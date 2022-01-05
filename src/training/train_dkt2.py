import argparse
import pandas as pd
from random import shuffle

import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence

from src.training.model_dkt2 import DKT2
from src.utils import *
from src.utils.data_loader import load_split
from src.utils.metrics import compute_metrics

from collections import defaultdict
import json


def get_data(df, split_id=0, dataset_name='elemmath_2021', randomize=True):
    """Extract sequences from dataframe.
    Arguments:
        df (pandas Dataframe): output by prepare_data.py
        train_split (float): proportion of data to use for training
    """
    item_ids, skill_ids, labels, indices = [], [], [], []
    for idx, u_df in df.groupby("user_id"):
        item_ids.append(torch.tensor(u_df["item_id"].values, dtype=torch.long))
        skill_ids.append(torch.tensor(u_df["hashed_skill_id"].values if 'skill_id' not in df.columns else u_df['skill_id'].values, dtype=torch.long))
        labels.append(torch.tensor(u_df["correct"].values, dtype=torch.long))
        indices.append(idx)

    item_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), i))[:-1] for i in item_ids]
    skill_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), s))[:-1] for s in skill_ids]
    label_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), l))[:-1] for l in labels]

    data = list(zip(indices, item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels))

    # Train-test split across users
    s = load_split(split_id, dataset_name)
    train_ids, test_ids = set(s['train_ids']), set(s['test_ids'])
    train_data, test_data = [], []
    for d in data:
        id = d[0]
        if id in train_ids:
            train_data.append(d[1:])
        else:
            test_data.append(d[1:])
    return train_data, test_data


def prepare_batches(data, batch_size, randomize=True):
    """Prepare batches grouping padded sequences.
    Arguments:
        data (list of lists of torch Tensor): output by get_data
        batch_size (int): number of sequences per batch
    Output:
        batches (list of lists of torch Tensor)
    """
    if randomize:
        shuffle(data)
    batches = []

    for k in range(0, len(data), batch_size):
        batch = data[k:k + batch_size]
        seq_lists = list(zip(*batch))
        inputs_and_ids = [pad_sequence(seqs, batch_first=True, padding_value=0)
                          for seqs in seq_lists[:-1]]
        labels = pad_sequence(seq_lists[-1], batch_first=True, padding_value=-1)  # Pad labels with -1
        batches.append([*inputs_and_ids, labels])

    return batches


def compute_loss(preds, labels, criterion):
    preds = preds[labels >= 0].flatten()
    labels = labels[labels >= 0].float()
    return criterion(preds, labels)


def train(train_data, val_data, model, optimizer, logger, saver, num_epochs, batch_size, print_every=50):
    """Train DKT model.
    Arguments:
        train_data (list of lists of torch Tensor)
        val_data (list of lists of torch Tensor)
        model (torch Module)
        optimizer (torch optimizer)
        logger: wrapper for TensorboardX logger
        saver: wrapper for torch saving
        num_epochs (int): number of epochs to train for
        batch_size (int)
    """
    criterion = nn.BCEWithLogitsLoss()
    metrics = Metrics()
    step = 0

    for epoch in range(num_epochs):
        train_batches = prepare_batches(train_data, batch_size)
        val_batches = prepare_batches(val_data, batch_size)

        # Training
        for item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels in train_batches:
            item_inputs = item_inputs.cuda()
            skill_inputs = skill_inputs.cuda()
            label_inputs = label_inputs.cuda()
            item_ids = item_ids.cuda()
            skill_ids = skill_ids.cuda()
            preds = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)

            loss = compute_loss(preds, labels.cuda(), criterion)
            acc, auc, nll, mse, f1 = compute_metrics(torch.sigmoid(preds[labels >= 0]).detach().cpu().numpy().flatten(),
                                                     labels[labels >= 0].float().numpy().flatten())

            model.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            metrics.store({'loss/train': loss.item()})
            metrics.store({'auc/train': auc})
            metrics.store({'acc/train': acc})
            metrics.store({'nll/train': nll})
            metrics.store({'mse/train': mse})
            metrics.store({'f1/train': f1})

            # Logging
            if print_every > 0 and step % print_every == 0:
                logger.log_scalars(metrics.average(), step)

        # Validation
        model.eval()
        all_preds = np.empty(0)
        all_label = np.empty(0)
        for item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels in val_batches:
            item_inputs = item_inputs.cuda()
            skill_inputs = skill_inputs.cuda()
            label_inputs = label_inputs.cuda()
            item_ids = item_ids.cuda()
            skill_ids = skill_ids.cuda()
            with torch.no_grad():
                preds = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
            preds = torch.sigmoid(preds[labels >= 0]).flatten().cpu().numpy()
            labels = labels[labels >= 0].float().flatten().cpu().numpy()
            all_preds = np.concatenate([all_preds, preds])
            all_label = np.concatenate([all_label, labels])
        acc, auc, nll, mse, f1 = compute_metrics(all_preds, all_label)
        metrics.store({'auc/val': auc})
        metrics.store({'acc/val': acc})
        metrics.store({'nll/val': nll})
        metrics.store({'mse/val': mse})
        metrics.store({'f1/val': f1})
        model.train()

        # Save model
        average_metrics = metrics.average()
        logger.log_scalars(average_metrics, step)
        stop = saver.save(average_metrics['auc/val'], model)
        if stop:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DKT.')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--logdir', type=str, default='runs/dkt')
    parser.add_argument('--savedir', type=str, default='save/dkt')
    parser.add_argument('--hid_size', type=int, default=200)
    parser.add_argument('--embed_size', type=int, default=200)
    parser.add_argument('--num_hid_layers', type=int, default=1)
    parser.add_argument('--drop_prob', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--total_split', type=int, default=5)
    parser.add_argument('--print_every', type=int, default=50)
    args = parser.parse_args()
    print(args.logdir, args.dataset, args.total_split)

    set_random_seeds(args.seed)
    print('Reading the data from data/{}/preparation'.format(args.dataset))
    full_df = pd.read_csv(os.path.join('data', args.dataset, 'preparation/preprocessed_data.csv'), sep="\t")

    metric_dic = defaultdict(lambda: defaultdict(list))
    for split_id in range(args.total_split):
        torch.cuda.empty_cache()
        print(f'Train model for split ID {split_id}')
        train_data, test_data = get_data(full_df, dataset_name=args.dataset, split_id=split_id)
        model = DKT2(int(full_df["item_id"].max()), int(full_df["skill_id"].max() if "skill_id" in full_df.columns else full_df["hashed_skill_id"].max()), args.hid_size,
                     args.embed_size, args.num_hid_layers, args.drop_prob).cuda()
        optimizer = Adam(model.parameters(), lr=args.lr)

        # Reduce batch size until it fits on GPU
        print('Fitting model on GPUs and start training')
        while True:
            try:
                # Train
                param_str = (f'{args.dataset},'
                             f'epochs={args.num_epochs},'
                             f'batch_size={args.batch_size},'
                             f'hid_size={args.hid_size},'
                             f'embed_size={args.embed_size},'
                             f'num_hid_layers={args.num_hid_layers},'
                             f'drop_prob={args.drop_prob}')
                print(split_id, param_str)
                logger = Logger(os.path.join(args.logdir, param_str + "_"
                                             + str(split_id)))
                saver = Saver(args.savedir, param_str + "_" + str(split_id))
                train(train_data, test_data, model, optimizer, logger, saver,
                      args.num_epochs, args.batch_size,
                      print_every=args.print_every)
                break
            except RuntimeError as e:
                print(f'Batch does not fit on gpu, reducing size to {args.batch_size}, because')
                print(e)
                args.batch_size = args.batch_size // 2
        logger.close()

        model = saver.load()
        train_batches = prepare_batches(train_data, args.batch_size, randomize=False)
        test_batches = prepare_batches(test_data, args.batch_size, randomize=False)

        # Predict on training and test set
        model.eval()

        def predict(batches):
            all_preds = np.empty(0)
            all_label = np.empty(0)
            for item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels in batches:
                with torch.no_grad():
                    item_inputs = item_inputs.cuda()
                    skill_inputs = skill_inputs.cuda()
                    label_inputs = label_inputs.cuda()
                    item_ids = item_ids.cuda()
                    skill_ids = skill_ids.cuda()
                    preds = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
                    preds = torch.sigmoid(preds[labels >= 0]).cpu().numpy()
                    labels = labels[labels >= 0].float().flatten().cpu().numpy()
                    all_preds = np.concatenate([all_preds, preds])
                    all_label = np.concatenate([all_label, labels])
            acc_train, auc_train, nll_train, mse_train, f1 = compute_metrics(all_preds, all_label)
            metrics_results = {
                "acc": acc_train,
                "auc": auc_train,
                "nll": nll_train,
                "mse": mse_train,
                "rmse": np.sqrt(mse_train),
                "f1": f1
            }
            print(''.join(['[{}: {:.4f}]'.format(k, v) for k, v in metrics_results.items()]))
            return metrics_results

        print('Evaluate on training set')
        metrics_train = predict(train_batches)
        for k, v in metrics_train.items():
            metric_dic['train'][k].append(v)

        print('Evaluate on test set')
        metrics_test = predict(test_batches)
        for k, v in metrics_test.items():
            metric_dic['test'][k].append(v)

    print('Calculating CV results')
    print('[Train]' + ''.join(['[{}: {:.4f}]'.format(k, np.mean(v)) for k, v in metric_dic['train'].items()]))
    print('[Test]' + ''.join(['[{}: {:.4f}]'.format(k, np.mean(v)) for k, v in metric_dic['test'].items()]))

    with open(f"results/DKT2_" + param_str + ".json", "w") as outfile:
        json.dump(metric_dic, outfile, indent=4)
