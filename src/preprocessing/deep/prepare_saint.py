import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm


def parse_df_to_seq(df_path, seq_len=128):
    df = pd.read_csv(df_path, sep='\t')
    df = df[['user_id', 'timestamp', 'item_id', 'skill_id', 'correct']]
    df = df.groupby('user_id').filter(lambda x: len(x) >= seq_len)
    res = None
    pbar = tqdm(total=len(pd.unique(df.user_id)))
    df = df.groupby('user_id')
    for name, grouped in df:
        grouped = grouped.sort_values('timestamp').reset_index()
        values = grouped[['item_id', 'skill_id', 'correct']].values
        # print(values.shape)
        seqs = np.stack([values[i - seq_len:i] for i in range(seq_len, values.shape[0] + 1)])
        if res is None:
            res = seqs
        else:
            res = np.vstack([res, seqs])
        pbar.update(1)
    pbar.close()
    return res


if __name__ == '__main__':
    train_df = parse_df_to_seq('./data/squirrel/preparation/preprocessed_data_train.csv')
    test_df = parse_df_to_seq('./data/squirrel/preparation/preprocessed_data_test.csv')

    with open('./data/squirrel/preparation/saint_data_train.pkl', 'wb') as f:
        pickle.dump(train_df, f)

    with open('./data/squirrel/preparation/saint_data_test.pkl', 'wb') as f:
        pickle.dump(test_df, f)
