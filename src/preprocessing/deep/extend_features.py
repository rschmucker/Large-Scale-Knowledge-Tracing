import argparse
from scipy import sparse
from scipy.sparse import load_npz, csr_matrix
import numpy as np

from embedding import inference
import constants as c

def extend_embeddings(X):
    print('X file shape: ' + str(X.shape))
    inter = inference.Inter('./data/embedding_time.csv', './data/shape.json', './data/embedding_time_2020-11-30_16_36_46.602000.pt')
    indicies = np.array([X.getcol(c.USER_ID).toarray(), X.getcol(c.TIMESTAMP).toarray()])
    embeddings = []
    for user_id, timestamp in indicies:
        embeddings.append(inter.infer(user_id, timestamp))
    X_extended = sparse.hstack([X, csr_matrix(embeddings)])
    print('new shape: ' + str(X_extended.shape))
    return X_extended


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='extend features on sparse feature matrix.')
    parser.add_argument('--X_file', type=str)
    args = parser.parse_args()

    # Load sparse dataset
    print("loading X file: " + args.X_file)
    X = csr_matrix(load_npz(args.X_file))

    sparse.save_npz(f"{args.X_file}_embedding", extend_embeddings(X))
