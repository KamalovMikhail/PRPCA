from PRPCA import PRPCA
from typing import *
import copy
import numpy as np
from sklearn.metrics.pairwise import  cosine_similarity , rbf_kernel
from sklearn import metrics as m


adj = np.load('data/CCT/adj.npy' )
Y = np.load('data/CCT/labels_binary.npy' )
features = np.load('data/CCT/X.npy')

Af = adj
nnodes = Af.shape[0]

d = features.shape[1]
# experiments with another similarity
MMCos = cosine_similarity(features) / (d-1)
MMrbf = rbf_kernel(features)/(d - 1)

# Split dataset as in paper Revisiting Semi-Supervised Learning with Graph Embeddings
def exclude_idx(idx: np.ndarray, idx_exclude_list: List[np.ndarray]) -> np.ndarray:
    idx_exclude = np.concatenate(idx_exclude_list)
    return np.array([i for i in idx if i not in idx_exclude])
def known_unknown_split(idx: np.ndarray, nknown, seed) -> Tuple[np.ndarray, np.ndarray]:
    rnd_state = np.random.RandomState(seed)
    known_idx = rnd_state.choice(idx, nknown, replace=False)
    unknown_idx = exclude_idx(idx, [known_idx])
    return known_idx, unknown_idx
def train_stopping_split(idx: np.ndarray, labels: np.ndarray, ntrain_per_class, nstopping, seed, ) -> Tuple[
    np.ndarray, np.ndarray]:
    rnd_state = np.random.RandomState(seed)
    train_idx_split = []
    for i in np.unique(labels):
        train_idx_split.append(
            rnd_state.choice(idx[labels == i], ntrain_per_class, replace=False)
        )
    train_idx = np.concatenate(train_idx_split)
    stopping_idx = rnd_state.choice(
        exclude_idx(idx, [train_idx]), nstopping, replace=False
    )
    return train_idx, stopping_idx
def gen_splits(labels: np.ndarray, idx_split_args: Dict[str, int], test) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_idx = np.arange(len(labels))
    known_idx, unknown_idx = known_unknown_split(
        all_idx, idx_split_args["nknown"], idx_split_args["seed"]
    )
    _, cnts = np.unique(labels[known_idx], return_counts=True)
    stopping_split_args = copy.copy(idx_split_args)
    del stopping_split_args["nknown"]
    train_idx, stopping_idx = train_stopping_split(
        known_idx, labels[known_idx], **stopping_split_args
    )
    if test:
        val_idx = unknown_idx
    else:
        val_idx = exclude_idx(known_idx, [train_idx, stopping_idx])
    return train_idx, stopping_idx, val_idx
idx_split_args = {
    "ntrain_per_class": 20,
    "nstopping": 500,
    "nknown": 1362,
    "seed": 0,
}
train_idx, _, test_idx = gen_splits(
    labels=Y, idx_split_args=idx_split_args, test=True
)

new_labels = train_idx
y_train = np.zeros((nnodes, 2))
for i in new_labels:
    c_ = int(Y[i] )
    y_train[i, c_] = 1
predicts = PRPCA(features, Af, y_train, delta=1,
                 sigma=1, alpha=0.9, svd_error=0.1,
                 iter_=10, tol=1e-03, beta=0.9)

print('accuracy',m.accuracy_score( Y[test_idx] , np.argmax(np.array(predicts[0])[test_idx], axis=-1) ))
print('time', predicts[1])

