from utils import *
from PRPCA import PRPCA
from sklearn.metrics.pairwise import  cosine_similarity , rbf_kernel
from sklearn import metrics as m

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('pubmed')
Af = adj.toarray()
nnodes = Af.shape[0]

d = features.toarray().shape[1]
# experiments with another similarity
MMCos = cosine_similarity(features.toarray()) / (d-1)
MMrbf = rbf_kernel(features.toarray())/(d - 1)

predicts = PRPCA(features.toarray(), Af, y_train, delta=1,
                 sigma=1, alpha=0.9, svd_error=0.1,
                 iter_=10, tol=1e-03, beta=0.9)

print('accuracy',m.accuracy_score(np.argmax(y_test[test_mask], axis=-1), np.argmax(np.array(predicts[0])[test_mask], axis=-1) ))
print('time', predicts[1])