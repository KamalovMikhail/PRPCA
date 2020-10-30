import numpy as np
import scipy.sparse as sp
import timeit
from scipy.sparse.linalg import gmres
from sklearn.decomposition import TruncatedSVD


def calc_A_hat(adj_matrix: sp.spmatrix, delta, sigma, MMx) -> sp.spmatrix:
    nnodes = adj_matrix.shape[0]
    A = adj_matrix + sp.eye(nnodes)  # Ω#@ D_invsqrt_corr
    D_vec = np.sum(A, axis=1).A1
    lsigma = sigma - 1
    rsigma = - sigma
    wsigma = -2 * sigma + 1

    D_l = sp.diags(np.power(D_vec, lsigma))
    D_r = sp.diags(np.power(D_vec, rsigma))
    Dw = sp.diags(np.power(D_vec, wsigma))
    S_ = MMx @ Dw

    return S_, D_l @ A @ D_r + delta * S_

def JOR(A, Z, Y, iter_, beta, alpha):
    A = np.copy(A)
    Z = np.copy(Z)
    Y = np.copy(Y)
    time_ = 0
    for _ in range(iter_):
        start = timeit.default_timer()
        Z = Z + beta * (alpha * A @ Z - Z + (1 - alpha) * Y)
        time_ += (timeit.default_timer() - start)
    return Z, time_

def GMRES(A, Y, alpha, k, tol):
    time_ = 0
    A = np.copy(A)
    Y = np.copy(Y)
    predicts = []
    for j in range(k):
        start = timeit.default_timer()
        temp_ = gmres(A, (1 - alpha) * Y[:, j], tol=tol)[0]
        time_ += (timeit.default_timer() - start)
        predicts.append([temp_])

    return np.concatenate(predicts).T, time_

def PRPCA(X, A, Y, delta = 1, sigma = 1, alpha= 0.9, svd_error = 0.1, iter_ = 10, tol = 1e-03, beta = 0.9):
    svd = TruncatedSVD(n_components=1, algorithm='randomized')
    Xn = np.copy(X)
    Xn = Xn.T
    Xn = Xn - np.median(Xn, axis=0)
    MMx = np.dot(Xn.T, Xn) / (Xn.shape[0] - 1)
    nnodes = A.shape[0]

    mmc, AHAT = calc_A_hat(A, delta, sigma, MMx)
    rex = (np.identity(nnodes) - alpha * AHAT)
    svd.fit(mmc)
    gamma = svd.singular_values_[0]
    if (1 + delta * (gamma - svd_error)) <= 1 / alpha:
        print('jor')
        Z, tm = JOR(A=AHAT, Z=Y, Y=Y, iter_=iter_, beta=beta, alpha=alpha)
    else:
        print('gmres')
        Z, tm = GMRES(A=rex, Y=Y, alpha=alpha, k=Y.shape[1], tol=tol)

    return Z, tm

