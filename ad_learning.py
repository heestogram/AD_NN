from data_gen import *

def ad_linear(X, A, R, K, alpha=1.0):
    n, p = X.shape
    V = simplex_vertices(K).numpy()           
    Y = np.zeros((n, K-1))
    for i in range(n):
        Y[i,:] = K * R[i] * V[A[i]]                  

    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(X, Y)

    def predict(X_new):
        f = model.predict(X_new)               
        scores = f @ V.T                            
        return scores.argmax(axis=1)
    return model, V, predict


import numpy as np

import numpy as np
from sklearn.linear_model import Ridge

def ad_linear_scalar(X, A, R, K=4, alpha=1.0):

    n, p = X.shape
    V = simplex_vertices(K).numpy()   

    # target = R_i
    y = R

    X_feat = np.zeros((n, p * (K - 1)))
    for i in range(n):
        w = V[A[i]]                   
        X_feat[i, :] = np.kron(X[i], w)

    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(X_feat, y)

    def predict(X_new):
        m = X_new.shape[0]
        scores = np.zeros((m, K))
        for k in range(K):
            w = V[k]                  
            W = model.coef_.reshape(p, K-1)
            fX = X_new @ W              
            scores[:, k] = fX @ w      
        return scores.argmax(axis=1)

    return model, V, predict

def _group_soft_threshold_rowwise(B, tau):
    """
    group sparsity
    """
    B_new = B.copy()
    norms = np.linalg.norm(B_new, axis=1)         
    scale = np.maximum(0.0, 1.0 - tau / (norms + 1e-12))
    B_new = scale[:, None] * B_new
    return B_new

def ad_linear_group(
    X, A, R, K,
    lam=0.05,            
    step=1e-2,      
    max_iter=1000,
    tol=1e-6,
    pi=None,
    alpha=1.0
):
    n, p = X.shape
    V = simplex_vertices(K).numpy() 
    if pi is None:
        pi = np.full(n, 1.0 / K, dtype=float)

    Y = np.zeros((n, K-1), dtype=float)
    for i in range(n):
        Y[i, :] = (R[i] / pi[i]) * V[A[i]]


    B = np.zeros((p, K-1), dtype=float)

    for it in range(max_iter):
        XB = X @ B
        grad = (2.0 / n) * (X.T @ (XB - Y))
        B_next = B - step * grad
        # group sparsity
        B_next = _group_soft_threshold_rowwise(B_next, tau=step * lam)

        denom = max(1.0, np.linalg.norm(B, ord='fro'))
        if np.linalg.norm(B_next - B, ord='fro') <= tol * denom:
            B = B_next
            break
        B = B_next


    def predict(X_new):
        F = X_new @ B            
        scores = F @ V.T       
        return scores.argmax(axis=1)


    model = {"B": B}
    return model, V, predict


import numpy as np

def _group_soft_threshold_rowwise(B, tau):
    """행(특징) 단위 L2 소프트임계: b_j <- (1 - tau/||b_j||)_+ * b_j"""
    norms = np.linalg.norm(B, axis=1)              # (p,)
    scale = np.maximum(0.0, 1.0 - tau / (norms + 1e-12))
    return scale[:, None] * B




def ad_linear_group_standardized(
    X, A, R, K,
    lam=1.0,        
    step=1e-3,     
    max_iter=2000,
    tol=1e-6,
    pi=None,      
    V=None,
    alpha=1.0
):
    """
    센터링 + 표준화 버전.
    """
    n, p = X.shape
    if V is None:
        V = simplex_vertices(K).numpy()  # (K, K-1)
    if pi is None:
        pi = np.full(n, 1.0 / K, dtype=float)

    # Target Y ∈ R^{n×(K-1)}
    Y = np.zeros((n, K-1), dtype=float)
    for i in range(n):
        Y[i, :] = (R[i] / pi[i]) * V[A[i]]

    #  표준화 
    X_mean = X.mean(axis=0, keepdims=True)       
    X_std = X.std(axis=0, keepdims=True) + 1e-12 
    Xs = (X - X_mean) / X_std                 

    B = np.zeros((p, K-1), dtype=float)

    # ISTA (prox-gradient)
    for _ in range(max_iter):
        XB = Xs @ B                           
        grad = (2.0 / n) * (Xs.T @ (XB - Y))        
        B_next = B - step * grad
        B_next = _group_soft_threshold_rowwise(B_next, tau=step * lam)

        if np.linalg.norm(B_next - B, ord='fro') <= tol * max(1.0, np.linalg.norm(B, ord='fro')):
            B = B_next
            break
        B = B_next

    def predict(X_new):
        Xs_new = (X_new - X_mean) / X_std
        F = Xs_new @ B
        scores = F @ V.T
        return scores.argmax(axis=1)


    model = {"B": B, "X_mean": X_mean, "X_std": X_std}
    return model, V, predict



def ad_linear_group_survival(
    X, A, T, Delta, K,
    lam=0.05,
    step=1e-2,
    max_iter=1000,
    tol=1e-6,
    pi=None,
):
    """
    AD-learning linear model for survival outcome.
    """
    n, p = X.shape
    V = simplex_vertices(K).numpy() 

    if pi is None:
        pi = np.full(n, 1.0 / K, dtype=float)
    pi = np.asarray(pi, dtype=float)

    # row-wise group soft-threshold
    def _group_soft_threshold_rowwise(B, tau):
        B_new = B.copy()
        norms = np.linalg.norm(B_new, axis=1)  
        scale = np.maximum(0.0, 1.0 - tau / (norms + 1e-12))
        B_new = scale[:, None] * B_new
        return B_new

    def _cox_grad_eta(eta, T, Delta, pi):
        """
        eta: (n,) linear predictor for actually received treatment
        T:   (n,) observed time
        Delta: (n,) event indicator
        pi:  (n,) propensity
        """
        n = len(eta)
        # weights w_i = Delta_i / (n * pi_i)
        w = Delta.astype(float) / (pi * n)

        # sort by time ascending
        order = np.argsort(T)
        T_s = T[order]
        eta_s = eta[order]
        w_s = w[order]

        exp_eta = np.exp(eta_s)

        cum_exp_rev = np.cumsum(exp_eta[::-1])[::-1]  # (n,)

        grad_s = -w_s.copy()  

        for i in range(n):
            if w_s[i] == 0.0:
                continue
            denom = cum_exp_rev[i]  
            contrib = w_s[i] * exp_eta[i:] / denom 
            grad_s[i:] += contrib

        grad = np.zeros_like(grad_s)
        grad[order] = grad_s

        return grad 
    
    def _cox_grad_eta_fast(eta, T, Delta, pi):

        n = len(eta)
        w = Delta.astype(float) / (pi * n)   # (n,)

        order = np.argsort(T)
        eta_s = eta[order]
        w_s   = w[order]

        exp_eta = np.exp(eta_s)

        cum_exp_rev = np.cumsum(exp_eta[::-1])[::-1]
        ratio = w_s / (cum_exp_rev + 1e-12)
        C_prefix = np.cumsum(ratio)
        grad_s = -w_s + exp_eta * C_prefix

        grad = np.zeros_like(grad_s)
        grad[order] = grad_s
        return grad

    # initialize B
    B = np.zeros((p, K-1), dtype=float)

    for it in range(max_iter):
        # forward
        F = X @ B        
        scores = F @ V.T       
        eta = scores[np.arange(n), A]  

        grad_eta = _cox_grad_eta_fast(eta, T, Delta, pi)   

        grad_scores = np.zeros_like(scores)
        grad_scores[np.arange(n), A] = grad_eta      
        grad_F = grad_scores @ V                   

        grad_B = X.T @ grad_F           

        # gradient step + group l2 proximal
        B_next = B - step * grad_B
        B_next = _group_soft_threshold_rowwise(B_next, tau=step * lam)

        denom = max(1.0, np.linalg.norm(B, ord='fro'))
        if np.linalg.norm(B_next - B, ord='fro') <= tol * denom:
            B = B_next
            break
        B = B_next

    def predict(X_new):

        F_new = X_new @ B          
        scores_new = F_new @ V.T   
        return scores_new.argmin(axis=1)

    model = {"B": B}
    return model, V, predict

