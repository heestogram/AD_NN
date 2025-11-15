import numpy as np

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

def delta_scenario(X, scenario: int, K: int,
                       smooth: bool = False,
                       temp: float = 0.25,    
                       amp: float = 1.0):       
    """
    """
    n, p = X.shape
    d = np.zeros((n, K))
    xs = [X[:, j % p] for j in range(max(p, 6))]
    x1,x2,x3,x4,x5,x6 = xs[:6]

    def _sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
    def _phi(u): return np.tanh(u)

    # ---------- Scenario 1 : Linear ----------
    if scenario == 1:

        if K >= 1: d[:,0] = 1 + x1 + x2 + x3 + x4
        if K >= 2: d[:,1] = 1 + x1 - x2 - x3 + x4
        if K >= 3: d[:,2] = 1 + x1 - x2 + x3 - x4
        if K >= 4: d[:,3] = 1 - x1 - x2 + x3 + x4


    # ---------- Scenario 2 : Tree ----------
    elif scenario == 2:
        if not smooth:
            if K >= 1: d[:,0] = 3.0 * ((x1 <= 0.5).astype(float) * ((x2 > -0.6).astype(float) - 1.0))
            if K >= 2: d[:,1] = ((x3 <= 1.0).astype(float)) * (2.0*(x4 <= -0.3).astype(float) - 1.0)
            if K >= 3: d[:,2] = (4.0*(x5 <= 0.0).astype(float) - 2.0)
            if K >= 4: d[:,3] = (4.0*(x6 <= 0.0).astype(float) - 2.0)
        else:
            s1 = _sigmoid((0.5 - x1)/temp)   
            s2 = _sigmoid((x2 + 0.6)/temp)    
            s3 = _sigmoid((1.0 - x3)/temp) 
            s4 = _sigmoid((-0.3 - x4)/temp)   
            s5 = _sigmoid((0.0 - x5)/temp)    
            s6 = _sigmoid((0.0 - x6)/temp)  
            if K >= 1: d[:,0] = amp * 3.0 * s1 * (2.0*s2 - 1.0)
            if K >= 2: d[:,1] = amp * s3 * (2.0*s4 - 1.0)
            if K >= 3: d[:,2] = amp * (4.0*s5 - 2.0)
            if K >= 4: d[:,3] = amp * (4.0*s6 - 2.0)

    # ---------- Scenario 3 : Polynomial ----------
    elif scenario == 3:
        if not smooth:
            if K >= 1: d[:,0] = 0.2 + x1**2 + x2**2 - x3**2 - x4**2
            if K >= 2: d[:,1] = 0.2 + x2**2 + x3**2 - x1**2 - x4**2
            if K >= 3: d[:,2] = 0.2 + x3**2 + x4**2 - x1**2 - x2**2
            if K >= 4: d[:,3] = 0.2 + x1**2 + x4**2 - x2**2 - x3**2
        else:
            # 교차항 + 합성 비선형
            c12 = _phi(0.8*x1*x2); c23 = _phi(0.8*x2*x3)
            c34 = _phi(0.8*x3*x4); c14 = _phi(0.8*x1*x4)
            h1  = _phi(0.7*x1 + 0.5*x2 - 0.4*x3 + 0.3*x4)
            h2  = _phi(-0.6*x1 + 0.7*x2 + 0.5*x3 - 0.4*x4)
            if K >= 1: d[:,0] = amp*(0.2 + x1**2 + x2**2 - x3**2 - x4**2 + 0.6*c12 + 0.4*h1)
            if K >= 2: d[:,1] = amp*(0.2 + x2**2 + x3**2 - x1**2 - x4**2 + 0.6*c23 + 0.4*h2)
            if K >= 3: d[:,2] = amp*(0.2 + x3**2 + x4**2 - x1**2 - x2**2 + 0.6*c34 + 0.4*h1)
            if K >= 4: d[:,3] = amp*(0.2 + x1**2 + x4**2 - x2**2 - x3**2 + 0.6*c14 + 0.4*h2)
    else:
        raise ValueError("scenario must be 1, 2, or 3")

    if K <= 4:
        return d

    def lin_arm(j):
        P = [j % p, (j+1) % p]
        M = [(j+2) % p, (j+3) % p]

        return 1.0 + X[:,P[0]] + X[:,P[1]] - X[:,M[0]] - X[:,M[1]]


    def poly_arm(j):
        P = [j % p, (j+1) % p]; M = [(j+2) % p, (j+3) % p]
        if not smooth:
            return 0.2 + X[:,P[0]]**2 + X[:,P[1]]**2 - X[:,M[0]]**2 - X[:,M[1]]**2
        cP = _phi(0.8*X[:,P[0]]*X[:,P[1]])
        h  = _phi(0.7*X[:,P[0]] - 0.6*X[:,P[1]] + 0.4*X[:,M[0]] - 0.3*X[:,M[1]])
        base = 0.2 + X[:,P[0]]**2 + X[:,P[1]]**2 - X[:,M[0]]**2 - X[:,M[1]]**2
        return amp*(base + 0.6*cP + 0.4*h)

    def tree_tmpl(t, j):
        jp = j % p; jn1 = (j+1) % p
        if not smooth:
            if t == 0:
                return 3.0 * ((X[:,jp] <= 0.5).astype(float) * ((X[:,jn1] > -0.6).astype(float) - 1.0))
            elif t == 1:
                return ((X[:,jp] <= 1.0).astype(float)) * (2.0*(X[:,jn1] <= -0.3).astype(float) - 1.0)
            else:
                return (4.0*(X[:,jp] <= 0.0).astype(float) - 2.0)
        if t == 0:
            sA = _sigmoid((0.5 - X[:,jp])/temp); sB = _sigmoid((X[:,jn1] + 0.6)/temp)
            return amp * (3.0 * sA * (2.0*sB - 1.0))
        elif t == 1:
            sA = _sigmoid((1.0 - X[:,jp])/temp); sB = _sigmoid((-0.3 - X[:,jn1])/temp)
            return amp * (sA * (2.0*sB - 1.0))
        else:
            sA = _sigmoid((0.0 - X[:,jp])/temp)
            return amp * (4.0*sA - 2.0)

    start_j = 0
    for k in range(4, K):
        j = (start_j + 2*(k-4))
        if scenario == 1:
            d[:,k] = lin_arm(j)
        elif scenario == 2:
            t = (k-4) % 4
            d[:,k] = tree_tmpl(t, j)
        elif scenario == 3:
            d[:,k] = poly_arm(j)

    return d


def simplex_vertices(K: int) -> torch.Tensor:
    w_list = []
    for j in range(1, K + 1):
        if j == 1:

            w_j = (1.0 / (K - 1) ** 0.5) * torch.ones(K - 1)
        else:

            term1 = -((1.0 + K ** 0.5) / ((K - 1) ** 1.5)) * torch.ones(K - 1)
            e = torch.zeros(K - 1)
            e[j - 2] = (K / (K - 1)) ** 0.5
            w_j = term1 + e
        w_list.append(w_j)
    return torch.stack(w_list) 

def delta_scenario_past(X, scenario: int, K: int):
    """
    원 논문의 시나리오 1~3을 구현해놓음.
    """
    n, p = X.shape
    d = np.zeros((n, K))

    x = [X[:, j % p] for j in range(max(p, 6))]  
    x1,x2,x3,x4,x5,x6 = x[0],x[1],x[2],x[3],x[4],x[5]

    if scenario == 1: 
        if K >= 1: d[:,0] = 1 + x1 + x2 + x3 + x4
        if K >= 2: d[:,1] = 1 + x1 - x2 - x3 + x4
        if K >= 3: d[:,2] = 1 + x1 - x2 + x3 - x4
        if K >= 4: d[:,3] = 1 - x1 - x2 + x3 + x4

    elif scenario == 2:  
        if K >= 1: d[:,0] = 3.0 * ((x1 <= 0.5).astype(float) * ((x2 > -0.6).astype(float) - 1.0))
        if K >= 2: d[:,1] = ((x3 <= 1.0).astype(float)) * (2.0*(x4 <= -0.3).astype(float) - 1.0)
        if K >= 3: d[:,2] = (4.0*(x5 <= 0.0).astype(float) - 2.0)
        if K >= 4: d[:,3] = (4.0*(x6 <= 0.0).astype(float) - 2.0)

    elif scenario == 3:  
        if K >= 1: d[:,0] = 0.2 + x1**2 + x2**2 - x3**2 - x4**2
        if K >= 2: d[:,1] = 0.2 + x2**2 + x3**2 - x1**2 - x4**2
        if K >= 3: d[:,2] = 0.2 + x3**2 + x4**2 - x1**2 - x2**2
        if K >= 4: d[:,3] = 0.2 + x1**2 + x4**2 - x2**2 - x3**2
    else:
        raise ValueError("scenario must be 1, 2, or 3")

    if K <= 4:
        return d


    def lin_arm(j):
        P = [j % p, (j+1) % p]
        M = [(j+2) % p, (j+3) % p]
        return 1.0 + X[:,P[0]] + X[:,P[1]] - X[:,M[0]] - X[:,M[1]]

    def poly_arm(j):
        P = [j % p, (j+1) % p]
        M = [(j+2) % p, (j+3) % p]
        return 0.2 + X[:,P[0]]**2 + X[:,P[1]]**2 - X[:,M[0]]**2 - X[:,M[1]]**2


    def tree_tmpl(t, j):
        jp = j % p; jn1 = (j+1) % p
        if t == 0:
            return 3.0 * ((X[:,jp] <= 0.5).astype(float) * ((X[:,jn1] > -0.6).astype(float) - 1.0))
        elif t == 1:
            return ((X[:,jp] <= 1.0).astype(float)) * (2.0*(X[:,jn1] <= -0.3).astype(float) - 1.0)
        elif t == 2:
            return (4.0*(X[:,jp] <= 0.0).astype(float) - 2.0)
        else:
            return (4.0*(X[:,jp] <= 0.0).astype(float) - 2.0)

    # k=5부터 채우기
    start_j = 0  
    for k in range(4, K):
        j = (start_j + 2*(k-4))  
        if scenario == 1:
            d[:,k] = lin_arm(j)
        elif scenario == 2:
            t = (k-4) % 4       
            d[:,k] = tree_tmpl(t, j)
        elif scenario == 3:
            d[:,k] = poly_arm(j)

    return d





def mu_default(X):
    """
    논문 기본설정: μ(x) = 1 + X1 + X2  (X는 (n,p), X1=X[:,0], X2=X[:,1])
    나머지 시나리오에서도 mu가 동일하게 적용되는지 모르겠음. appendix에서 알려준대놓고 안 알려줌;;
    """
    return 1.0 + X[:, 0] + X[:, 1]

# def mu_poly(X):
#     x1, x2 = X[:,0], X[:,1]
#     return 1.0 + 0.6*x1 + 0.6*x2 + 0.4*(x1**2) - 0.3*(x2**2) + 0.2*(x1*x2)


def simulate_data(K, n=2000, p=10, scenario=1, seed=0,
                  main_effect=mu_default, sigma_eps=1.0,
                  uniform_low=-1.0, uniform_high=1.0, smooth=False):

    rng = np.random.default_rng(seed)

    # X ~ Unif[-1, 1]
    X = rng.uniform(uniform_low, uniform_high, size=(n, p))

    deltas = delta_scenario(X, scenario, K, smooth)
    mu = main_effect(X)

    # A를 가능한 균등하게 분배
    base = n // K
    counts = np.full(K, base, dtype=int)
    counts[: (n % K)] += 1       
    A = np.repeat(np.arange(K), counts)
    rng.shuffle(A)                


    eps = rng.normal(0.0, sigma_eps, size=n)
    R = mu + deltas[np.arange(n), A] + eps

    # 최적 치료 (Ground truth)
    true_opt = np.argmax(deltas, axis=1)

    return X, A, R, true_opt


import numpy as np


def simulate_data_surv(
    K,
    n=2000,
    p=10,
    scenario=1,
    seed=0,
    main_effect=mu_default,  # μ(x)
    uniform_low=-1.0,
    uniform_high=1.0,
    theta_cens=50.0,
    smooth: bool = False,    
):
    """
    생존형 outcome을 delta 기반 구조로 생성.
    """


    rng = np.random.default_rng(seed)


    X = rng.uniform(uniform_low, uniform_high, size=(n, p))


    deltas = delta_scenario(X, scenario, K, smooth)

    mu = main_effect(X)     # (n,)
    mu_col = mu.reshape(-1, 1)


    lambda_mat = mu_col + deltas     # (n,K)
    true_opt = np.argmax(lambda_mat, axis=1)


    base = n // K
    counts = np.full(K, base, dtype=int)
    counts[:(n % K)] += 1
    A = np.repeat(np.arange(K), counts)
    rng.shuffle(A)

 
    lambdas = lambda_mat[np.arange(n), A]
    rates = np.exp(lambdas)


    R_true = np.exp(lambdas)


    C = rng.exponential(scale=theta_cens, size=n)


    T = np.minimum(R_true, C)
    Delta = (R_true <= C).astype(int)

    return X, A, T, Delta, true_opt, R_true, C
