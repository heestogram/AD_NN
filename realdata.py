import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from ad_learning import *
from ad_neural import *


def _infer_continuous_idx(df: pd.DataFrame, covs):
    """
    연속형 열 추정 규칙:
      - dtype이 수치형이고
      - 고유값 개수 > 6 이고
      - {0,1} 이진집합이 아닌 경우
    """
    cont_idx = []
    for j, c in enumerate(covs):
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            uniq = pd.unique(s.dropna())
            if set(np.unique(uniq)).issubset({0, 1}):
                continue
            if len(uniq) > 6:
                cont_idx.append(j)
    return cont_idx

def transform_reward_softplus(R, tau=10.0):
    """
    R>=0는 거의 보존, R<0는 0+로 스무스하게 압축.
    tau 작을수록 ReLU에 근접(음수 더 얇게), 클수록 완만.
    """
    R = np.asarray(R, dtype=float)
    return tau * np.log1p(np.exp(R / tau))

def actg_to_arrays(df: pd.DataFrame, covs=None, keep_R_pos=False, R_log=False, neg_scal=False):
    """
    ACTG175 데이터프레임 -> (X, A, R, meta)
    trt 열이 이미 {0,1,2,3}로 코딩되어 있다고 가정
    """
    A = df["trt"].to_numpy().astype(int)

    R = df["cd420"].astype(float) - df["cd40"].astype(float)

    X = df[covs].to_numpy(dtype=float)

    if keep_R_pos:
        mask = (R > 0)
        X, A, R = X[mask], A[mask], R[mask]
        if R_log:
            R = np.log1p(R)
    else:
        if neg_scal:
            R = transform_reward_softplus(R)

    cont_idx = _infer_continuous_idx(df, covs)
    meta = {"covs": covs, "K": len(np.unique(A)), "n": len(A), "cont_idx": cont_idx}
    return X, A, R, meta

def actg_to_arrays_surv(df: pd.DataFrame,
                             covs=None,
                             trt_col="trt",
                             time_col="time",
                             cens_col="cid"):
    """
    ACTG175 real dataset -> (X, A, T, Delta, meta)
    - 스케일링은 하지 않고, 연속형 열 인덱스(cont_idx)만 meta에 담아둔다.
    """
    cols_needed = covs + [trt_col, time_col, cens_col]
    df_ = df[cols_needed].dropna().copy()

    A_raw = df_[trt_col].to_numpy()
    if A_raw.min() == 1:
        A = (A_raw - 1).astype(int)
    else:
        A = A_raw.astype(int)
    K = len(np.unique(A))

    T = df_[time_col].astype(float).to_numpy()
    cid = df_[cens_col].astype(int).to_numpy() 
    Delta = 1 - cid                            

    X = df_[covs].to_numpy(dtype=float)

    cont_idx = _infer_continuous_idx(df_, covs)

    meta = {
        "covs": covs,
        "K": K,
        "n": len(df_),
        "cont_idx": cont_idx,
    }
    return X, A, T, Delta, meta



def value_uniform(pred, A, R, K):
    pred = np.asarray(pred).reshape(-1)
    A    = np.asarray(A, dtype=int).reshape(-1)
    R    = np.asarray(R, dtype=float).reshape(-1)
    assert pred.shape == A.shape == R.shape, "pred/A/R length mismatch"
    return np.mean((pred == A).astype(float) * R * K)


def _safe_predict(predict_fn, X_te):
    pred = predict_fn(X_te)
    pred = np.asarray(pred)
    if pred.ndim != 1:
        pred = pred.reshape(-1)
    return pred.astype(int)




def cv5_value_once(train_and_build_predict, X, A, R, K, seed,
                   verbose=False, standardize=False, cont_idx=None):
    """
    train_and_build_predict: (X_tr, A_tr, R_tr, K) -> predict_fn
    standardize=True 이고 cont_idx가 주어지면,
    각 fold의 train으로 StandardScaler를 fit 후 train/test에 동일 변환을 적용(연속형 열만).
    """
    X = np.asarray(X, dtype=float)
    A = np.asarray(A, dtype=int)
    R = np.asarray(R, dtype=float)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    vals, sizes = [], []

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, A), start=1):
        X_tr, A_tr, R_tr = X[tr_idx].copy(), A[tr_idx], R[tr_idx]
        X_te, A_te, R_te = X[te_idx].copy(), A[te_idx], R[te_idx]

        if standardize and cont_idx:
            scaler = StandardScaler()
            scaler.fit(X_tr[:, cont_idx])
            X_tr[:, cont_idx] = scaler.transform(X_tr[:, cont_idx])
            X_te[:, cont_idx] = scaler.transform(X_te[:, cont_idx])

        try:
            predict_fn = train_and_build_predict(X_tr, A_tr, R_tr, K)
            pred_te = _safe_predict(predict_fn, X_te)
            v = value_uniform(pred_te, A_te, R_te, K)
        except Exception as e:
            if verbose:
                uniq, cnts = np.unique(A_tr, return_counts=True)
                msg = (f"[cv5] fold={fold_idx} failed | "
                       f"train sizes per arm={dict(zip(uniq.tolist(), cnts.tolist()))} | "
                       f"err: {repr(e)}")
                print(msg)
            raise
        vals.append(v)
        sizes.append(len(te_idx))

    vals, sizes = np.asarray(vals), np.asarray(sizes)
    return float(np.sum(vals * sizes) / np.sum(sizes))

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

def value_uniform_surv(pred, A, T, K):
    pred = np.asarray(pred).reshape(-1)
    A    = np.asarray(A).reshape(-1)
    T    = np.asarray(T).astype(float).reshape(-1)
    return np.mean((pred == A).astype(float) * T * K)

def cv5_value_once_surv(train_and_build_predict, X, A, T, Delta, K, seed,
                             verbose=False, standardize=False, cont_idx=None):
    """
    real survival data용 5-fold CV.
    standardize=True 이고 cont_idx가 주어지면,
    각 fold의 train으로 scaler fit 후 연속형 열만 변환.
    """
    X = np.asarray(X, dtype=float)
    A = np.asarray(A, dtype=int)
    T = np.asarray(T, dtype=float)
    Delta = np.asarray(Delta, dtype=int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    vals, sizes = [], []

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, A), start=1):
        X_tr, A_tr, T_tr, D_tr = X[tr_idx].copy(), A[tr_idx], T[tr_idx], Delta[tr_idx]
        X_te, A_te, T_te, D_te = X[te_idx].copy(), A[te_idx], T[te_idx], Delta[te_idx]

        if standardize and cont_idx is not None and len(cont_idx) > 0:
            scaler = StandardScaler()
            scaler.fit(X_tr[:, cont_idx])
            X_tr[:, cont_idx] = scaler.transform(X_tr[:, cont_idx])
            X_te[:, cont_idx] = scaler.transform(X_te[:, cont_idx])

        try:
            predict_fn = train_and_build_predict(X_tr, A_tr, T_tr, D_tr, K)
            pred_te = _safe_predict(predict_fn, X_te)
            v = value_uniform_surv(pred_te, A_te, T_te, K) 
        except Exception as e:
            if verbose:
                uniq, cnts = np.unique(A_tr, return_counts=True)
                msg = (f"[cv5] fold={fold_idx} failed | "
                       f"train sizes per arm={dict(zip(uniq.tolist(), cnts.tolist()))} | "
                       f"err: {repr(e)}")
                print(msg)
            raise

        vals.append(v)
        sizes.append(len(te_idx))

    vals, sizes = np.asarray(vals), np.asarray(sizes)
    return float(np.sum(vals * sizes) / np.sum(sizes))



def repeated_cv_value(train_and_build_predict, X, A, R, K,
                      n_repeats=1000, base_seed=2025, verbose=False,
                      standardize=False, cont_idx=None):
    values = np.zeros(n_repeats, dtype=float)
    for r in range(n_repeats):
        print(f" Repeat {r+1}/{n_repeats}")
        seed = base_seed + r
        values[r] = cv5_value_once(
            train_and_build_predict, X, A, R, K, seed, verbose=verbose,
            standardize=standardize, cont_idx=cont_idx
        )
    return {
        "values": values,
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "n_repeats": int(n_repeats)
    }

def repeated_cv_value_surv(train_and_build_predict, X, A, T, Delta, K,
                                n_repeats=50, base_seed=2025, verbose=False, standardize=False, cont_idx=None):
    values = np.zeros(n_repeats, dtype=float)
    for r in range(n_repeats):
        print(f" Repeat {r+1}/{n_repeats}")
        seed = base_seed + r
        values[r] = cv5_value_once_surv(
            train_and_build_predict, X, A, T, Delta, K, seed, verbose=verbose, standardize=standardize, cont_idx=cont_idx
        )
    return {
        "values": values,
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "n_repeats": int(n_repeats),
    }

def build_predict_ad_linear(Xtr, Atr, Rtr, K, alpha=1.0):
    Xtr = np.asarray(Xtr, dtype=float)
    Atr = np.asarray(Atr, dtype=int)
    Rtr = np.asarray(Rtr, dtype=float)
    model, V, predict = ad_linear(Xtr, Atr, Rtr, K=K, alpha=alpha)
    return predict  

def build_predict_ad_nn(Xtr, Atr, Rtr, K, epochs=60, lr=1e-3, hidden=[128,128]):
    Xtr = np.asarray(Xtr, dtype=float)
    Atr = np.asarray(Atr, dtype=int)
    Rtr = np.asarray(Rtr, dtype=float)
    model, V, predict = ad_nn(Xtr, Atr, Rtr, K=K, epochs=epochs, lr=lr, hidden=hidden)
    return predict


def build_predict_ad_linear_surv(Xtr, Atr, Ttr, Dtr, K,
                                      lam=0.05, step=1e-2):
    Xtr = np.asarray(Xtr, dtype=float)
    Atr = np.asarray(Atr, dtype=int)
    Ttr = np.asarray(Ttr, dtype=float)
    Dtr = np.asarray(Dtr, dtype=int)

    model, V, predict = ad_linear_group_survival(
        Xtr, Atr, Ttr, Dtr, K,
        lam=lam, step=step, max_iter=1000, tol=1e-6, pi=None
    )
    return predict 

def build_predict_ad_nn_surv(Xtr, Atr, Ttr, Dtr, K,
                                  epochs=50, lr=1e-3, hidden=[128,128]):
    Xtr = np.asarray(Xtr, dtype=float)
    Atr = np.asarray(Atr, dtype=int)
    Ttr = np.asarray(Ttr, dtype=float)
    Dtr = np.asarray(Dtr, dtype=int)

    model, V, predict = ad_nn_survival(
        Xtr, Atr, Ttr, Dtr, K,
        epochs=epochs, lr=lr, pi=None, hidden=hidden
    )
    return predict