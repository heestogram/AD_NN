import numpy as np
from data_gen import *
from ad_neural import *
from ad_learning import *

def evaluate_policy(predict_fn, X, A, R, K, true_opt=None):
    pred = predict_fn(X)
    error_rate = None if true_opt is None else np.mean(pred != true_opt)
    value = np.mean(R * (pred == A) * K)
    return error_rate, value

def run_experiments(K, p=10, n=2000, n_test=10000, scenario_list=[1,2,3],
                    n_repeats=20, epochs=60, warmup_epochs=40, lr=1e-3, alpha=1.0, smooth=False, hidden=[128,128],
                    k_active=None, k_schedule=None, poly=False):
    """
    학습: 크기 n의 학습 데이터 전체 사용
    평가: 크기 n_test의 독립 테스트 데이터로만 평가
    """
    results = {}

    for scenario in scenario_list:
        err_lin_list, val_lin_list = [], []
        err_nn_list, val_nn_list = [], []

        for rep in range(n_repeats):
            print(f"[Scenario {scenario}] Repeat {rep+1}/{n_repeats}")

            train_seed = 42 + rep
            X_tr, A_tr, R_tr, opt_tr = simulate_data(
                K=K, n=n, p=p, scenario=scenario, seed=train_seed, smooth=smooth
            )


            # Linear AD
            lin_model, V_lin, lin_pred = ad_linear_group(X_tr, A_tr, R_tr, K=K, alpha=alpha)

            # NN AD
            if k_active:
                if poly:
                    nn_model, V_nn, nn_pred = ad_nn_with_warmup_poly(X_tr, A_tr, R_tr, K=K, epochs=epochs, warmup_epochs=warmup_epochs, 
                                            lr=lr, hidden=hidden, k_active=k_active)
                else:
                    nn_model, V_nn, nn_pred = ad_nn_with_warmup(X_tr, A_tr, R_tr, K=K, epochs=epochs, warmup_epochs=warmup_epochs, 
                                                            lr=lr, hidden=hidden, k_active=k_active)
            else:
                nn_model, V_nn, nn_pred = ad_nn(X_tr, A_tr, R_tr, K=K, epochs=epochs, lr=lr, hidden=hidden)


            test_seed = 1_000_000 + train_seed
            X_te, A_te, R_te, opt_te = simulate_data(
                K=K, n=n_test, p=p, scenario=scenario, seed=test_seed, smooth=smooth
            )

            # Evaluate on test only
            err_lin, val_lin = evaluate_policy(lin_pred, X_te, A_te, R_te, K, true_opt=opt_te)
            err_nn,  val_nn  = evaluate_policy(nn_pred,  X_te, A_te, R_te, K, true_opt=opt_te)

            err_lin_list.append(err_lin); val_lin_list.append(val_lin)
            err_nn_list.append(err_nn);   val_nn_list.append(val_nn)

        results[scenario] = {
            "Linear": {
                "Error_mean": float(np.mean(err_lin_list)),
                "Error_var":  float(np.var(err_lin_list, ddof=1)),
                "Value_mean": float(np.mean(val_lin_list)),
                "Value_var":  float(np.var(val_lin_list, ddof=1)),
            },
            "NN": {
                "Error_mean": float(np.mean(err_nn_list)),
                "Error_var":  float(np.var(err_nn_list, ddof=1)),
                "Value_mean": float(np.mean(val_nn_list)),
                "Value_var":  float(np.var(val_nn_list, ddof=1)),
            }
        }
    
    return results


import numpy as np

def evaluate_policy_survival(
    predict_fn,
    X,
    A,
    T,
    Delta,
    K,
    scenario,
    main_effect=mu_default,
    smooth=False,
    true_opt=None,
):
    """
    생존 설정에서 정책 평가.
    """
    pred = predict_fn(X)

    error_rate = None if true_opt is None else np.mean(pred != true_opt)

    deltas = delta_scenario(X, scenario, K, smooth)
    mu = main_effect(X)                             
    lambda_mat = mu[:, None] + deltas                

    chosen_lambda = lambda_mat[np.arange(X.shape[0]), pred]
    value = np.mean(np.exp(chosen_lambda))

    return error_rate, value

def run_experiments_survival(
    K,
    p=10,
    n=2000,
    n_test=10000,
    scenario_list=[1, 2, 3],
    n_repeats=20,
    epochs=60,
    k_active=None,
    warmup_epochs=40,
    lr=1e-3,
    lam=0.05,
    step=1e-2,
    smooth=False,
    theta_cens=3.0,
    poly=False,
    hidden = [128,128]
):
    """
    생존 outcome용 실험 루프.
    """
    results = {}

    for scenario in scenario_list:
        err_lin_list, val_lin_list = [], []
        err_nn_list,  val_nn_list  = [], []

        for rep in range(n_repeats):
            print(f"[Scenario {scenario}] Repeat {rep+1}/{n_repeats}")

            train_seed = 42 + rep
            X_tr, A_tr, T_tr, Delta_tr, opt_tr, r_true, c = simulate_data_surv(
                K=K,
                n=n,
                p=p,
                scenario=scenario,
                seed=train_seed,
                theta_cens=theta_cens,
                smooth=smooth,
            )

            # Linear AD (survival)
            lin_model, V_lin, lin_pred = ad_linear_group_survival(
                X_tr, A_tr, T_tr, Delta_tr, K=K,
                lam=lam, step=step, max_iter=1000, tol=1e-6, pi=None
            )

            #  NN AD (survival)
            if k_active:
                if poly:
                    nn_model, V_nn, nn_pred = ad_nn_survival_with_warmup_poly(
                        X_tr, A_tr, T_tr, Delta_tr, K=K,
                        epochs=epochs, warmup_epochs=warmup_epochs, lr=lr, pi=None, hidden=hidden, k_active=k_active
                    )
                else:
                    nn_model, V_nn, nn_pred = ad_nn_survival_with_warmup(
                        X_tr, A_tr, T_tr, Delta_tr, K=K,
                        epochs=epochs, warmup_epochs=warmup_epochs, lr=lr, pi=None, hidden=hidden, k_active=k_active
                    )
            else:
                nn_model, V_nn, nn_pred = ad_nn_survival(
                    X_tr, A_tr, T_tr, Delta_tr, K=K,
                    epochs=epochs, lr=lr, pi=None, hidden=hidden
                )


            test_seed = 1_000_000 + train_seed
            X_te, A_te, T_te, Delta_te, opt_te, r_true, c = simulate_data_surv(
                K=K,
                n=n_test,
                p=p,
                scenario=scenario,
                seed=test_seed,
                theta_cens=theta_cens,
                smooth=smooth,
            )


            err_lin, val_lin = evaluate_policy_survival(
                lin_pred, X_te, A_te, T_te, Delta_te,
                K=K, scenario=scenario,
                main_effect=mu_default,
                smooth=smooth,
                true_opt=opt_te,
            )
            err_nn,  val_nn  = evaluate_policy_survival(
                nn_pred,  X_te, A_te, T_te, Delta_te,
                K=K, scenario=scenario,
                main_effect=mu_default,
                smooth=smooth,
                true_opt=opt_te,
            )

            err_lin_list.append(err_lin); val_lin_list.append(val_lin)
            err_nn_list.append(err_nn);   val_nn_list.append(val_nn)

        results[scenario] = {
            "Linear": {
                "Error_mean": float(np.mean(err_lin_list)),
                "Error_var":  float(np.var(err_lin_list, ddof=1)),
                "Value_mean": float(np.mean(val_lin_list)),
                "Value_var":  float(np.var(val_lin_list, ddof=1)),
            },
            "NN": {
                "Error_mean": float(np.mean(err_nn_list)),
                "Error_var":  float(np.var(err_nn_list, ddof=1)),
                "Value_mean": float(np.mean(val_nn_list)),
                "Value_var":  float(np.var(val_nn_list, ddof=1)),
            }
        }

    return results

