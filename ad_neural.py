from data_gen import *
import torch.nn.functional as F

class FeatureGate(nn.Module):
    def __init__(self, p, k_active=None, init_logit=0.0):
        super().__init__()
        self.logit = nn.Parameter(torch.full((p,), init_logit))
        self.k_active = k_active  
    def forward(self, x):
        if self.k_active is None:
            # warmup step without featureGate
            return x

        prob = torch.sigmoid(self.logit)  

        if self.k_active >= prob.numel():
            gate = prob
        else:
            k = self.k_active
            topk_vals, topk_idx = torch.topk(prob, k)
            mask = torch.zeros_like(prob)
            mask[topk_idx] = 1.0
            gate = prob + (mask - prob).detach()

        return x * gate

    def get_prob(self):
        with torch.no_grad():
            return torch.sigmoid(self.logit).cpu().numpy()

class PolyFeatureGate(nn.Module):
    def __init__(self, degree=2):
        super().__init__()
        self.degree = degree

    def forward(self, x):
        feats = [x]
        if self.degree >= 2:
            feats.append(x ** 2)

        return torch.cat(feats, dim=1)  



class ADNet_feature(nn.Module):
    def __init__(self, p, out_dim, hidden=[128,128], k_active=None):
        super().__init__()
        self.gate = FeatureGate(p, k_active=k_active) 

        layers = []
        d = p
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.gate(x)         
        return self.net(x)

    def get_gate_values(self):
        """학습 후 feature importance 확인용"""
        with torch.no_grad():
            g = torch.sigmoid(self.gate.logit)
        return g.cpu().numpy()

class ADNetPoly(nn.Module):
    def __init__(self, p, out_dim, hidden=[128,128],
                 degree=2, k_active=None):
        super().__init__()
        self.p = p
        self.degree = degree

        self.gate = FeatureGate(p, k_active=k_active)
        self.poly = PolyFeatureGate(degree=degree)
        d_in = p * (2 if degree >= 2 else 1)

        layers = []
        d = d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.gate(x)       
        z = self.poly(x)       
        return self.net(z)

    
class ADNet(nn.Module):
    def __init__(self, p, out_dim, hidden=[128,128]):
        super().__init__()
        layers = []
        d = p
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)



def ad_nn_vector_target(X, A, R, K=4, epochs=60, lr=1e-3, hidden=[128,128]):
    """
    AD-NN (vector regression version, target = K*R*w_A)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n, p = X.shape
    V = torch.tensor(simplex_vertices(K), dtype=torch.float32, device=device)  # (K, K-1)

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    A_t = torch.tensor(A, dtype=torch.long, device=device)
    R_t = torch.tensor(R, dtype=torch.float32, device=device)

    # target: Y = K * R * w_A
    Y_t = (K * R_t[:, None]) * V[A_t]  

    model = ADNet(p=p, out_dim=K-1, hidden=hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train(); opt.zero_grad()
        f = model(X_t)                 
        loss = (((Y_t - f) ** 2)/(K-1)).mean() 
        loss.backward(); opt.step()
        if (ep+1) % 10 == 0:
            print(f"[NN-vec] epoch {ep+1}/{epochs} loss={loss.item():.4f}")

    def predict(X_new):
        X_new_t = torch.tensor(X_new, dtype=torch.float32, device=device)
        with torch.no_grad():
            f = model(X_new_t)    
            scores = f @ V.T         
            return scores.argmax(1).cpu().numpy()

    return model, V.cpu().numpy(), predict


def ad_nn(X, A, R, K, epochs=50, lr=1e-3, hidden=[128,128]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n, p = X.shape
    V = simplex_vertices(K).to(device)     

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    A_t = torch.tensor(A, dtype=torch.long, device=device)
    R_t = torch.tensor(R, dtype=torch.float32, device=device)

    model = ADNet(p, K-1, hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train(); opt.zero_grad()
        f = model(X_t)                        
        wA = V[A_t]                            
        pred = (f*wA).sum(1)                       
        loss = ((R_t - pred)**2).mean()

        loss.backward(); opt.step()
        if (ep+1) % 10 == 0:
            print(f"[NN] epoch {ep+1}/{epochs} loss={loss.item():.4f}")

    def predict(X_new):
        X_new_t = torch.tensor(X_new, dtype=torch.float32, device=device)
        with torch.no_grad():
            f = model(X_new_t)                  
            scores = f @ V.T                 
            return scores.argmax(1).cpu().numpy()
    return model, V.cpu().numpy(), predict

def ad_nn_feature(X, A, R, K, epochs=50, lr=1e-3, hidden=[128,128], k_active=10,
          lam_gate=1e-1):   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n, p = X.shape
    V = simplex_vertices(K).to(device)     

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    A_t = torch.tensor(A, dtype=torch.long, device=device)
    R_t = torch.tensor(R, dtype=torch.float32, device=device)

    model = ADNet_feature(p, K-1, hidden, k_active=k_active).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        opt.zero_grad()

        f = model(X_t)                
        wA = V[A_t]                      
        pred = (f * wA).sum(1)       
        mse = ((R_t - pred) ** 2).mean()
        loss = mse

        loss.backward()
        opt.step()

        if (ep+1) % 10 == 0:
            prob = torch.sigmoid(model.gate.logit).detach().cpu()
            hard = (prob >= prob.topk(k_active).values.min()).float()
            print(f"[ep {ep+1}] loss={loss.item():.4f}, "
                  f"mse={mse.item():.4f}, "
                  f"active_idx={hard.nonzero(as_tuple=True)[0].numpy()}")

    def predict(X_new):
        X_new_t = torch.tensor(X_new, dtype=torch.float32, device=device)
        with torch.no_grad():
            f = model(X_new_t)     
            scores = f @ V.T        
            return scores.argmax(1).cpu().numpy()

    return model, V.cpu().numpy(), predict

def ad_nn_with_warmup(
    X, A, R, K,
    epochs=80,
    warmup_epochs=40,     
    k_active=10,
    lr=1e-3,
    hidden=[128,128],
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n, p = X.shape
    V = simplex_vertices(K).to(device)

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    A_t = torch.tensor(A, dtype=torch.long, device=device)
    R_t = torch.tensor(R, dtype=torch.float32, device=device)

    model = ADNet_feature(p, K-1, hidden, k_active=k_active).to(device)
    model.gate.k_active = None

    opt = optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        opt.zero_grad()

        # warm-up이 끝나는 시점에 gate 초기화 & ON 
        if ep == warmup_epochs:
            with torch.no_grad():
                # 첫 Linear layer weight로 feature 중요도 계산
                fc1 = model.net[0]               
                W = fc1.weight                  
                scores = torch.sqrt((W ** 2).sum(dim=0)) 
                # 상위 k_active feature 인덱스
                topk = scores.topk(k_active).indices
                a = 3.0
                model.gate.logit.data[:] = -a
                model.gate.logit.data[topk] = a

                model.gate.k_active = k_active

            print(f"==> warm-up done, gate ON, k_active={k_active}, "
                  f"init topk={topk.cpu().numpy()}")

        f = model(X_t)
        wA = V[A_t]
        pred = (f * wA).sum(1)
        mse = ((R_t - pred) ** 2).mean()
        loss = mse

        loss.backward()
        opt.step()

        if (ep + 1) % 10 == 0:
            if model.gate.k_active is None:
                print(f"[ep {ep+1}] warmup, loss={loss.item():.4f}")
            else:
                prob = torch.sigmoid(model.gate.logit).detach().cpu()
                thr = prob.topk(k_active).values.min()
                active_idx = (prob >= thr).nonzero(as_tuple=True)[0].numpy()
                print(f"[ep {ep+1}] loss={loss.item():.4f}, "
                      f"active_idx={active_idx}")

    def predict(X_new):
        X_new_t = torch.tensor(X_new, dtype=torch.float32, device=device)
        with torch.no_grad():
            f = model(X_new_t)
            scores = f @ V.T
            return scores.argmax(1).cpu().numpy()

    return model, V.cpu().numpy(), predict

def ad_nn_with_warmup_poly(
    X, A, R, K,
    epochs=80,
    warmup_epochs=40,     
    k_active=10,
    lr=1e-3,
    hidden=[128,128],
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n, p = X.shape
    V = simplex_vertices(K).to(device)

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    A_t = torch.tensor(A, dtype=torch.long, device=device)
    R_t = torch.tensor(R, dtype=torch.float32, device=device)

    model = ADNetPoly(p, K-1, hidden, k_active=k_active).to(device)
    model.gate.k_active = None

    opt = optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        opt.zero_grad()

        # ====== warm-up이 끝나는 시점에 gate 초기화 & ON ======
        if ep == warmup_epochs:
            with torch.no_grad():
                fc1 = model.net[0]         
                W = fc1.weight              

                p = model.gate.logit.shape[0]
                d_in = W.shape[1]

                if d_in == p:
                    scores = torch.sqrt((W ** 2).sum(dim=0))   

                elif d_in == 2 * p:
                    W_lin  = W[:, :p]    
                    W_quad = W[:, p:]      
                    # 변수 j의 중요도 = linear+quadratic weight를 함께 본 L2 norm
                    scores = torch.sqrt((W_lin ** 2).sum(dim=0) +
                                        (W_quad ** 2).sum(dim=0))  

                else:
                    raise ValueError(f"Unexpected first-layer dim: d_in={d_in}, p={p}")

                topk = scores.topk(k_active).indices

                a = 3.0
                model.gate.logit.data[:] = -a
                model.gate.logit.data[topk] = a

                model.gate.k_active = k_active

            print(f"==> warm-up done, gate ON, k_active={k_active}, init topk={topk.cpu().numpy()}")


        f = model(X_t)
        wA = V[A_t]
        pred = (f * wA).sum(1)
        mse = ((R_t - pred) ** 2).mean()
        loss = mse

        loss.backward()
        opt.step()

        if (ep + 1) % 10 == 0:
            if model.gate.k_active is None:
                print(f"[ep {ep+1}] warmup, loss={loss.item():.4f}")
            else:
                prob = torch.sigmoid(model.gate.logit).detach().cpu()
                thr = prob.topk(k_active).values.min()
                active_idx = (prob >= thr).nonzero(as_tuple=True)[0].numpy()
                print(f"[ep {ep+1}] loss={loss.item():.4f}, "
                      f"active_idx={active_idx}")

    def predict(X_new):
        X_new_t = torch.tensor(X_new, dtype=torch.float32, device=device)
        with torch.no_grad():
            f = model(X_new_t)
            scores = f @ V.T
            return scores.argmax(1).cpu().numpy()

    return model, V.cpu().numpy(), predict



def ad_nn_survival(X, A, T, Delta, K, epochs=50, lr=1e-3, pi=None, hidden=[128,128]):
    """
    AD-learning 신경망 (생존 outcome, Cox partial likelihood)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = np.asarray(X, dtype=np.float32)
    A = np.asarray(A, dtype=np.int64)
    T = np.asarray(T, dtype=np.float32)
    Delta = np.asarray(Delta, dtype=np.float32)

    n, p = X.shape

    V = simplex_vertices(K).to(device)  

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    A_t = torch.tensor(A, dtype=torch.long, device=device)
    T_t = torch.tensor(T, dtype=torch.float32, device=device)
    Delta_t = torch.tensor(Delta, dtype=torch.float32, device=device)

    if pi is None:
        pi = np.full(n, 1.0 / K, dtype=np.float32)
    pi_t = torch.tensor(pi, dtype=torch.float32, device=device)

    model = ADNet(p, K-1, hidden=hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    def cox_ipw_loss(eta, T_t, Delta_t, pi_t):
        """
        eta : (n,) log-hazard for actually received treatment
        """
        n = eta.shape[0]
        order = torch.argsort(T_t)
        T_s = T_t[order]
        eta_s = eta[order]
        Delta_s = Delta_t[order]
        pi_s = pi_t[order]

        exp_eta = torch.exp(eta_s)
        cum_exp_rev = torch.cumsum(exp_eta.flip(0), dim=0).flip(0)

        # event만 골라서 계산
        event_mask = (Delta_s > 0)
        idx = torch.nonzero(event_mask, as_tuple=False).squeeze()

        eta_evt = eta_s[idx]
        log_risk_evt = torch.log(cum_exp_rev[idx])
        w_evt = 1.0 / pi_s[idx]    # IPW

        loss = - (w_evt * (eta_evt - log_risk_evt)).sum() / n
        return loss
    
    def cox_ipw_loss_2(eta, T_t, Delta_t, pi_t):
        """
        eta : (n,) log-hazard for actually received treatment
        """
        n = eta.shape[0]

        order   = torch.argsort(T_t)
        T_s     = T_t[order]
        eta_s   = eta[order]
        Delta_s = Delta_t[order]
        pi_s    = pi_t[order]

        c = torch.max(eta_s)             
        eta_centered = eta_s - c             
        exp_eta = torch.exp(eta_centered)    # overflow 방지

        cum_exp_rev = torch.cumsum(exp_eta.flip(0), dim=0).flip(0)

        # event만 골라서 계산
        event_mask = (Delta_s > 0)
        idx = torch.nonzero(event_mask, as_tuple=False).squeeze()

        if idx.numel() == 0:
            # 이벤트가 없으면 loss=0 (혹은 아주 작은 상수)
            return torch.tensor(0.0, device=eta.device)

        eta_evt = eta_s[idx]
        log_risk_evt = torch.log(cum_exp_rev[idx] + 1e-12) + c

        w_evt = 1.0 / pi_s[idx]    # IPW 

        loss = - (w_evt * (eta_evt - log_risk_evt)).sum() / n
        return loss

    for ep in range(epochs):
        model.train()
        opt.zero_grad()

        f = model(X_t)              
        scores = f @ V.T            
        eta = scores[torch.arange(n), A_t]  

        loss = cox_ipw_loss_2(eta, T_t, Delta_t, pi_t)

        loss.backward()
        opt.step()

        if (ep + 1) % 10 == 0:
            print(f"[AD-NN Survival] epoch {ep+1}/{epochs} loss={loss.item():.4f}")

    def predict(X_new):
        X_new = np.asarray(X_new, dtype=np.float32)
        X_new_t = torch.tensor(X_new, dtype=torch.float32, device=device)
        with torch.no_grad():
            f_new = model(X_new_t)          
            scores_new = f_new @ V.T     
            pred = torch.argmin(scores_new, dim=1)
        return pred.cpu().numpy()

    return model, V.cpu().numpy(), predict


def ad_nn_survival_with_warmup(
    X, A, T, Delta, K,
    epochs=80,
    warmup_epochs=40,     
    k_active=10,          
    lr=1e-3,
    hidden=[128,128],
    pi=None,
    a_init=3.0,           
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = np.asarray(X, dtype=np.float32)
    A = np.asarray(A, dtype=np.int64)
    T = np.asarray(T, dtype=np.float32)
    Delta = np.asarray(Delta, dtype=np.float32)
    n, p = X.shape

    V = simplex_vertices(K).to(device)

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    A_t = torch.tensor(A, dtype=torch.long, device=device)
    T_t = torch.tensor(T, dtype=torch.float32, device=device)
    Delta_t = torch.tensor(Delta, dtype=torch.float32, device=device)

    if pi is None:
        pi = np.full(n, 1.0/K, dtype=np.float32)
    pi_t = torch.tensor(pi, dtype=torch.float32, device=device)

    model = ADNet_feature(p, K-1, hidden=hidden, k_active=None).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    def cox_ipw_loss(eta, T_t, Delta_t, pi_t):
        n = eta.shape[0]
        order = torch.argsort(T_t)
        T_s = T_t[order]
        eta_s = eta[order]
        Delta_s = Delta_t[order]
        pi_s = pi_t[order]

        c = torch.max(eta_s)
        eta_centered = eta_s - c
        exp_eta = torch.exp(eta_centered)
        cum_exp_rev = torch.cumsum(exp_eta.flip(0), dim=0).flip(0)

        idx = torch.nonzero(Delta_s > 0, as_tuple=False).squeeze()
        if idx.numel() == 0:
            return torch.tensor(0.0, device=eta.device)

        eta_evt = eta_s[idx]
        log_risk_evt = torch.log(cum_exp_rev[idx] + 1e-12) + c
        w_evt = 1.0 / pi_s[idx]

        loss = - (w_evt * (eta_evt - log_risk_evt)).sum() / n
        return loss


    for ep in range(epochs):
        model.train()
        opt.zero_grad()

        if ep == warmup_epochs:

            with torch.no_grad():
                fc1 = model.net[0]      
                W = fc1.weight      
                scores = torch.sqrt((W**2).sum(dim=0)) 

                topk_idx = scores.topk(k_active).indices

                model.gate.logit.data[:] = -a_init
                model.gate.logit.data[topk_idx] = a_init

                model.gate.k_active = k_active

            print(f"==> Warm-up done. Gate ON with k={k_active}. "
                  f"initial topk={topk_idx.cpu().numpy()}")

        f = model(X_t)               
        scores = f @ V.T        
        eta = scores[torch.arange(n), A_t]  
        loss = cox_ipw_loss(eta, T_t, Delta_t, pi_t)

        loss.backward()
        opt.step()

        if (ep + 1) % 10 == 0:
            if model.gate.k_active is None:
                print(f"[Surv-NN] warmup ep {ep+1}/{epochs} loss={loss.item():.4f}")
            else:
                prob = torch.sigmoid(model.gate.logit).detach().cpu()
                thr = prob.topk(k_active).values.min()
                active = (prob >= thr).nonzero(as_tuple=True)[0].numpy()
                print(f"[Surv-NN] ep {ep+1}/{epochs} loss={loss.item():.4f}, "
                      f"active_idx={active}")

    def predict(X_new):
        X_new = np.asarray(X_new, dtype=np.float32)
        X_new_t = torch.tensor(X_new, dtype=torch.float32, device=device)
        with torch.no_grad():
            f_new = model(X_new_t)
            scores_new = f_new @ V.T
            return torch.argmin(scores_new, dim=1).cpu().numpy()

    return model, V.cpu().numpy(), predict


def ad_nn_survival_with_warmup_poly(
    X, A, T, Delta, K,
    epochs=80,
    warmup_epochs=40,      
    k_active=10,           
    lr=1e-3,
    hidden=[128,128],
    pi=None,
    a_init=3.0,        
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = np.asarray(X, dtype=np.float32)
    A = np.asarray(A, dtype=np.int64)
    T = np.asarray(T, dtype=np.float32)
    Delta = np.asarray(Delta, dtype=np.float32)
    n, p = X.shape

    V = simplex_vertices(K).to(device)

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    A_t = torch.tensor(A, dtype=torch.long, device=device)
    T_t = torch.tensor(T, dtype=torch.float32, device=device)
    Delta_t = torch.tensor(Delta, dtype=torch.float32, device=device)

    if pi is None:
        pi = np.full(n, 1.0/K, dtype=np.float32)
    pi_t = torch.tensor(pi, dtype=torch.float32, device=device)

    model = ADNetPoly(p, K-1, hidden=hidden, k_active=None).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    def cox_ipw_loss(eta, T_t, Delta_t, pi_t):
        n = eta.shape[0]
        order = torch.argsort(T_t)
        T_s = T_t[order]
        eta_s = eta[order]
        Delta_s = Delta_t[order]
        pi_s = pi_t[order]

        # log-sum-exp 안정화
        c = torch.max(eta_s)
        eta_centered = eta_s - c
        exp_eta = torch.exp(eta_centered)
        cum_exp_rev = torch.cumsum(exp_eta.flip(0), dim=0).flip(0)

        idx = torch.nonzero(Delta_s > 0, as_tuple=False).squeeze()
        if idx.numel() == 0:
            return torch.tensor(0.0, device=eta.device)

        eta_evt = eta_s[idx]
        log_risk_evt = torch.log(cum_exp_rev[idx] + 1e-12) + c
        w_evt = 1.0 / pi_s[idx]

        loss = - (w_evt * (eta_evt - log_risk_evt)).sum() / n
        return loss


    for ep in range(epochs):
        model.train()
        opt.zero_grad()

        # warm-up  gate OFF
        if ep == warmup_epochs:
            with torch.no_grad():
                fc1 = model.net[0]        
                W = fc1.weight          

                p_feat = model.gate.logit.shape[0]  
                d_in = W.shape[1]

                if d_in == p_feat:
                    scores = torch.sqrt((W ** 2).sum(dim=0))  

                elif d_in == 2 * p_feat:
                    W_lin  = W[:, :p_feat]     
                    W_quad = W[:, p_feat:]      
                    scores = torch.sqrt((W_lin ** 2).sum(dim=0) +
                                        (W_quad ** 2).sum(dim=0))  

                else:
                    raise ValueError(f"Unexpected first-layer dim: d_in={d_in}, p={p_feat}")

                k_eff = min(k_active, p_feat)

                topk_idx = scores.topk(k_eff).indices

                model.gate.logit.data[:] = -a_init
                model.gate.logit.data[topk_idx] = a_init

                model.gate.k_active = k_eff

            print(f"==> Warm-up done. Gate ON with k={k_eff}. "
                f"initial topk={topk_idx.cpu().numpy()}")


        f = model(X_t)             
        scores = f @ V.T          
        eta = scores[torch.arange(n), A_t]  
        loss = cox_ipw_loss(eta, T_t, Delta_t, pi_t)

        loss.backward()
        opt.step()

        if (ep + 1) % 10 == 0:
            if model.gate.k_active is None:
                print(f"[Surv-NN] warmup ep {ep+1}/{epochs} loss={loss.item():.4f}")
            else:
                prob = torch.sigmoid(model.gate.logit).detach().cpu()
                thr = prob.topk(k_active).values.min()
                active = (prob >= thr).nonzero(as_tuple=True)[0].numpy()
                print(f"[Surv-NN] ep {ep+1}/{epochs} loss={loss.item():.4f}, "
                      f"active_idx={active}")


    def predict(X_new):
        X_new = np.asarray(X_new, dtype=np.float32)
        X_new_t = torch.tensor(X_new, dtype=torch.float32, device=device)
        with torch.no_grad():
            f_new = model(X_new_t)
            scores_new = f_new @ V.T
            return torch.argmin(scores_new, dim=1).cpu().numpy()

    return model, V.cpu().numpy(), predict
