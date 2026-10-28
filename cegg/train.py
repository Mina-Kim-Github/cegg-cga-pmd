"""CGA-PMD: the training loop of Algorithm 1 and greedy evaluation.

The policy is a softmax of the learned per-node advantage scaled by
eta_k = beta_k / lambda_k, with beta_k = k+1 and lambda_k = lambda (k+1)^{3/2}.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from .env import CoupledEnv, build_state_tensor, build_coupling_matrix, build_coupling_matrix_inv
from .model import CGAAdvantageNet

DEVICE = torch.device('cpu')
MAX_EPISODES = 1000
PATIENCE = 100

def train_cga_pmd(G_D, G_S, phi, env_params, lr=3e-4, gamma=0.99, lam_scale=1.0):
    env = CoupledEnv(G_D, G_S, phi, **env_params)
    x_d, _ = build_state_tensor(G_D, np.zeros(env.N_D))
    x_s, _ = build_state_tensor(G_S, np.zeros(env.N_S))
    C_d2s = build_coupling_matrix(env); C_s2d = build_coupling_matrix_inv(env)
    psi_D = CGAAdvantageNet(x_d.shape[1], x_s.shape[1]).to(DEVICE)
    psi_S = CGAAdvantageNet(x_s.shape[1], x_d.shape[1]).to(DEVICE)
    opt = optim.Adam(list(psi_D.parameters()) + list(psi_S.parameters()), lr=lr)
    beta_sum = 0.0
    best_reward = -1e18; best_pD = None; best_pS = None; no_improve = 0

    for ep in range(MAX_EPISODES):
        env.reset()
        beta_k = float(ep + 1); lambda_k = lam_scale * (ep + 1)**1.5; beta_sum += beta_k
        lp_D = []; lp_S = []; values = []; rewards_total = []
        adv_scores_D = []; adv_scores_S = []
        done = False
        while not done:
            x_d, adj_d = build_state_tensor(G_D, env.mask_D)
            x_s, adj_s = build_state_tensor(G_S, env.mask_S)
            scores_D, val_D = psi_D(x_d, adj_d, x_s, adj_s, C_d2s)
            scores_S, val_S = psi_S(x_s, adj_s, x_d, adj_d, C_s2d)
            adv_scores_D.append(scores_D); adv_scores_S.append(scores_S)
            avail_D = [i for i in range(env.N_D) if env.mask_D[i] == 0]
            if avail_D:
                lg = torch.cat([scores_D[i].unsqueeze(0) for i in avail_D] + [torch.tensor([-2.0])])
                lg = lg * beta_k / lambda_k
                d = Categorical(logits=lg); idx = d.sample()
                action_D = [avail_D[idx.item()]] if idx.item() < len(avail_D) else []
                lp_D.append(d.log_prob(idx))
            else: action_D = []; lp_D.append(torch.tensor(0.0))
            avail_S = [i for i in range(env.N_S) if env.mask_S[i] == 0]
            if avail_S:
                lg = torch.cat([scores_S[i].unsqueeze(0) for i in avail_S] + [torch.tensor([-2.0])])
                lg = lg * beta_k / lambda_k
                d = Categorical(logits=lg); idx = d.sample()
                action_S = [avail_S[idx.item()]] if idx.item() < len(avail_S) else []
                lp_S.append(d.log_prob(idx))
            else: action_S = []; lp_S.append(torch.tensor(0.0))
            values.append(((val_D + val_S) / 2).item())
            _, r1, r2, done, _ = env.step(action_D, action_S)
            rewards_total.append(r1 + r2)

        total_W = sum(rewards_total)
        if total_W > best_reward:
            best_reward = total_W; no_improve = 0
            best_pD = copy.deepcopy(psi_D.state_dict())
            best_pS = copy.deepcopy(psi_S.state_dict())
        else: no_improve += 1
        if no_improve >= PATIENCE: break

        if lp_D:
            returns = []; G_t = 0
            for r in reversed(rewards_total): G_t = r + gamma * G_t; returns.insert(0, G_t)
            returns = torch.FloatTensor(returns); vals_t = torch.FloatTensor(values)
            adv = returns - vals_t
            if adv.std() > 1e-8: adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            lpD = torch.stack(lp_D); lpS = torch.stack(lp_S)
            pg_w = beta_k / beta_sum
            ploss = -(pg_w * (lpD + lpS) * adv).mean()
            pred_v = []
            for t in range(len(adv_scores_D)):
                pred_v.append(adv_scores_D[t].mean() + adv_scores_S[t].mean())
            if pred_v:
                pv = torch.stack(pred_v[:len(returns)])
                vloss = F.mse_loss(pv, returns[:len(pv)])
            else: vloss = torch.tensor(0.0)
            loss = ploss + 0.5 * vloss
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(list(psi_D.parameters()) + list(psi_S.parameters()), 0.5)
            opt.step()

    if best_pD: psi_D.load_state_dict(best_pD)
    if best_pS: psi_S.load_state_dict(best_pS)
    return psi_D, psi_S, ep + 1

def evaluate_cga_pmd(psi_D, psi_S, G_D, G_S, phi, env_params, n_trials=10):
    """Evaluate CGA+PDA (greedy based on advantage scores)."""
    env = CoupledEnv(G_D, G_S, phi, **env_params)
    C_d2s = build_coupling_matrix(env)
    C_s2d = build_coupling_matrix_inv(env)
    rewards = []
    for _ in range(n_trials):
        env.reset(); done = False; ep_r = 0
        psi_D.eval(); psi_S.eval()
        with torch.no_grad():
            while not done:
                x_d, adj_d = build_state_tensor(G_D, env.mask_D)
                x_s, adj_s = build_state_tensor(G_S, env.mask_S)
                scores_D, _ = psi_D(x_d, adj_d, x_s, adj_s, C_d2s)
                avail_D = [i for i in range(env.N_D) if env.mask_D[i] == 0]
                action_D = [avail_D[scores_D[avail_D].argmax().item()]] if avail_D else []
                scores_S, _ = psi_S(x_s, adj_s, x_d, adj_d, C_s2d)
                avail_S = [i for i in range(env.N_S) if env.mask_S[i] == 0]
                action_S = [avail_S[scores_S[avail_S].argmax().item()]] if avail_S else []
                _, r1, r2, done, _ = env.step(action_D, action_S)
                ep_r += r1 + r2
        rewards.append(ep_r)
        psi_D.train(); psi_S.train()
    return np.mean(rewards), np.std(rewards)
