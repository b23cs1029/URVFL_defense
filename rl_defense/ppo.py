"""Minimal PPO (clipped objective + GAE) for the discrete defence policy."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)

    def forward(self, x):
        z = self.body(x)
        return self.pi(z), self.v(z).squeeze(-1)


class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x, np.float64)
        if x.ndim == len(self.mean.shape):
            x = x[None]
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        d = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + d * bc / tot
        m2 = self.var * self.count + bv * bc + d ** 2 * self.count * bc / tot
        self.var = m2 / tot
        self.count = tot

    @property
    def std(self):
        return np.sqrt(self.var)


def gae(rewards, values, last_value, dones, gamma, lam):
    T = len(rewards)
    adv = np.zeros(T, np.float32)
    last = 0.0
    for t in reversed(range(T)):
        nv = last_value if t == T - 1 else values[t + 1]
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * nv * nonterm - values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + np.asarray(values, np.float32)


def ppo_update(net, opt, obs, acts, logp_old, adv, ret, epochs=4, mb=256, clip=0.2,
               ent_coef=0.01, vf_coef=0.5, device="cpu"):
    obs = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device)
    acts = torch.as_tensor(np.asarray(acts), dtype=torch.long, device=device)
    logp_old = torch.as_tensor(np.asarray(logp_old), dtype=torch.float32, device=device)
    adv = torch.as_tensor(adv, dtype=torch.float32, device=device)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    ret = torch.as_tensor(ret, dtype=torch.float32, device=device)
    n = len(obs)
    stats = []
    for _ in range(epochs):
        idx = torch.randperm(n, device=device)
        for s in range(0, n, mb):
            b = idx[s:s + mb]
            logits, v = net(obs[b])
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(acts[b])
            ratio = torch.exp(logp - logp_old[b])
            pg = -torch.min(ratio * adv[b], torch.clamp(ratio, 1 - clip, 1 + clip) * adv[b]).mean()
            vl = F.mse_loss(v, ret[b])
            ent = dist.entropy().mean()
            loss = pg + vf_coef * vl - ent_coef * ent
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            stats.append((pg.item(), vl.item(), ent.item()))
    return np.mean(stats, 0)
