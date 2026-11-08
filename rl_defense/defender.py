"""Passive-client defence layer.

The passive client owns ``target_model``.  The only thing an (honest or malicious)
active party can do to it is send back a gradient w.r.t. the embedding the passive
client uploaded.  This module sits between the received gradient and the passive
model update (PDF, section 5):

    embedding h  --(upload)-->  active party
    gradient g   <--(download)-- active party
           |
           v
    PassiveDefender: build state s_t -> policy -> action a_t
           |
           v
    transformed gradient / update  -->  target_opt.step()

Integration is non-invasive:
  * ``DefendedPassiveModel`` wraps the passive bottom model and registers a backward
    hook on every embedding it produces in training mode.  The hook calls the
    defender, which can transform the incoming gradient.
  * ``DefendedOptimizer`` wraps the passive optimizer.  ``step()`` applies the
    update-level part of the action (skip for Reject, scaled update for Downweight,
    norm-capped update for Clip).

The update-level part matters because the passive model is trained with Adam,
which is (approximately) invariant to a constant rescaling of the gradient: only
scaling the gradient would leave the attack speed unchanged.
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACCEPT, DOWNWEIGHT, CLIP, REJECT = 0, 1, 2, 3
ACTION_NAMES = ["accept", "downweight", "clip", "reject"]
N_ACTIONS = 4


# ----------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------

def _center_gram(k):
    n = k.shape[0]
    h = torch.eye(n, device=k.device, dtype=k.dtype) - 1.0 / n
    return h @ k @ h


def linear_cka_from_grams(k, l):
    kc, lc = _center_gram(k), _center_gram(l)
    num = (kc * lc).sum()
    den = kc.norm() * lc.norm() + 1e-12
    return (num / den).item()


def linear_cka(x, y):
    """Linear CKA between two batches of representations (B, ...)."""
    x = x.reshape(x.shape[0], -1).float()
    y = y.reshape(y.shape[0], -1).float()
    return linear_cka_from_grams(x @ x.t(), y @ y.t())


# Names of the entries of the observation vector (kept for logging / the report).
FEATURE_NAMES = [
    "log_grad_norm",          # log mean per-sample gradient norm
    "grad_norm_cv",           # coefficient of variation of per-sample norms
    "log_norm_ratio_hist",    # log(norm / EMA norm)  -> historical behaviour
    "cos_mean_grad_hist",     # cos(batch-mean gradient, EMA batch-mean gradient)
    "grad_pairwise_cos",      # mean pairwise cosine between per-sample gradients
    "cos_grad_emb",           # mean cos(g_i, h_i)
    "grad_highfreq_ratio",    # high-frequency share of the gradient energy
    "grad_channel_entropy",   # normalised entropy of per-channel gradient energy
    "log_grad_kurtosis",      # log kurtosis of gradient entries
    "grad_dead_unit_share",   # gradient energy on units where h == 0
    "cka_emb_ref",            # CKA(main embedding, reference embedding)
    "cka_emb_ref_trend",      # CKA(main, ref) - its EMA
    "cka_grad_emb",           # CKA(gradient Gram, embedding Gram)
    "cka_grad_ref",           # CKA(gradient Gram, reference Gram)
    "log_emb_norm_ratio",     # log(embedding norm / its EMA)  -> embedding drift
    "ema_cka_emb_ref_trend",  # slow EMA of the CKA trend
    "ema_grad_pairwise_cos",  # slow EMA of pairwise cosine
    "ema_cka_grad_ref",       # slow EMA of CKA(grad, ref)
    "ema_log_norm_ratio",     # slow EMA of the norm ratio
]
N_FEATURES = len(FEATURE_NAMES)
OBS_DIM = N_FEATURES + N_ACTIONS  # + one-hot of the previous action


class PassiveDefender:
    """Builds the RL state from an incoming gradient and applies the chosen action.

    policy: callable(obs: np.ndarray) -> int action, or None for "always accept".
            In RL training the environment installs a sampling callback here.
    reference_encoder: independent reference encoder (never receives gradients from
            the active party).  ``None`` disables the reference-based features.
    """

    def __init__(self, policy=None, reference_encoder=None, downweight=0.3,
                 clip_ratio=0.5, ema=0.9, slow_ema=0.98, normalizer=None,
                 record=True):
        self.policy = policy
        self.reference = reference_encoder
        self.downweight = downweight
        self.clip_ratio = clip_ratio
        self.ema = ema
        self.slow_ema = slow_ema
        self.normalizer = normalizer  # (mean, std) numpy arrays or None
        self.record = record
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self.ema_norm = None
        self.ema_mean_grad = None
        self.ema_cka = None
        self.ema_emb_norm = None
        self.ema_delta_norm = None
        self.slow = None
        self.prev_action = ACCEPT
        self.pending_action = None
        self.pending_update_scale = 1.0
        self.pending_update_cap = None
        self.last_obs = None
        self.last_raw = None
        self.log_actions = []
        self.log_features = []
        self.decision_time = 0.0
        self.n_decisions = 0
        self._cur_h = None
        self._cur_ref = None

    def observe_forward(self, x, h):
        """Called by the wrapped model on every training-mode forward pass."""
        self._cur_h = h.detach()
        if self.reference is not None:
            with torch.no_grad():
                was_training = self.reference.training
                self.reference.eval()
                self._cur_ref = self.reference(x.detach())
                self.reference.train(was_training)
        else:
            self._cur_ref = None

    @torch.no_grad()
    def _features(self, g):
        """All tensor work happens on the device; a single transfer to the CPU."""
        h = self._cur_h
        b = g.shape[0]
        g32 = g.float()
        gf = g32.reshape(b, -1)
        hf = h.reshape(b, -1).float()
        norms = gf.norm(dim=1) + 1e-12
        mean_norm = norms.mean()
        mg = gf.mean(0)
        if self.ema_mean_grad is None:
            self.ema_mean_grad = mg.clone()
        cos_hist = F.cosine_similarity(mg, self.ema_mean_grad, dim=0)
        self.ema_mean_grad.mul_(self.ema).add_((1 - self.ema) * mg)

        gn = gf / norms[:, None]
        cosm = gn @ gn.t()
        pair_cos = (cosm.sum() - cosm.diagonal().sum()) / (b * (b - 1))
        cos_ge = F.cosine_similarity(gf, hf + 1e-12, dim=1).mean()
        if g.dim() == 4 and g.shape[-1] % 2 == 0 and g.shape[-2] % 2 == 0:
            low = F.interpolate(F.avg_pool2d(g32, 2), scale_factor=2, mode="nearest")
            hfr = (g32 - low).pow(2).sum() / (g32.pow(2).sum() + 1e-12)
            ch = g32.pow(2).sum(dim=(0, 2, 3))
        else:
            hfr = torch.zeros((), device=g.device)
            ch = g32.pow(2).reshape(b, g.shape[1], -1).sum(dim=(0, 2))
        p = ch / (ch.sum() + 1e-12)
        ent = -(p * torch.log(p + 1e-12)).sum() / np.log(len(p))
        z = (gf - gf.mean()) / (gf.std() + 1e-12)
        kurt = torch.log(z.pow(4).mean() + 1e-12)
        g2 = gf.pow(2)
        dead = (g2 * (hf <= 0).float()).sum() / (g2.sum() + 1e-12)

        def cka(k, l):
            kc, lc = _center_gram(k), _center_gram(l)
            return (kc * lc).sum() / (kc.norm() * lc.norm() + 1e-12)

        gram_g, gram_h = gf @ gf.t(), hf @ hf.t()
        cka_gh = cka(gram_g, gram_h)
        if self._cur_ref is not None:
            rf = self._cur_ref.reshape(b, -1).float()
            gram_r = rf @ rf.t()
            cka_hr, cka_gr = cka(gram_h, gram_r), cka(gram_g, gram_r)
        else:
            cka_hr = cka_gr = torch.zeros((), device=g.device)
        en = hf.norm(dim=1).mean()
        v = torch.stack([mean_norm, norms.std() / mean_norm, cos_hist, pair_cos, cos_ge, hfr, ent,
                         kurt, dead, cka_hr, cka_gr, cka_gh, en]).cpu().numpy().astype(np.float64)
        (mean_norm, cv, cos_hist, pair_cos, cos_ge, hfr, ent, kurt, dead,
         cka_hr, cka_gr, cka_gh, en) = v

        f = {"log_grad_norm": np.log(mean_norm), "grad_norm_cv": cv}
        if self.ema_norm is None:
            self.ema_norm = mean_norm
        f["log_norm_ratio_hist"] = np.log(mean_norm / (self.ema_norm + 1e-12))
        self.ema_norm = self.ema * self.ema_norm + (1 - self.ema) * mean_norm
        f["cos_mean_grad_hist"] = cos_hist
        f["grad_pairwise_cos"] = pair_cos
        f["cos_grad_emb"] = cos_ge
        f["grad_highfreq_ratio"] = hfr
        f["grad_channel_entropy"] = ent
        f["log_grad_kurtosis"] = kurt
        f["grad_dead_unit_share"] = dead
        if self.ema_cka is None:
            self.ema_cka = cka_hr
        f["cka_emb_ref"] = cka_hr
        f["cka_emb_ref_trend"] = cka_hr - self.ema_cka
        self.ema_cka = self.ema * self.ema_cka + (1 - self.ema) * cka_hr
        f["cka_grad_emb"] = cka_gh
        f["cka_grad_ref"] = cka_gr
        if self.ema_emb_norm is None:
            self.ema_emb_norm = en
        f["log_emb_norm_ratio"] = np.log((en + 1e-12) / (self.ema_emb_norm + 1e-12))
        self.ema_emb_norm = self.ema * self.ema_emb_norm + (1 - self.ema) * en

        slow_keys = ["cka_emb_ref_trend", "grad_pairwise_cos", "cka_grad_ref", "log_norm_ratio_hist"]
        if self.slow is None:
            self.slow = {k: f[k] for k in slow_keys}
        else:
            for k in slow_keys:
                self.slow[k] = self.slow_ema * self.slow[k] + (1 - self.slow_ema) * f[k]
        f["ema_cka_emb_ref_trend"] = self.slow["cka_emb_ref_trend"]
        f["ema_grad_pairwise_cos"] = self.slow["grad_pairwise_cos"]
        f["ema_cka_grad_ref"] = self.slow["cka_grad_ref"]
        f["ema_log_norm_ratio"] = self.slow["log_norm_ratio_hist"]

        raw = np.array([f[k] for k in FEATURE_NAMES], dtype=np.float32)
        return raw, norms

    def build_obs(self, raw):
        x = raw.copy()
        if self.normalizer is not None:
            mean, std = self.normalizer
            x = np.clip((x - mean) / (std + 1e-6), -10, 10)
        onehot = np.zeros(N_ACTIONS, dtype=np.float32)
        onehot[self.prev_action] = 1.0
        return np.concatenate([x, onehot]).astype(np.float32)

    # ------------------------------------------------------------------ hook
    def on_gradient(self, g):
        """Backward hook on the passive embedding.  Returns the gradient that is
        actually allowed to flow into the passive model."""
        t0 = time.perf_counter()
        raw, norms = self._features(g)
        obs = self.build_obs(raw)
        self.last_raw, self.last_obs = raw, obs
        action = ACCEPT if self.policy is None else int(self.policy(obs))

        self.pending_action = action
        self.pending_update_scale = 1.0
        self.pending_update_cap = None
        if action == ACCEPT:
            out = g
        elif action == DOWNWEIGHT:
            out = g * self.downweight
            self.pending_update_scale = self.downweight
        elif action == CLIP:
            # per-sample clipping of the received gradient to clip_ratio * EMA norm ...
            c = self.clip_ratio * self.ema_norm
            factor = torch.clamp(c / norms, max=1.0).to(g.dtype)
            out = g * factor.view(-1, *([1] * (g.dim() - 1)))
            # ... and a cap on the resulting parameter-update norm
            if self.ema_delta_norm is not None:
                self.pending_update_cap = self.clip_ratio * self.ema_delta_norm
        elif action == REJECT:
            out = torch.zeros_like(g)
        else:
            raise ValueError(action)
        self.prev_action = action
        if self.record:
            self.log_actions.append(action)
            self.log_features.append(raw)
        self.decision_time += time.perf_counter() - t0
        self.n_decisions += 1
        return out


class DefendedPassiveModel(nn.Module):
    """Drop-in wrapper for the passive bottom model."""

    def __init__(self, model, defender):
        super().__init__()
        self.model = model
        self.defender = defender

    def forward(self, x):
        out = self.model(x)
        if self.training and torch.is_grad_enabled() and out.requires_grad:
            self.defender.observe_forward(x, out)
            out.register_hook(self.defender.on_gradient)
        return out


class DefendedOptimizer:
    """Wraps the passive optimizer and applies the update-level part of an action."""

    def __init__(self, optimizer, params, defender):
        self.opt = optimizer
        self.params = [p for p in params]
        self.defender = defender
        self.param_groups = optimizer.param_groups
        self.n_steps = 0
        self.n_skipped = 0
        self.keep_old = False   # simulator only: keep pre-update weights for counterfactuals
        self.last_old = None

    def zero_grad(self, set_to_none=True):
        self.opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.opt.state_dict()

    def load_state_dict(self, sd):
        self.opt.load_state_dict(sd)

    @torch.no_grad()
    def step(self, closure=None):
        d = self.defender
        action = d.pending_action
        self.n_steps += 1
        self.last_old = None
        if action is None:  # no gradient went through the defender this round
            return self.opt.step(closure)
        d.pending_action = None
        if action == REJECT:
            self.n_skipped += 1
            return None  # skip the update entirely (Adam state untouched)
        old = [p.detach().clone() for p in self.params]
        out = self.opt.step(closure)
        deltas = [p.detach() - o for p, o in zip(self.params, old)]
        dnorm = torch.sqrt(sum((x.float() ** 2).sum() for x in deltas)).item()
        scale = d.pending_update_scale
        if action == CLIP and d.pending_update_cap is None:   # first round: no history yet
            d.pending_update_cap = d.clip_ratio * dnorm
        if d.pending_update_cap is not None and dnorm * scale > d.pending_update_cap:
            scale = d.pending_update_cap / (dnorm + 1e-12)
        if scale != 1.0:
            for p, o, dl in zip(self.params, old, deltas):
                p.copy_(o + scale * dl)
        # running norm of the would-be (unmodified) update; Clip caps the update at
        # clip_ratio times this running norm, so it is effective even under Adam
        d.ema_delta_norm = dnorm if d.ema_delta_norm is None else 0.9 * d.ema_delta_norm + 0.1 * dnorm
        if self.keep_old:
            self.last_old = old
        return out


def attach_defence(target_model, target_opt, defender):
    """Return (wrapped_model, wrapped_optimizer).  The wrapped model shares its
    parameters with ``target_model``, so evaluation code can keep using either."""
    wm = DefendedPassiveModel(target_model, defender)
    wo = DefendedOptimizer(target_opt, list(target_model.parameters()), defender)
    return wm, wo


# ----------------------------------------------------------------------------------
# policies
# ----------------------------------------------------------------------------------

class FixedPolicy:
    """Non-learned baselines, e.g. FixedPolicy(CLIP) = simple clipping defence."""

    def __init__(self, action):
        self.action = action

    def __call__(self, obs):
        return self.action


class RLPolicy:
    """Deployment wrapper around a trained actor-critic (greedy by default)."""

    def __init__(self, net, device="cpu", greedy=True):
        self.net = net.to(device).eval()
        self.device = device
        self.greedy = greedy

    @torch.no_grad()
    def __call__(self, obs):
        logits, _ = self.net(torch.as_tensor(obs, device=self.device)[None])
        if self.greedy:
            return int(logits.argmax(-1).item())
        return int(torch.distributions.Categorical(logits=logits).sample().item())
