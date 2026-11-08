"""Steps 3-4 (PDF): train the RL defensive agent with PPO in the simulator.

Reward per round (PDF section 4):

    R = -alpha * L_task + beta * E_recon - gamma * L_false_rejection - delta * C_overhead

* L_task and E_recon are measured as the effect of the current action: on a fixed
  probe batch, the value after this round's passive update minus the value had the
  passive model not been updated (same attacker / server state, same BatchNorm
  buffers).  The raw per-round change of the level is dominated by things the
  action does not control (the attacker's discriminator learning, BatchNorm drift),
  which made the attack-side reward too noisy to learn from.  Values are divided
  by a running RMS per world type.
* E_recon comes from the simulated attacker's decoder and is only available
  offline.  It is NEVER part of the observation, and the trained policy does not
  need the simulator at deployment time.
* L_false_rejection penalises non-accept actions in honest worlds, so rejecting
  every gradient is not optimal.
* C_overhead is a small per-action computation cost.

Run:  python -m rl_defense.train_policy      (from the URVFL folder)
"""
import argparse
import csv
import json
import time

import numpy as np
import torch

from . import common as C
from .defender import (PassiveDefender, N_ACTIONS, OBS_DIM, N_FEATURES, FEATURE_NAMES,
                       ACTION_NAMES)
from .ppo import ActorCritic, RunningMeanStd, gae, ppo_update
from .reference import train_reference, load_reference_encoder
from .sim_env import SimData, HonestWorld, URVFLWorld, FSHAWorld, AGNWorld

FALSE_REJECT_COST = np.array([0.0, 0.7, 0.5, 1.0])   # share of the honest update lost
OVERHEAD_COST = np.array([0.0, 0.5, 1.0, 0.0])       # extra compute (relative)


class PolicyHook:
    """Sampling policy installed in every defender during training."""

    def __init__(self, net):
        self.net = net
        self.random = False
        self.last = None

    @torch.no_grad()
    def __call__(self, obs):
        o = torch.as_tensor(obs)[None]
        logits, v = self.net(o)
        if self.random:
            a = np.random.randint(N_ACTIONS)
            logp = float(np.log(1.0 / N_ACTIONS))
        else:
            dist = torch.distributions.Categorical(logits=logits)
            a = int(dist.sample().item())
            logp = float(dist.log_prob(torch.tensor(a)).item())
        self.last = (obs.copy(), a, logp, float(v.item()))
        return a


def build_worlds(sim, cfg, rl, ref, hook, device):
    def defender():
        return PassiveDefender(policy=hook, reference_encoder=ref, downweight=rl["downweight"],
                               clip_ratio=rl["clip_ratio"], record=False)
    honest = [HonestWorld(sim, defender(), cfg, device, lr=1e-4, name="honest_lr1e-4"),
              HonestWorld(sim, defender(), cfg, device, lr=3e-4, name="honest_lr3e-4")]
    kinds = tuple(rl["train_aux_kinds"])
    attacks = []
    if "urvfl" in rl["train_attacks"]:
        attacks.append((0.5, URVFLWorld(sim, defender(), cfg, device, aux_kinds=kinds, name="urvfl")))
    if "fsha" in rl["train_attacks"]:
        attacks.append((0.25, FSHAWorld(sim, defender(), cfg, device, aux_kinds=kinds, name="fsha")))
    if "agn" in rl["train_attacks"]:
        attacks.append((0.25, AGNWorld(sim, defender(), cfg, device, aux_kinds=kinds, name="agn")))
    w = np.array([a for a, _ in attacks]); w = w / w.sum()
    return honest, [x for _, x in attacks], w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--tag", default="ppo_defender")
    args = ap.parse_args()

    cfg, rl = C.base_config()
    cfg = dict(cfg); cfg["save"] = False
    if args.iterations:
        rl["ppo_iterations"] = args.iterations
    device = "cuda" if torch.cuda.is_available() else "cpu"
    C.set_seed(0)
    rng = np.random.RandomState(0)

    train_reference(rl["reference_epochs"], rl["reference_noise"], device)
    ref = load_reference_encoder(device)
    sim = SimData(device)

    net = ActorCritic(OBS_DIM, N_ACTIONS)
    opt = torch.optim.Adam(net.parameters(), lr=rl["ppo_lr"])
    hook = PolicyHook(net)
    honest, attacks, attack_w = build_worlds(sim, cfg, rl, ref, hook, device)
    all_worlds = honest + attacks
    life = {w.name: rng.randint(rl["world_life_min"], rl["world_life_max"]) for w in all_worlds}
    last_metric = {w.name: w.metric()[0] for w in all_worlds}
    rms = {w.name: RunningMeanStd() for w in all_worlds}   # per world type
    a_, b_, g_, d_ = (rl["reward_alpha_task"], rl["reward_beta_recon"],
                      rl["reward_gamma_false_reject"], rl["reward_delta_overhead"])

    def run_segment(world, n, learn=True):
        """Run n rounds in a world; returns trajectory lists."""
        traj = {"obs": [], "act": [], "logp": [], "val": [], "rew": [], "done": [], "raw": []}
        for t in range(n):
            if world.age >= life[world.name]:
                world.reset()
                life[world.name] = rng.randint(rl["world_life_min"], rl["world_life_max"])
                last_metric[world.name] = world.metric()[0]
                if traj["done"]:
                    traj["done"][-1] = 1.0
            world.step()
            obs, a, logp, v = hook.last
            m, m_cf = world.paired_metric()
            dm = m - m_cf                      # counterfactual effect of this round's action
            last_metric[world.name] = m
            rms[world.name].update([dm])
            scale = float(np.sqrt(rms[world.name].var + rms[world.name].mean ** 2)) + 1e-8  # RMS
            if world.kind == "honest":
                r = -a_ * np.clip(dm / scale, -5, 5) - g_ * FALSE_REJECT_COST[a]
            else:
                r = b_ * np.clip(dm / scale, -5, 5)
            r -= d_ * OVERHEAD_COST[a]
            traj["obs"].append(obs); traj["act"].append(a); traj["logp"].append(logp)
            traj["val"].append(v); traj["rew"].append(r); traj["done"].append(0.0)
            traj["raw"].append(world.defender.last_raw)
        return traj

    # ---------------------------------------------------------- feature normaliser
    print("[ppo] collecting warm-up statistics with a random policy ...")
    hook.random = True
    raws = []
    for w in all_worlds:
        raws += run_segment(w, 96)["raw"]
    raws = np.array(raws)
    normalizer = (raws.mean(0).astype(np.float32), raws.std(0).astype(np.float32) + 1e-6)
    for w in all_worlds:
        w.defender.normalizer = normalizer
    hook.random = False

    log_path = C.RESULTS_DIR / f"{args.tag}_training_log.csv"
    log_f = open(log_path, "w", newline="")
    logw = csv.writer(log_f)
    header = ["iter", "honest_reward", "attack_reward"] + \
             [f"honest_{n}" for n in ACTION_NAMES] + [f"attack_{n}" for n in ACTION_NAMES] + \
             ["pg_loss", "v_loss", "entropy", "seconds"]
    logw.writerow(header)

    t_start = time.time()
    for it in range(rl["ppo_iterations"]):
        batch = {"obs": [], "act": [], "logp": [], "adv": [], "ret": []}
        stats = {"honest": {"r": [], "a": []}, "attack": {"r": [], "a": []}}
        for _ in range(rl["ppo_segments_per_iter"]):
            if rng.rand() < rl["honest_fraction"]:
                world = honest[rng.randint(len(honest))]
            else:
                world = attacks[rng.choice(len(attacks), p=attack_w)]
            tr = run_segment(world, rl["ppo_segment_len"])
            with torch.no_grad():
                _, lv = net(torch.as_tensor(tr["obs"][-1])[None])
            adv, ret = gae(tr["rew"], tr["val"], float(lv.item()), tr["done"],
                           rl["ppo_gamma"], rl["ppo_lambda"])
            batch["obs"] += tr["obs"]; batch["act"] += tr["act"]; batch["logp"] += tr["logp"]
            batch["adv"].append(adv); batch["ret"].append(ret)
            stats[world.kind]["r"] += tr["rew"]; stats[world.kind]["a"] += tr["act"]
        pg, vl, ent = ppo_update(net, opt, batch["obs"], batch["act"], batch["logp"],
                                 np.concatenate(batch["adv"]), np.concatenate(batch["ret"]),
                                 epochs=rl["ppo_epochs"], ent_coef=rl["ppo_entropy"])

        def frac(acts):
            acts = np.array(acts, int)
            return [float((acts == k).mean()) if len(acts) else float("nan") for k in range(N_ACTIONS)]

        hr = float(np.mean(stats["honest"]["r"])) if stats["honest"]["r"] else float("nan")
        ar = float(np.mean(stats["attack"]["r"])) if stats["attack"]["r"] else float("nan")
        row = [it, hr, ar] + frac(stats["honest"]["a"]) + frac(stats["attack"]["a"]) + \
              [pg, vl, ent, time.time() - t_start]
        logw.writerow(row); log_f.flush()
        fh, fa = frac(stats["honest"]["a"]), frac(stats["attack"]["a"])
        print(f"[ppo] it {it:3d}  R_honest {hr:+.3f}  R_attack {ar:+.3f}  "
              f"honest acc/dw/clip/rej {np.round(fh, 2)}  attack {np.round(fa, 2)}  "
              f"ent {ent:.3f}  {time.time() - t_start:.0f}s")

        ckpt = {"state_dict": net.state_dict(), "normalizer_mean": normalizer[0],
                "normalizer_std": normalizer[1], "feature_names": FEATURE_NAMES,
                "obs_dim": OBS_DIM, "n_actions": N_ACTIONS, "rl_config": rl, "iteration": it}
        torch.save(ckpt, C.CACHE_DIR / f"{args.tag}.pt")
    log_f.close()
    print(f"[ppo] done. policy -> {C.CACHE_DIR / (args.tag + '.pt')}, log -> {log_path}")


def load_policy(tag="ppo_defender", device="cpu"):
    ck = torch.load(C.CACHE_DIR / f"{tag}.pt", map_location="cpu", weights_only=False)
    net = ActorCritic(ck["obs_dim"], ck["n_actions"])
    net.load_state_dict(ck["state_dict"])
    return net, (ck["normalizer_mean"], ck["normalizer_std"]), ck["rl_config"]


if __name__ == "__main__":
    main()
