"""Step 5 (PDF): deployment + evaluation of the RL defence on the REAL URVFL pipeline.

For every attack the script runs the ORIGINAL attack functions of this repository
several times from the same starting point (same seed, same data split, same
pre-trained attacker encoder/decoder/shadow model, same initial passive model and
discriminator).  Only the passive client's side differs between runs ("arms"):

    none     : no defence (original URVFL, untouched)
    rl       : trained RL policy between the received gradient and the update
    clip     : simple clipping baseline (always CLIP)
    splitguard (optional, --splitguard): SplitGuard detector, with the scoring bug of
                 defense.splitguard corrected (rl_defense/baselines.py)

Attacks:
    urvfl          URVFL / DAC, aux distribution as in the paper (in training distribution)
    urvfl_small    URVFL with a 1000-sample aux set          (held out from policy training)
    urvfl_half     URVFL with aux covering only 5 classes    (held out from policy training)
    sync           URVFL-sync variant (cls_sync)              (held out from policy training)
    fsha, agn      baseline attacks (fsha_attack, agn_attack)
    honest         honest VFL training -> utility and false-positive rate
    honest_splitguard  honest VFL training under SplitGuard -> SplitGuard false alarms

Usage (from the URVFL folder):
    python cifar10/rl_defense_eval_cifar.py --attacks urvfl,honest --epochs 25
"""
import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from baseline_module import *          # noqa: F401,F403  original code
from attack_module import *            # noqa: F401,F403  original code
from defense import split_module   # original code
from cifar_data_pre import data_pre    # original code
from models import *                   # noqa: F401,F403  original code
from util_cifar import set_seed, get_config, setup_paths  # original code

from rl_defense import common as C
from rl_defense.defender import (PassiveDefender, FixedPolicy, RLPolicy, attach_defence,
                                 ACTION_NAMES, CLIP, N_ACTIONS)
from rl_defense.reference import train_reference, load_reference_encoder
from rl_defense.train_policy import load_policy
from rl_defense.baselines import splitguard_fixed


# --------------------------------------------------------------------------- helpers
class ListLoader:
    """Iterable with the same interface as Distribute_MNIST (used for held-out aux sets)."""

    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def heldout_aux(smallloader, kind, seed, batch_size):
    xs1, xs2, ys = [], [], []
    for d, y in smallloader:
        xs1.append(d["client_1"]); xs2.append(d["client_2"]); ys.append(y)
    x1, x2, y = torch.cat(xs1), torch.cat(xs2), torch.cat(ys)
    rng = np.random.RandomState(seed)
    if kind == "small":
        idx = rng.permutation(len(y))[:1024]
    elif kind == "half":
        cls = rng.choice(10, 5, replace=False)
        idx = np.where(np.isin(y.numpy(), cls))[0]
        idx = rng.permutation(idx)
        print(f"[aux] held-out aux covers classes {sorted(cls.tolist())}, {len(idx)} samples")
    else:
        raise ValueError(kind)
    idx = torch.as_tensor(idx)
    b = []
    for s in range(0, len(idx) - batch_size + 1, batch_size):
        j = idx[s:s + batch_size]
        b.append(({"client_1": x1[j], "client_2": x2[j]}, y[j]))
    return ListLoader(b)


def make_defender(arm, ref, policy_bundle, rl, device):
    if arm == "none":
        return None
    if arm == "rl":
        net, normalizer, _ = policy_bundle
        return PassiveDefender(policy=RLPolicy(net, "cpu"), reference_encoder=ref,
                               downweight=rl["downweight"], clip_ratio=rl["clip_ratio"],
                               normalizer=normalizer)
    if arm == "clip":
        return PassiveDefender(policy=FixedPolicy(CLIP), reference_encoder=ref,
                               downweight=rl["downweight"], clip_ratio=rl["clip_ratio"])
    raise ValueError(arm)


def wrap(target_model, target_opt, defender):
    if defender is None:
        return target_model, target_opt
    return attach_defence(target_model, target_opt, defender)


def action_summary(defender, n_rounds_per_epoch):
    if defender is None:
        return None
    acts = np.array(defender.log_actions, int)
    out = {"overall": {ACTION_NAMES[k]: float((acts == k).mean()) for k in range(N_ACTIONS)},
           "per_epoch": [], "decision_ms_per_round": 1000 * defender.decision_time / max(1, defender.n_decisions)}
    for s in range(0, len(acts), n_rounds_per_epoch):
        e = acts[s:s + n_rounds_per_epoch]
        out["per_epoch"].append([float((e == k).mean()) for k in range(N_ACTIONS)])
    return out


def pretrained_attacker(kind, seed, enc, dec, shadow, aux, testloader, device, cfg, save_model_path,
                        load_saved_seed=None):
    """URVFL step 1 (original pretrain_model), cached per (seed, aux kind)."""
    if load_saved_seed is not None and kind == "paper":
        enc.load_state_dict(torch.load(save_model_path / f"{load_saved_seed}urvfl_enc.pt", map_location=device))
        dec.load_state_dict(torch.load(save_model_path / f"{load_saved_seed}urvfl_dec.pt", map_location=device))
        shadow.load_state_dict(torch.load(save_model_path / f"{load_saved_seed}urvfl_shadow.pt", map_location=device))
        print(f"[attacker] loaded the repository's saved URVFL models of seed {load_saved_seed}")
        return
    path = C.CACHE_DIR / f"eval_urvfl_parts_seed{seed}_{kind}_e{cfg['pretrain_epochs']}.pt"
    if path.exists():
        sd = torch.load(path, map_location=device)
        enc.load_state_dict(sd["enc"]); dec.load_state_dict(sd["dec"]); shadow.load_state_dict(sd["shadow"])
        print(f"[attacker] loaded cached pretrained attacker {path.name}")
        return
    c = dict(cfg); c["save"] = False
    pretrain_model(enc, dec, shadow, aux, testloader,
                   torch.optim.Adam(enc.parameters(), lr=1e-4), torch.optim.Adam(dec.parameters(), lr=1e-4),
                   torch.optim.Adam(shadow.parameters(), lr=1e-4), device, c, save_model_path, seed)
    torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(), "shadow": shadow.state_dict()}, path)


@torch.no_grad()
def save_recon_examples(parts, testloader, cfg, device, path, n=8):
    """Ground-truth passive halves and the attacker's reconstruction for every arm.
    parts: {arm: (attacker_encoder, attacker_decoder, passive_target_model)}"""
    data, _ = next(iter(testloader))
    adv, tgt = cfg["adv"], cfg["target"]
    x_adv, x_tgt = data[adv][:n].to(device), data[tgt][:n].to(device)
    out = {"ground_truth": x_tgt.cpu().numpy(), "adversary_half": x_adv.cpu().numpy()}
    for arm, (enc, dec, tm) in parts.items():
        enc.eval(); dec.eval(); tm.eval()
        rec = dec(torch.cat([enc(x_adv), tm(x_tgt)], -1))
        out[arm] = rec[:, :, :, cfg["dim"]:].cpu().numpy()
    if path.exists():                       # keep arms saved by earlier runs of this seed
        prev = dict(np.load(path))
        prev.update(out)
        out = prev
    np.savez_compressed(path, **out)


# --------------------------------------------------------------------------- attacks
def run_urvfl_family(name, args, cfg, rl, ref, policy_bundle, loaders, device, save_model_path, out_dir):
    trainloader, testloader, smallloader = loaders
    if name in ("urvfl_small", "urvfl_half"):
        aux = heldout_aux(smallloader, name.split("_")[1], args.seed, cfg["train_btz"])
    else:
        aux = smallloader
    aux_kind = {"urvfl": "paper", "sync": "paper", "urvfl_small": "small", "urvfl_half": "half"}[name]

    set_seed(args.seed)
    encoder, decoder, shadow = ClientNet(34, 10).to(device), fullDecoder().to(device), ClientNet(34, 10).to(device)
    target0 = ClientNet(34, 10)
    disc0 = CLSDiscriminator(10) if name != "sync" else CLSDiscriminator2(10)
    if name != "sync":
        pretrained_attacker(aux_kind, args.seed, encoder, decoder, shadow, aux, testloader, device,
                            cfg, save_model_path, args.load_saved_seed)
    enc0, dec0, sh0 = (copy.deepcopy(m.state_dict()) for m in (encoder, decoder, shadow))
    t0_state, d0_state = copy.deepcopy(target0.state_dict()), copy.deepcopy(disc0.state_dict())

    results, finals = {}, {}
    arms = list(args.arms)
    for arm in arms:
        print(f"\n===== attack={name}  passive defence={arm} =====")
        set_seed(args.seed)
        encoder.load_state_dict(enc0); decoder.load_state_dict(dec0); shadow.load_state_dict(sh0)
        target = ClientNet(34, 10); target.load_state_dict(t0_state)
        disc = CLSDiscriminator(10) if name != "sync" else CLSDiscriminator2(10)
        disc.load_state_dict(d0_state)
        target_opt = torch.optim.Adam(target.parameters(), lr=cfg["target_lr"])
        disc_opt = torch.optim.Adam(disc.parameters(), lr=cfg["disc_lr"])
        t_start = time.time()
        if arm == "splitguard":
            # corrected SplitGuard (see rl_defense/baselines.py for the bug in defense.splitguard)
            r = splitguard_fixed("urvfl", target, target_opt, trainloader, testloader, aux, device, cfg,
                                 args.epochs, shadow=shadow, disc=disc, disc_opt=disc_opt,
                                 encoder=encoder, decoder=decoder)
            r.update({"final_mse": r["mse"][-1], "final_psnr": r["psnr"][-1], "final_ssim": r["ssim"][-1],
                      "best_attack_mse": float(min(r["mse"]))})
            results[arm] = r
            finals[arm] = (copy.deepcopy(encoder), copy.deepcopy(decoder), target)
            continue

        defender = make_defender(arm, ref, policy_bundle, rl, device)
        wt, wo = wrap(target, target_opt, defender)
        if name == "sync":
            enc_opt = torch.optim.Adam(encoder.parameters(), lr=cfg["lr"])
            dec_opt = torch.optim.Adam(decoder.parameters(), lr=cfg["lr"])
            sh_opt = torch.optim.Adam(shadow.parameters(), lr=cfg["lr"])
            dis_ls, cos_ls, mse_ls, psnr_ls, ssim_ls = cls_sync(
                wt, encoder, decoder, shadow, disc, trainloader, aux, testloader,
                enc_opt, dec_opt, wo, sh_opt, disc_opt, device, cfg, save_model_path, args.seed)
        else:
            dis_ls, cos_ls, mse_ls, psnr_ls, ssim_ls = cls_pretrain(
                wt, encoder, decoder, shadow, disc, trainloader, aux, testloader,
                wo, disc_opt, device, cfg, save_model_path, args.seed)
        results[arm] = {"emb_mse_to_shadow": dis_ls, "emb_cos_to_shadow": cos_ls, "mse": mse_ls,
                        "psnr": psnr_ls, "ssim": ssim_ls, "final_mse": mse_ls[-1], "final_psnr": psnr_ls[-1],
                        "final_ssim": ssim_ls[-1], "best_attack_mse": float(min(mse_ls)),
                        "actions": action_summary(defender, len(trainloader)),
                        "seconds": time.time() - t_start}
        # (sync also trains the attacker's encoder/decoder, so keep each arm's copy)
        finals[arm] = (copy.deepcopy(encoder), copy.deepcopy(decoder), target)

    save_recon_examples(finals, testloader, cfg, device, out_dir / f"recon_{name}.npz")
    return results


def run_baseline_attack(name, args, cfg, rl, ref, policy_bundle, loaders, device, save_model_path, out_dir):
    trainloader, testloader, smallloader = loaders
    set_seed(args.seed)
    if name == "fsha":
        protos = {"shadow": ClientNet(34, 10), "encoder": ClientNet(34, 10), "decoder": fullDecoder(),
                  "disc": Discriminator1(), "target": ClientNet(34, 10)}
    else:
        protos = {"encoder": ClientNet(34, 10), "decoder": AGNDecoder(), "disc": Discriminator2(),
                  "target": ClientNet(34, 10)}
    init = {k: copy.deepcopy(v.state_dict()) for k, v in protos.items()}
    results, finals = {}, {}
    for arm in [a for a in args.arms if a != "splitguard"]:
        print(f"\n===== attack={name}  passive defence={arm} =====")
        set_seed(args.seed)
        m = {k: copy.deepcopy(v) for k, v in protos.items()}
        for k in m:
            m[k].load_state_dict(init[k])
        target_opt = torch.optim.Adam(m["target"].parameters(), lr=cfg["target_lr"])
        defender = make_defender(arm, ref, policy_bundle, rl, device)
        wt, wo = wrap(m["target"], target_opt, defender)
        c = dict(cfg); c["fsha_save"] = False; c["agn_save"] = False
        c["fsha_epochs"] = c["agn_epochs"] = args.epochs
        t_start = time.time()
        if name == "fsha":
            dis_ls, cos_ls, mse_ls, psnr_ls, ssim_ls, _ = fsha_attack(
                m["shadow"], m["encoder"], wt, m["decoder"], m["disc"], trainloader, testloader, smallloader,
                torch.optim.Adam(m["shadow"].parameters(), lr=cfg["lr"]),
                torch.optim.Adam(m["encoder"].parameters(), lr=cfg["lr"]), wo,
                torch.optim.Adam(m["decoder"].parameters(), lr=cfg["lr"]),
                torch.optim.Adam(m["disc"].parameters(), lr=cfg["disc_lr"]), device, c, save_model_path, args.seed)
            res = {"mse": mse_ls, "psnr": psnr_ls, "ssim": ssim_ls}
        else:
            mse_ls, _, _, _ = agn_attack(
                m["encoder"], wt, m["decoder"], m["disc"], trainloader, testloader, smallloader,
                torch.optim.Adam(m["encoder"].parameters(), lr=1e-4), wo,
                torch.optim.Adam(m["decoder"].parameters(), lr=1e-4),
                torch.optim.Adam(m["disc"].parameters(), lr=cfg["disc_lr"], amsgrad=True),
                device, c, save_model_path, args.seed)
            res = {"mse": mse_ls}
        res.update({"final_mse": res["mse"][-1], "best_attack_mse": float(min(res["mse"])),
                    "actions": action_summary(defender, len(trainloader)), "seconds": time.time() - t_start})
        results[arm] = res
    return results


def run_honest(args, cfg, rl, ref, policy_bundle, loaders, device, out_dir):
    """Honest VFL (original split_module) -> test accuracy and false-positive rate."""
    trainloader, testloader, _ = loaders
    set_seed(args.seed)
    protos = {"server": ServerNet(34, 10), "encoder": ClientNet(34, 10), "target": ClientNet(34, 10)}
    init = {k: copy.deepcopy(v.state_dict()) for k, v in protos.items()}
    results = {}
    for arm in [a for a in args.arms if a != "splitguard"]:
        print(f"\n===== honest VFL  passive defence={arm} =====")
        set_seed(args.seed)
        m = {k: copy.deepcopy(v).to(device) for k, v in protos.items()}
        for k in m:
            m[k].load_state_dict(init[k])
        server_opt = torch.optim.Adam(m["server"].parameters(), lr=1e-4, amsgrad=True)
        enc_opt = torch.optim.Adam(m["encoder"].parameters(), lr=1e-4)
        target_opt = torch.optim.Adam(m["target"].parameters(), lr=cfg["target_lr"])
        defender = make_defender(arm, ref, policy_bundle, rl, device)
        wt, wo = wrap(m["target"], target_opt, defender)
        accs, losses = [], []
        t_start = time.time()
        for ep in range(args.honest_epochs):
            tot = 0.0
            for d, y in trainloader:
                d = {k: v.to(device) for k, v in d.items()}
                for mm in (m["server"], m["encoder"], wt):
                    mm.train()
                _, loss = split_module(m["server"], m["encoder"], wt, server_opt, enc_opt, wo, d, y.to(device), cfg)
                tot += loss
            acc = float(get_score(m["encoder"], m["target"], m["server"], testloader, device, cfg))
            accs.append(acc); losses.append(tot / len(trainloader))
            print(f"[honest/{arm}] epoch {ep + 1}: train loss {losses[-1]:.4f}  test acc {acc:.2f}%")
        acts = action_summary(defender, len(trainloader))
        results[arm] = {"acc": accs, "loss": losses, "final_acc": accs[-1], "actions": acts,
                        "false_positive_rate": None if acts is None else 1.0 - acts["overall"]["accept"],
                        "seconds": time.time() - t_start}
    return results


def run_honest_splitguard(args, cfg, loaders, device):
    """False-positive check for the (corrected) SplitGuard baseline on HONEST training.
    SplitGuard stops training when it raises an alarm."""
    trainloader, testloader, smallloader = loaders
    set_seed(args.seed)
    server, encoder, target = ServerNet(34, 10), ClientNet(34, 10), ClientNet(34, 10)
    r = splitguard_fixed("normal", target, torch.optim.Adam(target.parameters(), lr=cfg["target_lr"]),
                         trainloader, testloader, smallloader, device, cfg, args.honest_epochs,
                         encoder=encoder, server=server,
                         server_opt=torch.optim.Adam(server.parameters(), lr=1e-4, amsgrad=True),
                         enc_opt=torch.optim.Adam(encoder.parameters(), lr=1e-4))
    r.update({"false_alarm": r["detected"], "final_acc": r["acc"][-1]})
    return {"splitguard": r}


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacks", default="urvfl,urvfl_small,urvfl_half,sync,honest")
    ap.add_argument("--arms", default="none,rl,clip")
    ap.add_argument("--splitguard", action="store_true", help="also run SplitGuard on URVFL")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--honest-epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--policy", default="ppo_defender")
    ap.add_argument("--load-saved-seed", type=int, default=None,
                    help="reuse the repository's saved URVFL encoder/decoder/shadow of this seed")
    ap.add_argument("--max-train-batches", type=int, default=None, help="debug: truncate training data")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, rl = C.base_config()
    cfg = dict(cfg)
    cfg["save"] = False                              # never overwrite the repository's saved models
    args.epochs = args.epochs or rl["eval_epochs"]
    args.honest_epochs = args.honest_epochs or rl["honest_eval_epochs"]
    cfg["epochs"] = args.epochs
    cfg["pretrain_epochs"] = rl["eval_pretrain_epochs"]
    args.seed = args.seed if args.seed is not None else random.randint(10, 1000)
    args.arms = args.arms.split(",") + (["splitguard"] if args.splitguard else [])
    print(f"Using device: {device}   seed: {args.seed}   arms: {args.arms}")

    set_seed(args.seed)
    _, save_model_path, _ = setup_paths()
    loaders = data_pre(["client_1", "client_2"], device, cfg["train_btz"], cfg["test_btz"], cfg["train_btz"],
                       num_samples=cfg["aux_size"])
    if args.max_train_batches:
        tl = loaders[0]
        tl.data_pointer = tl.data_pointer[:args.max_train_batches + 1]
        tl.labels = tl.labels[:args.max_train_batches + 1]
        tl.__class__ = type("Trunc", (tl.__class__,), {"__len__": lambda s: len(s.data_pointer) - 1})

    train_reference(rl["reference_epochs"], rl["reference_noise"], str(device))
    ref = load_reference_encoder(device)
    policy_bundle = load_policy(args.policy) if "rl" in args.arms else None

    out_dir = C.RESULTS_DIR / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "results.json"
    summary = json.load(open(summary_path)) if summary_path.exists() else {}
    meta = summary.get("_meta", {"epochs_per_attack": {}})
    meta.update({"seed": args.seed, "policy": args.policy, "device": str(device)})
    summary["_meta"] = meta
    for name in args.attacks.split(","):
        if name in ("urvfl", "urvfl_small", "urvfl_half", "sync"):
            arms_backup = args.arms
            if name != "urvfl":
                args.arms = [a for a in args.arms if a != "splitguard"]
            summary[name] = {**summary.get(name, {}),
                             **run_urvfl_family(name, args, cfg, rl, ref, policy_bundle, loaders, device,
                                                save_model_path, out_dir)}
            args.arms = arms_backup
        elif name in ("fsha", "agn"):
            summary[name] = {**summary.get(name, {}),
                             **run_baseline_attack(name, args, cfg, rl, ref, policy_bundle, loaders, device,
                                                   save_model_path, out_dir)}
        elif name == "honest":
            summary[name] = {**summary.get(name, {}),
                             **run_honest(args, cfg, rl, ref, policy_bundle, loaders, device, out_dir)}
        elif name == "honest_splitguard":
            summary[name] = run_honest_splitguard(args, cfg, loaders, device)
        else:
            raise ValueError(name)
        meta["epochs_per_attack"][name] = args.honest_epochs if name == "honest" else args.epochs
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=1, default=float)
        print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
