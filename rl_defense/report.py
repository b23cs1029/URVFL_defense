"""Figures + summary table from the evaluation results (for the with/without comparison).

python -m rl_defense.report --seed <seed>
"""
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

from . import common as C         # noqa: E402

ARM_COLOR = {"rl": "#2a78d6", "none": "#eb6834", "clip": "#1baf7a", "splitguard": "#eda100"}
ARM_LABEL = {"none": "No defence (original URVFL)", "rl": "RL defence (ours)",
             "clip": "Simple clipping", "splitguard": "SplitGuard"}
ATTACK_LABEL = {"urvfl": "URVFL", "urvfl_small": "URVFL, small aux (held-out)",
                "urvfl_half": "URVFL, 5-class aux (held-out)", "sync": "URVFL-sync (held-out)",
                "fsha": "FSHA", "agn": "AGN"}
ACTIONS = ["accept", "downweight", "clip", "reject"]
ACTION_COLOR = ["#2a78d6", "#1baf7a", "#eda100", "#eb6834"]

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "figure.dpi": 130})


def _img(x):
    return np.clip((np.transpose(x, (1, 2, 0)) + 1) / 2, 0, 1)


def mse_curves(res, name, out):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ends = []
    for arm, r in res.items():
        if "mse" not in r or len(r["mse"]) < 2:
            continue
        ep = np.arange(1, len(r["mse"]) + 1)
        label = ARM_LABEL.get(arm, arm)
        if arm == "splitguard":
            label += " (detected, stopped)" if r.get("detected") else " (not detected)"
        ax.plot(ep, r["mse"], lw=2, color=ARM_COLOR.get(arm, "#777"), label=label,
                ls="--" if arm == "splitguard" else "-")
        ends.append([r["mse"][-1], r["mse"][-1], ep[-1]])
    # end-of-line value labels, nudged apart so they never overlap
    lo, hi = ax.get_ylim()
    gap = 0.06 * (hi - lo)
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][1] = max(ends[i][1], ends[i - 1][1] + gap)
    for val, y, x in ends:
        ax.annotate(f"{val:.4f}", (x, val), xytext=(x + 0.4, y), textcoords="data",
                    fontsize=8, color="#52514e", va="center", annotation_clip=False)
    if "splitguard" in res and len(res["splitguard"]["mse"]) < 2:
        r = res["splitguard"]
        ax.axhline(r["final_mse"], color=ARM_COLOR["splitguard"], lw=1.5, ls="--",
                   label=f"SplitGuard ({'detected' if r['detected'] else 'not detected'})")
    ax.set_xlabel("attack epoch"); ax.set_ylabel("reconstruction MSE (higher = more private)")
    ax.set_title(f"{ATTACK_LABEL.get(name, name)}: attacker reconstruction error", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / f"fig_mse_{name}.png"); plt.close(fig)


def recon_grid(name, out):
    p = out / f"recon_{name}.npz"
    if not p.exists():
        return
    d = np.load(p)
    rows = ["ground_truth"] + [a for a in ("none", "rl", "clip", "splitguard") if a in d.files]
    n = d["ground_truth"].shape[0]
    fig, axes = plt.subplots(len(rows), n, figsize=(n * 0.9, len(rows) * 1.75))
    for i, row in enumerate(rows):
        for j in range(n):
            ax = axes[i, j]
            ax.imshow(_img(d[row][j])); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
        lab = "Private data\n(passive half)" if row == "ground_truth" else ARM_LABEL[row].replace(" (", "\n(")
        axes[i, 0].set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=8)
    fig.suptitle(f"{ATTACK_LABEL.get(name, name)}: what the attacker reconstructs", fontsize=10, x=0.02, ha="left")
    fig.tight_layout(); fig.savefig(out / f"fig_recon_{name}.png", bbox_inches="tight"); plt.close(fig)


def action_plot(res, name, out):
    r = res.get("rl", {})
    acts = r.get("actions")
    if not acts or not acts["per_epoch"]:
        return
    pe = np.array(acts["per_epoch"])
    fig, ax = plt.subplots(figsize=(6, 2.8))
    bottom = np.zeros(len(pe))
    x = np.arange(1, len(pe) + 1)
    for k in range(4):
        ax.bar(x, pe[:, k], bottom=bottom, color=ACTION_COLOR[k], label=ACTIONS[k], width=0.8,
               edgecolor="white", linewidth=1)
        bottom += pe[:, k]
    ax.set_ylim(0, 1); ax.set_xlabel("epoch"); ax.set_ylabel("share of rounds")
    ax.set_title(f"RL defence actions per epoch: {ATTACK_LABEL.get(name, name)}", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout(); fig.savefig(out / f"fig_actions_{name}.png"); plt.close(fig)


def honest_plot(res, out):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for arm, r in res.items():
        ep = np.arange(1, len(r["acc"]) + 1)
        ax.plot(ep, r["acc"], lw=2, color=ARM_COLOR.get(arm, "#777"),
                label="No defence" if arm == "none" else ARM_LABEL.get(arm, arm))
    ax.set_xlabel("epoch"); ax.set_ylabel("test accuracy (%)")
    ax.set_title("Honest VFL: utility with and without the defence", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_honest_accuracy.png"); plt.close(fig)


def final_bars(summary, out):
    names = [n for n in ATTACK_LABEL if n in summary]
    arms = [a for a in ("none", "rl", "clip") if any(a in summary[n] for n in names)]
    if not names:
        return
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(names)), 3.6))
    w = 0.8 / len(arms)
    for i, arm in enumerate(arms):
        vals = [summary[n].get(arm, {}).get("final_mse", np.nan) for n in names]
        xs = np.arange(len(names)) + (i - (len(arms) - 1) / 2) * w
        ax.bar(xs, vals, width=w * 0.92, color=ARM_COLOR[arm], label=ARM_LABEL[arm])
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([ATTACK_LABEL[n].replace(" (", "\n(") for n in names], fontsize=8)
    ax.set_ylabel("final reconstruction MSE")
    ax.set_title("Final attacker reconstruction error (higher = better privacy)", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_final_mse.png"); plt.close(fig)


def ppo_plot(tag="ppo_defender"):
    p = C.RESULTS_DIR / f"{tag}_training_log.csv"
    if not p.exists():
        return
    rows = list(csv.DictReader(open(p)))
    it = [int(r["iter"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for k, (kind, col) in enumerate((("honest", "#2a78d6"), ("attack", "#eb6834"))):
        axes[0].plot(it, [float(r[f"{kind}_reward"]) for r in rows], color=col, lw=2,
                     label=f"{kind} worlds")
    axes[0].set_title("Mean reward per round", fontsize=11, loc="left"); axes[0].set_xlabel("PPO iteration")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(it, [float(r["honest_accept"]) for r in rows], color="#2a78d6", lw=2,
                 label="accept rate, honest worlds")
    axes[1].plot(it, [1 - float(r["attack_accept"]) for r in rows], color="#eb6834", lw=2,
                 label="non-accept rate, attack worlds")
    axes[1].set_ylim(0, 1.02); axes[1].set_xlabel("PPO iteration")
    axes[1].set_title("Policy behaviour during training", fontsize=11, loc="left")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(C.RESULTS_DIR / f"fig_{tag}_training.png"); plt.close(fig)


def summary_table(summary, out):
    eps = summary.get("_meta", {}).get("epochs_per_attack", {})
    lines = ["| Attack | Defence | Final recon MSE | Final PSNR | Final SSIM | Lowest MSE reached | Accept / downweight / clip / reject | ms per round |",
             "|---|---|---|---|---|---|---|---|"]
    for n in ATTACK_LABEL:
        if n not in summary:
            continue
        for arm, r in summary[n].items():
            a = r.get("actions")
            mix = " / ".join(f"{a['overall'][k]:.2f}" for k in ACTIONS) if a else "n/a"
            ms = f"{a['decision_ms_per_round']:.2f}" if a else "n/a"
            ps = f"{r['final_psnr']:.2f}" if r.get("final_psnr") is not None else "n/a"
            ss = f"{r['final_ssim']:.3f}" if r.get("final_ssim") is not None else "n/a"
            best = f"{r['best_attack_mse']:.4f}" if "best_attack_mse" in r else "n/a"
            extra = ""
            if arm == "splitguard":
                extra = (f" (detected at round {r['detect_round']}, training stopped)" if r["detected"]
                         else " (not detected)")
            lines.append(f"| {ATTACK_LABEL[n]}, {eps.get(n, '?')} ep | {ARM_LABEL.get(arm, arm)}{extra} | {r['final_mse']:.4f} | {ps} | {ss} | {best} | {mix} | {ms} |")
    if "honest" in summary:
        lines += ["", "| Honest VFL | Final test accuracy (%) | Mean accuracy, last 3 epochs (%) | False-positive rate (non-accept share) |",
                  "|---|---|---|---|"]
        for arm, r in summary["honest"].items():
            fpr = "n/a" if r["false_positive_rate"] is None else f"{r['false_positive_rate']:.3f}"
            name = "No defence" if arm == "none" else ARM_LABEL.get(arm, arm)
            lines.append(f"| {name} | {r['final_acc']:.2f} | {np.mean(r['acc'][-3:]):.2f} | {fpr} |")
    if "honest_splitguard" in summary:
        r = summary["honest_splitguard"]["splitguard"]
        verdict = "false alarm, training aborted" if r["false_alarm"] else "no false alarm"
        lines.append(f"| SplitGuard ({verdict}) | {r['final_acc']:.2f} | n/a | {1.0 if r['false_alarm'] else 0.0:.3f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--policy", default="ppo_defender")
    a = ap.parse_args()
    out = C.RESULTS_DIR / f"seed{a.seed}"
    summary = json.load(open(out / "results.json"))
    for n in ATTACK_LABEL:
        if n in summary:
            mse_curves(summary[n], n, out); recon_grid(n, out); action_plot(summary[n], n, out)
    if "honest" in summary:
        honest_plot(summary["honest"], out)
    final_bars(summary, out)
    ppo_plot(a.policy)
    summary_table(summary, out)
    print(f"[report] figures in {out}")


if __name__ == "__main__":
    main()
