"""Corrected SplitGuard baseline (used by the evaluation; the original file is untouched).

``defense.splitguard`` in this repository appends a score of 0 for every regular
(non-fake) batch to the same list it averages for detection:

    else:
        scores.append(0)
    ...
    if len(scores) > 15:
        sg_value = sum(scores[-15:]) / 15
        if sg_value < 0.9: attack = False

With fake batches sent with probability 0.1, the 15-batch window is mostly zeros, so
the detector fires after about 15 batches whatever the active party does, honest or
not.  This module keeps the repository's SplitGuard protocol (fake batches with
probability 0.1 after N=20 batches, ``sg_score`` from util.py with mult=5, exp=2,
threshold 0.9 over the last 15 scores) and only averages real SplitGuard scores,
i.e. the scores of fake batches.
"""
import random
import time

import numpy as np
import torch

from attack_module import evaluate_img, test_dis            # original code
from baseline_module import get_score                       # original code
from defense import urvfl_module, split_module              # original code
from util import sg_score                                   # original code
from itertools import cycle


def splitguard_fixed(kind, target_model, target_opt, trainloader, testloader, smallloader, device, cfg,
                     epochs, shadow=None, disc=None, disc_opt=None, encoder=None, decoder=None,
                     server=None, server_opt=None, enc_opt=None, window=15, threshold=0.9,
                     p_fake=0.1, N=20, mult=5, exp=2):
    """kind: 'urvfl' (malicious active party) or 'normal' (honest split training).

    Returns a dict with: detected, detect_round, sg_scores, mse/psnr/ssim per epoch
    (urvfl) or acc per epoch (normal).  On detection the passive client stops
    training, as in the original implementation."""
    adv, tgt = cfg["adv"], cfg["target"]
    small_it = cycle(smallloader)
    r_1, r_2, fakes, sg_scores = [], [], [], []
    b_fake = int(cfg["train_btz"] * cfg["sg_ratio"])
    index, detected, detect_round = 0, False, None
    out = {"mse": [], "psnr": [], "ssim": [], "acc": []}
    for m in (target_model, shadow, disc, encoder, decoder, server):
        if m is not None:
            m.to(device)
    t0 = time.time()
    for epoch in range(epochs):
        for data, labels in trainloader:
            data = {k: v.to(device) for k, v in data.items()}
            labels = labels.to(device)
            local, local_labels = next(small_it)
            local = {k: v.to(device) for k, v in local.items()}
            local_labels = local_labels.to(device)
            index += 1
            send_fake = index > N and random.random() <= p_fake
            if send_fake:
                rand_labels = (labels + 1) % 2              # same randomisation as the repository
                labels = torch.cat((rand_labels[:b_fake], labels[b_fake:]))
            target_model.train()
            if kind == "urvfl":
                shadow.eval(); disc.train()
                g, _, _ = urvfl_module(shadow, target_model, disc, data, labels, local, local_labels,
                                       target_opt, disc_opt, True, cfg)
            else:
                server.train(); encoder.train()
                g, _ = split_module(server, encoder, target_model, server_opt, enc_opt, target_opt,
                                    data, labels, cfg)
            g = g.flatten().cpu()
            if send_fake:
                fakes.append(g)
                if r_1 and r_2:
                    sg_scores.append(float(sg_score(fakes, r_1, r_2, mult=mult, exp=exp, raw=False)))
            elif index > N:
                (r_1 if random.random() <= 0.5 else r_2).append(g)
            if len(sg_scores) >= window and np.mean(sg_scores[-window:]) < threshold:
                detected, detect_round = True, index
                break
        # per-epoch evaluation (also at the moment training is stopped)
        if kind == "urvfl":
            mse, psnr, ssim, _ = evaluate_img(encoder, decoder, target_model, shadow, testloader, device, cfg)
            out["mse"].append(mse); out["psnr"].append(psnr); out["ssim"].append(ssim)
        else:
            out["acc"].append(float(get_score(encoder, target_model, server, testloader, device, cfg)))
        print(f"[splitguard-fixed/{kind}] epoch {epoch + 1}: {len(sg_scores)} SG scores, "
              f"mean of last {window}: {np.mean(sg_scores[-window:]) if sg_scores else float('nan'):.3f}"
              f"{'  -> DETECTED, training stopped' if detected else ''}")
        if detected:
            break
    out.update({"detected": detected, "detect_round": detect_round, "sg_scores": sg_scores,
                "seconds": time.time() - t0})
    return out
