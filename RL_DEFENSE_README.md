# RL-based adaptive passive-client defence against URVFL

This folder adds a **passive-client-side defence** to the URVFL repository, following
the one-page research concept *"RL-Based Adaptive Defence Against Reconstruction
Attacks in VFL"*. The goal is to raise the attacker's reconstruction error, so the
active party cannot recover the passive client's private features, while honest VFL
training keeps working.

**The original attack code is not modified.** Every original file
(`attack_module.py`, `baseline_module.py`, `defense.py`, `cifar10/*.py`,
`cifar10/config_cifar.json`) is unchanged. The defence plugs in from the outside by
wrapping the passive model and its optimizer, and the original attack functions
(`cls_pretrain`, `cls_sync`, `fsha_attack`, `agn_attack`, `urvfl_module`, ...) are
then called exactly as published.

## How it maps to the research concept

| PDF section | Implementation |
|---|---|
| 1. Independent reference encoder | `rl_defense/reference.py`: a separate `ClientNet` trained as a denoising auto-encoder on the passive client's own half-images only. It never receives active-party gradients and uses no labels. |
| 2. Adversarial simulation environment | `rl_defense/sim_env.py`: simulated VFL worlds with an honest active party (`split_module`) or a malicious one (`urvfl_module`, `fsha_module`, `agn_module`, all called from the original `defense.py`). The attacker's auxiliary data distribution is varied between uniform and class-skewed in training. Small and 5-class aux sets are held out for testing. |
| 3. RL defensive agent | `rl_defense/defender.py`: the state is built from incoming-gradient statistics, gradient history, embedding drift, and CKA similarity to the reference encoder. The actions are **Accept / Downweight / Clip / Reject**. |
| 4. Reward | `rl_defense/train_policy.py`: `R = -α·L_task + β·E_recon - γ·L_false_reject - δ·C_overhead`. E_recon comes from the simulated attacker's decoder and is used offline only. The E_recon and L_task terms are measured as the counterfactual effect of the round's action, described below. |
| 5. Training, deployment, evaluation | PPO training in `rl_defense/train_policy.py`. Deployment and evaluation against no defence, simple clipping and SplitGuard, plus held-out attacks, in `cifar10/rl_defense_eval_cifar.py`. |

### Where the defence sits

```
passive client                                   active party (honest or URVFL)
  x_passive -> target_model -> embedding h  ---->  disc / server
                    ^                                   |
                    |   DefendedOptimizer.step()        |  gradient dL/dh
                    |   (skip / scaled / capped update) v
                    +---- PassiveDefender.on_gradient(g)  <-- backward hook
                          state s_t -> policy -> {accept, downweight, clip, reject}
```

* **Accept**: apply the gradient normally.
* **Downweight**: multiply the gradient and the resulting parameter update by 0.3.
* **Clip**: clip each per-sample gradient to 0.5 times the running gradient norm, and cap the parameter-update norm at 0.5 times its running value.
* **Reject**: drop the gradient and skip the optimizer step. The Adam state is left untouched.

The update-level part of Downweight and Clip is needed because the passive model uses
Adam. Adam normalises the gradient scale away, so scaling only the gradient would not
slow the attack.

### State features (observation)

The observation has 19 gradient, embedding and reference statistics plus the previous
action. They are listed in `FEATURE_NAMES` in `rl_defense/defender.py`. In the
simulator the most discriminative ones are:

* **Consistency of the batch-mean gradient direction over time.** Honest task gradients point in a different direction every batch, with a cosine of about 0.03 to their running mean. A reconstruction attacker pushes the embedding consistently towards its shadow distribution, with a cosine of about 0.87 for URVFL.
* **CKA between the main embedding and the reference encoder.** Under URVFL the passive embedding becomes increasingly similar to the locally trained reference encoder, at about 0.83 versus 0.62 for honest training. This happens because URVFL drives it towards an auto-encoder-like, information-preserving representation.

None of the features needs labels, the attacker's models, or any reconstruction, so the
policy can run at deployment time.

### Reward measurement and training attacks

Each simulated round evaluates a fixed probe batch twice: once with the passive update
that the action allowed, and once with the passive model's pre-update weights. Both
evaluations use the same attacker, server and BatchNorm state. The difference is the
effect of that action alone on E_recon (attack worlds) or L_task (honest worlds). The
plain round-to-round change of E_recon was tried first. It mostly reflected things the
action does not control, such as the attacker's discriminator learning, and the policy
could not learn to block the attack from it.

The default training mix uses URVFL worlds and honest worlds only. FSHA and AGN are
implemented in the simulator but excluded by default (`train_attacks` in
`config_rl_defense.json`). Over the first rounds, accepting their gradient *raises* the
reconstruction error against the attacker's current decoder. These attacks only pay
off after the decoder co-adapts over thousands of rounds, so a per-round reward would
teach the policy to accept them. They are instead evaluated as attacks the policy has
never seen. This is a deliberate deviation from section 2 of the concept.

## How to run

All commands run from the `URVFL` folder. The tested environment is conda `urvfl`
with Python 3.10, torch 2.3.0+cu121 and torchmetrics 0.11.4. `setuptools<70` is
needed for torchmetrics.

```bash
# 1. reference encoder (cached in rl_defense/cache/reference_encoder.pt)
python -m rl_defense.reference

# 2. train the RL policy with PPO in the simulator (cached in rl_defense/cache/ppo_defender.pt)
python -m rl_defense.train_policy

# 3. real URVFL attack with / without the defence, held-out attacks and honest utility
python cifar10/rl_defense_eval_cifar.py --attacks urvfl,honest --splitguard --seed 42 --epochs 25
python cifar10/rl_defense_eval_cifar.py --attacks urvfl_small,urvfl_half,sync,fsha,agn --seed 42 --epochs 10
python cifar10/rl_defense_eval_cifar.py --attacks honest_splitguard --arms splitguard --seed 42
#    every run with the same --seed merges into cifar10/rl_defense_result/seed42/results.json

# 4. figures + summary table  -> cifar10/rl_defense_result/seed42/
python -m rl_defense.report --seed 42
```

Hyper-parameters of the defence live in `cifar10/config_rl_defense.json`. The original
`cifar10/config_cifar.json` is read as is. The evaluation forces `save=False`, so the
repository's saved models are never overwritten.

## What the evaluation compares

For each attack, every arm starts from **the same** seed, data split, pretrained
attacker encoder/decoder/shadow model, initial passive model and initial
discriminator. Only the passive client's defence differs between arms:

* `none`: the original URVFL attack with no defence.
* `rl`: the trained RL policy.
* `clip`: always Clip, the simple clipping baseline.
* `splitguard`: the SplitGuard detector. It is optional and runs on URVFL, and on honest training through `--attacks honest_splitguard` to measure false alarms. SplitGuard stops all training when it raises an alarm.

**SplitGuard correction.** `defense.splitguard` appends a score of 0 for every regular
batch to the same list it averages for detection. With fake batches sent with
probability 0.1, that 15-entry window is mostly zeros, so the detector fires after
about 15 batches on any training run, honest or malicious. The evaluation therefore
uses `rl_defense/baselines.py`, which keeps the repository's SplitGuard protocol and
threshold and averages only the real scores from fake batches. The original
`defense.py` is not modified.

Metrics follow the PDF:

* **Attack success** is measured with the original `evaluate_img`: reconstruction MSE, PSNR and SSIM on the CIFAR-10 test set, recorded every epoch. A higher MSE means better privacy.
* **Honest classification accuracy** is measured by honest VFL training with `split_module` and `get_score`.
* **False-positive rate** is the share of honest rounds in which the policy did not choose Accept.
* **Overhead** is the defender's decision time per round.

Generalisation is tested on attack configurations that the policy never saw in
training: URVFL with a small aux set, URVFL with aux data covering only 5 classes, the
URVFL-sync attack with a different discriminator, and FSHA and AGN.

## Assumptions and limitations

* The simulator uses a fixed, labelled subset of the CIFAR-10 training set as public surrogate data for the defence designer. In the real evaluation the original `data_pre` split is used, so individual images can overlap. The policy only learns gradient behaviour, not images.
* The honest active party in the simulator uses the repository's `ServerNet`. A very different honest top model may produce different gradient statistics, and a false-positive check in a new deployment is advisable.
* An adaptive attacker who knows the defence could try to make the batch-mean gradient direction less consistent. That would be a natural next experiment.
* **Known failure: URVFL-sync.** In the seed-42 evaluation the policy rejected the held-out sync attack for its first four epochs, then switched to accepting it, and the attack finished close to its undefended result. The sync attacker keeps training its own shadow model and decoder while the passive model is frozen, so its gradients drift outside what the simulator showed the policy. The direct fix is a sync-style world in `rl_defense/sim_env.py` and in `train_attacks`.
