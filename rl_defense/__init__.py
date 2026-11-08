"""RL-based adaptive passive-client defence against reconstruction attacks in VFL.

Everything in this package is additive: the original URVFL and baseline attack
code is never modified. The defence is attached to the passive client by
wrapping its bottom model and optimizer (see defender.py), so the original
attack loops (cls_pretrain, fsha_module, ...) run unchanged.
"""
