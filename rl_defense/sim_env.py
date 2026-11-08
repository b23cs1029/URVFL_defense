"""Adversarial simulation environment (PDF, section 2).

A *world* is one simulated VFL run between the passive client (whose bottom model is
wrapped by the defender) and either an honest or a malicious active party.  Every
round the world calls the ORIGINAL step functions shipped with the repository
(``defense.py``: ``split_module``, ``urvfl_module``, ``fsha_module``, ``agn_module``),
so the simulated attacks are exactly the published ones.  Only the passive model and
optimizer objects passed into them are the defended wrappers.

Each world also exposes ``metric()``:
  * honest world  -> task loss of the joint VFL model on a fixed probe batch
  * attack world  -> reconstruction MSE of the simulated attacker's decoder on a
                     fixed probe batch of passive data (offline-only signal)
"""
import zlib
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn

from . import common as C
from .defender import attach_defence

import models as M                         # original code
import attack_module as AM                 # original code
from defense import (split_module, urvfl_module, fsha_module,  # original code
                     agn_module)


def _to(batch, device):
    data, labels = batch
    return {k: v.to(device) for k, v in data.items()}, labels.to(device)


class SimData:
    """Victim (passive) data, attacker aux pools and a fixed probe batch."""

    def __init__(self, device, batch_size=128, seed=1234):
        self.device = device
        train_aug = C.cifar_train()
        self.targets = np.array(train_aug.targets)
        victim_idx, self.aux_pool = C.simulation_split(seed=seed)
        self.victim = C.distributed(C.Subset(train_aug, victim_idx), batch_size)
        train_plain = C.cifar_train(C.TRANSFORM_TEST)
        # fixed probe batch (un-augmented), split into the two clients' halves
        x = torch.stack([train_plain[i][0] for i in victim_idx[:128]])
        y = torch.tensor([int(self.targets[i]) for i in victim_idx[:128]], dtype=torch.long)
        self.probe = ({"client_1": x[:, :, :, :16].to(device), "client_2": x[:, :, :, 16:].to(device)},
                      y.to(device))
        self.batch_size = batch_size
        self._aux_cache = {}
        self.train_aug = train_aug

    def aux_loader(self, kind, rng_seed):
        key = (kind, rng_seed)
        if key not in self._aux_cache:
            rng = np.random.RandomState(rng_seed)
            idx = C.aux_indices(self.aux_pool, self.targets, kind, rng)
            self._aux_cache[key] = C.distributed(C.Subset(self.train_aug, idx), self.batch_size)
        return self._aux_cache[key]


class World:
    kind = "honest"
    name = "world"

    def __init__(self, sim, defender, cfg, device):
        self.sim, self.defender, self.cfg, self.device = sim, defender, cfg, device
        self.adv, self.tgt = cfg["adv"], cfg["target"]
        self.age = 0

    def _fresh_passive(self, lr=1e-4):
        self.raw_target = C.client_net().to(self.device)
        opt = torch.optim.Adam(self.raw_target.parameters(), lr=lr)
        self.target, self.target_opt = attach_defence(self.raw_target, opt, self.defender)
        self.target_opt.keep_old = True
        self.defender.reset()
        self.age = 0

    @torch.no_grad()
    def paired_metric(self):
        """(metric after this round's passive update, metric had the passive model NOT
        been updated this round).  Both use the same attacker / server state and the
        same BatchNorm buffers, so their difference isolates the effect of the action."""
        m_new = self.metric()[0]
        old = self.target_opt.last_old
        if old is None:                       # rejected round: no update happened
            return m_new, m_new
        params = list(self.raw_target.parameters())
        cur = [p.detach().clone() for p in params]
        for p, o in zip(params, old):
            p.copy_(o)
        m_cf = self.metric()[0]
        for p, c in zip(params, cur):
            p.copy_(c)
        return m_new, m_cf

    def _batches(self, loader):
        return cycle(list(iter(loader)))

    def step(self):
        raise NotImplementedError

    def metric(self):
        raise NotImplementedError


class HonestWorld(World):
    kind = "honest"

    def __init__(self, sim, defender, cfg, device, lr=1e-4, name="honest"):
        super().__init__(sim, defender, cfg, device)
        self.lr, self.name = lr, name
        self.data_it = self._batches(sim.victim)
        self.reset()

    def reset(self):
        self._fresh_passive(self.lr)
        self.server = M.ServerNet(34, 10).to(self.device)
        self.server_opt = torch.optim.Adam(self.server.parameters(), lr=self.lr, amsgrad=True)
        self.encoder = C.client_net().to(self.device)
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)

    def step(self):
        data, labels = _to(next(self.data_it), self.device)
        for m in (self.server, self.encoder, self.target):
            m.train()
        split_module(self.server, self.encoder, self.target, self.server_opt,
                     self.encoder_opt, self.target_opt, data, labels, self.cfg)
        self.age += 1

    @torch.no_grad()
    def metric(self):
        data, labels = self.sim.probe
        for m in (self.server, self.encoder, self.raw_target):
            m.eval()
        out = self.server(torch.cat([self.encoder(data[self.adv]), self.raw_target(data[self.tgt])], -1))
        loss = nn.functional.cross_entropy(out, labels).item()
        acc = (out.argmax(1) == labels).float().mean().item()
        return loss, acc


class _AttackWorld(World):
    kind = "attack"

    def __init__(self, sim, defender, cfg, device, aux_kinds, aux_seeds):
        super().__init__(sim, defender, cfg, device)
        self.aux_kinds, self.aux_seeds = aux_kinds, aux_seeds
        self.data_it = self._batches(sim.victim)
        self.rng = np.random.RandomState(zlib.crc32(self.name.encode()))
        self.reset()

    def _pick_aux(self):
        kind = self.aux_kinds[self.rng.randint(len(self.aux_kinds))]
        seed = int(self.aux_seeds[self.rng.randint(len(self.aux_seeds))])
        self.aux_kind, self.aux_seed = kind, seed
        self.aux_it = self._batches(self.sim.aux_loader(kind, seed))

    def _next(self):
        data, labels = _to(next(self.data_it), self.device)
        local, local_labels = _to(next(self.aux_it), self.device)
        return data, labels, local, local_labels


def pretrained_urvfl_parts(sim, cfg, kind, seed, device, epochs=10):
    """Attacker's URVFL step-1 models (encoder, decoder, shadow) for one aux
    distribution, trained with the ORIGINAL ``pretrain_model`` and cached."""
    path = C.CACHE_DIR / f"sim_urvfl_{kind}_{seed}_e{epochs}.pt"
    enc, dec, sh = C.client_net(), M.fullDecoder(), C.client_net()
    if path.exists():
        sd = torch.load(path, map_location="cpu")
        enc.load_state_dict(sd["enc"]); dec.load_state_dict(sd["dec"]); sh.load_state_dict(sd["shadow"])
        return enc.to(device), dec.to(device), sh.to(device)
    c = dict(cfg); c["save"] = False; c["pretrain_epochs"] = epochs
    probe = [sim.probe]
    enc, dec, sh = enc.to(device), dec.to(device), sh.to(device)
    AM.pretrain_model(enc, dec, sh, sim.aux_loader(kind, seed), probe,
                      torch.optim.Adam(enc.parameters(), lr=1e-4),
                      torch.optim.Adam(dec.parameters(), lr=1e-4),
                      torch.optim.Adam(sh.parameters(), lr=1e-4), device, c, C.CACHE_DIR, 0)
    torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(), "shadow": sh.state_dict()}, path)
    return enc, dec, sh


class URVFLWorld(_AttackWorld):
    name = "urvfl"

    def __init__(self, sim, defender, cfg, device, aux_kinds=("uniform", "skewed"),
                 aux_seeds=(0, 1), disc_cls=None, name="urvfl"):
        self.name = name
        self.disc_cls = disc_cls or (lambda: M.CLSDiscriminator(10))
        self._parts = {}
        super().__init__(sim, defender, cfg, device, aux_kinds, aux_seeds)

    def reset(self):
        self._fresh_passive(self.cfg["target_lr"])
        self._pick_aux()
        key = (self.aux_kind, self.aux_seed)
        if key not in self._parts:
            self._parts[key] = pretrained_urvfl_parts(self.sim, self.cfg, *key, self.device)
        self.encoder, self.decoder, self.shadow = self._parts[key]
        self.disc = self.disc_cls().to(self.device)
        self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=self.cfg["disc_lr"])

    def step(self):
        data, labels, local, local_labels = self._next()
        self.shadow.eval(); self.target.train(); self.disc.train()
        urvfl_module(self.shadow, self.target, self.disc, data, labels, local, local_labels,
                     self.target_opt, self.disc_opt, True, self.cfg)
        self.age += 1

    @torch.no_grad()
    def metric(self):
        data, _ = self.sim.probe
        for m in (self.encoder, self.decoder, self.raw_target):
            m.eval()
        rec = self.decoder(torch.cat([self.encoder(data[self.adv]), self.raw_target(data[self.tgt])], -1))
        return nn.functional.mse_loss(rec[:, :, :, self.cfg["dim"]:], data[self.tgt]).item(), None


class FSHAWorld(_AttackWorld):
    name = "fsha"

    def __init__(self, sim, defender, cfg, device, aux_kinds=("uniform", "skewed"), aux_seeds=(0, 1),
                 name="fsha"):
        self.name = name
        super().__init__(sim, defender, cfg, device, aux_kinds, aux_seeds)

    def reset(self):
        self._fresh_passive(self.cfg["target_lr"])
        self._pick_aux()
        d = self.device
        self.shadow, self.encoder = C.client_net().to(d), C.client_net().to(d)
        self.decoder, self.disc = M.fullDecoder().to(d), M.Discriminator1().to(d)
        self.shadow_opt = torch.optim.Adam(self.shadow.parameters(), lr=1e-4)
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=1e-4)
        self.decoder_opt = torch.optim.Adam(self.decoder.parameters(), lr=1e-4, amsgrad=True)
        self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=1e-4)

    def step(self):
        data, labels, local, _ = self._next()
        for m in (self.shadow, self.encoder, self.decoder, self.disc, self.target):
            m.train()
        fsha_module(self.shadow, self.encoder, self.target, self.decoder, self.disc, data, local,
                    self.shadow_opt, self.encoder_opt, self.target_opt, self.decoder_opt,
                    self.disc_opt, True, self.cfg)
        self.age += 1

    @torch.no_grad()
    def metric(self):
        data, _ = self.sim.probe
        for m in (self.encoder, self.decoder, self.raw_target):
            m.eval()
        rec = self.decoder(torch.cat([self.encoder(data[self.adv]), self.raw_target(data[self.tgt])], -1))
        return nn.functional.mse_loss(rec[:, :, :, self.cfg["dim"]:], data[self.tgt]).item(), None


class AGNWorld(_AttackWorld):
    name = "agn"

    def __init__(self, sim, defender, cfg, device, aux_kinds=("uniform", "skewed"), aux_seeds=(0, 1),
                 name="agn"):
        self.name = name
        super().__init__(sim, defender, cfg, device, aux_kinds, aux_seeds)

    def reset(self):
        self._fresh_passive(self.cfg["target_lr"])
        self._pick_aux()
        d = self.device
        self.encoder, self.decoder, self.disc = C.client_net().to(d), M.AGNDecoder().to(d), M.Discriminator2().to(d)
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=1e-4)
        self.decoder_opt = torch.optim.Adam(self.decoder.parameters(), lr=1e-4)
        self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=self.cfg["disc_lr"], amsgrad=True)

    def step(self):
        data, labels, local, _ = self._next()
        for m in (self.encoder, self.decoder, self.disc, self.target):
            m.train()
        agn_module(self.encoder, self.target, self.decoder, self.disc, data, local,
                   self.encoder_opt, self.target_opt, self.decoder_opt, self.disc_opt, True, self.cfg)
        self.age += 1

    @torch.no_grad()
    def metric(self):
        data, _ = self.sim.probe
        for m in (self.encoder, self.decoder, self.raw_target):
            m.eval()
        rec = self.decoder(data[self.adv], self.encoder(data[self.adv]), self.raw_target(data[self.tgt]))
        return nn.functional.mse_loss(rec[:, :, :, self.cfg["dim"]:], data[self.tgt]).item(), None
