"""Step 1 (PDF): independent reference encoder.

The passive client trains a separate encoder on ITS OWN private features only
(the right half of each CIFAR-10 image), with a local denoising auto-encoding
objective.  No gradient from the active party ever touches it and no labels are
used.  Its representation is only a local baseline: the defender compares the
main VFL model's embeddings (and the received gradients) with it via CKA.
"""
import argparse

import torch
import torch.nn as nn
from tqdm import tqdm

from . import common as C


def train_reference(epochs=8, noise=0.1, device="cuda", batch_size=128, seed=0, force=False):
    path = C.CACHE_DIR / "reference_encoder.pt"
    ref = C.ReferenceAE().to(device)
    if path.exists() and not force:
        ref.load_state_dict(torch.load(path, map_location=device))
        return ref.eval()
    C.set_seed(seed)
    loader = torch.utils.data.DataLoader(C.cifar_train(), batch_size=batch_size, shuffle=True,
                                         num_workers=0, drop_last=True)
    opt = torch.optim.Adam(ref.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    for ep in range(epochs):
        ref.train()
        tot = 0.0
        for x, _ in tqdm(loader, desc=f"reference ep{ep}", leave=False):
            x = x[:, :, :, 16:].to(device)        # passive client's own half only
            noisy = x + noise * torch.randn_like(x)
            rec = ref.decoder(ref.encoder(noisy))
            loss = mse(rec, x)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"[reference] epoch {ep + 1}/{epochs}  recon loss {tot / len(loader):.4f}")
    torch.save(ref.state_dict(), path)
    return ref.eval()


def load_reference_encoder(device):
    ref = C.ReferenceAE().to(device)
    ref.load_state_dict(torch.load(C.CACHE_DIR / "reference_encoder.pt", map_location=device))
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref.encoder


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    _, rl = C.base_config()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train_reference(rl["reference_epochs"], rl["reference_noise"], dev, force=a.force)
