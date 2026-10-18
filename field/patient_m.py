"""Patient-M: the VQ-VAE patient encoder of GeSubNet (Chen et al.), as adapted for this study."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


def _block(n_in, n_out):
    return [nn.Linear(n_in, n_out), nn.BatchNorm1d(n_out), nn.ReLU()]


class PatientM(nn.Module):
    """Encoder -> per-dimension vector quantisation (EMA codebook) -> decoder."""

    def __init__(self, n_genes, latent_dim=32, code_dim=32, num_codes=32, commitment=0.25, ema_decay=0.99):
        super().__init__()
        self.latent_dim, self.code_dim, self.num_codes = latent_dim, code_dim, num_codes
        self.commitment, self.ema_decay = commitment, ema_decay
        self.encoder = nn.Sequential(*_block(n_genes, 512), nn.Dropout(0.2), *_block(512, 128), nn.Dropout(0.2),
                                     nn.Linear(128, latent_dim), nn.BatchNorm1d(latent_dim))
        self.proj = nn.Linear(latent_dim, latent_dim * code_dim, bias=False)
        self.codebook = nn.Embedding(num_codes, code_dim)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=1.0)
        self.unproj = nn.Linear(latent_dim * code_dim, latent_dim, bias=False)
        self.register_buffer("ema_count", torch.ones(num_codes))
        self.register_buffer("ema_sum", self.codebook.weight.data.clone())
        self.register_buffer("_init_done", torch.zeros(1, dtype=torch.bool))
        self.decoder = nn.Sequential(*_block(latent_dim, 128), *_block(128, 512), nn.Linear(512, n_genes))

    @torch.no_grad()
    def _init_codebook(self, z):
        """k-means++ initialisation of the codebook from the first batch."""
        chosen = [z[torch.randint(0, z.shape[0], (1,))].squeeze(0)]
        for _ in range(self.num_codes - 1):
            d = torch.stack([(z - c).pow(2).sum(1) for c in chosen]).min(dim=0).values
            chosen.append(z[torch.multinomial(d / (d.sum() + 1e-8), 1)].squeeze(0))
        self.codebook.weight.data.copy_(torch.stack(chosen))
        self.ema_sum.data.copy_(self.codebook.weight.data)
        self._init_done.fill_(True)

    def _codes(self, z):
        d = z.pow(2).sum(1, keepdim=True) + self.codebook.weight.pow(2).sum(1) - 2 * z @ self.codebook.weight.T
        return d.argmin(1)

    def _quantize(self, zc):
        flat = zc.reshape(-1, self.code_dim)
        if not self._init_done.item():
            self._init_codebook(flat.detach())
        idx = self._codes(flat)
        if self.training:  # EMA codebook update with unit-norm codes
            with torch.no_grad():
                one_hot = F.one_hot(idx, self.num_codes).float()
                g = self.ema_decay
                self.ema_count.mul_(g).add_((1 - g) * one_hot.sum(0))
                self.ema_sum.mul_(g).add_((1 - g) * one_hot.T @ flat.detach())
                codes = self.ema_sum / (self.ema_count.unsqueeze(1) + 1e-5)
                self.codebook.weight.data.copy_(codes / (codes.norm(dim=1, keepdim=True) + 1e-8))
        zq = self.codebook(idx).reshape(zc.shape)
        return zc + (zq - zc).detach(), self.commitment * F.mse_loss(zq.detach(), zc)

    def forward(self, x):
        ze = self.encoder(x)
        zq, vq_loss = self._quantize(self.proj(ze).reshape(len(x), self.latent_dim, self.code_dim))
        z = self.unproj(zq.reshape(len(x), -1))
        return z, vq_loss, self.decoder(z), ze

    @torch.no_grad()
    def reset_dead_codes(self, x, min_count=5.0):
        """Re-seed unused or near-duplicate (|cos| > 0.85) codes from encoder outputs."""
        flat = self.proj(self.encoder(x)).reshape(-1, self.code_dim)
        unit = self.codebook.weight / (self.codebook.weight.norm(dim=1, keepdim=True) + 1e-8)
        cos = (unit @ unit.T).abs().fill_diagonal_(0.0)
        reset = ((self.ema_count < min_count) | (cos.max(dim=1).values > 0.85)).nonzero().squeeze(-1)
        if len(reset):
            new = flat[torch.randperm(len(flat), device=flat.device)[:len(reset)]]
            new = new + torch.randn_like(new) * 0.01
            new = new / (new.norm(dim=1, keepdim=True) + 1e-8)
            self.codebook.weight.data[reset] = new
            self.ema_sum.data[reset] = new
            self.ema_count.data[reset] = min_count / 2.0


def effective_rank(z, var=0.95):
    s = np.linalg.svd(z - z.mean(0), compute_uv=False)
    return int(np.searchsorted(np.cumsum(s**2 / (s**2).sum()), var)) + 1


def train(X, cohort, target_rank, seed=42, epochs=200, lr=1e-3, batch_size=16, device="cpu",
          l_orth=0.05, l_div=0.7, l_var=0.10, l_rank=0.05):
    """Unsupervised training on an 80% split with cohort-balanced sampling (Supplementary Methods)."""
    perm = np.random.RandomState(seed).permutation(len(X))
    train_idx = perm[:int(0.8 * len(X))]
    x_train, cohort_train = X[train_idx], np.asarray(cohort)[train_idx]
    names, counts = np.unique(cohort_train, return_counts=True)
    weights = np.zeros(len(train_idx), dtype=np.float32)
    for name, n in zip(names, counts):
        weights[cohort_train == name] = 1.0 / n
    loader = DataLoader(TensorDataset(torch.tensor(x_train)), batch_size=batch_size,
                        sampler=WeightedRandomSampler(torch.tensor(weights), int(2 * counts.max()), replacement=True))

    model = PatientM(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_rank, best_state, min_loss = -1, None, float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total, ze_epoch = 0.0, []
        for (xb,) in loader:
            xb = xb.to(device)
            _, vq_loss, recon, ze = model(xb)
            unit = F.normalize(model.codebook.weight, dim=1)
            orth = (unit @ unit.T - torch.eye(model.num_codes, device=device)).pow(2).sum()
            zc = model.proj(ze).reshape(-1, model.code_dim)       # code-space decorrelation
            zc = zc - zc.mean(0, keepdim=True)
            cov = zc.T @ zc / len(zc)
            diag = torch.diagonal(cov)
            div = (1 - diag).pow(2).mean() + (cov - torch.diag(diag)).pow(2).sum() / (model.code_dim * (model.code_dim - 1))
            ze_d = ze.detach()                                   # monitoring terms (no gradient)
            s = torch.linalg.svdvals(ze_d.cpu())
            s = s / (s.sum() + 1e-8)
            rank_loss = (float(np.log(target_rank)) + (s * (s + 1e-8).log()).sum()).clamp(min=0.0).to(device)
            var_loss = (1.0 - ze_d.var(dim=0).mean()).abs()
            loss = (F.mse_loss(recon, xb) + (vq_loss + l_orth * orth + l_div * div)
                    + l_var * var_loss + l_rank * rank_loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
            ze_epoch.append(ze_d.cpu())
        sched.step()
        avg = total / len(x_train)
        if epoch % 25 == 0 and epoch <= epochs // 2:
            model.reset_dead_codes(torch.tensor(x_train, device=device))
        if epoch in (25, 50, 100, 150, 200):   # keep the best-rank checkpoint that also reconstructs well
            rank = effective_rank(torch.cat(ze_epoch).numpy())
            min_loss = min(min_loss, avg)
            if epoch >= 50 and avg < 0.90 and avg <= 1.10 * min_loss and rank >= target_rank and rank > best_rank:
                best_rank, best_state = rank, {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"epoch {epoch:3d}  loss {avg:.4f}  effective rank {rank}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.eval()


@torch.no_grad()
def encode(model, X, device="cpu"):
    """Quantised patient representation Z_p used for clustering."""
    return model.eval()(torch.tensor(X, dtype=torch.float32, device=device))[0].cpu().numpy()
