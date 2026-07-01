"""train.py -- torch data wrapping, training, zero-shot evaluation."""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .encoders import encode_ids, enc_coeff_vector, build_vocab, MECHANISMS
from .model import OperatorLearner, count_params


class OpDataset(Dataset):
    def __init__(self, bench, idxs, encoder, vocab, max_len):
        self.b = bench; self.idxs = idxs
        self.encoder = encoder; self.vocab = vocab; self.max_len = max_len

    def __len__(self): return len(self.idxs)

    def __getitem__(self, i):
        s = self.b.samples[self.idxs[i]]
        ic = torch.tensor(s.u0[:, None], dtype=torch.float32)       # (N,1)
        y = torch.tensor(s.traj, dtype=torch.float32)               # (T,N)
        if self.encoder == "coeff_vector":
            sym = torch.tensor(enc_coeff_vector(s.op))
            mask = torch.zeros(1)
        elif self.encoder == "none":
            sym = torch.zeros(self.max_len, dtype=torch.long); mask = torch.zeros(self.max_len)
        else:
            ids, m = encode_ids(s.op, self.encoder, self.vocab[self.encoder], self.max_len)
            sym = torch.tensor(ids); mask = torch.tensor(m)
        return ic, sym, mask, y, float(s.commutator), s.stratum


def make_model(bench, encoder, fusion, max_len, d_model=128, seed=0):
    torch.manual_seed(seed)
    N = bench.N; T = len(bench.t_eval)
    vsize = 1
    if encoder not in ("coeff_vector", "none"):
        from .encoders import ENCODERS
        vsize = max(len(build_vocab([s.op for s in bench.samples])[encoder]), 4)
    return OperatorLearner(N, T, vsize, d_model=d_model, symbol_kind=encoder,
                           fusion=fusion, n_mech=len(MECHANISMS), max_len=max_len)


def train_model(bench, encoder="grammar", fusion="xattn", epochs=40,
                lr=3e-4, batch=32, max_len=32, d_model=128, device="cpu",
                seed=0, verbose=True):
    vocab = build_vocab([s.op for s in bench.samples])
    tr = OpDataset(bench, bench.train_idx, encoder, vocab, max_len)
    dl = DataLoader(tr, batch_size=batch, shuffle=True)
    model = make_model(bench, encoder, fusion, max_len, d_model, seed).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = torch.nn.MSELoss()
    for ep in range(epochs):
        model.train(); tot = 0.0
        for ic, sym, mask, y, comm, strat in dl:
            ic, sym, mask, y = ic.to(device), sym.to(device), mask.to(device), y.to(device)
            opt.zero_grad()
            pred = model(ic, sym, mask)
            loss = lossf(pred, y)
            loss.backward(); opt.step(); tot += loss.item()
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  [{encoder}/{fusion}] ep{ep:3d} train_mse={tot/len(dl):.4e}")
    return model, vocab


@torch.no_grad()
def evaluate(model, bench, idxs, encoder, vocab, max_len, device="cpu",
             ablate_symbol=False):
    """Return list of dicts: per-sample relative L2 + commutator + stratum."""
    ds = OpDataset(bench, idxs, encoder, vocab, max_len)
    dl = DataLoader(ds, batch_size=32)
    rows = []
    model.eval()
    for ic, sym, mask, y, comm, strat in dl:
        ic, sym, mask, y = ic.to(device), sym.to(device), mask.to(device), y.to(device)
        pred = model(ic, sym, mask, ablate_symbol=ablate_symbol)
        num = torch.linalg.norm((pred - y).reshape(len(y), -1), dim=1)
        den = torch.linalg.norm(y.reshape(len(y), -1), dim=1) + 1e-9
        rel = (num / den).cpu().numpy()
        for r, c, st in zip(rel, comm.numpy(), strat):
            rows.append({"rel_l2": float(r), "commutator": float(c), "stratum": st})
    return rows
