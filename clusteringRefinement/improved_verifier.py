#!/usr/bin/env python3
"""
Improved pairwise verifier — does BETTER TRAINING (not new features) break 87%?

Upgrades over pairwise_verifier_gate.py, all on the SAME pooled embeddings:
  - richer symmetric pair features: per modality [cos, |a-b|, a*b] (was just [cos, |a-b|])
  - bigger net (1024-256-64) + weight decay
  - 15 epochs with validation-based early stopping (2 training shoots held out as val)
  - 3-seed ensemble (avg probabilities) — reduces variance, mildly decorrelates
Same LOMO protocol, k=40 candidate edges, end-to-end ARI via soft blend (α=0.3).

Compare to the gate baseline: ver_acc 0.872 (bi 0.863, Δ+0.009), end-to-end ARI 0.7354 (−0.069).
If this stays ~flat, it's strong evidence the limit is information, not training.
"""
from __future__ import annotations
import os, time, warnings
import numpy as np
import torch
import torch.nn as nn
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")
from lomo_common import PEG_W, COLOR_W, EVAL_SET, load_fold  # noqa: E402

B, K, ALPHA = 0.60, 40, 0.30
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS, BATCH, MAX_TRAIN, SEEDS, PATIENCE = 15, 8192, 400_000, 3, 3


def deployed_D(peg, col, proj):
    Gzs = (PEG_W**2 * (peg @ peg.T) + COLOR_W**2 * (col @ col.T)) / (PEG_W**2 + COLOR_W**2)
    D = 1.0 - ((1 - B) * Gzs + B * (proj @ proj.T))
    np.fill_diagonal(D, 0.0)
    return np.clip(0.5 * (D + D.T), 0.0, None)


def knn_edges(D, true, k):
    n = len(D); k = min(k, n - 2)
    nn_idx = np.argpartition(D, kth=k, axis=1)[:, :k + 1]
    seen, ii, jj = set(), [], []
    for i in range(n):
        if true[i] < 0:
            continue
        for j in nn_idx[i]:
            if j == i or true[j] < 0:
                continue
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in seen:
                seen.add((a, b)); ii.append(a); jj.append(b)
    ii, jj = np.array(ii), np.array(jj)
    return ii, jj, D[ii, jj], (true[ii] == true[jj]).astype(np.float32)


def ward_ari(D, N, true, grouped):
    Z = linkage(squareform(D, checks=False), method="ward")
    return adjusted_rand_score(true[grouped], fcluster(Z, t=N, criterion="maxclust")[grouped])


class Verifier(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, 1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def feats(PEG, COL, PROJ, ii, jj):
    out = []
    for X in (PEG, COL, PROJ):
        a, b = X[ii], X[jj]
        out += [(a * b).sum(1, keepdim=True), (a - b).abs(), a * b]
    return torch.cat(out, dim=1)


def prec(p, y): return float((p & y.astype(bool)).sum() / p.sum()) if p.sum() else 0.0
def rec(p, y): return float((p & y.astype(bool)).sum() / y.sum()) if y.sum() else 0.0


def main():
    print(f"device={DEV}  building k={K} edges...", flush=True)
    PEGs, COLs, PROJs, off, meta, edges = [], [], [], {}, {}, {}
    cur = 0
    for tgt in EVAL_SET:
        peg, col, proj, true, N = load_fold(tgt)
        D = deployed_D(peg, col, proj)
        ii, jj, dist, lab = knn_edges(D, true, K)
        off[tgt] = cur; meta[tgt] = (len(true), true, N)
        edges[tgt] = (ii + cur, jj + cur, dist, lab)
        PEGs.append(peg); COLs.append(col); PROJs.append(proj); cur += len(true); del D
    PEG = torch.tensor(np.concatenate(PEGs), device=DEV)
    COL = torch.tensor(np.concatenate(COLs), device=DEV)
    PROJ = torch.tensor(np.concatenate(PROJs), device=DEV)
    din = (PEG.shape[1] + COL.shape[1] + PROJ.shape[1]) * 2 + 3
    print(f"  bank={PEG.shape[0]} feat={din} (richer)  seeds={SEEDS} epochs≤{EPOCHS}", flush=True)

    def scores_of(models, ii, jj):
        acc = np.zeros(len(ii))
        for m in models:
            m.eval()
            with torch.no_grad():
                out = []
                for b0 in range(0, len(ii), 32768):
                    out.append(torch.sigmoid(m(feats(PEG, COL, PROJ, ii[b0:b0+32768], jj[b0:b0+32768]))).cpu().numpy())
            acc += np.concatenate(out)
        return acc / len(models)

    res = {}
    for tgt in EVAL_SET:
        t0 = time.time()
        tr_shoots = [s for s in EVAL_SET if s != tgt]
        val_shoots = tr_shoots[:2]; fit_shoots = tr_shoots[2:]
        def gather(shoots):
            I, J, Dd, L = [], [], [], []
            for s in shoots:
                i, j, d, l = edges[s]; I.append(i); J.append(j); Dd.append(d); L.append(l)
            return np.concatenate(I), np.concatenate(J), np.concatenate(Dd), np.concatenate(L)
        fi, fj, fd, fl = gather(fit_shoots)
        vi, vj, _, vl = gather(val_shoots)
        if len(fi) > MAX_TRAIN:
            sel = np.random.default_rng(0).choice(len(fi), MAX_TRAIN, replace=False)
            fi, fj, fd, fl = fi[sel], fj[sel], fd[sel], fl[sel]
        ti = torch.tensor(fi, device=DEV); tj = torch.tensor(fj, device=DEV); tl = torch.tensor(fl, device=DEV)
        vti = torch.tensor(vi, device=DEV); vtj = torch.tensor(vj, device=DEV)
        pos_w = torch.tensor([(tl == 0).sum() / max((tl == 1).sum(), 1)], device=DEV)

        models = []
        for seed in range(SEEDS):
            torch.manual_seed(seed)
            mdl = Verifier(din).to(DEV)
            opt = torch.optim.Adam(mdl.parameters(), lr=1e-3, weight_decay=1e-4)
            lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
            n = len(ti); best_acc, best_state, bad = -1, None, 0
            for ep in range(EPOCHS):
                mdl.train()
                perm = torch.randperm(n, device=DEV)
                for b0 in range(0, n, BATCH):
                    bi = perm[b0:b0 + BATCH]
                    opt.zero_grad()
                    lossf(mdl(feats(PEG, COL, PROJ, ti[bi], tj[bi])), tl[bi]).backward()
                    opt.step()
                va = (scores_of([mdl], vti, vtj) >= 0.5)
                vacc = (va == vl.astype(bool)).mean()
                if vacc > best_acc:
                    best_acc, best_state, bad = vacc, {k: v.detach().clone() for k, v in mdl.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= PATIENCE:
                        break
            mdl.load_state_dict(best_state); models.append(mdl)

        # thresholds on FIT set (ensemble), then eval held-out shoot
        s_fit = scores_of(models, ti, tj)
        bi_thr = sorted(np.quantile(fd, np.linspace(.02, .98, 80)),
                        key=lambda t: -((fd < t) == fl.astype(bool)).mean())[0]
        ei, ej, ed, ey = edges[tgt]
        ti_e = torch.tensor(ei, device=DEV); tj_e = torch.tensor(ej, device=DEV)
        s_te = scores_of(models, ti_e, tj_e)
        ver_pred = s_te >= 0.5; bi_pred = ed < bi_thr

        n, true_t, N = meta[tgt]; o = off[tgt]; grouped = true_t >= 0
        pe = PEG[o:o+n].cpu().numpy(); co = COL[o:o+n].cpu().numpy(); pr = PROJ[o:o+n].cpu().numpy()
        D = deployed_D(pe, co, pr)
        base_ari = ward_ari(D, N, true_t, grouped)
        Dv = D.copy(); li, lj = ei - o, ej - o
        bl = (1 - ALPHA) * D[li, lj] + ALPHA * (1.0 - s_te)
        Dv[li, lj] = bl; Dv[lj, li] = bl
        ver_ari = ward_ari(Dv, N, true_t, grouped)

        res[tgt] = dict(bi_acc=float((bi_pred == ey.astype(bool)).mean()),
                        ver_acc=float((ver_pred == ey.astype(bool)).mean()),
                        P=prec(ver_pred, ey), R=rec(ver_pred, ey),
                        base_ari=base_ari, ver_ari=ver_ari)
        r = res[tgt]
        print(f"{tgt:<5} bi={r['bi_acc']:.3f} ver={r['ver_acc']:.3f} (Δ{r['ver_acc']-r['bi_acc']:+.3f}) "
              f"P/R={r['P']:.2f}/{r['R']:.2f}  ARI {base_ari:.3f}→{ver_ari:.3f} ({ver_ari-base_ari:+.3f})  "
              f"({time.time()-t0:.0f}s)", flush=True)

    def M(k): return float(np.mean([res[t][k] for t in EVAL_SET]))
    print(f"\n================ MEAN over 17 (improved training) ================")
    print(f"  bi-encoder edge acc : {M('bi_acc'):.4f}")
    print(f"  verifier edge acc   : {M('ver_acc'):.4f}   (Δ {M('ver_acc')-M('bi_acc'):+.4f})   [gate was 0.872, Δ+0.009]")
    print(f"  verifier P / R      : {M('P'):.3f} / {M('R'):.3f}")
    print(f"  baseline ARI        : {M('base_ari'):.4f}")
    print(f"  verifier ARI        : {M('ver_ari'):.4f}   (Δ {M('ver_ari')-M('base_ari'):+.4f})   [gate was −0.069]")


if __name__ == "__main__":
    main()
