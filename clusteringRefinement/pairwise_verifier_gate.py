#!/usr/bin/env python3
"""
Pairwise "same-set?" verifier — LOMO gate (17-set, excl M7/M14/M15).

Decision gate for the cross-encoder direction (see oracle_ceiling.py): can a learned
verifier hit the accuracy the soft-integration table needs (~87-90% on gray-zone pairs,
less if precision-biased) on a HELD-OUT shoot?

This is the CHEAP lower bound: a verifier on the existing POOLED embeddings
(peg/color/proj) — no patches, no new extraction. Per fold it's trained on the other
16 shoots' candidate edges (each image's k=40 nearest under the deployed bi-encoder
distance — exactly the pairs a real verifier would judge) and evaluated on the held-out
shoot's candidate edges.

The number that matters: verifier accuracy / same-precision on held-out gray-zone pairs,
AND whether it beats the bi-encoder distance threshold on those same pairs (if it only
matches it, the verifier adds nothing). If the pooled verifier already clears the bar,
a patch cross-encoder is gravy; if it plateaus low, that's the signal to invest in
PE-G patch features (the real cross-encoder) before building the clustering integration.
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
EPOCHS, BATCH, MAX_TRAIN = 6, 8192, 400_000


def ward_ari(D, N, true, grouped):
    Z = linkage(squareform(D, checks=False), method="ward")
    pred = fcluster(Z, t=N, criterion="maxclust")
    return adjusted_rand_score(true[grouped], pred[grouped])


def deployed_D(peg, col, proj):
    Gzs = (PEG_W**2 * (peg @ peg.T) + COLOR_W**2 * (col @ col.T)) / (PEG_W**2 + COLOR_W**2)
    sim = (1 - B) * Gzs + B * (proj @ proj.T)
    D = 1.0 - sim
    np.fill_diagonal(D, 0.0)
    return np.clip(0.5 * (D + D.T), 0.0, None)


def knn_edges(D, true, k):
    """Candidate edges (i<j) among GROUPED images: i's k nearest j. Returns ii,jj,dist,label."""
    n = len(D)
    k = min(k, n - 2)
    nn_idx = np.argpartition(D, kth=k, axis=1)[:, :k + 1]
    seen = set()
    ii, jj = [], []
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
    dist = D[ii, jj]
    label = (true[ii] == true[jj]).astype(np.float32)
    return ii, jj, dist, label


class Verifier(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def pair_feats(PEG, COL, PROJ, ii, jj):
    pe = (PEG[ii] - PEG[jj]).abs(); co = (COL[ii] - COL[jj]).abs(); pr = (PROJ[ii] - PROJ[jj]).abs()
    cpe = (PEG[ii] * PEG[jj]).sum(1, keepdim=True)
    cco = (COL[ii] * COL[jj]).sum(1, keepdim=True)
    cpr = (PROJ[ii] * PROJ[jj]).sum(1, keepdim=True)
    return torch.cat([cpe, cco, cpr, pe, co, pr], dim=1)


def main():
    # Preload every shoot once; build one global embedding bank + per-shoot candidate edges.
    print(f"device={DEV}  building candidate edges (k={K}) for 17 shoots...", flush=True)
    PEGs, COLs, PROJs, off, meta = [], [], [], {}, {}
    edges = {}   # tgt -> (gii, gjj, dist, label) with GLOBAL indices
    cursor = 0
    for tgt in EVAL_SET:
        peg, col, proj, true, N = load_fold(tgt)
        D = deployed_D(peg, col, proj)
        ii, jj, dist, lab = knn_edges(D, true, K)
        off[tgt] = cursor
        meta[tgt] = (len(true), true, N)
        edges[tgt] = (ii + cursor, jj + cursor, dist, lab)
        PEGs.append(peg); COLs.append(col); PROJs.append(proj)
        cursor += len(true)
        del D
    PEG = torch.tensor(np.concatenate(PEGs), device=DEV)
    COL = torch.tensor(np.concatenate(COLs), device=DEV)
    PROJ = torch.tensor(np.concatenate(PROJs), device=DEV)
    din = 3 + PEG.shape[1] + COL.shape[1] + PROJ.shape[1]
    print(f"  bank={PEG.shape[0]} imgs, feat dim={din}", flush=True)

    res = {}
    for tgt in EVAL_SET:
        t0 = time.time()
        # train edges = all shoots except tgt
        tr_i, tr_j, tr_d, tr_lab = [], [], [], []
        for s in EVAL_SET:
            if s == tgt:
                continue
            i, j, d, l = edges[s]
            tr_i.append(i); tr_j.append(j); tr_d.append(d); tr_lab.append(l)
        tr_i, tr_j = np.concatenate(tr_i), np.concatenate(tr_j)
        tr_d, tr_lab = np.concatenate(tr_d), np.concatenate(tr_lab)
        if len(tr_i) > MAX_TRAIN:
            sel = np.random.default_rng(0).choice(len(tr_i), MAX_TRAIN, replace=False)
            tr_i, tr_j, tr_d, tr_lab = tr_i[sel], tr_j[sel], tr_d[sel], tr_lab[sel]
        ti = torch.tensor(tr_i, device=DEV); tj = torch.tensor(tr_j, device=DEV)
        tl = torch.tensor(tr_lab, device=DEV)
        pos_w = torch.tensor([(tl == 0).sum() / max((tl == 1).sum(), 1)], device=DEV)

        model = Verifier(din).to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        n = len(ti)
        for ep in range(EPOCHS):
            model.train()
            perm = torch.randperm(n, device=DEV)
            for b0 in range(0, n, BATCH):
                bi = perm[b0:b0 + BATCH]
                feats = pair_feats(PEG, COL, PROJ, ti[bi], tj[bi])
                opt.zero_grad()
                loss = lossf(model(feats), tl[bi])
                loss.backward(); opt.step()

        # train-set scores → pick operating thresholds on TRAIN (no test leakage)
        model.eval()
        def scores(ii, jj):
            out = []
            with torch.no_grad():
                for b0 in range(0, len(ii), 32768):
                    out.append(torch.sigmoid(model(pair_feats(
                        PEG, COL, PROJ, ii[b0:b0+32768], jj[b0:b0+32768]))).cpu().numpy())
            return np.concatenate(out)
        s_tr = scores(ti, tj); y_tr = tr_lab
        # bi-encoder best threshold on train (predict same if dist < thr)
        bi_thr = best_threshold(tr_d, y_tr, lower_is_same=True)
        # verifier high-precision threshold on train (same-precision >= 0.95)
        hp_thr = precision_threshold(s_tr, y_tr, target_p=0.95)

        # ---- evaluate on held-out shoot ----
        ei, ej, ed, ey = edges[tgt]
        ti_e = torch.tensor(ei, device=DEV); tj_e = torch.tensor(ej, device=DEV)
        s_te = scores(ti_e, tj_e)
        ver_pred = (s_te >= 0.5)
        bi_pred = (ed < bi_thr)
        hp_pred = (s_te >= hp_thr)

        # ---- end-to-end clustering: real verifier soft-blended into the distance ----
        n, true_t, N = meta[tgt]; o = off[tgt]; grouped = true_t >= 0
        pe = PEG[o:o+n].cpu().numpy(); co = COL[o:o+n].cpu().numpy(); pr = PROJ[o:o+n].cpu().numpy()
        D = deployed_D(pe, co, pr)
        base_ari = ward_ari(D, N, true_t, grouped)
        li, lj = ei - o, ej - o
        Dv = D.copy()
        blended = (1 - ALPHA) * D[li, lj] + ALPHA * (1.0 - s_te)   # verifier P(same)→distance
        Dv[li, lj] = blended; Dv[lj, li] = blended
        ver_ari = ward_ari(Dv, N, true_t, grouped)

        res[tgt] = dict(
            n=len(ey), same=float(ey.mean()),
            bi_acc=float((bi_pred == ey.astype(bool)).mean()),
            ver_acc=float((ver_pred == ey.astype(bool)).mean()),
            ver_P=prec(ver_pred, ey), ver_R=rec(ver_pred, ey),
            hp_P=prec(hp_pred, ey), hp_R=rec(hp_pred, ey),
            base_ari=base_ari, ver_ari=ver_ari)
        r = res[tgt]
        print(f"{tgt:<5} n={r['n']:<6} same={r['same']:.2f}  bi_acc={r['bi_acc']:.3f}  "
              f"ver_acc={r['ver_acc']:.3f}  ver(P/R)={r['ver_P']:.2f}/{r['ver_R']:.2f}  "
              f"ARI {base_ari:.3f}→{ver_ari:.3f} ({ver_ari-base_ari:+.3f})  ({time.time()-t0:.0f}s)", flush=True)

    def M(k): return float(np.mean([res[t][k] for t in EVAL_SET]))
    print(f"\n================ MEAN over 17 held-out shoots ================")
    print(f"  bi-encoder threshold accuracy : {M('bi_acc'):.4f}")
    print(f"  verifier accuracy             : {M('ver_acc'):.4f}   (Δ {M('ver_acc')-M('bi_acc'):+.4f})")
    print(f"  verifier same precision/recall: {M('ver_P'):.3f} / {M('ver_R'):.3f}")
    print(f"  same-recall @ same-prec≥0.95  : {M('hp_R'):.3f}  (realized P={M('hp_P'):.3f})")
    print(f"\n  ── ACTUAL clustering (real verifier soft-blended, α={ALPHA}) ──")
    print(f"  baseline ARI : {M('base_ari'):.4f}")
    print(f"  verifier ARI : {M('ver_ari'):.4f}   (Δ {M('ver_ari')-M('base_ari'):+.4f})")
    print(f"\nReads onto the soft-integration table: verifier ≈ {M('ver_acc')*100:.0f}% on gray-zone pairs.")

    out = os.path.join(os.path.dirname(__file__), "pairwise_verifier_gate.tsv")
    with open(out, "w") as f:
        cols = ["n", "same", "bi_acc", "ver_acc", "ver_P", "ver_R", "hp_P", "hp_R", "base_ari", "ver_ari"]
        f.write("dataset\t" + "\t".join(cols) + "\n")
        for t in EVAL_SET:
            f.write(t + "\t" + "\t".join(f"{res[t][c]:.4f}" for c in cols) + "\n")
    print(f"\nTSV: {out}")


# ---- small metric helpers ----
def prec(pred, y):
    pp = pred.sum()
    return float((pred & y.astype(bool)).sum() / pp) if pp else 0.0


def rec(pred, y):
    yy = y.sum()
    return float((pred & y.astype(bool)).sum() / yy) if yy else 0.0


def best_threshold(dist, y, lower_is_same=True):
    cand = np.quantile(dist, np.linspace(0.02, 0.98, 60))
    best, bt = -1, cand[0]
    for t in cand:
        pred = (dist < t) if lower_is_same else (dist > t)
        acc = (pred == y.astype(bool)).mean()
        if acc > best:
            best, bt = acc, t
    return bt


def precision_threshold(score, y, target_p=0.95):
    cand = np.quantile(score, np.linspace(0.5, 0.999, 80))
    for t in sorted(cand, reverse=False):
        pred = score >= t
        if pred.sum() and prec(pred, y) >= target_p:
            return t
    return cand[-1]


if __name__ == "__main__":
    main()
