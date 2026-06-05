#!/usr/bin/env python3
"""PE-G layer probe v2: dense layers around the L44 peak + better poolings.
Poolings: mean, max, gem(p=3), and the model's own attention-pool head applied to
intermediate tokens (removes the pooling confound vs final_proj). One forward pass;
deterministic cosine+Ward ARI @ oracle-N. Saves peak-layer features for reuse."""
import os, sys, json, glob, time
import numpy as np, torch
from PIL import Image
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score
import open_clip

BASE = "/Users/abdudh/Downloads/PicsStaging/ClusterBenchmarks"
DS = [("M22", "22-alexis"), ("M5", "5-lily"), ("M17", "17-dusha"), ("M25", "25-rae")]
LAYERS = [30, 34, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
SAVE_LAYER = 44
BS = 8
OUT = "/tmp/pe_probe2"; os.makedirs(OUT, exist_ok=True)
dev = torch.device("mps")

print("loading PE-Core-bigG-14-448 (meta)...", flush=True)
model, _, preprocess = open_clip.create_model_and_transforms(
    "PE-Core-bigG-14-448", pretrained="meta", device=dev)
model.eval(); model = model.half()

trunk = getattr(model.visual, "trunk", None)
attn_pool = getattr(trunk, "attn_pool", None)
final_norm = getattr(trunk, "norm", None)
print(f"attn_pool present: {attn_pool is not None}; final_norm present: {final_norm is not None}", flush=True)

best = None
for name, mod in model.named_modules():
    if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 12:
        if best is None or len(mod) > len(best[1]):
            best = (name, mod)
blocks = best[1]; N = len(blocks)
LAYERS = [i for i in LAYERS if i < N]
print(f"blocks '{best[0]}' depth={N}; probing {LAYERS}", flush=True)

captured = {}
def mk(i):
    def hook(m, inp, out):
        captured[i] = (out[0] if isinstance(out, (tuple, list)) else out).detach()
    return hook
for i in LAYERS:
    blocks[i].register_forward_hook(mk(i))

def to_BND(t, B):
    t = t.float()
    if t.dim() != 3:
        return None
    if t.shape[0] == B and t.shape[1] != B:
        return t
    if t.shape[1] == B:
        return t.transpose(0, 1)
    return t  # assume batch-first

def poolings(t, B):
    x = to_BND(t, B)
    out = {}
    if x is None:
        return out
    out["mean"] = x.mean(1)
    out["max"] = x.max(1).values
    out["gem3"] = x.clamp(min=1e-6).pow(3).mean(1).pow(1.0 / 3.0)
    if attn_pool is not None:
        try:
            xn = final_norm(x) if final_norm is not None else x
            ap = attn_pool(xn.half() if next(attn_pool.parameters()).dtype == torch.float16 else xn)
            if ap.dim() == 3:
                ap = ap.mean(1)
            out["attnpool"] = ap.float()
        except Exception as e:
            if not hasattr(poolings, "_warned"):
                print(f"  attn_pool readout failed ({e}); skipping it", flush=True); poolings._warned = True
    return out

def score(X, labels, ng):
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    Z = linkage(pdist(Xn, metric="cosine"), method="ward")
    return adjusted_rand_score(labels, fcluster(Z, t=ng, criterion="maxclust"))

POOLS = ["mean", "max", "gem3", "attnpool"]
rows = {}
for mid, suffix in DS:
    d = f"{BASE}/ClusteringBenchmark{suffix}"
    groups = json.load(open(f"{d}/.reorder-groups.json"))
    files = sum((glob.glob(f"{d}/*.{e}") for e in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG")), [])
    byname = {os.path.basename(p): p for p in files}
    fns, labels = [], []
    for gi, g in enumerate(groups):
        for fn in g["images"]:
            p = byname.get(fn) or byname.get(os.path.basename(fn))
            if p:
                fns.append(p); labels.append(gi)
    labels = np.array(labels); ng = len(groups); t0 = time.time()
    acc = {(i, pn): [] for i in LAYERS for pn in POOLS}
    finals = []
    with torch.no_grad():
        for s in range(0, len(fns), BS):
            tens = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in fns[s:s + BS]]).to(dev).half()
            finals.append(model.encode_image(tens).float().cpu().numpy())
            B = tens.shape[0]
            for i in LAYERS:
                for pn, v in poolings(captured[i], B).items():
                    acc[(i, pn)].append(v.cpu().numpy())
            if s % (BS * 25) == 0:
                print(f"  [{mid}] {s+B}/{len(fns)} ({(s+B)/max(time.time()-t0,1e-9):.1f} img/s)", flush=True)
    rep = {"final_proj": np.concatenate(finals)}
    for (i, pn), chunks in acc.items():
        if chunks:
            rep[f"L{i:02d}_{pn}"] = np.concatenate(chunks)
    rows[mid] = {r: score(X, labels, ng) for r, X in rep.items()}
    for pn in POOLS:
        k = f"L{SAVE_LAYER:02d}_{pn}"
        if k in rep:
            np.save(f"{OUT}/{mid}_L{SAVE_LAYER}_{pn}.npy", rep[k])
    np.save(f"{OUT}/{mid}_labels.npy", labels)
    print(f"=== {mid} {suffix}: n={len(fns)} g={ng} ({time.time()-t0:.0f}s) ===", flush=True)

# Tables: one per pooling, rows=layers, cols=datasets.
print("\n\nZERO-SHOT ARI @ oracle-N — by pooling and layer (final_proj baseline at top)")
fp = [rows[m]["final_proj"] for m, _ in DS]
print(f"{'final_proj':<14}" + "".join(f"{v:>8.3f}" for v in fp) + f"{np.mean(fp):>8.3f}")
for pn in POOLS:
    print(f"\n-- pooling: {pn} --   " + "".join(f"{m:>8}" for m, _ in DS) + f"{'MEAN':>8}")
    for i in LAYERS:
        k = f"L{i:02d}_{pn}"
        if all(k in rows[m] for m, _ in DS):
            vals = [rows[m][k] for m, _ in DS]
            print(f"  L{i:02d}        " + "".join(f"{v:>8.3f}" for v in vals) + f"{np.mean(vals):>8.3f}")
with open(f"{OUT}/results.tsv", "w") as f:
    f.write("representation\t" + "\t".join(m for m, _ in DS) + "\tmean\n")
    allk = ["final_proj"] + [f"L{i:02d}_{pn}" for pn in POOLS for i in LAYERS]
    for k in allk:
        if all(k in rows[m] for m, _ in DS):
            vals = [rows[m][k] for m, _ in DS]
            f.write(k + "\t" + "\t".join(f"{v:.4f}" for v in vals) + f"\t{np.mean(vals):.4f}\n")
print(f"\nSaved {OUT}/results.tsv and L{SAVE_LAYER} features for reuse.")
