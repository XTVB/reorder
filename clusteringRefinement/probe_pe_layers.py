#!/usr/bin/env python3
"""Zero-shot PE-G intermediate-layer probe (go/no-go for the 'tap an earlier layer'
ceiling lever). One forward pass per image captures ~9 layers (mean-pool + CLS);
each representation is scored deterministically: L2-norm -> cosine -> Ward ->
cut at oracle-N -> ARI. Compares against the official final projection (= deployed
pecore_g). No head, no training, no seed — per-dataset deltas are exact."""
import os, sys, json, glob, time
import numpy as np, torch
from PIL import Image
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score
import open_clip

BASE = "/Users/abdudh/Downloads/PicsStaging/ClusterBenchmarks"
DS = [("M22", "22-alexis"), ("M5", "5-lily"), ("M17", "17-dusha"), ("M25", "25-rae")]
FRACS = [0.25, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
BS = 8
dev = torch.device("mps")

print("loading PE-Core-bigG-14-448 (meta)...", flush=True)
model, _, preprocess = open_clip.create_model_and_transforms(
    "PE-Core-bigG-14-448", pretrained="meta", device=dev)
model.eval(); model = model.half()

# Locate the transformer block list (largest ModuleList under the visual tower).
best = None
for name, mod in model.named_modules():
    if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 12:
        if best is None or len(mod) > len(best[1]):
            best = (name, mod)
bname, blocks = best
N = len(blocks)
LAYERS = sorted(set(min(N - 1, round(f * (N - 1))) for f in FRACS))
print(f"block list: '{bname}'  depth={N}  probing layers {LAYERS}", flush=True)

captured = {}
def mk(i):
    def hook(m, inp, out):
        captured[i] = (out[0] if isinstance(out, (tuple, list)) else out).detach()
    return hook
for i in LAYERS:
    blocks[i].register_forward_hook(mk(i))

def pool(t, B):
    t = t.float()
    if t.dim() != 3:
        return t, t
    seqdim = 1 if (t.shape[0] == B and t.shape[1] != B) else (0 if t.shape[1] == B else 1)
    mean = t.mean(dim=seqdim)
    cls = t.index_select(seqdim, torch.tensor([0], device=t.device)).squeeze(seqdim)
    return mean, cls

def score(X, labels, ngroups):
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    Z = linkage(pdist(Xn, metric="cosine"), method="ward")
    pred = fcluster(Z, t=ngroups, criterion="maxclust")
    return adjusted_rand_score(labels, pred)

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
    labels = np.array(labels); ng = len(groups)
    t0 = time.time()
    finals, pm, pc = [], {i: [] for i in LAYERS}, {i: [] for i in LAYERS}
    with torch.no_grad():
        for s in range(0, len(fns), BS):
            batch = fns[s:s + BS]
            tens = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch]).to(dev).half()
            emb = model.encode_image(tens)
            finals.append(emb.float().cpu().numpy())
            B = tens.shape[0]
            for i in LAYERS:
                m, c = pool(captured[i], B)
                pm[i].append(m.cpu().numpy()); pc[i].append(c.cpu().numpy())
            if s % (BS * 25) == 0:
                print(f"  [{mid}] {s+B}/{len(fns)}  ({(s+B)/max(time.time()-t0,1e-9):.1f} img/s)", flush=True)
    reps = {"final_proj": np.concatenate(finals)}
    for i in LAYERS:
        reps[f"L{i:02d}_mean"] = np.concatenate(pm[i])
        reps[f"L{i:02d}_cls"] = np.concatenate(pc[i])
    rows[mid] = {r: score(X, labels, ng) for r, X in reps.items()}
    rows[mid]["_n"] = len(fns); rows[mid]["_g"] = ng
    print(f"=== {mid} {suffix}: n={len(fns)} groups={ng}  ({time.time()-t0:.0f}s) ===", flush=True)
    print("   " + "  ".join(f"{r}={v:.3f}" for r, v in sorted(rows[mid].items()) if not r.startswith("_")), flush=True)

# Summary table: representations x datasets, + mean.
reps_all = ["final_proj"] + [f"L{i:02d}_mean" for i in LAYERS] + [f"L{i:02d}_cls" for i in LAYERS]
print("\n\nZERO-SHOT ARI @ oracle-N  (PE-G representation alone; cosine+Ward)")
hdr = f"{'representation':<16}" + "".join(f"{m:>9}" for m, _ in DS) + f"{'MEAN':>9}"
print(hdr); print("-" * len(hdr))
for r in reps_all:
    vals = [rows[m][r] for m, _ in DS]
    print(f"{r:<16}" + "".join(f"{v:>9.3f}" for v in vals) + f"{np.mean(vals):>9.3f}")
out = "/tmp/pe_layer_probe.tsv"
with open(out, "w") as f:
    f.write("representation\t" + "\t".join(m for m, _ in DS) + "\tmean\n")
    for r in reps_all:
        vals = [rows[m][r] for m, _ in DS]
        f.write(f"{r}\t" + "\t".join(f"{v:.4f}" for v in vals) + f"\t{np.mean(vals):.4f}\n")
print(f"\nlayer depth={N}, probed={LAYERS}.  Saved {out}")
