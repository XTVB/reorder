#!/usr/bin/env python3
"""PE-G layer probe v3 — crash-safe. Forward each dataset ONCE, save all
layer×pooling features to an npz immediately (resumable), then score in a
separate pass (instant, re-runnable). Poolings: mean/max/gem3/attnpool
(attn-pool head deep-copied to fp32 so it can't overflow on intermediates)."""
import os, sys, json, glob, time, copy
import numpy as np, torch
from PIL import Image
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score
import open_clip

BASE = "/Users/abdudh/Downloads/PicsStaging/ClusterBenchmarks"
DS = [("M22", "22-alexis"), ("M5", "5-lily"), ("M17", "17-dusha"), ("M25", "25-rae")]
LAYERS = [30, 34, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
POOLS = ["mean", "max", "gem3", "attnpool"]
BS = 8
OUT = os.path.expanduser("~/.cache/reorder/pe_probe3"); os.makedirs(OUT, exist_ok=True)
dev = torch.device("mps")

def forward_all():
    print("loading PE-Core-bigG-14-448 (meta)...", flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "PE-Core-bigG-14-448", pretrained="meta", device=dev)
    model.eval(); model = model.half()
    trunk = getattr(model.visual, "trunk", None)
    ap = getattr(trunk, "attn_pool", None); fn = getattr(trunk, "norm", None)
    ap_f = copy.deepcopy(ap).float().to(dev) if ap is not None else None
    fn_f = copy.deepcopy(fn).float().to(dev) if fn is not None else None
    print(f"attn_pool(fp32 copy): {ap_f is not None}", flush=True)
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 12 and (best is None or len(mod) > len(best[1])):
            best = (name, mod)
    blocks = best[1]; N = len(blocks)
    layers = [i for i in LAYERS if i < N]
    print(f"blocks '{best[0]}' depth={N}; probing {layers}", flush=True)
    cap = {}
    def mk(i):
        def hook(m, inp, out): cap[i] = (out[0] if isinstance(out, (tuple, list)) else out).detach()
        return hook
    for i in layers:
        blocks[i].register_forward_hook(mk(i))

    def to_BND(t, B):
        t = t.float()
        if t.dim() != 3: return None
        if t.shape[0] == B and t.shape[1] != B: return t
        if t.shape[1] == B: return t.transpose(0, 1)
        return t
    def pools(t, B):
        x = to_BND(t, B); o = {}
        if x is None: return o
        o["mean"] = x.mean(1); o["max"] = x.max(1).values
        o["gem3"] = x.clamp(min=1e-6).pow(3).mean(1).pow(1 / 3)
        if ap_f is not None:
            xn = fn_f(x) if fn_f is not None else x
            a = ap_f(xn)
            o["attnpool"] = (a.mean(1) if a.dim() == 3 else a)
        return {k: v.float().cpu().numpy() for k, v in o.items()}

    for mid, suffix in DS:
        npz = f"{OUT}/{mid}.npz"
        if os.path.exists(npz):
            print(f"  [{mid}] cached, skip forward", flush=True); continue
        d = f"{BASE}/ClusteringBenchmark{suffix}"
        groups = json.load(open(f"{d}/.reorder-groups.json"))
        files = sum((glob.glob(f"{d}/*.{e}") for e in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG")), [])
        byname = {os.path.basename(p): p for p in files}
        fns, labels = [], []
        for gi, g in enumerate(groups):
            for f in g["images"]:
                p = byname.get(f) or byname.get(os.path.basename(f))
                if p: fns.append(p); labels.append(gi)
        labels = np.array(labels); ng = len(groups); t0 = time.time()
        acc = {(i, pn): [] for i in layers for pn in POOLS}; finals = []
        with torch.no_grad():
            for s in range(0, len(fns), BS):
                tens = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in fns[s:s + BS]]).to(dev).half()
                finals.append(model.encode_image(tens).float().cpu().numpy())
                B = tens.shape[0]
                for i in layers:
                    for pn, v in pools(cap[i], B).items(): acc[(i, pn)].append(v)
                if s % (BS * 25) == 0:
                    print(f"  [{mid}] {s+B}/{len(fns)} ({(s+B)/max(time.time()-t0,1e-9):.1f} img/s)", flush=True)
        save = {"labels": labels, "ngroups": np.array([ng]), "final_proj": np.concatenate(finals)}
        for (i, pn), ch in acc.items():
            if ch: save[f"L{i:02d}_{pn}"] = np.concatenate(ch)
        np.savez(npz, **save)
        print(f"=== saved {mid} ({suffix}) n={len(fns)} g={ng} ({time.time()-t0:.0f}s) → {npz}", flush=True)

def score_all():
    def score(X, labels, ng):
        X = np.asarray(X, np.float64)
        bad = int((~np.isfinite(X).all(axis=1)).sum())
        X = np.nan_to_num(X)
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        D = np.nan_to_num(pdist(Xn, "cosine"), nan=1.0, posinf=1.0, neginf=1.0)
        Z = linkage(D, "ward")
        return adjusted_rand_score(labels, fcluster(Z, t=ng, criterion="maxclust")), bad
    rows, badrep = {}, {}
    for mid, _ in DS:
        z = np.load(f"{OUT}/{mid}.npz"); labels = z["labels"]; ng = int(z["ngroups"][0])
        rows[mid] = {}
        for k in z.files:
            if k in ("labels", "ngroups"): continue
            ari, bad = score(z[k], labels, ng); rows[mid][k] = ari
            if bad: badrep[(mid, k)] = bad
    if badrep:
        print("NOTE non-finite rows sanitized:", {f"{m}/{k}": n for (m, k), n in badrep.items()})
    fp = [rows[m]["final_proj"] for m, _ in DS]
    print("\nZERO-SHOT ARI @ oracle-N — final_proj baseline then each pooling/layer")
    print(f"{'final_proj':<14}" + "".join(f"{v:>8.3f}" for v in fp) + f"{np.mean(fp):>8.3f}")
    layers = sorted({int(k[1:3]) for m, _ in DS for k in rows[m] if k.startswith("L")})
    for pn in POOLS:
        print(f"\n-- {pn} --        " + "".join(f"{m:>8}" for m, _ in DS) + f"{'MEAN':>8}")
        for i in layers:
            k = f"L{i:02d}_{pn}"
            if all(k in rows[m] for m, _ in DS):
                v = [rows[m][k] for m, _ in DS]
                star = "  *" if np.mean(v) > np.mean(fp) else ""
                print(f"  L{i:02d}        " + "".join(f"{x:>8.3f}" for x in v) + f"{np.mean(v):>8.3f}{star}")
    with open(f"{OUT}/results.tsv", "w") as f:
        f.write("rep\t" + "\t".join(m for m, _ in DS) + "\tmean\n")
        for k in ["final_proj"] + [f"L{i:02d}_{pn}" for pn in POOLS for i in layers]:
            if all(k in rows[m] for m, _ in DS):
                v = [rows[m][k] for m, _ in DS]
                f.write(k + "\t" + "\t".join(f"{x:.4f}" for x in v) + f"\t{np.mean(v):.4f}\n")
    print(f"\nSaved {OUT}/results.tsv  (features cached in {OUT}/*.npz for reuse)")

if __name__ == "__main__":
    if "--score-only" not in sys.argv:
        forward_all()
    score_all()
