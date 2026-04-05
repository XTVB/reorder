"""
Test DINOv2-reg patch-level features for group similarity.

Instead of comparing single CLS vectors, compare patch grids:
for each patch in image A, find its best match in image B.
The "patch match score" = mean of top-k matched patch similarities.

Usage: python scripts/test_patches.py /path/to/target/dir
"""

import sys
import json
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms

target_dir = Path(sys.argv[1])
groups_path = target_dir / ".reorder-groups.json"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# Load model
print("Loading DINOv2 ViT-B/14 with registers...")
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg", trust_repo=True)
model = model.to(device)
model.eval()

# DINOv2 preprocessing
transform = transforms.Compose([
    transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load groups
with open(groups_path) as f:
    groups = json.load(f)
group_map = {g["id"]: g for g in groups}

# Test pairs
test_pairs = [
    ("SAME", "1e2cee94-9dc6-4f2e-94f9-9146950d109b", "8eac2757-b071-44a6-a91d-b772fddde592", "PinkBikini x PinkBikini2"),
    ("SAME", "a6d1c65c-cbea-4105-af74-e0b9f9b95e51", "02e70989-426d-4805-b409-6215616f2a34", "WhiteSnow x WinterWhite2"),
    ("SAME", "01a12d5a-0e1f-40b3-8935-30e8d5d8c102", "2f66645e-504b-42c9-8b39-3e4e779105c7", "StrawRobe2 x StrawRobeStrip"),
    ("SAME", "d0cbe1d7-ea6d-4678-95d2-324b1c11684d", "ac74c692-ecb6-42e7-82c2-e8ecc1be7522", "PinkChains x Makima"),
    ("SIM",  "de990bc6-d3a7-4f12-b522-075263401ce3", "2a679639-574d-4c46-b5ca-1aed37de7fdb", "WhiteBody x TransWhite"),
    ("SIM",  "11298375-d7ed-490d-a681-29053f560a85", "3063db5d-737e-406e-9f54-8a3962925e41", "OrangeBikini x RedBikini"),
    ("SIM",  "7b065238-5677-46b9-a9cc-adbf97fbb542", "5944be9e-d18e-4179-9b35-10abc8ed1661", "PalePink x WhitePoolside"),
    ("SIM",  "69e87ffe-8124-42ca-bc48-50a81c263c4e", "d0cbe1d7-ea6d-4678-95d2-324b1c11684d", "DevilPrep x PinkChains"),
    ("NEVR", "01803486-5fe0-4ea2-8b6e-a22fabf13653", "40e5c93e-0490-4767-9c3f-29caaccce340", "ValentG x SnowElves"),
    ("NEVR", "e1853e6f-9458-416b-83f7-02f5bf9b0d29", "547c6d66-243d-43dc-9ec3-7d416f96a916", "NeonLatex x SexyShower"),
    ("NEVR", "d0cbe1d7-ea6d-4678-95d2-324b1c11684d", "dbf6d6a2-a543-4e6d-a4ca-a3022f79351e", "PinkChains x BlackDevil"),
    ("NEVR", "70928dcf-5f8b-4396-bb8a-e5b2c4b9f9f0", "233e9056-030d-4d38-b018-11ef40038979", "Venom x NaturalWaves"),
]

# Extract features for all images we need
needed_ids = set()
for _, a, b, _ in test_pairs:
    needed_ids.add(a)
    needed_ids.add(b)

# Cache: group_id -> list of (cls_token, patch_tokens) per image
# cls_token: [768], patch_tokens: [N_patches, 768]
group_features = {}

print("Extracting features...")
for gid in needed_ids:
    g = group_map[gid]
    features = []
    for img_name in g["images"]:
        img_path = target_dir / img_name
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        inp = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            # Get CLS token
            cls = model(inp)  # [1, 768]

            # Get patch tokens (last layer)
            patch_out = model.get_intermediate_layers(inp, n=1, reshape=False)
            patches = patch_out[0][:, 5:]  # skip CLS + 4 register tokens -> [1, N_patches, 768]

        cls = F.normalize(cls, dim=-1).cpu()
        patches = F.normalize(patches, dim=-1).cpu()
        features.append((cls.squeeze(0), patches.squeeze(0)))

    group_features[gid] = features
    print(f"  {g['name']}: {len(features)} images, {features[0][1].shape[0]} patches each")


def patch_match_score(patches_a, patches_b):
    """
    For each patch in A, find its best cosine match in B.
    Return mean of all best-match similarities.
    """
    # patches_a: [Na, 768], patches_b: [Nb, 768]
    sim = patches_a @ patches_b.T  # [Na, Nb], values in [-1, 1]
    best_a_to_b = sim.max(dim=1).values  # best match in B for each patch in A
    best_b_to_a = sim.max(dim=0).values  # best match in A for each patch in B
    return (best_a_to_b.mean().item() + best_b_to_a.mean().item()) / 2


def image_pair_scores(feat_a, feat_b):
    """Compare two images using CLS and patch features."""
    cls_a, patches_a = feat_a
    cls_b, patches_b = feat_b

    cls_sim = (cls_a @ cls_b).item()
    patch_sim = patch_match_score(patches_a, patches_b)

    return cls_sim, patch_sim


def group_pair_analysis(gid_a, gid_b):
    """Compare all image pairs between two groups."""
    feats_a = group_features[gid_a]
    feats_b = group_features[gid_b]

    cls_sims = []
    patch_sims = []

    for fa in feats_a:
        for fb in feats_b:
            cs, ps = image_pair_scores(fa, fb)
            cls_sims.append(cs)
            patch_sims.append(ps)

    cls_sims.sort(reverse=True)
    patch_sims.sort(reverse=True)
    n = len(cls_sims)

    return {
        "cls_median": cls_sims[n // 2],
        "cls_p75": cls_sims[n * 3 // 4],  # 75th percentile (lower = worse)
        "patch_median": patch_sims[n // 2],
        "patch_p75": patch_sims[n * 3 // 4],
        "n_pairs": n,
    }


# Run analysis
print("\nResults (higher similarity = more similar):")
print(f"{'cat':<5} {'pair':<28} {'cls_med':>8} {'cls_p75':>8} {'patch_med':>10} {'patch_p75':>10}")
print("-" * 75)

for cat, gid_a, gid_b, name in test_pairs:
    r = group_pair_analysis(gid_a, gid_b)
    print(f"{cat:<5} {name:<28} {r['cls_median']:>8.4f} {r['cls_p75']:>8.4f} {r['patch_median']:>10.4f} {r['patch_p75']:>10.4f}")

# Also show what we WANT:
print("\nGoal: SAME > SIM > NEVR for patch scores")
print("If patch_median for SAME pairs is higher than NEVR pairs, patches help!")
