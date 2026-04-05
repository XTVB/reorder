"""
Test DINOv3 patch matching with average-pooled 7x7 grid vs full 14x14.
Compare results on the test cases to see if spatial reduction preserves the signal.

Usage: python scripts/test_patches_pooled.py /path/to/target/dir
"""

import sys
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from transformers import AutoModel, AutoImageProcessor

target_dir = Path(sys.argv[1])
groups_path = target_dir / ".reorder-groups.json"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

model_path = "/tmp/dinov3-weights/facebook/dinov3-vitb16-pretrain-lvd1689m"
print("Loading DINOv3 ViT-B/16...")
model = AutoModel.from_pretrained(model_path)
processor = AutoImageProcessor.from_pretrained(model_path)
model = model.to(device)
model.eval()

with open(groups_path) as f:
    groups = json.load(f)
group_map = {g["id"]: g for g in groups}

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

needed_ids = set()
for _, a, b, _ in test_pairs:
    needed_ids.add(a)
    needed_ids.add(b)

# Extract features: store both full 14x14 and pooled 7x7 patches
group_features_full = {}   # 196 patches
group_features_pooled = {} # 49 patches

print("Extracting features...")
for gid in needed_ids:
    g = group_map[gid]
    feats_full = []
    feats_pooled = []
    for img_name in g["images"]:
        img_path = target_dir / img_name
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state
            patch_tokens = hidden[:, 5:, :]  # [1, 196, 768]

            # L2-normalize
            patch_tokens = F.normalize(patch_tokens, dim=-1)

            # Full 14x14
            full = patch_tokens.squeeze(0).cpu()  # [196, 768]

            # Pooled 7x7: reshape to [1, 14, 14, 768] -> [1, 768, 14, 14] -> avg_pool2d -> [1, 768, 7, 7]
            grid = patch_tokens.view(1, 14, 14, 768).permute(0, 3, 1, 2)  # [1, 768, 14, 14]
            pooled = F.avg_pool2d(grid, kernel_size=2, stride=2)  # [1, 768, 7, 7]
            pooled = pooled.permute(0, 2, 3, 1).reshape(1, 49, 768)  # [1, 49, 768]
            pooled = F.normalize(pooled, dim=-1)  # re-normalize after averaging
            pooled = pooled.squeeze(0).cpu()  # [49, 768]

        feats_full.append(full)
        feats_pooled.append(pooled)

    group_features_full[gid] = feats_full
    group_features_pooled[gid] = feats_pooled
    print(f"  {g['name']}: {len(feats_full)} images")


def patch_match_score(patches_a, patches_b):
    sim = patches_a @ patches_b.T
    best_a_to_b = sim.max(dim=1).values
    best_b_to_a = sim.max(dim=0).values
    return (best_a_to_b.mean().item() + best_b_to_a.mean().item()) / 2


def group_pair_analysis(feats_dict, gid_a, gid_b):
    feats_a = feats_dict[gid_a]
    feats_b = feats_dict[gid_b]
    scores = []
    for pa in feats_a:
        for pb in feats_b:
            scores.append(patch_match_score(pa, pb))
    scores.sort(reverse=True)
    n = len(scores)
    return scores[n // 2]  # median


print(f"\n{'cat':<5} {'pair':<28} {'14x14':>8} {'7x7':>8} {'delta':>8}")
print("-" * 62)

for cat, gid_a, gid_b, name in test_pairs:
    med_full = group_pair_analysis(group_features_full, gid_a, gid_b)
    med_pooled = group_pair_analysis(group_features_pooled, gid_a, gid_b)
    delta = med_pooled - med_full
    print(f"{cat:<5} {name:<28} {med_full:>8.4f} {med_pooled:>8.4f} {delta:>+8.4f}")

print("\nGoal: SAME min > NEVER max for both columns")
