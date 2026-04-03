#!/usr/bin/env python3
"""Precompute CLIP text embeddings for the auto-naming vocabulary.

Run once (or when vocabulary changes). Output is a JSON file that
the Bun server reads for TF-IDF auto-naming.

Usage:
    python3 precompute_text_embeddings.py <output_path>
"""

import json
import sys

import numpy as np
import open_clip
import torch


VOCABULARY = [
    # ── Colors ────────────────────────────────────────────────────────────
    "red", "blue", "green", "pink", "purple", "orange", "yellow",
    "white", "black", "gold", "silver", "teal", "turquoise", "brown",
    "pastel colors", "neon colors", "monochrome", "colorful rainbow",

    # ── Lighting / mood ───────────────────────────────────────────────────
    "red lighting", "blue lighting", "pink lighting", "purple lighting",
    "green lighting", "neon lighting", "warm golden lighting",
    "natural daylight", "sunset lighting", "candlelight",
    "studio flash lighting", "ring light", "dark moody low key",
    "bright high key", "backlit silhouette", "colored gel lighting",
    "window light", "overhead lighting", "harsh shadows",

    # ── Settings / locations ──────────────────────────────────────────────
    "kitchen", "bedroom", "bathroom", "living room", "hallway",
    "staircase", "balcony", "rooftop", "garage", "laundry room",
    "swimming pool", "hot tub", "sauna", "shower",
    "tropical beach", "ocean shoreline", "desert", "mountain",
    "forest", "garden", "field of flowers", "park",
    "city street", "alleyway", "parking lot",
    "gym", "yoga studio", "dance studio",
    "office", "library", "classroom",
    "hotel room", "cabin", "tent camping",
    "car interior", "motorcycle", "boat",
    "studio backdrop plain", "studio with props",
    "bar nightclub", "restaurant", "cafe",
    "abandoned building", "warehouse", "industrial",
    "castle", "church", "dungeon",
    "bathtub", "bed", "couch sofa", "chair armchair",
    "floor carpet", "table counter", "window sill",
    "mirror reflection", "doorway", "curtains",

    # ── Hair ──────────────────────────────────────────────────────────────
    "blonde hair", "brunette hair", "black hair", "red hair",
    "pink hair", "blue hair", "green hair", "purple hair",
    "white silver hair", "orange hair", "multicolored hair",
    "long straight hair", "long wavy hair", "long curly hair",
    "short hair", "bob haircut", "pixie cut", "buzz cut",
    "pigtails", "twin tails", "ponytail", "braids",
    "bun updo", "messy hair", "wet hair", "bangs fringe",
    "wig", "hair extensions", "hair accessories",

    # ── Clothing / outfits ────────────────────────────────────────────────
    "bikini", "one piece swimsuit", "swim cover up",
    "bra and panties", "lingerie set", "lace lingerie",
    "thong", "g-string", "boyshorts",
    "corset", "bustier", "garter belt", "stockings thigh highs",
    "bodysuit", "leotard", "catsuit",
    "crop top", "tank top", "t-shirt", "blouse",
    "dress", "mini skirt", "long skirt", "shorts",
    "jeans", "leggings", "yoga pants",
    "jacket", "hoodie", "blazer", "vest",
    "robe bathrobe", "kimono", "towel wrapped",
    "pajamas", "nightgown", "sleepwear",
    "workout clothes", "sports bra",
    "formal dress gown", "cocktail dress",
    "leather outfit", "latex outfit", "PVC vinyl",
    "fishnet", "mesh sheer", "see through",
    "fur coat", "feather boa",
    "harness straps", "chains jewelry", "choker collar",
    "high heels", "boots", "thigh high boots", "sneakers",
    "sandals", "barefoot",
    "gloves", "arm warmers", "leg warmers",
    "hat", "cap", "headband", "crown tiara",
    "sunglasses", "glasses",
    "mask", "blindfold", "veil",

    # ── Costume themes ────────────────────────────────────────────────────
    "angel wings", "devil horns", "fairy wings", "butterfly wings",
    "cat ears", "bunny ears", "fox ears", "wolf ears",
    "cat tail", "bunny tail", "fox tail",
    "witch hat", "wizard robe", "vampire fangs", "zombie",
    "mermaid tail", "elf ears pointy",
    "pirate costume", "cowgirl western", "astronaut space",
    "nurse uniform", "maid outfit", "schoolgirl uniform",
    "police officer", "firefighter", "military uniform",
    "cheerleader", "flight attendant", "secretary",
    "superhero costume", "villain costume",
    "princess costume", "queen costume",
    "ninja", "samurai", "knight armor",
    "christmas santa", "halloween costume", "easter bunny",
    "valentine themed", "birthday celebration",
    "cosplay costume", "anime character", "video game character",
    "fantasy warrior", "sci-fi futuristic", "steampunk",
    "gothic lolita", "punk rock", "hippie bohemian",

    # ── Body / skin ───────────────────────────────────────────────────────
    "tattoos visible", "no tattoos clean skin",
    "body oil shiny", "tan skin", "pale skin", "dark skin",
    "freckles", "body paint", "glitter sparkle",
    "muscular fit", "slim petite", "curvy voluptuous",

    # ── Props / objects ───────────────────────────────────────────────────
    "phone selfie", "camera tripod", "ring light visible",
    "food drink", "wine champagne", "coffee tea",
    "flowers bouquet", "candles", "balloons",
    "christmas tree decorations", "pumpkins halloween",
    "stuffed animal plush", "pillow cushion",
    "book reading", "musical instrument", "microphone",
    "sword weapon", "shield", "bow arrow",
    "umbrella", "whip", "rope",
    "handcuffs", "collar leash",
    "pole dance pole", "yoga mat", "exercise equipment",
    "bubbles", "smoke fog", "water splash",
    "confetti", "sparklers fireworks",

    # ── Composition / framing ─────────────────────────────────────────────
    "close up face", "head and shoulders portrait",
    "upper body", "full body", "legs focused",
    "from above looking down", "from below looking up",
    "from behind back view", "side profile",
    "over the shoulder", "between the legs",
    "wide angle", "telephoto compressed",
    "shallow depth of field bokeh", "everything in focus",
    "dutch angle tilted", "symmetrical centered",

    # ── Photo style ───────────────────────────────────────────────────────
    "professional photoshoot", "amateur selfie",
    "film grain vintage", "high contrast dramatic",
    "soft dreamy ethereal", "sharp crisp detailed",
    "black and white", "sepia toned", "cross processed",
    "HDR vivid", "matte flat", "glossy saturated",
    "polaroid style", "disposable camera aesthetic",

    # ── Scene type ────────────────────────────────────────────────────────
    "solo one person", "duo two people", "group multiple people",
    "with pet animal", "nature landscape background",
    "urban cityscape background", "plain solid background",
    "messy room background", "decorated room",
    "outdoor sunny day", "outdoor cloudy overcast",
    "outdoor night", "indoor well lit", "indoor dimly lit",
]


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <output_path>", file=sys.stderr)
        sys.exit(1)

    output_path = sys.argv[1]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    text_tokens = tokenizer(VOCABULARY).to(device)
    with torch.no_grad():
        text_embs = model.encode_text(text_tokens)
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

    text_np = text_embs.cpu().numpy().astype(float)

    output = {
        "terms": VOCABULARY,
        "embeddings": text_np.tolist(),  # list of lists, each 512 floats
    }

    with open(output_path, "w") as f:
        json.dump(output, f)

    print(f"Saved {len(VOCABULARY)} text embeddings to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
