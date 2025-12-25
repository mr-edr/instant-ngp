#!/usr/bin/env python3
"""
GA v4 — Encoding-only Genetic Algorithm for Instant-NGP (HashGrid).

✔ Optimizes ONLY encoding parameters
✔ Hard-clamped feasible ranges (prevents dead configs)
✔ Uses run_one_frame_psnr.py for fitness
✔ Designed for research / publication stability
"""

import argparse
import json
import os
import random
import subprocess
import sys
import copy
import math
import csv
from pathlib import Path

# -----------------------------
# SAFE ENCODING SEARCH SPACE
# -----------------------------
ENCODING_LIMITS = {
    "n_levels": (8, 16),
    "n_features_per_level": [2, 4],
    "log2_hashmap_size": (15, 19),
    "base_resolution": (16, 64),
}

ENCODING_PATH = ["encoding"]

# -----------------------------
# Utility helpers
# -----------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, int(v)))

def deep_get(d, path):
    cur = d
    for p in path:
        cur = cur.get(p, {})
    return cur

def deep_set(d, path, value):
    cur = d
    for p in path[:-1]:
        cur = cur.setdefault(p, {})
    cur[path[-1]] = value

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# -----------------------------
# Individual generation
# -----------------------------
def random_encoding(base_cfg):
    cfg = copy.deepcopy(base_cfg)
    enc = deep_get(cfg, ENCODING_PATH)

    enc["otype"] = "HashGrid"
    enc["n_levels"] = random.randint(*ENCODING_LIMITS["n_levels"])
    enc["n_features_per_level"] = random.choice(ENCODING_LIMITS["n_features_per_level"])
    enc["log2_hashmap_size"] = random.randint(*ENCODING_LIMITS["log2_hashmap_size"])
    enc["base_resolution"] = random.randint(*ENCODING_LIMITS["base_resolution"])

    deep_set(cfg, ENCODING_PATH, enc)
    return cfg

def sanitize_encoding(cfg):
    enc = deep_get(cfg, ENCODING_PATH)

    enc["otype"] = "HashGrid"
    enc["n_levels"] = clamp(enc.get("n_levels", 12), *ENCODING_LIMITS["n_levels"])
    enc["n_features_per_level"] = (
        enc["n_features_per_level"]
        if enc.get("n_features_per_level") in ENCODING_LIMITS["n_features_per_level"]
        else 4
    )
    enc["log2_hashmap_size"] = clamp(enc.get("log2_hashmap_size", 18), *ENCODING_LIMITS["log2_hashmap_size"])
    enc["base_resolution"] = clamp(enc.get("base_resolution", 32), *ENCODING_LIMITS["base_resolution"])

    deep_set(cfg, ENCODING_PATH, enc)
    return cfg

# -----------------------------
# Genetic operators
# -----------------------------
def mutate(cfg, rate=0.3):
    enc = deep_get(cfg, ENCODING_PATH)

    if random.random() < rate:
        enc["n_levels"] += random.choice([-1, 1])
    if random.random() < rate:
        enc["log2_hashmap_size"] += random.choice([-1, 1])
    if random.random() < rate:
        enc["base_resolution"] *= random.choice([0.5, 2.0])
    if random.random() < rate:
        enc["n_features_per_level"] = random.choice(ENCODING_LIMITS["n_features_per_level"])

    deep_set(cfg, ENCODING_PATH, enc)
    return sanitize_encoding(cfg)

def crossover(a, b):
    c = copy.deepcopy(a)
    ea, eb = deep_get(a, ENCODING_PATH), deep_get(b, ENCODING_PATH)
    ec = {}

    for k in ea:
        ec[k] = ea[k] if random.random() < 0.5 else eb.get(k, ea[k])

    deep_set(c, ENCODING_PATH, ec)
    return sanitize_encoding(c)

# -----------------------------
# Fitness evaluation
# -----------------------------
def evaluate(scene, cfg_path, steps, frame_idx):
    cmd = [
        sys.executable,
        "instant-ngp/scripts/run_one_frame_psnr.py",
        "--scene", scene,
        "--config", cfg_path,
        "--steps", str(steps),
        "--frame_idx", str(frame_idx),
    ]

    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=900
        )
    except subprocess.TimeoutExpired:
        return -1.0, "TIMEOUT"

    out = proc.stdout
    for line in out.splitlines():
        if "=== PSNR" in line:
            return float(line.split()[-1]), out

    return -1.0, out

# -----------------------------
# Main GA loop
# -----------------------------
def run_ga(args):
    with open(args.base) as f:
        base_cfg = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    log_csv = open(os.path.join(args.out_dir, "ga_log.csv"), "w", newline="")
    writer = csv.writer(log_csv)
    writer.writerow(["gen", "ind", "psnr", "n_levels", "features", "hash", "base_res"])

    population = [sanitize_encoding(random_encoding(base_cfg)) for _ in range(args.pop_size)]
    best_psnr = -1.0

    for gen in range(1, args.generations + 1):
        print(f"\n=== Generation {gen}/{args.generations} ===")
        scored = []

        for i, ind in enumerate(population):
            cfg_path = os.path.join(args.out_dir, f"gen{gen:02d}_ind{i:02d}.json")
            save_json(cfg_path, ind)

            psnr, out = evaluate(args.scene, cfg_path, args.eval_steps, args.frame_idx)
            enc = deep_get(ind, ENCODING_PATH)

            print(
                f"Ind {i:02d} | PSNR={psnr:.6f} | "
                f"L={enc['n_levels']} F={enc['n_features_per_level']} "
                f"H={enc['log2_hashmap_size']} R={enc['base_resolution']}"
            )

            writer.writerow([
                gen, i, psnr,
                enc["n_levels"],
                enc["n_features_per_level"],
                enc["log2_hashmap_size"],
                enc["base_resolution"],
            ])
            log_csv.flush()

            scored.append((psnr, ind))

            if psnr > best_psnr:
                best_psnr = psnr
                save_json(os.path.join(args.out_dir, "best_encoding.json"), ind)

        scored.sort(key=lambda x: x[0], reverse=True)
        elites = [copy.deepcopy(scored[i][1]) for i in range(max(1, args.pop_size // 4))]

        next_pop = elites.copy()
        while len(next_pop) < args.pop_size:
            a, b = random.sample(elites, 2)
            child = mutate(crossover(a, b))
            next_pop.append(child)

        population = next_pop

    log_csv.close()
    print("\n=== GA finished ===")
    print("Best PSNR:", best_psnr)

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--pop_size", type=int, default=8)
    p.add_argument("--generations", type=int, default=8)
    p.add_argument("--eval_steps", type=int, default=2000)
    p.add_argument("--frame_idx", type=int, default=0)
    return p.parse_args()

if __name__ == "__main__":
    run_ga(parse_args())
