#!/usr/bin/env python3
"""
GA v3.1 — Optimizer-only GA for Instant-NGP (baseline, non-stagnating).

Search space:
- learning_rate
- l2_reg

Evaluation:
- Train N steps
- Render ONE fixed frame
- Compute PSNR via run_one_frame_psnr.py

This script intentionally avoids pyngp internals.
"""

import argparse
import os
import sys
import json
import random
import subprocess
import math
import csv
from copy import deepcopy

# -------------------------
# Search ranges (SAFE)
# -------------------------
LR_RANGE = (1e-4, 5e-2)      # log-uniform
L2_RANGE = (0.0, 1e-4)       # uniform

OPTIMIZER_PATH = ["optimizer", "nested", "nested"]  # Adam block

# -------------------------
# JSON helpers
# -------------------------
def deep_get(d, path):
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return {}
        cur = cur.get(p, {})
    return cur

def deep_set(d, path, value):
    cur = d
    for p in path[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[path[-1]] = value

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def log_uniform(lo, hi):
    return math.exp(random.uniform(math.log(lo), math.log(hi)))

# -------------------------
# Genome
# -------------------------
def sample_individual(base_cfg):
    cfg = deepcopy(base_cfg)
    opt = deep_get(cfg, OPTIMIZER_PATH)
    opt = dict(opt)

    opt["learning_rate"] = float(log_uniform(*LR_RANGE))
    opt["l2_reg"] = float(random.uniform(*L2_RANGE))

    deep_set(cfg, OPTIMIZER_PATH, opt)
    return cfg

def mutate(ind, rate=0.3):
    cfg = deepcopy(ind)
    opt = deep_get(cfg, OPTIMIZER_PATH)
    opt = dict(opt)

    if random.random() < 0.8:
        factor = math.exp(random.uniform(-rate, rate))
        opt["learning_rate"] = float(
            max(LR_RANGE[0], min(LR_RANGE[1], opt["learning_rate"] * factor))
        )

    if random.random() < 0.5:
        opt["l2_reg"] = float(
            max(L2_RANGE[0], min(L2_RANGE[1], opt["l2_reg"] + random.uniform(-1e-5, 1e-5)))
        )

    deep_set(cfg, OPTIMIZER_PATH, opt)
    return cfg

def crossover(a, b):
    cfg = deepcopy(a)
    opt_a = deep_get(a, OPTIMIZER_PATH)
    opt_b = deep_get(b, OPTIMIZER_PATH)

    opt = {}
    opt["learning_rate"] = random.choice([opt_a["learning_rate"], opt_b["learning_rate"]])
    opt["l2_reg"] = random.choice([opt_a["l2_reg"], opt_b["l2_reg"]])

    deep_set(cfg, OPTIMIZER_PATH, opt)
    return cfg

# -------------------------
# Fitness evaluation
# -------------------------
def evaluate(scene, cfg_path, steps, frame_idx, width, height, spp, timeout):
    cmd = [
        sys.executable,
        os.path.join("scripts", "run_one_frame_psnr.py"),
        "--scene", scene,
        "--config", cfg_path,
        "--steps", str(steps),
        "--frame_idx", str(frame_idx),
        "--width", str(width),
        "--height", str(height),
        "--spp", str(spp),
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )

    out = proc.stdout
    psnr = None

    for line in out.splitlines():
        if "PSNR" in line:
            try:
                psnr = float(line.strip().split()[-1])
            except Exception:
                pass

    return psnr, out

# -------------------------
# GA loop
# -------------------------
def run_ga(args):
    random.seed(args.seed)

    with open(args.base, "r") as f:
        base_cfg = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "ga_log.csv")

    pop = [sample_individual(base_cfg) for _ in range(args.pop_size)]

    with open(csv_path, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["gen", "idx", "psnr", "lr", "l2", "cfg_path"])

        best_psnr = -1e9
        best_cfg = None

        for gen in range(1, args.generations + 1):
            print(f"\n=== Generation {gen}/{args.generations} ===")
            scored = []

            gen_dir = os.path.join(args.out_dir, f"gen{gen:02d}")
            os.makedirs(gen_dir, exist_ok=True)

            for i, ind in enumerate(pop):
                cfg_path = os.path.join(gen_dir, f"ind{i:02d}.json")
                save_json(cfg_path, ind)

                psnr, out = evaluate(
                    args.scene, cfg_path,
                    args.eval_steps,
                    args.frame_idx,
                    args.eval_width,
                    args.eval_height,
                    args.eval_spp,
                    args.eval_timeout
                )

                lr = deep_get(ind, OPTIMIZER_PATH).get("learning_rate")
                l2 = deep_get(ind, OPTIMIZER_PATH).get("l2_reg")

                print(f"Ind {i:02d} | PSNR={psnr} | lr={lr:.2e} l2={l2:.2e}")
                writer.writerow([gen, i, psnr, lr, l2, cfg_path])
                cf.flush()

                if psnr is not None and psnr > best_psnr:
                    best_psnr = psnr
                    best_cfg = deepcopy(ind)
                    save_json(os.path.join(args.out_dir, "best_ga_config.json"), best_cfg)
                    print(f" NEW BEST PSNR={psnr:.4f}")

                scored.append((psnr if psnr is not None else -1e9, ind))

            scored.sort(key=lambda x: x[0], reverse=True)

            elites = [deepcopy(scored[i][1]) for i in range(args.elite_count)]
            next_pop = elites[:]

            while len(next_pop) < args.pop_size:
                a = random.choice(elites)
                b = random.choice(scored)[1]
                child = crossover(a, b)
                child = mutate(child, args.mutation_rate)
                next_pop.append(child)

            pop = next_pop

    print("\n=== GA finished ===")
    print("Best PSNR:", best_psnr)

# -------------------------
# CLI
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--pop_size", type=int, default=8)
    p.add_argument("--generations", type=int, default=6)
    p.add_argument("--elite_count", type=int, default=1)
    p.add_argument("--mutation_rate", type=float, default=0.3)
    p.add_argument("--eval_steps", type=int, default=3000)
    p.add_argument("--eval_width", type=int, default=64)
    p.add_argument("--eval_height", type=int, default=64)
    p.add_argument("--eval_spp", type=int, default=1)
    p.add_argument("--eval_timeout", type=int, default=1200)
    p.add_argument("--frame_idx", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    run_ga(args)

if __name__ == "__main__":
    main()
