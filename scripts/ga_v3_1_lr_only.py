#!/usr/bin/env python3
"""
GA v3.1 — Learning-rate–only Genetic Algorithm for Instant-NGP
Fitness = single-frame PSNR via run_one_frame_psnr.py

Place at:
  instant-ngp/scripts/ga_v3_1_lr_only.py

Run from project root:
  python instant-ngp/scripts/ga_v3_1_lr_only.py ...
"""

import argparse
import os
import sys
import json
import random
import subprocess
import math
import csv
import re
from copy import deepcopy

# ------------------------------------------------------------
# PSNR extraction (FIXED — this was your bug)
# ------------------------------------------------------------
def extract_psnr(output: str):
    # Primary pattern (your evaluator prints this)
    m = re.search(
        r"PSNR\s*\(frame\s*\d+\)\s*===\s*([0-9eE+\-.]+)",
        output
    )
    if m:
        return float(m.group(1))

    # Fallbacks
    m = re.search(r"PSNR\s*[:=]\s*([0-9eE+\-.]+)", output, re.IGNORECASE)
    if m:
        return float(m.group(1))

    return None


# ------------------------------------------------------------
# Optimizer-only genome
# ------------------------------------------------------------
def random_genome():
    return {
        "learning_rate": 10 ** random.uniform(-4, -1.3),  # ~1e-4 to 5e-2
        "l2_reg": 10 ** random.uniform(-8, -4)
    }


def mutate(genome, rate=0.2):
    g = genome.copy()
    if random.random() < rate:
        g["learning_rate"] *= math.exp(random.uniform(-0.3, 0.3))
        g["learning_rate"] = min(max(g["learning_rate"], 1e-5), 5e-2)

    if random.random() < rate:
        g["l2_reg"] *= math.exp(random.uniform(-0.5, 0.5))
        g["l2_reg"] = min(max(g["l2_reg"], 0.0), 1e-3)

    return g


# ------------------------------------------------------------
# Apply genome to base.json
# ------------------------------------------------------------
def apply_genome(base_cfg, genome):
    cfg = deepcopy(base_cfg)

    # Adam optimizer path used by instant-ngp base.json
    opt = cfg.setdefault("optimizer", {})
    nested = opt.setdefault("nested", {})
    inner = nested.setdefault("nested", {})

    inner["otype"] = "Adam"
    inner["learning_rate"] = float(genome["learning_rate"])
    inner["l2_reg"] = float(genome["l2_reg"])

    return cfg


# ------------------------------------------------------------
# Run evaluator
# ------------------------------------------------------------
def evaluate(scene, cfg_path, args):
    cmd = [
        sys.executable,
        os.path.join("instant-ngp", "scripts", "run_one_frame_psnr.py"),
        "--scene", scene,
        "--config", cfg_path,
        "--steps", str(args.eval_steps),
        "--frame_idx", str(args.frame_idx),
        "--width", str(args.eval_width),
        "--height", str(args.eval_height),
        "--spp", str(args.eval_spp),
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=args.eval_timeout
    )

    psnr = extract_psnr(proc.stdout)
    return psnr, proc.stdout


# ------------------------------------------------------------
# GA loop
# ------------------------------------------------------------
def run_ga(args):
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.base) as f:
        base_cfg = json.load(f)

    population = [random_genome() for _ in range(args.pop_size)]
    best_psnr = -1e9
    best_genome = None

    csv_path = os.path.join(args.out_dir, "ga_log.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["generation", "idx", "psnr", "learning_rate", "l2_reg"])

        for gen in range(1, args.generations + 1):
            print(f"\n=== Generation {gen}/{args.generations} ===")
            scored = []

            for i, genome in enumerate(population):
                cfg = apply_genome(base_cfg, genome)
                cfg_path = os.path.join(args.out_dir, f"gen{gen:02d}_ind{i:02d}.json")

                with open(cfg_path, "w") as f:
                    json.dump(cfg, f, indent=2)

                psnr, out = evaluate(args.scene, cfg_path, args)
                print(
                    f"Ind {i:02d} | PSNR={psnr} | "
                    f"lr={genome['learning_rate']:.2e} "
                    f"l2={genome['l2_reg']:.2e}"
                )

                writer.writerow([gen, i, psnr, genome["learning_rate"], genome["l2_reg"]])
                csvfile.flush()

                if psnr is not None:
                    scored.append((psnr, genome))
                    if psnr > best_psnr:
                        best_psnr = psnr
                        best_genome = genome
                        with open(os.path.join(args.out_dir, "best_ga_config.json"), "w") as bf:
                            json.dump(cfg, bf, indent=2)

            # Selection
            if scored:
                scored.sort(reverse=True, key=lambda x: x[0])
                elites = [scored[0][1]]
            else:
                elites = [random_genome()]

            # Reproduce
            next_pop = elites.copy()
            while len(next_pop) < args.pop_size:
                parent = random.choice(elites)
                child = mutate(parent, args.mutation_rate)
                next_pop.append(child)

            population = next_pop

    print("\n=== GA finished ===")
    print("Best PSNR:", best_psnr)
    print("Best genome:", best_genome)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--out_dir", required=True)

    p.add_argument("--pop_size", type=int, default=8)
    p.add_argument("--generations", type=int, default=6)
    p.add_argument("--mutation_rate", type=float, default=0.3)

    p.add_argument("--eval_steps", type=int, default=2000)
    p.add_argument("--eval_width", type=int, default=64)
    p.add_argument("--eval_height", type=int, default=64)
    p.add_argument("--eval_spp", type=int, default=1)
    p.add_argument("--eval_timeout", type=int, default=1200)
    p.add_argument("--frame_idx", type=int, default=0)

    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


if __name__ == "__main__":
    run_ga(parse_args())
