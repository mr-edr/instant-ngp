#!/usr/bin/env python3
"""
Train for N steps, render one frame and return PSNR.

Place at: scripts/run_one_frame_psnr.py
Run from project root (so scripts/run.py is reachable).

Example:
python scripts/run_one_frame_psnr.py --scene data/nerf/fox --config configs/nerf/base.json --steps 2000 --frame_idx 0
"""
import os
import sys
import subprocess
import argparse
import time
import json
import re
from PIL import Image
import numpy as np
import uuid
import glob

def call_run_py(args_list, timeout=None):
    py = sys.executable
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    run_py = os.path.join(SCRIPT_DIR, "run.py")

    if not os.path.exists(run_py):
        raise FileNotFoundError(f"scripts/run.py not found at {run_py}")
    cmd = [py, run_py] + args_list
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    out_lines = []
    start = time.time()
    try:
        for line in iter(proc.stdout.readline, ''):
            if line == '':
                break
            out_lines.append(line)
            # optional: mirror to console
            print(line, end="")
            if timeout and (time.time() - start) > timeout:
                proc.kill()
                return False, "".join(out_lines) + "\n*** TIMEOUT ***"
        proc.wait()
    except KeyboardInterrupt:
        try:
            proc.kill()
        except Exception:
            pass
        raise
    return proc.returncode == 0, "".join(out_lines)

def compute_psnr(img_a_path, img_b_path):
    a = Image.open(img_a_path).convert("RGB")
    b = Image.open(img_b_path).convert("RGB")

    # resize GT to match rendered image size
    if a.size != b.size:
        b = b.resize(a.size, Image.BICUBIC)

    a = np.asarray(a).astype(np.float32) / 255.0
    b = np.asarray(b).astype(np.float32) / 255.0

    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def find_transforms(transforms_arg, scene):
    if transforms_arg:
        transforms_path = transforms_arg
    else:
        # default to scene/transforms.json
        cand = os.path.join(scene, "transforms.json")
        if os.path.exists(cand):
            transforms_path = cand
        else:
            raise FileNotFoundError("No transforms.json found. Provide --transforms <file> or ensure scene has transforms.json.")
    return transforms_path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True, help="Path to NeRF scene folder (contains transforms.json)")
    p.add_argument("--config", required=True, help="Path to network JSON (instant-ngp config)")
    p.add_argument("--steps", type=int, default=2000, help="Training steps")
    p.add_argument("--frame_idx", type=int, default=0, help="Index of frame in transforms.json to evaluate")
    p.add_argument("--width", type=int, default=64, help="Render width (small for GA fitness)")
    p.add_argument("--height", type=int, default=64, help="Render height")
    p.add_argument("--spp", type=int, default=1, help="Samples per pixel for screenshot")
    p.add_argument("--transforms", type=str, default=None, help="Path to transforms.json (optional)")
    p.add_argument("--timeout_train", type=int, default=3600, help="Timeout for training (s)")
    p.add_argument("--timeout_eval", type=int, default=300, help="Timeout for eval (s)")
    args = p.parse_args()

    # create snapshot path unique per run
    out_dir = os.path.join("ga_eval_snaps")
    os.makedirs(out_dir, exist_ok=True)
    snap_path = os.path.join(out_dir, f"snap_{uuid.uuid4().hex[:8]}.ingp")

    # 1) Train and save snapshot
    train_args = [
        "--scene", args.scene,
        "--network", args.config,
        "--train",
        "--n_steps", str(args.steps),
        "--width", str(args.width), "--height", str(args.height),
        "--save_snapshot", snap_path
    ]
    print(">>> Training (will save snapshot):", snap_path)
    ok, train_out = call_run_py(train_args, timeout=args.timeout_train)
    if not ok:
        print("Warning: training returned non-zero or timed out; still attempting evaluation.", file=sys.stderr)

    # 2) Find transforms.json and the ground-truth image for the requested frame index
    transforms_path = find_transforms(args.transforms, args.scene)
    with open(transforms_path, "r") as f:
        transforms = json.load(f)

    frames = transforms.get("frames") or transforms.get("images") or None
    if not frames:
        print("ERROR: transforms.json contains no 'frames' list.", file=sys.stderr)
        sys.exit(2)
    if args.frame_idx < 0 or args.frame_idx >= len(frames):
        print(f"ERROR: frame_idx {args.frame_idx} out of range (0..{len(frames)-1})", file=sys.stderr)
        sys.exit(2)
    frame = frames[args.frame_idx]
    # field may be 'file_path' or 'file_path' variants
    gt_rel = frame.get("file_path") or frame.get("file") or frame.get("filePath") or None
    if not gt_rel:
        print("ERROR: selected frame does not contain 'file_path' to ground-truth image.", file=sys.stderr)
        sys.exit(2)
    transforms_dir = os.path.dirname(transforms_path)
    gt_path = gt_rel if os.path.isabs(gt_rel) else os.path.normpath(os.path.join(transforms_dir, gt_rel))

    # try common image extensions
    if not os.path.exists(gt_path):
        for ext in [".png", ".jpg", ".jpeg"]:
            if os.path.exists(gt_path + ext):
                gt_path = gt_path + ext
                break

    if not os.path.exists(gt_path):
        alt = os.path.join(args.scene, os.path.basename(gt_path))
        if os.path.exists(alt):
            gt_path = alt
        else:
            print(f"ERROR: ground-truth image not found at {gt_path}", file=sys.stderr)
            sys.exit(2)

    # 3) Render single frame from saved snapshot (no training)
    tmp_dir = os.path.join("ga_eval_renders")
    os.makedirs(tmp_dir, exist_ok=True)
    # Use screenshot_transforms with screenshot_frames to render specific frames to tmp_dir
    eval_args = [
        "--scene", args.scene,
        "--load_snapshot", snap_path,
        "--screenshot_transforms", transforms_path,
        "--screenshot_dir", tmp_dir,
        "--screenshot_frames", str(args.frame_idx),
        "--screenshot_spp", str(args.spp),
        "--width", str(args.width),
        "--height", str(args.height),
        "--n_steps", "0"
    ]
    print(">>> Rendering single frame to dir:", tmp_dir)
    ok_eval, eval_out = call_run_py(eval_args, timeout=args.timeout_eval)
    if not ok_eval:
        print("Warning: eval run returned non-zero or timed out; parsing outputs anyway.", file=sys.stderr)

    # 4) Find rendered image file in tmp_dir
    pngs = sorted(glob.glob(os.path.join(tmp_dir, "*.png")))
    rendered_path = None
    if pngs:
        # prefer file names that match ground-truth basename
        gt_basename = os.path.basename(gt_path)
        for p in pngs:
            if gt_basename in os.path.basename(p):
                rendered_path = p
                break
        if not rendered_path:
            rendered_path = pngs[0]
    else:
        # fallback: maybe run.py wrote ref.png/out.png in cwd
        if os.path.exists("out.png"):
            rendered_path = "out.png"
        elif os.path.exists(os.path.join(tmp_dir, "out.png")):
            rendered_path = os.path.join(tmp_dir, "out.png")

    if not rendered_path or not os.path.exists(rendered_path):
        print("ERROR: could not find rendered PNG in", tmp_dir, file=sys.stderr)
        # print eval stdout for debugging
        print(eval_out)
        sys.exit(2)

    # 5) Compute PSNR between rendered_path and gt_path
    try:
        psnr_val = compute_psnr(rendered_path, gt_path)
    except Exception as e:
        print("ERROR computing PSNR:", e, file=sys.stderr)
        sys.exit(2)

    print(f"\n=== PSNR (frame {args.frame_idx}) === {psnr_val:.6f}")
    # exit 0 success
    sys.exit(0)

if __name__ == "__main__":
    main()
