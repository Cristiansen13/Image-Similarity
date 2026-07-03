"""Rigorous multi-seed, multi-dataset evaluation suite, run on the remote GPU
instance. Reports mean +/- 95% CI per metric per dataset, following the spirit
of Musgrave et al. "A Metric Learning Reality Check" (2020): multiple seeds
(not one lucky run), R@1/2/4/8 + MAP@R (not just R@1), and an explicit
zero-shot control alongside our trained recipe.

For each dataset, runs:
  - one zero-shot DINOv2 control (no training)
  - N seeds of our best recipe (patch-level Proxy-Anchor + augment + cosine LR
    schedule), each: train -> encode+cache -> two-stage R@1/2/4/8/MAP@R -> cleanup

SOP skips the eval script's own brute-force recall pass (too expensive per
seed, ~57min) -- it's killed right after the embeddings cache is written, then
scored via the already-validated two-stage retrieval (matches brute-force
within noise, see two_stage_retrieval.py). CUB/CARS keep the eval script's
quick brute-force pass too (cheap at their scale) as an extra sanity check
alongside the two-stage numbers.

Usage: python run_rigorous_suite.py --root /workspace/image-similarity
"""
import argparse
import json
import os
import shutil
import subprocess
import time

import torch


def run(cmd, cwd, env, log_path, kill_after_cache=False):
    """Runs cmd, streaming to log_path. If kill_after_cache, terminates the
    process as soon as "Saving embeddings to cache" appears (used to skip
    SOP's expensive brute-force pass -- see module docstring)."""
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            if kill_after_cache and "Saving embeddings to cache" in line:
                time.sleep(5)  # let the save actually finish writing to disk
                proc.terminate()
                proc.wait(timeout=30)
                return 0
        proc.wait()
        return proc.returncode


def mean_ci95(values):
    n = len(values)
    if n == 0:
        return None, None
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = var ** 0.5
    ci95 = 1.96 * std / (n ** 0.5)
    return mean, ci95


def cub_dataset(root):
    cub_dir = f"{root}/data/cub_raw/CUB_200_2011"
    return {
        "name": "cub",
        "train_cmd": lambda out, seed: [
            "python", "-u", "scripts/finetune_cub_proxy.py", "--cub-dir", cub_dir,
            "--unfreeze-last-n", "4", "--P", "16", "--K", "4", "--steps", "1000",
            "--lr", "2e-5", "--proxy-lr", "5e-2", "--M", "8", "--alpha", "32",
            "--delta", "0.1", "--augment", "--lr-schedule", "--seed", str(seed),
            "--out-dir", out],
        "eval_cmd": lambda ckpt: [
            "python", "-u", "scripts/eval_cub_test.py", "--cub-dir", cub_dir,
            "--checkpoint", ckpt, "--batch-size", "128"],
        "two_stage_cmd": lambda cache: [
            "python", "-u", "scripts/two_stage_retrieval.py", "--dataset", "cub",
            "--labels-path", cub_dir, "--embeddings-cache", cache, "--top-k", "100"],
        "cache_name": "cub_embeddings_cache.pt",
        "seeds": [0, 1, 2, 3, 4],
        "skip_brute_force": False,
    }


def cars_dataset(root):
    cars_dir = f"{root}/data/cars_raw"
    return {
        "name": "cars",
        "train_cmd": lambda out, seed: [
            "python", "-u", "scripts/finetune_cars_proxy.py", "--cars-dir", cars_dir,
            "--unfreeze-last-n", "4", "--P", "16", "--K", "4", "--steps", "1000",
            "--lr", "2e-5", "--proxy-lr", "5e-2", "--M", "8", "--alpha", "32",
            "--delta", "0.1", "--augment", "--lr-schedule", "--seed", str(seed),
            "--out-dir", out],
        "eval_cmd": lambda ckpt: [
            "python", "-u", "scripts/eval_cars_test.py", "--cars-dir", cars_dir,
            "--checkpoint", ckpt, "--batch-size", "128"],
        "two_stage_cmd": lambda cache: [
            "python", "-u", "scripts/two_stage_retrieval.py", "--dataset", "cars",
            "--labels-path", cars_dir, "--embeddings-cache", cache, "--top-k", "100"],
        "cache_name": "cars_embeddings_cache.pt",
        "seeds": [0, 1, 2, 3, 4],
        "skip_brute_force": False,
    }


def sop_dataset(root):
    sop_root = f"{root}/data/sop_raw/Stanford_Online_Products"
    return {
        "name": "sop",
        "train_cmd": lambda out, seed: [
            "python", "-u", "scripts/finetune_sop_proxy.py",
            "--ebay-info", f"{sop_root}/Ebay_train.txt", "--images-root", sop_root,
            "--unfreeze-last-n", "4", "--P", "32", "--K", "4", "--steps", "3000",
            "--lr", "2e-5", "--proxy-lr", "5e-2", "--M", "8", "--alpha", "32",
            "--delta", "0.1", "--proxy-chunk", "1024", "--augment", "--lr-schedule",
            "--seed", str(seed), "--out-dir", out],
        "eval_cmd": lambda ckpt: [
            "python", "-u", "scripts/eval_full_test.py",
            "--ebay-test", f"{sop_root}/Ebay_test.txt", "--images-root", sop_root,
            "--checkpoint", ckpt, "--batch-size", "128"],
        "two_stage_cmd": lambda cache: [
            "python", "-u", "scripts/two_stage_retrieval.py", "--dataset", "sop",
            "--labels-path", f"{sop_root}/Ebay_test.txt", "--embeddings-cache", cache,
            "--top-k", "100"],
        "cache_name": "all_embeddings_cache.pt",
        "seeds": [0, 1, 2],
        "skip_brute_force": True,
    }


def run_one_seed(ds, root, out_base, seed, log_dir):
    out_dir = f"{out_base}/{ds['name']}_seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_HOME"] = f"{root}/.hf_home"
    cwd = f"{root}/patch-image-similarity"

    print(f"[{ds['name']} seed={seed}] training...")
    rc = run(ds["train_cmd"](out_dir, seed), cwd, env, f"{log_dir}/{ds['name']}_seed{seed}_train.log")
    if rc != 0:
        print(f"[{ds['name']} seed={seed}] TRAIN FAILED, skipping")
        return None

    ckpt = f"{out_dir}/backbone_final.pt"
    print(f"[{ds['name']} seed={seed}] evaluating (encode + cache)...")
    run(ds["eval_cmd"](ckpt), cwd, env, f"{log_dir}/{ds['name']}_seed{seed}_eval.log",
        kill_after_cache=ds["skip_brute_force"])

    cache_path = f"{out_dir}/{ds['cache_name']}"
    if not os.path.exists(cache_path):
        print(f"[{ds['name']} seed={seed}] embeddings cache missing, skipping two-stage")
        return None

    print(f"[{ds['name']} seed={seed}] two-stage metrics...")
    run(ds["two_stage_cmd"](cache_path), cwd, env, f"{log_dir}/{ds['name']}_seed{seed}_twostage.log")

    metrics_files = [f for f in os.listdir(out_dir) if f.startswith("two_stage_metrics")]
    result = None
    if metrics_files:
        with open(f"{out_dir}/{metrics_files[0]}") as f:
            result = json.load(f)

    # Cleanup: keep only backbone_final.pt + the metrics json, drop the big stuff.
    for fname in os.listdir(out_dir):
        if fname.startswith("backbone_step") or fname == ds["cache_name"]:
            os.remove(f"{out_dir}/{fname}")

    return result


def run_zero_shot(ds, root, out_base, log_dir):
    out_dir = f"{out_base}/{ds['name']}_zeroshot"
    os.makedirs(out_dir, exist_ok=True)
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_HOME"] = f"{root}/.hf_home"
    cwd = f"{root}/patch-image-similarity"

    print(f"[{ds['name']} zero-shot] evaluating...")
    run(ds["eval_cmd"]("zero-shot"), cwd, env, f"{log_dir}/{ds['name']}_zeroshot_eval.log",
        kill_after_cache=ds["skip_brute_force"])
    # zero-shot cache is written relative to the eval script's CWD (cwd, not this
    # driver's own CWD) per its os.path.dirname("zero-shot") == "" logic.
    written_cache_path = f"{cwd}/{ds['cache_name']}"
    if not os.path.exists(written_cache_path):
        print(f"[{ds['name']} zero-shot] cache not found at {written_cache_path}, skipping")
        return None
    shutil.move(written_cache_path, f"{out_dir}/{ds['cache_name']}")

    print(f"[{ds['name']} zero-shot] two-stage metrics...")
    run(ds["two_stage_cmd"](f"{out_dir}/{ds['cache_name']}"), cwd, env,
        f"{log_dir}/{ds['name']}_zeroshot_twostage.log")

    metrics_files = [f for f in os.listdir(out_dir) if f.startswith("two_stage_metrics")]
    result = None
    if metrics_files:
        with open(f"{out_dir}/{metrics_files[0]}") as f:
            result = json.load(f)
    os.remove(f"{out_dir}/{ds['cache_name']}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/image-similarity")
    ap.add_argument("--out-base", default=None)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--datasets", nargs="+", default=["cub", "cars", "sop"])
    args = ap.parse_args()

    out_base = args.out_base or f"{args.root}/rigorous_suite"
    log_dir = args.log_dir or f"{args.root}/rigorous_suite_logs"
    os.makedirs(out_base, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    builders = {"cub": cub_dataset, "cars": cars_dataset, "sop": sop_dataset}
    all_results = {}

    for name in args.datasets:
        ds = builders[name](args.root)
        all_results[name] = {"zero_shot": None, "seeds": []}

        zs = run_zero_shot(ds, args.root, out_base, log_dir)
        all_results[name]["zero_shot"] = zs
        print(f"[{name} zero-shot] {zs}")

        for seed in ds["seeds"]:
            t0 = time.time()
            result = run_one_seed(ds, args.root, out_base, seed, log_dir)
            elapsed = time.time() - t0
            print(f"[{name} seed={seed}] done in {elapsed:.0f}s: {result}")
            if result is not None:
                all_results[name]["seeds"].append(result)

        # Aggregate this dataset's seeds now, in case of interruption later.
        metrics = ["recall_at_1", "recall_at_2", "recall_at_4", "recall_at_8", "map_at_r"]
        summary = {}
        for m in metrics:
            vals = [r[m] for r in all_results[name]["seeds"] if r and m in r]
            mean, ci95 = mean_ci95(vals)
            summary[m] = {"mean": mean, "ci95": ci95, "n": len(vals), "values": vals}
        all_results[name]["summary"] = summary

        with open(f"{out_base}/results_so_far.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n=== {name} summary ===")
        for m in metrics:
            s = summary[m]
            if s["mean"] is not None:
                print(f"  {m}: {s['mean']:.4f} +/- {s['ci95']:.4f} (n={s['n']})")

    with open(f"{out_base}/results_final.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved final results to {out_base}/results_final.json")


if __name__ == "__main__":
    main()
