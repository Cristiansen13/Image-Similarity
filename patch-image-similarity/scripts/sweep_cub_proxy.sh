#!/bin/bash
# Tune the patch-level Proxy-Anchor knobs on CUB (cheap: ~8min/config) before
# baking a config into the expensive SOP run. Baseline to beat: the first-guess
# proxy config (alpha=16 M=8 proxy-lr=1e-2) hit 0.8130. All configs keep the
# proven augment + cosine-LR schedule and the winning CUB backbone recipe
# (P=16 K=4 steps=1000 lr=2e-5 unfreeze=4). Each config trains, evaluates full
# Recall@1 on the disjoint 101-200 test split, then cleans its large files.
set -e
cd /workspace/image-similarity/patch-image-similarity
source /venv/main/bin/activate
export HF_HUB_OFFLINE=1 HF_HOME=/workspace/.hf_home
CUB_DIR=/workspace/image-similarity/data/cub_raw/CUB_200_2011
BASE=/workspace/image-similarity/checkpoints_cub_proxy_sweep
mkdir -p "$BASE"

run_config () {
  NAME=$1; M=$2; ALPHA=$3; PLR=$4
  OUTDIR="$BASE/$NAME"; mkdir -p "$OUTDIR"
  echo "=== Config $NAME: M=$M alpha=$ALPHA proxy_lr=$PLR ==="
  python -u scripts/finetune_cub_proxy.py --cub-dir "$CUB_DIR" \
    --unfreeze-last-n 4 --P 16 --K 4 --steps 1000 --lr 2e-5 \
    --proxy-lr "$PLR" --M "$M" --alpha "$ALPHA" --delta 0.1 \
    --augment --lr-schedule --out-dir "$OUTDIR" 2>&1 | tail -4
  python -u scripts/eval_cub_test.py --cub-dir "$CUB_DIR" \
    --checkpoint "$OUTDIR/backbone_final.pt" --batch-size 128 2>&1 | tail -3
  RECALL=$(python -c "import json; print(json.load(open('$OUTDIR/cub_test_recall.json'))['recall_1'])")
  echo "RESULT $NAME M=$M alpha=$ALPHA proxy_lr=$PLR recall_1=$RECALL"
  rm -f "$OUTDIR"/backbone_step*.pt "$OUTDIR"/cub_embeddings_cache.pt
}

echo "SWEEP_START"
# reference (first-guess, already known ~0.8130) is A; vary one knob at a time
run_config A  8 16 1e-2   # reference config
run_config B  8 32 1e-2   # alpha up to paper default (now logsumexp-safe)
run_config C  8  8 1e-2   # alpha down (gentler for compressed score band)
run_config D  4 16 1e-2   # fewer prototype tokens
run_config E 16 16 1e-2   # more prototype tokens
run_config F  8 16 5e-2   # faster-moving proxies
echo "SWEEP_END"
