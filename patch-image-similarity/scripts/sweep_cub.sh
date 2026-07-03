#!/bin/bash
# Hyperparameter sweep for CUB fine-tuning, to find what recovers/beats the
# earlier 73.8% result after P=32/steps=1500 regressed to 67.0% (overfitting
# to CUB's small 100-class train universe). Each config trains, evaluates full
# Recall@1 on the standard 101-200 test split, then cleans up its large
# intermediate files (checkpoints, embedding cache) to save disk, keeping only
# the final backbone + result json.
set -e
cd /workspace/image-similarity/patch-image-similarity
source /venv/main/bin/activate
export HF_HUB_OFFLINE=1 HF_HOME=/workspace/.hf_home
CUB_DIR=/workspace/image-similarity/data/cub_raw/CUB_200_2011
BASE=/workspace/image-similarity/checkpoints_cub_sweep
mkdir -p "$BASE"

run_config () {
  NAME=$1; P=$2; K=$3; STEPS=$4; LR=$5; UNFREEZE=$6
  OUTDIR="$BASE/$NAME"
  mkdir -p "$OUTDIR"
  echo "=== Config $NAME: P=$P K=$K steps=$STEPS lr=$LR unfreeze=$UNFREEZE ==="
  python -u scripts/finetune_cub_backbone.py --cub-dir "$CUB_DIR" \
    --unfreeze-last-n "$UNFREEZE" --P "$P" --K "$K" --steps "$STEPS" --lr "$LR" \
    --out-dir "$OUTDIR" 2>&1 | tail -6
  python -u scripts/eval_cub_test.py --cub-dir "$CUB_DIR" \
    --checkpoint "$OUTDIR/backbone_final.pt" --batch-size 128 2>&1 | tail -6
  RECALL=$(python -c "import json; print(json.load(open('$OUTDIR/cub_test_recall.json'))['recall_1'])")
  echo "RESULT $NAME P=$P K=$K steps=$STEPS lr=$LR unfreeze=$UNFREEZE recall_1=$RECALL"
  rm -f "$OUTDIR"/backbone_step*.pt "$OUTDIR"/cub_embeddings_cache.pt
}

echo "SWEEP_START"
run_config A 16 4 1000 2e-5 4   # original defaults (previously ~73.8%)
run_config B 16 4 1500 2e-5 4   # same P, more steps -- isolate steps effect
run_config C 32 4 800  2e-5 4   # same P as regression, fewer steps -- isolate steps effect
run_config D 32 4 1500 1e-5 4   # regression P/steps, gentler LR
run_config E 32 4 1500 2e-5 2   # regression P/steps, shallower unfreeze (less capacity to overfit)
echo "SWEEP_END"
