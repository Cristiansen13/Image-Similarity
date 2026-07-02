# Patch-level image similarity — Phase 0

Baseline: frozen DINOv2 (`facebook/dinov2-base`) patch tokens + symmetric
MaxSim (Chamfer similarity), compared against a global CLS-token cosine
similarity baseline. See `../patch_image_similarity_spec.md` for the full
multi-phase plan.

## Setup

```
pip install torch transformers sentence-transformers scikit-image pillow
```

## Run

```
python scripts/make_test_pairs.py   # regenerate data/test_images/ + pairs.json
python scripts/run_phase0.py        # encode pairs, print MaxSim + global cosine table
```

Test pairs are built from scikit-image's bundled sample photos (no
downloads) plus PIL transforms: JPEG/resize/noise for near-duplicates, real
stereo parallax (`stereo_motorcycle`) for a same-scene-shifted pair, crops for
same-scene-cropped, and a foreground crop composited onto a texture
background for same-object-different-background. See
`data/test_images/pairs.json` for the manifest and expected-similarity labels.

## Phase 0 results (11 sanity pairs)

Both MaxSim and global cosine rank all pairs correctly by bucket (high >
medium > low) with no crossover — see `data/test_images/phase0_results.json`
for the full numbers. Next step (Phase 1) is adding LLM-generated per-region
captions and re-running the same MaxSim scorer over text embeddings, to see
whether it changes these numbers or only adds interpretability.
