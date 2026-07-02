# Patch-level image similarity via late interaction — project spec (v0)

**How to use this doc:** hand this whole file to Claude Code with an instruction like
"implement Phase 0 first, get it working end to end on a handful of test images,
show me the results, then we'll move to Phase 1." Don't ask it to build everything
at once — the phases are ordered so each one produces a working, testable result
before the next one adds complexity.

## 1. Research question

Does comparing images via **patch-level late interaction** (matching local regions
between two images instead of collapsing each image into one global vector) work
better than standard global embeddings — and specifically, does adding an
**LLM-generated text description per patch** improve on raw vision-encoder patch
embeddings, or does it only add interpretability without improving accuracy?

Two separate questions bundled together on purpose:
1. Does patch-level matching beat a single global embedding? (Probably yes for
   some cases — spatially shifted/cropped content especially.)
2. Does routing patches through language before comparing them help, hurt, or
   just add a "why" on top of a similar accuracy number? (Genuinely open — this
   is the actual novel contribution.)

## 2. Why this architecture (context, not to re-litigate)

- Global embeddings (CLIP, DINOv2, SigLIP2) are the strong existing baseline —
  cheap, one forward pass, no hallucination risk. Any patch-based method needs to
  be compared against this directly, not against weaker tools like perceptual
  hashing.
- Late interaction / MaxSim (ColBERT → ColPali) is an established, zero-shot,
  no-training way to compare two sets of vectors by best-match rather than
  pooling — this is the mechanism to build first, before considering any learned
  attention layer.
- Learned cross-attention matching (SuperGlue/LoFTR/LightGlue) is the more
  powerful but heavier alternative — real prior art exists here, but it needs
  training data and is a Phase 2+ concern, not where to start.

## 3. Overall pipeline

```
Image A ──► patch encoder ──► patch vectors A ─┐
                                                 ├──► late interaction (MaxSim) ──► similarity score
Image B ──► patch encoder ──► patch vectors B ─┘                                  + best-matching patch pairs
```

"Patch encoder" is deliberately vague above because it changes between phases:
in Phase 0 it's a frozen vision transformer, in Phase 1 it's that same vision
transformer *plus* an LLM-generated caption per patch, embedded as text.

---

## Phase 0 — Baseline: zero-shot vision-encoder patches + MaxSim

**Goal:** a working similarity score with zero LLM calls, in an afternoon. This
is the number every later phase has to beat or meaningfully differentiate from.

**Steps:**

1. Load a pretrained ViT that exposes patch tokens, not just a pooled embedding.
   Recommended: **DINOv2** (e.g. `facebook/dinov2-base` via `transformers`) — it
   has shown stronger performance than CLIP specifically on similarity/retrieval
   tasks in prior benchmarking. Keep CLIP or SigLIP2 available as a
   swappable alternative for comparison later.
2. Run both images through the encoder. Take the **patch token embeddings**
   (drop the CLS/global token). For a 224×224 image with 14px patches, this is a
   16×16 = 256-vector grid per image. Downsample resolution or use a
   coarser-patch model variant if 256×256 comparisons are too slow to iterate on.
3. L2-normalize every patch vector.
4. Implement **symmetric MaxSim (Chamfer similarity)**:

   For patch sets `A = {a_1 ... a_n}` and `B = {b_1 ... b_m}`, similarity matrix
   `S[i,j] = a_i · b_j` (cosine, since normalized).

   ```
   score(A→B) = mean_i( max_j S[i,j] )   # each patch in A finds its best match in B
   score(B→A) = mean_j( max_i S[i,j] )   # symmetric
   similarity(A,B) = 0.5 * (score(A→B) + score(B→A))
   ```

   Symmetric because you're not doing query→document retrieval, you're doing
   image↔image comparison, and asymmetric scoring will bias results if the two
   images have different amounts of "clutter."

5. Also compute standard **global cosine similarity** (CLS token or mean-pooled
   patch embedding) as a sanity-check baseline sitting right next to MaxSim —
   you want both numbers on every pair, always, for comparison.
6. Sanity test on ~10 manually chosen pairs: a near-duplicate pair, a
   same-object-different-background pair, a same-scene-cropped pair, and a few
   clearly unrelated pairs. Confirm MaxSim ranks them sensibly before moving on.

**Output of this phase:** a script that takes two image paths and returns both a
MaxSim score and a global-embedding cosine score, plus a quick eyeball check that
the ranking looks right.

---

## Phase 1 — Add LLM-generated patch/region descriptions

**Goal:** find out whether routing patches through language changes the score
in a meaningful way, and produce the interpretability payoff (readable
patch-to-patch match explanations).

**Steps:**

1. **Don't caption patches one-by-one.** A 16×16 grid is 256 LLM calls per
   image — slow, expensive, and each isolated patch is often ambiguous without
   surrounding context (a gray textured patch could be fur, fabric, or smoke).
2. Instead, use a coarser grid (e.g. 4×4 or 6×6 — pick based on cost/detail
   tradeoff) and generate **all region descriptions in a single call**: overlay
   numbered grid lines on the full image (e.g. with PIL), send the whole image
   once, and prompt the model to return a structured JSON object with one short
   description per numbered cell, written *in the context of the whole image it
   can see*. This is grounded/referring captioning, not blind patch captioning —
   the model sees the full picture, just reports on each region.
3. Keep each region description short and structured (a phrase, not a
   paragraph) — e.g. "gray tabby cat's back, mid-frame" rather than a free-form
   essay. Long free-form captions are measurably more prone to hallucination the
   longer they get, and you don't want that noise here.
4. Embed each region description with a sentence embedding model (e.g.
   `sentence-transformers/all-MiniLM-L6-v2` for fast iteration, or
   `BAAI/bge-base-en-v1.5` / `thenlper/gte-base` for better quality once the
   pipeline works).
5. Run the exact same symmetric MaxSim scoring function from Phase 0, but over
   these text-embedding vectors instead of (or concatenated with) the raw vision
   patch embeddings.
6. For every test pair, log which region in A matched which region in B (the
   argmax pairs from the MaxSim computation) along with both regions' text
   descriptions — this readable trace is the actual interpretability deliverable.
7. Use temperature 0 (or as close to deterministic as the model allows) for the
   captioning call — you want the same image to produce the same regions
   descriptions run to run, since you're caching/reusing these.

**Output of this phase:** the same two-image → score function as Phase 0, plus a
"why" trace, plus a direct number-vs-number comparison against Phase 0 on the
same test pairs.

---

## Phase 2 — Optional, later: learned cross-attention matching

Only pursue this if zero-shot MaxSim clearly underperforms and you want to
close the gap. This is where you'd be re-deriving something closer to
SuperGlue/LoFTR: a small transformer that takes both patch sets jointly and
learns to score correspondences, trained on positive/negative image pairs
(can bootstrap positives cheaply via augmentation — crops, color jitter,
rotation — of the same source image). Not needed for an MVP; flag as future
work rather than building now.

---

## 4. Evaluation plan (keep this lightweight to start)

Use **triplet accuracy**: curate a set of `(anchor, positive, negative)`
triplets where positive is "should score as similar to anchor" and negative is
"should score as dissimilar." For each triplet, check whether
`score(anchor, positive) > score(anchor, negative)`. Accuracy = fraction of
triplets where that holds.

- **Cheap synthetic triplets** (free, generate as many as you want): anchor +
  an augmented/cropped/rotated version of itself as positive, + any other random
  image as negative. Tests robustness to spatial transforms — exactly the case
  MaxSim should be strong at versus a global embedding.
- **Hand-picked semantic triplets** (~20–50, worth the manual effort): cases
  where "similar" means same object/scene/style rather than same pixels — this
  is the harder, more interesting evaluation and where Phase 1 vs Phase 0 should
  actually differ if the language layer is adding anything.

Report triplet accuracy for: global embedding baseline, Phase 0 MaxSim, Phase 1
MaxSim. Three numbers side by side is the whole point of the comparison.

## 5. Suggested project structure

```
patch-image-similarity/
  data/
    test_images/
    triplets.json          # anchor/positive/negative file paths + labels
  src/
    encoders.py             # DINOv2/CLIP wrapper -> patch embeddings
    caption_regions.py      # grid overlay + single-call LLM captioning
    text_embed.py            # sentence-transformer wrapper
    maxsim.py                 # symmetric MaxSim / Chamfer scoring
    eval.py                     # triplet accuracy harness
  scripts/
    run_phase0.py
    run_phase1.py
  README.md
```

## 6. Task checklist for Claude Code

- [ ] Set up environment: `torch`, `transformers`, `sentence-transformers`, `Pillow`
- [ ] `encoders.py`: load DINOv2, return normalized patch embeddings for an image path
- [ ] `maxsim.py`: implement symmetric MaxSim scoring exactly as specified in Phase 0 step 4
- [ ] Manually gather ~10 test image pairs spanning near-duplicate → unrelated
- [ ] Run Phase 0 end to end, eyeball-check the ranking makes sense
- [ ] Build `triplets.json` with a handful of synthetic + hand-picked triplets
- [ ] `eval.py`: compute triplet accuracy for global-embedding baseline vs Phase 0 MaxSim
- [ ] `caption_regions.py`: grid overlay + single structured captioning call, temperature 0
- [ ] `text_embed.py` + rerun `maxsim.py` over text embeddings for Phase 1
- [ ] Re-run `eval.py` including Phase 1, compare all three numbers
- [ ] Log and print match traces (which region matched which, with captions) for a few pairs

## 7. Known pitfalls to design around from the start

- Long, unconstrained captions get *less* reliable the longer they are —
  keep region descriptions short and structured, not essay-length.
- A patch/region captioned in isolation is ambiguous — always caption with the
  full image visible, never a bare crop.
- Fixed-grid patches won't align positionally between differently-composed
  images — that's precisely why MaxSim (content-based best-match) is used
  instead of comparing patch `i` to patch `i`.
- Normalize embeddings before any cosine similarity — easy to forget and get
  silently wrong numbers.
- Keep the global-embedding baseline computed alongside every experiment, not
  as an afterthought — it's the number that determines whether any of this was
  worth building.
