
# Notebooks

There are two notebooks. Run them in order on Google Colab with a T4 GPU
(Runtime → Change runtime type → T4 GPU).

## How to run

1. **`Milestone1_generative.ipynb`** (data prep): downloads COCO val2017 captions, picks 2,000
   image–caption pairs, resizes and center-crops the images to 512×512 RGB, and writes
   `manifest.csv` plus `preprocessing_config.json`.
2. **`02_text_embeddings.ipynb`** (model lead): reads `manifest.csv` from step 1 and builds the
   text→embedding baseline.

Both notebooks use the same Drive folder: `MyDrive/IE_7615_generative/data`.
Notebook 2 needs `manifest.csv`, so run notebook 1 first (or copy `data/manifest.csv` from this
repo into that Drive folder). Run all cells top to bottom. Install the pinned libraries with
`pip install -r requirements.txt` (the first cell of notebook 2 also does this).

## What `Milestone1_generative.ipynb` does
1. Downloads the COCO val2017 caption annotations (only the annotation file) to `raw/` in the Drive data folder.
2. Picks one caption per image using a fixed rule: the first caption (lowest annotation ID).
3. Cleans and selects the subset:
   * strips whitespace and collapses repeated spaces
   * drops empty captions, captions under 3 words, and duplicates (compared after lowercasing and removing punctuation)
   * shuffles the images with seed 42 and keeps the first 2,000 that pass
4. Downloads only the 2,000 selected images, resizes the shorter side to 512, center-crops to 512×512, converts to RGB, and saves them as lossless PNGs in `processed_512/`. A full run prints `Processed: 2000 | Failed: 0`.
5. Writes the two files the rest of the team reads from:
   * `manifest.csv`: `image_id`, `file_path` (relative to the data folder), `caption`
   * `preprocessing_config.json`: dataset, seed, caption rule, minimum words, image size, dropped counts

## What `02_text_embeddings.ipynb` does

1. Installs `torch`, `diffusers`, `transformers` and `accelerate` and writes the pinned
   versions to `requirements.txt`.
2. Loads the CLIP tokenizer and text encoder from the Stable Diffusion v1.5 checkpoint
   (`stable-diffusion-v1-5/stable-diffusion-v1-5`), so the embeddings match what the U-Net receives.
3. Turns each caption into tokens (padded or truncated to 77) and then into embeddings:
   - tokens: `[B, 77]`
   - per-token hidden states: `[B, 77, 768]` (what the U-Net's cross-attention uses)
   - pooled vector: `[B, 768]` (the end-of-text token's hidden state)
4. Sanity-checks the embeddings with cosine similarity: paraphrased captions score about
   0.84–0.86, unrelated captions about 0.22–0.37. Nearest neighbours of real captions are
   semantically related.
5. Counts truncation: 0 of 2,000 captions exceed 77 tokens (max 30, median 13).
6. Embeds all 2,000 captions and saves them to `embeddings/` in the Drive data folder:
   - `text_embeddings.pt`: hidden states, pooled vectors, image IDs (same order as the manifest),
     and the empty-prompt embedding for classifier-free guidance later
   - `token_stats.csv` and `embedding_config.json` (also copied to `results/` in this repo)

`text_embeddings.pt` (~236 MB) is not in the repo. It is on Drive.
