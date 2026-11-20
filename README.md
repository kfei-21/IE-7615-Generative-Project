# IE-7615 Generative Project: From Words to Worlds

Text-to-image generation using Transformer text encoders (CLIP / OpenCLIP) and pre-trained latent
diffusion models. Two checkpoints are compared on the same captions and settings: **Stable Diffusion v1.5**
and **Stable Diffusion v2.1-base**.

Course: IE 7615 Deep Learning for AI.

## Status
- Milestone 1 (setup, dataset preparation, text→embedding baseline, sample testing): complete.
- Milestone 2 (baseline conditional generation, guidance and noise-schedule tuning): complete.

## Repository structure

```
.
├── README.md
├── requirements.txt                 # pinned library versions
├── data/
│   ├── manifest.csv                 # 2,000 image–caption pairs (image_id, file_path, caption)
│   ├── preprocessing_config.json    # rules used to build the manifest
│   ├── prompt_set_m2.csv            # the 10 Milestone 2 test prompts
│   └── fid_reference_ids.json       # 1,000-image FID reference set for Milestone 3
├── notebooks/
│   ├── 01_data_prep.ipynb
│   ├── 02_text_embeddings_sd15.ipynb
│   ├── 02_text_embeddings_sd21.ipynb
│   ├── 03_eval_sets.ipynb
│   ├── 04_generate.ipynb
│   └── 05_evaluate.ipynb
├── results/
│   ├── milestone1/                  # per-model configs, token stats, figures
│   └── milestone2/                  # generation logs, CLIP scores, plots, samples, report
└── src/                             # reusable modules (later milestones)
```

Images and embeddings are not stored in this repo (see [Data and large files](#data-and-large-files)).

## Setup

Developed and tested on Google Colab with a T4 GPU.

### 1. Google Drive working folder
All notebooks read and write a shared Drive folder at <https://drive.google.com/drive/folders/10ZaS0nFiGMs0Bc-1G9cuR4TB_wbmR2VU?usp=sharing>:

```
MyDrive/IE7615_Project2/
├── requirements.txt
├── notebooks/            # Colab copies of the notebooks in this repo
├── data/
│   ├── manifest.csv, preprocessing_config.json, prompt_set_m2.csv, fid_reference_ids.json
│   ├── raw/              # captions_val2017.json
│   └── processed_512/    # 2,000 processed images (Drive only)
├── embeddings/           # text_embeddings_*.pt (Drive only)
└── results/
    ├── milestone1/       # configs, token stats, figures, individual samples
    └── milestone2/       # logs, scores, plots, samples, report
        └── images/       # all 420 generated images and review sheets (Drive only)
```

The folder is shared with team members. A shared folder appears under **Shared with me**, not
**My Drive**, so the path above will not resolve until you add a shortcut. Right-click the
folder → **Organize → Add shortcut** → **My Drive**.

The path is set by `ROOT` in each notebook's first code cell.

### 2. Libraries
Each notebook installs the pinned versions from `requirements.txt` before importing anything:

```
torch==2.11.0
torchvision==0.26.0
diffusers==0.42.1
transformers==5.22.0
accelerate==1.15.0
```

If pip reports that it installed or changed a package, restart the session and run all again.

## How to run

Run the notebooks in order, each with **Runtime → Restart session**, then **Run all**.

### Milestone 1

#### `01_data_prep.ipynb`
Selects one cleaned caption per COCO val2017 image (lowest annotation ID), keeps a seeded 2,000-pair
subset (seed 42, 6 duplicate captions dropped), downloads each image, resizes the shorter side to
512 (LANCZOS) and center-crops to 512×512 RGB PNG.

- **Writes:** `data/manifest.csv`, `data/preprocessing_config.json`, `data/processed_512/`,
  `results/milestone1/example_pairs.png`
- **Rerunning:** existing images are skipped. If a manifest already exists, the new run is compared
  against it and the notebook stops if they differ. Expected output:
  `Existing manifest identical to this run: True`.
- Uses `data/raw/captions_val2017.json` if present; otherwise downloads the COCO annotations.

#### `02_text_embeddings_sd15.ipynb` / `02_text_embeddings_sd21.ipynb`
Loads the tokenizer and text encoder from each pinned Stable Diffusion checkpoint, embeds all 2,000
captions, verifies the saved embeddings end to end, and generates 5 sample images.

The two notebooks are identical except for three config lines:

| | SD 1.5 | SD 2.1-base |
|---|---|---|
| `CKPT` | `stable-diffusion-v1-5/stable-diffusion-v1-5` | `sd2-community/stable-diffusion-2-1-base` |
| `REVISION` | `451f4fe16113bff5a5d2269ed5ad43b0592e9a14` | `4e63672c03103b6c636b8fb4119ba982469b2955` |
| `MODEL_TAG` | `sd15` | `sd21` |
| Text encoder | OpenAI CLIP ViT-L/14, 12 layers, 768-d | OpenCLIP ViT-H, 23 layers (penultimate), 1024-d |
| Padding token | end-of-text (ID 49407) | `!` (ID 0) |

Both checkpoints are community-maintained mirrors: the original RunwayML SD 1.5 and Stability AI
SD 2.x repositories have been removed from Hugging Face. Revisions are pinned to exact commits so the
weights cannot change underneath the project.

Other shared settings: `SEED = 42`, `STEPS = 30`, `GUIDANCE = 7.5`, `CAPTION_COL = "caption"`, 512×512 output.

- **Writes:** `embeddings/text_embeddings_{MODEL_TAG}_caption.pt`,
  `results/milestone1/embedding_config_{MODEL_TAG}_caption.json`,
  `results/milestone1/token_stats_{MODEL_TAG}_caption.csv`, sample images and figures in
  `results/milestone1/`
- **Rerunning:** with `OVERWRITE = False` (the default), an existing embedding file is not replaced;
  instead it is checked against the fresh run. Expected output: `matches this run: True`.
  Set `OVERWRITE = True` only to regenerate deliberately, then set it back.
- **Pipeline note:** the SD 1.5 pipeline includes a safety checker; SD 2.1-base does not.

### Milestone 2

#### `03_eval_sets.ipynb`
Fixes the 10 test prompts (the five Milestone 1 difficulty types plus colour binding, visible text,
crowd, food close-up and cluttered indoor) and a 1,000-image FID reference set for Milestone 3.

- **Writes:** `data/prompt_set_m2.csv`, `data/fid_reference_ids.json`,
  `results/milestone2/prompt_set_m2.png`
- **Rerunning:** both files refuse to change once saved; a rerun checks them instead.

#### `04_generate.ipynb`
Passes the saved Milestone 1 embeddings to the U-Net as `prompt_embeds` and generates every setting
for the 10 prompts at seed 42, 512×512, fp16. Run it once with `MODEL_TAG = "sd15"` and once with
`MODEL_TAG = "sd21"`; nothing else changes between the two runs.

| Sweep | Settings |
|---|---|
| Baseline | DDIM, 30 steps, guidance 7.5 |
| Guidance | 1, 3, 5, 7.5, 10, 15 at DDIM, 30 steps (1 = guidance off) |
| Scheduler | PNDM, DDIM, Euler, DPM-Solver++ 2M × 10, 20, 30, 50 steps at guidance 7.5 |

That is 21 settings × 10 prompts = 210 images per model. Changing the scheduler swaps the sampler
and its timestep spacing; the checkpoint's trained noise schedule is unchanged. The safety checker is
off so timings are comparable and no image is blacked out.

- **Writes:** `results/milestone2/images/m2_{MODEL_TAG}_v1/` (Drive only),
  `results/milestone2/run_log_m2_{MODEL_TAG}_v1.csv`, `results/milestone2/config_m2_{MODEL_TAG}_v1.json`,
  `results/milestone2/run_log.csv` (both models merged), `baseline_grid_m2_{MODEL_TAG}_v1.png`
- **Rerunning:** existing images are skipped, so an interrupted run resumes where it stopped.
  To regenerate a model, move its `images/m2_{MODEL_TAG}_v1/` folder and log out of the way first.

#### `05_evaluate.ipynb`
Scores every image with CLIP (`openai/clip-vit-base-patch32` @ `3d74acf9a2`, score = 100 × cosine of
image and caption), merges the logs, summarises each setting, draws the plots and builds the
submission strips. It does not generate images.

- **Writes:** `clip_scores_m2_{MODEL_TAG}_v1.csv`, `clip_scorer.json`, `summary_by_setting.csv`,
  `plot_clip_vs_guidance.png`, `plot_quality_time_vs_steps.png`, `submission/`, and the step
  review sheets in `images/review/` (Drive only), all under `results/milestone2/`
- **Rerunning:** images that already have a score are skipped (`RESCORE = False`). The strip
  prompts and models are set in the config cell; section 8 clears `submission/` before rebuilding it.

### Checks each run should pass

| Notebook | Check | Expected |
|---|---|---|
| 01 | Manifest | identical to the saved one |
| 02 | Shapes | `[2000, 77, 768]` (SD 1.5) / `[2000, 77, 1024]` (SD 2.1) |
| 02 | Truncation | 0 / 2000 captions over 77 tokens |
| 02 | Saved embeddings | match the fresh run (`allclose = True`) |
| 02 | End-to-end | images from saved embeddings are visually identical to images from the prompt string |
| 03 | Prompt and FID sets | unchanged from the saved files |
| 04 | Completeness | `planned 210 \| missing images 0 \| missing log rows 0 \| setting mismatches 0` |
| 04 | Baseline vs Milestone 1 | baseline images repeat the Milestone 1 samples (mean pixel difference a few units of 255) |
| 05 | CLIP scorer | each real image matches its own caption better than another's (10 / 10) |
| 05 | Scoring | 420 rows, none without a score |

Regenerating all 210 SD 2.1 images in a second session gave pixel-identical images and identical
CLIP scores; only the time per image changed (about 9%), so timings are compared within a session only.

## Results

### Milestone 1
Stored in `results/milestone1/`:

- `example_pairs.png`: six sample image–caption pairs from the manifest
- `grid_sd15_caption.png`, `grid_sd21_caption.png`: real vs generated images for the five test prompts
  (image IDs 1761, 97924, 25181, 3661, 5600: counting, people and action, spatial relation, simple
  scene, two separate objects)
- `e2e_diffmap_sd15_caption.png`, `e2e_diffmap_sd21_caption.png`: where generation from saved embeddings
  differs from generation from the prompt string
- `embedding_config_*_caption.json`: checkpoint, revision, library versions, encoder and token statistics
  for each run
- `token_stats_*_caption.csv`: per-caption token count and truncation flag

Individual generated samples (`sample_*.png`) are kept on Drive only; each also appears in a grid.

### Milestone 2
Stored in `results/milestone2/`:

- `Milestone2_Report.pdf`: the submitted report (setup, method, results, early samples, generation log)
- `run_log.csv`: generation log, 420 rows, one per image (model, scheduler, steps, guidance, seed,
  file name, generation time, peak GPU memory, GPU, CLIP score); per-model logs in `run_log_m2_*_v1.csv`
- `summary_by_setting.csv`: the log aggregated by model and setting (mean CLIP score, paired difference
  from the baseline with its standard error, mean time, peak memory)
- `config_m2_*_v1.json`, `clip_scorer.json`, `clip_scores_m2_*_v1.csv`: run settings, scorer and per-image scores
- `plot_clip_vs_guidance.png`, `plot_quality_time_vs_steps.png`: the two sweeps
- `baseline_grid_m2_*_v1.png`, `prompt_set_m2.png`: baseline images per model and the real images
- `submission/`: the two early-sample strips (SD 1.5 guidance sweep, prompt 3661; SD 2.1 scheduler
  sweep at 10 steps, prompt 214200) and the 10 images in them

Settings fixed for Milestone 3: DDIM, 30 steps, guidance 7.5, seed 42 for both models.

## Data and large files

Images and embeddings are not stored in this repo. They live on Google Drive and can be regenerated
by running the notebooks.

| File | Size | Location |
|---|---|---|
| `data/processed_512/` | 2,000 PNGs, 512×512 | <https://drive.google.com/drive/folders/1xVKOTaHNhwKxA3evtWOuCRykhnaNRFEL?usp=drive_link> |
| `embeddings/text_embeddings_sd15_caption.pt` | 2000×77×768, fp16, ~236 MB | <https://drive.google.com/file/d/1fI-lKd30xSl6aw_Qo8Gc0ruHwRw4tbXJ/view?usp=drive_link> |
| `embeddings/text_embeddings_sd21_caption.pt` | 2000×77×1024, fp16, ~315 MB | <https://drive.google.com/file/d/1elwu75heCKvm4fYQqzOIxI07avkxgeFc/view?usp=drive_link> |
| `results/milestone2/images/` | 420 PNGs, 512×512 (210 per model) | <https://drive.google.com/drive/folders/10HhBUZYjT8WpyO17np3HQuW2jyM_O8Vo?usp=drive_link> |

Embeddings are stored in the same row order as `manifest.csv` and include the `image_id` list, the
empty-prompt embedding for classifier-free guidance, the checkpoint and the revision.

## Dataset
COCO 2017 validation captions and images ([cocodataset.org](https://cocodataset.org)), used under the
COCO terms of use.
