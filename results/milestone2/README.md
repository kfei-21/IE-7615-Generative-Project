### Milestone 2 – Experimental Results and Evaluation

Milestone 2 evaluates Stable Diffusion 1.5 and Stable Diffusion 2.1 through controlled experiments involving guidance scale, inference steps, and scheduler selection. The generated images are evaluated using CLIP to measure image caption alignment, alongside generation time and GPU memory consumption.

The results directory contains the generated images, individual evaluation scores, consolidated experiment logs, configuration files, summary statistics, and visualizations.

#### 1. CLIP Scoring Results

| File | Description |
|---|---|
| `clip_scorer.json` | Stores the CLIP model identifier, checkpoint revision, scoring formula, and mean CLIP score obtained from the original COCO images. This ensures consistency of the scoring model across evaluation sessions. |
| `clip_scores_m2_sd15_v1.csv` | Contains the individual CLIP scores for all 210 images generated using Stable Diffusion 1.5. Each record includes the generated image filename and its corresponding image–caption similarity score. |
| `clip_scores_m2_sd21_v1.csv` | Contains the individual CLIP scores for all 210 images generated using Stable Diffusion 2.1. The same scoring methodology is applied to both models to enable consistent comparison. |
