# Data

- `manifest.csv`: 2,000 image–caption pairs with columns `image_id`, `file_path`, `caption`. `file_path` is relative to the data folder on Drive.
- `preprocessing_config.json`: the exact rules used (dataset, seed, caption rule, minimum words, image size, dropped counts).
- `example_pairs.png`: six randomly sampled image caption pairs from the processed subset.

Source: COCO Captions, val2017 split. Selection: shuffled with seed 42, first caption per image, captions under 3 words and duplicates dropped. Images are resized and center-cropped to 512×512 RGB.


