# Text free speech to image Training with SpokenCOCO Dataset

## Data

Three things are needed:

1. **SpokenCOCO**: spoken captions (wav files) and JSON metadata (`SpokenCOCO.tar.gz` from the authors, MIT).
2. **MS-COCO 2014 images** (`train2014`, `val2014`): downloaded separately, not included in SpokenCOCO.
3. **Karpathy split** (optional): for the standard train/val/test division.

## TODO

- [ ] Verify sample rate and path existence on a subset of files
- [ ] Choose `max_sec` from the audio length distribution
- [ ] Apply the Karpathy split (filter by image ID)
- [ ] Match image normalization to the chosen image encoder
- [ ] Check input requirements of the chosen audio encoder
- [ ] Config wandb
- [ ] Implement Visual Encoder (ResNet)
- [ ] Implement Audio Encoder (VG-HuBERT)
- [ ] Training with wandb
- [ ] Experiments
- [ ] Visalisation
