### Training LightGlue
```bash
python -m gluefactory.train sp+lg_homography \
 --conf gluefactory/configs/superpoint+lightglue_homography.yaml
```
### Fine-tuning LightGlue on MegaDepth dataset:
```bash
python -m gluefactory.train sp+lg_megadepth \
 --conf gluefactory/configs/superpoint+lightglue_megadepth.yaml \
    train.load_experiment=sp+lg_homography
```

### Dataset preparation
- Subset smaller 100 times for an hour training
- 