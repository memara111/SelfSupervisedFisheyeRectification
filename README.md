# SelfSupervisedFisheyeRectification
An official pytorch implementation of the paper "Self-Supervised Fisheye Image Rectification by Reconstructing Coordinate Relations"

![results.png](https://raw.githubusercontent.com/MasakiHosono/SelfSupervisedFisheyeRectification/main/statics/results.png?token=AE3JGTNHNWMGTDYSXFK7PHDAOVLIQ "results.png")

Our network is based on single parameter division model, architecture is shown below.

![net_arch_full.png](https://raw.githubusercontent.com/MasakiHosono/SelfSupervisedFisheyeRectification/main/statics/net_arch_full.png?token=AE3JGTIG7EIOR2BT5B2DDMDAOVLLC "net_arch_full.png")

### Quick start
1. Clone the project.
   ```
   git clone https://github.com/MasakiHosono/SelfSupervisedFisheyeRectification.git
   ```

1. Install the dependencies.
   ```
   python3 -m venv .venv && . .venv/bin/activate
   pip3 install -r requirements.txt
   ```
   `torch`/`torchvision` need the wheel index matching your CUDA setup, e.g.
   `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu` for a CPU only box.
   `matplotlib` is optional: it is used for the loss curve only.

1. Prepare dataset
   Download [cityscapes dataset](https://www.cityscapes-dataset.com) and unpack
   `leftImg8bit_trainvaltest.zip` under `data/cityscapes/`. Then run the following script.
   ```
   python3 tools/prepare_cityscapes.py
   ```
   It reshapes the archive layout and writes `train.lst`, `val.lst` and `test.lst`.
   Images are moved by default; use `--mode copy` to keep the download intact,
   `--mode link` to hardlink, `--mode dry_run` to only see what would happen, and
   `--data_root` if the dataset lives somewhere else.

1. Training.
   ```
   python3 src/train.py cfg/cityscapes.yml
   ```
   Checkpoints, `losses.png` and `losses.json` land in `outputs/<dataset name>/`.
   Add `--resume` to continue from an existing checkpoint.

1. Testing.
   ```
   python3 src/test.py cfg/cityscapes.yml
   ```
   Reports the distortion estimation error (MAE / median / max) together with the
   residual of the coordinate relation the network is trained on, and writes
   `results/metrics.csv` plus `*_org.jpg` / `*_dis.jpg` / `*_rec.jpg` triplets.

1. Rectifying your own images.
   ```
   python3 src/predict.py cfg/cityscapes.yml -i path/to/fisheye_images -ow 1280 -oh 720
   ```
   Outputs go to `path/to/fisheye_images/rectified`; `--save-distortions` also writes the
   estimated parameter per image. Add `--no_crop` for images that were not rendered by
   `FisheyeEffector` (i.e. real fisheye frames, which were never cropped and magnified).

1. Sanity checks.
   ```
   python3 tests/run_tests.py
   ```
   No dataset required: it verifies the geometry, the loss, the label/loss agreement and
   a full train/test round trip on synthetic images.

### Kaggle notebook

`notebook534800d0d8.ipynb` runs this quick start end to end on Kaggle using **NYU Depth V2**
instead of Cityscapes (no account, no 2 GB download): it writes `data/nyu2/{train,val,test}.lst`
and a matching `cfg/nyu2.yml`, trains, tests, and finishes with `src/predict.py` on a photo.
It pins the clone to the branch that carries the fixes above and keeps a patch cell that only
applies to a pre-fix clone.

### Troubleshooting
* **Every prediction sticks at `+/-MODEL.MAX_DISTORTION`.** The head is squashed into that
  interval, so a too-high learning rate (or `MODEL.FREEZE_VGG: true` with the random VGG
  features) saturates it. Lower `TRAIN.LEARNING_RATE`, or set `MODEL.VGG_PRETRAINED: true`.
* **`TRAIN.CURRICULUM.MAX_DISTORTION` must stay below `MODEL.MAX_DISTORTION`.** Otherwise
  the hardest targets sit exactly at the bound, which a squashed output can only approach
  asymptotically. `train.py` warns and clamps when it sees this.
* **`outputs/<dataset>/checkpoint.pth.tar` is where `test.py` and `predict.py` look**, and
  `MODEL.MAX_DISTORTION` is recorded in it; the two scripts refuse a mismatch rather than
  silently rescaling every prediction.
* **`TEST.OUTPUT_SIZE` need not equal `DATASET.HEIGHT/WIDTH`**: `test.py` resamples the image
  to it before rectifying, so the saved `_rec.jpg` can be an upsized deliverable. Feeding a
  differently sized image to `FisheyeEffector` yourself works too (`applyToArray` resamples),
  but with nearest neighbour, so resize first if quality matters.

### Conventions
* The model parameter is the single coefficient `k` of the division model
  `original = distorted / (1 - k * |distorted|^2)`, `k` in `[-1, 1]` (`k > 0` is
  barrel/fisheye, `k < 0` is pincushion). Both directions and the key point labels are
  derived from it in one place, `src/core/division_model.py`.
* Coordinates are isotropic normalised units (`x` in `(-1, 1)`, `y` scaled by `H/W`),
  so radii are aspect correct for non-square frames.
* The loss is *self-supervised*: it never sees the ground truth `k`, only the relations
  (vertical alignment + equal spacing) that the three key points must satisfy after
  rectification.

### Citation
```
@inproceedings{hosono2021self,
  title={Self-Supervised Deep Fisheye Image Rectification Approach using Coordinate Relations},
  author={Hosono, Masaki and Simo-Serra, Edgar and Sonoda, Tomonari},
  booktitle={2021 17th International Conference on Machine Vision and Applications (MVA)},
  pages={1--5},
  year={2021},
  organization={IEEE}
}
```
