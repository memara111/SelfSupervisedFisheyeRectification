# Self-Supervised Fisheye Rectification on NYU Depth V2 - Kaggle Notebooks

This repository contains Jupyter notebooks for training and inference of a self-supervised fisheye image rectification model using the NYU Depth V2 dataset on Kaggle.

## Overview

Based on the paper "Self-Supervised Fisheye Image Rectification by Reconstructing Coordinate Relations" ([original repo](https://github.com/memara111/SelfSupervisedFisheyeRectification)).

## Notebooks

### 1. Training Notebook: `train_fisheye_nyu.ipynb`

This notebook handles the complete training pipeline:
- **Dataset Preparation**: Automatically loads NYU Depth V2 dataset from Kaggle input
- **Train/Val Split**: Creates 90/10 split for training and validation
- **Model Training**: Trains the Parameters Estimation Module (PEM)
- **Checkpoint Saving**: Saves model state, optimizer state, and losses after each epoch
- **Loss Visualization**: Plots training and validation loss curves in real-time
- **Resume Support**: Can resume from the last saved checkpoint if interrupted

**Key Features:**
- ✅ Automatic dataset preparation from `/kaggle/input/datasets/soumikrakshit/nyu-depth-v2/nyu_data/data/nyu2_train`
- ✅ Stop and continue from last epoch (checkpoint-based)
- ✅ Real-time loss plotting (train & val)
- ✅ Curriculum learning support
- ✅ Progress bars with tqdm

### 2. Inference Notebook: `inference_undistort.ipynb`

This notebook performs inference on distorted images:
- **Load Trained Model**: Loads checkpoint from training
- **Predict Distortion**: Estimates distortion parameter for each image
- **Undistort Images**: Applies inverse distortion to rectify images
- **Visualization**: Shows before/after comparison plots
- **Save Results**: Saves undistorted images and comparison plots

**Key Features:**
- ✅ Batch processing of multiple images
- ✅ Side-by-side comparison plots
- ✅ Manual distortion testing (experiment with different values)
- ✅ Automatic result saving

## Usage on Kaggle

### Step 1: Setup Kaggle Notebook

1. Create a new Kaggle Notebook
2. Add the NYU Depth V2 dataset as input:
   - Dataset: `soumikrakshit/nyu-depth-v2`
   - Path will be: `/kaggle/input/datasets/soumikrakshit/nyu-depth-v2/nyu_data/data/nyu2_train`

### Step 2: Training

1. Upload `train_fisheye_nyu.ipynb` to Kaggle
2. Run all cells
3. The notebook will:
   - Clone the repository
   - Prepare the dataset
   - Train the model
   - Save checkpoints to `/kaggle/working/outputs/nyu_depth_v2/`

**To Resume Training:**
- If the notebook stops, simply re-run it
- It will automatically detect the checkpoint and resume from the last epoch

### Step 3: Inference

1. After training completes, upload `inference_undistort.ipynb`
2. Update the `INPUT_IMAGE_PATH` variable to point to your images
3. Run all cells
4. Undistorted images will be saved to `/kaggle/working/undistorted_images/`

## Configuration

Edit the config in the training notebook to adjust:
- `MAX_EPOCH`: Total number of epochs (default: 50)
- `BATCH_SIZE`: Batch size (default: 8)
- `LEARNING_RATE`: Learning rate (default: 0.0001)
- `CURRICULUM.SWITCH_EPOCH`: When to switch distortion patterns (default: 10)
- `DATASET.HEIGHT/WIDTH`: Input image dimensions (default: 256x512)

## Output Files

### Training Outputs (`/kaggle/working/outputs/nyu_depth_v2/`)
- `checkpoint.pth.tar`: Model checkpoint with epoch, losses, state_dict, optimizer
- `losses.png`: Training curve (updated each epoch)
- `final_losses.png`: Final loss plot

### Inference Outputs (`/kaggle/working/undistorted_images/`)
- `undistorted_*.jpg/png`: Rectified images
- `comparison_*.png`: Before/after comparison plots
- `manual_distortion_comparison.png`: Manual distortion experiments

## Model Architecture

The model consists of:
1. **Encoder**: Convolutional layers for feature extraction
2. **Decoder**: Transposed convolutions for upsampling
3. **VGG11 Backbone**: Pretrained feature extractor
4. **Regression Head**: Predicts single distortion parameter

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{hosono2021self,
  title={Self-Supervised Deep Fisheye Image Rectification Approach using Coordinate Relations},
  author={Hosono, Masaki and Simo-Serra, Edgar and Sonoda, Tomonari},
  booktitle={2021 17th International Conference on Machine Vision and Applications (MVA)},
  pages={1--5},
  year={2021},
  organization={IEEE}
}
```

## Requirements

The notebooks will automatically install:
- PyTorch (CUDA 11.8)
- torchvision
- opencv-python-headless
- pillow
- scipy
- pyyaml
- matplotlib
- tqdm
- scikit-learn

## Troubleshooting

### Out of Memory
- Reduce `BATCH_SIZE` in the config
- Reduce image dimensions (`HEIGHT`, `WIDTH`)

### Slow Training
- Ensure GPU is enabled in Kaggle settings
- Check that CUDA is available (first cell output)

### Checkpoint Not Loading
- Verify the checkpoint file exists at `/kaggle/working/outputs/nyu_depth_v2/checkpoint.pth.tar`
- Make sure you're running in the same notebook session or have saved the output

## License

This project follows the license of the original repository.
