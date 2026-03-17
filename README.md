# NTIRE 2026 Event-Based Image Deblurring Challenge

## Team Information

**Team Name:** KLETech-CEVI

---

## 1. Introduction

This repository presents our solution for the **NTIRE 2026 Event-Based Image Deblurring Challenge**.
The objective of this challenge is to recover sharp images from motion-blurred inputs by leveraging both conventional RGB frames and event data.

Our approach is based on a **NAFNet architecture**, adapted to incorporate event information for improved deblurring performance.

---

## 2. Method Overview

We adopt a hybrid framework combining:

* Image-based restoration using **NAFNet**
* Event-guided enhancement using voxelized event representations

### Key Features

* Encoder–Decoder architecture
* Residual learning blocks
* Event-image fusion mechanism
* Multi-scale feature extraction

The proposed model effectively utilizes complementary information from event streams to improve restoration quality under challenging motion conditions.

---

## 3. Repository Structure

```id="finalstruct1"
.
├── basicsr/                # Core training and testing framework
├── datasets/               # Dataset configuration files
├── options/                # YAML config files (train/test)
├── scripts/                # Data preparation & utilities
├── figs/                   # Figures and visualizations
├── requirements.txt        # Python dependencies
├── setup.py                # Package installation script
├── setup.cfg               # Package configuration
├── .gitignore              # Git ignore rules
├── LICENSE                 # License information
├── VERSION                 # Version file
├── README.md               # Project documentation
```

---

## 4. Environment Setup

### Step 1: Create Conda Environment

```bash id="finalenv1"
conda create -n ntire python=3.8 -y
conda activate ntire
```

### Step 2: Install Dependencies

```bash id="finalenv2"
pip install -r requirements.txt
```

---

## 5. Dataset Preparation

Ensure the dataset follows the structure:

```id="finaldata1"
HighREV_test/
├── blur/
├── event/
├── voxel/
```

Update dataset paths in:

```id="finaldata2"
options/test/HighREV/
options/train/HighREV/
```

---

## 6. Training

To train the model, run:

```bash id="finaltrain"
python basicsr/train.py -opt options/train/HighREV/NAFNet_ImageOnly.yml
```

---

## 7. Testing / Inference

To perform testing or generate submission results:

```bash id="finaltest"
python basicsr/test.py -opt options/test/HighREV/NAFNet_200k_test.yml
```

---

## 8. Validation

Validation is performed during training based on the configuration file.

**Metrics used:**

* PSNR (Peak Signal-to-Noise Ratio)
* SSIM (Structural Similarity Index)

---

## 9. Pretrained Models (Weights)

The pretrained model weights can be downloaded from:

👉 https://drive.google.com/drive/folders/1v_nDto_vPvXVopttu138RYCetn-1IC1G?usp=drive_link

After downloading, place the weights in:

```id="finalweights"
experiments/NAFNet_ImageOnly/models/
```

---

## 10. Results

The generated results for the NTIRE 2026 submission can be accessed here:

👉 https://drive.google.com/drive/folders/1OkeP6q4BmpUEzVuLJjXPr3Qw2PekVyrO?usp=drive_link

These correspond to the restored output images obtained on the test dataset.

---

## 11. Important Notes

* Ensure dataset paths are correctly configured before execution
* GPU is recommended for faster training and inference
* Tested on CUDA-enabled systems
* Large files (logs, checkpoints) are excluded via `.gitignore`

---

## 12. Reproducibility

Steps to reproduce results:

1. Setup environment
2. Prepare dataset
3. Download pretrained weights
4. Place weights in `experiments/NAFNet_ImageOnly/models/`
5. Run testing script

---

## 13. Acknowledgements

This work is based on:

* BasicSR framework
* NAFNet architecture

We thank the NTIRE organizers for providing the dataset and evaluation platform.

---

## 14. License

This project follows the license provided in the repository.

---
