# Stereo Disparity and Depth Estimation from Rectified Stereo Images

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)]()
[![License](https://img.shields.io/badge/License-Educational-orange.svg)]()

A Computer Vision project for **CO543/CO5430** that estimates disparity and relative depth from rectified stereo image pairs using classical stereo matching algorithms.

---

## 📌 Project Information

**Course:** CO543/CO5430 Computer Vision

**Project ID:** P12

**Group:** G03

### Team Members

- E/23/127 – H.M.K.I. Herath
- E/23/188 – K.M.M.Y. Kumarasinge
- E/23/343 – S.B.N.S. Samarawickrama
- E/23/347 – S.D.M.P. Sandanayake

---

# Project Overview

Stereo vision estimates the depth of objects by comparing two images captured from slightly different viewpoints.

This project implements two classical stereo matching algorithms:

- Stereo Block Matching (StereoBM)
- Semi-Global Block Matching (StereoSGBM)

The generated disparity maps are evaluated using the Middlebury Stereo 2021 dataset and compared against the provided ground-truth disparity maps.

---

# Objectives

## Minimum Goal

- Load rectified stereo image pairs
- Generate disparity maps using StereoBM
- Evaluate using RMSE and Bad Pixel Rate

## Expected Goal

- Implement StereoSGBM
- Tune parameters
- Compare StereoBM and StereoSGBM

## Stretch Goal

- Post-processing
- Parameter ablation study
- Relative depth visualization
- Optional KITTI evaluation

---

# Dataset

This project uses the **Middlebury Stereo 2021 Dataset**.

The dataset is **not included** in this repository due to its size.

Download it from:

https://vision.middlebury.edu/stereo/data/scenes2021/

After downloading, organize it as:

```text
data/
└── raw/
    └── <Scene Name>/
        ├── im0.png
        ├── im1.png
        ├── disp0.pfm
        └── calib.txt
```

---

# Repository Structure

```text
stereo-depth-estimation/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── outputs/
│
├── scripts/
│   └── run_bm.py
│
├── src/
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── stereo_bm.py
│   ├── stereo_sgbm.py
│   ├── evaluation.py
│   └── visualization.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/<username>/e23-co5430-Stereo-disparity_depth-estimation.git

cd e23-co5430-Stereo-disparity_depth-estimation
```

Create a virtual environment

```bash
python3 -m venv venv
```

Activate it

Linux/macOS

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running

Run the StereoBM pipeline

```bash
python -m scripts.run_bm
```

Future scripts

```bash
python -m scripts.run_sgbm
```

---

# Project Workflow

```text
Download Dataset
        │
        ▼
Load Stereo Images
        │
        ▼
Preprocessing
        │
        ▼
StereoBM Baseline
        │
        ▼
Evaluation
        │
        ▼
StereoSGBM
        │
        ▼
Parameter Tuning
        │
        ▼
Post Processing
        │
        ▼
Final Evaluation
```

---

# Evaluation Metrics

The project compares the generated disparity maps against the provided ground truth using:

- Root Mean Square Error (RMSE)
- Bad Pixel Rate
- Qualitative disparity map comparison
- Runtime comparison

---

# Current Progress

- [x] Repository setup
- [x] Python virtual environment
- [x] Initial dataset loader
- [ ] Dataset integration
- [ ] Preprocessing
- [ ] StereoBM implementation
- [ ] StereoSGBM implementation
- [ ] Evaluation metrics
- [ ] Parameter tuning
- [ ] Final report

---

# Technologies

- Python
- OpenCV
- NumPy
- Matplotlib
- Pandas

---

# References

1. Middlebury Stereo Dataset (2021)

   https://vision.middlebury.edu/stereo/data/scenes2021/

2. OpenCV Documentation

   https://docs.opencv.org/

---

# License

This repository is developed solely for academic purposes as part of the **CO543/CO5430 Computer Vision** course at the **University of Peradeniya**.
