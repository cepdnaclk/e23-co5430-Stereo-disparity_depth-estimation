---
title: Stereo Disparity and Depth Estimation
emoji: 👁️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
hardware: cpu-basic
---

# Stereo Disparity and Depth Estimation from Rectified Stereo Images

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]()
[![Gradio](https://img.shields.io/badge/Gradio-5.49.1-orange.svg)]()
[![License](https://img.shields.io/badge/License-Educational-lightgrey.svg)]()

A comprehensive Computer Vision project for **CO543 / CO5430** that estimates disparity and relative depth from rectified stereo image pairs using both classical stereo matching algorithms (**StereoBM**, **StereoSGBM**) and deep learning (**RAFT-Stereo** with few-shot domain adaptation).

---

## 📌 Project Information

* **Course:** CO543 / CO5430 Computer Vision
* **Institution:** Department of Computer Engineering, University of Peradeniya
* **Project ID:** P12
* **Group:** G03

### Team Members
- **E/23/127** – H.M.K.I. Herath
- **E/23/188** – K.M.M.Y. Kumarasinge
- **E/23/343** – S.B.N.S. Samarawickrama
- **E/23/347** – S.D.M.P. Sandanayake

---

## 🌟 Features & Web Application

This repository includes a feature-rich, interactive **Gradio Web Application** (`app.py`):
1. **Interactive Disparity Estimation**: Upload any stereo pair or choose from bundled examples. Includes real-time parameter tuning for SGBM (`blockSize`, `numDisparities`, `uniquenessRatio`).
2. **Interactive Image Slider**: Compare Left Camera View vs. Colorized Disparity Map with a split comparison slider.
3. **4-Way Model Comparison**: Run StereoBM, StereoSGBM, Pre-trained RAFT, and Fine-tuned RAFT side-by-side.
4. **Ground Truth Benchmark**: Upload a Middlebury `.pfm` ground-truth file to compute **RMSE**, **Absolute Relative Error (AbsRel)**, and **Bad-3-Pixel Rate (D1 Metric)**, alongside an error heatmap.

---

## 🚀 Running the Web Application

### Option A: In Google Colab (Recommended for GPU)
Run these commands in a Colab notebook cell:
```python
# 1. Clone this repository
!git clone https://github.com/cepdnaclk/e23-co5430-Stereo-disparity_depth-estimation.git
%cd e23-co5430-Stereo-disparity_depth-estimation

# 2. Install dependencies
!pip install -r requirements.txt

# 3. Launch Gradio (generates a public https://xxxx.gradio.live link)
!python app.py
```

### Option B: Deploy to Hugging Face Spaces (24/7 Permanent URL)
1. Create a new Space on [Hugging Face](https://huggingface.co/new-space):
   * Select **Gradio** as the Space SDK.
2. Add your Hugging Face Space as a git remote:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space main
   ```
   Hugging Face will automatically build and host the interactive web app!

Keep `sdk_version` above and the Gradio pin in `requirements.txt` at the same
version. Gradio 5.49.1 provides the built-in `ImageSlider` and handles boolean
JSON schemas that caused the Gradio 4.44.0 startup crash (`TypeError: argument
of type 'bool' is not iterable`). The subsequent localhost/share-link error
was a consequence of the failed HTTP response; Spaces does not need `share=True`.
For an existing Space, deploy both updated files and rebuild it. For a local
installation, rerun `python -m pip install -r requirements.txt`.

### Option C: Run Locally
```bash
# Clone the repository
git clone https://github.com/cepdnaclk/e23-co5430-Stereo-disparity_depth-estimation.git
cd e23-co5430-Stereo-disparity_depth-estimation

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch web app
python app.py
```
Open `http://127.0.0.1:7860` in your web browser.

---

## 📁 Repository Structure

```text
e23-co5430-Stereo-disparity_depth-estimation/
├── app.py                     # Interactive Gradio Web Application (HF Spaces & Colab ready)
├── requirements.txt           # Python package requirements
├── packages.txt               # Debian system packages for Hugging Face Spaces
├── README.md                  # Project documentation & Hugging Face metadata
│
├── src/                       # Modular source code
│   ├── dataset.py             # Middlebury PFM loader, image pair loader, calibration parser
│   ├── preprocessing.py       # Spatial cropping (384x512) and padding utilities
│   ├── stereo_bm.py           # Classical StereoBM implementation
│   ├── stereo_sgbm.py         # Classical StereoSGBM implementation with dynamic P1/P2
│   ├── raft_wrapper.py        # RAFT-Stereo deep learning inference wrapper (CPU/CUDA)
│   ├── evaluation.py          # Benchmark metrics: RMSE, AbsRel, Bad-3-Pixel Rate (D1)
│   ├── visualization.py       # Plasma/turbo colormapping and error heatmaps
│   └── temp ipynb/            # Development notebooks from Milestones 1–3
│
├── examples/                  # Bundled sample stereo pairs for 1-click web testing
│   ├── sample_left.png
│   └── sample_right.png
│
└── Documents/                 # Academic milestone deliverables
    ├── Project Proposal.pdf   # Approved project proposal
    ├── M2.pptx                # Milestone 2 presentation slides
    └── M3.pptx                # Milestone 3 presentation slides
```

---

## 🔬 Benchmark Results on Middlebury 2021 (`artroom1`)

| Method | Type | RMSE (px) | AbsRel | Bad-3-Pixel Rate (D1) | Runtime |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **StereoBM** | Classical Sliding Window | ~42.1 px | ~0.29 | ~35.4% | **~15 ms** |
| **StereoSGBM** | Classical Semi-Global | **38.64 px** | **0.2378** | **30.23%** | **~65 ms** |
| **RAFT-Stereo (Pre-trained)** | Deep Learning (SceneFlow) | **3.98 px** | **0.0091** | **4.46%** | **~280 ms** |
| **RAFT-Stereo (Fine-Tuned)** | Deep Learning (Middlebury 24-crop) | Domain-adapted | Domain-adapted | Domain-adapted | **~280 ms** |

---

## 🎯 Current Progress

- [x] Repository setup & modular structure
- [x] Middlebury 2021 `.pfm` ground-truth reader and calibration parser (`src/dataset.py`)
- [x] Preprocessing with memory-safe spatial crops (`src/preprocessing.py`)
- [x] StereoBM baseline implementation (`src/stereo_bm.py`)
- [x] StereoSGBM implementation with parameter ablation (`src/stereo_sgbm.py`)
- [x] Benchmark quantitative evaluation metrics (`src/evaluation.py`)
- [x] RAFT-Stereo deep learning integration (`src/raft_wrapper.py`)
- [x] Interactive Gradio Web Application with split image slider (`app.py`)
- [x] Bundled test examples for 1-click evaluation (`examples/`)
- [x] Hugging Face Spaces & Google Colab compatibility

---

## 📜 References

1. G. Pan, T. Sun, T. Weed, and D. Scharstein, *"2021 Mobile Stereo Datasets with Ground Truth,"* Middlebury Stereo Datasets, 2021.
2. H. Hirschmuller, *"Stereo Processing by Semiglobal Matching and Mutual Information,"* IEEE TPAMI, 2008.
3. L. Lipson, Z. Teed, and J. Deng, *"RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching,"* 3DV, 2021.
4. OpenCV Documentation, OpenCV 4.x, 2025.

---

## 📄 License

This repository is developed for academic purposes as part of the **CO543 / CO5430 Computer Vision** course at the **University of Peradeniya**.
