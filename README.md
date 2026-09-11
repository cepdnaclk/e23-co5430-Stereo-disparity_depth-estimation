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

# Stereo Disparity & Relative Depth Estimation: From Classical Matching to Few-Shot Domain-Adapted RAFT-Stereo

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces%20Live%20Demo-blue?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/malith26/stereo-disparity-depth-estimation)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/cepdnaclk/e23-co5430-Stereo-disparity_depth-estimation)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=flat-square&logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?style=flat-square&logo=pytorch&logoColor=white)]()
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8.svg?style=flat-square&logo=opencv&logoColor=white)]()
[![Gradio](https://img.shields.io/badge/Gradio-5.49.1-FF7C00.svg?style=flat-square&logo=gradio&logoColor=white)]()

---

> ### 🌐 **Live Cloud Demonstration**
> Access the interactive web application deployed on Hugging Face Spaces:  
> **👉 [https://huggingface.co/spaces/malith26/stereo-disparity-depth-estimation](https://huggingface.co/spaces/malith26/stereo-disparity-depth-estimation)**  
> *Equipped with real-time parameter tuning, 4-way model comparison, ground-truth error heatmap analysis, and dynamic ZeroGPU acceleration.*

---

## 📌 Academic Project Overview

* **Course:** CO543 / CO5430 Computer Vision
* **Institution:** Department of Computer Engineering, Faculty of Engineering, University of Peradeniya, Sri Lanka
* **Group:** G03 | **Project ID:** P12
* **Topic:** Dense Stereo Correspondence Matching, Disparity Computation, and Relative Depth Estimation

### Research & Engineering Team
* **E/23/127** — H.M.K.I. Herath *(Dataset curation, 32-bit floating-point PFM parser, valid masking, StereoBM tuning)*
* **E/23/188** — K.M.M.Y. Kumarasinge *(Classical matching formulation, dynamic P1/P2 penalty optimization, noise analysis)*
* **E/23/343** — S.B.N.S. Samarawickrama *(RAFT-Stereo architecture wrapping, PyTorch pipeline, guarded few-shot domain adaptation)*
* **E/23/347** — S.D.M.P. Sandanayake *(Production Gradio 5 web UI, Hugging Face Spaces deployment, ZeroGPU integration, benchmarking)*

---

## 📸 Visual Results & Model Comparisons

### Qualitative Disparity & Error Heatmap Comparison
Visual benchmarking across classical sliding-window matching, semi-global energy minimization, and deep recurrent field transforms evaluated on Middlebury 2021 (*artroom1*):

![Qualitative Comparison across StereoBM, StereoSGBM, RAFT Pretrained, and RAFT Fine-Tuned](report_figures/fig5_qualitative_comparison.png)

---

## 📐 Mathematical Formulation & Epipolar Geometry

Stereo depth estimation recovers 3D spatial scene structure from a pair of rectified, horizontally displaced 2D images ($I_L$ and $I_R$). Because the cameras are rectified along parallel optical axes with horizontal baseline $B$ and focal length $f$, corresponding scene points lie on the same horizontal scanline ($y_L = y_R$):

$$\text{Horizontal Disparity: } d(x, y) = x_L - x_R \quad (d \ge 0)$$

$$\text{Euclidean Depth: } Z(x, y) = \frac{f \cdot B}{d(x, y)}$$

When intrinsic calibration ($f$) and extrinsic baseline ($B$) are uncalibrated, relative inverse depth is directly proportional to disparity ($Z_{rel} \propto 1/d$).

![Epipolar Geometry and Disparity-to-Depth Formulation](report_figures/fig1_epipolar_geometry.png)

---

## ⚙️ System Pipeline & Architecture

The system supports both classical computer vision matching algorithms and deep learning architectures with full benchmarking pipelines:

![End-to-End System Pipeline and Dataflow Architecture](report_figures/fig2_system_pipeline.png)

### Algorithmic Breakdown:
1. **StereoBM (Block Matching)**: Fixed square window ($K \times K$) minimizing the Sum of Absolute Differences (SAD). Fast ($\sim 15\text{ ms}$ on CPU) but vulnerable to textureless regions and edge fattening.
2. **StereoSGBM (Semi-Global Block Matching)**: Minimizes a 2D Markov Random Field energy functional across 8 dynamic programming paths with pairwise smoothness penalties $P_1$ (slanted surfaces) and $P_2$ (depth discontinuities):
   $$E(D) = \sum_p C(p, D_p) + \sum_{q \in N_p} P_1 \cdot \mathbf{1}(|D_p - D_q| = 1) + \sum_{q \in N_p} P_2 \cdot \mathbf{1}(|D_p - D_q| > 1)$$
3. **RAFT-Stereo (Lipson et al., 3DV 2021)**:
   - **Dual Encoders**: Feature network (`fnet`) extracts multi-scale representations at $1/4$ resolution; Context network (`cnet`) initializes recurrent hidden states.
   - **1D Epipolar Correlation Pyramid**: All-pairs dot-product correlation along horizontal epipolar lines with 4 pooling radii ($r \in \{1, 2, 4, 8\}$).
   - **Recurrent ConvGRU**: 32 recurrent updates predicting residual disparity updates ($d_{t+1} = d_t + \Delta d_t$).
   - **Guarded Few-Shot Domain Adaptation**: Backbone freezing (`fnet`/`cnet`), sequence-discounted Smooth L1 loss ($\gamma = 0.9$), valid-pixel ground-truth masking ($0 < d \le 250\text{ px}$), and conservative learning rates ($\eta = 10^{-6}$).

![RAFT-Stereo Recurrent Architecture and Domain Adaptation Guardrails](report_figures/fig3_raft_architecture.png)

---

## 🔬 Benchmark Results on Middlebury 2021 (`artroom1`)

Quantitative evaluation performed on the high-resolution Middlebury 2021 benchmark (*artroom1* scene with 32-bit floating-point PFM ground truth):

| Algorithm | Model Type | RMSE (px) $\downarrow$ | AbsRel $\downarrow$ | Bad-3px (D1 %) $\downarrow$ | EPE (px) $\downarrow$ | Inference Speed |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **StereoBM** | Classical Sliding Window | 44.12 | 0.3120 | 42.15% | 22.40 | **$\sim 15\text{ ms}$ (CPU)** |
| **StereoSGBM** | Classical Semi-Global Matching | 38.64 | 0.2378 | 30.23% | 14.82 | $\sim 45\text{ ms}$ (CPU) |
| **RAFT-Stereo (Pre-trained)** | Deep Learning (SceneFlow) | 3.98 | 0.0091 | 4.46% | 1.28 | $\sim 120\text{ ms}$ (T4 GPU) |
| **RAFT-Stereo (Fine-Tuned)** | Deep Learning (Few-Shot M3) | **3.65** | **0.0084** | **3.89%** | **1.14** | $\sim 120\text{ ms}$ (T4 GPU) |

![Quantitative Evaluation Benchmark Metrics](report_figures/fig4_quantitative_metrics.png)

*Metrics:*
* **RMSE**: Root Mean Squared Error $\sqrt{\frac{1}{|M|} \sum (d_{\text{pred}} - d_{\text{gt}})^2}$
* **AbsRel**: Absolute Relative Error $\frac{1}{|M|} \sum \frac{|d_{\text{pred}} - d_{\text{gt}}|}{d_{\text{gt}}}$
* **Bad-3px (D1)**: Percentage of valid pixels with disparity error $> 3.0\text{ px}$
* **EPE**: End-Point Error $\frac{1}{|M|} \sum |d_{\text{pred}} - d_{\text{gt}}|$

---

## 💻 Web Application Features

The production Gradio 5 application (`app.py`) provides four interactive tabs:

1. **Single Model Disparity**: Upload any stereo pair (or click the 1-click sample button), adjust parameters (`blockSize`, `numDisparities`, `uniquenessRatio`, colormaps: `plasma`, `turbo`, `magma`, `viridis`), and compute colorized disparity maps.
2. **4-Way Model Comparison**: Compare StereoBM, StereoSGBM, RAFT Pre-trained, and RAFT Fine-tuned side-by-side on identical stereo inputs.
3. **Ground Truth Benchmark**: Upload a ground-truth disparity map (`.pfm` or `.png`) alongside a stereo pair to compute RMSE, AbsRel, Bad-3px rate, and render an interactive error heatmap.
4. **Theory & Architecture**: Integrated technical documentation and academic references.

---

## 🚀 Getting Started & Local Execution

### Option 1: Live Cloud Demo
No local setup needed. Open:  
👉 **[https://huggingface.co/spaces/malith26/stereo-disparity-depth-estimation](https://huggingface.co/spaces/malith26/stereo-disparity-depth-estimation)**

### Option 2: Run in Google Colab
```python
# 1. Clone repository
!git clone https://github.com/cepdnaclk/e23-co5430-Stereo-disparity_depth-estimation.git
%cd e23-co5430-Stereo-disparity_depth-estimation

# 2. Install dependencies
!pip install -r requirements.txt

# 3. Launch interactive app
!python app.py
```

### Option 3: Run Locally (Linux / macOS / Windows)
```bash
# 1. Clone repository
git clone https://github.com/cepdnaclk/e23-co5430-Stereo-disparity_depth-estimation.git
cd e23-co5430-Stereo-disparity_depth-estimation

# 2. Set up virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch web application
python app.py
```
Open **`http://localhost:7860`** in your browser.

### Option 4: Deploy to Hugging Face Spaces
To update or deploy to your own Hugging Face Space:
```bash
# Linux / macOS / WSL:
./deploy_to_hf.sh

# Windows Command Prompt:
deploy_to_hf.bat

# Windows PowerShell:
.\deploy_to_hf.ps1
```

---

## 📁 Repository Structure

```text
e23-co5430-Stereo-disparity_depth-estimation/
├── app.py                     # Production Gradio 5 application with ZeroGPU hooks
├── requirements.txt           # Python dependency specifications
├── packages.txt               # Debian system dependencies for Spaces container
├── deploy_to_hf.sh            # Automated deployment script for Hugging Face Spaces
├── deploy_to_hf.bat           # Windows Command Prompt deployment script
├── deploy_to_hf.ps1           # Windows PowerShell deployment script
├── README.md                  # Project documentation, benchmarks & HF metadata
│
├── core/                      # Embedded RAFT-Stereo deep learning core
│   ├── corr.py                # 1D epipolar correlation pyramid
│   ├── extractor.py           # Multi-scale residual feature & context networks
│   ├── raft_stereo.py         # Full RAFT-Stereo PyTorch model definition
│   ├── update.py              # Recurrent ConvGRU update operator
│   └── utils/                 # Spatial frame padding & geometry utilities
│
├── src/                       # Modular computer vision library
│   ├── dataset.py             # 32-bit floating-point PFM ground truth reader
│   ├── preprocessing.py       # Spatial cropping (384x512) & image normalization
│   ├── stereo_bm.py           # Classical StereoBM implementation
│   ├── stereo_sgbm.py         # Classical StereoSGBM implementation (dynamic P1/P2)
│   ├── raft_wrapper.py        # RAFT-Stereo PyTorch inference wrapper (CPU/CUDA)
│   ├── evaluation.py          # Benchmark metrics: RMSE, AbsRel, Bad-3px D1, EPE
│   └── visualization.py       # Colorized disparity maps & error heatmaps
│
├── demo_images/               # Middlebury test stereo pairs & ground-truth PFM
│   ├── artroom1_ground_truth.pfm
│   ├── stereobm_left.png / stereobm_right.png
│   ├── stereosgbm_left.png / stereosgbm_right.png
│   ├── raft_pretrained_left.png / raft_pretrained_right.png
│   └── raft_finetuned_left.png / raft_finetuned_right.png
│
├── report_figures/            # High-resolution architectural figures & diagrams
│   ├── fig1_epipolar_geometry.png
│   ├── fig2_system_pipeline.png
│   ├── fig3_raft_architecture.png
│   ├── fig4_quantitative_metrics.png
│   └── fig5_qualitative_comparison.png
│
├── Documents/                 # Academic milestone deliverables & reports
│   ├── Project Proposal.pdf   # Approved project proposal
│   ├── M2.pptx                # Milestone 2 presentation slides
│   ├── M3.pptx                # Milestone 3 presentation slides
│   └── CO5430_P12_Final_Report_IEEE.docx # Comprehensive IEEE technical report
│
└── tests/
    └── test_app_startup.py    # Automated startup and API regression tests
```

---

## 📜 References

1. D. Scharstein and R. Szeliski, *"A taxonomy and evaluation of dense two-frame stereo correspondence algorithms,"* International Journal of Computer Vision (IJCV), vol. 47, no. 1-3, pp. 7–42, 2002.
2. H. Hirschmüller, *"Stereo processing by semiglobal matching and mutual information,"* IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), vol. 30, no. 2, pp. 328–341, 2008.
3. L. Lipson, Z. Teed, and J. Deng, *"RAFT-Stereo: Multilevel recurrent field transforms for stereo matching,"* in International Conference on 3D Vision (3DV), IEEE, 2021, pp. 218–227.
4. Z. Teed and J. Deng, *"RAFT: Recurrent all-pairs field transforms for optical flow,"* in European Conference on Computer Vision (ECCV), Springer, 2020, pp. 402–419.
5. G. Pan, T. Sun, T. Weed, and D. Scharstein, *"2021 Mobile Stereo Datasets with Ground Truth,"* Middlebury Stereo Datasets, 2021.
