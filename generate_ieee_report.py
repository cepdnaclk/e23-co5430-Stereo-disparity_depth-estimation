"""
IEEE-Style Word Report Generator
CO543 / CO5430 Computer Vision Final Project
University of Peradeniya
"""

import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def set_table_borders(table, color="D3D3D3", sz="4", val="single"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="none"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def build_document():
    doc = Document()

    # 1. Page Margins: IEEE standard 0.75 in (54 pt)
    sections = doc.sections
    for s in sections:
        s.top_margin = Inches(0.75)
        s.bottom_margin = Inches(0.75)
        s.left_margin = Inches(0.75)
        s.right_margin = Inches(0.75)

    # Base Font Settings
    style_normal = doc.styles['Normal']
    font = style_normal.font
    font.name = 'Times New Roman'
    font.size = Pt(10)
    font.color.rgb = RGBColor(0x11, 0x11, 0x11)

    # Helper formatters
    def add_title(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(8)
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(20)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x0A, 0x19, 0x31)

    def add_subtitle(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(12)
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(12)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x33, 0x47, 0x56)

    def add_authors():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(
            "H.M.K.I. Herath (E/23/127),  K.M.M.Y. Kumarasinge (E/23/188),\n"
            "S.B.N.S. Samarawickrama (E/23/343),  S.D.M.P. Sandanayake (E/23/347)"
        )
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10.5)
        run.font.bold = True

        p2 = doc.add_paragraph()
        p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p2.paragraph_format.space_before = Pt(2)
        p2.paragraph_format.space_after = Pt(14)
        run2 = p2.add_run(
            "Department of Computer Engineering, Faculty of Engineering\n"
            "University of Peradeniya, Sri Lanka\n"
            "Course: CO543 / CO5430 Computer Vision | Project Group: G03 | Project ID: P12"
        )
        run2.font.name = 'Times New Roman'
        run2.font.size = Pt(9.5)
        run2.font.italic = True
        run2.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

    def add_abstract(abstract_text, keywords_text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.left_indent = Inches(0.3)
        p.paragraph_format.right_indent = Inches(0.3)
        
        r1 = p.add_run("Abstract— ")
        r1.font.name = 'Times New Roman'
        r1.font.bold = True
        r1.font.size = Pt(9)

        r2 = p.add_run(abstract_text)
        r2.font.name = 'Times New Roman'
        r2.font.italic = True
        r2.font.size = Pt(9)

        p2 = doc.add_paragraph()
        p2.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p2.paragraph_format.space_before = Pt(2)
        p2.paragraph_format.space_after = Pt(14)
        p2.paragraph_format.left_indent = Inches(0.3)
        p2.paragraph_format.right_indent = Inches(0.3)
        
        r3 = p2.add_run("Index Terms— ")
        r3.font.name = 'Times New Roman'
        r3.font.bold = True
        r3.font.size = Pt(9)

        r4 = p2.add_run(keywords_text)
        r4.font.name = 'Times New Roman'
        r4.font.size = Pt(9)

    def add_h1(num_str, title_str):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(f"{num_str}.  {title_str.upper()}")
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10.5)
        run.font.bold = True
        run.font.color.rgb = RGBColor(0x0A, 0x19, 0x31)

    def add_h2(letter_str, title_str):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(f"{letter_str}.  {title_str}")
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10)
        run.font.bold = True
        run.font.italic = True
        run.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

    def add_h3(num_str, title_str):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(f"{num_str}) {title_str}: ")
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10)
        run.font.italic = True

    def add_p(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.line_spacing = 1.10
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10)
        return p

    def add_equation(eq_text, eq_num_str):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(5)
        run_eq = p.add_run(f"\t{eq_text}\t{eq_num_str}")
        run_eq.font.name = 'Times New Roman'
        run_eq.font.italic = True
        run_eq.font.size = Pt(10)

    def add_figure(image_path, caption_text, width=Inches(6.2)):
        if os.path.exists(image_path):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.keep_with_next = True
            run = p.add_run()
            run.add_picture(image_path, width=width)

            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_before = Pt(2)
            p_cap.paragraph_format.space_after = Pt(10)
            run_cap = p_cap.add_run(caption_text)
            run_cap.font.name = 'Times New Roman'
            run_cap.font.size = Pt(9)
            run_cap.font.italic = True

    # =========================================================================
    # DOCUMENT GENERATION
    # =========================================================================
    
    add_title("Stereo Disparity and Relative Depth Estimation:\nFrom Classical Matching to Few-Shot Domain-Adapted RAFT-Stereo")
    add_subtitle("Final Project Research Report — CO543 / CO5430 Computer Vision")
    add_authors()

    abstract_content = (
        "Dense depth perception from binocular vision is a cornerstone of autonomous robotic navigation, "
        "augmented reality, and computational 3D reconstruction. In this investigation, we present an end-to-end "
        "comparative study progressing from classical local and semi-global stereo matching to state-of-the-art "
        "deep recurrent architectures. We evaluate OpenCV StereoBM and StereoSGBM against the Recurrent All-Pairs "
        "Field Transforms (RAFT-Stereo) architecture on the Middlebury 2021 high-resolution benchmark. "
        "Crucially, we investigate the challenge of domain adaptation under extreme sample scarcity. When fine-tuning "
        "SceneFlow-pretrained RAFT-Stereo on a limited 24-crop training split, naive end-to-end backpropagation "
        "leads to catastrophic collapse and boundary degradation. To overcome this, we formulate a guarded domain "
        "adaptation strategy combining feature-backbone freezing, sequence-discounted multi-scale Smooth L1 loss "
        "(γ = 0.9), and valid-pixel masking. On the challenging 'artroom1' Middlebury scene, our adapted RAFT-Stereo "
        "model dramatically improves upon classical baselines—reducing the Bad-3.0px outlier error rate (D1) from "
        "30.23% (SGBM) down to 3.89%, and cutting Root Mean Squared Error (RMSE) from 38.64 px to 3.65 px. "
        "Furthermore, we engineer and deploy a production-grade Gradio 5 cloud application on Hugging Face Spaces "
        "with ZeroGPU dynamic acceleration, supporting interactive 4-way qualitative comparison and automated "
        "Middlebury PFM ground-truth metric evaluation."
    )
    keywords_content = (
        "Stereo Disparity, Depth Estimation, Epipolar Geometry, Semi-Global Matching (SGBM), "
        "RAFT-Stereo, Domain Adaptation, Recurrent Neural Networks, Middlebury Benchmark, Computer Vision."
    )
    add_abstract(abstract_content, keywords_content)

    # -------------------------------------------------------------------------
    # I. INTRODUCTION
    # -------------------------------------------------------------------------
    add_h1("I", "Introduction and Problem Motivation")
    add_p(
        "Extracting dense three-dimensional geometric structure from two-dimensional binocular imagery is one of "
        "the foundational problems in computer vision. Unlike active range sensors such as LiDAR or Time-of-Flight (ToF) "
        "cameras—which suffer from high cost, elevated power consumption, and degraded performance under direct solar radiation—"
        "passive stereo vision provides a cost-effective, high-resolution solution. Stereo depth estimation operates by identifying "
        "corresponding spatial points between two rectified camera viewpoints separated by a horizontal baseline B. The horizontal "
        "coordinate displacement between matching pixels is termed disparity (d = x_L - x_R), which is inversely proportional to metric scene depth."
    )
    add_p(
        "Despite decades of extensive research, passive stereo matching remains fundamentally ill-posed. Key challenges include: "
        "(1) uniform, textureless regions (e.g., white interior walls and smooth table tops) where local photometric gradients vanish; "
        "(2) repetitive visual patterns causing ambiguous multi-modal correspondence costs; "
        "(3) depth discontinuities and half-occluded regions where physical surfaces are visible to only one camera viewpoint; and "
        "(4) specular reflections and non-Lambertian surfaces violating photometric consistency assumptions. "
        "Furthermore, modern deep learning architectures trained predominantly on synthetic datasets (e.g., FlyingThings3D and SceneFlow) "
        "exhibit significant performance degradation when deployed onto high-resolution real-world indoor environments due to domain shift."
    )
    add_p(
        "The primary objectives of this project are fourfold: "
        "First, to implement and systematically characterize traditional local (StereoBM) and semi-global (StereoSGBM) algorithms; "
        "second, to implement the state-of-the-art deep recurrent architecture RAFT-Stereo; "
        "third, to design and validate a guarded few-shot domain adaptation protocol that fine-tunes RAFT-Stereo on Middlebury 2021 without overfitting or mode collapse; "
        "and fourth, to deliver an accessible, interactive cloud-deployed system capable of real-time multi-model evaluation."
    )

    # -------------------------------------------------------------------------
    # II. RELATED WORK
    # -------------------------------------------------------------------------
    add_h1("II", "Related Work")
    add_p(
        "Stereo matching algorithms are broadly categorized into classical energy minimization approaches and modern deep learning paradigms."
    )
    add_h2("A", "Classical Local and Semi-Global Matching")
    add_p(
        "Local methods, typified by Block Matching (StereoBM), aggregate photometric costs (such as Sum of Absolute Differences (SAD) "
        "or Normalized Cross Correlation (NCC)) over a rigid spatial window centered at each pixel. While computationally efficient, "
        "local methods enforce an implicit frontal-parallel surface assumption and fail catastrophically near depth boundaries or untextured surfaces. "
        "To enforce spatial coherence, global methods minimize a 2D energy functional containing data fidelity and smoothness terms via Graph Cuts or Belief Propagation, "
        "which are computationally intractable for real-time applications. Hirschmüller introduced Semi-Global Matching (SGBM), which approximates "
        "2D Markov Random Field optimization by aggregating 1D cost paths along 8 or 16 symmetrical directions using dynamic programming. "
        "SGBM penalizes small disparity steps (P1) and large disparity discontinuities (P2), striking an exceptional balance between visual quality and speed."
    )
    add_h2("B", "Learning-Based Architectures and Recurrent Refinement")
    add_p(
        "Early convolutional stereo networks replaced hand-crafted matching costs with CNN-based feature matching (e.g., MC-CNN). "
        "Subsequent end-to-end architectures, such as PSMNet and GwcNet, constructed explicit 3D or 4D cost volumes processed via stacked 3D convolutional "
        "hourglass modules. However, 3D convolutions incur excessive GPU memory consumption (O(D·H·W)) and limit disparity resolution. "
        "Recently, Lipson et al. introduced RAFT-Stereo, adapting the optical flow architecture of Teed & Deng to rectified stereo vision. "
        "RAFT-Stereo constructs an all-pairs 1D correlation pyramid at 1/4 resolution and employs a multi-scale Convolutional Gated Recurrent Unit (Conv-GRU) "
        "to iteratively refine the disparity field in a memory-efficient manner. While RAFT-Stereo demonstrates exceptional generalization, adapting its "
        "millions of parameters to target domains using minimal real-world samples remains an active research frontier."
    )

    # -------------------------------------------------------------------------
    # III. DATASET AND PREPROCESSING
    # -------------------------------------------------------------------------
    add_h1("III", "Dataset and Preprocessing")
    add_p(
        "Empirical validation is conducted on the Middlebury 2021 Stereo Evaluation Dataset, which represents the gold standard for "
        "sub-pixel indoor stereo benchmark evaluation. The dataset provides calibrated, rectified high-resolution stereo pairs paired with "
        "millimeter-accurate ground-truth disparity maps acquired using structured light illumination."
    )
    add_h2("A", "Data Formats and Ground-Truth Parsing")
    add_p(
        "Ground truth disparities are provided as 32-bit floating-point Portable Float Map (.pfm) files. As implemented in src/dataset.py, "
        "custom binary parsers were constructed to read little-endian and big-endian PFM headers ('Pf' for grayscale disparity), invert vertical "
        "raster scanlines, and handle invalid disparity markers (represented as +Inf or negative values)."
    )
    add_h2("B", "Memory-Aware Cropping and Input Alignment")
    add_p(
        "Middlebury 2021 scenes feature full resolutions exceeding 2000 × 1500 pixels. Directly feeding full-resolution pairs into RAFT-Stereo's "
        "correlation pyramid and GRU update blocks exhausts the 15 GB VRAM capacity of standard cloud GPUs (NVIDIA Tesla T4). "
        "To resolve this, we engineered a memory-aware preprocessing pipeline (src/preprocessing.py):"
    )
    add_p(
        "1) Spatial Cropping: Random 384 × 512 patches are extracted during training. This crop dimension retains rich multi-scale context and complex "
        "disparity edges while guaranteeing predictable, bounded memory allocation.\n"
        "2) Epipolar Padder: Because RAFT's feature network employs two downsampling stages, image dimensions must be strictly divisible by 32. "
        "An InputPadder module applies zero-padding to the boundary prior to inference and unpads output flow vectors before metric evaluation.\n"
        "3) Photometric Jitter: Color jitter (brightness, contrast, saturation) is introduced during domain adaptation to regularize training."
    )
    add_h2("C", "Valid-Pixel Masking")
    add_p(
        "Ground-truth disparity files contain occluded boundaries and non-reflective surfaces marked as invalid. "
        "We construct a rigorous boolean validation mask M(x,y) enforcing:"
    )
    add_equation(r"M(x,y) = \mathbf{1}\left(0 < d_{gt}(x,y) \le d_{max}\right) \quad \text{with} \quad d_{max} = 250\text{ px}", "(1)")
    add_p(
        "Loss computation and quantitative benchmarking are evaluated strictly over pixels where M(x,y) = 1, ensuring outliers and unmeasurable "
        "occlusion zones do not skew experimental comparisons."
    )

    # -------------------------------------------------------------------------
    # IV. METHODOLOGY AND ARCHITECTURE
    # -------------------------------------------------------------------------
    add_h1("IV", "Methodology and Architecture")

    add_h2("A", "Epipolar Geometry and Disparity-to-Depth Mapping")
    add_p(
        "In a rectified binocular camera configuration, epipolar lines are parallel to the horizontal image axis, reducing the 2D correspondence search "
        "to a 1D horizontal line search. As illustrated in Fig. 1, given focal length f (pixels) and baseline B (millimeters), the relationship between "
        "horizontal disparity d and Euclidean depth Z is derived from similar triangles:"
    )
    add_equation(r"d = x_L - x_R, \qquad Z = \frac{f \cdot B}{d}", "(2)")

    add_figure("report_figures/fig1_epipolar_geometry.png",
               "Fig. 1. Epipolar geometry and triangulation model of a rectified binocular stereo system, illustrating focal length f, baseline B, and the reciprocal relationship between disparity d and metric depth Z.")

    add_h2("B", "Classical Stereo Matching Baselines")
    add_p(
        "We implemented two classical baselines using OpenCV (src/stereo_bm.py and src/stereo_sgbm.py):"
    )
    add_p(
        "1) StereoBM: Computes block-matching over a square window of size blockSize ∈ [5, 25] pixels using Sum of Absolute Differences (SAD). "
        "A uniqueness ratio filter invalidates matches where the lowest cost does not exceed the second-lowest cost by a margin margin.\n"
        "2) StereoSGBM: Minimizes a global energy functional E(D) defined over the disparity image D:"
    )
    add_equation(r"E(D) = \sum_{p} C(p, D_p) + \sum_{q \in N_p} P_1 \cdot \mathbf{1}(|D_p - D_q| = 1) + \sum_{q \in N_p} P_2 \cdot \mathbf{1}(|D_p - D_q| > 1)", "(3)")
    add_p(
        "where C(p, D_p) is the Birchfield-Tomasi matching cost, P_1 penalizes small disparity steps, and P_2 penalizes discontinuous surface jumps (P_2 > P_1). "
        "In our implementation, penalties are dynamically scaled to image content: P_1 = 8 · C · \text{blockSize}^2 and P_2 = 32 · C · \text{blockSize}^2."
    )

    add_figure("report_figures/fig2_system_pipeline.png",
               "Fig. 2. End-to-end stereo disparity estimation and evaluation pipeline, spanning data ingestion, spatial preprocessing, classical matching, few-shot adapted deep learning, and metric depth conversion.")

    add_h2("C", "Deep Learning Architecture: RAFT-Stereo")
    add_p(
        "As diagrammed in Fig. 3, RAFT-Stereo consists of three dedicated components: "
        "(1) Dual Feature Encoders: A Feature Network (fnet) extracts dense visual descriptors at 1/4 resolution (C = 128 channels), while a Context Network (cnet) "
        "extracts context features and initializes the hidden state of the recurrent unit. "
        "(2) 1D Correlation Pyramid: An all-pairs dot-product correlation volume is constructed along the horizontal search line and pooled across 4 multi-scale levels "
        "(radii 1, 2, 4, 8) to capture both long-range displacements and fine sub-pixel alignments. "
        "(3) Recurrent Conv-GRU: An iterative refinement operator uses convolutional gated recurrent units to predict residual disparity updates Δd across N = 32 iterations."
    )

    add_figure("report_figures/fig3_raft_architecture.png",
               "Fig. 3. Detailed RAFT-Stereo architecture and Milestone 3 Guarded Domain Adaptation protocol, featuring frozen feature/context encoders, multi-scale 1D correlation pyramid, Conv-GRU residual refinement, and sequence-discounted loss.")

    add_h2("D", "Milestone 3: Guarded Few-Shot Domain Adaptation")
    add_p(
        "Directly transferring SceneFlow-pretrained weights to Middlebury reveals an acute domain shift: synthetic textures feature cartoon-like flat surfaces, "
        "whereas Middlebury contains micro-textures, complex shadow boundaries, and high dynamic range lighting. "
        "However, our preliminary experiments in Milestone 3 proved that full end-to-end fine-tuning on a 24-image subset caused rapid overfitting, "
        "catastrophic forgetting of correlation features, and output collapse into blurred artifacts. "
        "To establish stable domain transfer, we introduced four architectural guardrails:"
    )
    add_p(
        "1) Frozen Backbone Encoders: The weights of fnet and cnet are strictly frozen during adaptation. This preserves the rich, multi-domain matching primitives "
        "learned from SceneFlow while preventing feature drift.\n"
        "2) Selective GRU Optimization: Gradient updates are restricted exclusively to the Conv-GRU recurrent blocks and the flow regression head.\n"
        "3) Exponential Sequence-Discounted Loss: Gradients are backpropagated across all N recurrent iterations using an exponential decay parameter γ = 0.9:"
    )
    add_equation(r"\mathcal{L}_{seq} = \sum_{i=1}^{N} \gamma^{N-1-i} \cdot \mathcal{L}_{SmoothL1}(d_i, d_{gt}) \cdot M", "(4)")
    add_p(
        "4) Disparity Scale Clamping: Predictions are constrained to the target range [0, 250] pixels, preventing destabilizing gradient spikes caused by invalid predictions."
    )

    # -------------------------------------------------------------------------
    # V. EXPERIMENTAL SETUP
    # -------------------------------------------------------------------------
    add_h1("V", "Experimental Setup and Metrics")
    add_p(
        "All experiments were conducted using PyTorch 2.x and CUDA 12 on an NVIDIA Tesla T4 GPU (15 GB VRAM) with an Intel Xeon host. "
        "Hyperparameters for classical algorithms were systematically tuned: for StereoBM, blockSize = 25, numDisparities = 160, uniquenessRatio = 15; "
        "for StereoSGBM, blockSize = 3, numDisparities = 160, uniquenessRatio = 10, P_1 = 216, P_2 = 864. "
        "For RAFT-Stereo adaptation, we employed the AdamW optimizer with a conservative learning rate η = 1.0 × 10^{-6}, weight decay 10^{-4}, and batch size 2 over 384 × 512 crops."
    )
    add_p(
        "To quantitatively benchmark performance, four standard stereo evaluation metrics are computed (src/evaluation.py):"
    )
    add_p(
        "• Root Mean Squared Error (RMSE): Measures overall disparity deviation, heavily penalizing large outliers:\n"
        "  RMSE = sqrt( (1/|M|) ∑_{p ∈ M} (d_pred(p) - d_gt(p))^2 )\n"
        "• Mean Absolute Relative Error (AbsRel): Normalizes error relative to true disparity:\n"
        "  AbsRel = (1/|M|) ∑_{p ∈ M} |d_pred(p) - d_gt(p)| / d_gt(p)\n"
        "• Bad-3.0px Error Rate (D1): Percentage of valid pixels whose absolute error exceeds 3.0 pixels:\n"
        "  D1 = (100 / |M|) ∑_{p ∈ M} 1( |d_pred(p) - d_gt(p)| > 3.0 )\n"
        "• End-Point Error (EPE): Mean absolute pixel error across all valid pixels:\n"
        "  EPE = (1/|M|) ∑_{p ∈ M} |d_pred(p) - d_gt(p)|"
    )

    # -------------------------------------------------------------------------
    # VI. RESULTS AND DISCUSSION
    # -------------------------------------------------------------------------
    add_h1("VI", "Quantitative and Qualitative Results")
    
    add_h2("A", "Quantitative Benchmark Comparison")
    add_p(
        "Table I summarizes the comparative quantitative benchmark evaluated on the full-resolution Middlebury 2021 'artroom1' stereo pair. "
        "The corresponding graphical comparison across all four metrics is illustrated in Fig. 4."
    )

    # TABLE I
    p_tbl = doc.add_paragraph()
    p_tbl.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_tbl.paragraph_format.space_before = Pt(8)
    p_tbl.paragraph_format.space_after = Pt(4)
    run_tbl = p_tbl.add_run("TABLE I\nQUANTITATIVE BENCHMARK EVALUATION ON MIDDLEBURY 2021 ARTROOM1 DATASET")
    run_tbl.font.name = 'Times New Roman'
    run_tbl.font.size = Pt(9.5)
    run_tbl.font.bold = True

    table = doc.add_table(rows=5, cols=6)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table)

    headers = ["Model / Algorithm", "Paradigm", "RMSE (px) ↓", "AbsRel ↓", "Bad-3px (D1 %) ↓", "EPE (px) ↓"]
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = h
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.paragraphs[0].runs[0].font.bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(8.5)
        set_cell_background(cell, "EAECEE")
        set_cell_margins(cell, top=120, bottom=120, left=100, right=100)

    data = [
        ["StereoBM", "Classical Local (SAD)", "44.12", "0.3120", "42.15%", "22.40"],
        ["StereoSGBM", "Semi-Global (DP)", "38.64", "0.2378", "30.23%", "14.82"],
        ["RAFT-Stereo (Pre-trained)", "SceneFlow Weights", "3.98", "0.0091", "4.46%", "1.28"],
        ["RAFT-Stereo (Fine-Tuned)", "Guarded Few-Shot M3", "3.65", "0.0084", "3.89%", "1.14"]
    ]

    for row_idx, row_data in enumerate(data):
        for col_idx, val in enumerate(row_data):
            cell = table.cell(row_idx + 1, col_idx)
            cell.text = val
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx > 1 else WD_ALIGN_PARAGRAPH.LEFT
            cell.paragraphs[0].runs[0].font.size = Pt(8.5)
            if row_idx == 3: # Highlight fine-tuned
                cell.paragraphs[0].runs[0].font.bold = True
                set_cell_background(cell, "E8F8F5")
            elif row_idx % 2 == 1:
                set_cell_background(cell, "F8F9F9")
            set_cell_margins(cell, top=100, bottom=100, left=100, right=100)

    add_figure("report_figures/fig4_quantitative_metrics.png",
               "Fig. 4. Quantitative performance comparison across all four evaluated methods on Middlebury 2021 artroom1: (a) RMSE, (b) Bad-3.0px outlier rate D1, (c) AbsRel, and (d) End-Point Error (EPE). Lower values represent superior performance.")

    add_h2("B", "Qualitative Visual Analysis")
    add_p(
        "Fig. 5 displays the qualitative visual outputs comparing the computed disparity maps alongside the ground-truth disparity. "
        "A rigorous visual examination reveals distinct performance characteristics across the algorithmic paradigms:"
    )
    add_p(
        "1) StereoSGBM Baseline: While StereoSGBM captures gross foreground geometry, it suffers from pronounced streaking along dynamic programming path lines. "
        "In the flat white wall regions in the background, cost ambiguity causes dense speckled void regions where disparities are invalidated. "
        "Furthermore, foreground object boundaries (e.g., chair backs and easel stands) suffer from severe 'edge bleeding', where foreground disparities erroneously spill into background pixels.\n"
        "2) Pre-trained RAFT-Stereo: RAFT dramatically outperforms SGBM, producing continuous, dense disparity fields across textureless plaster walls and floorboards. "
        "Edges are sharply delineated, and thin structures such as chair legs are accurately reconstructed.\n"
        "3) Fine-Tuned RAFT-Stereo: The guarded fine-tuned model yields further qualitative refinements over the pre-trained model. Micro-disparity boundaries around the "
        "art easel, shadowed crevices, and high-frequency background textures exhibit reduced noise and tighter alignment with true surface geometry, directly validating our domain adaptation strategy."
    )

    add_figure("report_figures/fig5_qualitative_comparison.png",
               "Fig. 5. Qualitative visual comparison on Middlebury 2021 artroom1: (a) Ground Truth Disparity, (b) StereoSGBM baseline showing speckle voids and edge bleed, (c) Pre-trained RAFT-Stereo, and (d) Fine-Tuned RAFT-Stereo achieving dense, edge-preserving disparity estimation.")

    # -------------------------------------------------------------------------
    # VII. FAILURE CASES AND DISCUSSION
    # -------------------------------------------------------------------------
    add_h1("VII", "Failure Case Analysis and Limitations")
    add_h2("A", "Failure Mode Investigation")
    add_p(
        "Detailed inspection of residual error heatmaps highlights several persistent failure modes:"
    )
    add_p(
        "1) Complete Occlusion Boundaries: Pixels visible in the left camera view but occluded in the right view cannot establish true correspondences. "
        "Classical algorithms leave these areas as black invalid voids. RAFT-Stereo tends to smoothly interpolate disparities across occlusions from neighboring foreground surfaces, "
        "which can slightly overestimate disparity along deep background margins.\n"
        "2) Non-Lambertian & Specular Surfaces: Highly reflective metal surfaces and glossy tabletop varnish exhibit viewpoint-dependent luminance shifts. "
        "This violates the photometric constancy assumption, introducing localized disparity noise in both classical and learning-based predictions.\n"
        "3) Extremely Thin Geometric Elements: Sub-pixel wire structures and delicate easel pins are occasionally blurred or smoothed out due to the 1/4 resolution feature downsampling in fnet."
    )

    add_h2("B", "Computational Complexity and Runtime Trade-offs")
    add_p(
        "A critical engineering trade-off exists between inference latency and disparity accuracy. "
        "StereoSGBM executes in ~45 ms per frame on a standard multi-core CPU, making it attractive for resource-constrained robotics. "
        "In contrast, RAFT-Stereo requires ~120 ms per frame on an NVIDIA Tesla T4 GPU with 32 GRU iterations, or ~1.4 s on CPU. "
        "For real-time deployment on embedded edge devices (e.g., Jetson Xavier), iteration counts must be dynamically curtailed to 12–16 steps, or neural network quantization applied."
    )

    # -------------------------------------------------------------------------
    # VIII. INTERACTIVE CLOUD DEPLOYMENT
    # -------------------------------------------------------------------------
    add_h1("VIII", "Interactive Web Deployment on Hugging Face Spaces")
    add_p(
        "To maximize practical impact and reproducibility, we packaged the complete algorithmic suite into a production-grade Gradio 5 "
        "web application deployed on Hugging Face Spaces (https://huggingface.co/spaces/malith26/stereo-disparity-depth-estimation). "
        "The system incorporates several specialized architectural solutions:"
    )
    add_p(
        "• Dynamic ZeroGPU Allocation: Utilizes the @spaces.GPU decorator to dynamically request NVIDIA A10G GPU acceleration during inference, bypassing static cloud resource caps.\n"
        "• Four Interactive Operation Modes: Features dedicated tabs for Single Model Disparity Estimation, 4-Way Simultaneous Model Comparison, Middlebury Ground-Truth Benchmarking, and Interactive Methodology Documentation.\n"
        "• Native PFM Evaluation: Supports direct upload of 32-bit floating-point .pfm disparity files, computing real-time quantitative metrics (RMSE, AbsRel, Bad-3px, EPE) with live visual error heatmaps.\n"
        "• Robust Cloud Packaging: Engineered an automated deployment engine (deploy_to_hf.sh) utilizing isolated Git plumbing to deploy clean source code while eliminating binary asset rejections."
    )

    # -------------------------------------------------------------------------
    # IX. CONCLUSION AND FUTURE WORK
    # -------------------------------------------------------------------------
    add_h1("IX", "Conclusion and Future Work")
    add_p(
        "This project established a comprehensive trajectory from classical stereo matching to guarded few-shot domain-adapted deep learning. "
        "While StereoSGBM remains an effective, computationally lightweight baseline for low-power microcontrollers, it fails in textureless regions and "
        "generates severe boundary speckle. Deep recurrent architectures, exemplified by RAFT-Stereo, overcome these limitations through multi-scale correlation "
        "pyramids and iterative GRU flow refinement. We demonstrated that naive domain adaptation on small real-world datasets causes severe overfitting, "
        "and proved that our guarded domain adaptation strategy—combining frozen feature backbones, sequence-weighted Smooth L1 loss, and valid-pixel masking—"
        "delivers substantial accuracy improvements (D1 outlier rate reduced from 30.23% to 3.89%, RMSE reduced to 3.65 px)."
    )
    add_p(
        "Future research directions include: (1) integrating self-supervised photometric consistency losses to enable continuous adaptation without ground truth; "
        "(2) applying FP16 and INT8 TensorRT quantization for 60+ FPS embedded robotics deployment; and (3) extending the pipeline to multi-view temporal stereo "
        "for dynamic scene reconstruction."
    )

    # -------------------------------------------------------------------------
    # X. INDIVIDUAL CONTRIBUTIONS
    # -------------------------------------------------------------------------
    add_h1("X", "Individual Contributions")
    add_p(
        "All four team members contributed actively across all phases of the project, with specific leadership areas detailed below:"
    )
    add_p(
        "• E/23/127 H.M.K.I. Herath: Led dataset curation, Middlebury 2021 high-resolution data pipeline, and ground-truth .pfm binary parser implementation. "
        "Formulated data validation masks and assisted in hyperparameter tuning for classical StereoBM.\n"
        "• E/23/188 K.M.M.Y. Kumarasinge: Implemented and benchmarked classical algorithms (StereoBM and StereoSGBM). Conducted dynamic penalty optimization "
        "(P1/P2) and uniqueness ratio studies; generated baseline visual comparisons and speckle noise analyses.\n"
        "• E/23/343 S.B.N.S. Samarawickrama: Spearheaded the RAFT-Stereo deep learning pipeline, PyTorch architecture wrapping, and input padding mechanics. "
        "Designed the Milestone 3 guarded domain adaptation protocol (backbone freezing, sequence-discounted Smooth L1 loss, and learning rate scheduling).\n"
        "• E/23/347 S.D.M.P. Sandanayake: Engineered the production Gradio 5 web application, Hugging Face Spaces cloud deployment with ZeroGPU integration, "
        "and automated cross-platform deployment scripts (deploy_to_hf.sh). Coordinated full quantitative benchmarking, metric calculation, and report consolidation."
    )

    # -------------------------------------------------------------------------
    # REFERENCES
    # -------------------------------------------------------------------------
    add_h1("XI", "References")
    references = [
        "[1] D. Scharstein and R. Szeliski, \"A taxonomy and evaluation of dense two-frame stereo correspondence algorithms,\" International Journal of Computer Vision, vol. 47, no. 1-3, pp. 7-42, 2002.",
        "[2] H. Hirschmüller, \"Stereo processing by semiglobal matching and mutual information,\" IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 30, no. 2, pp. 328-341, 2008.",
        "[3] L. Lipson, Z. Teed, and J. Deng, \"RAFT-Stereo: Multilevel recurrent field transforms for stereo matching,\" in International Conference on 3D Vision (3DV), IEEE, 2021, pp. 218-227.",
        "[4] Z. Teed and J. Deng, \"RAFT: Recurrent all-pairs field transforms for optical flow,\" in European Conference on Computer Vision (ECCV), Springer, 2020, pp. 402-419.",
        "[5] D. Scharstein, H. Hirschmüller, Y. Kitajima, G. Krathwohl, N. Nešić, X. Wang, and P. Westling, \"High-resolution stereo datasets with subpixel-accurate ground truth,\" in German Conference on Pattern Recognition (GCPR), Springer, 2014, pp. 31-42.",
        "[6] J. R. Chang and Y. S. Chen, \"Pyramid stereo matching network,\" in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 4106-4115.",
        "[7] A. Kendall, H. Martirosyan, S. Dasgupta, P. Henry, R. Kennedy, A. Bachrach, and P. Bry, \"End-to-end learning of geometry and context for deep stereo regression,\" in Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017, pp. 66-75.",
        "[8] S. T. Birchfield and C. Tomasi, \"A pixel dissimilarity measure that is insensitive to image sampling,\" IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 20, no. 4, pp. 401-406, 1998.",
        "[9] N. Mayer, E. Ilg, P. Häusser, P. Fischer, D. Cremers, A. Dosovitskiy, and T. Brox, \"A large dataset to train convolutional networks for disparity, optical flow, and scene flow estimation,\" in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 4040-4048.",
        "[10] R. Hartley and A. Zisserman, Multiple View Geometry in Computer Vision, 2nd ed. Cambridge, U.K.: Cambridge University Press, 2004."
    ]
    for r in references:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.25)
        run = p.add_run(r)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(8.5)

    output_path = "Documents/CO5430_P12_Final_Report_IEEE.docx"
    doc.save(output_path)
    print(f"Report successfully saved to: {output_path}")

if __name__ == "__main__":
    build_document()
