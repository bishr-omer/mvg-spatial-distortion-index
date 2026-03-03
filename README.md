<div align="center">

# 🔭 MVG-Spa
### No-Reference Multivariate Gaussian-Based Spatial Distortion Index for Pansharpened Images

[![Paper](https://img.shields.io/badge/Paper-LNCS_2026-2196F3?style=flat-square&logo=readthedocs&logoColor=white)](#citation)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2024a+-orange?style=flat-square&logo=mathworks&logoColor=white)](https://www.mathworks.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-NBU_Pansharpening-purple?style=flat-square)](https://doi.org/10.1109/MGRS.2020.3008355)
[![Related](https://img.shields.io/badge/Related-MVG--SDI_(Spectral)-red?style=flat-square)](https://github.com/bishromer/MVG-SDI)

<br/>

**Bishr Omer Abdelrahman Adam · Xu Li\* · Yuchao Wang · Xinyan Yang**

*School of Electronics and Information, Northwestern Polytechnical University, Xi'an 710129, China*

✉ lixu@nwpu.edu.cn

<br/>

> **TL;DR** — MVG-Spa is the spatial counterpart to [MVG-SDI](https://github.com/bishromer/MVG-SDI). It assesses **spatial** distortions (blur, blocking, ghosting, misregistration) in pansharpened images using a 48D feature set — Log-Gabor + LBP + edge + statistical features — fitted to a Multivariate Gaussian model and compared via Mahalanobis distance against the reference PAN image. No ground-truth HR MS required.

</div>

---

## 🧭 Context: The MVG Quality Assessment Suite

This repo is **Part 2** of a two-part framework for complete, decoupled pansharpening quality assessment:

| Repo | Distortion Type | Reference | Features |
|---|---|---|---|
| [**MVG-SDI**](https://github.com/bishromer/MVG-SDI) | Spectral (colour shifts, radiometric errors) | MS image | FDD (Benford) + Color Moments · 21D |
| **MVG-Spa** *(this repo)* | Spatial (blur, blocking, ghosting, misregistration) | PAN image | Log-Gabor + LBP + Edge + Stats · 48D |

Together they form **MVG-QNR** — a holistic NR quality framework that avoids the spectral–spatial coupling that plagues QNR and its variants.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Method](#-method)
- [Quick Start](#-quick-start)
- [Dataset](#-dataset)
- [Results](#-results)
- [Degradation Robustness](#-degradation-robustness)
- [Citation](#-citation)

---

## 🔭 Overview

Pansharpened images suffer from spatial artifacts that standard metrics often miss:

| Artifact | Cause | Visual Effect |
|---|---|---|
| **Blur** | MTF mismatch in MRA low-pass filter | Loss of texture and sharpness |
| **Blocking** | Nearest-neighbour upsampling | Grid-like / zigzag patterns |
| **Ghosting** | Time lag between PAN & MS capture | Faint duplicate of moving objects |
| **Misregistration** | Sensor/terrain/timing displacement | Misplaced features, double edges |

Existing NR metrics (QNRs, FQNRs, MQNRs) fail to rank these distortions monotonically. MVG-Spa detects all four using a spatially-aware feature set within a statistical MVG framework.

---

## 🔧 Method

### Pipeline

```
PAN Image  ──► 32×32 Patches ──► Spatial Features (48D) ──► MVG Model (μ_PAN, Ψ_PAN)
                                                                         │
Fused Band ──► 32×32 Patches ──► Spatial Features (48D) ──► MVG Model (μ_i,   Ψ_i)
                                                                         │
                                               Mahalanobis Distance D_i ─┘
                                                                         │
                                    Quality = mean(D_1, D_2, …, D_Nb) ◄─┘
```

### Feature Extraction — 48 Dimensions

**Log-Gabor Features (24D)**

Multi-scale, multi-orientation texture decomposition — the primary detector for blur and ghosting.

$$G_r(\rho) = \exp\!\left(-\frac{(\log(\rho/f_0))^2}{2\sigma_f^2}\right), \qquad G_a(\theta) = \exp\!\left(-\frac{(\Delta\theta)^2}{2\sigma_\theta^2}\right)$$

Configuration: 2 scales `[1, 0.25]` × 4 orientations `[0, π/4, π/2, 3π/4]` → 8 filters × 3 stats (μ, σ, energy) = **24 features**

**LBP Features (15D)**

Circular LBP with 4 neighbours at radius 2 captures micro-patterns and structural artifacts like blocking.

$$LBP = \sum_{p=0}^{P-1} s(g_p - g_c) \cdot 2^p, \qquad s(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$$

**Basic Statistical Features (5D)**

Per-patch descriptors: $[\mu,\; \sigma,\; S,\; K,\; H]$ — mean, std, skewness, kurtosis, entropy.

**Edge Features (4D)**

Canny edge detector → $[\mu_e,\; \sigma_e,\; S_e,\; K_e]$ — statistics of edge magnitudes, sensitive to sharpness loss.

**Combined vector per patch:**
$$\mathbf{x} = [\mathbf{x}_\text{LogGabor},\; \mathbf{x}_\text{LBP},\; \mathbf{x}_\text{Stats},\; \mathbf{x}_\text{Edge}] \in \mathbb{R}^{48}$$

### Score Computation

Per-band Mahalanobis distance against the PAN reference MVG model:

$$\boxed{D_i = \sqrt{(\mu_\text{PAN} - \mu_{\text{fused},i})^\top \;\Psi_\text{pooled,i}^{-1}\; (\mu_\text{PAN} - \mu_{\text{fused},i})}, \qquad \text{quality} = \frac{1}{N_b}\sum_{i=1}^{N_b} D_i}$$

> **Lower score = better spatial fidelity.**

### Implementation Parameters

| Parameter | Value | Description |
|---|---|---|
| Patch size | 32 × 32 | Non-overlapping blocks |
| Log-Gabor scales | `[1, 0.25]` | Coarse + fine texture |
| Log-Gabor orientations | `[0, π/4, π/2, 3π/4]` | 4-direction coverage |
| σ_f (bandwidth) | 0.65 | Radial frequency bandwidth |
| LBP neighbours | 4 | At radius 2 px |
| **Total feature dim** | **48** | Per patch |

---

## 🚀 Quick Start

```matlab
% Add to MATLAB path
addpath('matlab');

% Load images (double)
PAN   = double(imread('pan_image.tif'));    % H×W, single band
Fused = double(imread('fused_image.tif')); % H×W×B, multi-band

% Compute MVG-Spa  —  lower score = less spatial distortion
[quality, band_scores] = MQNR_Spa(PAN, Fused);

fprintf('MVG-Spa (overall): %.4f\n', quality);
fprintf('Per-band scores:   ');
fprintf('%.4f  ', band_scores);
fprintf('\n');
```

### Run on Full NBU Sensor Subset

```matlab
sensors = {'IK', 'WV2', 'QB'};
methods = {'BDSD', 'GS', 'GSA', 'AWLP', 'MTF-GLP', 'FE-HPM', 'PWMBF', 'TV'};

for s = 1:numel(sensors)
    for m = 1:numel(methods)
        PAN   = load_pan(sensors{s});
        Fused = load_fused(sensors{s}, methods{m});
        [q, ~] = MQNR_Spa(PAN, Fused);
        fprintf('%s / %s : %.4f\n', sensors{s}, methods{m}, q);
    end
end
```

---

## 🗄️ Dataset

Experiments use the **[NBU Pansharpening Benchmark](https://doi.org/10.1109/MGRS.2020.3008355)** (Meng et al., IEEE GRSM 2021).

| Sensor | Image Pairs | PAN Size | MS Size | MS Bands |
|---|:---:|---|---|:---:|
| IKONOS (IK) | 200 | 1024 × 1024 | 256 × 256 | 4 |
| WorldView-2 (WV-2) | 500 | 1024 × 1024 | 256 × 256 | 8 |
| QuickBird (QB) | 500 | 1024 × 1024 | 256 × 256 | 4 |
| **Total** | **1 200** | | | |

Download: [NBU official GitHub](https://github.com/xingxingmeng/NBU-Dataset)

---

## 📊 Results

10 pansharpening algorithms evaluated: 3 CS, 4 MRA, 3 VO.

### IK Dataset

| Type | Method | SSIM ↑ | sCC ↑ | QNRs ↓ | FQNRs ↓ | RQNRs ↓ | MQNRs | **Ours ↓** |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CS | BDSD | 0.6095 | 0.8931 | 0.0777 | 0.0719 | 0.0123 | 1.8833 | 0.4949 |
| CS | GS | 0.6176 | 0.8753 | 0.1025 | 0.0796 | 0.0207 | 1.3249 | 0.4098 |
| CS | GSA | 0.6412 | 0.8863 | 0.1164 | 0.0562 | 0.0017 | 2.0698 | 0.7447 |
| MRA | AWLP | 0.6284 | 0.8933 | 0.1234 | 0.0595 | 0.0345 | 2.3447 | 0.2713 |
| MRA | MTF-GLP | 0.6232 | 0.8899 | 0.1372 | 0.0608 | 0.0375 | 2.2723 | 0.3044 |
| MRA | MTF-GLP-CBD | 0.6376 | 0.8861 | 0.1111 | 0.0463 | 0.0247 | 2.5327 | 0.8762 |
| VO | FE-HPM | **0.6795** | **0.9007** | 0.1179 | 0.0524 | 0.0250 | 3.2963 | 0.5524 |
| VO | TV | 0.6593 | 0.9012 | 0.9199 | 0.9103 | 0.9401 | 2.2413 | 1.8092 |

### WV-2 Dataset

| Type | Method | SSIM ↑ | sCC ↑ | QNRs ↓ | FQNRs ↓ | RQNRs ↓ | MQNRs | **Ours ↓** |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CS | BDSD | 0.8755 | 0.9150 | 0.1284 | 0.0382 | 0.0918 | 3.3933 | 1.0167 |
| CS | GS | 0.8711 | 0.9088 | 0.0444 | 0.0738 | 5.1e-04 | 1.9270 | 0.4215 |
| CS | **GSA** | 0.8684 | 0.9165 | 0.0477 | 0.0479 | **2.8e-04** | 1.7887 | **0.1508** |
| MRA | MTF-GLP | **0.8729** | **0.9289** | 0.0394 | 0.0268 | 0.0444 | 1.9453 | 0.3881 |
| VO | TV | 0.8809 | 0.9279 | 0.9012 | 0.9321 | 0.9110 | 2.6667 | 0.7795 |

> MVG-Spa agrees with RQNRs on the best performer (GSA) for WV-2 — and ranks all distortion severities monotonically, which QNRs/FQNRs fail to do.

---

## 🧪 Degradation Robustness

MVG-Spa is tested against four simulated spatial degradations on the QB dataset. A robust metric must rank severities **monotonically** (score increases as distortion worsens).

| Degradation | Simulation | QNRs | FQNRs | RQNRs | MQNRs | **Ours** |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Blur** | Gaussian σ = 0.5→1.5 | ❌ | ⚠️ | ✅ | ⚠️ | ✅ |
| **Blocking** | Block size 1→3 px | ❌ | ⚠️ | ✅ | ⚠️ | ✅ |
| **Misregistration** | Shift 1→3 px | ⚠️ | ❌ | ✅ | ➖ | ✅ |
| **Ghosting** | α = 1.0→3.0 | ⚠️ | ❌ | ✅ | ⚠️ | ✅ |

✅ Monotonic · ⚠️ Partially monotonic · ❌ Fails · ➖ Flat/insensitive

### QB / GS Method — Detailed

| Condition | Param | QNRs | FQNRs | RQNRs | MQNRs | **Ours** |
|---|---|:---:|:---:|:---:|:---:|:---:|
| Blur | σ=0.5 | 0.0661 | 0.0664 | 0.0457 | 1.8465 | 1.0169 |
| Blur | σ=1.0 | 0.0501 | 0.0977 | 0.1553 | 2.1896 | 2.2489 |
| Blur | σ=1.5 | 0.1098 | 0.1743 | 0.2296 | 2.4763 | 2.4992 |
| Blocking | 1 px | 0.1025 | 0.0796 | 0.0207 | 1.6566 | 0.3186 |
| Blocking | 2 px | 0.0388 | 0.0701 | 0.1652 | 2.7592 | 1.1927 |
| Blocking | 3 px | 0.1021 | 0.1601 | 0.2560 | 3.0750 | **1.8904** |
| Ghost | α=1.0 | 0.5354 | 0.2243 | 0.9860 | 1.5542 | 0.4615 |
| Ghost | α=2.0 | 0.8603 | 0.1497 | 1.4686 | 1.5922 | 0.7912 |
| Ghost | α=3.0 | 0.8850 | 0.1684 | 2.0535 | 2.7571 | **1.0481** |

---

## ⚠️ Limitations

- **Spatial only** — does not measure spectral fidelity. Pair with [MVG-SDI](https://github.com/bishromer/MVG-SDI) for complete assessment.
- Higher computational cost than simpler NR metrics due to Log-Gabor filterbank and LBP extraction (persistent filterbank caching mitigates repeated calls).
- Validated on IK, WV-2, and QB sensors; additional sensors (WV-3/4) covered in the companion spectral repo.

---

## 🙏 Acknowledgements

Supported by the Practice and Innovation Funds for Graduate Students of Northwestern Polytechnical University, the Key R&D Program of Shaanxi Province (No. 2025CY-YBXM-079), and the National College Students' Innovation and Entrepreneurship Training Program (No. 202410699209).

Fusion implementations from the [PanCollection](https://github.com/liangjiandeng/PanCollection) MATLAB toolkit.

---

## 📎 Citation

```bibtex
@inproceedings{adam2026mvgspa,
  title     = {Multivariate Gaussian-Based No-Reference Quality Assessment
               for Spatial Distortion in Pansharpened Images},
  author    = {Adam, Bishr Omer Abdelrahman and Li, Xu and
               Wang, Yuchao and Yang, Xinyan},
  booktitle = {Lecture Notes in Computer Science},
  year      = {2026},
  publisher = {Springer}
}
```

If you also use the spectral index, please cite [MVG-SDI](https://github.com/bishromer/MVG-SDI):

```bibtex
@article{adam2026mvgsdi,
  title   = {A No-Reference Multivariate Gaussian-Based Spectral Distortion
             Index for Pansharpened Images},
  author  = {Adam, Bishr Omer Abdelrahman and Li, Xu and
             Wu, Jingying and Hao, Xiankun},
  journal = {Sensors},
  volume  = {26}, number = {3}, pages = {1002}, year = {2026},
  doi     = {10.3390/s26031002}
}
```

---

<div align="center">
<sub>© 2026 Adam et al. · Northwestern Polytechnical University · Part of the MVG pansharpening quality assessment suite</sub>
</div>
