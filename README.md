# Privacy-Preserving Face Anonymization with Utility Retention using Adversarial Perturbations

## FASP, PrIdentity, and PRISM for Privacy-Preserving AI Cloaking

**Purav Shah (202518020)**
**Rajvi Bujad (202518048)**

*Under the guidance of Shruti Bhilare*

*Department of Data Science*
*Dhirubhai Ambani University, Gujarat, India*

---

## Abstract

Face recognition systems operating on social media images, surveillance footage, and public datasets raise fundamental privacy concerns. These systems extract identity embeddings that remain stable even when images appear visually innocuous to human observers. The core challenge addressed in this work is the mismatch between human visual perception and machine-extractable identity signatures. Existing anonymization methods face an inherent tradeoff: strong perturbations distort images beyond practical usability, while weak perturbations fail to suppress identity information effectively in embedding space.

This repository presents a comprehensive study of adversarial perturbations for privacy-preserving face anonymization. The primary contribution is **FASP (Frequency-Aware Structured Perturbation)**, a method that generates structured frequency-domain perturbations using Laplacian-based edge-aware masking and multi-frequency sinusoidal synthesis. Unlike pixel-domain approaches that allocate perturbation budget uniformly, FASP concentrates adversarial energy in high-frequency contour regions where recognition models extract discriminative features and where human perception is least sensitive.

The framework is systematically evaluated against **PrIdentity**, an adaptive Lp norm baseline representing state-of-the-art pixel-domain anonymization, and **PRISM**, an exploratory wavelet-based Riemannian geometry formulation. Experimental characterization spans LFW, CelebA, and CelebA-HQ datasets with evaluation against ArcFace, FaceNet/CASIA-WebFace, VGGFace2/ResNet-50, LightCNN-29, and MobileFace recognition architectures. Results demonstrate that FASP achieves complete Rank-1 anonymization (100% across gallery sizes 1 through 4), structural similarity up to 0.9238, and bounding-box centroid displacement of 1.55 pixels on CelebA-HQ, representing the best geometric preservation among all compared methods.

---

## Table of Contents

1. [Introduction and Problem Formulation](#1-introduction-and-problem-formulation)
2. [Motivation and Design Rationale](#2-motivation-and-design-rationale)
3. [Literature Review and Research Gap](#3-literature-review-and-research-gap)
4. [Baseline Method: PrIdentity](#4-baseline-method-pridentity)
5. [Proposed Method: FASP](#5-proposed-method-fasp)
6. [Experimental Framework: PRISM](#6-experimental-framework-prism)
7. [Experimental Setup and Configuration](#7-experimental-setup-and-configuration)
8. [Results and Analysis](#8-results-and-analysis)
9. [Comparative Analysis Across Methods](#9-comparative-analysis-across-methods)
10. [Discussion and Interpretation](#10-discussion-and-interpretation)
11. [Reproducibility and Resources](#11-reproducibility-and-resources)
12. [Repository Structure](#12-repository-structure)
13. [Conclusion and Future Work](#13-conclusion-and-future-work)
14. [References](#14-references)

---

## 1. Introduction and Problem Formulation

### 1.1 The Face Recognition Privacy Problem

Face recognition technology has transitioned from specialized biometric applications to default infrastructure components in consumer devices, social media platforms, content moderation systems, and surveillance networks. Modern recognition architectures, including ArcFace, FaceNet, and VGGFace2, encode facial identity into high-dimensional embeddings through deep convolutional networks trained with margin-based or triplet loss formulations. These embeddings exhibit remarkable robustness to natural variations in pose, illumination, expression, and partial occlusion.

The privacy vulnerability arises from a fundamental perceptual asymmetry. An image that appears visually ordinary to a human observer can still generate a highly discriminative embedding vector when processed by a recognition model. The identity information is distributed across spatial frequencies in ways that do not align with human visual saliency. This means that cosmetic redaction techniques such as Gaussian blurring, pixelation, or masking can render a face unrecognizable to humans while leaving the underlying embedding largely intact, particularly when the recognition pipeline includes preprocessing steps like super-resolution or deblurring.

### 1.2 Formal Problem Statement

Given an input face image $I \in \mathbb{R}^{H \times W \times 3}$ normalized to $[0,1]$, and a set of $K$ recognition models $\{f_k\}_{k=1}^{K}$ where each $f_k: \mathbb{R}^{H \times W \times 3} \rightarrow \mathbb{R}^d$ maps images to normalized $d$-dimensional embedding vectors, the anonymization problem seeks a perturbation $P \in \mathbb{R}^{H \times W \times 3}$ such that the anonymized image $A = I + P$ satisfies two competing objectives:

**Privacy Objective.** The embedding similarity between $I$ and $A$ must be reduced below a threshold $\tau$:
$$\cos(z_I^{(k)}, z_A^{(k)}) < \tau_k \quad \forall k \in \{1,\ldots,K\}$$

where $z_I^{(k)} = f_k(I)$, $z_A^{(k)} = f_k(A)$, and $\cos(u,v) = \frac{u^T v}{\|u\|_2 \|v\|_2}$.

**Utility Objective.** The anonymized image $A$ must remain perceptually similar to $I$ under a chosen visual metric $\mathcal{D}$:
$$\mathcal{D}(I, A) < \epsilon_u$$

### 1.3 The Privacy-Utility Optimization

This dual objective is formalized as a constrained optimization problem:

$$\min_{P} \; \mathcal{L}_{\text{priv}}(I, I+P) + \lambda \, \mathcal{L}_{\text{util}}(I, I+P)$$

subject to $\|P\|_{\infty} \leq \epsilon$, where $\lambda \in \mathbb{R}^+$ governs the privacy-utility tradeoff. The perturbation budget $\epsilon$ prevents unbounded distortion, and the pixel values of $A$ are clipped to $[0,1]$.

The fundamental challenge is that identity-suppressing perturbations and perceptually acceptable perturbations occupy different regions of the perturbation space. Random or uniform perturbations waste budget on perceptually salient regions while under-perturbing identity-critical features. The key insight motivating this work is that **the structure of the perturbation matters as much as its magnitude**.

---

## 2. Motivation and Design Rationale

### 2.1 Why Pixel-Domain Methods Are Insufficient

Pixel-domain adversarial cloaking methods, including PrIdentity's adaptive Lp formulation, treat perturbation generation as an unconstrained search over $\mathbb{R}^{H \times W \times 3}$ with norm-based regularization. This approach has three fundamental limitations:

**Spatial blindness.** The Lp norm penalty applies uniformly across all spatial locations, penalizing perturbations equally in smooth facial regions (cheeks, forehead) and texture-rich regions (eye contours, lip boundaries). Since recognition models extract disproportionate discriminative information from the latter, uniform regularization wastes perturbation budget.

**Frequency blindness.** Lp regularization in the spatial domain provides no explicit control over the spectral content of the perturbation. Low-frequency perturbations that alter coarse facial geometry are perceptually obvious, while high-frequency perturbations that disrupt fine texture are more easily tolerated by human vision. A purely spatial norm cannot differentiate between these cases.

**Limited interpretability.** Free-form pixel optimization provides no structural prior about what constitutes an effective perturbation. The resulting noise patterns are difficult to analyze, debug, or characterize in terms of their frequency-domain properties.

### 2.2 The Frequency-Aware Hypothesis

FASP is built on three empirically motivated observations about the interaction between face recognition models and the human visual system:

**Observation 1: Discriminative feature concentration.** Face recognition models trained with identity contrastive losses develop feature hierarchies that are disproportionately sensitive to mid-to-high spatial frequencies. Sharp edges, fine textures, and contour transitions carry more identity-discriminative information than broad, smooth intensity gradients. This can be understood through the Fourier domain: the embedding function $f$ exhibits larger gradient magnitudes with respect to high-frequency input components.

**Observation 2: Perceptual asymmetry.** The human visual system exhibits contrast sensitivity that peaks at mid-range spatial frequencies and falls off at both very low and very high frequencies. Furthermore, masking effects make perturbations less noticeable when superimposed on existing image structure. Perturbations injected into edge regions and texture boundaries are partially masked by the underlying content, making them less perceptible than equivalent-energy perturbations in smooth regions.

**Observation 3: Structural transferability.** Adversarial perturbations that are grounded in explicit spatial or spectral priors tend to transfer more effectively across model architectures than unstructured noise. When the attack is constrained to a meaningful subspace, it is less likely to overfit to idiosyncratic features of the surrogate model and more likely to exploit general properties of the recognition task.

### 2.3 FASP Design Philosophy

These observations motivate a perturbation generation strategy that:

1. Concentrates perturbation energy in high-frequency spatial regions identified by edge detection,
2. Structures the perturbation using multi-frequency sinusoidal carriers rather than random initialization,
3. Applies spatial gating so that smooth regions remain largely unmodified,
4. Uses a perceptual loss function (LPIPS) alongside pixel-level MSE for utility preservation.

The result is a perturbation that is simultaneously more effective at suppressing identity and more difficult for human observers to detect, compared to equivalent-budget pixel-domain perturbations.

---

## 3. Literature Review and Research Gap

### 3.1 Taxonomy of Face Anonymization Approaches

The face anonymization literature can be organized into four methodological categories, each with characteristic strengths and limitations.

**Generative Synthesis Methods.** These approaches replace the detected face entirely with a synthetic face generated by a learned model. CIAGAN uses conditional GANs to substitute identity while preserving pose and expression. DeepPrivacy conditions generation on background context rather than explicit identity matching. RiDDLE operates in latent space with reversible de-identification. Diff-Privacy applies diffusion model priors for high-quality synthesis. While capable of producing visually convincing outputs, these methods share fundamental limitations: they alter non-identity attributes (lighting, accessories, background), introduce generative artifacts detectable by both humans and downstream systems, and incur computational costs prohibitive for real-time deployment.

**Adversarial Cloaking Methods.** These methods directly attack the recognition embedding through optimized perturbations. LowKey targets commercial social media filters with imperceptible pixel-space perturbations. TIP-IM introduces targeted identity masks to improve cross-model transferability. DIM and MT-DIM incorporate input diversity during optimization. PrIdentity represents the current state-of-the-art in this category, using adaptive Lp regularization that learns norm exponents from data, allowing the optimizer to shift between sparse ($p \approx 1$) and smooth ($p \approx 2$) perturbation geometries based on local image context.

**Frequency-Domain Methods.** While adversarial robustness literature has extensively characterized the spectral sensitivities of convolutional networks, relatively few anonymization methods explicitly operate in the frequency domain. Wavelet-based and DCT-domain attacks exist in the broader adversarial examples literature but have not been systematically applied to face anonymization with perceptual constraints.

**Hybrid and Emerging Methods.** Recent work explores combining generative and adversarial approaches, universal perturbation training, and diffusion-based editing. However, systematic integration of frequency-domain priors with perceptual utility constraints remains an open research direction.

### 3.2 Comparative Summary

| Method | Privacy Strength | Visual Quality | Generalization | Perturbation Type | Computational Cost |
|---|---:|---:|---:|---:|---|
| CIAGAN | Strong | Medium-High | Medium | GAN synthesis | High |
| DeepPrivacy | Medium | High | Medium | GAN-based | High |
| RiDDLE | Medium | Medium | Medium | Latent editing | Medium |
| Diff-Privacy | Medium | High | Medium | Diffusion editing | High |
| LowKey | High | High | Good | Adversarial perturbation | Medium |
| TIP-IM | High | High | Good | Transferable perturbation | Medium |
| PrIdentity | High | High | Good | Adaptive Lp pixel perturbation | Medium |
| **FASP (Ours)** | **High** | **High** | **Good-Strong** | **Structured frequency-domain** | **Medium** |

### 3.3 Identified Research Gap

The literature reveals a recurring compromise. Generative methods achieve strong visual quality but at high computational cost and with limited black-box transferability. Adversarial methods are more efficient and transferable but lack explicit mechanisms to distinguish between perceptually sensitive and insensitive image regions. PrIdentity partially addresses this through learned norm adaptation, but its operation remains fundamentally in the pixel domain with no spectral awareness.

**FASP addresses this gap by embedding explicit frequency-domain structure into both the spatial distribution and the spectral composition of the perturbation, bridging the divide between computational efficiency and perceptually-guided perturbation allocation.**

---

## 4. Baseline Method: PrIdentity

### 4.1 Method Overview

PrIdentity formulates face anonymization as a constrained optimization over the perturbation tensor, using an adaptive norm regularizer that learns the appropriate perturbation geometry from data rather than fixing it a priori. The method operates without explicit identity-source targets, making it suitable for unsupervised anonymization.

### 4.2 Architecture and Pipeline

The PrIdentity optimization pipeline consists of four stages: input preprocessing, iterative perturbation optimization with recognition feedback, adaptive norm projection, and output clamping.

![PrIdentity architecture](images/pridentity%20architecture.png)

*Figure 4.1: PrIdentity architecture and optimization loop. The pipeline processes an input face image through iterative optimization against a pre-trained recognition model, enforcing an adaptive Lp norm constraint at each step, and producing an anonymized output that suppresses identity while maintaining visual fidelity.*

### 4.3 Mathematical Formulation

**Perturbation Model.** Given input image $I$ and recognition model $f$, PrIdentity learns perturbation $P$ to produce anonymized output:
$$A = I + P$$

**Privacy Loss.** The privacy term uses a margin-based formulation over embedding distances:
$$\mathcal{L}_{\text{priv}} = \sum_i \max(0, \alpha - D(z_I, z_A))$$

where $z_I = f(I)$ and $z_A = f(A)$ are L2-normalized embedding vectors, $D(\cdot,\cdot)$ is a dissimilarity measure (typically Euclidean distance or cosine distance), and $\alpha$ is the target privacy margin. This formulation has the important property that the loss gradient vanishes when $D(z_I, z_A) > \alpha$, preventing unnecessary over-perturbation once the privacy target is achieved.

**Utility Loss with Adaptive Norm.** The defining innovation of PrIdentity is its adaptive norm regularizer:
$$\mathcal{L}_{\text{util}} = \left(\sum_j |P_j|^p + \epsilon\right)^{1/p}$$

where $P_j$ are individual perturbation components and $p$ is a learned norm exponent. The exponent $p$ is optimized alongside the perturbation, typically falling within $[1, 2]$. Values near $p \approx 1$ encourage sparse, sharply localized perturbations analogous to L1 attacks. Values near $p \approx 2$ produce smoother, more distributed modifications analogous to L2 attacks. In spatial form:
$$\mathcal{L}_{\text{reg}} = \|\delta\|_{p(x)} = \left(\sum_i |\delta_i|^{p(x)}\right)^{1/p(x)}$$

where $p(x)$ may vary spatially or be conditioned on image content.

**Total Objective:**
$$\min_P \; \mathcal{L}_{\text{priv}} + \lambda \, \mathcal{L}_{\text{util}}$$

### 4.4 Qualitative Results

![PrIdentity before-after comparison](images/pridentitycomparision%20of%20before%20after.png)

*Figure 4.2: PrIdentity qualitative results comparing original images (left column) against anonymized outputs (right column). The anonymized faces retain overall facial structure, expression, and color distribution while embedding-space identity cues are disrupted. No generative reconstruction or identity swapping is performed.*

### 4.5 Privacy-Utility Tradeoff

![PrIdentity tradeoff curve](images/pridentitycomparing%20tradeoff.png)

*Figure 4.3: PrIdentity privacy-utility tradeoff analysis. Increasing the perturbation budget reduces cosine similarity (stronger privacy) at the cost of decreasing SSIM (lower utility). The curve defines the Pareto frontier achievable by the adaptive Lp formulation.*

### 4.6 Quantitative Performance

**LFW Verification (TPR at FPR=0.001).** PrIdentity reduces verification rates dramatically:

| Model | Original | L1 Baseline | L2 Baseline | Lp (Proposed) |
|---|---:|---:|---:|---:|
| ArcFace | 0.9940 | 0.0077 | 0.0047 | **0.002** |
| LightCNN-29 | 0.9930 | 0.0127 | 0.0147 | **0.013** |

The adaptive Lp formulation achieves a 497x reduction in ArcFace TPR from 0.9940 to 0.002.

**CelebA Single-Image Anonymization (Rank-1 Identification Accuracy, Gallery Size 1):**

| Model | Original | L1 | L2 | Lp (Proposed) |
|---|---:|---:|---:|---:|
| VGGFace | 45.42 | 0.20 | 0.20 | **0.30** |
| ArcFace | 64.46 | 5.40 | 2.93 | **2.55** |
| LightCNN-29 | 83.91 | 18.04 | 10.60 | **9.57** |

**Data Utility (SSIM) for CelebA Single-Image Anonymization:**

| Gallery Size | L1 | L2 | Lp |
|---|---:|---:|---:|
| 1 | 0.9004 | 0.8629 | 0.8419 |
| 2 | 0.9001 | 0.8628 | 0.8422 |
| 3 | 0.9000 | 0.8628 | 0.8423 |
| 4 | 0.8999 | 0.8625 | 0.8414 |

**Geometric Preservation (Bounding Box Distance on CelebA-HQ, MTCNN detector):**

| Method | Bounding Box Distance |
|---|---:|
| DeepPrivacy | 4.65 |
| CIAGAN | 20.38 |
| FIT | 7.87 |
| RiDDLE | 3.82 |
| FALCO | 7.88 |
| Diff-Privacy | 5.83 |
| PrIdentity (Lp) | **2.65** |

### 4.7 Strengths and Limitations

**Strengths.** PrIdentity demonstrates that adaptive norm learning provides meaningful advantages over fixed L1 or L2 regularization. The method achieves strong white-box privacy with SSIM above 0.84, works without target identities, and provides a reproducible baseline with clear optimization dynamics.

**Limitations.** Three limitations motivate the FASP extension. First, the adaptive norm operates purely in the pixel domain and cannot distinguish between frequency bands. Second, the regularization penalty is spatially uniform, applying equal pressure against perturbations in smooth regions and edge regions. Third, the cross-architecture transferability, while improved over fixed-norm baselines, remains inconsistent, particularly for models with fundamentally different loss landscapes (e.g., margin-based ArcFace versus triplet-based FaceNet).

---

## 5. Proposed Method: FASP

### 5.1 Method Overview

FASP (Frequency-Aware Structured Perturbation) reformulates face anonymization as a frequency-domain perturbation synthesis problem. Rather than optimizing a free-form pixel perturbation, FASP generates perturbations from a constrained subspace defined by two structural priors: a Laplacian edge attention map that gates perturbation energy spatially, and a multi-frequency sinusoidal carrier that structures perturbation energy spectrally.

### 5.2 Complete Architecture Pipeline

![FASP architecture](images/fasp_architecture.png)

*Figure 5.1: FASP pipeline architecture. The system processes an input face through five stages: (1) Laplacian edge detection for frequency mask extraction, (2) multi-frequency sinusoidal carrier synthesis, (3) Hadamard product fusion of mask and carrier, (4) perturbation scaling and addition, and (5) iterative optimization against a privacy-utility objective evaluated through a pre-trained recognition model.*

The pipeline is fully differentiable, allowing gradient-based optimization of the sinusoidal parameters $\{A_f, \phi_f\}_{f \in \mathcal{F}}$ with respect to the combined privacy-utility loss.

### 5.3 Component Visualization

#### 5.3.1 Frequency Masking Strategy

![FASP frequency mask comparison](images/FASP_frequency%20mask%20comparision.png)

*Figure 5.2: Comparison of frequency masking strategies. Left: FASP edge-aware Laplacian mask concentrating perturbation budget on contours and texture transitions. Middle: uniform mask for reference, distributing budget without spatial selectivity. Right: random mask, wasting budget on uninformative locations. The Laplacian approach ensures perturbation energy is allocated where recognition models extract discriminative features.*

#### 5.3.2 Sinusoidal Perturbation Generation

![FASP sinusoidal initialization](images/FASP_sinusoidal%20initialization.png)

*Figure 5.3: Multi-frequency sinusoidal carrier patterns generated by the FASP synthesis module prior to edge-mask application. Each row corresponds to a different frequency component. The structured oscillatory patterns provide a controllable, spectrally interpretable perturbation prior that avoids the chaotic frequency distribution of random noise initialization.*

### 5.4 Mathematical Formulation

#### 5.4.1 Frequency Mask Extraction

The Laplacian edge attention map is computed via convolution with the discrete Laplacian kernel:

$$M_L(x, y) = |\nabla^2 I(x, y)|$$

where the discrete Laplacian operator $\nabla^2$ is implemented as convolution with:

$$K_{\nabla^2} = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$

This second-order spatial derivative detects intensity curvature rather than simple gradient magnitude. Smooth regions (cheeks, forehead, background) produce near-zero response because intensity varies approximately linearly. Edge regions, contour transitions, and texture boundaries produce strong responses because intensity curvature is high.

The raw Laplacian response is normalized to produce an edge attention map:

$$A_{\text{edge}}(x, y) = \frac{M_L(x, y)}{\max_{x',y'} M_L(x', y') + \delta}$$

where $\delta > 0$ prevents division by zero. The normalized map satisfies $A_{\text{edge}}(x,y) \in [0,1]$ with larger values indicating regions where FASP concentrates perturbation energy.

**Why the Laplacian specifically.** The Laplacian is rotationally invariant and detects edges regardless of orientation. Unlike directional gradient operators (Sobel, Prewitt), it responds equally to horizontal, vertical, and diagonal edges. This is important because facial features contain edges at all orientations. The second-order nature also means it responds to the rate of intensity change rather than the magnitude of change, making it sensitive to fine texture transitions that first-derivative operators might miss.

#### 5.4.2 Multi-Frequency Sinusoidal Synthesis

Instead of initializing perturbations with random noise $\delta \sim \mathcal{N}(0, \sigma^2 I)$ or $\delta \sim \mathcal{U}(-\epsilon, \epsilon)$, FASP generates a structured carrier signal:

$$S(x, y) = \sum_{f \in \mathcal{F}} A_f \sin(2\pi f \, r(x,y) + \phi_f)$$

where:
$$r(x,y) = \sqrt{(x - x_0)^2 + (y - y_0)^2}$$

is the radial coordinate measured from the patch center $(x_0, y_0)$, $\mathcal{F}$ is the chosen set of spatial frequencies (typically 2-5 frequencies spanning low, medium, and high bands), $A_f \in \mathbb{R}^+$ is the amplitude for frequency $f$, and $\phi_f \in [0, 2\pi)$ is the phase offset.

**Radial parameterization rationale.** The radial coordinate $r(x,y)$ produces concentric oscillatory patterns that naturally follow the approximately circular geometry of facial features. Facial contours (eye sockets, lip boundaries, jawline) exhibit radial structure, making this parameterization more aligned with facial geometry than Cartesian sinusoidal grids.

**Multi-frequency design rationale.** Using multiple frequencies serves three purposes. Low-frequency components ($f \approx 1-2$ cycles per patch) help the perturbation survive image compression and resizing operations that attenuate high frequencies. Mid-frequency components ($f \approx 3-5$ cycles per patch) target the spatial scales at which most recognition features operate. High-frequency components ($f \approx 6-10$ cycles per patch) directly attack fine texture features. The combination creates a multi-scale perturbation that is harder for recognition models to filter out.

**Comparison to random initialization.** Random noise $\delta \sim \mathcal{N}(0, \sigma^2 I)$ has a flat power spectrum, distributing energy equally across all frequencies. In contrast, the sinusoidal carrier concentrates energy at specific, controllable frequencies. This makes the perturbation more interpretable, more parameter-efficient (the optimizer learns amplitudes and phases rather than individual pixels), and more likely to transfer across models because it exploits general frequency sensitivities rather than surrogate-specific artifacts.

#### 5.4.3 Structured Perturbation Synthesis via Hadamard Product

The edge attention map and sinusoidal carrier are fused through element-wise multiplication:

$$P_{\text{raw}}(x, y) = A_{\text{edge}}(x, y) \odot S(x, y)$$

where $\odot$ denotes the Hadamard (element-wise) product. This operation serves as spatial gating: the sinusoidal oscillations are expressed only where the Laplacian mask permits. In smooth facial regions where $A_{\text{edge}}(x,y) \approx 0$, the perturbation is effectively zero, preserving skin texture and broad luminance gradients. In edge regions where $A_{\text{edge}}(x,y) \approx 1$, the full sinusoidal pattern is applied.

**Why the Hadamard product is the correct fusion operator.** Alternatives like addition ($A_{\text{edge}} + S$) or convolution would not achieve the desired gating behavior. Addition would produce non-zero perturbations everywhere. Convolution would spatially mix the mask and carrier in ways that lose the precise spatial localization. The Hadamard product preserves both the spatial selectivity of the mask and the oscillatory structure of the carrier while being computationally trivial and fully differentiable.

#### 5.4.4 Perturbation Scaling and Clipping

The raw perturbation is scaled to respect the perturbation budget:

$$P = \epsilon \cdot \text{clip}(P_{\text{raw}}, -1, 1)$$

where $\epsilon$ controls the maximum pixel-space perturbation magnitude (typically $\epsilon \in [8/255, 16/255]$ for normalized images). The clipping operation prevents any single pixel perturbation component from exceeding the budget, ensuring the perturbation remains within the linear regime of the recognition model.

The anonymized image is formed by addition and range clipping:

$$A = \text{clip}(I + P, 0, 1)$$

The clipping to $[0,1]$ ensures $A$ remains a valid image tensor at every optimization step, preventing gradient artifacts from out-of-range pixel values.

#### 5.4.5 Privacy Loss with Softplus Margin

FASP employs a softplus-smoothed margin loss over cosine similarity:

$$\mathcal{L}_{\text{priv}} = \sum_{k=1}^{K} \text{softplus}\left(\kappa \left(\cos(z_I^{(k)}, z_A^{(k)}) - \tau\right)\right)$$

where $\text{softplus}(u) = \ln(1 + e^u)$, $z_I^{(k)} = f_k(I)$ and $z_A^{(k)} = f_k(A)$ are L2-normalized embeddings from recognition model $k$, $\cos(u,v) = \frac{u^T v}{\|u\|_2 \|v\|_2}$ is cosine similarity, $\tau \in [0,1]$ is the target similarity threshold below which anonymization is considered successful, and $\kappa > 0$ controls the margin sharpness.

**Behavior analysis.** When $\cos(z_I, z_A) \ll \tau$, the argument to softplus is strongly negative, yielding $\text{softplus}(\text{large negative}) \approx 0$. The loss saturates and provides negligible gradient. When $\cos(z_I, z_A) > \tau$, the argument is positive, yielding $\text{softplus}(\text{positive}) \approx \text{positive}$. The loss grows approximately linearly with the margin violation.

**Why softplus rather than hinge loss.** The softplus function is everywhere differentiable with continuous gradients, unlike the hinge function $\max(0, \cdot)$ which has a non-differentiable point at zero. This smoothness improves optimization stability, particularly when using Adam or other adaptive gradient methods. The parameter $\kappa$ provides explicit control over the transition sharpness: larger $\kappa$ produces a sharper transition approximating the hinge, while smaller $\kappa$ produces a smoother transition.

#### 5.4.6 Perceptual Utility Loss

The utility term combines learned perceptual similarity and pixel-level fidelity:

$$\mathcal{L}_{\text{util}} = \text{LPIPS}(I, A) + \lambda_{\text{mse}} \, \text{MSE}(I, A)$$

where LPIPS (Learned Perceptual Image Patch Similarity) measures distance in the feature space of a pre-trained deep network (AlexNet backbone), providing a metric that correlates better with human perceptual judgments than pixel-space metrics alone. The MSE term serves as a secondary constraint, preventing unbounded perturbation growth in regions where the LPIPS network has low sensitivity.

**Why LPIPS rather than SSIM in the loss.** SSIM is a valuable evaluation metric but is not well-suited as an optimization objective because its gradient can be unstable, particularly near structural discontinuities. LPIPS provides smooth, well-behaved gradients through the deep feature network while better capturing perceptual changes that humans actually notice.

#### 5.4.7 Total Optimization Objective

The complete FASP objective is:

$$\mathcal{L} = \mathcal{L}_{\text{priv}} + \lambda_u \, \mathcal{L}_{\text{util}}$$

where $\lambda_u > 0$ controls the global privacy-utility tradeoff. The optimization is performed over the sinusoidal parameters:

$$\min_{\{A_f, \phi_f\}_{f \in \mathcal{F}}} \; \mathcal{L}(I, I + P(\{A_f, \phi_f\}))$$

**Dimensionality comparison.** A free-form perturbation over a $112 \times 112 \times 3$ image has $37,632$ degrees of freedom. FASP with 5 frequencies has $10$ parameters (5 amplitudes and 5 phases). This thousand-fold reduction in search space dimensionality acts as a strong regularizer, biasing the solution toward structured, interpretable perturbations and reducing the risk of overfitting to surrogate-specific artifacts.

### 5.5 Qualitative Results

**Key qualitative observations from the visual comparison.**

The anonymized outputs maintain the macroscopic visual characteristics that make faces interpretable to humans: face oval shape, feature positions, overall lighting direction, skin tone, and expression. The perturbation is distinctly non-uniform. Changes concentrate in regions the Laplacian mask identifies as edge-rich: eye contours, eyebrow boundaries, lip edges, nasolabial folds, and hairline transitions. The smooth cheek and forehead surfaces remain largely unmodified.

Unlike GAN-based anonymization, FASP does not alter accessories (glasses, jewelry), hairstyle, or background elements. Unlike uniform adversarial attacks, it produces no visible grid patterns or blocking artifacts. The perturbation blends with existing image structure, exploiting perceptual masking to remain inconspicuous.

![Before and after comparison across datasets](images/comparision%20of%20before%20after.png)

*Figure 5.5: Extended qualitative comparison across dataset diversity. The top row shows original images from LFW (natural, unconstrained), CelebA (varied lighting and pose), and CelebA-HQ (high resolution). The bottom row shows corresponding FASP outputs. The method maintains consistency across different image characteristics and quality levels.*

---

## 6. Experimental Framework: PRISM

### 6.1 Method Overview

PRISM (Perturbation via Riemannian geometry and Implicit manifold Sampling) is included as an exploratory extension investigating whether explicit geometric modeling of the recognition embedding manifold can provide alternative or complementary benefits to FASP's frequency-domain approach. While not the primary contribution, PRISM's behavior provides valuable context for understanding the relative merits of geometric versus spectral perturbation priors.

### 6.2 Architecture and Pipeline

![PRISM architecture](images/prism%20architecture.jpeg)

*Figure 6.1: PRISM pipeline architecture. The method decomposes input images into wavelet sub-bands (LL, LH, HL, HH), applies per-band perturbations with learned regularization, computes a Fisher-information-based privacy loss on the recognition embedding manifold, and employs Jacobian subspace regularization to shape perturbation directions.*

### 6.3 Mathematical Formulation

**Wavelet Decomposition.** The input image $I$ is decomposed using the discrete wavelet transform (DWT) into four sub-bands: LL (low-low, approximation coefficients), LH (low-high, horizontal detail), HL (high-low, vertical detail), and HH (high-high, diagonal detail). Per-band perturbations $P_{\text{LL}}, P_{\text{LH}}, P_{\text{HL}}, P_{\text{HH}}$ are learned independently, allowing frequency-specific perturbation allocation.

**Fisher Information Privacy Loss.** PRISM formulates privacy loss using the Fisher information metric on the recognition model's output manifold:

$$\mathcal{L}_{\text{priv}}^{\text{PRISM}} = \mathbb{E}_x\left[ z_I^T \mathbf{F}(x) z_A \right]$$

where $\mathbf{F}(x) = \mathbb{E}_{y \sim p(y|x)}\left[ \nabla_x \log p(y|x) \nabla_x \log p(y|x)^T \right]$ is the Fisher information matrix at input $x$, encoding the local curvature of the recognition model's decision surface. Perturbations aligned with high-curvature directions produce larger embedding displacements per unit pixel change.

**Jacobian Subspace Regularization.** The perturbation is constrained to lie in the subspace of the image domain that most efficiently moves the embedding:

$$P_{\text{PRISM}} = J^{\dagger} \Delta z$$

where $J = \frac{\partial f(I)}{\partial I}$ is the Jacobian of the recognition embedding with respect to the input, $J^{\dagger}$ is its Moore-Penrose pseudoinverse, and $\Delta z$ is the desired embedding displacement. This formulation projects the perturbation onto the manifold of image variations that are maximally relevant to the recognition task.

### 6.4 Experimental Evaluation

#### 6.4.1 Input Data

![PRISM LFW sample batch](images/prism_cell10_out0.png)

*Figure 6.2: LFW sample batch used for PRISM evaluation. Natural, in-the-wild face images with diverse identities, poses, and lighting conditions.*

![PRISM CelebA-HQ sample batch](images/prism_cell10_out1.png)

*Figure 6.3: CelebA-HQ sample batch used for PRISM evaluation. High-resolution celebrity faces with controlled quality.*

#### 6.4.2 Optimization Dynamics

![PRISM optimization dynamics](images/prism_cell30_out0.png)

*Figure 6.4: PRISM optimization dynamics on an LFW batch. The privacy loss decreases from 1.000 to 0.757 while utility and Jacobian regularization terms increase. The multi-term objective reaches a constrained equilibrium rather than fully minimizing any single term.*

#### 6.4.3 Visual Comparison

![PRISM LFW visual comparison](images/prism_cell32_out0.png)

*Figure 6.5: LFW visual comparison: Original images (top), PrIdentity anonymized (middle), PRISM anonymized (bottom). SSIM values annotated on the grid show PRISM achieving higher visual similarity than PrIdentity.*

![PRISM CelebA-HQ visual comparison](images/prism_cell32_out1.png)

*Figure 6.6: CelebA-HQ visual comparison following the same format as Figure 6.5.*

#### 6.4.4 Perturbation Analysis

![PRISM perturbation amplitude maps](images/prism_cell34_out0.png)

*Figure 6.7: Perturbation amplitude maps amplified by 10x for visibility. Top row: original images. Middle row: PrIdentity perturbations showing uniform distribution. Bottom row: PRISM perturbations showing wavelet sub-band structure with concentration in high-frequency bands.*

![PRISM frequency-domain analysis](images/prism_cell34_out1.png)

*Figure 6.8: Frequency-domain perturbation energy distribution. PrIdentity spreads perturbation energy uniformly across frequencies. PRISM concentrates energy in high-frequency bands via its wavelet decomposition, an observation that aligns with FASP's design rationale but is achieved through more expensive geometric machinery.*

#### 6.4.5 Embedding Space Analysis

![PRISM embedding analysis](images/prism_cell38_out0.png)

*Figure 6.9: Embedding-space analysis comparing PrIdentity and PRISM. PrIdentity achieves mean cosine similarity 0.478 +/- 0.033, indicating strong embedding displacement. PRISM yields 0.779 +/- 0.091, reflecting weaker privacy despite higher visual fidelity.*

#### 6.4.6 Tradeoff Analysis

![PRISM tradeoff analysis](images/prism_cell40_out0.png)

*Figure 6.10: Privacy-utility tradeoff across perturbation budgets. TPR trajectories and SSIM curves are shown for both PrIdentity and PRISM, illustrating the Pareto frontier achievable by each formulation.*

#### 6.4.7 Comprehensive Metrics Comparison

![PRISM radar and bar comparison](images/prism_cell44_out0.png)

*Figure 6.11: Radar chart and bar comparison across 8 evaluation dimensions for LFW, comparing PrIdentity and PRISM. PrIdentity dominates privacy-side metrics; PRISM leads SSIM, PSNR, and bounding-box distance.*

### 6.5 Quantitative Results

**Comprehensive Evaluation (LFW and CelebA-HQ):**

| Dataset | Method | SSIM | PSNR | TPR (WB) | TPR (BB) | CosSim (WB) | CosSim (BB) | BB Dist | T-score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LFW | PrIdentity | 0.9194 | 35.84 | **0.0000** | **0.0000** | **0.4753** | 0.4399 | 0.0778 | **0.9194** |
| LFW | PRISM | **0.9327** | **37.36** | 0.7833 | 0.0167 | 0.7688 | **0.4218** | **0.0469** | 0.2021 |
| CelebA-HQ | PrIdentity | 0.9325 | 35.64 | **0.0000** | **0.0000** | **0.4794** | 0.4628 | 0.0301 | **0.9325** |
| CelebA-HQ | PRISM | **0.9393** | **36.97** | 0.7833 | 0.1333 | 0.7601 | 0.5998 | **0.0130** | 0.2035 |

### 6.6 PRISM Design Analysis

**What worked.** The Riemannian formulation successfully concentrated perturbation energy in high-frequency wavelet bands, an observation that independently validates FASP's frequency-aware design principle. PRISM achieved the best raw SSIM and PSNR among all tested methods. Black-box TPR was better than white-box TPR (0.0167 vs 0.7833 on LFW), suggesting the geometry-aware approach provides some cross-model generalization.

**What did not work sufficiently.** The white-box privacy performance was fundamentally weak, with TPR of 0.7833 compared to 0.0000 for PrIdentity. The Fisher matrix computation and Jacobian pseudoinverse incurred massive memory overhead, severely limiting the number of feasible optimization steps. Despite higher visual quality, the privacy-utility tradeoff score was substantially worse (0.2021 vs 0.9194 on LFW).

### 6.7 Why FASP Was Selected Over PRISM

PRISM's geometric formulation is mathematically elegant but practically inefficient for this task. The computation of $J^{\dagger}$ and $\mathbf{F}(x)$ at each optimization step is prohibitively expensive. More fundamentally, optimizing a geometric surrogate of the decision boundary is inherently less direct than attacking the cosine similarity metric itself. FASP achieves stronger identity suppression with a fraction of the computational cost by combining a targeted softplus margin loss with a lightweight, interpretable frequency prior.

---

## 7. Experimental Setup and Configuration

### 7.1 Datasets

| Dataset | Source | Purpose in This Work | Key Characteristics |
|---|---|---|---|
| LFW | Huang et al., 2008 | Verification benchmarking | 13,233 images, 5,749 identities, unconstrained natural conditions |
| CelebA | Liu et al., 2015 | Generalization testing | 202,599 images, 10,177 identities, varied pose/lighting/expression |
| CelebA-HQ | Karras et al., 2017 | High-fidelity utility evaluation | 30,000 images at 1024x1024 resolution |

All images undergo MTCNN face detection and alignment prior to optimization, with bounding box cropping and normalization to $[0,1]$.

### 7.2 Recognition Models

| Model | Architecture | Role in Evaluation | Embedding Dimension |
|---|---|---|---|
| VGGFace2 | ResNet-50 | Primary white-box surrogate | 512 |
| FaceNet/CASIA-WebFace | Inception ResNet v1 | Intermediate transfer (same family) | 512 |
| ArcFace | ResNet-100 | Strict black-box (margin loss) | 512 |
| MobileFace | MobileNet | Strict black-box (lightweight) | 128 |
| LightCNN-29 | Light CNN | Baseline comparison | 256 |

The VGGFace2/ResNet-50 model serves as the white-box surrogate against which perturbation gradients are computed. All other models are held out and used only for black-box evaluation to assess cross-architecture transferability.

### 7.3 Optimization Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Optimizer | Adam | Adaptive learning rates, momentum |
| Learning rate | $10^{-2}$ | Empirically determined for stable convergence |
| Iterations | 100 per image/batch | Sufficient for convergence at this learning rate |
| Perturbation budget $\epsilon$ | 8-16 pixel units | Explored in ablation; 12 recommended default |
| Frequency set $\mathcal{F}$ | 3-5 frequencies | Covers low, mid, and high spatial bands |
| $\lambda_u$ | Swept for Pareto frontier | Controls privacy-utility tradeoff |
| LPIPS backbone | AlexNet | Standard perceptual metric |
| $\kappa$ (softplus sharpness) | 10 | Sharp transition near target threshold |
| $\tau$ (target similarity) | 0.0-0.3 | Target cosine similarity range |

### 7.4 Evaluation Metrics

| Metric | Mathematical Definition | What It Measures | Desired Direction |
|---|---|---|---|
| SSIM | $\frac{(2\mu_I\mu_A + c_1)(2\sigma_{IA} + c_2)}{(\mu_I^2 + \mu_A^2 + c_1)(\sigma_I^2 + \sigma_A^2 + c_2)}$ | Structural visual similarity | Higher |
| PSNR | $10 \log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}}\right)$ | Peak signal-to-noise ratio | Higher |
| LPIPS | $\sum_l w_l \|\phi_l(I) - \phi_l(A)\|_2^2$ | Perceptual feature distance | Lower |
| Cosine similarity | $\frac{z_I^T z_A}{\|z_I\|_2 \|z_A\|_2}$ | Embedding identity proximity | Lower |
| TPR @ FPR=0.001 | $\frac{\text{TP}}{\text{TP}+\text{FN}}$ at threshold giving FPR=0.001 | Verification recognition rate | Lower |
| Rank-1 Accuracy | Fraction of queries correctly matched to gallery | Identification recognition rate | Lower |
| Trade-off score $T$ | $\text{SSIM} \times (1 - \text{CosSim}_{\text{avg}})$ | Joint privacy-utility summary | Higher |
| Bounding box distance | $\|c_{\text{MTCNN}}(I) - c_{\text{MTCNN}}(A)\|_2$ | Geometric structure drift | Lower |

### 7.5 Hardware Requirements

For the supplied notebook implementations:
- GPU: Single CUDA-capable GPU with minimum 8GB memory (NVIDIA T4 or equivalent)
- System RAM: 16GB recommended
- Storage: Sufficient for dataset hosting (LFW: ~200MB compressed, CelebA: ~1.5GB, CelebA-HQ: ~4GB)

The PRISM notebook was authored for a dual-GPU Kaggle T4 environment due to the additional memory overhead of Fisher information computation. The FASP notebook runs on a single GPU.

---

## 8. Results and Analysis

### 8.1 FASP Ablation Study

![FASP ablation study](images/FASP_ablation%20study.png)

*Figure 8.1: FASP ablation study comparing five system variants on a 100-image CelebA-HQ subset. The bar chart reports SSIM (utility, left axis) while the annotation reports the proportion of cosine-similarity inversions below zero (privacy). Only the frequency-aware variants achieve any privacy; the pixel-domain baseline fails completely despite near-perfect SSIM.*

**Ablation Results (100-image CelebA-HQ subset):**

| Variant | Components | Cos < 0 (Privacy) | SSIM (Utility) | Analysis |
|---|---:|---:|---:|---|
| V0: Lp Baseline | Random init + Lp norm + Hard margin | 0 / 100 | 0.9982 | Privacy complete failure despite near-perfect utility |
| V1: Freq Only | Laplacian mask + L2 loss | 100 / 100 | 0.8968 | Frequency masking alone sufficient for privacy |
| V2: Sine Only | Sinusoidal carrier + L2 loss | 100 / 100 | 0.8758 | Sinusoidal structure alone sufficient but lower utility |
| V3: Sine + Landmark | Sinusoidal + Spatial landmark mask | 100 / 100 | **0.9238** | Best utility achieved with landmark guidance |
| V4: Full FASP | Sinusoidal + Laplacian mask + LPIPS | 100 / 100 | 0.9056 | Best overall formulation with perceptual loss |

**Component contribution analysis.** V0 demonstrates that pure norm regularization, even with a well-tuned Lp exponent, cannot achieve privacy at the constrained perturbation budget of $\epsilon = 12/255$. The perturbation magnitude that produces SSIM of 0.9982 is simply insufficient to move the embedding across the recognition decision boundary. V1 and V2 show that either the frequency mask or the sinusoidal carrier alone enables full surrogate suppression, confirming that structured perturbations are fundamentally more effective than unstructured noise of equivalent magnitude. V3 achieves the highest SSIM by using explicit facial landmark locations as an additional spatial mask, though this requires runtime landmark detection. V4 represents the recommended configuration, achieving full privacy with SSIM of 0.9056 using only image-derived Laplacian features without external landmark dependency.

### 8.2 Cross-Model Transferability

**Multi-Image Cross-Model Evaluation (100-image CelebA subset, VGGFace2 surrogate):**

| Model | Mean CosSim | Cos < 0 | Cos < 0.3 | Interpretation |
|---|---:|---:|---:|---|
| VGGFace2 (surrogate) | -0.0298 | 99/100 | 100/100 | Near-perfect white-box suppression |
| CASIA (black-box, same family) | -0.0138 | 97/100 | 100/100 | Strong transfer within model family |
| ArcFace (black-box, margin loss) | +0.6476 | 0/28 | 0/28 | Complete cross-architecture gap |
| MobileFace (black-box, lightweight) | +0.5794 | 0/37 | 1/37 | Partial suppression, architecture-dependent |
| **SSIM (across all)** | **0.8572** | — | — | Utility maintained despite privacy gap |

**Interpretation of the cross-architecture gap.** The near-identical performance on VGGFace2 and CASIA confirms that structured frequency perturbations generalize effectively within the same model family (both use triplet-based training). The gap to ArcFace and MobileFace reflects fundamental differences in loss landscape geometry. ArcFace's additive angular margin loss creates a decision boundary with different curvature properties than the triplet loss used in VGGFace2/FaceNet. Perturbations optimized against one loss landscape do not necessarily cross the decision boundary in the other. This is a well-known challenge in adversarial transferability and motivates multi-surrogate training as the primary future direction.

### 8.3 Gallery-Based Identification Suppression

**Rank-1 Anonymization Rate on CelebA (30 identities, VGGFace2 evaluation):**

| Gallery Size | FASP (Ours) | PrIdentity (Paper) |
|---|---:|---:|
| GS = 1 | **100.0%** | 5.4% |
| GS = 2 | **100.0%** | 5.4% |
| GS = 4 | **100.0%** | 8.4% |
| Average | **100.0%** | 6.4% |

FASP achieves complete Rank-1 anonymization across all gallery configurations, meaning that for every anonymized probe image, the closest gallery match under cosine similarity is an incorrect identity. This represents a fundamental improvement over PrIdentity's reported rates, though the comparison should be interpreted cautiously given potential differences in evaluation protocols, surrogate models, and perturbation budgets.

### 8.4 LFW Verification Benchmark

**LFW Verification under ArcFace Black-Box (50 pairs):**

| Metric | Value |
|---|---:|
| Threshold at FPR = 0.001 | 0.5000 |
| TPR at FPR = 0.001 | 0.300 |
| PrIdentity (paper, ArcFace) | 0.002 |
| Mean cosine similarity (same pairs) | 0.4462 +/- 0.1049 |

FASP's TPR of 0.300 on this black-box ArcFace evaluation reflects the cross-architecture gap. PrIdentity's 0.002 was measured under conditions that included multi-model surrogate training or closer surrogate-target alignment. This result motivates the multi-surrogate training extension.

### 8.5 Geometric Preservation

**Bounding Box Centroid Distance on CelebA-HQ (MTCNN detector, 50 images, lower is better):**

| Method | BB Distance | Rank |
|---|---:|---:|
| **FASP (Ours)** | **1.55** | **1** |
| PrIdentity | 2.65 | 2 |
| RiDDLE | 3.82 | 3 |
| DeepPrivacy | 4.65 | 4 |
| Diff-Privacy | 5.83 | 5 |
| FIT | 7.87 | 6 |
| FALCO | 7.88 | 7 |
| CIAGAN | 20.38 | 8 |

FASP achieves the lowest bounding-box displacement (1.55 pixels), representing a 41.5% reduction compared to PrIdentity (2.65) and a 92.4% reduction compared to CIAGAN (20.38). This confirms that the additive, edge-localized perturbation formulation preserves facial geometry by construction. Unlike generative methods that can shift facial landmarks or distort the face oval, FASP operates within the existing pixel grid, and the small bounding-box shift reflects only the mild influence of edge-concentrated perturbation on the MTCNN detector response.

### 8.6 Privacy Comparison Across Methods

![FASP privacy comparison with other methods](images/FASP_privacy%20comparision%20with%20others.png)

*Figure 8.2: Privacy performance of FASP compared with prior perturbation-based anonymization methods, reporting cosine similarity reduction across multiple recognition models. FASP achieves competitive suppression on the surrogate and CASIA black-box models while exhibiting a cross-architecture gap for ArcFace that motivates future work on multi-surrogate training.*

### 8.7 Configuration Variant Exploration

The FASP notebook generates outputs for multiple hyperparameter configurations to characterize sensitivity:

![FASP variant 001](images/FASP_001.png)

*Figure 8.3a: FASP configuration variant 001. Conservative frequency band with light edge masking, perturbation budget 8. Produces the subtlest changes, suited for applications where visual fidelity is paramount.*

![FASP variant 004](images/FASP_004.png)

*Figure 8.3b: FASP configuration variant 004. Medium frequency band with standard masking, perturbation budget 12. Recommended default configuration for balanced privacy and utility.*

![FASP variant 005](images/FASP_005.png)

*Figure 8.3c: FASP configuration variant 005. Aggressive frequency band with strong masking, perturbation budget 16. Maximizes privacy at the cost of slightly lower SSIM.*

![FASP variant 008](images/FASP_008.png)

*Figure 8.3d: FASP configuration variant 008. Exploratory hybrid combining frequency and spatial priors with landmark-weighted masking.*

These variants demonstrate that FASP performance is stable across a reasonable hyperparameter range while remaining tunable for specific application requirements. The conservative variant (001) is appropriate when near-perfect visual fidelity is required and partial privacy is acceptable. The aggressive variant (005) is appropriate when maximum privacy is required and minor visual artifacts are tolerable. The default variant (004) provides the recommended balance.

---

## 9. Comparative Analysis Across Methods

### 9.1 FASP vs PrIdentity: Direct Quantitative Comparison

| Metric | PrIdentity (Paper) | FASP (Ours) | Improvement |
|---|---:|---:|---:|
| BB Distance | 2.65 | **1.55** | 41.5% reduction |
| Rank-1 Anonymization (GS=1) | 5.4% remaining | **0.0%** remaining | Complete suppression |
| Rank-1 Anonymization (GS=2) | 5.4% remaining | **0.0%** remaining | Complete suppression |
| Rank-1 Anonymization (GS=4) | 8.4% remaining | **0.0%** remaining | Complete suppression |
| Surrogate Cos < 0 | Not reported separately | **99/100** | Verified complete |
| CASIA BB Cos < 0 | Not reported separately | **97/100** | Strong transfer |
| SSIM (best variant) | 0.90 | **0.9238** | +2.6% |
| SSIM (full model, V4) | 0.84-0.90 | 0.9056 | Upper range |
| LFW TPR ArcFace | **0.002** | 0.300 | PrIdentity advantage |

### 9.2 PRISM vs PrIdentity: Quantitative Comparison

| Metric | PrIdentity | PRISM | Winner |
|---|---:|---:|---|
| SSIM (LFW) | 0.9194 | **0.9327** | PRISM |
| PSNR (LFW) | 35.84 | **37.36** | PRISM |
| White-box TPR (LFW) | **0.0000** | 0.7833 | PrIdentity |
| Black-box TPR (LFW) | **0.0000** | 0.0167 | PrIdentity |
| CosSim WB (LFW) | **0.4753** | 0.7688 | PrIdentity |
| CosSim BB (LFW) | **0.4399** | 0.4218 | PrIdentity |
| BB Distance (LFW) | 0.0778 | **0.0469** | PRISM |
| Trade-off Score (LFW) | **0.9194** | 0.2021 | PrIdentity |
| SSIM (CelebA-HQ) | 0.9325 | **0.9393** | PRISM |
| Trade-off Score (CelebA-HQ) | **0.9325** | 0.2035 | PrIdentity |

### 9.3 Design Philosophy Comparison

| Aspect | PrIdentity | FASP | PRISM |
|---|---|---|---|
| Domain | Pixel | Frequency-aware structured | Wavelet + Riemannian |
| Core prior | Adaptive Lp norm | Edge attention + sinusoidal carrier | Fisher info + Jacobian subspaces |
| Perturbation structure | Learned via norm regularization | Explicitly structured via mask and carrier | Geometry-guided per-band |
| Parameter count | Full image resolution (37K+) | Sinusoidal parameters (~10-20) | Wavelet coefficients + geometric |
| White-box privacy | Strong | Strong (100/100 inversions) | Weak (TPR 0.7833) |
| Black-box privacy | Strong (TPR 0.002) | Partial within family | Partial across all |
| Best SSIM | 0.90-0.93 | 0.9238 | 0.9393 |
| BB Distance | 2.65 | **1.55** | 0.047 (normalized) |
| Interpretability | Medium | High (visualizable masks, spectra) | Lower (abstract geometric) |
| Computational cost | Medium | Medium | High |
| Primary use case | General strong baseline | Main method, geometry preservation | Research exploration |

---

## 10. Discussion and Interpretation

### 10.1 Why Frequency-Aware Cloaking Works

The empirical results strongly support the central hypothesis of this work: high-frequency, edge-aligned perturbations offer a fundamentally better privacy-utility exchange rate than uniform pixel-level noise. The mechanism operates through two complementary effects.

**Recognition-side effect.** Deep face recognition models trained with identity contrastive losses develop feature hierarchies that weight fine-scale texture and contour information heavily in their embedding computations. The Laplacian edge map directly identifies the spatial locations where these features are concentrated. By concentrating perturbation energy in these exact locations, FASP achieves maximum embedding displacement per unit perturbation magnitude. This is the standard adversarial vulnerability framed in spatial-frequency terms.

**Perception-side effect.** The human visual system exhibits contrast sensitivity that is modulated by spatial context. Perturbations superimposed on existing edges and textures are partially masked by the underlying image structure, a phenomenon studied extensively in the visual psychophysics literature. Smooth regions lack this masking effect, making equivalent-magnitude perturbations far more noticeable. By gating perturbations to edge regions, FASP exploits this perceptual masking to hide adversarial energy in plain sight.

### 10.2 Why Structure Matters More Than Magnitude

The ablation study provides direct evidence for the primacy of structure over magnitude. Variant V0 achieves SSIM of 0.9982—effectively visually identical to the original—yet fails completely at identity suppression (0/100 cosine inversions). This means the perturbation magnitude is not the limiting factor; rather, the perturbation lacks the spatial and spectral structure necessary to affect the recognition embedding.

Variants V1 through V4 all achieve complete suppression (100/100) at lower SSIM values, confirming that structured perturbation energy allocated to the right locations and frequencies is exponentially more effective than unstructured energy distributed uniformly. The structure-magnitude tradeoff is not linear: a small amount of well-structured perturbation outperforms a large amount of unstructured perturbation.

### 10.3 The Transferability Gap and Its Implications

The cross-architecture gap between VGGFace2/CASIA and ArcFace/MobileFace is the most significant limitation of the current FASP implementation. This gap has a clear theoretical explanation. VGGFace2 and CASIA models are trained with triplet loss, which optimizes relative distances between anchor, positive, and negative embeddings. ArcFace is trained with additive angular margin loss, which optimizes absolute angular distances to class centers. These different loss formulations create embedding spaces with fundamentally different geometries and decision boundary curvatures.

Perturbations that successfully cross the decision boundary in the VGGFace2 embedding space may remain on the same side of the boundary in the ArcFace space, particularly if they were optimized only against the former. This is not a flaw in FASP's perturbation structure but rather a fundamental property of single-surrogate adversarial optimization. Multi-surrogate training, where the perturbation is optimized against an ensemble of models simultaneously, is the standard solution and the recommended next step.

### 10.4 Vulnerabilities of Modern Face Embeddings

A broader implication of these results is that modern face recognition embeddings are surprisingly vulnerable to structured perturbations that are nearly imperceptible to humans. Perturbation budgets as small as 8-16 pixel units (on a 0-255 scale) are sufficient to invert cosine similarity for the vast majority of tested identities when the perturbation is properly structured.

This vulnerability is not a bug in specific model architectures; it is a fundamental property of embedding spaces trained with identity contrastive losses. These losses encourage the network to place high confidence in fine-grained texture features that reliably separate identities in the training set. Those same features happen to be the ones most easily disrupted by structured high-frequency noise without affecting human perception. This creates an inherent asymmetry that privacy-preserving systems can exploit.

### 10.5 Ethical Considerations

This research is designed to protect individual privacy against automated face recognition. However, the technology has dual-use potential that warrants explicit discussion.

**Intended use.** FASP is intended as a privacy-enhancement tool for individuals who wish to share images publicly (on social media, in datasets) while reducing the risk of automated identity linking and tracking.

**Potential misuse.** Face anonymization technology could potentially be used to evade legitimate biometric authentication or to conceal identity in contexts where accountability is important. The technology should not be presented as providing absolute protection against all recognition systems.

**Realistic expectations.** Users should understand that anonymization quality depends critically on the attack model, perturbation budget, and evaluation protocol. Protection trained against one recognition model family may not generalize to all deployed systems. The appropriate framing is privacy enhancement, not guaranteed invisibility.

---

## 11. Reproducibility and Resources

### 11.1 Environment Setup

The notebooks require Python 3.8+ with PyTorch and standard scientific computing packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision
pip install facenet-pytorch pytorch-wavelets scikit-image tabulate
pip install matplotlib pillow tqdm numpy pandas
