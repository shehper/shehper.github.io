---
layout: post
title: "What Makes Larger Models Better? A Features' Perspective"
date: 2025-01-15
---

Notes of a talk I recently gave on feature superposition (a microscopic phenomenon) and neural scaling laws (a macroscopic phenomenon). 

## Guiding Question For My Research

Can feature superposition (a microscopic phenomenon) explain neural scaling laws (a macroscopic phenomenon)?

## Specific Questions For This Talk

Consider Transformer models of fixed depth but different widths.

1. Do different models store the same features in different fashion?
2. OR: Do larger models store more features, giving them higher capability?
3. In either case, can we see its effect on scaling laws?

## Outline

1. **Background**
   - Neural Scaling Laws
   - Feature Superposition
2. **This work**
   - Feature Importance and Universality
   - Relationship with Scaling Laws

## Background

### Neural Scaling Laws

Neural scaling laws describe how model performance improves with scale (Kaplan et al., 2020; Hoffman et al., 2023, and others).

### Definition: Neural Network Features

![Feature Definition Diagrams](/images/presentation2/page_7_img_1.png)

- Activations can be decomposed into (overcomplete) bases.
- For token *j*, we can write activations as a sum over features *i*.
- Each feature *d_i* has a specific interpretation.
- *f_i(x)* ≥ 0 represents feature activations.
- For a given token, only a few features are active.

![Feature Activation Diagram](/images/presentation2/page_8_img_1.png)

**Example:** The Golden-Gate-Bridge feature of Claude 3 Sonnet (Templeton et al., 2024). Activating this feature to 10x its value changes the model behavior.

![Golden Gate Bridge Feature Example](/images/presentation2/page_10_img_1.jpeg)

![Golden Gate Bridge Feature Activation](/images/presentation2/page_11_img_1.jpeg)

### Definition: Superposition

![Superposition Diagrams](/images/presentation2/page_12_img_1.png)

- Neural networks store more features than the number of available dimensions.
- Hence, some features interfere with others.
- Intuitively, larger models perform better as they have more "capacity": they can store more features without interference (Elhage et al., 2022).

![Superposition Visualization](/images/presentation2/page_12_img_2.png)

**This talk:** Make this more precise.

### How are features learned?

![SAE Diagram](/images/presentation2/page_13_img_1.png)

Reconstruct activations using autoencoders and let the decomposition be sparse — **Sparse Autoencoders (SAEs)** (Bricken et al., 2023; Cunningham et al., 2023).

![SAE Architecture](/images/presentation2/page_14_img_1.png)

## New Work

### Feature Importance

Can we define a notion of importance of a feature for features of a real model?

- Important features must be more universal across models of different widths.
- Important features may be learned early in training. (Not answered today, but I have some observations.)
- Hence, scaling laws could be studied from the perspective of feature importance.

### Proposal (Definition): Feature Importance

![Feature Importance Definition](/images/presentation2/page_17_img_1.png)

Let the importance of feature *i* be its maximum activation value over a large dataset.

### Experiments: Transformer Models

![Experiment Setup](/images/presentation2/page_19_img_1.png)

- Train four 8-layer models with varying embedding dimensions: 128, 256, 512, 768
- Train SAEs on MLP outputs of 6th layer, all with 24,768 latents.
- Reconstructed Losses with the SAEs spliced on.

![Model Architecture](/images/presentation2/page_19_img_2.png)

![Reconstructed Losses](/images/presentation2/page_20_img_1.png)

### Are important features more universal?

**Measure of universality:** activation similarity. Roughly, how much do the activations of two features correlate?

More mathematically:
- Assign to each feature, a vector of length |X| of its activations.
- Compute Pearson Correlation between features of one model with another.

**Results:**

![Feature Importance vs Universality Plot](/images/presentation2/page_23_img_1.jpeg)

Relative Feature Importance (x-axis) vs Maximum Activation Similarity (y-axis)

- Fitted Equation: *y = 1 - e^(-b(x-c))*
- *r = 0.7002*

![Correlation Analysis](/images/presentation2/page_24_img_1.jpeg)

**Table of Correlation Coefficients:**

| width \ width | 128   | 256   | 512   | 768   |
|---------------|-------|-------|-------|-------|
| 128           | -     | 0.7002| 0.7096| 0.6975|
| 256           | 0.7216| -     | 0.7152| 0.7136|
| 512           | 0.7352| 0.7026| -     | 0.6646|
| 768           | 0.7640| 0.7751| 0.7735| -     |

### New Insight #1

**Important features tend to be more universal amongst models of fixed depth and various widths.**

### Dependence of Features Importance on Scaling Laws

![Scaling Laws Plot](/images/presentation2/page_28_img_1.png)

- Larger models perform better for any fixed number of features.
- Reason: less interference.
- In fact, models' differences increases as you pack more features.
- From the same number of features, larger models gain more capabilities.

![Feature Packing Comparison](/images/presentation2/page_29_img_1.jpeg)

**But do they also store more features?**

![More Features in Larger Models](/images/presentation2/page_29_img_2.png)

**Yes!** Larger models also pack more features!

![Feature Storage Analysis](/images/presentation2/page_31_img_1.jpeg)

### New Insight #2

**Larger models store more features. But even for a smaller number of features, they extract more performance.**

## Answer to Our Questions

Consider Transformer models of fixed depth but different widths.

**It's a combination of both!**

Neural Scaling Laws get contribution from both factors:
- Larger models extract more from a similar set of features.
- Larger models also store more features.

Quantifying the contribution from both is an interesting problem.

## Some Limitations

- We ignored any inductive biases SAEs bring to the family of features learned.
- We ignored part of original loss that is not reconstructed by the SAEs.
- We studied only 4 models of varying widths.

## References

- Kaplan et al. (2020) - Neural Scaling Laws
- Hoffman et al. (2023) - Neural Scaling Laws
- Bricken et al. (2023) - Sparse Autoencoders
- Cunningham et al. (2023) - Sparse Autoencoders
- Elhage et al. (2022) - Feature Superposition
- Templeton et al. (2024) - Golden-Gate-Bridge Feature

