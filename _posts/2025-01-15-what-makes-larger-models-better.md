---
layout: post
title: "A Features' Perspective on Neural Scaling Laws"
date: 2025-01-15
figure_stylesheet: /assets/css/scaling-law-figures.css
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

Neural scaling laws describe how model performance improves with scale ([Kaplan et al., 2020](#ref-kaplan-2020){: #cite-kaplan2020}; [Hoffmann et al., 2022](#ref-hoffmann-2022){: #cite-hoffmann2022}, and others).

<figure class="scaling-figure scaling-figure--wide">
  <a href="/images/scaling-laws/scaling-laws-background.png" title="View full-size figure" aria-label="Loss scales with compute, dataset size, and parameter count — view full-size image">
    <img src="/images/scaling-laws/scaling-laws-background.png" width="2182" height="700" alt="Loss scales with compute, dataset size, and parameter count" loading="lazy" decoding="async">
  </a>
</figure>

### Definition: Neural Network Features

- Activations can be decomposed into (overcomplete) bases.
- For token $j$, we can write activations as a sum over features $i$.
- Each feature $\mathbf{d}_i$ has a specific interpretation.
- $f_i(x) \ge 0$ represents feature activations.
- For a given token, only a few features are active.

<div class="scaling-equation">
\[
\mathbf{x}^{j} \approx \mathbf{b} + \sum_i f_i(\mathbf{x}^{j})\mathbf{d}_i
\]
</div>

**Example:** The Golden-Gate-Bridge feature of Claude 3 Sonnet ([Templeton et al., 2024](#ref-templeton-2024){: #cite-templeton2024}). Activating this feature to 10x its value changes the model behavior.

<figure class="scaling-figure scaling-figure--wide">
  <a href="/images/scaling-laws/golden-gate-activations.png" title="View full-size figure" aria-label="Text activating the Golden Gate Bridge feature — view full-size image">
    <img src="/images/scaling-laws/golden-gate-activations.png" width="1530" height="414" alt="Text activating the Golden Gate Bridge feature" loading="lazy" decoding="async">
  </a>
</figure>

<figure class="scaling-figure scaling-figure--wide">
  <a href="/images/scaling-laws/golden-gate-behavior.png" title="View full-size figure" aria-label="Claude output before and after amplifying the Golden Gate Bridge feature — view full-size image">
    <img src="/images/scaling-laws/golden-gate-behavior.png" width="1642" height="534" alt="Claude output before and after amplifying the Golden Gate Bridge feature" loading="lazy" decoding="async">
  </a>
</figure>

### Definition: Superposition

- Neural networks store more features than the number of available dimensions.
- Hence, some features interfere with others.
- Intuitively, larger models perform better as they have more "capacity": they can store more features without interference ([Elhage et al., 2022](#ref-elhage-2022){: #cite-elhage2022}).

<figure class="scaling-figure scaling-figure--diagram">
  <a href="/images/scaling-laws/superposition.png" title="View full-size figure" aria-label="Interference between features sharing a two-dimensional space — view full-size image">
    <img src="/images/scaling-laws/superposition.png" width="526" height="520" alt="Interference between features sharing a two-dimensional space" loading="lazy" decoding="async">
  </a>
</figure>

**This talk:** Make this more precise.

### How are features learned?

Reconstruct activations using autoencoders and let the decomposition be sparse — **Sparse Autoencoders (SAEs)** ([Bricken et al., 2023](#ref-bricken-2023){: #cite-bricken2023}; [Cunningham et al., 2023](#ref-cunningham-2023){: #cite-cunningham2023}).

<div class="scaling-equation">
\[
\mathcal{L} = \frac{1}{|X|} \sum_{x \in X} \left\lVert \mathbf{x} - \hat{\mathbf{x}} \right\rVert_2^2 + \lambda \left\lVert \mathbf{f} \right\rVert_1
\]
</div>

<div class="scaling-equation">
\[
f_i(x) = \operatorname{ReLU}\!\left(W_e(\mathbf{x} - \mathbf{b}_d) + \mathbf{b}_e\right)_i
\]
</div>

## New Work

### Feature Importance

Can we define a notion of importance of a feature for features of a real model?

- Important features must be more universal across models of different widths.
- Important features may be learned early in training. (Not answered today, but I have some observations.)
- Hence, scaling laws could be studied from the perspective of feature importance.

### Proposal (Definition): Feature Importance

<div class="scaling-equation">
\[
I_i = \max_{x \in X} f_i(x)
\]
</div>

Let the importance of feature $i$ be its maximum activation value over a large dataset.

### Experiments: Transformer Models

<figure class="scaling-figure scaling-figure--table">
  <a href="/images/scaling-laws/model-losses-table.png" title="View full-size figure" aria-label="Parameter counts and training and validation losses for four model widths — view full-size image">
    <img src="/images/scaling-laws/model-losses-table.png" width="946" height="300" alt="Parameter counts and training and validation losses for four model widths" loading="lazy" decoding="async">
  </a>
</figure>

- Train four 8-layer models with varying embedding dimensions: 128, 256, 512, 768
- Train SAEs on the residual stream entering the seventh Transformer block, all with 24,576 latents.
- Reconstructed Losses with the SAEs spliced on.

<div class="scaling-figure-pair">
<figure class="scaling-figure scaling-figure--chart">
  <a href="/images/scaling-laws/training-loss.svg" title="View full-size figure" aria-label="Training loss versus non-embedding parameter count — view full-size image">
    <img src="/images/scaling-laws/training-loss.svg" width="1854" height="1263" alt="Training loss versus non-embedding parameter count" loading="lazy" decoding="async">
  </a>
</figure>
<figure class="scaling-figure scaling-figure--chart">
  <a href="/images/scaling-laws/reconstructed-losses.png" title="View full-size figure" aria-label="Original and SAE-reconstructed losses versus non-embedding parameter count — view full-size image">
    <img src="/images/scaling-laws/reconstructed-losses.png" width="640" height="480" alt="Original and SAE-reconstructed losses versus non-embedding parameter count" loading="lazy" decoding="async">
  </a>
</figure>
</div>

### Are important features more universal?

**Measure of universality:** activation similarity. Roughly, how much do the activations of two features correlate?

More mathematically:
- Assign to each feature, a vector of length $\lvert X\rvert$ of its activations.
- Compute Pearson Correlation between features of one model with another.

**Results:**

<figure class="scaling-figure scaling-figure--chart">
  <a href="/images/scaling-laws/importance-universality.svg" title="View full-size figure" aria-label="Relative feature importance versus maximum activation similarity — view full-size image">
    <img src="/images/scaling-laws/importance-universality.svg" width="976" height="912" alt="Relative feature importance versus maximum activation similarity" loading="lazy" decoding="async">
  </a>
  <figcaption>Relative Feature Importance (x-axis) vs Maximum Activation Similarity (y-axis)</figcaption>
</figure>

- Fitted Equation: $y = 1 - e^{-b(x-c)}$
- $r = 0.7002$

<figure class="scaling-figure scaling-figure--wide">
  <a href="/images/scaling-laws/importance-universality-grid.svg" title="View full-size figure" aria-label="Pairwise feature importance and activation similarity across four model widths — view full-size image">
    <img src="/images/scaling-laws/importance-universality-grid.svg" width="4016" height="3920" alt="Pairwise feature importance and activation similarity across four model widths" loading="lazy" decoding="async">
  </a>
</figure>

**Table of Correlation Coefficients:**

<div class="scaling-results-table" markdown="1">

| width \ width | 128   | 256   | 512   | 768   |
|---------------|-------|-------|-------|-------|
| 128           | -     | 0.7735| 0.7751| 0.7640|
| 256           | 0.6446| -     | 0.7026| 0.7352|
| 512           | 0.7136| 0.7152| -     | 0.7216|
| 768           | 0.6975| 0.7096| 0.7002| -     |

</div>

### New Insight #1

**Important features tend to be more universal amongst models of fixed depth and various widths.**

### Dependence of Features Importance on Scaling Laws

<figure class="scaling-figure scaling-figure--chart">
  <a href="/images/scaling-laws/feature-retention-scaling.png" title="View full-size figure" aria-label="Training loss versus model size for different fractions of retained features — view full-size image">
    <img src="/images/scaling-laws/feature-retention-scaling.png" width="1158" height="642" alt="Training loss versus model size for different fractions of retained features" loading="lazy" decoding="async">
  </a>
</figure>

- Larger models perform better for any fixed number of features.
- Reason: less interference.
- In fact, models' differences increases as you pack more features.
- From the same number of features, larger models gain more capabilities.

<figure class="scaling-figure scaling-figure--chart">
  <a href="/images/scaling-laws/feature-retention-exponent.png" title="View full-size figure" aria-label="Scaling exponent versus the fraction of retained features — view full-size image">
    <img src="/images/scaling-laws/feature-retention-exponent.png" width="874" height="658" alt="Scaling exponent versus the fraction of retained features" loading="lazy" decoding="async">
  </a>
</figure>

**But do they also store more features?**

<figure class="scaling-figure scaling-figure--chart">
  <a href="/images/scaling-laws/feature-retention-losses.png" title="View full-size figure" aria-label="Reconstructed loss versus the fraction of retained features for each model width — view full-size image">
    <img src="/images/scaling-laws/feature-retention-losses.png" width="854" height="674" alt="Reconstructed loss versus the fraction of retained features for each model width" loading="lazy" decoding="async">
  </a>
</figure>

**Yes!** Larger models also pack more features!

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

1. <a id="ref-kaplan-2020"></a>Jared Kaplan, et al. *Scaling Laws for Neural Language Models.* arXiv:2001.08361 (2020). [arXiv](https://arxiv.org/abs/2001.08361)&nbsp;[↩](#cite-kaplan2020)
2. <a id="ref-hoffmann-2022"></a>Jordan Hoffmann, et al. *Training Compute-Optimal Large Language Models.* arXiv:2203.15556 (2022). [arXiv](https://arxiv.org/abs/2203.15556)&nbsp;[↩](#cite-hoffmann2022)
3. <a id="ref-bricken-2023"></a>Trenton Bricken, et al. *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning.* Transformer Circuits (2023). [Article](https://transformer-circuits.pub/2023/monosemantic-features/index.html)&nbsp;[↩](#cite-bricken2023)
4. <a id="ref-cunningham-2023"></a>Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey. *Sparse Autoencoders Find Highly Interpretable Features in Language Models.* arXiv:2309.08600 (2023). [arXiv](https://arxiv.org/abs/2309.08600)&nbsp;[↩](#cite-cunningham2023)
5. <a id="ref-elhage-2022"></a>Nelson Elhage, et al. *Toy Models of Superposition.* Transformer Circuits (2022). [Article](https://transformer-circuits.pub/2022/toy_model/index.html)&nbsp;[↩](#cite-elhage2022)
6. <a id="ref-templeton-2024"></a>Adly Templeton, et al. *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet.* Transformer Circuits (2024). [Article](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)&nbsp;[↩](#cite-templeton2024)
