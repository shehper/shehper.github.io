---
layout: post
title: "Revisiting Neural Network Parameterizations for Optimal Performance"
date: 2026-05-24
description: "A modified standard parameterization admits hyperparameter transfer and outperforms μP — over both width and depth."
---

<div class="callout" markdown="1">
<span class="callout-label meta">Acknowledgements:</span> This work was done at Essential AI. I would like to thank Ashish Vaswani for guidance and the rest of the team for support during this project.
</div>

<div class="callout" markdown="1">
<span class="callout-label green">TL;DR:</span> A common understanding amongst practitioners is that $\mu P$ (maximal update parameterization; [Yang et al., 2022](#ref-yang-2022){: #cite-yang2022}) is the unique parameterization that admits hyperparameter transfer. [Everett et al. (2024)](#ref-everett-2024){: #cite-everett2024} showed that certain assumptions in the derivation of $\mu P$ are too strong in practice, and when those assumptions are relaxed, other parameterizations can also admit hyperparameter transfer. They also showed that the modified standard parameterization consistently outperforms $\mu P$ as measured by the final training loss of large-scale training runs.

This blogpost reproduces results of [Everett et al. (2024)](#ref-everett-2024), albeit at a smaller scale. It further argues that modified SP outperforms vanilla standard parameterization at large widths, and hence, presents a better alternative to the commonly used scaling laws for predicting hyperparameters of a large-scale training run. It also shows that when combined with CompleteP ([Dey et al., 2025](#ref-dey-2025){: #cite-dey2025}), modified SP can be extended to transfer hyperparameters over depth. The resulting large-depth models achieve better performance compared to their alternatives in terms of the final training loss.
</div>

When training large neural networks, we want to figure out the optimal hyperparameters for our largest models. It's not feasible in practice to tune the hyperparameters for the largest model directly. Hence, two techniques exist:

1. Use a parameterization such as $\mu P$ ([Yang et al., 2022](#ref-yang-2022)) that permits hyperparameter transfer: tune learning rate, initial standard deviation, etc. at small scale and use the same optimal values at large scale. (The word on the street is that OpenAI used this technique to train GPT-4.)
2. Observe empirically how hyperparameters scale as a function of width, depth, batch size, etc., and use these scaling laws to predict hyperparameters for the largest models. ([DeepSeek LLM (2024)](#ref-deepseek-2024){: #cite-deepseek2024} and [Qwen 2.5 (2024)](#ref-qwen-2024){: #cite-qwen2024} use this approach, for example.)

This blogpost focuses on the former technique, but it also argues that the former may actually be better in practice than the latter.

So let's take a step back and look at what a neural network parameterization is and how it affects hyperparameter tuning. A parameterization is a rule for how the quantities you don't scale (learning rate, weight decay, initial variance of weights, etc.) depend on the quantities you do scale (width, depth, batch size, training length) when scaling up model training. The default choice is *Standard Parameterization (SP)* that sets the initial variance of weights to $\frac{1}{\text{fan-in}}$, and keeps other hyperparameters such as learning rate, weight decay, etc. independent of scale. As a result, as we scale up width, the optimal learning rate shifts to smaller values (Figure 1a). Keeping the learning rate constant instead results in final logits exploding as width increases (Figure 1b), which can cause training instabilities.

<div class="fig-row">
  <figure>
    <img src="/images/parameterizations/fig1a.svg" alt="Optimal learning rate vs. width under Standard Parameterization">
    <figcaption><strong>Figure 1a.</strong> Standard Parameterization: as width increases, the optimal learning rate shifts to smaller values — i.e., the learning rate does not transfer across width.</figcaption>
  </figure>
  <figure>
    <img src="/images/parameterizations/fig1b_toplegend.svg" alt="Final logit magnitude vs. width at fixed learning rate under Standard Parameterization">
    <figcaption><strong>Figure 1b.</strong> Standard Parameterization: at a fixed learning rate, the final logits grow as width increases.</figcaption>
  </figure>
</div>

[Yang et al. (2022)](#ref-yang-2022) set out to solve the latter problem by imposing by hand the following requirements:

- **(stability)** every activation and pre-activation has $\Theta(1)$-sized coordinates as functions of width,
- **(non-triviality)** output logits have $O(1)$-sized coordinates as functions of width, and
- **(feature learning)** parameter updates are as large as possible without leading to divergence.

As a result, they obtained the following parameterization: when training with Adam, scale the learning rate of hidden and unembedding weights and the initial variance of embedding and hidden weights with $\frac{1}{\text{fan-in}}$. Also, scale the initial variance of the unembedding weights with $\frac{1}{\text{fan-in}^2}$. They argued theoretically and showed empirically that this parameterization admits hyperparameter transfer. Empirically, this results in the hyperparameters at small scale being optimal at large scale (Figure 2a). The final logits also stay constant as model width is increased (Figure 2b).[^coordcheck]

<div class="fig-row">
  <figure>
    <img src="/images/parameterizations/fig2a.svg" alt="Optimal learning rate vs. width under μP">
    <figcaption><strong>Figure 2a.</strong> $\mu P$: the optimal learning rate is stable across widths — the learning rate transfers.</figcaption>
  </figure>
  <figure>
    <img src="/images/parameterizations/fig2b_toplegend.svg" alt="Final logit magnitude vs. width under μP">
    <figcaption><strong>Figure 2b.</strong> $\mu P$: the final logits stay roughly constant as width increases (a coordinate check).</figcaption>
  </figure>
</div>

[Yang et al. (2022)](#ref-yang-2022) argued that under certain technical assumptions, their parameterization is the unique parameterization that satisfies the three criteria above. They named it "maximal update parameterization" or $\mu P$ and exhibited its empirical success by training a 6.7B model, which outperformed the original GPT-3 6.7B.[^gpt3fig]

We observe in our reproduction, however, that at each given width, the best model trained with SP (Figure 1a) performs better than its $\mu P$ counterpart (Figure 2a). Others have also reported similar results in the past. (See [Vlassis et al. (2025)](#ref-vlassis-2025){: #cite-vlassis2025} and [an open issue on the mup GitHub repository](https://github.com/microsoft/mup/issues/76).) So does $\mu P$'s original reported success come with any caveats?

Luckily, [Everett et al. (2024)](#ref-everett-2024) already did a deep dive on this issue. They reported that the original success of $\mu P$ in their GPT-3-scale experiment is due to them tuning a lot more hyperparameters than just the learning rate and initial variance at small scale. Indeed, Appendix F.4 of [Yang et al. (2022)](#ref-yang-2022) lists all the hyperparameters tuned by them: attention temperature, output temperature, embedding multiplier, etc. Technically, of course, this is not a problem. **Yet, it's good to be aware that $\mu P$, when applied without tuning extra parameters, can give you worse performance at scale.** [Everett et al. (2024)](#ref-everett-2024) confirm this hypothesis by conducting thorough experiments at significantly large scales.

Theoretically, they also revisit one of the main assumptions in the derivation of $\mu P$ — namely, that updates to the unembedding matrix are perfectly aligned with its pre-activations — and show that it does not hold well in practice. Relaxing this condition gives them many more parameterizations which satisfy the stability, non-triviality, and feature learning criteria above. For example, Standard Parameterization admits a simple modification: set, as before, all initial variances to $\frac{1}{\text{fan-in}}$, but also scale Adam's learning rate for hidden and unembedding matrices by the same $\frac{1}{\text{fan-in}}$ factor. The resulting parameterization, they show, admits hyperparameter transfer.

We checked whether this modification indeed allows hyperparameter transfer (Figure 3). We observe that our results are consistent with theirs, with the final loss achieved being significantly better than that of models trained with $\mu P$ at each scale.

<figure>
  <img src="/images/parameterizations/fig3.svg" alt="Optimal learning rate vs. width under modified Standard Parameterization">
  <figcaption><strong>Figure 3.</strong> Modified SP: the optimal learning rate transfers across widths, while achieving lower final loss than $\mu P$.</figcaption>
</figure>

In fact, the best performing models with modified SP (Figure 3) also outperform their vanilla SP counterparts (Figure 1a). Table 1 reports the final loss obtained at each width with SP and modified SP, along with the percentage improvement of the latter over the former. We observe that the advantage of modified SP over SP grows at larger scales.

| width | SP | Modified SP | % improvement |
|------:|------:|------:|------:|
| 256 | 2.083 | 2.085 | −0.10% |
| 512 | 1.907 | 1.843 | +3.36% |
| 1024 | 1.773 | 1.702 | +4.00% |
| 2048 | 1.665 | 1.595 | +4.20% |
| 4096 | 1.603 | 1.523 | +4.99% |

**Table 1.** Final training loss at each width under SP and modified SP, with the percentage improvement of modified SP over SP. The advantage grows with scale.
{: .table-caption}

In the beginning of this post, I shared two common techniques for determining the hyperparameters of a large training run: hyperparameter transfer or hyperparameter scaling laws. I believe Table 1 presents evidence for the superiority of the former approach. By using hyperparameter scaling laws, you can at best get a model as good as the best model trained with vanilla SP.[^scalinglaws] Our modified Standard Parameterization, however, outperforms these best models. Hence, **for large-scale training runs, a good parameterization is likely a better approach than using scaling laws to discover the hyperparameters.**

In our experiments, we tuned and transferred only the global learning rate. But as is shown by [Yang et al. (2022)](#ref-yang-2022), one could tune many more hyperparameters to get the maximum benefit out of $\mu P$. Does this extra leverage change the balance of performance in favor of $\mu P$? We did not conduct these experiments, but [Everett et al. (2024)](#ref-everett-2024) showed that their **modified SP outperforms $\mu P$ at any scale provided that an equivalent number of hyperparameters is tuned and transferred.**[^perlayer]

### Transfer over depth

So far the discussion has been about transferring hyperparameters over width. What if we want to transfer hyperparameters over depth instead? Recent works such as Depth-$\mu P$ ([Yang et al., 2024](#ref-yang-2024){: #cite-yang2024}) and CompleteP ([Dey et al., 2025](#ref-dey-2025)) give recipes to extend $\mu P$ for depth-transfer. For example, CompleteP proposes scaling Adam's learning rate for hidden matrices as $\frac{1}{\text{fan-in}} \times L^{\alpha-1}$, where $L$ is the number of residual blocks; $\alpha \in \lbrace 0.5, 1 \rbrace$ is a hyperparameter; and the $\frac{1}{\text{fan-in}}$ factor comes from width-transfer over $\mu P$.[^completep]

What if we extend modified SP instead of $\mu P$ to obtain an alternative to CompleteP?[^modsp] We sweep over learning rates and look for depth-transfer with SP, $\mu P$, and modified SP in Figures 4a, 4b, and 4c respectively. We see again that depth-transfer built on top of modified SP performs better than its vanilla SP and $\mu P$ counterparts.

<div class="fig-row cols-3">
  <figure>
    <img src="/images/parameterizations/fig4a.svg" alt="Learning-rate transfer over depth with Standard Parameterization">
    <figcaption><strong>Figure 4a.</strong> Learning rate does not transfer over depth with vanilla SP.</figcaption>
  </figure>
  <figure>
    <img src="/images/parameterizations/fig4b.svg" alt="Learning-rate transfer over depth with μP / CompleteP">
    <figcaption><strong>Figure 4b.</strong> Learning rate transfers over depth with $\mu P$ (CompleteP).</figcaption>
  </figure>
  <figure>
    <img src="/images/parameterizations/fig4c.svg" alt="Learning-rate transfer over depth with modified Standard Parameterization">
    <figcaption><strong>Figure 4c.</strong> Learning rate transfers over depth with modified SP.</figcaption>
  </figure>
</div>

### Future Directions

For this blogpost, we only studied dense models trained with Adam. The next natural step is to test the ideas presented here on larger models with mixture-of-expert modules and more modern optimizers such as Muon. Given compute and resources, this should be easy to do.

On a more ambitious front, I believe that there exist parameterizations that are even more performant than modified SP, and that it's possible to find them. Some of the hidden clues are already in the [Everett et al. (2024)](#ref-everett-2024) paper. But more on this set of ideas later!

---

### Appendix: Training Details

All width-scaling experiments were trained on a 4-layer Transformer model with a Gemma 2 decoder block containing both pre-layer and post-layer norms. Each model is trained with a sequence length of 8192 for 2B tokens, with a global batch size of 196K tokens. I observed similar results on 8-layer models but have not included them here. We used untied embeddings in all experiments and initialized the unembedding matrix with zeros when training with $\mu P$, as is the recommendation in [Yang et al. (2022)](#ref-yang-2022).

Depth-scaling experiments were trained on the same architecture as the width-scaling experiments, but with a fixed embedding dimension of 256 and MLP feed-forward dimension of 1024. These choices were made to be consistent with depth-scaling experiments of [Dey et al. (2025)](#ref-dey-2025). Following their work, I used $\alpha=1$ when testing CompleteP, but when replacing $\mu P$ by modified SP for depth transfer, I found $\alpha = 0.5$ to be a better choice.

### References

1. <a id="ref-yang-2022"></a>Greg Yang, Edward J. Hu, et al. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer.* arXiv:2203.03466 (2022). [arXiv](https://arxiv.org/abs/2203.03466)&nbsp;[↩](#cite-yang2022)
2. <a id="ref-everett-2024"></a>Katie Everett, et al. *Scaling Exponents Across Parameterizations and Optimizers.* arXiv:2407.05872 (2024). [arXiv](https://arxiv.org/abs/2407.05872)&nbsp;[↩](#cite-everett2024)
3. <a id="ref-dey-2025"></a>Nolan Dey, et al. *Don't Be Lazy: CompleteP Enables Compute-Efficient Deep Transformers.* arXiv:2505.01618 (2025). [arXiv](https://arxiv.org/abs/2505.01618)&nbsp;[↩](#cite-dey2025)
4. <a id="ref-deepseek-2024"></a>DeepSeek-AI. *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism.* arXiv:2401.02954 (2024). [arXiv](https://arxiv.org/abs/2401.02954)&nbsp;[↩](#cite-deepseek2024)
5. <a id="ref-qwen-2024"></a>Qwen Team. *Qwen2.5 Technical Report.* arXiv:2412.15115 (2024). [arXiv](https://arxiv.org/abs/2412.15115)&nbsp;[↩](#cite-qwen2024)
6. <a id="ref-vlassis-2025"></a>Georgios Vlassis, David Belius, and Vladyslav Fomichov. *A thorough reproduction and evaluation of μP.* Transactions on Machine Learning Research (2025). [OpenReview](https://openreview.net/forum?id=AFxEdJwQcp)&nbsp;[↩](#cite-vlassis2025)
7. <a id="ref-yang-2024"></a>Greg Yang, Dingli Yu, Chen Zhu, and Soufiane Hayou. *Tensor Programs VI: Feature Learning in Infinite-Depth Neural Networks.* ICLR 2024. [OpenReview](https://openreview.net/forum?id=17pVDnpwwl)&nbsp;[↩](#cite-yang2024)

[^coordcheck]: The stability of final logits in Figure 2b can also be seen as a check for the correctness of our implementation. Checks of this kind were called "coordinate checks" by [Yang et al. (2022)](#ref-yang-2022).
[^gpt3fig]: See Figure 15 in [Yang et al. (2022)](#ref-yang-2022).
[^perlayer]: [Everett et al. (2024)](#ref-everett-2024) call these extra hyperparameters "per-layer learning rates"; see Figure 3 in their paper for results.
[^scalinglaws]: The assumption here is that hyperparameter scaling laws are conducted with vanilla SP, which, to my best knowledge, is the common practice.
[^completep]: See Table 1 in [Dey et al. (2025)](#ref-dey-2025) for the complete set of rules.
[^modsp]: That is, we keep all $m_L$-dependent factors the same as in Table 1 of [Dey et al. (2025)](#ref-dey-2025), but replace the $m_N$-dependent factors by those suggested by modified SP.
