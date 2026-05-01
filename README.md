# Temporal Preference Optimization for Unsupervised Retrieval (TPOUR)

> **HyunJin Kim, Jaejun Shim, Young Jin Kim, JinYeong Bak**
> *ICML 2026*

---

## Overview

**TPOUR (Temporal Preference Optimization for Unsupervised Retrieval)** is a framework for learning **temporally-aware dense retrievers** without requiring explicit timestamp supervision solely based on corpus-level temporal signal (i.e., data collected at a specific time).

Traditional unsupervised retrievers (e.g., contrastive learning–based models) focus purely on **semantic similarity**, often retrieving documents that are **temporally misaligned** with the query. TPOUR addresses this limitation by introducing **temporal preference learning** into the retrieval objective.

---

## Motivation

* Queries often contain **explicit** (e.g., "in 2019") or **implicit** (e.g., "this year") temporal intent
* Standard retrievers ignore this signal → **temporal misalignment**
* Supervised temporal retrieval requires labeled timestamps → **not scalable**

![TPOUR Overview](./figure/Comparison.pdf)

Figure: Comparison between TPOUR aligned at 2019 and a time-unaware retriever for queries with explicit (e.g., in 2019) or implicit (e.g., this year) temporal information. **Left**: A mixed-timestamp document collection containing (i) semantically and temporally aligned documents (green), (ii) semantically relevant but temporally misaligned documents (yellow), and (iii) irrelevant documents (red). **Right**: Ranked retrieval results. The time-unaware retriever, trained solely for semantic similarity, struggles to rank the temporally aligned document (green) over the misaligned (yellow). In contrast, the TPOUR-trained retriever prioritizes the temporally aligned document.

---

## Method

### Temporal Retrieval Preference Optimization (TRPO)

TPOUR integrates **contrastive learning** with a **preference optimization objective**:

* **Contrastive loss** → semantic similarity
* **TRPO loss** → temporal alignment based on preference learning

The model is trained to:

* Prefer **aligned document** $D^t$
* Over **misaligned document** $D^{t'}$

---

## Model Architecture

![TPOUR Method](./figure/Training_method.pdf)

Figure: Overview of TPOUR. Given a query $Q_i$ and two documents $D_i^t$ (temporally aligned) and $D_i^{t'}$ (temporally misaligned), each input is encoded using both the main encoder $\pi_\theta$ and the reference encoder $\pi_{\text{ref}}$. (1) Similarity scores are computed between the query and each document using $\pi_\theta$. (2) A contrastive loss $L_{\text{CE}}$, which calculate semantic similarity between $Q_i$ and $D_i^t$, and a TRPO loss $L_{\text{TPRO}}$ for preferring temporally aligned documents are calculated to get combined loss $L_{\text{total}}$. (3) The reference embeddings $\pi_{\text{ref}}(D_i^t)$ and $\pi_{\text{ref}}(D_i^{t'})$ are added to a queue as negatives for future batches. (4) The encoder $\pi_\theta$ is updated via $L_{\text{total}}$, and $\pi_{\text{ref}}$ is updated via momentum from $\pi_\theta$.

### Key Components

* **Encoder** $\pi_\theta$: learns joint semantic + temporal representations
* **Reference encoder** $\pi_{\text{ref}}$: momentum-updated (MoCo-style)
* **Preference pairs**: constructed from documents across time periods
* **Loss function**:

  * $L_{CE} = -\log \frac{e^{S_\theta(y_i^w)}}{e^{S_\theta(y_i^w)} + \sum_{j<i} (e^{S_{\mathrm{ref}}(y_j^w)} + e^{S_{\mathrm{ref}}(y_j^l)})}$: contrastive learning
  * $L_{\mathrm{TRPO}} = -\log \sigma\big(\beta [S_\theta(y_i^w) - S_\theta(y_i^l) - (S_{\mathrm{ref}}(y_i^w) - S_{\mathrm{ref}}(y_i^l))]\big)$: temporal preference alignment
  * $L_{total} = \lambda L_{CE} + (1 - \lambda)L_{TRPO}$

---

## Continuous Temporal Generalization

TPOUR introduces **time vector interpolation** (1) to enable smooth adaptation to **intermediate time periods** and (2) without training:

* Extract temporal shift:
  $\tau_t = \theta_t - \theta_{\text{base}}$

* Interpolate between time periods:
  $\theta_{mid} = \theta_{\text{base}} + (1-\alpha)\tau_{t_1} + \alpha\tau_{t_2}$

---

## Installation & Usage

Code and data will be released soon

## Citation

```bibtex
@inproceedings{kim2026tpour,
  title={Temporal Preference Optimization for Unsupervised Retrieval},
  author={Kim, HyunJin and Shim, Jaejun and Kim, Young Jin and Bak, JinYeong},
  booktitle={ICML},
  year={2026}
}
```
