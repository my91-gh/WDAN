# Wavelet-based Disentangled Adaptive Normalization for Non-stationary Time Series Forecasting

**Authors:**  
Junpeng Lin¹, Tian Lan¹, Bo Zhang¹, Ke Lin², Dandan Miao², Huiru He³, Jiantao Ye³, Chen Zhang¹*, Yan-fu Li¹  
¹ Department of Industrial Engineering, Tsinghua University  
² Shanghai Research Center, Huawei Technologies  
³ Songshan Lake Research Center, Huawei Technologies  
Emails: {linjp22,lant23}@mails.tsinghua.edu.cn, {bz16328,zhangchen01,liyanfu}@tsinghua.edu.cn, {linke2,miaodandan,hehuiru,yejiantao}@huawei.com  
*Corresponding author  

---

## Abstract

Forecasting non-stationary time series is a challenging task because their statistical properties often change over time, making it hard for deep models to generalize well. Instance-level normalization techniques can help address shifts in temporal distribution. However, most existing methods overlook the multi-component nature of time series, where different components exhibit distinct non-stationary behaviors.

In this paper, we propose **Wavelet-based Disentangled Adaptive Normalization (WDAN)**, a model-agnostic framework designed to address non-stationarity in time series forecasting. WDAN uses discrete wavelet transforms to break down the input into low-frequency trends and high-frequency fluctuations, applying tailored normalization strategies to each part.

For trend components that exhibit strong non-stationarity, we apply first-order differencing to extract stable features used for predicting normalization parameters. Extensive experiments on multiple benchmarks demonstrate that WDAN consistently improves forecasting accuracy across various backbone models.  
Code is available at: [https://github.com/MonBG/WDAN](https://github.com/MonBG/WDAN)

---

## Introduction

Time series forecasting is essential in many domains such as energy, economics, transportation, and healthcare. While deep learning models have made significant progress in capturing temporal patterns, real-world time series remain difficult to predict due to their **non-stationary characteristics** — statistical properties such as mean and variance change over time, hindering model generalization.

Most existing research focuses on complex architectures to capture temporal dependencies but neglects the non-stationarity problem itself.

Real-world time series often contain **multiple components** (slow trends, periodic patterns, high-frequency noise), each with unique non-stationary behaviors. Traditional methods that treat the series as a single entity fail to model these distinct behaviors effectively.

**Time-frequency analysis** offers an approach to separate the components of a time series at different temporal scales, revealing the structure of non-stationarity. Discrete Wavelet Transform (DWT) provides such decomposition — isolating low-frequency trends and high-frequency fluctuations.

An example (Figure 1 in the paper) shows that:
- The **trend component** captures slow, non-stationary changes.
- The **residual (high-frequency)** component is more stationary.
- The **first-order differenced trend** appears more stationary, suggesting it helps stabilize training.

However, conventional normalization applies uniform transformations to the entire sequence, losing component-specific information.  

To overcome these limitations, we introduce **WDAN**, which:
1. Decomposes time series into low- and high-frequency components via DWT.
2. Applies **component-wise normalization** based on their statistical behavior.
3. Uses **first-order differencing** for trend stabilization.
4. Employs a **statistics prediction module** to adaptively denormalize model outputs.

Our contributions:
- Introduce a novel perspective separating statistical behaviors of different components.
- Propose a model-agnostic normalization framework for disentangling and treating components individually.
- Demonstrate consistent forecasting accuracy improvements across datasets and backbone models.

---

## Related Work

### Time Series Forecasting

Time series forecasting has long been a central problem across domains like finance, energy, and transportation.  

Traditional statistical models such as **ARIMA** rely on linear autoregressive structures to capture temporal dependencies. They are simple and interpretable but limited by strict assumptions about stationarity and linearity.

Modern **deep learning models**—including recurrent neural networks (RNNs), temporal convolutional networks (TCNs), and Transformer architectures—have achieved impressive results due to their nonlinear modeling power:

- **RNNs:** (Hochreiter & Schmidhuber 1997; Cho et al. 2014; Chung et al. 2015) capture sequential dependencies.
- **TCNs:** (Bai et al. 2018) and (Sen et al. 2019) extend convolutional models to temporal tasks.
- **Transformers:** (Vaswani et al. 2017; Zhou et al. 2021, 2022) adapt self-attention for sequence forecasting.

Further improvements include:
- Incorporating **decomposition** (Wu et al. 2021; Zeng et al. 2023) or **time–frequency representations** (Zhou et al. 2022).
- Modeling **cross-variable dependencies** via attention or graph neural networks (Bai et al. 2020; Wu et al. 2020).

However, the authors note that **channel-independent approaches** are often more robust to distribution shifts and non-stationarity (Wen et al. 2023; Han et al. 2024), and this motivates their design choice.

---

### Non-stationary Time Series Forecasting

**Non-stationarity**—where data distribution changes over time—remains a core challenge because it leads to poor generalization and unstable forecasts.

#### Traditional Approaches
Methods like **differencing** and **seasonal decomposition** (Box et al. 2015) pre-process data to remove trends and achieve stationarity.

#### Normalization-based Methods
Normalization has become the dominant deep-learning technique for mitigating non-stationarity:

- **RevIN** (Kim et al. 2021): symmetric normalization/denormalization per instance.  
- **DAIN** (Passalis et al. 2019): nonlinear transformation on statistics to adaptively stabilize inputs.  
- **Dish-TS** (Fan et al. 2023): learns mappings between input and output distributions (means & variances).  
- **SAN** (Liu et al. 2023c): learns slice-level distribution mappings across short time windows.

These methods reduce distribution shifts but **apply a single normalization to the whole sequence**, overlooking component-specific behaviors.

#### Dynamical-systems-based Methods
Some research models non-stationarity through **Koopman operators**, treating sequences as nonlinear dynamical systems (Liu et al. 2023b; Wang et al. 2022). These require strong domain knowledge and handcrafted measurement functions.

#### Key Limitation
Existing methods do not explicitly distinguish components with different degrees of non-stationarity.

---

### Summary

The WDAN framework differs from prior work by:
- **Disentangling** trend and residual components via wavelet decomposition.  
- **Applying tailored normalization** strategies to each.  
- Incorporating **first-order differencing** for trend modeling.  
- Remaining **model-agnostic**, compatible with various deep forecasting architectures.

---

## Methodology

We propose the **Wavelet-based Disentangled Adaptive Normalization (WDAN)** framework for non-stationary multivariate time series forecasting.

Given an input sequence:

- \( X \in \mathbb{R}^{N \times T} = (x_1, \dots, x_N) \),  
  where **N** is the number of variables and **T** the input length,  
- The goal is to predict the future sequence  
  \( Y \in \mathbb{R}^{N \times H} = (y_1, \dots, y_N) \)  
  with horizon length **H**.

The key idea is to **disentangle non-stationary components** using wavelet transforms, and to **adaptively predict future normalization statistics** for accurate denormalization.

---

### 1. Overall Framework

WDAN is a **model-agnostic normalization scheme**, meaning it can wrap around any deep forecasting backbone (RNNs, TCNs, Transformers, etc.).

The procedure consists of:
1. **Wavelet decomposition** of the input series into trend (low-frequency) and residual (high-frequency) parts.  
2. **Component-wise normalization** — extracting mean and variance from the decomposed parts.  
3. Feeding normalized sequences to a forecasting backbone.  
4. **Predicting future normalization parameters** to denormalize outputs.

Unlike existing methods that compute global normalization statistics directly from raw inputs, WDAN **uses the DWT-decomposed components** to derive more representative, component-level statistics.

---

### 2. Wavelet-based Disentangled Normalization

Each input variable \( x_i \) is decomposed using **Discrete Wavelet Transform (DWT)** into:

- Low-frequency (trend) coefficients \( c^i_l \)  
- High-frequency (detail) coefficients \( \{ c^i_{h,k} \}_{k=1}^{K} \)  

Formally:

\[
c^i_{l,k}, c^i_{h,k} = DWT_{\phi_l, \phi_h}(c^i_{l,k-1}), \quad \forall k = 1, \dots, K
\]

with:
\[
c^i_{l,0} = x_i, \quad c^i_l = c^i_{l,K}
\]

Inverse DWT reconstructs interpretable components:
\[
x^i_l = IDWT(c^i_l, 0), \quad
x^i_h = \sum_{k=1}^{K} IDWT(0, c^i_{h,k})
\]

Where:
- \( x^i_l \): **trend component** (captures long-term non-stationarity)
- \( x^i_h \): **residual component** (captures short-term variations)

#### Why Wavelets?
Unlike Fourier-based methods, DWT preserves **local temporal features** and can handle **non-stationary signals** adaptively.

---

### 3. Component-wise Normalization

The normalization statistics are derived **separately** from trend and residual components.

#### Mean estimation:
For each time step \( t \),
\[
\mu^i_x[t] = x^i_l[t]
\]

#### Variance estimation:
A sliding window of size \( 2w+1 \) is used on the high-frequency component:
\[
\mu^i_h[t] = \frac{1}{2w+1} \sum_{j=-w}^{w} x^i_h[t+j]
\]

\[
(\sigma^i_x[t])^2 = \frac{1}{2w+1} \sum_{j=-w}^{w} (x^i_h[t+j] - \mu^i_h[t])^2
\]

#### Normalization:
\[
\bar{x}^i[t] = \frac{x^i[t] - \mu^i_x[t]}{\sigma^i_x[t] + \epsilon}
\]

Here, \( \epsilon \) avoids division by zero.

Thus, WDAN **normalizes at the point level**, preserving detailed local dynamics and improving stationarity.

The normalized sequence \( \bar{X} = (\bar{x}_1, \dots, \bar{x}_N) \) is then passed into the backbone forecasting model \( g_\theta \).

---

### 4. Adaptive Non-stationarity Reconstruction

To counter distribution shifts, WDAN **predicts future normalization statistics** using a lightweight MLP network, and applies them during denormalization.

#### Statistics prediction with differencing

Rather than predicting future statistics directly, WDAN predicts **residual differences** relative to the current statistics’ global mean:

\[
\bar{\mu}^i = \frac{1}{T} \sum_{t=1}^T \mu^i_x[t], \quad
\bar{\sigma}^i = \frac{1}{T} \sum_{t=1}^T \sigma^i_x[t]
\]

Differences:
\[
\tilde{\mu}^i_x[t] = \mu^i_x[t] - \bar{\mu}^i, \quad
\tilde{\sigma}^i_x[t] = \sigma^i_x[t] - \bar{\sigma}^i
\]
\[
\Delta \tilde{\mu}^i_x[t] = \tilde{\mu}^i_x[t] - \tilde{\mu}^i_x[t-1]
\]

Predicted parameters:
\[
\hat{\mu}^i_y = MLP_\mu(MLP_1(\tilde{\mu}^i_x) \| MLP_2(\Delta \tilde{\mu}^i_x) \| MLP_3(x^i_h)) + \bar{\mu}^i
\]
\[
\hat{\sigma}^i_y = MLP_\sigma(MLP_4(\tilde{\sigma}^i_x) \| MLP_1(\tilde{\mu}^i_x) \| MLP_3(x^i_h)) + \bar{\sigma}^i
\]

Here “∥” denotes concatenation.  
Predictions are made pointwise for the future horizon \( H \).

---

### 5. De-normalization

After prediction:
\[
\bar{Y} = g_\theta(\bar{X}), \quad \bar{Y} = (\bar{y}_1, \dots, \bar{y}_N)
\]
\[
\hat{y}^i = \bar{y}^i \odot (\hat{\sigma}^i_y + \epsilon) + \hat{\mu}^i_y
\]

where \( \odot \) is elementwise multiplication.

The final forecast is \( \hat{Y} = (\hat{y}_1, \dots, \hat{y}_N) \).

---

### 6. Three-stage Training Strategy

Because the forecasting and normalization prediction modules are interdependent, WDAN uses a **three-stage training process**:

1. **Pretrain the statistics prediction module**  
   using ground-truth normalization statistics with loss \( l_{sp} \) (MAE or MSE).

2. **Train the backbone forecasting model**  
   with fixed prediction module, minimizing forecasting loss \( l_{fc} \).

3. **Joint fine-tuning**  
   of both modules with a smaller learning rate for stable convergence.

This structured training decouples learning of temporal dynamics and normalization parameters, leading to better stability and accuracy.

---

## Experiments

This section evaluates the proposed **WDAN** framework across multiple benchmark datasets and forecasting models.

---

### 1. Experimental Setup

#### Datasets

WDAN was tested on several widely used multivariate time series datasets, each exhibiting different levels of non-stationarity:

| Dataset | Variables | Sampling | Length | ADF Statistic | Notes |
|----------|------------|-----------|---------|----------------|-------|
| Exchange | 8 | 1 day | 7588 | -1.90 | Daily exchange rates of 8 countries (1990–2016) |
| ETTh1 | 7 | 1 hour | 17420 | -5.91 | Transformer temperature and load data |
| ETTh2 | 7 | 1 hour | 17420 | -4.13 | Same source as ETTh1, different split |
| ETTm1 | 7 | 15 mins | 69680 | -14.98 | Finer time resolution |
| ETTm2 | 7 | 15 mins | 69680 | -5.66 | Moderate non-stationarity |
| Weather | 21 | 10 mins | 52696 | -26.68 | Weather data with 21 indicators |
| Electricity | 321 | 1 hour | 26304 | -8.44 | Power usage for 321 clients |

*A smaller absolute value of the ADF statistic indicates stronger non-stationarity.*

Data splits follow chronological order:
- **ETT datasets:** 60% train, 20% validation, 20% test.
- **Others:** 70% train, 10% validation, 20% test.

A global **z-score normalization** is applied based on training statistics for comparability, though WDAN later replaces this with its adaptive normalization.

---

#### Baseline Forecasting Models

WDAN is designed to be **model-agnostic** and can integrate with any deep forecasting architecture.  
The paper evaluates WDAN on four representative backbones:

1. **FEDformer** (Zhou et al., 2022): combines seasonal-trend decomposition with frequency analysis.  
2. **Crossformer** (Zhang & Yan, 2022): models temporal and inter-variable dependencies via dual attention.  
3. **PatchTST** (Nie et al., 2023): Transformer variant that uses patch-level inputs for better temporal structure.  
4. **iTransformer** (Liu et al., 2023a): treats each variate as a token in the attention mechanism.

To highlight WDAN’s normalization benefits, comparisons were made with two other normalization frameworks:
- **SAN** (Liu et al., 2023c): segment-level adaptive normalization.
- **DDN** (Dai et al., 2024): dual-domain normalization (time and frequency).

---

#### Metrics

Two metrics are used:
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**

All models are tested at prediction horizons:
**H = {96, 192, 336, 720}**,  
with a fixed input window length of **L = 720**.

Each experiment is repeated three times to ensure stability.

---

### 2. Main Results

#### Overall Findings

WDAN consistently **improves forecasting accuracy** across all backbone models and datasets.

- The performance gains **increase with longer prediction horizons**, indicating that WDAN better handles long-term distribution shifts.
- Even models that already include normalization (like RevIN in iTransformer) see additional improvements when replaced by WDAN.

For example:
- On **ETTh1**, WDAN reduces iTransformer’s MSE by 19.6% at horizon 720.
- On **Exchange**, it improves FEDformer’s MSE by over 40% at long horizons.

#### Trend Observation

When prediction length increases from 96 → 720:
- WDAN’s relative improvement grows,
- showing that **handling non-stationarity** is especially critical for long-term forecasting.

---

### 3. Comparison with Other Normalization Frameworks

WDAN outperforms **SAN** and **DDN** across nearly all datasets and models.

Summary of average MSE improvements (↓ is better):

| Backbone | WDAN vs SAN | WDAN vs DDN | Comment |
|-----------|--------------|--------------|----------|
| iTransformer | +2–5% improvement | +3–4% improvement | Strong consistency |
| PatchTST | +3–10% | +5–9% | Large gains on Exchange, ETTh2 |
| Crossformer | +2–6% | +3–4% | Improved robustness |
| FEDformer | +2–10% | +3–5% | Large boost on long horizons |

Notably, the **largest gains** appear on the most **non-stationary datasets** (Exchange, ETTh2), validating WDAN’s motivation.

---

### 4. Ablation Study

To evaluate the contribution of each WDAN component, two simplified variants were compared:

1. **MovingAvg:** replaces DWT with a simple moving-average decomposition.  
2. **NoDiff:** removes first-order differencing in the trend modeling.

| Dataset | iTransformer (MSE) | Crossformer (MSE) | Notes |
|----------|--------------------|--------------------|-------|
| Exchange | 0.407 (WDAN) vs 0.438 (NoDiff) | Similar pattern | Differencing crucial for non-stationary data |
| ETTh2 | 0.334 (WDAN) vs 0.360 (MovingAvg) | Same trend | Wavelets outperform simple averages |
| Weather | Minimal differences | Low non-stationarity | WDAN stable even when not needed |

→ Both **wavelet decomposition** and **trend differencing** are essential for capturing multi-scale non-stationary dynamics.

---

### 5. Hyperparameter Sensitivity

Tests varied:
- number of MLP layers,
- hidden dimensions,
- DWT levels.

Results:
- Strongly non-stationary datasets (e.g., Exchange) are **sensitive** to hyperparameters.
- More stable datasets (e.g., Weather) remain **robust**.

→ Hyperparameter tuning is important when non-stationarity is strong.

---

### 6. Effect of Input Sequence Length

Longer input windows can worsen model performance in normal Transformers due to distribution drift.  
However, WDAN **stabilizes** or even **improves** performance as input length increases.

Example:  
On the Weather dataset (H = 720),
- iTransformer with WDAN reduces MSE by 11% when input length grows from 192 → 1080.
- Average improvement across backbones ≈ 10%.

→ WDAN mitigates degradation from long inputs by correcting distribution shifts dynamically.

---

### 7. Training Strategy Ablation

Different training strategies were compared:

| Strategy | Description | Performance |
|-----------|-------------|-------------|
| **Three-stage (WDAN default)** | Pretrain → Train → Joint Fine-tune | Best overall |
| Two-stage (Alternate) | Alternate between modules | Less stable |
| Two-stage (Co-train) | Train together after pretraining | Moderate |
| Single-stage | Train everything end-to-end | Unstable, worse results |

The **three-stage process** yields the best convergence and accuracy, confirming the benefit of decoupled optimization.

---

### 8. Summary of Findings

- WDAN improves all tested backbones, especially on long-term forecasts.
- Handles severe non-stationarity more effectively than existing normalization methods.
- Components (wavelet + differencing) both contribute measurably.
- Hyperparameter sensitivity is dataset-dependent.
- Training strategy significantly influences performance stability.

---

## Conclusion

This work presents **WDAN (Wavelet-based Disentangled Adaptive Normalization)** —  
a novel, model-agnostic framework for improving **non-stationary time series forecasting**.

By combining **wavelet decomposition**, **component-wise normalization**, and **differenced adaptive statistics prediction**, WDAN explicitly models the varying degrees of non-stationarity within different signal components.

**Key contributions:**
1. Introduced a wavelet-based method to disentangle trend and residual components.  
2. Proposed adaptive normalization tailored to each component’s behavior.  
3. Designed a statistics-prediction module leveraging first-order differencing to enhance stability.  
4. Achieved consistent improvements across diverse datasets and backbone architectures.

Experiments across seven benchmarks and four modern forecasting models show that WDAN significantly enhances performance—especially for long-term prediction and highly non-stationary datasets—demonstrating its robustness and general applicability.

Future work may explore:
- Extending WDAN to **spatio-temporal forecasting** problems (e.g., traffic or climate).  
- Combining WDAN with **online learning** for dynamic adaptation to real-time data.  
- Developing **lightweight variants** suitable for edge deployment.

---

## Appendix Summary

### A. Implementation Details

- Implemented in **PyTorch**, trained on **NVIDIA RTX 3090 GPUs**.  
- Optimizer: **AdamW**, initial learning rate \(1 \times 10^{-4}\), decayed by cosine schedule.  
- Batch size: 32 (iTransformer, PatchTST), 16 (Crossformer, FEDformer).  
- Early stopping with patience 5 on validation loss.  
- DWT implemented with `pywt` (Daubechies-4 wavelet).  
- MLP hidden dimension: 128, two layers per block.

---

### B. Wavelet Decomposition Visualization

Figures in the paper illustrate how DWT separates long-term trends from high-frequency noise on datasets like Exchange and ETTh1:
- The trend part shows **gradual changes** in mean.
- The residual part captures **short-term oscillations**.
- The **first-order differenced trend** becomes visibly more stationary.

---

### C. Hyperparameter Analysis

- **Window size (w):** Tested values 3–15, with 7 yielding best trade-off.  
- **DWT level (K):** Increasing K enhances decomposition depth but adds computation. K=2 used by default.  
- **Hidden dim:** 64–256 range tested; 128 optimal in most cases.

Performance remains stable under moderate hyperparameter changes, confirming WDAN’s robustness.

---

### D. Complexity Analysis

WDAN adds only minor computational overhead compared to vanilla normalization:
- **Time complexity:** \( O(NT) \) for DWT-based decomposition (linear in sequence length).  
- **Memory overhead:** negligible due to small auxiliary networks (≈0.2M parameters).

Thus, WDAN is efficient enough for integration into real-world forecasting systems.

---

### E. Additional Case Studies

#### Case 1: Electricity Dataset
WDAN maintains smooth long-term predictions without drift, whereas baseline models gradually diverge.

#### Case 2: Exchange Dataset
Denormalization guided by predicted statistics stabilizes forecasts during regime shifts (e.g., exchange rate spikes).

These visual results align with the quantitative improvements reported in the experiments.

---

### F. Limitations

- Performance gains depend on the degree of non-stationarity; improvements are smaller on stationary datasets.  
- Prediction of normalization statistics assumes stable relationships between past and future components, which may not hold under extreme shifts.

However, the authors note WDAN’s modular design makes it compatible with **future extensions** such as adaptive decomposition or self-supervised training.

---

## References

Key cited works include:

- Bai et al., 2018 — Temporal Convolutional Networks  
- Box et al., 2015 — Time Series Analysis: Forecasting and Control  
- Kim et al., 2021 — RevIN: Revisiting Instance Normalization  
- Liu et al., 2023a — iTransformer  
- Liu et al., 2023c — SAN: Sliced Adaptive Normalization  
- Nie et al., 2023 — PatchTST  
- Passalis et al., 2019 — DAIN: Adaptive Input Normalization  
- Zhou et al., 2022 — FEDformer  
- Zeng et al., 2023 — TimesNet: Temporal 2D Variation Modeling  

(Full bibliography available in the arXiv source.)

---

## Summary

**WDAN** introduces a new paradigm for dealing with non-stationarity in deep time series forecasting by explicitly modeling and normalizing distinct frequency components.  
It is simple to integrate, computationally efficient, and consistently improves both accuracy and robustness across datasets and models.

---