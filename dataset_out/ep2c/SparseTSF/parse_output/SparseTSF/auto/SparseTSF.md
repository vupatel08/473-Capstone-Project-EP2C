# SparseTSF: Modeling Long-term Time Series Forecasting with 1k Parameters

Shengsheng Lin 1 Weiwei Lin 1 2 Wentai Wu 3 Haojun Chen 1 Junjie Yang 1

# Abstract

This paper introduces SparseTSF, a novel, extremely lightweight model for Long-term Time Series Forecasting (LTSF), designed to address the challenges of modeling complex temporal dependencies over extended horizons with minimal computational resources. At the heart of SparseTSF lies the Cross-Period Sparse Forecasting technique, which simplifies the forecasting task by decoupling the periodicity and trend in time series data. This technique involves downsampling the original sequences to focus on crossperiod trend prediction, effectively extracting periodic features while minimizing the model’s complexity and parameter count. Based on this technique, the SparseTSF model uses fewer than $I k$ parameters to achieve competitive or superior performance compared to state-of-the-art models. Furthermore, SparseTSF showcases remarkable generalization capabilities, making it wellsuited for scenarios with limited computational resources, small samples, or low-quality data. The code is publicly available at this repository: https://github.com/lss-1138/SparseTSF.

els to provide an extended predictive view for advanced planning (Zhou et al., 2021).

Although a longer predictive horizon offers convenience, it also introduces greater uncertainty (Lin et al., 2023b). This demands models capable of extracting more extensive temporal dependencies from longer historical windows. Consequently, modeling becomes more complex to capture these long-term temporal dependencies. For instance, Transformer-based models often have millions or tens of millions of parameters, limiting their practical usability, especially in scenarios with restricted computational resources (Deng et al., 2024).

In fact, the basis for accurate long-term time series forecasting lies in the inherent periodicity and trend of the data. For example, long-term forecasts of household electricity consumption are feasible due to the clear daily and weekly patterns in such data. Particularly for daily patterns, if we resample the electricity consumption at a certain time of the day into a daily sequence, each subsequence exhibits similar or consistent trends. In this case, the original sequence’s periodicity and trend are decomposed and transformed. That is, periodic patterns are transformed into inter-subsequence dynamics, while trend patterns are reinterpreted as intrasubsequence characteristics. This decomposition offers a novel perspective for designing lightweight LTSF models.

# 1. Introduction

Time series forecasting holds significant value in domains such as traffic flow, product sales, and energy consumption, as accurate predictions enable decision-makers to plan proactively. Achieving precise forecasts typically relies on powerful yet complex deep learning models, such as RNNs (Zhang et al., 2023), TCNs (Bai et al., 2018; Franceschi et al., 2019), and Transformers (Wen et al., 2022). In recent years, there has been a growing interest in Longterm Time Series Forecasting (LTSF), which demands mod

In this paper, we pioneer the exploration of how to utilize this inherent periodicity and decomposition in data to construct specialized lightweight time series forecasting models. Specifically, we introduce SparseTSF, an extremely lightweight LTSF model. Technically, we propose the CrossPeriod Sparse Forecasting technique (hereinafter referred to as Sparse technique). It first downsamples the original sequences with constant periodicity into subsequences, then performs predictions on each downsampled subsequence, simplifying the original time series forecasting task into a cross-period trend prediction task. This approach yields two benefits: (i) effective decoupling of data periodicity and trend, enabling the model to stably identify and extract periodic features while focusing on predicting trend changes, and (ii) extreme compression of the model’s parameter size, significantly reducing the demand for computational resources. As shown in Figure 1, SparseTSF achieves near state-of-the-art prediction performance with less than $\boldsymbol { { \mathit { 1 } } k }$ trainable parameters, which makes it $l { \sim } 4$ orders of magnitude smaller than its counterparts.

In summary, our contributions in this paper are as follows:

![](images/a652c7db5216814bc1e3f834906af3495d0832d1910081cdc77b67b92364068e.jpg)  
Figure 1: Comparison of MSE and parameters between SparseTSF and other mainstream models on the Electricity dataset with a forecast horizon of 720.

patch strategy, a technique that is prevalently employed in the realm of computer vision (Dosovitskiy et al., 2020; He et al., 2022). Besides Transformer architectures, Convolutional Neural Networks (CNNs) and Multilayer Perceptrons (MLPs) are also mainstream approaches, including SCINet (Liu et al., 2022a), TimesNet (Wu et al., 2023), MICN (Wang et al., 2022), TiDE (Das et al., 2023), and HDMixer (Huang et al., 2024a). Recent studies have shown that transferring pretrained Large Language Models (LLMs) to the time series domain can also yield commendable results (Chang et al., 2024; Jin et al., 2023; Xue & Salim, 2023). Moreover, recent works have revealed that RNN and GNN networks can also perform well in LTSF tasks, as exemplified by SegRNN (Lin et al., 2023b) and CrossGNN (Huang et al., 2024b).

• We propose a novel Cross-Period Sparse Forecasting technique, which downsamples the original sequences to focus on cross-period trend prediction, effectively extracting periodic features while minimizing the model’s complexity and parameter count.

• Based on the Sparse technique, we present the SparseTSF model, which requires fewer than $I k$ parameters, significantly reducing the computational resource demand of forecasting models.

• The proposed SparseTSF model not only attains competitive or surpasses state-of-the-art predictive accuracy with a remarkably minimal parameter scale but also demonstrates robust generalization capabilities.

# 2. Related Work

Development of Long-term Time Series Forecasting The LTSF tasks, which aim at predicting over an extended horizon, are inherently more challenging. Initially, the Transformer architecture (Vaswani et al., 2017), known for its robust long-term dependency modeling capabilities, gained widespread attention in the LTSF domain. Models such as Informer (Zhou et al., 2021), Autoformer (Wu et al., 2021), and FEDformer (Zhou et al., 2022b) have modified the native structure of Transformer to suit time series forecasting tasks. More recent advancements, like PatchTST (Nie et al., 2023) and PETformer (Lin et al., 2023a), demonstrate that the original Transformer architecture can achieve impressive results with an appropriate

Progress in Lightweight Forecasting Models Since DLinear (Zeng et al., 2023) demonstrated that simple models could already extract strong temporal periodic dependencies, numerous studies have been pushing LTSF models towards lightweight designs, including LightTS(Zhang et al., 2022), TiDE (Das et al., 2023), TSMixer (Ekambaram et al., 2023), and HDformer (Deng et al., 2024). Recently, FITS emerged as a milestone in the lightweight LTSF process, being the first to reduce the LTSF model scale to the 10k parameter level while maintaining excellent predictive performance (Xu et al., 2024). FITS achieved this by transforming time-domain forecasting tasks into frequency-domain ones and using low-pass filters to reduce the required number of parameters. In this paper, our proposed SparseTSF model takes lightweight model design to the extreme. Utilizing the Cross-Period Sparse Forecasting technique, it’s the first to reduce model parameters to below $I k$ .

# 3. Methodology

# 3.1. Preliminaries

Long-term Time Series Forecasting The task of LTSF involves predicting future values over an extended horizon using previously observed multivariate time series (MTS) data. It is formalized as $\bar { x } _ { t + 1 : t + H } = f ( x _ { t - L + 1 : t } )$ , where xt−L+1:t ∈ RL×C and $\bar { x } _ { t + 1 : t + H } \in \mathbb { R } ^ { H \times C }$ . In this formulation, $L$ represents the length of the historical observation window, $C$ is the number of distinct features or channels, and $H$ is the length of the forecast horizon. The main goal of LTSF is to extend the forecast horizon $H$ as it provides richer and more advanced guidance in practical applications. However, an extended forecast horizon $H$ also increases the complexity of the model, leading to a significant increase in parameters in mainstream models. To address this challenge, our research focuses on developing models that are not only extremely lightweight but also robust and effective.

Channel Independent Strategy Recent advancements in the field of LTSF have seen a shift towards a Channel Independent (CI) approach, especially when dealing with multivariate time series data (Han et al., 2024). This strategy simplifies the forecasting process by focusing on individual univariate time series within the dataset. Instead of the traditional approach, which utilizes the entire multivariate historical data to predict future outcomes, the CI method finds a shared function $f : x _ { t - L + 1 : t } ^ { ( i ) } \in \mathbb { R } ^ { L } \to \bar { x } _ { t + 1 : t + H } ^ { ( i ) } \in \mathbb { R } ^ { H }$ for each univariate series. This approach provides a more targeted and simplified prediction model for each channel, reducing the complexity of accounting for inter-channel relationships.

As a result, the main goal of mainstream state-of-the-art models in recent years has shifted towards effectively predict by modeling long-term dependencies, including periodicity and trends, in univariate sequences. For instance, models like DLinear achieve this by extracting dominant periodicity from univariate sequences using a single linear layer (Zeng et al., 2023). More advanced models, such as PatchTST (Nie et al., 2023) and TiDE (Das et al., 2023), employ more complex structures on single channels to extract temporal dependencies, aiming for superior predictive performance. In this paper, we adopt this CI strategy as well and focus on how to create an even more lightweight yet effective approach for capturing long-term dependencies in singlechannel time series.

# 3.2. SparseTSF

Given that the data to be forecasted often exhibits constant, periodicity a priori (e.g., electricity consumption and traffic flow typically have fixed daily cycles), we propose the Cross-Period Sparse Forecasting technique to enhance the extraction of long-term sequential dependencies while reducing the model’s parameter scale. Utilizing a single linear layer to model the LTSF task within this framework leads to our SparseTSF model, as illustrated in Figure 2.

Cross-Period Sparse Forecasting Assuming that the time series $x _ { t - L + 1 : t } ^ { ( i ) }$ has a known periodicity $w$ , the first step is to downsample the original series into $w$ subsequences of length $n = \left\lfloor { \frac { L } { w } } \right\rfloor$ . A model with shared parameters is then applied to these subsequences for prediction. After prediction, the $w$ subsequences, each of length $\begin{array} { r } { m = \left\lfloor \frac { H } { w } \right\rfloor } \end{array}$ are upsampled back to a complete forecast sequence of length $H$ .

Intuitively, this forecasting process appears as a sliding forecast with a sparse interval of $w$ , performed by a fully connected layer with parameter sharing within a constant period $w$ . This can be viewed as a model performing sparse sliding prediction across periods.

Technically, the downsampling process is equivalent to reshaping $x _ { t - L + 1 : t } ^ { ( i ) }$ into a $n \times w$ matrix, which is then transposed to a $w \times n$ matrix. The sparse sliding prediction is equivalent to applying a linear layer of size $n \times m$ on the last dimension of the matrix, resulting in a $w \times m$ matrix. The upsampling step is equivalent to transposing the $w \times m$ matrix and reshaping it back into a complete forecast sequence of length $H$ .

However, this approach currently still faces two issues: (i) loss of information, as only one data point per period is utilized for prediction, while the rest are ignored; and (ii) amplification of the impact of outliers, as the presence of extreme values in the downsampled subsequences can directly affect the prediction.

To address these issues, we additionally perform a sliding aggregation on the original sequence before executing sparse prediction, as depicted in Figure 2. Each aggregated data point incorporates information from other points within its surrounding period, addressing issue (i). Moreover, as the aggregated value is essentially a weighted average of surrounding points, it mitigates the impact of outliers, thus resolving issue (ii). Technically, this sliding aggregation can be implemented using a 1D convolution with zero-padding and a kernel size of $2 \times \ \lfloor { \frac { w } { 2 } } \rfloor \ + \ 1$ . The process can be formulated as follows:

$$
\begin{array} { r } { x _ { t - L + 1 : t } ^ { ( i ) } = x _ { t - L + 1 : t } ^ { ( i ) } + \mathrm { C o n v } 1 \mathrm { D } ( x _ { t - L + 1 : t } ^ { ( i ) } ) } \end{array}
$$

Instance Normalization Time series data often exhibit distributional shifts between training and testing datasets. Recent studies have shown that employing simple sample normalization strategies between the input and output of models can help mitigate this issue (Kim et al., 2021; Zeng et al., 2023). In our work, we also utilize a straightforward normalization strategy. Specifically, we subtract the mean of the sequence from itself before it enters the model and add it back after the model’s output. This process is formulated as follows:

$$
\begin{array} { r l } & { x _ { t - L + 1 : t } ^ { ( i ) } = x _ { t - L + 1 : t } ^ { ( i ) } - \mathbb { E } _ { t } ( x _ { t - L + 1 : t } ^ { ( i ) } ) , } \\ & { \bar { x } _ { t + 1 : t + H } ^ { ( i ) } = \bar { x } _ { t + 1 : t + H } ^ { ( i ) } + \mathbb { E } _ { t } ( x _ { t - L + 1 : t } ^ { ( i ) } ) . } \end{array}
$$

Loss Function In alignment with current mainstream practices in the field, we adopt the classic Mean Squared Error (MSE) as the loss function for SparseTSF. This func$\bar { x } _ { t + 1 : t + H } ^ { ( i ) }$ ures the discrepancy betweeand the actual ground truth $y _ { t + 1 : t + H } ^ { ( i ) }$ icted values. It is formulated as:

$$
\mathcal { L } = \frac { 1 } { C } \sum _ { i = 1 } ^ { C } \Big | \Big | y _ { t + 1 : t + H } ^ { ( i ) } - \bar { x } _ { t + 1 : t + H } ^ { ( i ) } \Big | \Big | _ { 2 } ^ { 2 } .
$$

![](images/72d29ed6073646f6a29f9f584f3708a07bb38db09750cffefe04490e24b567eb.jpg)  
Figure 2: SparseTSF architecture.

# 3.3. Theoretical Analysis

In this section, we provide a theoretical analysis of the SparseTSF model, focusing on its parameter efficiency and the effectiveness of the Sparse technique. The relevant theoretical proofs are provided in Appendix B.

# 3.3.1. PARAMETER EFFICIENCY OF SPARSETSF

Theorem 3.1. Given a historical look-back window length $L _ { i }$ , a forecast horizon $H$ , and a constant periodicity $w$ , the total number of parameters required for the SparseTSF model is $\begin{array} { r } { \left\lfloor { \frac { L } { w } } \right\rfloor \times \left\lfloor { \frac { H } { w } } \right\rfloor + 2 \times \left\lfloor { \frac { w } { 2 } } \right\rfloor + 1 } \end{array}$ .

In LTSF tasks, the look-back window length $L$ and forecast horizon $H$ are usually quite large, for instance, up to 720, while the intrinsic periodicity $w$ of the data is also typically large, such as 24. In this scenario, $\begin{array} { r } { \left\lfloor \frac { L } { w } \right\rfloor \times \left\lfloor \frac { H } { w } \right\rfloor + 2 \times \left\lfloor \frac { w } { 2 } \right\rfloor \dot { + } } \end{array}$ $1 \ll L \times H$ . This means that the parameter scale of the SparseTSF model is much lighter than even the simplest single-layer linear model. This demonstrates the lightweight architecture of the SparseTSF model.

# 3.3.2. EFFECTIVENESS OF SPARSETSF

The time series targeted for long-term forecasting often exhibits constant periodicity. Here, we first define the representation of such a sequence $X$ .

Definition 3.2. Consider a univariate time series $X$ with a known period $w$ , which can be decomposed into a periodic component $P ( t )$ and a trend component $T ( t )$ , such that $X ( t ) = P ( t ) + T ( t )$ . Here, $P ( t )$ represents the periodic part and satisfies the condition:

$$
P ( t ) = P ( t + w ) .
$$

Furthermore, we can derive the form of the modeling task after downsampling.

In the context of a truncated subsequence $x _ { t - L + 1 : t }$ of $X ( t )$ and its corresponding future sequence $x _ { t + 1 : t + H }$ to be forecasted, the conventional approach involves using $x _ { t - L + 1 : t }$

directly to predict $x _ { t + 1 : t + H }$ , essentially estimating the function:

$$
x _ { t + 1 : t + H } = f ( x _ { t - L + 1 : t } )
$$

However, with the application of the Sparse technique, this forecasting task transforms into predicting downsampled subsequences, as per Lemma 3.3.

Lemma 3.3. The SparseTSF model reformulates the forecasting task into predicting downsampled subsequences, namely:

$$
x _ { t + 1 : t + m } ^ { \prime } = f ( x _ { t - n + 1 : t } ^ { \prime } )
$$

Combining Definition 3.2 and Lemma 3.3, we can further deduce Theorem 3.4.

Theorem 3.4. Given a time series dataset that satisfies Definition 3.2, the SparseTSF model’s formulation becomes:

$$
p _ { t + 1 : t + m } ^ { \prime } + t _ { t + 1 : t + m } ^ { \prime } = f ( p _ { t - n + 1 : t } ^ { \prime } + t _ { t - n + 1 : t } ^ { \prime } )
$$

where, for any $i \in [ t - n + 1 : t + m ]$ and $j \in [ t - n + 1$ $t + m ]$ , satisfying:

$$
p _ { i } ^ { \prime } = p _ { j } ^ { \prime }
$$

Theorem 3.4 implies that the task of the SparseTSF model effectively transforms into predicting future trend components (i.e., $t ^ { \prime }$ ), using the constant periodic components (i.e., $p ^ { \prime } )$ as a reference. This process effectively separates the periodic components, which are no longer explicitly modeled, allowing the model to focus more on the trend variations.

Intuitively, We can further validate this finding from the perspective of autocorrelation, a powerful tool for identifying patterns such as seasonality or periodicity in time series data.

Definition 3.5 (AutoCorrelation Function (ACF) (Madsen, 2007)). Given a time series $\{ X _ { t } \}$ , where $t$ represents discrete time points, the ACF at lag $k$ is defined as:

$$
\operatorname { A C F } ( k ) = { \frac { \sum _ { t = 1 } ^ { N - k } ( X _ { t } - \mu ) ( X _ { t + k } - \mu ) } { \sum _ { t = 1 } ^ { N } ( X _ { t } - \mu ) ^ { 2 } } }
$$

where $N$ is the total number of observations in the time series, $X _ { t }$ is the value of the series at time $t$ , $X _ { t + k }$ is the value of the series at time $t + k$ , and $\mu$ is the mean of the series $\{ X _ { t } \}$ .

![](images/0df19de9cdc0d57534cce49af7135c787b65c2a71201d0b0d4c578715a46afd3.jpg)  
Figure 3: Comparison of autocorrelation in original and downsampled subsequences for the first channel in the ETTh1 dataset.

The lag time $k$ in the ACF reveals the periodic patterns in the series, that is, when $k$ equals the periodic length of the series, the ACF value typically shows a significant peak. As shown in Figure 3, the original sequence exhibits clear periodicity, while the downsampled subsequences retain only trend characteristics. This demonstrates that, through its downsampling strategy, the SparseTSF model can efficiently separate and extract accurate periodic features from time series data. This not only reduces the complexity of the model but also enables it to focus on predicting trend variations, thereby exhibiting impressive performance in LTSF tasks.

In summary, the SparseTSF model’s design, characterized by its parameter efficiency and focus on decoupling periodic features, makes it well-suited for LTSF tasks, especially in scenarios where the data exhibits clear periodic patterns.

# 4. Experiments

In this section, we present the experimental results of SparseTSF on mainstream LTSF benchmarks. Additionally, we discuss the efficiency advantages brought by the lightweight architecture of SparseTSF. Furthermore, we conduct ablation studies and analysis to further reveal the effectiveness of the Sparse technique.

# 4.1. Experimental Setup

Datasets We conducted experiments on four mainstream LTSF datasets that exhibit daily periodicity. These datasets include ETTh1&ETTh21, Electricity2, and Traffic3. The details of these datasets are presented in Table 1.

Table 1: Summary of datasets.   

<table><tr><td>Datasets</td><td>ETTh1 &amp; ETTh2</td><td>Electricity</td><td>Traffic</td></tr><tr><td>Channels</td><td>7</td><td>321</td><td>862</td></tr><tr><td>Frequency</td><td>hourly</td><td>hourly</td><td>hourly</td></tr><tr><td>Timesteps</td><td>17,420</td><td>26,304</td><td>17,544</td></tr></table>

Baselines We compared our approach with state-of-theart and representative methods in the field. These include Informer (Zhou et al., 2021), Autoformer (Wu et al., 2021), Pyraformer (Liu et al., 2022b), FEDformer (Zhou et al., 2022b), Film (Zhou et al., 2022a), TimesNet (Wu et al., 2023), and PatchTST (Nie et al., 2023). Additionally, we specifically compared SparseTSF with lightweight models, namely DLinear (Zeng et al., 2023) and FITS (Xu et al., 2024). Following FITS, SparseTSF defaults to a look-back length of 720.

Environment All experiments in this study were implemented using PyTorch (Paszke et al., 2019) and conducted on a single NVIDIA RTX 4090 GPU with 24GB of memory. More experimental details are provided in Appendix A.2.

# 4.2. Main Results

Table 2 presents a performance comparison between SparseTSF and other baseline models4. It is observable that SparseTSF ranks within the top two in all scenarios, achieving or closely approaching state-of-the-art levels with a significantly smaller parameter scale. This emphatically demonstrates the superiority of the Sparse technique proposed in this paper. Specifically, the Sparse technique is capable of more effectively extracting the periodicity and trends from data, thereby enabling exceptional predictive performance in long horizon scenarios. Additionally, the standard deviation of SparseTSF’s results is notably small. In most cases, the standard deviation across 5 runs is within 0.001, which strongly indicates the robustness of the SparseTSF model.

# 4.3. Efficiency Advantages of SparseTSF

Beyond its powerful predictive performance, another significant benefit of the SparseTSF model is its extreme lightweight nature. Previously, Figure 1 visualized the parameter-performance comparison of SparseTSF with other mainstream models. Here, we further present a comprehensive comparison between SparseTSF and these baseline models in terms of both static and runtime metrics, including:

Table 2: MSE results of multivariate long-term time series forecasting comparing SparseTSF with other mainstream models. The top two results are highlighted in bold. The reported results of SparseTSF are averaged over 5 runs with standard deviation included. ’Imp.’ denotes the improvement compared to the best-performing baseline models.   

<table><tr><td>Dataset</td><td colspan="4">ETTh1</td><td colspan="4">ETTh2</td><td colspan="4">Electricity</td><td colspan="4">Traffic</td></tr><tr><td>Horizon</td><td>96</td><td>192</td><td>336</td><td>720</td><td>96</td><td>192</td><td>336</td><td>720</td><td>96</td><td>192</td><td>336</td><td>720</td><td>96</td><td>192</td><td>336</td><td>720</td></tr><tr><td>Informer (2021)</td><td>0.865</td><td>1.008</td><td>1.107</td><td>1.181</td><td>3.755</td><td>5.602</td><td>4.721</td><td>3.647</td><td>0.274</td><td>0.296</td><td>0.300</td><td>0.373</td><td>0.719</td><td>0.696</td><td>0.777</td><td>0.864</td></tr><tr><td>Autoformer (2021)</td><td>0.449</td><td>0.500</td><td>0.521</td><td>0.514</td><td>0.358</td><td>0.456</td><td>0.482</td><td>0.515</td><td>0.201</td><td>0.222</td><td>0.231</td><td>0.254</td><td>0.613</td><td>0.616</td><td>0.622</td><td>0.660</td></tr><tr><td>Pyraformer (2022b)</td><td>0.664</td><td>0.790</td><td>0.891</td><td>0.963</td><td>0.645</td><td>0.788</td><td>0.907</td><td>0.963</td><td>0.386</td><td>0.386</td><td>0.378</td><td>0.376</td><td>2.085</td><td>0.867</td><td>0.869</td><td>0.881</td></tr><tr><td>FEDformer (2022b)</td><td>0.376</td><td>0.420</td><td>0.459</td><td>0.506</td><td>0.346</td><td>0.429</td><td>0.496</td><td>0.463</td><td>0.193</td><td>0.201</td><td>0.214</td><td>0.246</td><td>0.587</td><td>0.604</td><td>0.621</td><td>0.626</td></tr><tr><td>FiLM (2022a)</td><td>0.371</td><td>0.414</td><td>0.442</td><td>0.465</td><td>0.284</td><td>0.357</td><td>0.377</td><td>0.439</td><td>0.154</td><td>0.164</td><td>0.188</td><td>0.236</td><td>0.416</td><td>0.408</td><td>0.425</td><td>0.520</td></tr><tr><td>TimesNet (2023)</td><td>0.384</td><td>0.436</td><td>0.491</td><td>0.521</td><td>0.340</td><td>0.402</td><td>0.452</td><td>0.462</td><td>0.168</td><td>0.184</td><td>0.198</td><td>0.220</td><td>0.593</td><td>0.617</td><td>0.629</td><td>0.640</td></tr><tr><td>PatchTST (2023)</td><td>0.370</td><td>0.413</td><td>0.422</td><td>0.447</td><td>0.274</td><td>0.341</td><td>0.329</td><td>0.379</td><td>0.129</td><td>0.147</td><td>0.163</td><td>0.197</td><td>0.360</td><td>0.379</td><td>0.392</td><td>0.432</td></tr><tr><td>DLinear (2023)</td><td>0.374</td><td>0.405</td><td>0.429</td><td>0.440</td><td>0.338</td><td>0.381</td><td>0.400</td><td>0.436</td><td>0.140</td><td>0.153</td><td>0.169</td><td>0.203</td><td>0.410</td><td>0.423</td><td>0.435</td><td>0.464</td></tr><tr><td>FITS (2024)</td><td>0.375</td><td>0.408</td><td>0.429</td><td>0.427</td><td>0.274</td><td>0.333</td><td>0.340</td><td>0.374</td><td>0.138</td><td>0.152</td><td>0.166</td><td>0.205</td><td>0.401</td><td>0.407</td><td>0.420</td><td>0.456</td></tr><tr><td>SparseTSF (ours)</td><td>0.359</td><td>0.397</td><td>0.404</td><td>0.417</td><td>0.267</td><td>0.314</td><td>0.312</td><td>0.370</td><td>0.138</td><td>0.146</td><td>0.164</td><td>0.203</td><td>0.382</td><td>0.388</td><td>0.402</td><td>0.445</td></tr><tr><td>Imp.</td><td>±0.006 +0.011</td><td>±0.002 +0.008</td><td>±0.001 +0.018</td><td>±0.001 +0.010</td><td>±0.005 +0.007</td><td>±0.003 +0.019</td><td>±0.004 +0.017</td><td>±0.001 +0.004</td><td>±0.001 -0.009</td><td>±0.001 +0.001</td><td>±0.001 -0.001</td><td>±0.001 -0.006</td><td>±0.001 -0.022</td><td>±0.001 -0.009</td><td>±0.001 -0.010</td><td>±0.002 -0.013</td></tr></table>

1. Parameters: The total number of trainable parameters in the model, representing the model’s size. 2. MACs (Multiply-Accumulate Operations): A common measure of computational complexity in neural networks, indicating the number of multiply-accumulate operations required by the model. 3. Max Memory: The maximum memory usage during the model training process. 4. Epoch Time: The training duration for a single epoch. This metric was averaged over 3 runs.

Table 3: Static and runtime metrics of SparseTSF and other mainstream models on the Electricity Dataset with a forecast horizon of 720. Here, the look-back length for each model is set to be consistent with their respective official papers, such as 336 for DLinear and 720 for FITS.   

<table><tr><td>Model</td><td>Parameters</td><td>MACs</td><td>Max Mem.(MB)</td><td>Epoch Time(s)</td></tr><tr><td>Informer (2021)</td><td>12.53 M</td><td>3.97 G</td><td>969.7</td><td>70.1</td></tr><tr><td>Autoformer (2021)</td><td>12.22 M</td><td>4.41 G</td><td>2631.2</td><td>107.7</td></tr><tr><td>FEDformer (2022b)</td><td>17.98 M</td><td>4.41 G</td><td>1102.5</td><td>238.7</td></tr><tr><td>FiLM (2022a)</td><td>12.22 M</td><td>4.41 G</td><td>1773.9</td><td>78.3</td></tr><tr><td>PatchTST (2023)</td><td>6.31 M</td><td>11.21 G</td><td>10882.3</td><td>290.3</td></tr><tr><td>DLinear (2023)</td><td>485.3 K</td><td>156.0 M</td><td>123.8</td><td>25.4</td></tr><tr><td>FITS (2024)</td><td>10.5 K</td><td>79.9 M</td><td>496.7</td><td>35.0</td></tr><tr><td>SparseTSF (Ours)</td><td>0.92 K</td><td>12.71 M</td><td>125.2</td><td>31.3</td></tr></table>

Table 3 displays the comparative results. It is evident that SparseTSF significantly outperforms other models in terms of static metrics like the number of parameters and MACs, being over ten times smaller than the next best model. This characteristic allows SparseTSF to be deployed on devices with very limited computational resources. Furthermore, in terms of runtime metrics, Max Memory and Epoch Time,

SparseTSF significantly outperforms other mainstream models, rivaling the existing lightweight models (i.e., DLinear and FITS). Herein, DLinear benefits from a shorter lookback length, achieving the lowest overhead, while FITS and SparseTSF incur additional overhead due to extra operations (i.e., Fourier transformation and resampling).

Table 4: Comparison of the scale of parameters on Electricity dataset between SparseTSF and FITS models under different configurations of look-back length and forecast horizon, where SparseTSF operates with $w = 2 4$ and FITS employs COF at the $2 ^ { t h }$ harmonic.   

<table><tr><td>Model</td><td colspan="4">SparseTSF (Ours)</td><td colspan="4">FITS (2024)</td></tr><tr><td>Look-back Horizon</td><td rowspan="2">96</td><td rowspan="2">192</td><td rowspan="2">336</td><td rowspan="2">720</td><td rowspan="2">96</td><td rowspan="2">192</td><td rowspan="2">336</td><td rowspan="2">720</td></tr><tr><td></td></tr><tr><td>96</td><td>41</td><td>57</td><td>81</td><td>145</td><td>840</td><td>1,218</td><td>2,091</td><td>5,913</td></tr><tr><td>192</td><td>57</td><td>89</td><td>137</td><td>265</td><td>1,260</td><td>1,624</td><td>2,542</td><td>6,643</td></tr><tr><td>336</td><td>81</td><td>137</td><td>221</td><td>445</td><td>1,890</td><td>2,233</td><td>3,280</td><td>7,665</td></tr><tr><td>720</td><td>145</td><td>265</td><td>445</td><td>925</td><td>3,570</td><td>3,857</td><td>5,125</td><td>10,512</td></tr></table>

Additionally, we conducted a comprehensive comparison with FITS, a recent milestone work in the field of LTSF model lightweight progression. The results in Table 4 reveal that SparseTSF significantly surpasses FITS in terms of parameter scale under any input-output length configuration. Therefore, SparseTSF marks another significant advancement in the journey towards lightweight LTSF models.

# 4.4. Ablation Studies and Analysis

Beyond its ultra-lightweight characteristics, the Sparse technique also possesses a robust capability to extract periodic features, which we will delve further into in this section.

Effectiveness of the Sparse Technique The Sparse technique, combined with a simple single-layer linear model, forms the core of our proposed model, SparseTSF. Additionally, the Sparse technique can be integrated with other foundational models, including the Transformer (Vaswani et al., 2017) and GRU (Cho et al., 2014) models. As demonstrated in the results of Table 5, the incorporation of the Sparse technique significantly enhances the performance of all models, including Linear, Transformer, and GRU. Specifically, the Linear model showed an average improvement of $4 . 7 \%$ , the Transformer by $2 1 . 4 \%$ , and the GRU by $12 . 4 \%$ . These results emphatically illustrate the efficacy of the Sparse technique. Therefore, the Sparse technique can substantially improve the performance of base models in LTSF tasks.

Table 5: Ablation MSE results of the Sparse technique. All results are collected with a unified channel-independent and instance normalization strategy. The ’Boost’ indicates the percentage of performance improvement after incorporating the Sparse technique.   

<table><tr><td>Dataset</td><td colspan="4">ETTh1</td><td colspan="4">ETTh2</td></tr><tr><td>Horizon</td><td>96</td><td>192</td><td>336</td><td>720</td><td>96</td><td>192</td><td>336</td><td>720</td></tr><tr><td>Linear</td><td>0.371</td><td>0.460</td><td>0.417</td><td>0.424</td><td>0.257</td><td>0.337</td><td>0.336</td><td>0.391</td></tr><tr><td>+sparse</td><td>0.359</td><td>0.397</td><td>0.404</td><td>0.417</td><td>0.267</td><td>0.314</td><td>0.312</td><td>0.370</td></tr><tr><td>Boost</td><td>3.3%</td><td>13.8%</td><td>3.1%</td><td>1.7%</td><td>-3.9%</td><td>6.9%</td><td>7.1%</td><td>5.3%</td></tr><tr><td>Transformer</td><td>0.697</td><td>0.732</td><td>0.714</td><td>0.770</td><td>0.340</td><td>0.376</td><td>0.366</td><td>0.468</td></tr><tr><td>+sparse</td><td>0.406</td><td>0.442</td><td>0.446</td><td>0.489</td><td>0.322</td><td>0.380</td><td>0.353</td><td>0.432</td></tr><tr><td>Boost</td><td>41.7%</td><td>39.6%</td><td>37.5%</td><td>36.5%</td><td>5.2%</td><td>-1.0%</td><td>3.6%</td><td>7.7%</td></tr><tr><td>GRU</td><td>0.415</td><td>0.529</td><td>0.512</td><td>0.620</td><td>0.296</td><td>0.345</td><td>0.363</td><td>0.454</td></tr><tr><td>+sparse</td><td>0.356</td><td>0.391</td><td>0.437</td><td>0.455</td><td>0.282</td><td>0.332</td><td>0.356</td><td>0.421</td></tr><tr><td>Boost</td><td>14.1%</td><td>26.1%</td><td>14.7%</td><td>26.7%</td><td>4.8%</td><td>3.7%</td><td>1.9%</td><td>7.2%</td></tr></table>

Representation Learning of the Sparse Technique In Section 3.3, we theoretically analyzed the reasons why the Sparse technique can enhance the performance of forecasting tasks. Here, we further reveal the role of the Sparse technique from a representation learning perspective. Figure 3 shows the distribution of normalized weights for both the trained Linear model and the SparseTSF model. The weight of the Linear model is an $L \times H$ matrix, which can be directly obtained. However, as the SparseTSF model is a sparse model, we need to acquire its equivalent weights. To do this, we first input $H$ one-hot encoded vectors of length $L$ into the SparseTSF model (when $L$ equals $H$ , this can be simplified to a diagonal matrix, i.e., diagonal elements are 1, and other elements are 0). We then obtain and transpose the corresponding output to get the equivalent $L \times H$ weight matrix of SparseTSF. When $L$ equals $H$ , this process is formulated as:

$$
\boldsymbol { w e i g h t ^ { \prime } } = S p a r s e T S F ( \begin{array} { l l l l } { 1 } & { 0 } & { \dots } & { 0 } \\ { 0 } & { 1 } & { \dots } & { 0 } \\ { \dots } & { \dots } & { \dots } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 1 } \end{array} ) ^ { \top } .
$$

From the visualization in Figure 4, two observations can be made: (i) The Linear model can learn evenly spaced weight distribution stripes (i.e., periodic features) from the data, indicating that single linear layer can already extract the primary periodic characteristics from a univariate series with the CI strategy. These findings are consistent with previous research conclusions (Zeng et al., 2023). (ii) Compared to the Linear model, SparseTSF learns more distinct evenly spaced weight distribution stripes, indicating that SparseTSF has a stronger capability in extracting periodic features. This phenomenon aligns with the conclusions of Section 3.3.

Therefore, the Sparse technique can enhance the model’s performance in LTSF tasks by strengthening its ability to extract periodic features from data.

Impact of the Hyperparameter $w$ The Sparse technique relies on the manual setting of the hyperparameter $w$ , which represents the a priori main period. Here, we delve into the influence of different values of $w$ on the forecast outcomes. As indicated in the results from Table 6, SparseTSF exhibits optimal performance when $w = 2 4$ , aligning with the intrinsic main period of the data. Conversely, when $w$ diverges from 24, a slight decline in performance is observed. This suggests that the hyperparameter $w$ should ideally be set consistent with the data’s a priori main period.

Table 6: MSE results of SparseTSF on ETTh1 with varied hyperparameters $w$ .   

<table><tr><td>Horizon</td><td>SparseTSF (w=6)</td><td>SparseTSF (w=12)</td><td>SparseTSF (w=24)</td><td>SparseTSF (w=48)</td><td>FITS (2024)</td><td>DLinear (2023)</td><td>PatchTST (2023)</td></tr><tr><td>96</td><td>0.376</td><td>0.369</td><td>0.359</td><td>0.380</td><td>0.375</td><td>0.374</td><td>0.370</td></tr><tr><td>192</td><td>0.410</td><td>0.402</td><td>0.397</td><td>0.400</td><td>0.408</td><td>0.405</td><td>0.413</td></tr><tr><td>336</td><td>0.408</td><td>0.406</td><td>0.404</td><td>0.399</td><td>0.429</td><td>0.429</td><td>0.422</td></tr><tr><td>720</td><td>0.427</td><td>0.423</td><td>0.417</td><td>0.427</td><td>0.427</td><td>0.440</td><td>0.447</td></tr><tr><td>Avg.</td><td>0.405</td><td>0.400</td><td>0.394</td><td>0.402</td><td>0.410</td><td>0.412</td><td>0.413</td></tr></table>

In practical scenarios, datasets requiring long-term forecasting often exhibit inherent periodicity, such as daily or weekly cycles, common in domains like electricity, transportation, energy, and consumer goods consumption. Therefore, empirically identifying the predominant period and setting the appropriate $w$ for such data is both feasible and straightforward. However, for data lacking clear periodicity and patterns, such as financial data, current LTSF models may not be effective (Zeng et al., 2023). Thus, the SparseTSF model may not be the preferred choice for these types of data. Nonetheless, we will further discuss the existing limitations and potential improvements of the SparseTSF model in the Section 5.1.

Generalization Ability of the SparseTSF Model The Sparse technique enhances the model’s ability to extract periodic features from data. Therefore, the generalization capability of a trained SparseTSF model on different datasets with the same principal periodicity is promising. To investigate this, we further studied the cross-domain generalization performance of the SparseTSF model (i.e., training on a dataset from one domain and testing on a dataset from another). Specifically, we examined the performance from

![](images/34234cdb806c68c8e329db38c3f032eb65954c2f3979ad0423531cc46cc728cd.jpg)  
Figure 4: Visualization of normalized weights of the model trained on the ETTh1 dataset with both look-back length (X-axis) and forecast horizon (Y-axis) of 96.

ETTh2 to ETTh1, which are datasets of the same type but collected from different machines, each with 7 variables. Additionally, we explored the performance from Electricity to ETTh1, where these datasets originate from different domains and have a differing number of variables (i.e., Electricity has 321 variables). On datasets with different numbers of variables, models trained with traditional nonCI strategies (like Informer) cannot transfer, whereas those trained with CI strategies (like PatchTST) can, due to the decoupling of CI strategies from channel relationships. These datasets all have a daily periodicity, i.e., a prior predominant period of $w = 2 4$ .

Table 7: Comparison of generalization capabilities between SparseTSF and other mainstream models. ’Dataset $\Delta $ Dataset $\mathbf { B } ^ { \prime }$ indicates training and validation on the training and validation sets of Dataset A, followed by testing on the test set of Dataset B.   

<table><tr><td>Dataset</td><td colspan="4">ETTh2 → ETTh1</td><td colspan="4">Electricity → ETTh1</td></tr><tr><td>Horizon</td><td>96</td><td>192</td><td>336</td><td>720</td><td>96</td><td>192</td><td>336</td><td>720</td></tr><tr><td>Informer (2021)</td><td>0.844</td><td>0.921</td><td>0.898</td><td>0.829</td><td></td><td>1///</td><td>////</td><td></td></tr><tr><td>Autoformer (2021)</td><td>0.978</td><td>1.058</td><td>0.944</td><td>0.921</td><td></td><td></td><td></td><td></td></tr><tr><td>FEDformer (2022b)</td><td>0.878</td><td>0.927</td><td>0.939</td><td>0.967</td><td>1/7.</td><td></td><td></td><td></td></tr><tr><td>FiLM (2022a)</td><td>0.876</td><td>0.904</td><td>0.919</td><td>0.925</td><td>|</td><td></td><td></td><td>1//7</td></tr><tr><td>PatchTST (2023)</td><td>0.449</td><td>0.478</td><td>0.482</td><td>0.476</td><td>0.400</td><td>0.424</td><td>0.475</td><td>0.472</td></tr><tr><td>DLinear (2023)</td><td>0.430</td><td>0.478</td><td>0.458</td><td>0.506</td><td>0.397</td><td>0.428</td><td>0.447</td><td>0.470</td></tr><tr><td>Fits (2024)</td><td>0.419</td><td>0.427</td><td>0.428</td><td>0.445</td><td>0.380</td><td>0.414</td><td>0.440</td><td>0.448</td></tr><tr><td>SparseTSF (Ours)</td><td>0.370</td><td>0.401</td><td>0.412</td><td>0.419</td><td>0.373</td><td>0.409</td><td>0.433</td><td>0.439</td></tr></table>

Experimental results, as shown in Table 7, reveal that SparseTSF outperforms other models in both similar domain generalization (ETTh2 to ETTh1) and less similar domain generalization (Electricity to ETTh1). It is expected that performance on ETTh2 to ETTh1 would be superior to Electricity to ETTh1. Furthermore, in both scenarios, the generalization performance of SparseTSF is nearly on par with the performance of direct modeling in the SparseTSF source domain as shown in Table 2 and surpasses other baselines that model directly in the source domain. This robustly demonstrates the generalization capability of SparseTSF, indirectly proving the Sparse technique’s ability to extract stable periodic features.

Therefore, the SparseTSF model exhibits outstanding generalization capabilities. This characteristic is highly beneficial for the application of the SparseTSF model in scenarios involving small samples and low-quality data.

# 5. Discussion

# 5.1. Limitations and Future Work

The SparseTSF model proposed in this paper excels in handling data with a stable main period, demonstrating enhanced feature extraction capabilities and an extremely lightweight architecture. However, there are two scenarios where SparseTSF may not be as effective:

1. Ultra-Long Periods: In cases involving ultra-long periods (for example, periods exceeding 100), the Sparse technique results in overly sparse parameter connections. Consequently, SparseTSF does not perform optimally in such scenarios.

2. Multiple Periods: SparseTSF may struggle with data that intertwines multiple periods, as the Sparse technique can only downsample and decompose one main period.

We have further investigated the performance of SparseTSF in these scenarios in Appendix C and concluded that: (1) in ultra-long period scenarios, a denser connected model would be a better choice; (2) SparseTSF can still perform excellently in some multi-period scenarios (such as daily periods superimposed with weekly periods).

Finally, one of our key future research directions is to further address the these potential limitations by designing additional modules to enhance SparseTSF’s ability, thus achieving a balance between performance and parameter size.

# 5.2. Differences Compared to Existing Methods

The Sparse technique proposed in this paper involves downsampling/upsampling to achieve periodicity/trend decoupling. It may share a similar idea with existing methods, as downsampling/upsampling and periodic/trend decomposition techniques are prevalent in related literature nowadays. Specifically, we provide a detailed analysis of the differences with respect to N-HiTS (Challu et al., 2023) and OneShotSTL (He et al., 2023) as follows, and present the comparison results in Appendix D.4.

SparseTSF Compared to N-HiTS N-HiTS incorporates novel hierarchical interpolation and multi-rate data sampling techniques to achieve better results (Challu et al., 2023). The downsampling and upsampling techniques proposed in SparseTSF are indeed quite different from those used in N-HiTS, including:

• The downsampling and upsampling in SparseTSF occur before and after the model’s prediction process, respectively, whereas N-HiTS conducts these operations within internally stacked modules.

• SparseTSF’s downsampling involves resampling by a factor of $w$ to $w$ subsequences of length $L / w$ , which is technically equivalent to matrix reshaping and transposition, whereas N-HiTS employs downsampling through max-pooling.

• SparseTSF’s upsampling involves transposing and reshaping the predicted subsequences back to the original sequence, whereas N-HiTS achieves upsampling through interpolation.

SparseTSF Compared to OneShotSTL Seasonal-trend decomposition (STD) is a classical and powerful tool for time series forecasting, and OneShotSTL makes a great contribution to advancing the lightweight long-term forecasting process, featuring fast, lightweight, and powerful capabilities (He et al., 2023). However, SparseTSF differs significantly from OneShotSTL in several aspects:

• SparseTSF is a neural network model while OneShotSTL is a non-neural network method focused on online forecasting.

• OneShotSTL minimizes residuals and calculates trend and seasonal subseries separately from the original sequence with lengths of $L$ , whereas our SparseTSF resamples the original sequence into $w$ subseries of length $L / w$ with a constant period $w$ .

• OneShotSTL accelerates inference by optimizing the original computation for online processing, while SparseTSF achieves lightweighting by using parametersharing linear layers for prediction across all subseries.

# 6. Conclusion

In this paper, we introduce the Cross-Period Sparse Forecasting technique and the corresponding SparseTSF model. Through detailed theoretical analysis and experimental validation, we demonstrated the lightweight nature of the SparseTSF model and its capability to extract periodic features effectively. Achieving competitive or even surpassing the performance of current state-of-the-art models with a minimal parameter scale, SparseTSF emerges as a strong contender for deployment in computation resourceconstrained environments. Additionally, SparseTSF exhibits potent generalization capabilities, opening new possibilities for applications in transferring to small samples and low-quality data scenarios. SparseTSF stands as another milestone in the journey towards lightweight models in the field of long-term time series forecasting. Finally, we aim to further tackle the challenges associated with extracting features from ultra-long-periodic and multi-periodic data in the future, striving to achieve an optimal balance between model performance and parameter size.

# Acknowledgements

This work is supported by Guangdong Major Project of Basic and Applied Basic Research (2019B030302002), National Natural Science Foundation of China (62072187), Guangzhou Development Zone Science and Technology Project (2021GH10) and the Major Key Project of PCL, China under Grant PCL2023A09.

# Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

# References

Bai, S., Kolter, J. Z., and Koltun, V. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv preprint arXiv:1803.01271,

2018.

Challu, C., Olivares, K. G., Oreshkin, B. N., Ramirez, F. G., Canseco, M. M., and Dubrawski, A. Nhits: Neural hierarchical interpolation for time series forecasting. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pp. 6989–6997, 2023.

Chang, C., Wang, W.-Y., Peng, W.-C., and Chen, T.-F. Llm4ts: Aligning pre-trained llms as data-efficient timeseries forecasters, 2024.

Cho, K., Van Merrienboer, B., Gulcehre, C., Bahdanau, ¨ D., Bougares, F., Schwenk, H., and Bengio, Y. Learning phrase representations using rnn encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078, 2014.

Das, A., Kong, W., Leach, A., Mathur, S., Sen, R., and Yu, R. Long-term forecasting with tide: Time-series dense encoder. arXiv preprint arXiv:2304.08424, 2023.

Deng, J., Song, X., Tsang, I. W., and Xiong, H. The bigger the better? rethinking the effective model scale in long-term time series forecasting. arXiv preprint arXiv:2401.11929, 2024.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Ekambaram, V., Jati, A., Nguyen, N., Sinthong, P., and Kalagnanam, J. Tsmixer: Lightweight mlp-mixer model for multivariate time series forecasting. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 459–469, 2023.

Franceschi, J.-Y., Dieuleveut, A., and Jaggi, M. Unsupervised scalable representation learning for multivariate time series. Advances in neural information processing systems, 32, 2019.

Han, L., Ye, H.-J., and Zhan, D.-C. The capacity and robustness trade-off: Revisiting the channel independent strategy for multivariate time series forecasting. IEEE Transactions on Knowledge and Data Engineering, 2024.

He, K., Chen, X., Xie, S., Li, Y., Dollar, P., and Girshick, ´ R. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 16000–16009, 2022.

He, X., Li, Y., Tan, J., Wu, B., and Li, F. Oneshotstl: One-shot seasonal-trend decomposition for online time series anomaly detection and forecasting. arXiv preprint arXiv:2304.01506, 2023.

Huang, Q., Shen, L., Zhang, R., Cheng, J., Ding, S., Zhou, Z., and Wang, Y. Hdmixer: Hierarchical dependency with extendable patch for multivariate time series forecasting. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 12608–12616, 2024a.

Huang, Q., Shen, L., Zhang, R., Ding, S., Wang, B., Zhou, Z., and Wang, Y. Crossgnn: Confronting noisy multivariate time series via cross interaction refinement. Advances in Neural Information Processing Systems, 36, 2024b.

Jin, M., Wang, S., Ma, L., Chu, Z., Zhang, J. Y., Shi, X., Chen, P.-Y., Liang, Y., Li, Y.-F., Pan, S., et al. Time-llm: Time series forecasting by reprogramming large language models. arXiv preprint arXiv:2310.01728, 2023.

Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., and Choo, J. Reversible instance normalization for accurate time-series forecasting against distribution shift. In International Conference on Learning Representations, 2021.

Kingma, D. P. and Ba, J. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Lin, S., Lin, W., Wu, W., Wang, S., and Wang, Y. Petformer: Long-term time series forecasting via placeholderenhanced transformer. arXiv preprint arXiv:2308.04791, 2023a.

Lin, S., Lin, W., Wu, W., Zhao, F., Mo, R., and Zhang, H. Segrnn: Segment recurrent neural network for long-term time series forecasting. arXiv preprint arXiv:2308.11200, 2023b.

Liu, M., Zeng, A., Chen, M., Xu, Z., Lai, Q., Ma, L., and Xu, Q. Scinet: Time series modeling and forecasting with sample convolution and interaction. Advances in Neural Information Processing Systems, 35:5816–5828, 2022a.

Liu, S., Yu, H., Liao, C., Li, J., Lin, W., Liu, A. X., and Dustdar, S. Pyraformer: Low-complexity pyramidal attention for long-range time series modeling and forecasting. In International conference on learning representations, 2022b.

Madsen, H. Time series analysis. CRC Press, 2007.

Nie, Y., H. Nguyen, N., Sinthong, P., and Kalagnanam, J. A time series is worth 64 words: Long-term forecasting with transformers. In International Conference on Learning Representations, 2023.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019.

Qiu, X., Hu, J., Zhou, L., Wu, X., Du, J., Zhang, B., Guo, C., Zhou, A., Jensen, C. S., Sheng, Z., et al. Tfb: Towards comprehensive and fair benchmarking of time series forecasting methods. arXiv preprint arXiv:2403.20150, 2024.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Wang, H., Peng, J., Huang, F., Wang, J., Chen, J., and Xiao, Y. Micn: Multi-scale local and global context modeling for long-term series forecasting. In The Eleventh International Conference on Learning Representations, 2022.

Wen, Q., Zhou, T., Zhang, C., Chen, W., Ma, Z., Yan, J., and Sun, L. Transformers in time series: A survey. arXiv preprint arXiv:2202.07125, 2022.

Wu, H., Xu, J., Wang, J., and Long, M. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. Advances in neural information processing systems, 34:22419–22430, 2021.

Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J., and Long, M. Timesnet: Temporal 2d-variation modeling for general time series analysis. In International Conference on Learning Representations, 2023.

Xu, Z., Zeng, A., and Xu, Q. Fits: Modeling time series with $1 0 k$ parameters. In The Twelfth International Conference on Learning Representations, 2024.

Xue, H. and Salim, F. D. Promptcast: A new promptbased learning paradigm for time series forecasting. IEEE Transactions on Knowledge and Data Engineering, 2023.

Zeng, A., Chen, M., Zhang, L., and Xu, Q. Are transformers effective for time series forecasting? In Proceedings of the AAAI conference on artificial intelligence, volume 37, pp. 11121–11128, 2023.

Zhang, T., Zhang, Y., Cao, W., Bian, J., Yi, X., Zheng, S., and Li, J. Less is more: Fast multivariate time series forecasting with light sampling-oriented mlp structures. arXiv preprint arXiv:2207.01186, 2022.

Zhang, X., Zhong, C., Zhang, J., Wang, T., and Ng, W. W. Robust recurrent neural networks for time series forecasting. Neurocomputing, 526:143–157, 2023.

Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., and Zhang, W. Informer: Beyond efficient transformer for long sequence time-series forecasting. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pp. 11106–11115, 2021.

Zhou, T., Ma, Z., Wen, Q., Sun, L., Yao, T., Yin, W., Jin, R., et al. Film: Frequency improved legendre memory model for long-term time series forecasting. Advances in Neural Information Processing Systems, 35:12677– 12690, 2022a.

Zhou, T., Ma, Z., Wen, Q., Wang, X., Sun, L., and Jin, R. Fedformer: Frequency enhanced decomposed transformer for long-term series forecasting. In International conference on machine learning, pp. 27268–27286. PMLR, 2022b.

# A. More Details of SparseTSF

# A.1. Overall Workflow

The complete workflow of SparseTSF is outlined in Algorithm 1, which takes a univariate historical look-back window $x _ { t - L + 1 : t }$ as input and outputs the corresponding forecast results $\bar { x } _ { t + 1 : t + H }$ . By integrating the CI strategy, i.e., modeling multiple channels using a model with shared parameters, multivariate time series forecasting can be effectively achieved.

# Algorithm 1 The Overall Pseudocode of SparseTSF

Require: Historical look-back window $\boldsymbol { x } _ { t - L + 1 : t } \in \mathbb { R } ^ { L }$ Ensure: Forecasting horizon $\bar { x } _ { t + 1 : t + H } \in \mathbb { R } ^ { H }$

1: $e _ { t } \gets \frac { \sum _ { i = t - L + 1 } ^ { t } { x _ { i } } } { L }$ /\* Calculate the mean of the look-back window $^ { * }$   
2: $x _ { t - L + 1 : t } \gets x _ { t - L + 1 : t } - e _ { t }$ $/ { * }$ Subtract the mean from each element $^ { * } /$   
3: $\begin{array} { l l l } { { x _ { t - L + 1 : t } } } & { {  } } & { { C o n \nu I d ( x _ { t - L + 1 : t } , 2 \times \lfloor \frac { w } { 2 } \rfloor + 1 ) \ + } } \end{array}$ $x _ { t - L + 1 : t }$ $/ *$ Apply 1D convolution on the original window $^ { * } /$   
4: $X ~ \gets ~ R e s h a p e ( x _ { t - L + 1 : t } , ( n , w ) )$ $/ { * }$ Reshape $x _ { t - L + 1 : t }$ into a $n \times w$ matrix $^ { * } /$   
5: $Y \gets L i n e a r ( X ^ { \top } ) ^ { \top } \qquad I ^ { * }$ Transpose, apply linear transformation $n  m$ , and transpose back $^ { * }$   
6: $\bar { x } _ { t + 1 : t + H } ~ \gets ~ R e s h a p e ( Y , ( H ) )$ $/ { * }$ Reshape $Y$ back into a length $H$ sequence $^ { * }$   
7: $\bar { x } _ { t + 1 : t + H } \gets \bar { x } _ { t + 1 : t + H } + e _ { t }$ $/ { * }$ Add the mean back to each element $^ { * } /$

Additionally, intuitively, SparseTSF can be perceived as a sparsely connected linear layer performing sliding prediction across periods, as depicted in Figure 5.

![](images/91364455a1552c2d946a337e6f7dfb39bfae8746f70cd9989d8ebf1d01d42ac2.jpg)  
Figure 5: Schematic illustration of SparseTSF.

# A.2. Experimental Details

We implemented SparseTSF in PyTorch (Paszke et al., 2019) and trained it using the Adam optimizer (Kingma & Ba, 2014) for 30 epochs, with a learning rate decay of 0.8 after the initial 3 epochs, and early stopping with a patience of 5. The dataset splitting follows the procedures of FITS and Autoformer, where the ETT datasets are divided into proportions of 6:2:2, while the other datasets are split into

proportions of 7:1:2.

SparseTSF has minimal hyperparameters due to its simple design. The period $w$ is set to the inherent cycle of the data (e.g., $w = 2 4$ for ETTh1) or to a smaller value if the data has an extremely long cycle (e.g., $w = 4$ for ETTm1). The choice of batch size depends on the size of the data samples (i.e., the number of channels). For datasets with fewer than 100 channels (such as ETTh1), the batch size is set to 256, while for datasets with fewer than 300 channels (such as Electricity), the batch size is set to 128. This setting maximizes the utilization of GPU parallel computing capabilities while avoiding GPU out-of-memory issues (i.e., with NVIDIA RTX 4090, 24GB). Additionally, the learning rate needs to be set relatively large (i.e., 0.02) due to the very small number of learnable parameters in SparseTSF. The complete details can be found in our official repository5.

The baseline results in this paper are from the first version of the FITS paper6, where FITS adopted a uniform input length of 720 (we also use an input length of 720 for fair comparison with it). Here, the input lengths of other baselines are set to be consistent with their respective official input lengths.

# B. Theoretical Proofs

# Proof of Theorem 3.1

Proof. The SparseTSF model consists of two main components: a 1D convolutional layer for sliding aggregation and a linear layer for sparse sliding prediction. The number of parameters in the 1D convolutional layer (without bias) is determined by the kernel size, which is $2 \times \left\lfloor { \frac { w } { 2 } } \right\rfloor + 1$ . For the linear layer (without bias), the number of parameters is the prodand $\begin{array} { r } { m = \left[ \frac { H } { w } \right] } \end{array}$ nput and output sizes, which are , respectively. Thus, the total n $\begin{array} { r } { n = \left\lfloor { \frac { L } { w } } \right\rfloor } \end{array}$ parameters in the linear layer is $n \times m$ .

By combining the parameters from both layers, the total count is: $\begin{array} { r } { n \times \bar { m } + 2 \dot { { \times } } \left\lfloor \frac { w } { 2 } \right\rfloor + 1 = \left\lfloor \frac { L } { w } \right\rfloor \times \left\lfloor \frac { H } { w } \right\rfloor + 2 \times \left\lfloor \frac { w } { 2 } \right\rfloor + } \end{array}$ 1.

# Proof of Lemma 3.3

Proof. Given the original time series $x _ { t - L + 1 : t }$ with length $L$ , the downsampling process segments it into $w$ subsequences, each of which contains every $w$ -th data point from the original series. The length of each downsampled subsequence, denoted as $n$ , is therefore $\left\lfloor { \frac { L } { w } } \right\rfloor$ , as it collects one data point from every $w$ time steps from the original series of length $L$ .

The SparseTSF model then applies a forecasting function $f$ on each of these downsampled subsequences. The forecasting function $f$ is designed to predict future values of the time series based on its past values. Specifically, it predicts the future subsequence $x _ { t + 1 : t + m } ^ { \prime }$ using the past subsequence $x _ { t - n + 1 : t } ^ { \prime }$ . Here, $m$ is the length of the forecast horizon for the downsampled subsequences and is given by $\left\lfloor { \frac { H } { w } } \right\rfloor$ , where $H$ is the original forecast horizon.

Therefore, the SparseTSF model effectively reformulates the original forecasting task of predicting $x _ { t + 1 : t + H }$ from $x _ { t - L + 1 : t }$ into a series of smaller tasks. Each of these smaller tasks involves using the downsampled past subsequence $x _ { t - n + 1 : t } ^ { \prime }$ to predict the downsampled future subsequence $x _ { t + 1 : t + m } ^ { \prime }$ . This is represented mathematically as:

$$
\begin{array} { r } { x _ { t + 1 : t + m } ^ { \prime } = f ( x _ { t - n + 1 : t } ^ { \prime } ) . } \end{array}
$$

# Proof of Theorem 3.4

Proof. Theorem 3.4 is established based on the assumption of a time series dataset that can be decomposed into a periodic component $P ( t )$ and a trend component $T ( t )$ , as defined in Definition 3.2. This decomposition implies that any time point in the series $X ( t )$ can be expressed as the sum of its periodic and trend components, i.e., $X ( t ) = P ( t ) + T ( t )$

Therefore, for the downsampled subsequences $x _ { t - n + 1 : t } ^ { \prime }$ and $x _ { t + 1 : t + m } ^ { \prime }$ based on a periodicity $w$ , we have:

$$
\begin{array} { r } { x _ { t - n + 1 : t } ^ { \prime } = p _ { t - n + 1 : t } ^ { \prime } + t _ { t - n + 1 : t } ^ { \prime } , } \\ { x _ { t + 1 : t + m } ^ { \prime } = p _ { t + 1 : t + m } ^ { \prime } + t _ { t + 1 : t + m } ^ { \prime } . } \end{array}
$$

Hence, by combining with Lemma 3.3, the task formulation of the SparseTSF model can be expressed as:

$$
p _ { t + 1 : t + m } ^ { \prime } + t _ { t + 1 : t + m } ^ { \prime } = f ( p _ { t - n + 1 : t } ^ { \prime } + t _ { t - n + 1 : t } ^ { \prime } ) .
$$

Due to the periodic nature of $P ( t )$ as defined in Equation 5, for any two points $i$ and $j$ in the downsampled sequence (where $i , j \in [ t - n + 1 : t + m ] )$ , the periodic component remains constant, i.e., $p _ { i } ^ { \prime } = p _ { j } ^ { \prime }$ .

This indicates that the task of the SparseTSF model is to predict future trend components while utilizing a constant periodic component as a reference.

# C. Case Study

# C.1. Multi-Period Scenarios

In this section, we specifically examine the performance of the SparseTSF model in scenarios involving multiple periods. Specifically, we study its performance on the Traffic dataset, as traffic flow data not only exhibits distinct daily periodicity but also demonstrates significant weekly cycles. For instance, the morning and evening rush hours represent intra-day cycles, while the different patterns between weekdays and weekends exemplify weekly cycles.

Figure 6 displays the autocorrelation in the original and dayperiod downsampled traffic flow data. It can be observed that even after downsampling with a daily period, the data still exhibits a clear weekly cycle $w ^ { \prime } = 7 ,$ ). Under these circumstances, with SparseTSF only decoupling the primary daily cycle, will it outperform the original fully connected linear model?

![](images/073087a7664da3d19a218b343719160251f6290ef98c3b8b1b33ff9fba97b423.jpg)  
Figure 6: Comparison of autocorrelation in original and downsampled subsequences for the last channel in the Traffic dataset.

The results, as shown in Figure 7, indicate that the SparseTSF model captures stronger daily and weekly periodic patterns (evident as more pronounced equidistant stripes) compared to the original approach. This is because, in the original method, a single linear layer is tasked with extracting both daily and weekly periodic patterns. In contrast, the SparseTSF model, by decoupling the daily cycle, simplifies the task for its inherent linear layer to only extract the remaining weekly periodic features. Therefore, even in scenarios with multiple periods, SparseTSF can still achieve remarkable performance.

# C.2. Ultra-Long Period Scenarios

This section is dedicated to examining the SparseTSF model’s performance in scenarios characterized by ultralong periods. Specifically, our focus is on the ETTm1&ETTm27 and Weather8 datasets, as detailed in Table 8. These datasets are distinguished by their primary periods extending up to 96 and 144, respectively. We evaluate the SparseTSF model’s performance under various settings of the hyperparameter $w$ .

As illustrated in Table 9, when $w$ is set to a large value (for instance, 144, which aligns with the intrinsic primary period of the Weather dataset), the performance of the SparseTSF model tends to deteriorate. This decline is attributed to the excessive sparsity in connections caused by a large $w$ , limiting the information available for the model to base its predictions on, thereby impairing its performance. Interestingly, as $w$ increases, there is a noticeable improvement in the SparseTSF model’s performance. This observation suggests that employing denser connections within the SparseTSF framework could be a more viable option for datasets with longer periods.

![](images/69f58b5113ce960676512843858bd6621c1634fccf5c8d05c506c1a3fe76efeb.jpg)  
Figure 7: Visualization of normalized weights of the model trained on the Traffic dataset with both look-back length (X-axis) and forecast horizon (Y-axis) of 336.

Table 8: Summary of datasets with ultra-long periods.   

<table><tr><td>Datasets</td><td>ETTm1</td><td>ETTm2</td><td>Weather</td></tr><tr><td>Channels</td><td>7</td><td>7</td><td>21</td></tr><tr><td>Frequency</td><td>15 mins</td><td>15 mins</td><td>10 mins</td></tr><tr><td>Timesteps</td><td>69,680</td><td>69,680</td><td>52,696</td></tr></table>

Furthermore, an intriguing phenomenon is observed when $w = 1$ , which corresponds to the scenario of employing a fully connected linear layer for prediction. The performance in this case is inferior compared to sparse connection-based predictions. This indicates that an appropriate level of sparsity in connections (even when the sparse interval does not match the dataset’s inherent primary period) can enhance the model’s predictive accuracy. This could be due to the redundant nature of time series data, especially when data sampling is dense. In such cases, executing sparse predictions might help eliminate some redundant information. However, these findings necessitate further investigation and exploration in future work.

The findings above suggest that employing a denser sparse strategy would be beneficial in such cases. Therefore, we present in Table 10 a comparative performance of

Table 9: MSE results of SparseTSF on ultra-long period datasets with varied hyperparameters $w$ . The forecast horizon is set as 720.   

<table><tr><td rowspan="2">Dataset</td><td colspan="8">Parameter w</td></tr><tr><td>144</td><td>72</td><td>48</td><td>24</td><td>12</td><td>6</td><td>2</td><td>1</td></tr><tr><td>ETTm1</td><td>0.450</td><td>0.450</td><td>0.422</td><td>0.422</td><td>0.421</td><td>0.415</td><td>0.415</td><td>0.429</td></tr><tr><td>ETTm2</td><td>0.375</td><td>0.371</td><td>0.373</td><td>0.352</td><td>0.354</td><td>0.349</td><td>0.349</td><td>0.357</td></tr><tr><td>Weather</td><td>0.332</td><td>0.329</td><td>0.325</td><td>0.321</td><td>0.319</td><td>0.319</td><td>0.318</td><td>0.322</td></tr></table>

SparseTSF against other models under the setting of $w = 4$ where SparseTSF ranks within the top 3 in most cases. In this scenario, SparseTSF remains significantly lighter compared to other mainstream models. This indicates that the Sparse forecasting technique not only effectively reduces parameter size but also enhances prediction accuracy in most scenarios.

# D. More Results and Analysis

# D.1. Comparison Results after Fixing the Code Bug

Recent research has discovered a long-standing bug in the popular codebase used in the field since the introduction of the Informer (Zhou et al., 2021). This bug, which affected the calculation of test set metrics, caused the data that did not fill an entire batch to be discarded (Qiu et al., 2024). As a result, the batch size setting influenced the results. Theoretically, the larger the batch size, the more test data might be discarded, leading to incorrect results. This bug significantly improved the performance on ETTh1 and ETTh2 datasets when the batch size was large, while the impact on other datasets was relatively minor.

To reassess the performance of SparseTSF, we present the performance of SparseTSF and existing models after fixing this bug in Table 11. Here, we reran FITS under the conditions of lookback $L = 7 2 0$ and cutoff frequency $C O F = 5$ (where the parameter count of SparseTSF is still tens of times smaller than that of FITS) for a fair comparison with SparseTSF. The results for other baselines were sourced from FITS’ reproduction, where they reran the baselines’ results after fixing the bug (Xu et al., 2024). As shown, after fixing the code bug, SparseTSF still achieves impressive performance with minimal overhead, aligning with the conclusions of Table 2.

Table 10: MSE results on ultra-long period datasets comparing SparseTSF $\omega = 4$ ) with other mainstream models. The ranking of SparseTSF’s performance is shown in parentheses.   

<table><tr><td>Dataset</td><td colspan="4">ETTm1</td><td colspan="4">ETTm2</td><td colspan="4">Weather</td></tr><tr><td>Horizon</td><td>96</td><td>192</td><td>336</td><td>720</td><td>96</td><td>192</td><td>336</td><td>720</td><td>96</td><td>192</td><td>336</td><td>720</td></tr><tr><td>Informer (2021)</td><td>0.672</td><td>0.795</td><td>1.212</td><td>1.166</td><td>0.365</td><td>0.533</td><td>1.363</td><td>3.379</td><td>0.300</td><td>0.598</td><td>0.578</td><td>1.059</td></tr><tr><td>Autoformer (2021)</td><td>0.505</td><td>0.553</td><td>0.621</td><td>0.671</td><td>0.255</td><td>0.281</td><td>0.339</td><td>0.433</td><td>0.266</td><td>0.307</td><td>0.359</td><td>0.419</td></tr><tr><td>Pyraformer (2022b)</td><td>0.543</td><td>0.557</td><td>0.754</td><td>0.908</td><td>0.435</td><td>0.730</td><td>1.201</td><td>3.625</td><td>0.896</td><td>0.622</td><td>0.739</td><td>1.004</td></tr><tr><td>FEDformer (2022b)</td><td>0.379</td><td>0.426</td><td>0.445</td><td>0.543</td><td>0.203</td><td>0.269</td><td>0.325</td><td>0.421</td><td>0.217</td><td>0.276</td><td>0.339</td><td>0.403</td></tr><tr><td>TimesNet (2023)</td><td>0.338</td><td>0.374</td><td>0.410</td><td>0.478</td><td>0.187</td><td>0.249</td><td>0.321</td><td>0.408</td><td>0.172</td><td>0.219</td><td>0.280</td><td>0.365</td></tr><tr><td>PatchTST (2023)</td><td>0.293</td><td>0.333</td><td>0.369</td><td>0.416</td><td>0.166</td><td>0.223</td><td>0.274</td><td>0.362</td><td>0.149</td><td>0.194</td><td>0.245</td><td>0.314</td></tr><tr><td>DLinear (2023)</td><td>0.299</td><td>0.335</td><td>0.369</td><td>0.425</td><td>0.167</td><td>0.221</td><td>0.274</td><td>0.368</td><td>0.176</td><td>0.218</td><td>0.262</td><td>0.323</td></tr><tr><td>FITS (2024)</td><td>0.305</td><td>0.339</td><td>0.367</td><td>0.418</td><td>0.164</td><td>0.217</td><td>0.269</td><td>0.347</td><td>0.145</td><td>0.188</td><td>0.236</td><td>0.308</td></tr><tr><td>SparseTSF (ours)</td><td>0.314(4)</td><td>0.343(4)</td><td>0.369(2)</td><td>0.418(2)</td><td>0.165(2)</td><td>0.218(2)</td><td>0.272(2)</td><td>0.35(2)</td><td>0.172(3)</td><td>0.215(3)</td><td>0.26(3)</td><td>0.318(3)</td></tr></table>

Table 11: MSE results of multivariate long-term time series forecasting comparing SparseTSF with other mainstream models after fixing code bug. The top two results are highlighted in bold.   

<table><tr><td>Dataset</td><td colspan="4">ETTh1</td><td colspan="4">ETTh2</td><td colspan="4">Electricity</td><td colspan="4">Traffic</td></tr><tr><td>Horizon</td><td>96</td><td>192</td><td>336</td><td>720</td><td>96</td><td>192</td><td>336</td><td>720</td><td>96</td><td>192</td><td>336</td><td>720</td><td>96</td><td>192</td><td>336</td><td>720</td></tr><tr><td>FEDformer (2022b)</td><td>0.375</td><td>0.427</td><td>0.459</td><td>0.484</td><td>0.340</td><td>0.433</td><td>0.508</td><td>0.480</td><td>0.188</td><td>0.197</td><td>0.212</td><td>0.244</td><td>0.573</td><td>0.611</td><td>0.621</td><td>0.630</td></tr><tr><td>TimesNet (2023)</td><td>0.384</td><td>0.436</td><td>0.491</td><td>0.521</td><td>0.340</td><td>0.402</td><td>0.452</td><td>0.462</td><td>0.168</td><td>0.184</td><td>0.198</td><td>0.220</td><td>0.593</td><td>0.617</td><td>0.629</td><td>0.640</td></tr><tr><td>PatchTST (2023)</td><td>0.385</td><td>0.413</td><td>0.440</td><td>0.456</td><td>0.274</td><td>0.338</td><td>0.367</td><td>0.391</td><td>0.129</td><td>0.149</td><td>0.166</td><td>0.210</td><td>0.366</td><td>0.388</td><td>0.398</td><td>0.457</td></tr><tr><td>DLinear (2023</td><td>0.384</td><td>0.443</td><td>0.446</td><td>0.504</td><td>0.282</td><td>0.350</td><td>0.414</td><td>0.588</td><td>0.140</td><td>0.153</td><td>0.169</td><td>0.204</td><td>0.413</td><td>0.423</td><td>0.437</td><td>0.466</td></tr><tr><td>FITS (2024)</td><td>0.382</td><td>0.417</td><td>0.436</td><td>0.433</td><td>0.272</td><td>0.333</td><td>0.355</td><td>0.378</td><td>0.145</td><td>0.159</td><td>0.175</td><td>0.212</td><td>0.398</td><td>0.409</td><td>0.421</td><td>0.457</td></tr><tr><td>SparseTSF (ours)</td><td>0.362</td><td>0.403</td><td>0.434</td><td>0.426</td><td>0.294</td><td>0.339</td><td>0.359</td><td>0.383</td><td>0.138</td><td>0.151</td><td>0.166</td><td>0.205</td><td>0.389</td><td>0.398</td><td>0.411</td><td>0.448</td></tr></table>

# D.2. Impacts of Varying Look-Back Length

The look-back length determines the richness of historical information the model can utilize. Generally, models are expected to perform better with longer input lengths if they possess robust long-term dependency modeling capabilities. Table 12 presents the performance of SparseTSF at different look-back lengths.

It can be observed that two phenomena occur: (i) longer look-back windows perform better, indicating SparseTSF’s ability in long-term dependency modeling, and (ii) the performance of the ETTh1 & ETTh2 datasets remains relatively stable across different look-back windows, while the performance of the Traffic & Electricity datasets varies significantly, especially with a look-back of 96, where the accuracy notably decreases.

In fact, we can further discuss the reasons behind the second point. As illustrated in Figure 3, ETTh1 only exhibits a significant daily periodic pattern $\omega = 2 4$ ). In this case, look-back lengths of 96 can achieve good results because they fully encompass the daily periodic pattern. However, as shown in Figure 7, Traffic not only has a significant daily periodic pattern $w = 2 4 )$ ) but also a noticeable weekly periodic pattern $w = 1 6 8 ,$ . In this case, a look-back of 96 cannot cover the entire weekly periodic pattern, leading to a significant performance drop. This underscores the necessity of sufficiently long look-back lengths (at least covering the entire cycle length) for accurate prediction. Given the extreme lightweight nature of SparseTSF, we strongly recommend providing sufficiently long look-back windows whenever feasible.

# D.3. Impacts of Instance Normalization

Instance Normalization (IN) strategy has become popular in mainstream methods. We also employ this strategy in SparseTSF to enhance its performance on datasets with significant distribution drift. We showcase the impact of the IN strategy in Table 13.

It can be observed that IN is necessary for smaller datasets, namely ETTh1 and ETTh2 datasets. However, its effect is relatively limited on larger datasets such as Traffic and Electricity datasets. It must be clarified that, although the IN strategy is one of the factors contributing to SparseTSF’s success, it is not the key differentiator of SparseTSF’s core contributions compared to other models.

Table 12: MSE results of SparseTSF with varied look-back lengths.   

<table><tr><td>Dataset</td><td colspan="4">ETTh1</td><td colspan="4">ETTh2</td><td colspan="4">Electricity</td><td colspan="4">Traffic</td></tr><tr><td>Look-back Horizon</td><td>96</td><td>192</td><td>336</td><td>720</td><td>96</td><td>192</td><td>336</td><td>720</td><td>96</td><td>192</td><td>336</td><td>720</td><td>96</td><td>192</td><td>336</td><td>720</td></tr><tr><td>96</td><td>0.380</td><td>0.371</td><td>0.393</td><td>0.354</td><td>0.288</td><td>0.285</td><td>0.272</td><td>0.278</td><td>0.209</td><td>0.160</td><td>0.146</td><td>0.138</td><td>0.672</td><td>0.455</td><td>0.412</td><td>0.383</td></tr><tr><td>192</td><td>0.433</td><td>0.434</td><td>0.418</td><td>0.398</td><td>0.363</td><td>0.346</td><td>0.323</td><td>0.315</td><td>0.202</td><td>0.166</td><td>0.154</td><td>0.147</td><td>0.608</td><td>0.453</td><td>0.415</td><td>0.388</td></tr><tr><td>336</td><td>0.447</td><td>0.420</td><td>0.390</td><td>0.405</td><td>0.366</td><td>0.335</td><td>0.314</td><td>0.311</td><td>0.217</td><td>0.184</td><td>0.172</td><td>0.164</td><td>0.609</td><td>0.468</td><td>0.428</td><td>0.403</td></tr><tr><td>720</td><td>0.451</td><td>0.426</td><td>0.413</td><td>0.418</td><td>0.407</td><td>0.389</td><td>0.372</td><td>0.371</td><td>0.259</td><td>0.223</td><td>0.210</td><td>0.205</td><td>0.650</td><td>0.493</td><td>0.462</td><td>0.446</td></tr><tr><td>Avg.</td><td>0.428</td><td>0.413</td><td>0.404</td><td>0.394</td><td>0.356</td><td>0.339</td><td>0.320</td><td>0.319</td><td>0.222</td><td>0.183</td><td>0.171</td><td>0.163</td><td>0.635</td><td>0.467</td><td>0.429</td><td>0.405</td></tr></table>

Table 13: Ablation results of IN strategy in SparseTSF.   

<table><tr><td>Dataset</td><td colspan="2">ETTh1</td><td colspan="2">ETTh2</td><td colspan="2">Electricity</td><td colspan="2">Traffic</td></tr><tr><td>Horizon</td><td>w/IN</td><td>w/o IN</td><td>w/IN</td><td>w/o IN</td><td>w/IN</td><td>w/o IN</td><td>w/IN</td><td>w/o IN</td></tr><tr><td>96</td><td>0.359</td><td>0.37</td><td>0.267</td><td>0.327</td><td>0.138</td><td>0.138</td><td>0.382</td><td>0.382</td></tr><tr><td>192</td><td>0.397</td><td>0.413</td><td>0.314</td><td>0.426</td><td>0.146</td><td>0.146</td><td>0.388</td><td>0.387</td></tr><tr><td>336</td><td>0.404</td><td>0.431</td><td>0.312</td><td>0.482</td><td>0.164</td><td>0.163</td><td>0.402</td><td>0.401</td></tr><tr><td>720</td><td>0.417</td><td>0.462</td><td>0.37</td><td>0.866</td><td>0.203</td><td>0.198</td><td>0.445</td><td>0.444</td></tr></table>

# D.4. Comparison Results with N-HiTS and OneShotSTL

Table 14: Comparison Results with N-HiTS and OneShotSTL. In this comparison, SparseTSF and N-HiTS are evaluated based on multivariate prediction results (MSE), while SparseTSF and OneShotSTL are compared using univariate prediction results (MAE). Their results are sourced from their respective official papers.

<table><tr><td>Dataset</td><td>Horizon</td><td>Nhit</td><td>SparseTSF</td><td>OneShotSTL</td><td>SparseTSF</td></tr><tr><td rowspan="4">ETTm2</td><td>96</td><td>0.176</td><td>0.165</td><td>0.211</td><td>0.187</td></tr><tr><td>192</td><td>0.245</td><td>0.218</td><td>0.244</td><td>0.233</td></tr><tr><td>336</td><td>0.295</td><td>0.272</td><td>0.273</td><td>0.268</td></tr><tr><td>720</td><td>0.401</td><td>0.350</td><td>0.321</td><td>0.324</td></tr><tr><td rowspan="4">Electricity</td><td>96</td><td>0.147</td><td>0.138</td><td>0.331</td><td>0.314</td></tr><tr><td>192</td><td>0.167</td><td>0.146</td><td>0.355</td><td>0.334</td></tr><tr><td>336</td><td>0.186</td><td>0.164</td><td>0.389</td><td>0.366</td></tr><tr><td>720</td><td>0.243</td><td>0.203</td><td>0.444</td><td>0.416</td></tr><tr><td rowspan="4">Traffic</td><td>96</td><td>0.402</td><td>0.382</td><td>0.181</td><td>0.179</td></tr><tr><td>192</td><td>0.42</td><td>0.388</td><td>0.181</td><td>0.175</td></tr><tr><td>336</td><td>0.448</td><td>0.402</td><td>0.182</td><td>0.184</td></tr><tr><td>720</td><td>0.539</td><td>0.445</td><td>0.199</td><td>0.203</td></tr></table>

Here, we present the comparison results between SparseTSF and N-HiTS and OneShotSTL in Table 14. It can be observed that in most cases, SparseTSF outperforms these methods, demonstrating the superiority of the SparseTSF approach.