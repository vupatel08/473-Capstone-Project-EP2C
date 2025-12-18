# RETHINKING THE UNIFORMITY METRIC IN SELFSUPERVISED LEARNING

Xianghong Fang The Chinese University of Hong Kong, Shenzhen fangxianghong2@gmail.com

Jian Li   
Tencent AI Lab   
lijianjack@gmail.com

Qiang Sun∗ University of Toronto & MBZUAI qsunstats@gmail.com

Benyou Wang∗   
The Chinese University of Hong Kong, Shenzhen & SRIBD   
wangbenyou@cuhk.edu.cn

# ABSTRACT

Uniformity plays an important role in evaluating learned representations, providing insights into self-supervised learning. In our quest for effective uniformity metrics, we pinpoint four principled properties that such metrics should possess. Namely, an effective uniformity metric should remain invariant to instance permutations and sample replications while accurately capturing feature redundancy and dimensional collapse. Surprisingly, we find that the uniformity metric proposed by Wang & Isola (2020) fails to satisfy the majority of these properties. Specifically, their metric is sensitive to sample replications, and can not account for feature redundancy and dimensional collapse correctly. To overcome these limitations, we introduce a new uniformity metric based on the Wasserstein distance, which satisfies all the aforementioned properties. Integrating this new metric in existing self-supervised learning methods effectively mitigates dimensional collapse and consistently improves their performance on downstream tasks involving CIFAR-10 and CIFAR-100 datasets. Code is available at https://github.com/statsle/WassersteinSSL.

# 1 INTRODUCTION

Self-supervised learning excels in acquiring invariant representations to various augmentations (Chen et al., 2020; He et al., 2020; Caron et al., 2020; Grill et al., 2020; Zbontar et al., 2021). It has been outstandingly successful across a wide range of domains, such as multimodality learning, object detection, and segmentation (Radford et al., 2021; Li et al., 2022; Xie et al., 2021; Wang et al., 2021; Yang et al., 2021; Zhao et al., 2021). To gain a deeper understanding of self-supervised learning, thoroughly evaluating the learned representations is necessary (Wang & Isola, 2020; Gao et al., 2021; Tian et al., 2021; Jing et al., 2022).

Alignment, a metric quantifying the similarities between positive pairs, holds significant importance in the evaluation of learned representations (Wang & Isola, 2020). It ensures that positive pairs are mapped to similar features, making them invariant to unnecessary details (Hadsell et al., 2006; Chen et al., 2020). However, relying solely on alignment proves inadequate for effectively assessing the representations. This limitation becomes evident in the presence of extremely small alignment values in collapsing solutions, as observed in Siamese networks (Hadsell et al., 2006), where all outputs collapse to a single point (Chen & He, 2021), as illustrated in Figure 1. In such cases, the learned representations exhibit optimal alignment but fail to provide meaningful information for any downstream tasks. This underscores the necessity of incorporating additional metrics when evaluating learned representations.

![](images/5d0a2edc611e8abb25340bf74e601d091b80c5ee057bdac7d58ac05c11442003.jpg)  
Figure 1: The left figure presents constant collapse, and the right figure visualizes dimensional collapse.

To further evaluate the learned representations, Wang & Isola (2020) formally introduced a uniformity metric based on the logarithm of the average pairwise Gaussian potential (Cohn & Kumar, 2007). Uniformity assesses how feature embeddings are distributed uniformly across the unit hypersphere, and higher uniformity indicates more information from the data is preserved. Since its introduction, uniformity has played a pivotal role in understanding self-supervised learning and mitigating constant collapse (Arora et al., 2019; Wang & Isola, 2020; Gao et al., 2021). Nevertheless, the effectiveness of this particular uniformity metric warrants further examination.

To delve deeper into the existing uniformity metric proposed by Wang & Isola (2020), we introduce four principled properties that an effective uniformity metric should possess. Guided by these properties, we conduct a theoretical analysis, unveiling key limitations of this metric, particularly its inability to capture feature redundancy and dimensional collapse (Hua et al., 2021). Dimensional collapse refers to the scenario where representations occupy a lower-dimensional subspace rather than the entire embedding space (Jing et al., 2022); see Figure 1. We reinforce our theoretical findings with empirical evidence, demonstrating, for instance, the existing metric’s inability to differentiate between different degrees of dimensional collapse. Subsequently, we propose a novel uniformity metric based on the quadratic Wasserstein distance that satisfies all four properties, thereby surpassing the existing one. Finally, integrating the proposed uniformity metric as an auxiliary loss within existing self-supervised learning methods consistently enhances their performance in downstream tasks.

Our main contributions are summarized as follows. (i) We identify four principled properties that an effective uniformity metric should possess, providing new guidelines on designing such metrics. (ii) Surprisingly, we find that the existing uniformity metric (Wang & Isola, 2020) fails to meet the majority of these properties. For example, it can not correctly capture dimensional collapse. (iii) We propose a new uniformity metric based on the Wasserstein distance that satisfies all four properties, addressing key limitations of the existing metric. (iv) Our proposed uniformity metric can seamlessly integrate as an auxiliary loss in various self-supervised learning methods, resulting in improved performance in downstream tasks.

# 2 BACKGROUND

# 2.1 SELF-SUPERVISED REPRESENTATION LEARNING

Self-supervised learning leverages the idea that similar samples should have similar representations that are invariant to unnecessary details (Wang & Isola, 2020). For instance, the Siamese network (Hadsell et al., 2006) takes as input positive pairs $\left( \mathbf { x } ^ { a } , \mathbf { x } ^ { b } \right)$ , often obtained by taking two augmented views of the same sample $\mathbf { x }$ . These positive pairs are then processed by an encoder network $f$ consisting of a backbone (e.g., ResNet (He et al., 2016)) and a projection MLP head (Chen et al., 2020), yielding representations $( \mathbf { z } ^ { a } = f ( \mathbf { x } ^ { a } ) , \mathbf { z } ^ { b } = f ( \mathbf { x } ^ { b } ) ^ { 1 }$ . To enforce invariance, a natural approach is to minimize the following alignment loss, defined as the expected distance between positive pairs:

$$
\begin{array} { r } { \mathcal { L } _ { \boldsymbol { A } } : = \mathbb { E } _ { ( \mathbf { z } ^ { a } , \mathbf { z } ^ { b } ) \sim p _ { \mathrm { p o s } } } \left\| \mathbf { z } _ { i } ^ { a } - \mathbf { z } _ { i } ^ { b } \right\| _ { 2 } ^ { 2 } , } \end{array}
$$

where $p _ { \mathrm { p o s } } ( \cdot , \cdot )$ is the distribution of positive pairs.

However, optimizing the above alignment loss alone may lead to an undesired collapsing solution, where all representations collapse into a single point, as shown in Figure 1.

# 2.2 EXISTING SOLUTIONS TO CONSTANT COLLAPSE

To prevent constant collapse, existing solutions include contrastive learning, asymmetric model architecture, and redundancy reduction.

Contrastive Learning Contrastive learning offers a potent solution to mitigate constant collapse. The key idea is to leverage negative pairs. For example, SimCLR (Chen et al., 2020) introduced an in-batch negative sampling strategy that utilizes samples within a batch as negative samples. However, its effectiveness is contingent on the use of a large batch size. To address this limitation,

MoCo (He et al., 2020) used a memory bank, which stores additional representations as negative samples. Recent research endeavors have also explored clustering-based contrastive learning, which combines a clustering objective with contrastive learning techniques (Li et al., 2021; Caron et al., 2020).

Asymmetric Model Architecture The use of asymmetric model architecture represents another strategy to combat constant collapse. One plausible explanation for its effectiveness is that such an asymmetric design encourages encoding more information (Grill et al., 2020). To maintain this asymmetry, BYOL (Grill et al., 2020) introduces the concept of using an additional predictor in one branch of the Siamese network while employing momentum updates and stop-gradient operators in the other branch. DINO (Caron et al., 2021), takes this asymmetry a step further by applying it to two encoders, distilling knowledge from the momentum encoder into the other one (Hinton et al., 2015). SimSiam (Chen & He, 2021) removes the momentum update from BYOL, and shows that the momentum update may not be essential in preventing constant collapse. However, MirrorSimSiam (Zhang et al., 2022a) swaps the stop-gradient operator to the other branch. Its failure challenges the assertion made in SimSiam (Chen & He, 2021) that the stop-gradient operator is the key component for preventing constant collapse. Tian et al. (2021) provides a theoretical examination to elucidate why an asymmetric model architecture can effectively avoid constant collapse.

Redundancy Reduction The fundamental principle behind redundancy reduction to mitigate constant collapse is to maximize the information preserved by the representations. The key idea is to decorrelate the learned representations. Barlow Twins (Zbontar et al., 2021) aims to achieve decorrelation by focusing on the cross-correlation matrix, while VICReg (Bardes et al., 2022) focuses on the covariance matrix. Zero-CL (Zhang et al., 2022b) takes a hybrid approach, combining instance-wise and feature-wise whitening techniques.

# 2.3 THE EXISTING UNIFORMITY METRIC

While the aforementioned solutions effectively prevent constant collapse, they are not as effective in preventing dimensional collapse, wherein representations occupy a lower-dimensional subspace instead of the entire space. This phenomenon has been observed in contrastive learning by visualizing the singular value spectra of representations (Jing et al., 2022; Tian et al., 2021).

To quantitatively measure the degree of collapse, Wang & Isola (2020) introduced a uniformity loss based on the logarithm of the average pairwise Gaussian potential. Given (normalized) feature representations $\left\{ \mathbf { z } _ { 1 } , \mathbf { z } _ { 2 } , . . . , \mathbf { z } _ { n } \right\}$ , their proposed empirical uniformity loss is:

$$
\mathcal { L } _ { \mathcal { U } } : = \log \frac { 1 } { n ( n - 1 ) / 2 } \sum _ { i = 2 } ^ { n } \sum _ { j = 1 } ^ { i - 1 } e ^ { - t \| \mathbf { z } _ { i } - \mathbf { z } _ { j } \| _ { 2 } ^ { 2 } } ,
$$

where $t > 0$ is a fixed parameter, often set to 2. Then $\mathrm { - } \mathcal { L } _ { \mathcal { U } }$ serves as the corresponding uniformity metric, with a higher value indicating greater uniformity.

We demonstrate in this work that this metric is insensitive to dimensional collapse, both theoretically in Section 3.2 and empirically in Section 5.2.

# 3 WHAT MAKES AN EFFECTIVE UNIFORMITY METRIC?

In this section, we begin by presenting four fundamental properties that an effective uniformity metric should possess. Leveraging these properties as a lens, we then scrutinize the existing uniformity metric $\mathrm { - } \mathcal { L } _ { \mathcal { U } }$ , shedding light on its limitations.

# 3.1 FOUR PROPERTIES FOR UNIFORMITY

A uniformity metric $\mathcal { U } : \mathbb { R } ^ { m ^ { n } }  \mathbb { R }$ is a function that maps a set of learned representations to a scalar indicator of uniformity. In the following section, we introduce four principled properties that an effective uniformity metric should possess. Let $\mathcal { D } = \mathbf { z } _ { 1 } , . . . , \mathbf { z } _ { n } \in \mathbb { R } ^ { m n }$ represent the learned representations. To avoid the trivial case, we assume that $\mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { n }$ are not all equal, meaning that not all points collapse to a single constant point.

First, an effective uniformity metric should be invariant to the permutation of instances, as the distribution of representations should not be affected by permutations.

Property 1 (Instance Permutation Constraint (IPC)). An effective uniformity metric U should satisfy

$$
\mathcal { U } ( \pi ( \mathcal { D } ) ) = \mathcal { U } ( \mathcal { D } ) ,
$$

where π is a permutation over the instances.

Second, an effective uniformity metric should be invariant to instance clones, as instance cloning does not vary the distribution of representations.

Property 2 (Instance Cloning Constraint (ICC)). An effective uniformity metric U should satisfy

$$
\mathcal { U } ( \mathcal { D } \not \Rightarrow \mathcal { D } ) = \mathcal { U } ( \mathcal { D } ) ,
$$

where ${ \mathcal { D } } \not \\sqcup { \mathcal { D } } : = \{ { \mathbf { z } } _ { 1 } , { \mathbf { z } } _ { 2 } , . . . , { \mathbf { z } } _ { n } , { \mathbf { z } } _ { 1 } , { \mathbf { z } } _ { 2 } , . . . , { \mathbf { z } } _ { n } \} .$

Third, an effective uniformity metric should strictly decrease as feature-level cloning for each instance occurs, as this duplication introduces redundancy, which corresponds to dimensional collapse (Zbontar et al., 2021; Bardes et al., 2022).

Property 3 (Feature Cloning Constraint (FCC)). An effective uniformity metric $\mathcal { U }$ should satisfy

$$
\mathcal { U } ( \mathcal { D } \oplus \mathcal { D } ) < \mathcal { U } ( \mathcal { D } ) ,
$$

$$
\mathcal { D } \oplus \mathcal { D } : = \{ \mathbf { z } _ { 1 } \oplus \mathbf { z } _ { 1 } , \mathbf { z } _ { 2 } \oplus \mathbf { z } _ { 2 } , . . . , \mathbf { z } _ { n } \oplus \mathbf { z } _ { n } \} \ a n d \mathbf { z } _ { i } \oplus \mathbf { z } _ { i } : = ( z _ { i 1 } , \cdot \cdot , z _ { i m } , z _ { i 1 } , \cdot \cdot \cdot , z _ { i m } ) ^ { \mathrm { { T } } } \in \mathbb { R } ^ { 2 m } .
$$

Fourth, an effective uniformity metric should strictly decrease with the addition of constant features for each instance, as this introduces uninformative and thus redundant features, which again corresponds to dimensional collapse.

Property 4 (Feature Baby Constraint (FBC)). An effective uniformity metric $\mathcal { U }$ should satisfy

$$
\begin{array} { r } { \mathcal { U } ( \mathcal { D } \oplus \mathbf { 0 } ^ { k } ) < \mathcal { U } ( \mathcal { D } ) , \quad k \in \mathbb { N } ^ { + } , } \end{array}
$$

where $\oplus$ is defined in Property 3, that is, $\mathcal { D } \oplus \mathbf { 0 } ^ { k } = \{ \mathbf { z } _ { 1 } \oplus \mathbf { 0 } ^ { k } , \mathbf { z } _ { 2 } \oplus \mathbf { 0 } ^ { k } , . . . , \mathbf { z } _ { n } \oplus \mathbf { 0 } ^ { k } \}$ and $\mathbf { z } _ { i } \oplus \mathbf { 0 } ^ { k } =$ (zi1, zi2, ..., zim, 0, 0, ..., 0)T ∈ Rm+k.

Intuitively, Properties 1 and 2 ensure that the uniformity metric should remain insensitive to instance permutations and sample replications, respectively. Meanwhile, Properties 3 and 4 ensure that feature redundancy and dimensional collapse reduce the uniformity metric, as they make the distribution of the representations less uniform. These four properties constitute intuitive yet principled characteristics of an effective uniformity metric.

# 3.2 EXAMINING THE UNIFORMITY METRIC $\mathrm { - } \mathcal { L } _ { \mathcal { U } }$

We employ the four properties introduced earlier to analyze the uniformity metric $\mathrm { - } \mathcal { L } _ { \mathcal { U } }$ defined in Eqn. (2). The following theorem summarizes our findings.

Theorem 1. The uniformity metric $\mathrm { - } \mathcal { L } _ { \mathcal { U } }$ satisfies Property 1, but violates Properties 2, 3, and 4.

The proof of the above theorem is provided in Appendix C. The violation of Property 2 indicates that the uniformity metric $\mathrm { - } \mathcal { L } _ { \mathcal { U } }$ is sensitive to sample replications, while the violations of Properties 3 and 4 suggest that feature redundancy and dimensional collapse do not reduce the uniformity metric $\mathrm { - } \mathcal { L } _ { \mathcal { U } }$ , making this uniformity metric unable to correctly reflect feature redundancy and dimensional collapse. Therefore, there is a pressing need to develop a new uniformity metric.

# 4 A NEW UNIFORMITY METRIC

In this section, we introduce a new uniformity metric to address the limitations of $\mathrm { - } \mathcal { L } _ { \mathcal { U } }$

![](images/e9ff42e151516bd79d7a88b795e1d1f766aa86e4328b5fe441e20ed39d9fca37.jpg)  
Figure 2: The KL divergence and Wasserstein distance between $Y _ { i }$ and $\widehat { Y } _ { i }$ w.r.t. various dimensions.

# 4.1 THE UNIFORM SPHERICAL DISTRIBUTION AND AN APPROXIMATION

As pointed out by (Wang & Isola, 2020), feature vectors should be roughly uniformly distributed on the unit hypersphere $S ^ { \bar { m } - 1 }$ , preserving as much information of the data as possible. Therefore, we adopt the uniform spherical distribution as our target distribution.

Our approach utilizes the quadratic Wasserstein distance, a form of statistical distance, between the feature distribution and the target distribution as the new uniformity loss. However, computing any statistical distances involving the uniform spherical distribution can be challenging. To address this, we first establish an asymptotic equivalence between the uniform spherical distribution and the isotropic Gaussian distribution. By adopting a Gaussian distribution for the representations, we then exploit the fact that the quadratic Wasserstein distance between two Gaussian distributions has a closed form involving only the means and covariance matrices, leading to a new and simple uniformity loss. We need the following fact.

Fact 1. If $\mathbf { Z } \sim { \mathcal { N } } ( \mathbf { 0 } , \sigma ^ { 2 } \mathbf { I } _ { m } )$ , then $\mathbf { Y } : = \mathbf { Z } / \| \mathbf { Z } \| _ { 2 }$ is uniformly distributed on the unit hypersphere Sm−1.

Because the average length of $\| \mathbf Z \| _ { 2 }$ is roughly $\sigma { \sqrt { m } }$ (Chandrasekaran et al., 2012), that is,

$$
{ \frac { m } { \sqrt { m + 1 } } } \leq \| \mathbf { Z } \| _ { 2 } / \sigma \leq { \sqrt { m } } ,
$$

we expect that $\mathbf { Z } / ( \sigma { \sqrt { m } } ) \sim { \mathcal { N } } ( \mathbf { 0 } , \mathbf { I } _ { m } / m )$ provides a reasonable approximation to $\mathbf { Z } / \lVert \mathbf { Z } \rVert _ { 2 }$ , and thus to the uniform spherical distribution. This is partially justified by the following theorem.

Theorem 2. Let $Y _ { i }$ be the $i$ -th coordinate of $\mathbf { Y } = \mathbf { Z } / \| \mathbf { Z } \| _ { 2 } \in \mathbb { R } ^ { m }$ , where $\mathbf { Z } \sim { \mathcal { N } } ( \mathbf { 0 } , \sigma ^ { 2 } \mathbf { I } _ { m } )$ . Then the quadratic Wasserstein distance between $Y _ { i }$ and $\widehat { Y } _ { i } \sim { \mathcal { N } } ( 0 , 1 / m )$ converges to zero as $m  \infty$ , that is,

$$
\operatorname* { l i m } _ { m  \infty } \mathcal { W } _ { 2 } ( Y _ { i } , \widehat { Y } _ { i } ) = 0 .
$$

Theorem 2 suggests that $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { m } / m )$ approximates the distribution of each coordinate of the uniform spherical distribution as $m  \infty$ . It can be proven by first employing the Talagrand $T _ { 2 }$ inequality (Van Handel, 2016) to upper bound the quadratic Wasserstein distance using the Kullback-Leibler (KL) divergence, and then establishing that the Kullback-Leibler (KL) divergence converges to 0. The proof is provided in Appendix B.

We empirically compare the distributions of $Y _ { i }$ and $\widehat { Y } _ { i }$ across various dimensions $m \in$ 2, 4, 8, 16, 32, 64, 128, 256. For each $m$ , we sample 200,000 data points from both $Y _ { i }$ and $\widehat { Y } _ { i }$ , bin them into 51 groups, and calculate the empirical KL divergence and Wasserstein distance. Figure 2 plots both distances versus increasing dimensions. We observe that both distances converge to 0 as $m$ increases. Specifically, these results indicate that the distribution of $\widehat { Y } _ { i }$ provides a reasonable approximation to that of $Y _ { i }$ when $m \geq 2 ^ { 4 } = 1 6$ . Further comparisons between $\mathbf { Y }$ and $\widehat { \mathbf Y }$ can be found in Appendix D.

# 4.2 A NEW METRIC FOR UNIFORMITY

In this section, we discuss how to use the quadratic Wasserstein distance between the distribution of learned representations and $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { m } / m )$ , in place of the uniform spherical distribution $\operatorname { U n i f } ( S ^ { m - 1 } )$ , as our new uniformity loss.

To facilitate computation, we adopt a Gaussian hypothesis for the learned representations and assume they follow $\scriptstyle { \mathcal { N } } ( \mu , \Sigma )$ . With this assumption, we employ the quadratic Wasserstein distance2 to measure the distance between two distributions. We need the following well-known lemma (Olkin & Pukelsheim, 1982).

Lemma 1. Then the quadratic Wasserstein distance between $\scriptstyle { \mathcal { N } } ( \mu , \Sigma )$ and $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } / m )$ is

$$
\sqrt { \| \pmb { \mu } \| _ { 2 } ^ { 2 } + 1 + \operatorname { t r } ( \pmb { \Sigma } ) - \frac { 2 } { \sqrt { m } } \operatorname { t r } ( \pmb { \Sigma } ^ { \frac { 1 } { 2 } } ) } .
$$

The lemma above indicates that the quadratic Wasserstein distance can be easily computed using the population mean and covariance of the representations. In practice, we estimate the population mean and covariance by using the sample mean $\widehat { \pmb { \mu } }$ and covariance matrix $\widehat { \pmb { \Sigma } }$ , respectively. Specifically, the empirical quadratic Wasserstein distance serves as the new empirical uniformity loss:

$$
\mathcal { W } _ { 2 } : = \sqrt { \| \widehat { \pmb { \mu } } \| _ { 2 } ^ { 2 } + 1 + \operatorname { t r } ( \widehat { \pmb { \Sigma } } ) - \frac { 2 } { \sqrt { m } } \operatorname { t r } ( \widehat { \pmb { \Sigma } } ^ { \frac { 1 } { 2 } } ) } .
$$

Thus, $- \mathcal { W } _ { 2 }$ can be utilized as the new uniformity metric, with larger values indicating greater uniformity. Moreover, our new uniformity loss can be seamlessly integrated into various existing self-supervised learning methods to enhance their performance.

# 5 COMPARING TWO METRICS

5.1 THEORETICAL COMPARISON

We examine the proposed metric $- \mathcal { W } _ { 2 }$ in terms of the four properties introduced earlier. The following theorem summarizes our findings.

Theorem 3. The uniformity metric $- \mathcal { W } _ { 2 }$ satisfies all four properties, that is, Properties 1–4.

The proof of the above theorem is similar to that of Theorem 1, and is provided in Appendix C.2. Table 1 compares $\mathrm { - } \mathcal { L } _ { \mathcal { U } }$ and $- \mathcal { W } _ { 2 }$ . It is important to highlight that our new uniformity metric is invariant to instance permutations and sample replications, while effectively capturing feature redundancy and dimensional collapse.

Taking dimensional collapse as an example, we consider $\mathcal { D } \oplus \mathbf { 0 } ^ { k }$ versus $\mathcal { D }$ . Here, a larger $k$ indicates a more severe dimensional collapse. However, $\mathrm { - } \mathcal { L } _ { \mathcal { U } }$ fails to identify this issue, as $- \mathscr { L } _ { \mathcal { U } } ( { \mathcal { D } } \oplus \mathbf { 0 } ^ { k } ) = - \mathscr { L } _ { \mathcal { U } } ( { \mathcal { D } } )$ . In stark contrast, our proposed metric can accurately detect this dimensional collapse, as $- \mathcal { W } _ { 2 } ( \mathcal { D } \oplus \mathbf { 0 } ^ { k } ) < - \mathcal { W } _ { 2 } ( \mathcal { D } )$

Table 1: Comparing the two uniformity metrics.   

<table><tr><td>Properties</td><td>IPC</td><td>ICC</td><td>FCC</td><td>FBC</td></tr><tr><td>-Lu</td><td>√</td><td>X</td><td>X</td><td>✗</td></tr><tr><td>-W2</td><td>√</td><td>V</td><td>V</td><td></td></tr></table>

# 5.2 EMPIRICAL COMPARISONS VIA SYNTHETIC STUDIES

We perform synthetic experiments to investigate the two uniformity metrics. An empirical examination of the correlation between these metrics shows that data points following an isotropic Gaussian distribution exhibit better uniformity compared to those from other distributions; see Appendix E for detailed results. Additionally, we generate data vectors from this distribution to enable a thorough comparison between the two metrics.

On Dimensional Collapse Degrees To generate data reflecting varying degrees of dimensional collapse, we sample data vectors from an isotropic Gaussian distribution, normalize them to have $\ell _ { 2 }$ norms3, and then zero out a proportion of the coordinates. As the proportion of zero-value coordinates, denoted by $\eta$ , increases, dimensional collapse becomes more pronounced, while the proportion of non-zero coordinates is $1 - \eta$ . In Figure 3(a) and Figure 3(b), we observe that $- \mathcal { W } _ { 2 }$ effectively captures different collapse degrees, whereas $\mathrm { - } \mathcal { L } _ { \mathcal { U } }$ remains almost unchanged even with $8 0 \%$ collapse $( \eta = 8 0 \%$ ), indicating that $\mathrm { - } \mathcal { L } _ { \mathcal { U } }$ is insensitive to the degrees of dimensional collapse.

![](images/51f06022e9c51fa98beff7c6f4e2a1c6cf3b1a17205d3cd44c2941bf4d5396ac.jpg)  
Figure 3: Sensitivity to dimensional collapse degrees: $- \mathcal { W } _ { 2 }$ is more sensitive than $\mathrm { - } \mathcal { L } _ { \mathcal { U } }$ .

![](images/c9c84bcbf587ec913b0ceab79e7e7948c85278701e2708982efebfc0c6cced9e.jpg)  
Figure 4: Effectiveness of the metrics when increasing dimension $m$ : $\mathrm { - } \mathcal { L } _ { \mathcal { U } }$ fails to distinguish different dimensional collapse degrees for large $m$ , while $- \mathcal { W } _ { 2 }$ is always able to.

On Sensitiveness of Dimensions Figure 4 demonstrates that $\mathrm { - } \mathcal { L } _ { \mathcal { U } }$ can not distinguish between different degrees of dimensional collapse $\eta = 2 5 \%$ , $5 0 \%$ , and $7 5 \%$ ) as the dimension $m$ increases (e.g., $m \geq \bar { 2 ^ { 8 } } = 2 5 6 )$ . In contrast, $- \mathcal { W } _ { 2 }$ only depends on the degree of dimensional collapse and is independent of the dimensions $m$ .

To complement the theoretical comparisons between the two metrics discussed in Section 5.1, we also conduct empirical comparisons in terms of FCC and FBC. ICC comparisons are collected in Appendix E.

On Feature Cloning Constraint We investigate the impact of feature cloning by creating multiple feature clones of the dataset, such as $\mathcal { D } \oplus \mathcal { D }$ and $\mathcal { D } \oplus \mathcal { D } \oplus \mathcal { D }$ , corresponding to one and two times cloning, respectively. Figure 5(a) demonstrates that the value of $\mathrm { - } \mathcal { L } _ { \mathcal { U } }$ increases as the number of clones increases, which violates the strict decline in Eqn. (5). In contrast, in Figure 5(b), our proposed metric $- \mathcal { W } _ { 2 }$ decreases, satisfying the property.

On Feature Baby Constraint We proceed to analyze the effect of feature baby, where we insert $k$ dimensional zero vectors into each instance of $\mathcal { D }$ . This modified dataset is denoted as $\mathcal { D } \oplus \mathbf { 0 } ^ { k }$ , and we examine the impact of $k$ on both metrics. Figure 6(a) shows that the value of $\mathrm { - } \mathcal { L } _ { \mathcal { U } }$ remains constant as $k$ increases, violating the strict inequality constraint in Eqn. (6). In contrast, Figure 6(b) shows that our proposed metric $- \mathcal { W } _ { 2 }$ decreases, satisfying the constraint.

Summary of Synthetic Studies In summary, our empirical results corroborate our theoretical analysis, confirming that our proposed metric $- \mathcal { W } _ { 2 }$ outperforms the existing metric $\mathrm { - } \mathcal { L } _ { \mathcal { U } }$ in capturing feature redundancy and dimensional collapse.

![](images/3597c8ac77f93ffc82c9092a3516504e384b8dc79657d90349f429fae63efb74.jpg)  
Figure 5: FCC analysis.

![](images/28db5e5e628e3939af256dbe8eae06d0bb7d3ce357cb95505da397feff01ecfb.jpg)  
Figure 6: FBC analysis.

# 6 EXPERIMENTS

In this section, we integrate the proposed uniformity loss as an auxiliary term into various existing self-supervised methods. We then conduct experiments on CIFAR-10 and CIFAR-100 datasets to demonstrate its effectiveness.

Models We conduct experiments on a series of self-supervised representation learning models: (i) AlignUniform (Wang & Isola, 2020), which incorporates both alignment and uniformity losses in its objective function; (ii) three contrastive learning methods, namely SimCLR (Chen et al., 2020), MoCo (He et al., 2020), and NNCLR (Dwibedi et al., 2021); (iii) two asymmetric models, BYOL (Grill et al., 2020) and SimSiam (Chen & He, 2021); (iv) two methods based on redundancy reduction, BarlowTwins (Zbontar et al., 2021) and Zero-CL (Zhang et al., 2022b). To investigate the behavior of the proposed Wasserstein uniformity loss in self-supervised learning, we integrate it as an auxiliary loss into the following models: MoCo v2, BYOL, BarlowTwins, and Zero-CL. Additionally, we propose using linear decay to weight the Wasserstein uniformity loss during training. This is achieved by setting $\alpha _ { t } = \alpha _ { \operatorname* { m a x } } - t , ( \alpha _ { \operatorname* { m a x } } - \alpha _ { \operatorname* { m i n } } ) / T$ , where $t$ , $T .$ , $\alpha _ { \mathrm { m a x } }$ , $\alpha _ { \mathrm { { m i n } } }$ , and $\alpha _ { t }$ represent the current epoch, maximum epochs, maximum weight, minimum weight, and current weight, respectively. Further details on the experimental settings can be found in Appendix F.1.

Accuracy and representation capacity We assess the aforementioned methods using two distinct criteria: accuracy and representation quality/capacity. Accuracy is gauged through linear evaluation accuracy, quantified by Top-1 accuracy $( \operatorname { A c c } @ 1 )$ and Top-5 accuracy $\left( \operatorname { A c c } @ 5 \right)$ . On the other hand, representation quality/capacity is evaluated using the uniformity losses $\mathcal { L } _ { \mathcal { U } }$ and $\mathcal { W } _ { 2 }$ , along with the alignment loss $\mathcal { L } _ { A }$ . .

Main Results As depicted in Table 2, incorporating $\mathcal { W } _ { 2 }$ as an additional loss consistently yields superior performance compared to models without this loss or those with $\mathcal { L } _ { \mathcal { U } }$ as the additional term. Intriguingly, although it marginally compromises alignment, it enhances uniformity and accuracy in downstream tasks. This underscores the effectiveness of $\mathcal { W } _ { 2 }$ as a uniformity loss. Notably, integrating the Wasserstein uniformity loss does not impede training or inference efficiency.

Convergence Analysis We evaluate the Top-1 accuracy of these models on CIFAR-10 and CIFAR100 using the linear evaluation protocol, as described in Appendix F.2, across different training epochs. Figure 15 illustrates the results. By incorporating $\mathcal { W } _ { 2 }$ as an additional loss for these models, we observe faster convergence compared to the raw models, particularly for MoCo v2 and BYOL, which exhibit significant collapse issues. Our experiments demonstrate that imposing the proposed Wasserstein uniformity metric as an auxiliary penalty loss greatly enhances uniformity but may compromise alignment. We further analyze uniformity and alignment throughout all training epochs in Appendix F.3.

Table 2: Main results on CIFAR-10 and CIFAR-100. Proj. and Pred. are the hidden dimensions in projector and predictor. $\cdot$ and $\downarrow$ indicates gains and losses, respectively.   

<table><tr><td rowspan="2">Methods</td><td rowspan="2">Proj.</td><td rowspan="2">Pred.</td><td colspan="6">CIFAR-10</td><td colspan="6">CIFAR-100</td></tr><tr><td>Acc@1↑</td><td>Acc@5↑</td><td>W2↓</td><td>Lut</td><td>LA↓</td><td></td><td>Acc@1↑</td><td>Acc@5↑</td><td>W2↓</td><td>Lu↓</td><td></td><td>LA↓</td></tr><tr><td>SimCLR</td><td>256</td><td>X</td><td>89.85</td><td>99.78</td><td>1.04</td><td></td><td>-3.75 0.47</td><td></td><td>63.43</td><td>88.97</td><td>1.05</td><td>-3.75</td><td></td><td>0.50</td></tr><tr><td>NNCLR</td><td>256</td><td>256</td><td>87.46</td><td>99.63</td><td>1.23</td><td></td><td>-3.12</td><td>0.38</td><td>54.90</td><td>83.81</td><td>1.23</td><td></td><td>-3.18</td><td>0.43</td></tr><tr><td>SimSiam</td><td>256</td><td>256</td><td>86.71</td><td>99.67</td><td>1.19</td><td></td><td>-3.33</td><td>0.39</td><td>56.10</td><td>84.34</td><td>1.21</td><td></td><td>-3.29</td><td>0.42</td></tr><tr><td>AlignUniform</td><td>256</td><td>×</td><td>90.37</td><td>99.76</td><td>0.94</td><td></td><td>-3.82</td><td>0.51</td><td>65.08</td><td>90.15</td><td>0.95</td><td></td><td>-3.82 0.53</td><td></td></tr><tr><td>MoCo v2</td><td>256</td><td>X</td><td>90.65</td><td>99.81</td><td>1.06</td><td></td><td>-3.75 0.51</td><td></td><td>60.27</td><td>86.29</td><td>1.07</td><td></td><td>-3.60 0.46</td><td></td></tr><tr><td>MoCo v2 + Lu</td><td>256</td><td>×</td><td>90.98 to.33</td><td>99.67</td><td>0.98to.08</td><td></td><td>-3.82</td><td>0.53 ↓0.02</td><td>61.21 †0.94</td><td>87.32</td><td>0.98 0.09</td><td></td><td>-3.81</td><td>0.52↓0.06</td></tr><tr><td>MoCo v2 + W2</td><td>256</td><td>×</td><td>91.41 ↑0.76</td><td>99.68</td><td>0.33†0.73</td><td></td><td>-3.84</td><td>0.63 ↓0.12</td><td>63.68 ↑3.41</td><td>88.48</td><td>0.28 o.79</td><td></td><td>-3.86 0.66</td><td>↓0.20</td></tr><tr><td>BYOL</td><td>256</td><td>256</td><td>89.53</td><td>99.71</td><td>1.21</td><td></td><td>-2.99</td><td>0.31</td><td>63.66</td><td>88.81</td><td>1.20</td><td></td><td>-2.87 0.33</td><td></td></tr><tr><td>BY Cu</td><td>256</td><td>×</td><td>90.09 To.56</td><td>99.75</td><td>1.09</td><td>†o.12</td><td>-3.66</td><td>0.40 ↓0.09</td><td>62.68 Lo.98</td><td>88.44</td><td>1.08</td><td>↑o.12 -3.70</td><td></td><td>0.51 ↓0.18</td></tr><tr><td>BYOL+W2</td><td>256</td><td>256</td><td>90.31 ↑0.78</td><td>99.77</td><td>0.38</td><td>†o.83</td><td>-3.90</td><td>0.65 ↓0.34</td><td>65.16 ↑1.50</td><td>89.25</td><td>0.36</td><td>T0.84</td><td>-3.91 0.69</td><td>↓0.36</td></tr><tr><td>BarlowTwins</td><td>256</td><td>X</td><td>91.16</td><td>99.80</td><td>0.22</td><td></td><td>-3.91</td><td>0.75</td><td>68.19</td><td>90.64</td><td>0.23</td><td></td><td>-3.91 0.75</td><td></td></tr><tr><td>BarlowTwins + Lu</td><td>256</td><td>×</td><td>91.38 10.22</td><td>99.77</td><td>0.21</td><td>10.01</td><td>-3.92</td><td>0.76 ↓0.01</td><td>68.41 T0.22</td><td>90.99</td><td>0.22</td><td>To.01</td><td>-3.91</td><td>0.76 ↓0.01</td></tr><tr><td>BarlowTwins + W2</td><td>256</td><td>×</td><td>91.43↑0.27</td><td>99.78</td><td>0.190.03</td><td></td><td>-3.92</td><td>0.76 ↓0.01</td><td>68.47 ↑0.28</td><td>90.64</td><td>0.19</td><td>↑0.04</td><td>-3.91</td><td>0.79 ↓0.04</td></tr><tr><td>Zero-CL</td><td>256</td><td>X</td><td>91.35</td><td>99.74</td><td>0.15</td><td></td><td>-3.94</td><td>0.70</td><td>68.50</td><td>90.97</td><td>0.15</td><td></td><td>-3.93 0.75</td><td></td></tr><tr><td>Zero-CL + Lu</td><td>256</td><td>×</td><td>91.28 ↓0.07</td><td>99.74</td><td>0.15</td><td></td><td>-3.94</td><td>0.72 ↓0.02</td><td>68.44 ↓0.06</td><td>90.91</td><td>0.15</td><td></td><td>-3.93</td><td>0.74†o.01</td></tr><tr><td>Zero-CL + W2</td><td>256</td><td>×</td><td>91.42 T0.07</td><td>99.82</td><td>0.14</td><td>T0.01</td><td>-3.94</td><td>0.71 ↓0.01</td><td>68.55 T0.05</td><td>91.02</td><td>0.14</td><td>T0.01</td><td>-3.94 0.76</td><td>↓0.01</td></tr></table>

![](images/be9a4af53a953efb71e63a310802f27b63b8f853a64bdf7f458174c16f64ee17.jpg)  
Figure 7: Dimensional collapse analysis on CIFAR-100 dataset.

Dimensional Collapse Analysis We visualize the singular value spectra of the learned representations (Jing et al., 2022) for various models. These spectra contain the singular values of the covariance matrix of representations from CIFAR-100 dataset, sorted in logarithmic scale order. As shown in Figure 7(a), most singular values collapse to zeros in most models, indicating a large number of collapsed coordinates in most models. To further understand how the additional loss $\mathcal { W } _ { 2 }$ helps prevent dimensional collapse, we add $\mathcal { W } _ { 2 }$ as an additional loss for Moco v2 and BYOL, the numbers of collapsed coordinates decrease to zeros in both cases; see Figure 7(b) and Figure 7(c). This verifies that our proposed uniformity loss $\mathcal { W } _ { 2 }$ can effectively address the dimensional collapse issue for Moco v2 and BYOL. In contrast, $\mathcal { L } _ { \mathcal { U } }$ can not effectively prevent dimensional collapse.

# 7 CONCLUSION

In this paper, we have identified four principled properties that an effective uniformity metric should possess. Namely, an effective uniformity metric should remain invariant to instance permutations and sample replications while accurately capturing feature redundancy and dimensional collapse. Surprisingly, the popular uniformity metric proposed by Wang & Isola (2020) fails to meet the majority of these properties, unveiling its limitations. Empirical investigations corroborate our theoretical findings. To overcome these limitations, we introduce a new uniformity metric that satisfies all four properties. Particularly, this new metric demonstrates remarkable abilities to capture feature redundancy and dimensional collapse. Integrating it as an auxiliary loss in various selfsupervised learning methods effectively mitigates dimensional collapse and consistently improves their performance on downstream tasks. Nonetheless, it is worth noting that the four identified properties may not encompass a comprehensive characterization of an ideal uniformity metric, warranting further exploration.

# ACKNOWLEDGEMENT

Benyou Wang was partially supported by the Shenzhen Science and Technology Program (JCYJ20220818103001002), Shenzhen Doctoral Startup Funding (RCBS20221008093330065), and Tianyuan Fund for Mathematics of National Natural Science Foundation of China (NSFC) (12326608). Qiang Sun was partially supported in part by the Natural Sciences and Engineering Research Council of Canada under Grant RGPIN-2018-06484 and a Data Sciences Institute Catalyst Grant.

# REFERENCES

Sanjeev Arora, Hrishikesh Khandeparkar, Mikhail Khodak, Orestis Plevrakis, and Nikunj Saunshi. A theoretical analysis of contrastive unsupervised representation learning. In ICML, 2019.

Adrien Bardes, Jean Ponce, and Yann LeCun. Vicreg: Variance-invariance-covariance regularization for self-supervised learning. In ICLR, 2022.

A. Bhattacharyya. On a measure of divergence between two statistical populations defined by their probability distributions. Bulletin of the Calcutta Mathematical Society, 1943.

Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. In NeurIPS, 2020.

Mathilde Caron, Hugo Touvron, Ishan Misra, Herv’e J’egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021.

Venkat Chandrasekaran, Benjamin Recht, Pablo A Parrilo, and Alan S Willsky. The convex geometry of linear inverse problems. Foundations of Computational mathematics, 12:805–849, 2012.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey E. Hinton. A simple framework for contrastive learning of visual representations. In ICML, 2020.

Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. In CVPR, 2021.

Henry Cohn and Abhinav Kumar. Universally optimal distribution of points on spheres. Journal of the American Mathematical Society, 2007.

Victor Guilherme Turrisi da Costa, Enrico Fini, Moin Nabi, N. Sebe, and Elisa Ricci. Solo-learn: A library of self-supervised methods for visual representation learning. JMLR, 2022.

Debidatta Dwibedi, Yusuf Aytar, Jonathan Tompson, Pierre Sermanet, and Andrew Zisserman. With a little help from my friends: Nearest-neighbor contrastive learning of visual representations. In ICCV, 2021.

Tianyu Gao, Xingcheng Yao, and Danqi Chen. Simcse: Simple contrastive learning of sentence embeddings. In ArXiv, 2021.

Jean-Bastien Grill, Florian Strub, Florent Altch’e, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi ´ Azar, Bilal Piot, Koray Kavukcuoglu, Remi Munos, and Michal Valko. Bootstrap your own latent: ´ A new approach to self-supervised learning. In NeurIPS, 2020.

Raia Hadsell, Sumit Chopra, and Yann LeCun. Dimensionality reduction by learning an invariant mapping. In CVPR, 2006.

Kaiming He, X. Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016.

Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross B. Girshick. Momentum contrast for unsupervised visual representation learning. In CVPR, 2020.

Geoffrey E. Hinton, Oriol Vinyals, and Jeffrey Dean. Distilling the knowledge in a neural network. ArXiv, abs/1503.02531, 2015.

Tianyu Hua, Wenxiao Wang, Zihui Xue, Sucheng Ren, Yue Wang, and Hang Zhao. On feature decorrelation in self-supervised learning. In ICCV, 2021.

Li Jing, Pascal Vincent, Yann LeCun, and Yuandong Tian. Understanding dimensional collapse in contrastive self-supervised learning. In ICLR, 2022.

Junnan Li, Pan Zhou, Caiming Xiong, Richard Socher, and Steven C. H. Hoi. Prototypical contrastive learning of unsupervised representations. In ICLR, 2021.

Yangguang Li, Feng Liang, Lichen Zhao, Yufeng Cui, Wanli Ouyang, Jing Shao, Fengwei Yu, and Junjie Yan. Supervision exists everywhere: A data efficient contrastive language-image pre-training paradigm. In ICLR, 2022.

David Lindley and Solomon Kullback. Information theory and statistics. Journal of the American Statistical Association, 54:825, 1959.

Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. In ICLR, 2017.

Ingram Olkin and Friedrich Pukelsheim. The distance between two random vectors with given dispersion matrices. Linear Algebra and its Applications, 48:257–263, 1982.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021.

Yuandong Tian, Xinlei Chen, and Surya Ganguli. Understanding self-supervised learning dynamics without contrastive pairs. In ICML, 2021.

Ramon Van Handel. Probability in high dimension. Lecture Notes (Princeton University), 2016.

Tongzhou Wang and Phillip Isola. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In ICML, 2020.

Xinlong Wang, Rufeng Zhang, Chunhua Shen, Tao Kong, and Lei Li. Dense contrastive learning for self-supervised visual pre-training. In CVPR, 2021.

Enze Xie, Jian Ding, Wenhai Wang, Xiaohang Zhan, Hang Xu, Zhenguo Li, and Ping Luo. Detco: Unsupervised contrastive learning for object detection. In ICCV, 2021.

Ceyuan Yang, Zhirong Wu, Bolei Zhou, and Stephen Lin. Instance localization for self-supervised detection pretraining. In CVPR, 2021.

Yang You, Igor Gitman, and Boris Ginsburg. Scaling sgd batch size to 32k for imagenet training.   
ArXiv, 2017.

Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, and Stephane Deny. Barlow twins: Self-supervised ´ learning via redundancy reduction. In ICML, 2021.

Chaoning Zhang, Kang Zhang, Chenshuang Zhang, Trung X. Pham, Chang D. Yoo, and In So Kweon. How does simsiam avoid collapse without negative samples? a unified understanding with self-supervised contrastive learning. In ICLR, 2022a.

Shaofeng Zhang, Feng Zhu, Junchi Yan, Rui Zhao, and Xiaokang Yang. Zero-CL: Instance and feature decorrelation for negative-free symmetric contrastive learning. In ICLR, 2022b.

Xiangyu Zhao, Raviteja Vemulapalli, P. A. Mansfield, Boqing Gong, Bradley Green, Lior Shapira, and Ying Wu. Contrastive learning for label efficient semantic segmentation. In ICCV, 2021.

# Appendix

# Table of Contents

A Statistical distances over Gaussian distributions 1 2   
B Proof of Theorem 2 1 3   
B.1 Proofs for supporting lemmas 14   
C Examining the four properties for two uniformity metrics 15   
C.1 Proof of Theorem 1: Examining the four properties for $- \mathcal { L } _ { \mathcal { U } }$ . 15   
C.2 Proof of Theorem 3: Examining the four properties for $- \mathcal { W } _ { 2 }$ 16   
D Further comparisons between $\mathbf { Y }$ and $\widehat { \mathbf Y }$ 17   
E Additional synthetic studies 17   
E.1 Correlation between $- \mathcal { L } _ { \mathcal { U } }$ and $- \mathcal { W } _ { 2 }$ 17   
E.2 On Instance Cloning Constraint 18   
E.3 Understanding Property 4: Why does it relate to dimensional collapse? 19   
E.4 Understanding $\mathcal { W } _ { 2 }$ : Large means may lead to collapse 19

# F Experiment settings and convergence analysis

F.1 Experiment settings 20   
F.2 Convergence analysis for Top-1 accuracy 20   
F.3 Convergence analysis for uniformity and alignment . . 21

# A STATISTICAL DISTANCES OVER GAUSSIAN DISTRIBUTIONS

We first introduce the Wasserstein distance or the earth mover distance.

Definition 1. The Wasserstein distance or earth-mover distance with $p$ norm is defined as below:

$$
W _ { p } ( \mathbb { P } _ { r } , \mathbb { P } _ { g } ) = \big ( \operatorname* { i n f } _ { \gamma \in \Pi ( \mathbb { P } _ { r } , \mathbb { P } _ { g } ) } \mathbb { E } _ { ( x , y ) \sim \gamma } \big [ \| x - y \| ^ { p } \big ] \big ) ^ { 1 / p } \ .
$$

where $\Pi ( \mathbb { P } _ { r } , \mathbb { P } _ { g } )$ denotes the set of all joint distributions $\gamma ( x , y )$ whose marginals are respectively $\mathbb { P } _ { r }$ and $\mathbb { P } _ { g }$ . Intuitively, when viewing each distribution as a unit amount of earth/soil, the Wasserstein distance or earth-mover distance takes the minimum cost of transporting “mass” from $x$ to $y$ to transform the distribution $\mathbb { P } _ { r }$ into the distribution $\mathbb { P } _ { g }$ . This distance is also called the quadratic Wasserstein distance when $p = 2$ .

In this paper, we mainly exploit the quadratic Wasserstein distance over Gaussian distributions. Besides this distance, we also discuss other distribution distances as uniformity metrics and make comparisons with the Wasserstein distance. Specifically, the Kullback-Leibler divergence and the Bhattacharyya distance over Gaussian distributions are provided in Lemma 2 and Lemma 3 respectively. Both distances require full-rank covariance matrices, making them impropriate to conduct dimensional collapse analysis. In contrast, our quadratic Wasserstein distance-based uniformity metric is free of such a requirement.

Lemma 2 (Kullback-Leibler divergence (Lindley & Kullback, 1959)). Suppose two random variables $\mathbf { Z } _ { 1 } \sim \mathcal { N } ( \pmb { \mu } _ { 1 } , \pmb { \Sigma } _ { 1 } )$ and $\mathbf { Z } _ { 2 } \sim \mathcal { N } ( \mu _ { 2 } , \Sigma _ { 2 } )$ obey multivariate normal distributions, then KullbackLeibler divergence between Z1 and $\mathbf { Z } _ { 2 }$ is:

$$
D _ { \mathrm { K L } } ( \mathbf { Z } _ { 1 } , \mathbf { Z } _ { 2 } ) = { \frac { 1 } { 2 } } { \big ( } ( { \boldsymbol { \mu } } _ { 1 } - { \boldsymbol { \mu } } _ { 2 } ) ^ { T } { \boldsymbol { \Sigma } } _ { 2 } ^ { - 1 } ( { \boldsymbol { \mu } } _ { 1 } - { \boldsymbol { \mu } } _ { 2 } ) + \operatorname { t r } ( { \boldsymbol { \Sigma } } _ { 2 } ^ { - 1 } { \boldsymbol { \Sigma } } _ { 1 } - \mathbf { I } ) + \ln { \frac { \operatorname* { d e t } { \boldsymbol { \Sigma } } _ { 2 } } { \operatorname* { d e t } { \boldsymbol { \Sigma } } _ { 1 } } } { \big ) } .
$$

Lemma 3 (Bhattacharyya Distance (Bhattacharyya, 1943)). Suppose two random variables $\mathbf { Z } _ { 1 } \sim$ $\mathcal { N } ( \mu _ { 1 } , \Sigma _ { 1 } )$ and $\mathbf { Z } _ { 2 } \sim \mathcal { N } ( \pmb { \mu } _ { 2 } , \pmb { \Sigma } _ { 2 } )$ obey multivariate normal distributions, $\begin{array} { r } { \Sigma = \frac { 1 } { 2 } ( \Sigma _ { 1 } + \Sigma _ { 2 } ) } \end{array}$ , then bhattacharyya distance between $\mathbf { Z } 1$ and $\mathbf { Z } _ { 2 }$ is:

$$
\mathcal { D } _ { B } ( \mathbf { Z } _ { 1 } , \mathbf { Z } _ { 2 } ) = \frac { 1 } { 8 } ( \mu _ { 1 } - \mu _ { 2 } ) ^ { T } \Sigma ^ { - 1 } ( \mu _ { 1 } - \mu _ { 2 } ) + \frac { 1 } { 2 } \ln \frac { \operatorname* { d e t } \Sigma } { \sqrt { \operatorname* { d e t } \Sigma _ { 1 } \operatorname* { d e t } \Sigma _ { 2 } } } .
$$

# B PROOF OF THEOREM 2

We first need the following lemma, whose proof is collected in the end of this section.

Lemma 4. Let $\mathbf { Z } \sim { \mathcal { N } } ( \mathbf { 0 } , \sigma ^ { 2 } \mathbf { I } _ { m } )$ and $\mathbf { Y } = \mathbf { Z } / \lVert \mathbf { Z } \rVert _ { 2 }$ . Then the probability density function of $Y _ { i }$ , the $i$ -th coordinate of $\mathbf { Y }$ is:

$$
f _ { Y _ { i } } ( y _ { i } ) = \frac { \Gamma ( m / 2 ) } { \sqrt { \pi } \Gamma ( ( m - 1 ) / 2 ) } ( 1 - y _ { i } ^ { 2 } ) ^ { ( m - 3 ) / 2 } , \forall y _ { i } \in [ - 1 , 1 ] .
$$

We are ready to prove Theorem 2.

Proof of Theorem 2. According to the Lemma 4, the pdf of $Y _ { i }$ and $\widehat { Y } _ { i }$ are:

$$
f _ { Y _ { i } } ( y ) = \frac { \Gamma ( m / 2 ) } { \sqrt { \pi } \Gamma ( ( m - 1 ) / 2 ) } ( 1 - y ^ { 2 } ) ^ { ( m - 3 ) / 2 } , \quad f _ { \hat { r } _ { i } } ( y ) = \sqrt { \frac { m } { 2 \pi } } \exp \{ - \frac { m y ^ { 2 } } { 2 } \} .
$$

Then the Kullback-Leibler divergence between $Y _ { i }$ and $\widehat { Y } _ { i }$ is

$$
\begin{array} { l } { { \displaystyle D _ { \mathrm { K L } } ( Y _ { i } \| \widehat { Y _ { i } } ) = \int _ { - 1 } ^ { 1 } f _ { Y _ { i } } ( y ) [ \log f _ { Y _ { i } } ( y ) - \log f _ { Y _ { i } } ( y ) ] d y } \ ~ } \\ { { \displaystyle ~ = \int _ { - 1 } ^ { 1 } f _ { Y _ { i } } ( y ) [ \log \frac { \Gamma ( m / 2 ) } { \sqrt { \pi } \Gamma ( ( m - 1 ) / 2 ) } + \frac { m - 3 } { 2 } \log ( 1 - y ^ { 2 } ) - \log \sqrt { \frac { m } { 2 \pi } } + \frac { m y ^ { 2 } } { 2 } ] d y } \ ~ } \\ { { \displaystyle ~ = \log \sqrt { \frac { 2 } { m } } \frac { \Gamma ( m / 2 ) } { \Gamma ( ( m - 1 ) / 2 ) } + \int _ { - 1 } ^ { 1 } f _ { Y _ { i } } ( y ) [ \frac { m - 3 } { 2 } \log ( 1 - y ^ { 2 } ) + \frac { m y ^ { 2 } } { 2 } ] d y } . } \end{array}
$$

Letting $\mu = { y } ^ { 2 }$ , we have $y = { \sqrt { \mu } }$ and $\begin{array} { r } { d y = \frac { 1 } { 2 } \mu ^ { - \frac { 1 } { 2 } } d u } \end{array}$ . Thus,

$$
\begin{array} { l } { A : = \displaystyle \int _ { - 1 } ^ { 1 } f _ { \gamma _ { i } } ( y ) [ \frac { m - 3 } { 2 } \log ( 1 - y ^ { 2 } ) + \frac { m y ^ { 2 } } { 2 } ] d y } \\ { = 2 \displaystyle \int _ { 0 } ^ { 1 } \frac { \Gamma ( m / 2 ) } { \sqrt { \pi \Gamma ( ( m - 1 ) / 2 ) } } ( 1 - y ^ { 2 } ) ^ { \frac { m - 3 } { 2 } } [ \frac { m - 3 } { 2 } \log ( 1 - y ^ { 2 } ) + \frac { m y ^ { 2 } } { 2 } ] d y } \\ { = \displaystyle \frac { \Gamma ( m / 2 ) } { \sqrt { \pi \Gamma ( ( m - 1 ) / 2 ) } } \int _ { 0 } ^ { 1 } ( 1 - \mu ) ^ { \frac { m - 3 } { 2 } } [ \frac { m - 3 } { 2 } \log ( 1 - \mu ) + \frac { m } { 2 } \mu ] \mu ^ { - \frac 1 2 } d \mu } \\ { = \displaystyle \frac { \Gamma ( m / 2 ) } { \sqrt { \pi \Gamma ( ( m - 1 ) / 2 ) } } \frac { m - 3 } { 2 } \int _ { 0 } ^ { 1 } ( 1 - \mu ) ^ { \frac { m - 3 } { 2 } } \mu ^ { - \frac 1 2 } \log ( 1 - \mu ) d \mu } \\ { \displaystyle + \frac { \Gamma ( m / 2 ) } { \sqrt { \pi \Gamma ( ( m - 1 ) / 2 ) } } \frac { m } { 2 } \int _ { 0 } ^ { 1 } ( 1 - \mu ) ^ { \frac { m - 3 } { 2 } } \mu ^ { \frac 1 2 } d \mu . } \end{array}
$$

By using the property of Beta distribution, and the inequality that $\begin{array} { r } { \frac { - \mu } { 1 - \mu } \leq \log ( 1 - \mu ) \leq - \mu } \end{array}$ , we have

$$
\begin{array} { l } { { A _ { 1 } : = \displaystyle \frac { \Gamma ( m / 2 ) } { \sqrt { \pi } \Gamma ( ( m - 1 ) / 2 ) } \displaystyle \frac { m - 3 } { 2 } \int _ { 0 } ^ { 1 } ( 1 - \mu ) ^ { \frac { m - 3 } { 2 } } \mu ^ { - \frac { 1 } { 2 } } \log ( 1 - \mu ) d \mu } } \\ { { \ \leq - \displaystyle \frac { \Gamma ( m / 2 ) } { \sqrt { \pi } \Gamma ( ( m - 1 ) / 2 ) } \displaystyle \frac { m - 3 } { 2 } \int _ { 0 } ^ { 1 } ( 1 - \mu ) ^ { \frac { m - 3 } { 2 } } \mu ^ { \frac { 1 } { 2 } } d \mu } } \\ { { \ = - \displaystyle \frac { \Gamma ( m / 2 ) } { \sqrt { \pi } \Gamma ( ( m - 1 ) / 2 ) } \displaystyle \frac { m - 3 } { 2 } B ( \frac { 3 } { 2 } , \frac { m - 1 } { 2 } ) \mathrm { a n d } } } \\ { { \ \mathcal { A } _ { 2 } : = \displaystyle \frac { \Gamma ( m / 2 ) } { \sqrt { \pi } \Gamma ( ( m - 1 ) / 2 ) } \displaystyle \frac { m } { 2 } \int _ { 0 } ^ { 1 } ( 1 - \mu ) ^ { \frac { m - 3 } { 2 } } \mu ^ { \frac { 1 } { 2 } } d \mu } } \\ { { \ = \displaystyle \frac { \Gamma ( m / 2 ) } { \sqrt { \pi } \Gamma ( ( m - 1 ) / 2 ) } \displaystyle \frac { m } { 2 } B ( \frac { 3 } { 2 } , \frac { m - 1 } { 2 } ) . } } \end{array}
$$

Then, for $\mathcal { A }$ , we have

$$
\begin{array} { l } { { A = A _ { 1 } + A _ { 2 } \le - \displaystyle \frac { \Gamma ( m / 2 ) } { \sqrt { \pi \Gamma ( ( m - 1 ) / 2 ) } } \displaystyle \frac { m - 3 } { 2 } B ( \frac { 3 } { 2 } , \frac { m - 1 } { 2 } ) + \displaystyle \frac { \Gamma ( m / 2 ) } { \sqrt { \pi \Gamma ( ( m - 1 ) / 2 ) } } \displaystyle \frac { m } { 2 } B ( \frac { 3 } { 2 } , \frac { m - 1 } { 2 } ) } } \\ { { = \displaystyle \frac { 3 } { 2 } \displaystyle \frac { \Gamma ( m / 2 ) } { \sqrt { \pi \Gamma ( ( m - 1 ) / 2 ) } } B ( \frac { 3 } { 2 } , \frac { m - 1 } { 2 } ) = \displaystyle \frac { 3 } { 2 } \displaystyle \frac { \Gamma ( m / 2 ) } { \sqrt { \pi } \Gamma ( ( m - 1 ) / 2 ) } \displaystyle \frac { \Gamma ( 3 / 2 ) \Gamma ( ( m - 1 ) / 2 ) } { \Gamma ( ( m + 2 ) / 2 ) } } } \\ { { = \displaystyle \frac { 3 } { 2 } \displaystyle \frac { \Gamma ( 3 / 2 ) \Gamma ( m / 2 ) } { \sqrt { \pi \Gamma ( ( m + 2 ) / 2 ) } } = \displaystyle \frac { 3 } { 2 } \displaystyle \frac { \Gamma ( \pi / 2 ) \Gamma ( m / 2 ) } { \sqrt { \pi } \Gamma ( ( m + 2 ) / 2 ) } = \displaystyle \frac { 3 } { 4 } \displaystyle \frac { \Gamma ( m / 2 ) } { \Gamma ( ( m + 2 ) / 2 ) } . } } \end{array}
$$

Using the Stirling formula, we have $\Gamma ( x + \alpha )  \Gamma ( x ) x ^ { \alpha }$ as $x \to \infty$ and thus

$$
\begin{array} { r l } { \underset { m  \infty } { \operatorname* { l i m } } D _ { \mathrm { K L } } ( Y _ { i } \Vert \widehat { Y _ { i } } ) = \underset { m  \infty } { \operatorname* { l i m } } \log \sqrt { \frac { 2 } { m } } \frac { \Gamma ( m / 2 ) } { \Gamma ( ( m - 1 ) / 2 ) } + \underset { m  \infty } { \operatorname* { l i m } } \mathcal { A } } & { } \\ { \leq \underset { m  \infty } { \operatorname* { l i m } } \log \sqrt { \frac { 2 } { m } } \frac { \Gamma ( ( m - 1 ) / 2 ) ( \frac { m - 1 } { 2 } ) ^ { 1 / 2 } } { \Gamma ( ( m - 1 ) / 2 ) } + \underset { m  \infty } { \operatorname* { l i m } } \frac { 3 } { 4 } \frac { \Gamma ( m / 2 ) } { \Gamma ( ( m + 2 ) / 2 ) } } & { } \\ { = \underset { m  \infty } { \operatorname* { l i m } } \log \sqrt { \frac { 2 } { m } } \sqrt { \frac { m - 1 } { 2 } } + \frac { 3 } { 4 } \frac { \Gamma ( m / 2 ) } { \Gamma ( m / 2 ) m } = \underset { m  \infty } { \operatorname* { l i m } } \log \sqrt { \frac { m - 1 } { m } } + \frac { 3 } { 4 m } = 0 . } \end{array}
$$

We further use $T _ { 2 }$ inequality (Van Handel, 2016, Theorem 4.31) to derive the quadratic Wasserstein metric (Van Handel, 2016, Definition 4.29) as:

$$
\operatorname* { l i m } _ { m  \infty } \mathcal { W } _ { 2 } ( Y _ { i } , \widehat { Y _ { i } } ) \leq \operatorname* { l i m } _ { m  \infty } \sqrt { \frac { 2 } { m } D _ { \mathrm { K L } } ( Y _ { i } \| \widehat { Y _ { i } } ) } = 0 .
$$

# B.1 PROOFS FOR SUPPORTING LEMMAS

Proof of Lemma 4. Let $\mathbf { Z } = \vert Z _ { 1 } , Z _ { 2 } , \cdot \cdot \cdot , Z _ { m } \vert \sim { \mathcal { N } } ( \mathbf { 0 } , \sigma ^ { 2 } \mathbf { I } _ { m } )$ , then $Z _ { i } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ) , \forall i \in [ 1 , m ]$ . Let $\dot { U } \stackrel { \cdot } { = } Z _ { i } / \sigma \sim { \mathcal N } ( 0 , 1 )$ , $\begin{array} { r } { V \dot { = } \sum _ { j \neq i } ^ { m } ( Z _ { j } / \sigma ) ^ { 2 } \sim \mathcal { X } ^ { 2 } ( m - 1 ) , } \end{array}$ , then $U$ and $V$ are independent with each other. The random variable $\begin{array} { r } { \dot { T } = \frac { U } { \sqrt { V / ( m - 1 ) } } } \end{array}$ follows the Student’s t-distribution with $m - 1$ degrees of freedom, and its probability density function (pdf) is:

$$
f _ { T } ( t ) = \frac { \Gamma ( m / 2 ) } { \sqrt { ( m - 1 ) \pi } \Gamma ( ( m - 1 ) / 2 ) } ( 1 + \frac { t ^ { 2 } } { m - 1 } ) ^ { - m / 2 } .
$$

For random variable $Y _ { i }$ , we have

$$
Y _ { i } = \frac { Z _ { i } } { \sqrt { \sum _ { i = 1 } ^ { m } Z _ { i } ^ { 2 } } } = \frac { Z _ { i } } { \sqrt { Z _ { i } ^ { 2 } + \sum _ { j \neq i } ^ { m } Z _ { j } ^ { 2 } } } = \frac { Z _ { i } / \sigma } { \sqrt { ( Z _ { i } / \sigma ) ^ { 2 } + \sum _ { j \neq i } ^ { m } ( Z _ { j } / \sigma ) ^ { 2 } } } = \frac { U } { \sqrt { U ^ { 2 } + V } } ,
$$

and then $\begin{array} { r } { T = \frac { U } { \sqrt { V / ( m - 1 ) } } = \frac { \sqrt { m - 1 } Y _ { i } } { \sqrt { 1 - Y _ { i } ^ { 2 } } } } \end{array}$ , $\begin{array} { r } { Y _ { i } = \frac { T } { \sqrt { T ^ { 2 } + m - 1 } } } \end{array}$ . Therefore, the cumulative distribution function (cdf) of $T$ is:

$$
\begin{array} { r l } { F _ { Y _ { i } } ( y _ { i } ) = P ( \{ Y _ { i } \leq y _ { i } \} ) = \{ P ( \{ Y _ { i } \leq y _ { i } \} )  } & { y _ { k } \leq 0 } \\ & { = \{ P ( \{ Y _ { i } \leq 0 \} ) + P ( \{ 0 < Y _ { i } \leq y _ { k } \} )  } & { y _ { k } > 0 } \\ & { = \{ P ( \{ \frac { { Y } } { \sqrt { 7 ^ { 2 } + m - 1 } } \leq y _ { k } \} )  } & { y _ { k } \leq 0 } \\ {  P ( \{ \frac { { Y } } { \sqrt { 7 ^ { 2 } + m - 1 } } \leq 0 \} ) + P ( \{ 0 < \frac { { T } } { \sqrt { 7 ^ { 2 } + m - 1 } } \leq y _ { k } \} )  } & { y _ { k } > 0 } \\ & { = \{ P ( \{ \frac { { Y } } { \sqrt { 7 ^ { 2 } + m - 1 } } > y _ { k } ^ { 2 } , T \leq 0 \} ) \} } & { y _ { k } \leq 0 } \\ {  P ( \{ Y \leq 0 \} + P ( \{ \frac { { T } ^ { 2 } } { \sqrt { 1 - y _ { k } ^ { 2 } } } \leq y _ { k } ^ { 2 } , T > 0 \} ) \quad \frac { y _ { k } > 0 } { y _ { k } > 0 }  } \\ & { = \{ P ( \{ Y \leq \frac { { \sqrt { m - 1 } } y _ { k } } { \sqrt { 1 - y _ { k } ^ { 2 } } } \} ) \quad  } & { y _ { k } \leq 0  } \\ {  P ( \{ X \leq 0 \} + P ( \{ 0 < T \leq \frac { { \sqrt { m - 1 } } y _ { k } } { \sqrt { 1 - y _ { k } ^ { 2 } } } \} ) \quad \frac { y _ { k } > 0 } { y _ { k } > 0 }   } \\ &  =  P ( \{ Y \leq \frac { { \sqrt { m } - 1 } y _ { k } } { \sqrt { 1 - y _ { k } ^ { 2 } } } \} ) = F _ { T } ( \frac   \sqrt  m  \end{array}
$$

The probability density function of $Y _ { i }$ can then be derived as:

$$
\begin{array} { l } { \displaystyle f _ { Y _ { i } } ( y _ { i } ) = \frac { d } { d y _ { i } } F _ { Y _ { i } } ( y _ { i } ) = \frac { d } { d y _ { i } } F _ { T } ( \frac { \sqrt { m - 1 } y _ { i } } { \sqrt { 1 - y _ { i } ^ { 2 } } } ) } \\ { \displaystyle ~ = f _ { T } ( \frac { \sqrt { m - 1 } y _ { i } } { \sqrt { 1 - y _ { i } ^ { 2 } } } ) \frac { d } { d y _ { i } } ( \frac { \sqrt { m - 1 } y _ { i } } { \sqrt { 1 - y _ { i } ^ { 2 } } } ) } \\ { \displaystyle ~ = [ \frac { \Gamma ( m / 2 ) } { \sqrt { ( m - 1 ) \pi } \Gamma ( ( m - 1 ) / 2 ) } ( 1 - y _ { i } ^ { 2 } ) ^ { m / 2 } ] [ \sqrt { m - 1 } ( 1 - y _ { i } ^ { 2 } ) ^ { - 3 / 2 } ] } \\ { \displaystyle ~ = \frac { \Gamma ( m / 2 ) } { \sqrt { \pi \Gamma ( ( m - 1 ) / 2 ) } } ( 1 - y _ { i } ^ { 2 } ) ^ { ( m - 3 ) / 2 } . } \end{array}
$$

# C EXAMINING THE FOUR PROPERTIES FOR TWO UNIFORMITY METRICS

C.1 PROOF OF THEOREM 1: EXAMINING THE FOUR PROPERTIES FOR $\mathrm { - } \mathcal { L } _ { \mathcal { U } }$

Property 1 can be easily verified for $\mathrm { - } \mathcal { L } _ { \mathcal { U } }$ and thus we skip the verification. We only examine the other three properties for the uniformity metric $\mathrm { - } \mathcal { L } _ { \mathcal { U } }$ .

First, we prove that $\mathrm { - } \mathcal { L } _ { \mathcal { U } }$ does not satisfy Property 2. Due to the definition of $\mathcal { L } _ { \mathcal { U } }$ in Eqn. (2), we have

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { \mathcal { U } } ( \mathcal { D } \uplus \mathcal { D } ) : = \log \frac { 1 } { 2 n ( 2 n - 1 ) / 2 } \left( 4 \sum _ { i = 2 } ^ { n } \sum _ { j = 1 } ^ { i - 1 } e ^ { - t \| \mathbf { z } _ { i } - \mathbf { z } _ { j } \| _ { 2 } ^ { 2 } } + \sum _ { i = 1 } ^ { n } e ^ { - t \| \mathbf { z } _ { i } - \mathbf { z } _ { i } \| _ { 2 } ^ { 2 } } \right) } \\ { \displaystyle \qquad = \log \frac { 1 } { 2 n ( 2 n - 1 ) / 2 } \left( 4 \sum _ { i = 2 } ^ { n } \sum _ { j = 1 } ^ { i - 1 } e ^ { - t \| \mathbf { z } _ { i } - \mathbf { z } _ { j } \| _ { 2 } ^ { 2 } } + n \right) . } \end{array}
$$

Letting G = Pni=2 Pi−1j=1 e $\begin{array} { r } { G = \sum _ { i = 2 } ^ { n } \sum _ { j = 1 } ^ { i - 1 } e ^ { - t \| \mathbf { z } _ { i } - \mathbf { z } _ { j } \| _ { 2 } ^ { 2 } } } \end{array}$ , we have

$$
G = \sum _ { i = 2 } ^ { n } \sum _ { j = 1 } ^ { i - 1 } e ^ { - t \| \mathbf z _ { i } - \mathbf z _ { j } \| _ { 2 } ^ { 2 } } \leq \sum _ { i = 2 } ^ { n } \sum _ { j = 1 } ^ { i - 1 } e ^ { - t \| \mathbf z _ { i } - \mathbf z _ { i } \| _ { 2 } ^ { 2 } } = n ( n - 1 ) / 2 ,
$$

and $G = n ( n - 1 ) / 2$ if and only if $\mathbf { z } _ { 1 } = \mathbf { z } _ { 2 } = \mathbf { . ~ . ~ } = \mathbf { z } _ { n }$ . Thus

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathcal { U } } ( \mathcal { D } \mathbin { \uplus } \mathcal { D } ) - \mathcal { L } _ { \mathcal { U } } ( \mathcal { D } ) = \log \frac { 4 G + n } { 2 n ( 2 n - 1 ) / 2 } - \log \frac { G } { n ( n - 1 ) / 2 } } \\ & { \qquad = \log \frac { ( 4 G + n ) n ( n - 1 ) / 2 } { 2 n G ( 2 n - 1 ) / 2 } = \log \frac { ( 4 G + n ) ( n - 1 ) } { 4 n G - 2 G } } \\ & { \qquad = \log \frac { 4 n G - 4 G + n ^ { 2 } - n } { 4 n G - 2 G } \geq \log 1 = 0 . } \end{array}
$$

The above equality holds if and only if $G = n ( n - 1 ) / 2$ , which requires $\mathbf { z } _ { 1 } = \mathbf { z } _ { 2 } = { \ldots } = \mathbf { z } _ { n }$ , a trivial case when all representations collapse to one constant point. We have excluded this trivial case, and thus $- \mathscr { L } _ { \mathcal { U } } ( \mathscr { D } \uplus \mathscr { D } ) < - \mathscr { L } _ { \mathcal { U } } ( \mathscr { D } )$ . Therefore, the uniformity metric $\mathrm { - } \mathcal { L } _ { \mathcal { U } }$ does not satisfy Property 2.

Second, we prove that $\mathrm { - } \mathcal { L } _ { \mathcal { U } }$ does not satisfy Property 3. Letting ${ \widehat { \mathbf { z } } } _ { i } = \mathbf { z } _ { i } \oplus \mathbf { z } _ { i }$ and $\widehat { \mathbf { z } } _ { j } = \mathbf { z } _ { j } \oplus \mathbf { z } _ { j }$ , we have

$$
\mathcal { L } _ { \mathcal { U } } ( \mathcal { D } \oplus \mathcal { D } ) : = \log \frac { 1 } { n ( n - 1 ) / 2 } \sum _ { i = 2 } ^ { n } \sum _ { j = 1 } ^ { i - 1 } e ^ { - t | | \widehat { \mathbf { z } } _ { i } - \widehat { \mathbf { z } } _ { j } | | _ { 2 } ^ { 2 } } .
$$

By the definitions of $\widehat { \mathbf { z } } _ { i }$ and $\widehat { \mathbf { z } } _ { j }$ , we have $\| \widehat { \mathbf { z } } _ { i } \| _ { 2 } = \sqrt { 2 } \| \mathbf { z } _ { i } \| _ { 2 }$ , $\| \widehat { \mathbf { z } } _ { j } \| _ { 2 } = \sqrt { 2 } \| \mathbf { z } _ { j } \| _ { 2 }$ , and $\langle \widehat { \mathbf { z } } _ { i } , \widehat { \mathbf { z } } _ { j } \rangle =$ $2 \langle \mathbf { z } _ { i } , \mathbf { z } _ { j } \rangle$ . Thus

$$
\| \widehat { \mathbf { z } } _ { i } - \widehat { \mathbf { z } } _ { j } \| _ { 2 } ^ { 2 } = 2 \| \mathbf { z } _ { i } \| _ { 2 } ^ { 2 } + 2 \| \mathbf { z } _ { j } \| _ { 2 } ^ { 2 } - 4 \langle \mathbf { z } _ { i } , \mathbf { z } _ { j } \rangle = 2 \| \mathbf { z } _ { i } - \mathbf { z } _ { j } \| _ { 2 } ^ { 2 } \geq \| \mathbf { z } _ { i } - \mathbf { z } _ { j } \| _ { 2 } ^ { 2 } .
$$

Therefore, $- \mathcal { L } _ { \mathcal { U } } ( \mathcal { D } \oplus \mathcal { D } ) \ge - \mathcal { L } _ { \mathcal { U } } ( \mathcal { D } )$ , indicating that the uniformity metric $\mathrm { - } \mathcal { L } _ { \mathcal { U } }$ does not satisfy the Property 3.

Third, we prove that the existing metric $\mathrm { - } \mathcal { L } _ { \mathcal { U } }$ does not satisfy the Property 4. Letting $\widehat { \mathbf { z } } _ { i } = \mathbf { z } _ { i } \oplus \mathbf { 0 } ^ { k }$ and $\widehat { \mathbf z } _ { j } = \dot { \mathbf z } _ { j } \oplus \mathbf 0 ^ { k }$ , we have

$$
\mathcal { L } _ { \mathcal { U } } ( \mathcal { D } \oplus \mathbf { 0 } ^ { k } ) : = \log \frac { 1 } { n ( n - 1 ) / 2 } \sum _ { i = 2 } ^ { n } \sum _ { j = 1 } ^ { i - 1 } e ^ { - t | | \widehat { \mathbf { z } } _ { i } - \widehat { \mathbf { z } } _ { j } | | _ { 2 } ^ { 2 } } .
$$

By the definitions of $\widehat { \mathbf { z } } _ { i }$ and $\widehat { \mathbf { z } } _ { j }$ , we have $\| \widehat { \mathbf { z } } _ { i } \| _ { 2 } = \| \mathbf { z } _ { i } \| _ { 2 } , \| \widehat { \mathbf { z } } _ { j } \| _ { 2 } = \| \mathbf { z } _ { j } \| _ { 2 } , \langle \widehat { \mathbf { z } } _ { i } , \widehat { \mathbf { z } } _ { j } \rangle = \langle \mathbf { z } _ { i } , \mathbf { z } _ { j } \rangle$ , and thus

$$
\| \widehat { \mathbf { z } } _ { i } - \widehat { \mathbf { z } } _ { j } \| _ { 2 } ^ { 2 } = \| \widehat { \mathbf { z } } _ { i } \| _ { 2 } ^ { 2 } + \| \widehat { \mathbf { z } } _ { j } \| _ { 2 } ^ { 2 } - 2 \langle \widehat { \mathbf { z } } _ { i } , \widehat { \mathbf { z } } _ { j } \rangle = \| \mathbf { z } _ { i } \| _ { 2 } ^ { 2 } + \| \mathbf { z } _ { j } \| _ { 2 } ^ { 2 } - 2 \langle \mathbf { z } _ { i } , \mathbf { z } _ { j } \rangle = \| \mathbf { z } _ { i } - \mathbf { z } _ { j } \| _ { 2 } ^ { 2 } .
$$

Therefore, $- \mathscr { L } _ { \mathcal { U } } ( \mathscr { D } \oplus \mathbf { 0 } ^ { k } ) = - \mathscr { L } _ { \mathcal { U } } ( \mathscr { D } )$ , indicating that the uniformity metric $\mathrm { - } \mathcal { L } _ { \mathcal { U } }$ does not satisfy Property 4.

C.2 PROOF OF THEOREM 3: EXAMINING THE FOUR PROPERTIES FOR $- \mathcal { W } _ { 2 }$

Property 1 can be easily verified for $- \mathcal { W } _ { 2 }$ , and thus the proof is skipped. We only examine the rest three properties for the proposed uniformity metric $- \mathcal { W } _ { 2 }$ .

First, we prove that our proposed metric $- \mathcal { W } _ { 2 }$ satisfies Property 2. Let $\widehat { \mu }$ and $\widehat { \pmb { \Sigma } }$ be defined as above, for $\mathcal { D } \not \sqcup \mathcal { D } = \{ \mathbf { z } _ { 1 } , \mathbf { z } _ { 2 } , . . . , \mathbf { z } _ { n } , \mathbf { z } _ { 1 } , \mathbf { z } _ { 2 } , . . . , \mathbf { z } _ { n } \}$ b, the mean and covariance estimators are

$$
\widetilde { \pmb { \mu } } = \frac { 1 } { 2 n } \sum _ { i = 1 } ^ { n } 2 \mathbf { z } _ { i } = \widehat { \pmb { \mu } } , \quad \widetilde { \pmb { \Sigma } } = \frac { 1 } { 2 n } \sum _ { i = 1 } ^ { n } 2 ( \mathbf { z } _ { i } - \widetilde { \pmb { \mu } } ) ^ { T } ( \mathbf { z } _ { i } - \widetilde { \pmb { \mu } } ) = \widehat { \pmb { \Sigma } } ,
$$

which agree with those for $\mathcal { D }$ . Then we have

$$
\mathcal { W } _ { 2 } ( \mathcal { D } \uplus \mathcal { D } ) : = \sqrt { \| \hat { \mu } \| _ { 2 } ^ { 2 } + 1 + \mathrm { t r } ( \widehat { \Sigma } ) - \frac { 2 } { \sqrt { m } } \mathrm { t r } ( \widehat { \Sigma } ^ { 1 / 2 } ) } = \mathcal { W } _ { 2 } ( \mathcal { D } ) .
$$

Therefore, our proposed metric $- \mathcal { W } _ { 2 }$ satisfies Property 2.

Second, we prove that $- \mathcal { W } _ { 2 }$ satisfies Property 3. Let $\widetilde { \mathbf { z } } _ { i } = \mathbf { z } _ { i } \oplus \mathbf { z } _ { i } \in \mathbb { R } ^ { 2 m }$ . For $\mathcal { D } \oplus \mathcal { D }$ , the mean and covariance estimators are:

$$
\widetilde { \pmb { \mu } } = \left( \frac { \widehat { \pmb { \mu } } } { \widehat { \pmb { \mu } } } \right) , \quad \widetilde { \pmb { \Sigma } } = \left( \begin{array} { c c } { \widehat { \pmb { \Sigma } } } & { \widehat { \pmb { \Sigma } } } \\ { \widehat { \pmb { \Sigma } } } & { \widehat { \pmb { \Sigma } } } \end{array} \right) .
$$

We easily have

$\widetilde { \pmb { \Sigma } } ^ { 1 / 2 } = \left( \frac { \widehat { \pmb { \Sigma } } ^ { 1 / 2 } / \sqrt { 2 } } { \widehat { \pmb { \Sigma } } ^ { 1 / 2 } / \sqrt { 2 } } \quad \widehat { \pmb { \Sigma } } ^ { 1 / 2 } / \sqrt { 2 } \right) , \mathrm { t r } ( \widetilde { \pmb { \Sigma } } ) = 2 \mathrm { t r } ( \widehat { \pmb { \Sigma } } )$ , and $\mathrm { t r } ( \widetilde { \Sigma } ^ { 1 / 2 } ) = \sqrt { 2 } \mathrm { t r } ( \widehat { \Sigma } ^ { 1 / 2 } )$ .

Thus

$$
\begin{array} { l } { \displaystyle \mathcal { W } _ { 2 } ( D \oplus D ) : = \sqrt { \| \widetilde { \mu } \| _ { 2 } ^ { 2 } + 1 + \mathrm { t r } ( \widetilde { \Sigma } ) - \frac { 2 } { \sqrt { 2 m } } \mathrm { t r } ( \widetilde { \Sigma } ^ { 1 / 2 } ) } } \\ { \displaystyle \quad = \sqrt { 2 \| \widehat { \mu } \| _ { 2 } ^ { 2 } + 1 + 2 \mathrm { t r } ( \widehat { \Sigma } ) - \frac { 2 \sqrt { 2 } } { \sqrt { 2 m } } \mathrm { t r } ( \widehat { \Sigma } ^ { 1 / 2 } ) } } \\ { \displaystyle \quad > \sqrt { \| \widehat { \mu } \| _ { 2 } ^ { 2 } + 1 + \mathrm { t r } ( \widehat { \Sigma } ) - \frac { 2 } { \sqrt { m } } \mathrm { t r } ( \widehat { \Sigma } ^ { 1 / 2 } ) } = \mathcal { W } _ { 2 } ( D ) . } \end{array}
$$

Therefore, $- \mathcal { W } _ { 2 } ( \mathcal { D } \oplus \mathcal { D } ) < - \mathcal { W } _ { 2 } ( \mathcal { D } )$ , indicating that our proposed metric $- \mathcal { W } _ { 2 }$ could satisfy the Property 3.

Third, we prove that our proposed metric $- \mathcal { W } _ { 2 }$ satisfies Property 4. Let $\widetilde { \mathbf { z } } _ { i } = \mathbf { z } _ { i } \oplus \mathbf { 0 } ^ { k } \in \mathbb { R } ^ { m + k }$ with an overload of notation. For $\mathcal { D } \oplus \mathbf { 0 } ^ { k }$ , the sample mean and covariance estimators are

$$
\widetilde { \pmb { \mu } } = \left( \frac { \widehat { \pmb { \mu } } } { \mathbf { 0 } ^ { k } } \right) , \quad \widetilde { \pmb { \Sigma } } = \left( \begin{array} { c c } { \widehat { \pmb { \Sigma } } } & { \mathbf { 0 } ^ { m \times k } } \\ { \mathbf { 0 } ^ { k \times m } } & { \mathbf { 0 } ^ { k \times k } } \end{array} \right) ,
$$

where $\widehat { \mu }$ and $\widehat { \pmb { \Sigma } }$ are defined previously. Therefore, we have $\operatorname { t r } ( \widetilde { \Sigma } ) = \operatorname { t r } ( \widehat { \Sigma } )$ , $\mathrm { t r } ( \widetilde { \Sigma } ^ { 1 / 2 } ) = \mathrm { t r } ( \widehat { \Sigma } ^ { 1 / 2 } )$ , band thus

$$
\begin{array} { l } { \displaystyle \mathcal { W } _ { 2 } ( { \mathcal D } \oplus { \bf 0 } ^ { k } ) : = \sqrt { \| \widetilde { \pmb { \mu } } \| _ { 2 } ^ { 2 } + 1 + \mathrm { t r } ( \widetilde { \pmb { \Sigma } } ) - \frac { 2 } { \sqrt { m + k } } \mathrm { t r } ( \widetilde { \pmb { \Sigma } } ^ { 1 / 2 } ) } } \\ { \displaystyle \qquad = \sqrt { \| \widehat { \pmb { \mu } } \| _ { 2 } ^ { 2 } + 1 + \mathrm { t r } ( \widehat { \pmb { \Sigma } } ) - \frac { 2 } { \sqrt { m + k } } \mathrm { t r } ( \widehat { \pmb { \Sigma } } ^ { 1 / 2 } ) } } \\ { \displaystyle \qquad > \sqrt { \| \widehat { \pmb { \mu } } \| _ { 2 } ^ { 2 } + 1 + \mathrm { t r } ( \widehat { \pmb { \Sigma } } ) - \frac { 2 } { \sqrt { m } } \mathrm { t r } ( \widehat { \pmb { \Sigma } } ^ { 1 / 2 } ) } = \mathcal { W } _ { 2 } ( { \mathcal D } ) . } \end{array}
$$

Therefore, $- \mathcal { W } _ { 2 } ( \mathcal { D } \oplus \mathbf { 0 } ^ { k } ) \ < \ - \mathcal { W } _ { 2 } ( \mathcal { D } )$ , indicating that our proposed metric $- \mathcal { W } _ { 2 }$ satisfies the Property 4.

# D FURTHER COMPARISONS BETWEEN Y AND $\widehat { \mathbf Y }$

This section further compares the distributions of $\mathbf { Y }$ and $\widehat { \mathbf Y }$ .

We visually compare the distributions of $Y _ { i }$ and $\widehat { Y } _ { i }$ . To estimate the distributions of $Y _ { i }$ and $\widehat { Y } _ { i }$ , we bin 200,000 sampled data points into 51 groups. Figure 8 compares the binning densities of $Y _ { i }$ and $\widehat { Y } _ { i }$ when $m \in \{ 2 , 4 , 8 , 1 6 , 3 2 , 6 4 , 1 2 8 , 2 5 6 \}$ . We can observe that two distributions are highly overlapped when $m$ is moderately large, e.g., $m \geq 8$ or $m \geq 1 6$ .

By binning 2,000,000 data points into $5 1 \times 5 1$ groups in two-axis, we also analyze the joint binning densities and present 2D joint binning densities of $( Y _ { i } , Y _ { j } )$ $( i \neq j )$ ) in Figure 9(a) and $( \widehat { Y } _ { i } , \widehat { Y } _ { j } ) \ : ( i \neq j )$ in Figure 9(b). Even if $m$ is relatively small (i.e., 32), the densities of the two distributions are close.

# E ADDITIONAL SYNTHETIC STUDIES

E.1 CORRELATION BETWEEN $\mathrm { - } \mathcal { L } _ { \mathcal { U } }$ AND $- \mathcal { W } _ { 2 }$

![](images/ca2887b81fbc40f67bfc1556ace4589149c81b2d52ee161150c870c5ad213e2e.jpg)  
Figure 8: Comparing the binning densities of $Y _ { i }$ and $\widehat { Y } _ { i }$ with various dimensions.

![](images/c954488a5455b09fafe6e696e2b394f4af3ad5b4bb7c76120842ba7aaa0a1ccd.jpg)  
Figure 9: Visualization of two arbitrary dimensions for $\mathbf { Y }$ and $\widehat { \mathbf Y }$ when $m = 3 2$

We employ synthetic experiments to study the uniformity metrics across different distributions. Specifically, we sample 50,000 data vectors $m = 2 5 6 )$ ) from different distributions, such as the isotropic Gaussian distribution $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , the uniform distribution on the hyperrectangle $[ \mathbf { 0 } , \mathbf { 1 } ]$ , and the mixture of Gaussians, etc. Then we normalize these data vectors, and estimate the uniformity of different distributions by two metrics. As shown in Fig. 10, isotropic Gaussian distribution achieves the maximum values for both $- \mathcal { W } _ { 2 }$ and $\mathrm { - } \mathcal { L } _ { \mathcal { U } }$ , which indicates that isotropic Gaussian distribution achieves larger uniformity than other distributions. This empirical result is consistent with Fact 1 that the isotropic Gaussian distribution (approximately) achieves the maximum uniformity.

![](images/b322f6501803a453c7e4acf83fda33759409e81458834f3e56e50a0760e3167d.jpg)  
Figure 10: Uniformity analysis for various distributions by two metrics.

# E.2 ON INSTANCE CLONING CONSTRAINT

In this section, we compare the two metrics in terms of Property 2 (ICC). Specifically, we randomly sample 1,000 data vectors from the isotropic Gaussian distribution ( $m = 3 2$ ) and then mask $5 0 \%$ of their coordinates with zeros, forming a new dataset $\mathcal { D }$ with an overload of notation. To investigate the impact of instance cloning, we create multiple clones of the dataset, such as $\mathcal { D } \uplus \mathcal { D }$ and ${ \mathcal { D } } \uplus { \mathcal { D } } \uplus { \mathcal { D } }$ , which correspond to one and two times cloning, respectively. We evaluate the two metrics on these datasets. Figure 11 shows that the value of $\mathrm { - } \mathcal { L } _ { \mathcal { U } }$ slightly decreases as the number of clones increases, indicating that $\mathrm { - } \mathcal { L } _ { \mathcal { U } }$ violates the equality in Equation 4. In contrast, our proposed metric $- \mathcal { W } _ { 2 }$ remains constant, satisfying the equality.

![](images/d8f59858a33f4bb0dedd8ffac000a784f2728376169fecb2fe50c3a8d769a101.jpg)  
Figure 11: ICC analysis.

![](images/11f82bedd261049e22e5b36230d619541caf3ecd02fbd33b1ddcf0281b16400c.jpg)  
Figure 12: A case study for Property 4 and blue points are data vectors.

E.3 UNDERSTANDING PROPERTY 4: WHY DOES IT RELATE TO DIMENSIONAL COLLAPSE?

This section delves into Property 4 through case studies. Let us begin with a thought experiment. Consider a dataset $\mathcal { D }$ with instances uniformly distributed on the unit hypersphere, thereby possessing (almost) maximal uniformity. When additional coordinates with zeros are inserted to each instance of $\mathcal { D }$ , forming a new dataset $\mathcal { D } \oplus \mathbf { 0 } ^ { k }$ , it can no longer maintain maximal uniformity. This is because, the new dataset only occupies a small area of the unit hypersphere. Consequently, as $k$ increases, the uniformity of the dataset would decrease significantly.

Let us visualize this thought experiment using synthetic studies. In Figure 12(a), we present 400 data vectors $( \mathcal { D } _ { 1 } )$ sampled from $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { 2 } )$ , which are also nearly uniformly distributed on $S ^ { 1 }$ . By inserting one zero-coordinate to each instance of $\mathcal { D } _ { 1 }$ , we obtain a new dataset $\mathcal { D } _ { 1 } \oplus \mathbf { 0 } ^ { 1 }$ , as depicted in Figure 12(b). We also construct another dataset $\mathcal { D } _ { 2 }$ consisting of 400 data vectors sampled from $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { 3 } )$ , visualized in Figure 12(c). Notably, $\mathcal { D } _ { 1 } \oplus \mathbf { 0 } ^ { 1 }$ forms a ring on $S ^ { 2 }$ , while $\mathcal { D } _ { 2 }$ is almost uniformly distributed over $S ^ { 2 }$ . Naturally, $\mathcal { U } ( \mathcal { D } _ { 2 } ) > \mathcal { U } ( \mathcal { D } _ { 1 } \oplus \mathbf { 0 } ^ { 1 } )$ . If $\bar { \mathcal { U } } ( \mathcal { D } _ { 1 } ) = \mathcal { U } ( \mathcal { D } _ { 2 } ) ^ { 4 }$ , then $\mathcal { U } ( \mathcal { D } _ { 1 } ) = \mathcal { U } ( \mathcal { D } _ { 2 } ) > \mathcal { U } ( \mathcal { D } _ { 1 } \oplus \mathbf { 0 } ^ { 1 } )$ . This partially confirms the validity of Property 4.

Additionally, increasing the value of $k$ in Property 4 exacerbates the degree of dimensional collapse. To illustrate, consider a dataset $\mathcal { D }$ sampled from a multivariate Gaussian distribution $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { m } / m )$ , exhibiting a collapse degree close to $0 \%$ . However, upon inserting $m$ -dimensional zero-value vectors to each instance of $\mathcal { D }$ , denoted as $\mathcal { D } \oplus \mathbf { 0 } ^ { m }$ , half of the dimensions collapse. Consequently, the collapse degree increases to $5 0 \%$ . Figure 13 visually represents the collapse of $\mathcal { D } \oplus \mathbf { 0 } ^ { k }$ using the singular value spectra of the representations. It is evident that a larger $k$ results in a more pronounced mensional collapse. In summary, Property 4 corresponds to dimensional collapse.

![](images/5ca0f6e3a43c2a1978978b733d1ef342a5e9f7cdb9a9db084f3b7d7a1772e6f2.jpg)  
Figure 13: Singular value spectrum of $\mathcal { D } \oplus$ 0k .

# E.4 UNDERSTANDING $\mathcal { W } _ { 2 }$ : LARGE MEANS MAY LEAD TO COLLAPSE

In this section, we explore our uniformity loss $\mathcal { W } _ { 2 }$ . This loss embodies two primary constraints. Firstly, it promotes the covariance matrix to be isotropic (specifically ${ \bf I } _ { m } / m \vert$ ). Secondly, it enforces the mean to be zero. The latter constraint on the mean is crucial. To illustrate, we present a case study demonstrating that deviating the mean from zero compromises uniformity, even if the covariance matrix is precisely ${ \bf I } _ { m } / m$ and thus isotropic. Means deviating from zero may result in dimensional collapse and even constant collapse.

![](images/03c4fcff5ba3376e7312e5eb178c471abc13caff932f3d3c36c24f190c6daba6.jpg)  
Figure 14: Visualizing $\ell _ { 2 }$ normalized Gaussian vectors with different means.

able 3: Parameter settings for various models in the experiments.   

<table><tr><td>Models</td><td>MoCo v2</td><td>BYOL</td><td>BarlowTwins</td><td>Zero-CL</td></tr><tr><td>αmax</td><td>1.0</td><td>0.2</td><td>30.0</td><td>30.0</td></tr><tr><td>αmin</td><td>1.0</td><td>0.2</td><td>0</td><td>30.0</td></tr></table>

Assuming $\mathbf { X } \in \mathbb { R } ^ { 2 }$ follows a Gaussian distribution $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { 2 } )$ , let $\mathbf { Y } = \mathbf { X } + \boldsymbol { k } \cdot \mathbf { 1 }$ such that $\mathbf { Y } \sim$ $\mathcal { N } ( k \cdot { \bf 1 } , \bar { \bf 1 } _ { 2 } )$ , where $\mathbf { 1 } \in \mathbb { R } ^ { k }$ represents a vector of all ones. We vary $k$ from 0 to 32 and visualize the $\ell _ { 2 }$ -normalized $\mathbf { Y }$ ’s in Figure 14 (by generating multiple independent copies). It is clear that an excessively large means will cause representations to collapse to a single point, even if the covariance matrix is isotropic.

# F EXPERIMENT SETTINGS AND CONVERGENCE ANALYSIS

# F.1 EXPERIMENT SETTINGS

To ensure fair comparisons, all experiments in Section 6 are conducted on a single 1080 GPU. Additionally, we maintain consistency in network architecture across all models, utilizing ResNet18 (He et al., 2016) as the backbone and a three-layer MLP as the projector. The LARS optimizer (You et al., 2017) is employed with a base learning rate of 0.2, accompanied by a cosine decay learning rate schedule (Loshchilov & Hutter, 2017) for all models. Evaluation follows a linear evaluation protocol, where models are pre-trained for 500 epochs. Evaluation involves adding a linear classifier and training the classifier for 100 epochs while preserving the learned representations. The same augmentation strategy is deployed across all models, encompassing various operations such as color distortion, rotation, and cutout. Following da Costa et al. (2022), we set the temperature $t = 0 . 2$ for all contrastive learning methods. For MoCo (He et al., 2020) and NNCLR (Dwibedi et al., 2021), which require an additional queue to store negative samples, we set the queue size to $2 ^ { 1 2 }$ . Regarding the linear decay for weighting the quadratic Wasserstein distance, refer to Table 3 for the parameter settings.

# F.2 CONVERGENCE ANALYSIS FOR TOP-1 ACCURACY

Here we illustrate the convergence of Top-1 accuracy across all training epochs in Fig 15. Throughout the training, we capture the model checkpoint at the end of each epoch to train a linear classifier. We subsequently evaluate the Top-1 accuracy on unseen images from the test set (either CIFAR-10 or CIFAR-100).

For both CIFAR-10 and CIFAR-100, we observe that integrating the proposed uniformity metric as an auxiliary loss significantly enhances the Top-1 accuracy, particularly in the initial stages of training.

# F.3 CONVERGENCE ANALYSIS FOR UNIFORMITY AND ALIGNMENT

This section presents the convergence of the uniformity metric and alignment loss across all training epochs in Figure 16 and Figure 17, respectively. Throughout the training, we record the model checkpoint at the end of each epoch to evaluate the uniformity using the proposed metric $\mathcal { W } _ { 2 }$ and alignment (Wang & Isola, 2020) on unseen images from the test set (either CIFAR-10 or CIFAR-100).

For both CIFAR-10 and CIFAR-100, we observe that integrating the proposed uniformity metric as an auxiliary loss significantly improves uniformity. However, it also slightly compromises alignment (where a smaller alignment loss indicates better alignment). It should be noted that improved uniformity often leads to worse alignment.

![](images/f3c108314252345fd7cce76affaae971abc57cddf1b4e909aa049cfb24a8e468.jpg)

![](images/9ac6a97ef4499c03f5b85e86386f81cb19f72f9f21942b835b9852fd6540f580.jpg)  
Figure 15: Convergence analysis for Top-1 accuracy during training.   
Figure 16: Visualizing uniformity during training

![](images/6789a277e4da688ee05e9fbd0f081643e0bf0fd1d29bead41a78a39eb0fce3b1.jpg)  
Figure 17: Visualizing alignment during training.