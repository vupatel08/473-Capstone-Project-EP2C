# Graph Distillation with Eigenbasis Matching

Yang Liu \* 1 Deyu Bo \* 1 Chuan Shi 1

# Abstract

The increasing amount of graph data places requirements on the efficient training of graph neural networks (GNNs). The emerging graph distillation (GD) tackles this challenge by distilling a small synthetic graph to replace the real large graph, ensuring GNNs trained on real and synthetic graphs exhibit comparable performance. However, existing methods rely on GNN-related information as supervision, including gradients, representations, and trajectories, which have two limitations. First, GNNs can affect the spectrum (i.e., eigenvalues) of the real graph, causing spectrum bias in the synthetic graph. Second, the variety of GNN architectures leads to the creation of different synthetic graphs, requiring traversal to obtain optimal performance. To tackle these issues, we propose Graph Distillation with Eigenbasis Matching (GDEM), which aligns the eigenbasis and node features of real and synthetic graphs. Meanwhile, it directly replicates the spectrum of the real graph and thus prevents the influence of GNNs. Moreover, we design a discrimination constraint to balance the effectiveness and generalization of GDEM. Theoretically, the synthetic graphs distilled by GDEM are restricted spectral approximations of the real graphs. Extensive experiments demonstrate that GDEM outperforms state-of-the-art GD methods with powerful crossarchitecture generalization ability and significant distillation efficiency. Our code is available at https://github.com/liuyang-tian/GDEM.

# 1. Introduction

Graph neural networks (GNNs) are proven effective in a variety of graph-related tasks (Kipf & Welling, 2017; Velickovic et al., 2018). However, the non-Euclidean nature of graph structure presents challenges to the efficiency and scalability of GNNs (Hamilton et al., 2017). To accelerate training, one data-centric approach is to summarize the large-scale graph into a much smaller one. Traditional methods primarily involve sparsification (Spielman & Srivastava, 2011; Yu et al., 2022) and coarsening (Loukas, 2019; Kumar et al., 2023). However, these methods are typically designed to optimize some heuristic metrics, e.g., spectral similarity (Loukas, 2019) and pair-wise distance (Ahmed et al., 2020), which may be irrelevant to downstream tasks, leading to sub-optimal performance.

Recently, graph distillation (GD), a.k.a., graph condensation, has attracted considerable attention in graph reduction due to its remarkable compression ratio and lossless performance (Gao et al., 2024). Generally, GD aims to synthesize a small graph wherein GNNs trained on it exhibit comparable performance to those trained on the real large graph. To this end, existing methods are designed to optimize the synthetic graphs by matching some GNN-related information, such as gradients (Jin et al., 2022b;a), representations (Liu et al., 2022a), and training trajectories (Zheng et al., 2023), between the real and synthetic graphs. As a result, the synthetic graph aligns its distribution with the real graph and also incorporates information from downstream tasks.

Despite the considerable progress, existing GD methods require pre-selecting a specific GNN as the distillation model, introducing two limitations: (1) GNNs used for distillation affect the real spectrum, leading to spectrum bias in the synthetic graph, i.e., a few eigenvalues dominate the data distribution. Figure 1 illustrates the total variation (TV) (Gutman & Zhou, 2006) of the real and synthetic graphs. Notably, TV reflects the smoothness of the signal over a graph. A small value of TV indicates a low-frequency distribution, and vice versa. We can observe that the values of TV in the synthetic graph distilled by a low-pass filter consistently appear lower than those in the real graph, while the opposite holds for the high-pass filter, thus verifying the existence of spectrum bias (See Section 3 for a theoretical analysis). (2) The optimal performance is obtained by traversing various GNN architectures, resulting in non-negligible computational costs. Table 1 presents the cross-architecture results of GCOND (Jin et al., 2022b) across six well-known GNNs, including GCN (Kipf & Welling, 2017), SGC (Wu et al., 2019), PPNP (Klicpera et al., 2019), ChebyNet (Defferrard et al., 2016), BernNet (He et al., 2021), and GPRGNN (Chien et al., 2021). It can be seen that the evaluation performance of different GNNs varies greatly. As a result, existing GD methods need to distill and traverse various GNN architectures to obtain optimal performance, which significantly improves the time overhead. See Appendix A.1 for the definition of TV and Appendix A.2 for more experimental details.

![](images/4f2946a95a879769f9475800e3e091a4f55b35b9e6a73fce242fbbdb50638392.jpg)  
Figure 1. Data distribution of the real and synthetic graphs in Pubmed dataset, where the average TV of the real graph is 0.87. Left: Synthetic graph distilled by a low-pass filter has a lower value of TV (0.75). Right: Synthetic graph distilled by a high-pass filter has a higher value of TV (1.02). For clarity, only the first 100-dimensional features are visualized. Best viewed in color.

Table 1. Cross-architecture performance $( \% )$ of GCOND with various distillation (D) and evaluation (E) GNNs in Pubmed dataset. Bold indicates the best in each column.   

<table><tr><td>DE</td><td>GCN</td><td>SGC</td><td>PPNP</td><td>Cheb.</td><td>Bern.</td><td>GPR.</td></tr><tr><td>GCN</td><td>74.57</td><td>71.70</td><td>75.53</td><td>70.13</td><td>68.40</td><td>71.73</td></tr><tr><td>SGC</td><td>77.72</td><td>77.60</td><td>77.34</td><td>76.03</td><td>74.42</td><td>76.52</td></tr><tr><td>PPNP</td><td>72.70</td><td>70.40</td><td>77.46</td><td>73.38</td><td>70.56</td><td>74.02</td></tr><tr><td>Cheb.</td><td>73.60</td><td>70.62</td><td>75.10</td><td>77.30</td><td>77.62</td><td>78.10</td></tr><tr><td>Bern.</td><td>67.68</td><td>73.76</td><td>74.30</td><td>77.20</td><td>78.12</td><td>78.28</td></tr><tr><td>GPR.</td><td>76.04</td><td>72.20</td><td>77.94</td><td>75.92</td><td>77.12</td><td>77.96</td></tr></table>

Once the weaknesses of existing methods are identified, it is natural to ask: How to distill graphs without being affected by different GNNs? To answer this question, we propose Graph Distillation with Eigenbasis Matching (GDEM). Specifically, GDEM decomposes the graph structure into eigenvalues and eigenbasis. During distillation, GDEM matches the eigenbasis and node features of real and synthetic graphs, which equally preserves the information of different frequencies, thus addressing the spectrum bias. Additionally, a discrimination loss is jointly optimized to improve the performance of GDEM and balance its effectiveness and generalization. Upon completing the matching, GDEM leverages the real graph spectrum and synthetic eigenbasis to construct a complete synthetic graph, which prevents the spectrum from being affected by GNNs and ensures the uniqueness of the synthetic graph, thus avoiding the traversal requirement and improving the distillation efficiency.

The contributions of our paper are as follows. (1) We systematically analyze the limitations of existing distillation methods, including spectrum bias and traversal requirement. (2) We propose GDEM, a novel graph distillation framework, which mitigates the dependence on GNNs by matching the eigenbasis instead of the entire graph structure. Additionally, it is theoretically demonstrated that GDEM preserves essential spectral similarity during distillation. (3) Extensive experiments on seven graph datasets validate the superiority of GDEM over state-of-the-art GD methods in terms of effectiveness, generalization, and efficiency.

# 2. Preliminary

Before describing our framework in detail, we first introduce some notations and concepts used in this paper. Specifically, we focus on the node classification task, where the goal is to predict the labels of the nodes in a graph. Assume that there is a graph $\mathcal { G } = ( \nu , \mathcal { E } , \mathbf { X } )$ , where $\nu$ is the set of nodes with $| \nu | = N$ , $\mathcal { E }$ indicates the set of edges, and $\mathbf { X } \in \mathbb { R } ^ { N \times d }$ is the node feature matrix. The adjacency matrix of $\mathcal { G }$ is defined as $\mathbf { A } \in \{ 0 , 1 \} ^ { N \times N }$ , where $A _ { i j } = 1$ if there is an edge between nodes $i$ and $j$ , and $A _ { i j } = 0$ otherwise. The corresponding normalized Laplacian matrix is defined as ${ \bf L } = { \dot { \bf I } } _ { N } ^ { - } - { \bf D } ^ { - { \frac { 1 } { 2 } } } { \bf A } { \bf D } ^ { - { \frac { 1 } { 2 } } }$ , where ${ \mathbf { I } } _ { N }$ is an identity matrix and $\mathbf { D }$ is the degree matrix with $\begin{array} { r } { D _ { i i } = \sum _ { j } A _ { i j } } \end{array}$ for $i \in \mathcal V$ and $D _ { i j } = 0$ for $i \neq j$ . Without loss of generality, we assume that $\mathcal { G }$ is undirected and all the nodes are connected.

Eigenbasis and Eigenvalue. The normalized graph Laplacian can be decomposed as $\begin{array} { r } { { \bf L } = { \bf U } { \bf \Lambda } { \bf A } { \bf U } ^ { \top } = \sum _ { i = 1 } ^ { \breve { N } } \dot { \lambda } _ { i } { \bf u } _ { i } { \bf { \bar { u } } } _ { i } ^ { \top } } \end{array}$ where $\mathbf { \Lambda } \mathbf { \Lambda } = \mathrm { d i a g } ( \{ \lambda _ { i } \} _ { i = 1 } ^ { N } )$ are the eigenvalues and ${ \textbf { U } } =$ $[ { \bf u } _ { 1 } , \cdot \cdot \cdot , { \bf u } _ { N } ] \in \mathbb { R } ^ { N \times N }$ is the eigenbasis, consisting of a set of eigenvectors. Each eigenvector ${ \bf u } _ { i } \in \mathbb { R } ^ { N }$ has a corresponding eigenvalue $\lambda _ { i }$ , such that $\mathbf { L } \mathbf { u } _ { i } = \lambda _ { i } \mathbf { u } _ { i }$ . Without loss of generality, we assume $0 \le \lambda _ { 1 } \le \cdots \le \lambda _ { N } \le 2$ .

Graph Distillation. GD aims to distill a small synthetic graph $\mathcal { G } ^ { \prime } = ( \mathcal { V } ^ { \prime } , \mathcal { E } ^ { \prime } , { \bf X } ^ { \prime } )$ , where $| \mathcal { V } ^ { \prime } | ~ = ~ N ^ { \prime } ~ \ll ~ N$ and $\bar { \mathbf { X } } ^ { \prime } \in \mathbb { R } ^ { N ^ { \prime } \times d }$ , from the real large graph $\mathcal { G }$ . Meanwhile, GNNs trained on $\mathcal { G }$ and $\mathcal { G } ^ { \prime }$ will have comparable performance, thus accelerating the training of GNNs. Existing frameworks can be divided into three categories: gradient matching, distribution matching, and trajectory matching. See Appendix C for more detailed descriptions.

# 3. Spectrum Bias in Gradient Matching

In this section, we give a detailed analysis of the objective of gradient matching in graph data, which motivates the design of our method. We start with a vanilla example, which adopts a one-layer GCN as the distillation model and simplifies the objective of GNNs into the MSE loss:

$$
\mathcal { L } = \frac { 1 } { 2 } \left. \mathbf { A X W } - \mathbf { Y } \right. _ { F } ^ { 2 } ,
$$

where W is the model parameter. The gradients on the real and synthetic graphs are calculated as follows:

$$
\nabla _ { \mathbf { W } } = \left( \mathbf { A X } \right) ^ { T } \left( \mathbf { A X W } - \mathbf { Y } \right) ,
$$

$$
\nabla _ { \mathbf { W } } ^ { \prime } = \left( \mathbf { A ^ { \prime } X ^ { \prime } } \right) ^ { T } \left( \mathbf { A ^ { \prime } X ^ { \prime } W } - \mathbf { Y ^ { \prime } } \right) .
$$

Assume that the objective of gradient matching is the MSE loss between two gradients, i.e., $\mathcal { L } _ { G M } = \| \nabla _ { \mathbf { W } } - \nabla _ { \mathbf { W } } ^ { \prime } \| _ { F } ^ { 2 }$ To further characterize its properties, we analyze the following upper-bound of $\mathcal { L } _ { G M }$ :

$$
\begin{array} { r l } & { \mathcal { L } _ { G M } \leq \| \mathbf { W } \| _ { F } ^ { 2 } \| \mathbf { X } ^ { \top } \mathbf { A } ^ { 2 } \mathbf { X } - { \mathbf { X ^ { \prime } } } ^ { \top } { \mathbf { A ^ { \prime } } } ^ { 2 } \mathbf { X ^ { \prime } } \| _ { F } ^ { 2 } } \\ & { \qquad + \| \mathbf { X } ^ { \top } \mathbf { A } \mathbf { Y } - { \mathbf { X ^ { \prime } } } ^ { \top } \mathbf { A ^ { \prime } } \mathbf { Y ^ { \prime } } \| _ { F } ^ { 2 } , } \end{array}
$$

where $\mathbf { X } ^ { \top } \mathbf { A } ^ { 2 } \mathbf { X }$ and $\mathbf { X } ^ { \top } \mathbf { A Y }$ are two target distributions in the real graph, which are used to supervise the update of the synthetic graph. However, both of them will be dominated by a few eigenvalues, resulting in spectrum bias.

Lemma 3.1. The target distribution of GCN is dominated by the smallest eigenvalue after stacking multiple layers. Proof. The target distribution can be reformulated as:

$$
\mathbf { X } ^ { \top } \mathbf { A } ^ { 2 t } \mathbf { X } = \sum _ { i = 1 } ^ { N } ( 1 - \lambda _ { i } ) ^ { 2 t } \mathbf { X } ^ { \top } \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } \mathbf { X } ,
$$

where $t$ is the number of layers. When $t$ goes to infinity, only the smallest eigenvalue $\lambda _ { 0 } = 0$ preserves its coefficient $( 1 - \lambda _ { 0 } ) ^ { 2 t } = 1$ and other coefficients tend to 0. Hence, the target distribution $\mathbf { X } ^ { \top } \mathbf { A } ^ { 2 t } \mathbf { X }$ is dominated by $\mathbf { X } ^ { \top } \mathbf { u } _ { 0 } \mathbf { u } _ { 0 } ^ { \top } \mathbf { X }$ The same analysis can be applied for $\mathbf { X } ^ { \top } \mathbf { A } ^ { t } \mathbf { \bar { Y } }$ . □

Lemma 3.2. Suppose the distillation GNN has an analytic filtering function $g ( \cdot )$ . Then the target distributions will be dominated by the eigenvalues whose filtered values are greater than $^ { l }$ , i.e., $g ( \lambda _ { i } ) \geq 1$ .

Proof. The objective function of distillation GNN is ${ \mathcal { L } } =$ $\begin{array} { r } { \frac { 1 } { 2 } \| g ( { \bf L } ) { \bf X } { \bf W } - { \bf Y } \| _ { F } ^ { 2 } } \end{array}$ . Then the target distributions become $\bar { \mathbf { X } } ^ { \top } g ( \mathbf { L } ) ^ { 2 t } \mathbf { X }$ and $\mathbf { X } ^ { \top } g ( \mathbf { L } ) ^ { t } \mathbf { Y }$ as $g$ is analytic. Therefore, the filtered eigenvalues with values $g ( \lambda _ { i } ) \geq 1$ retain their coefficients and dominate the target distributions. □

Lemmas 3.1 and 3.2 state that leveraging the information of GNNs in distillation will introduce a spectral bias in the target distributions. As a result, the synthetic graph can only match part of the data distribution of the real graph, leaving its structural information incomplete.

# 4. The Proposed Method: GDEM

In this section, we introduce the proposed method GDEM. Compared with previous methods, e.g., gradient matching (Figure 2(a)) and distribution matching (Figure 2(b)), GDEM, illustrated in 2(c), does not rely on specific GNNs, whose distillation process can be divided into two steps: (1) Matching the eigenbasis and node features between the real and synthetic graphs. (2) Constructing the synthetic graph by using the synthesized eigenbasis and real spectrum.

![](images/b78491fa1bdcb6454710d9151813d2159462cbb5571eb0a52cec6d0f71838f74.jpg)  
Figure 2. Comparison between different graph distillation methods, where the red characters represent the synthetic data, the solid black lines, and red dotted lines indicate the forward and backward passes, respectively.

# 4.1. Eigenbasis Matching

The eigenbasis of a graph represents its crucial structural information. For example, eigenvectors corresponding to smaller eigenvalues reflect the global community structure, while eigenvectors corresponding to larger eigenvalues encode local details (Bo et al., 2021). Generally, the number of eigenvectors is the same as the number of nodes in a graph, suggesting that we cannot preserve all the real eigenbasis in the synthetic graph. Therefore, GDEM is designed to match eigenvectors with the $K _ { 1 }$ smallest and the $K _ { 2 }$ largest eigenvalues, where $K _ { 1 }$ and $K _ { 2 }$ are hyperparameters, and $K _ { 1 } + K _ { 2 } = K \leq N ^ { \prime }$ . This approach has been proven effective in both graph coarsening (Jin et al., 2020) and spectral GNNs (Bo et al., 2023). We initialize a matrix $\mathbf { U } _ { K } ^ { \prime } \ = \ [ \mathbf { u } _ { 1 } ^ { \prime } , \cdot \cdot \cdot , \mathbf { u } _ { N ^ { \prime } } ^ { \prime } ] \ \in \ \mathbb { R } ^ { N ^ { \prime } \times K }$ to match the principal eigenbasis of the real graph, denoted as $\mathbf { U } _ { K } = [ \mathbf { u } _ { 1 } , \therefore \cdot \cdot , \mathbf { u } _ { K _ { 1 } } , \mathbf { u } _ { N - K _ { 2 } } , \cdot \cdot \cdot , \mathbf { u } _ { N } ] \in \mathbb { R } ^ { N \times K }$ .

To eliminate the influence of GNNs, GDEM does not use the spectrum information during distillation. Therefore, the first term in Equation 3 becomes:

$$
\mathcal { L } _ { e } = \sum _ { k = 1 } ^ { K } \left. \mathbf { X } ^ { \top } \mathbf { u } _ { k } \mathbf { u } _ { k } ^ { \top } \mathbf { X } - { \mathbf { X ^ { \prime } } } ^ { \top } \mathbf { u } _ { k } ^ { \prime } { \mathbf { u } _ { k } ^ { \prime } } ^ { \top } \mathbf { X ^ { \prime } } \right. _ { F } ^ { 2 } ,
$$

where ${ \mathbf { u } } _ { k } { \mathbf { u } } _ { k } ^ { \top }$ and $\mathbf { u } _ { k } ^ { \prime } \mathbf { u } _ { k } ^ { \prime } ^ { \top }$ are the subspaces induced by the $k$ -th eigenvector in the real and synthetic graphs.

Additionally, as the basis of graph Fourier transform, eigenvectors are naturally normalized and orthogonal to each other. However, directly optimizing $\mathbf { U } _ { K } ^ { \prime }$ via gradient de

scent cannot preserve this property. Therefore, an additional regularization is used to constrain the representation space:

$$
\mathcal { L } _ { o } = \left. \mathbf { U } _ { K } ^ { \prime } ^ { \top } \mathbf { U } _ { K } ^ { \prime } - \mathbf { I } _ { K } \right. _ { F } ^ { 2 } .
$$

See Appendix A.3 for more implementation details.

# 4.2. Discrimination Constraint

In practice, we find that eigenbasis matching improves the cross-architecture generalization of GDEM but contributes less to the performance of node classification as it only preserves the global distribution, i.e., $\mathbf { X } ^ { \top } \mathbf { u } \mathbf { u } ^ { \top } \mathbf { X }$ , without considering the information of downstream tasks. Therefore, we need to approximate the second term in Equation 3. Interestingly, we find that $\mathbf { X } ^ { \top } \mathbf { A } \mathbf { Y } \in \mathbb { R } ^ { d \times C }$ indicates the category-level representations, which assigns each category a $d$ -dimensional representation. However, the MSE loss only emphasizes the intra-class similarity between the real and synthetic graphs and ignores the inter-class dissimilarity.

Based on this discovery, we design a discrimination constraint to effectively preserve the category-level information, which can also be treated as a class-aware regularization technique (Zhao et al., 2023; Wang et al., 2022). Specifically, we first learn the category-level representations of the real and synthetic graphs:

$$
\mathbf { H } = \mathbf { Y } ^ { \top } \mathbf { A } \mathbf { X } , \quad \mathbf { H } ^ { \prime } = { \mathbf { Y } ^ { \prime } } ^ { \top } \sum _ { k = 1 } ^ { K } ( 1 - \lambda _ { k } ) { \mathbf { u } _ { k } ^ { \prime } } { \mathbf { u } _ { k } ^ { \prime } } ^ { \top } \mathbf { X } ^ { \prime } ,
$$

where $\lambda _ { k }$ is the $k$ -th eigenvalue of the real graph Laplacian. We then constrain the cosine similarity between $\mathbf { H }$ and $\mathbf { H } ^ { \prime }$ :

$$
\mathcal { L } _ { d } = \sum _ { i = 1 } ^ { C } \left( 1 - \frac { \mathbf { H } _ { i } ^ { \top } \cdot \mathbf { H } _ { i } ^ { \prime } } { \vert \vert \mathbf { H } _ { i } \vert \vert \vert \vert \mathbf { H } _ { i } ^ { \prime } \vert \vert } \right) + \sum _ { i , j = 1 \atop i \neq j } ^ { C } \frac { \mathbf { H } _ { i } ^ { \top } \cdot \mathbf { H } _ { j } ^ { \prime } } { \vert \vert \mathbf { H } _ { i } \vert \vert \vert \vert \mathbf { H } _ { j } ^ { \prime } \vert \vert } .
$$

Note that the discrimination constraint introduces the spectrum information in the distillation process, which conflicts with the eigenbasis matching. However, we find that adjusting the weights of eigenbasis matching and the discrimination constraint can balance the performance and generalization of GDEM. Ablation studies can be seen in Section 6.5.

# 4.3. Final Objective and Synthetic Graph Construction

In summary, the overall loss function of GDEM is formulated as the weighted sum of three regularization terms:

$$
\mathcal { L } _ { t o t a l } = \alpha \mathcal { L } _ { e } + \beta \mathcal { L } _ { d } + \gamma \mathcal { L } _ { o } ,
$$

where $\alpha , \beta$ , and $\gamma$ are the hyperparameters. The pseudocode of GDEM is presented in Algorithm 1.

Upon minimizing the total loss function, the outputs of GDEM are the eigenbasis and node features of the synthetic

Input: Real graph $\mathcal { G } \ = \ ( \mathbf { A } , \mathbf { X } , \mathbf { Y } )$ with eigenvalues   
$\{ \lambda _ { i } \} _ { i = 1 } ^ { K }$ and eigenbasis ${ \bf U } _ { K }$   
Init: Synthetic graph $\mathcal { G } ^ { \prime }$ with eigenbasis $\mathbf { U } _ { K } ^ { \prime }$ , node fea  
tures $\mathbf { X } ^ { \prime }$ , and labels $\mathbf { Y } ^ { \prime }$   
for $t = 1$ to $T$ do Compute $\mathcal { L } _ { e } , \mathcal { L } _ { o }$ , and $\mathcal { L } _ { d }$ via Eqs. 5, 6, and 8 Compute $\mathcal { L } _ { t o t a l } = \alpha \mathcal { L } _ { e } + \beta \mathcal { L } _ { d } + \gamma \mathcal { L } _ { o }$ if $t \% ( \tau _ { 1 } + \tau _ { 2 } ) < \tau _ { 1 }$ then Update $\mathbf { U } _ { K } ^ { \prime }  \mathbf { U } _ { K } ^ { \prime } - \eta _ { 1 } \nabla _ { \mathbf { U } _ { K } ^ { \prime } } \mathcal { L } _ { t o t a l }$ else Update $\mathbf { X } ^ { \prime }  \mathbf { X } ^ { \prime } - \eta _ { 2 } \nabla _ { \mathbf { X } ^ { \prime } } \mathcal { L } _ { t o t a l }$ end if   
end for   
Compute $\begin{array} { r } { \mathbf { A } ^ { \prime } = \sum _ { k = 1 } ^ { K } ( 1 - \lambda _ { k } ) \mathbf { u } _ { k } ^ { \prime } \mathbf { u } _ { k } ^ { \prime } ^ { \top } } \end{array}$   
Return: A′, X′

graph. However, the data remains incomplete due to the absence of the graph spectrum. Essentially, the graph spectrum encodes the global shape of a graph (Martinkus et al., 2022). Ideally, if the synthetic graph preserves the distribution of the real graph, they should have similar spectrums. Therefore, we directly replicate the real spectrum for the synthetic graph to construct its Laplacian matrix or adjacency matrix:

$$
\mathbf { L } ^ { \prime } = \sum _ { k = 1 } ^ { K } \lambda _ { k } \mathbf { u } _ { k } ^ { \prime } \mathbf { u } _ { k } ^ { \prime } { } ^ { \top } , \quad \mathbf { A } ^ { \prime } = \sum _ { k = 1 } ^ { K } ( 1 - \lambda _ { k } ) \mathbf { u } _ { k } ^ { \prime } { } \mathbf { u } _ { k } ^ { \prime } { } ^ { \top } .
$$

# 4.4. Discussion

Complexity. The complexity of decomposition is $\mathcal { O } ( N ^ { 3 } )$ However, given that we only utilize the $K$ smallest or largest eigenvalues, the complexity reduces to $\mathcal { O } ( K N ^ { 2 } )$ . Additionally, $\mathbf { u } _ { k } ^ { \top } \mathbf { X }$ in Equation 5 and $\mathbf { H }$ in Equation 8 cost $\mathcal { O } ( K N d )$ and $\mathcal { O } ( E d )$ in pre-processing. During distillation, the complexity of $\mathcal { L } _ { e } , \mathcal { L } _ { d }$ and $\scriptstyle { \mathcal { L } } _ { o }$ are $\mathcal { O } ( K N ^ { \prime } d + K d ^ { 2 } )$ , $\mathcal { O } ( K N ^ { \prime } d ^ { \prime } + C d ^ { 2 } )$ , and $\mathcal { O } ( K N ^ { \prime 2 } )$ , respectively.

Relation to Message Passing. Message-passing (MP) is the most popular paradigm for GNNs. Although GDEM does not explicitly perform message-passing during distillation, eigenbasis matching already encodes the information of neighbors as most MP operators rely on the combination of the out product of eigenvectors, e.g., $\begin{array} { r } { \mathbf { L } = \sum _ { i = 1 } ^ { N } \lambda _ { i } \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } } \end{array}$ Therefore, GDEM not only inherits the expressive power of MP but also addresses the weaknesses of the previous distillation methods.

Limitations. Hereby we discuss the limitations of GDEM. (1) The decomposition of the real graph introduces additional computational costs for distillation. (2) In scenarios with extremely high compression rates, the synthetic graphs can only match a limited number of real eigenbasis, resulting in performance degradation.

# 5. Theoretical Analysis

In this section, we give a theoretical analysis of GDEM and prove that it preserves the restricted spectral similarity.

Definition 5.1. (Spectral Similarity (Spielman & Srivastava, 2011)) Let A, $\mathbf { B } \in \mathbb { R } ^ { N \times N }$ be two square matrices. Matrix $\mathbf { B }$ is considered a spectral approximation of A if there exists a positive constant $\epsilon$ , such that for any vector $\mathbf { x } \in \mathbb { R } ^ { N }$ , the following inequality holds:

$$
( 1 - \epsilon ) \mathbf { x } ^ { \top } \mathbf { A } \mathbf { x } < \mathbf { x } ^ { \top } \mathbf { B } \mathbf { x } < ( 1 + \epsilon ) \mathbf { x } ^ { \top } \mathbf { A } \mathbf { x } .
$$

However, it is impossible to satisfy this condition for all $\mathbf { x } \in \mathbb { R } ^ { N }$ (Loukas, 2019). Therefore, we only consider a restricted version of spectral similarity in the feature space.

Definition 5.2. (Restricted Spectral Similarity, RSS 1) The synthetic graph Laplacian $\mathbf { L } ^ { \prime }$ preserves RSS of the real graph Laplacian $\mathbf { L }$ , if there exists an $\epsilon > 0$ such that:

$$
\begin{array} { r } { ( 1 { - } \epsilon ) \mathbf { x } ^ { \top } \mathbf { L } \mathbf { x } < { \mathbf { x } ^ { \prime } } ^ { \top } \mathbf { L } ^ { \prime } \mathbf { x } ^ { \prime } < ( 1 { + } \epsilon ) \mathbf { x } ^ { \top } \mathbf { L } \mathbf { x } \quad \forall \mathbf { x } , \mathbf { x } ^ { \prime } \in \mathbf { X } , \mathbf { X } ^ { \prime } . } \end{array}
$$

Proposition 5.3. The synthetic graph distilled by GDEM is a restricted $\epsilon$ -spectral approximation of the real graph.

Proof. We first characterize the spectral similarity of node features in the real and synthetic graphs, respectively. Notably, here we use the principal $K$ eigenvalues and eigenvectors as a truncated representation of the real graph

$$
\begin{array} { c } { { \displaystyle { \bf x } ^ { \top } { \bf L x } = { \bf x } ^ { \top } \sum _ { k = 1 } ^ { N } \lambda _ { k } { \bf u } _ { k } { \bf u } _ { k } ^ { \top } { \bf x } \approx \sum _ { k = 1 } ^ { K } \lambda _ { k } { \bf x } ^ { \top } { \bf u } _ { k } { \bf u } _ { k } ^ { \top } { \bf x } , } } \\ { { \displaystyle { \bf x } ^ { \prime } { } ^ { \top } { \bf L } ^ { \prime } { \bf x } ^ { \prime } = { \bf x } ^ { \prime } { } ^ { \top } \langle \sum _ { k = 1 } ^ { N ^ { \prime } } \lambda _ { k } { \bf u } _ { k } ^ { \prime } { \bf u } _ { k } ^ { \prime } { } ^ { \top } + \tilde { \bf U } { \bf \boldsymbol { \Lambda } } { \tilde { \bf U } } ^ { \top } \rangle { \bf x } ^ { \prime } } } \\ { { \displaystyle ~ \approx \sum _ { k = 1 } ^ { K } \lambda _ { k } { \bf x } ^ { \prime } { } ^ { \top } { \bf u } _ { k } ^ { \prime } { } ^ { \top } { \bf x } ^ { \prime } + \Delta , } } \end{array}
$$

where $\Delta \ = \ \mathbf { x ^ { \prime } } ^ { \top } \tilde { \mathbf { U } } \mathbf { A } \tilde { \mathbf { U } } ^ { \top } \mathbf { x } ^ { \prime }$ and $\tilde { \textbf { U } }$ represents the nonorthogonal terms of the eigenbasis $\mathbf { U } _ { K } ^ { \prime }$ , which means that $\mathbf { U } _ { K } ^ { \prime } + \tilde { \mathbf { U } }$ is strictly orthogonal.

Combining Equations 11 and 12, we have

$$
\begin{array} { r l } & { ~ \left| { \mathbf { x } } ^ { \top } { \mathbf { L } } { \mathbf { x } } - { \mathbf { x } } ^ { \top } { \mathbf { L } } ^ { \prime } { \mathbf { x } } ^ { \prime } \right| } \\ & { \approx \left| \displaystyle \sum _ { k = 1 } ^ { K } \lambda _ { k } { \mathbf { x } } ^ { \top } { \mathbf { u } } _ { k } { \mathbf { u } } _ { k } ^ { \top } { \mathbf { x } } - \displaystyle \sum _ { k = 1 } ^ { K } \lambda _ { k } { \mathbf { x } } ^ { \prime \top } { \mathbf { u } } _ { k } ^ { \prime } { \mathbf { u } } _ { k } ^ { \prime } { ^ { \top } } { \mathbf { x } } ^ { \prime } - \Delta \right| } \\ & { \leq \displaystyle \sum _ { k = 1 } ^ { K } \lambda _ { k } \left| { \mathbf { x } } ^ { \top } { \mathbf { u } } _ { k } { \mathbf { u } } _ { k } ^ { \top } { \mathbf { x } } - { \mathbf { x } } ^ { \prime \top } { \mathbf { u } } _ { k } ^ { \prime } { \mathbf { u } } _ { k } ^ { \prime } { ^ { \top } } { \mathbf { x } } ^ { \prime } \right| + \left| \Delta \right| . } \end{array}
$$

The above inequality shows that the objective of eigenbasis matching is the upper bound of the spectral discrepancy between the real and synthetic graphs. Optimizing

$\mathcal { L } _ { e }$ and $\mathcal { L } _ { o }$ makes the bound tighter and preserves the spectral similarity of the real graph. The synthetic graph is a restricted $\epsilon$ -spectral approximation of the real graph with

$$
\begin{array} { r } { \epsilon = \sum _ { k = 1 } ^ { K } \lambda _ { k } \left| \mathbf { x } ^ { \top } \mathbf { u } _ { k } \mathbf { u } _ { k } ^ { \top } \mathbf { x } - \mathbf { x } ^ { \prime \top } \mathbf { u } _ { k } ^ { \prime } { \mathbf { u } _ { k } ^ { \prime } } ^ { \top } \mathbf { x } ^ { \prime } \right| + | \bar { \Delta } | . } \end{array}
$$

# 6. Experiments

In this section, we conduct experiments on a variety of graph datasets to validate the effectiveness, generalization, and efficiency of the proposed GDEM.

# 6.1. Experimental Setup

Datasets. To evaluate the effectiveness of our GDEM, we select seven representative graph datasets, including five homophilic graphs, i.e., Citeseer, Pubmed (Kipf & Welling, 2017), Ogbn-arxiv (Hu et al., 2020), Filckr (Zeng et al., 2020), and Reddit (Hamilton et al., 2017), and two heterophilic graphs, i.e., Squirrel (Rozemberczki et al., 2021) and Gamers (Lim et al., 2021).

Baselines. We benchmark our model against several competitive baselines, which can be divided into two categories: (1) Traditional graph reduction methods, including three coreset methods, i.e., Random, Herding, and KCenter (Welling, 2009; Sener & Savarese, 2018), and one coarsening method (Loukas, 2019). (2) Graph distillation methods, including two gradient matching methods, i.e., GCOND (Jin et al., 2022b) and SGDD (Yang et al., 2023), and one trajectory matching method, i.e., SFGC (Zheng et al., 2023). See Appendix A.6 for more details.

Evaluation Protocol. To fairly evaluate the quality of synthetic graphs, we perform the following two steps for all methods: (1) Distillation step, where we apply the distillation methods in the training set of the real graphs. (2) Evaluation step, where we train GNNs on the synthetic graph from scratch and then evaluate their performance on the test set of real graphs. In the node classification experiment (Section 6.2), we follow the settings of the original papers (Jin et al., 2022b; Zheng et al., 2023; Yang et al., 2023). In the generalization experiment (Section 6.3), we use six representative GNNs, including three spatial GNNs, i.e., GCN, SGC, and PPNP, and three spectral GNNs, i.e., ChebyNet, BernNet, and GPR-GNN. See Appendix A.7 for more detailed description.

Settings and Hyperparameters. To eliminate randomness, in the distillation step, we run the distillation methods 10 times and yield 10 synthetic graphs. Moreover, we set $K _ { 1 } + K _ { 2 } = N ^ { \prime }$ . To reduce the tuning complexity, we treat $r _ { k } = \{ 0 . 8 , 0 . 8 5 , 0 . 9 , 0 . 9 5 , 1 . 0 \}$ as a hyperparameter and set $K _ { 1 } = r _ { k } N ^ { \prime }$ , $K _ { 2 } = ( 1 - r _ { k } ) N ^ { \prime }$ for eigenbasis matching. In the evaluation step, spatial GNNs have two aggregation layers and the polynomial order of spectral GNNs is set to 10. For more details, see Appendix A.8.

Table 2. Node classification performance of different distillation methods, mean accuracy $( \% ) \pm$ standard deviation. Bold indicates the best performance and underline means the runner-up.   

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Ratio (r)</td><td colspan="4">Traditional Methods</td><td colspan="4">Graph Distillation Methods</td><td rowspan="2">Whole Dataset</td></tr><tr><td>Random (A&#x27;, X′)</td><td>Coarsening (A&#x27;,X′)</td><td>Herding (A&#x27;,X&#x27;)</td><td>K-Center (A&#x27;,X′)</td><td>GCOND (A&#x27;,X′)</td><td>SFGC (X′)</td><td>SGDD (A&#x27;,X′)</td><td>GDEM (U&#x27;′, X′)</td></tr><tr><td rowspan="3">Citeseer</td><td>0.90%</td><td>54.4±4.4</td><td>52.2±0.4</td><td>57.1±1.5</td><td>52.4±2.8</td><td>70.5±1.2</td><td>71.4±0.5</td><td>69.5±0.4</td><td>72.3±0.3</td><td rowspan="3">71.7±0.1</td></tr><tr><td>1.80%</td><td>64.2±1.7</td><td>59.0±0.5</td><td>66.7±1.0</td><td>64.3±1.0</td><td>70.6±0.9</td><td>72.4±0.4</td><td>70.2±0.8</td><td>72.6±0.6</td></tr><tr><td>3.60%</td><td>69.1±0.1</td><td>65.3±0.5</td><td>69.0±0.1</td><td>69.1±0.1</td><td>69.8±1.4</td><td>70.6±0.7</td><td>70.3±1.7</td><td>72.6±0.5</td></tr><tr><td rowspan="3">Pubmed</td><td>0.08%</td><td>69.4±0.2</td><td>18.1±0.1</td><td>76.7±0.7</td><td>64.5±2.7</td><td>76.5±0.2</td><td>76.4±1.2</td><td>77.1±0.5</td><td>77.7±0.7</td><td rowspan="3">79.3±0.2</td></tr><tr><td>0.15%</td><td>73.3±0.7</td><td>28.7±4.1</td><td>76.2±0.5</td><td>69.4±0.7</td><td>77.1±0.5</td><td>77.5±0.4</td><td>78.0±0.3</td><td>78.4±1.8</td></tr><tr><td>0.30%</td><td>77.8±0.3</td><td>42.8±4.1</td><td>78.0±0.5</td><td>78.2±0.4</td><td>77.9±0.4</td><td>77.9±0.3</td><td>77.5±0.5</td><td>78.2±0.8</td></tr><tr><td rowspan="3">Ogbn-arxiv</td><td>0.05%</td><td>47.1±3.9</td><td>35.4±0.3</td><td>52.4±1.8</td><td>47.2±3.0</td><td>59.2±1.1</td><td>65.5±0.7</td><td>60.8±1.3</td><td>63.7±0.8</td><td rowspan="3">71.4±0.1</td></tr><tr><td>0.25%</td><td>57.3±1.1</td><td>43.5±0.2</td><td>58.6±1.2</td><td>56.8±0.8</td><td>63.2±0.3</td><td>66.1±0.4</td><td>65.8±1.2</td><td>63.8±0.6</td></tr><tr><td>0.50%</td><td>60.0±0.9</td><td>50.4±0.1</td><td>60.4±0.8</td><td>60.3±0.4</td><td>64.0±0.4</td><td>66.8±0.4</td><td>66.3±0.7</td><td>64.1±0.3</td></tr><tr><td rowspan="3">Flickr</td><td>0.10%</td><td>41.8±2.0</td><td>41.9±0.2</td><td>42.5±1.8</td><td>42.0±0.7</td><td>46.5±0.4</td><td>46.6±0.2</td><td>46.9±0.1</td><td>49.9±0.8</td><td></td></tr><tr><td>0.50%</td><td>44.0±0.4</td><td>44.5±0.1</td><td>43.9±0.9</td><td>43.2±0.1</td><td>47.1±0.1</td><td>47.0±0.1</td><td>47.1±0.3</td><td>49.4±1.3</td><td>47.2±0.1</td></tr><tr><td>1.00%</td><td>44.6±0.2</td><td>44.6±0.1</td><td>44.4±0.6</td><td>44.1±0.4</td><td>47.1±0.1</td><td>47.1±0.1</td><td>47.1±0.1</td><td>49.9±0.6</td><td></td></tr><tr><td rowspan="3">Reddit</td><td>0.05%</td><td>46.1±4.4</td><td>40.9±0.5</td><td>53.1±2.5</td><td>46.6±2.3</td><td>88.0±1.8</td><td>89.7±0.2</td><td>91.8±1.9</td><td>92.9±0.3</td><td></td></tr><tr><td>0.10%</td><td>58.0±2.2</td><td>42.8±0.8</td><td>62.7±1.0</td><td>53.0±3.3</td><td>89.6±0.7</td><td>90.0±0.3</td><td>91.0±1.6</td><td>93.1±0.2</td><td>93.9±0.0</td></tr><tr><td>0.50%</td><td>66.3±1.9</td><td>47.4±0.9</td><td>71.0±1.6</td><td>58.5±2.1</td><td>90.1±0.5</td><td>89.9±0.4</td><td>91.6±1.8</td><td>93.2±0.4</td><td></td></tr><tr><td rowspan="3">Squirrel</td><td>0.60%</td><td>22.4±1.6</td><td>20.9±1.1</td><td>21.3±1.1</td><td>21.8±0.3</td><td>27.0±1.3</td><td>24.0±0.4</td><td>24.1±2.3</td><td>28.4±2.0</td><td></td></tr><tr><td>1.20%</td><td>25.0±0.2</td><td>21.1±0.4</td><td>21.4±2.1</td><td>22.8±0.9</td><td>25.7±2.3</td><td>26.9±2.5</td><td>24.7±2.5</td><td>28.2±2.4</td><td>33.0±0.4</td></tr><tr><td>2.50%</td><td>26.9±1.4</td><td>21.5±0.3</td><td>22.4±1.6</td><td>22.9±1.7</td><td>25.3±0.8</td><td>26.1±0.8</td><td>25.8±1.8</td><td>27.8±1.6</td><td></td></tr><tr><td rowspan="3">Gamers</td><td>0.05%</td><td>56.6±1.8</td><td>56.1±0.1</td><td>56.7±1.7</td><td>52.5±4.2</td><td>58.5±1.5</td><td>58.2±1.1</td><td>57.5±1.8</td><td>59.3±1.9</td><td></td></tr><tr><td>0.25%</td><td>60.5±1.0</td><td>56.9±3.0</td><td>57.5±2.0</td><td>57.2±2.3</td><td>58.9±1.8</td><td>58.8±0.5</td><td>57.7±1.0</td><td>60.8±0.4</td><td>62.6±0.0</td></tr><tr><td>0.50%</td><td>60.0±0.5</td><td>57.1±0.4</td><td>58.6±1.3</td><td>57.8±1.7</td><td>58.5±1.9</td><td>59.9±0.3</td><td>58.4±1.7</td><td>61.2±0.3</td><td></td></tr></table>

# 6.2. Node Classification

The node classification performance is reported in Table 2, in which we have the following observations:

First, the GD methods consistently outperform the traditional methods, including coreset and coarsening. The reasons are two-fold: On the one hand, GD methods can leverage the powerful representation learning ability of GNNs to synthesize the graph data. On the other hand, the distillation process involves the downstream task information. In contrast, the traditional methods can only leverage the structural information.

Second, GDEM achieves state-of-the-art performance in 6 out of 7 graph datasets, demonstrating its effectiveness in preserving the distribution of real graphs. Existing GD methods heavily rely on the information of GNNs to distill synthetic graphs. However, the results of GDEM reveal that matching eigenbasis can also yield good synthetic graphs. Furthermore, some results of GDEM are better than those on the entire dataset, which may be due to the use of highfrequency information.

Third, GDEM performs slightly worse on Ogbn-arxiv but achieves promising results on other large-scale graphs. We conjecture this is because, under the compression ratios of

$0 . 0 5 \% - \ 0 . 5 0 \%$ , there are only hundreds of eigenvectors for eigenbasis matching, which is not enough to cover all the useful subspaces in Ogbn-arxiv. See Appendix A.9 for further experimental verification.

# 6.3. Cross-architecture Generalization

We evaluate the generalization ability of the synthetic graphs distilled by four different GD methods, including GCOND, SFGC, SGDD, and GDEM. In particular, each synthetic graph is evaluated by six GNNs, and the average accuracy and variance of the evaluation results are shown in Table 3.

First, GDEM stands out by exhibiting the highest average accuracy across datasets except for Ogbn-arxiv, indicating that the synthetic graphs distilled by GDEM can consistently benefit a variety of GNNs. Moreover, GDEM significantly reduces the performance gap between different GNNs. For example, the variance of GCOND is 2-6 times higher than that of GDEM. On the other hand, SGDD broadcasts the structural information to synthetic graphs and exhibits better generalization ability than GCOND, implying that preserving graph structures can improve the generalization of synthetic graphs. SFGC proposes structure-free distillation. However, this strategy may lead to restricted application scenarios due to the lack of explicit graph structures.

Table 3. Generalization of different distillation methods across GNNs. $\uparrow$ means higher the better and $\downarrow$ means lower the better. Avg., Std. and Impro. indicate average accuracy, standard deviation, and absolute performance improvement.   

<table><tr><td rowspan="2">Dataset (Ratio)</td><td rowspan="2">Methods</td><td colspan="3">Spatial GNNs</td><td colspan="3">Spectral GNNs</td><td rowspan="2">Avg. (↑)</td><td rowspan="2">Std. (↓)</td><td rowspan="2">Impro. (↑)</td></tr><tr><td>GCN</td><td>SGC</td><td>PPNP</td><td>ChebyNet</td><td>BernNet</td><td>GPR-GNN</td></tr><tr><td rowspan="4">Citeseer (r = 1.80%)</td><td>GCOND</td><td>70.5</td><td>70.3</td><td>69.6</td><td>68.3</td><td>63.1</td><td>67.2</td><td>68.17</td><td>2.54</td><td>(+) 4.21</td></tr><tr><td>SFGC</td><td>71.6</td><td>71.8</td><td>70.5</td><td>71.8</td><td>71.1</td><td>71.7</td><td>71.42</td><td>0.47</td><td>(+) 0.96</td></tr><tr><td>SGDD</td><td>70.2</td><td>71.3</td><td>69.2</td><td>70.5</td><td>64.7</td><td>69.7</td><td>69.27</td><td>2.14</td><td>(+) 3.11</td></tr><tr><td>GDEM</td><td>72.6</td><td>72.1</td><td>72.6</td><td>71.4</td><td>72.6</td><td>73.0</td><td>72.38</td><td>0.51</td><td>-</td></tr><tr><td rowspan="4">Pubmed (r = 0.15%)</td><td>GCOND</td><td>77.7</td><td>77.6</td><td>77.3</td><td>76.0</td><td>74.4</td><td>76.5</td><td>76.58</td><td>1.15</td><td>(+) 1.34</td></tr><tr><td>SFGC</td><td>77.5</td><td>77.4</td><td>77.6</td><td>77.3</td><td>76.4</td><td>78.6</td><td>77.47</td><td>0.64</td><td>(+) 0.45</td></tr><tr><td>SGDD</td><td>78.0</td><td>76.6</td><td>78.7</td><td>76.9</td><td>75.5</td><td>77.0</td><td>77.12</td><td>1.02</td><td>(+) 0.80</td></tr><tr><td>GDEM</td><td>78.4</td><td>76.1</td><td>78.1</td><td>78.1</td><td>78.2</td><td>78.6</td><td>77.92</td><td>0.83</td><td>-</td></tr><tr><td rowspan="4">Ogbn-arxiv (r = 0.25%)</td><td>GCOND</td><td>63.2</td><td>63.7</td><td>63.4</td><td>54.9</td><td>55.0</td><td>60.5</td><td>60.12</td><td>3.80</td><td>(+) 2.90</td></tr><tr><td>SFGC</td><td>65.1</td><td>64.8</td><td>63.9</td><td>60.7</td><td>63.8</td><td>64.9</td><td>63.87</td><td>1.50</td><td>(-) 0.85</td></tr><tr><td>SGDD</td><td>65.8</td><td>64.0</td><td>63.6</td><td>56.4</td><td>62.0</td><td>64.0</td><td>62.63</td><td>3.00</td><td>(+) 0.39</td></tr><tr><td>GDEM</td><td>63.8</td><td>62.9</td><td>63.5</td><td>62.4</td><td>61.9</td><td>63.6</td><td>63.02</td><td>0.69</td><td>-</td></tr><tr><td rowspan="4">Flickr (r = 0.50%)</td><td>GCOND</td><td>47.1</td><td>46.1</td><td>45.9</td><td>42.8</td><td>44.3</td><td>46.4</td><td>45.43</td><td>1.45</td><td>(+) 3.90</td></tr><tr><td>SFGC</td><td>47.1</td><td>42.5</td><td>40.7</td><td>45.4</td><td>45.7</td><td>46.4</td><td>44.63</td><td>2.27</td><td>(+) 4.70</td></tr><tr><td>SGDD</td><td>47.1</td><td>46.5</td><td>44.3</td><td>45.3</td><td>46.0</td><td>46.8</td><td>46.00</td><td>0.96</td><td>(+) 3.33</td></tr><tr><td>GDEM</td><td>49.4</td><td>50.3</td><td>49.4</td><td>48.3</td><td>49.6</td><td>49.0</td><td>49.33</td><td>0.60</td><td>-</td></tr><tr><td rowspan="4">Reddit (r = 0.10%)</td><td>GCOND</td><td>89.4</td><td>89.6</td><td>87.8</td><td>75.5</td><td>67.1</td><td>78.8</td><td>81.37</td><td>8.35</td><td>(+) 10.10</td></tr><tr><td>SFGC</td><td>89.7</td><td>89.5</td><td>88.3</td><td>82.8</td><td>87.8</td><td>85.4</td><td>87.25</td><td>2.44</td><td>(+) 4.22</td></tr><tr><td>SGDD</td><td>91.0</td><td>89.4</td><td>89.2</td><td>78.4</td><td>72.4</td><td>81.4</td><td>83.63</td><td>6.80</td><td>(+) 7.84</td></tr><tr><td>GDEM</td><td>93.1</td><td>90.0</td><td>92.6</td><td>90.0</td><td>92.7</td><td>90.4</td><td>91.47</td><td>1.35</td><td>-</td></tr><tr><td rowspan="4">Squirrel (r = 1.20%)</td><td>GCOND</td><td>25.7</td><td>27.2</td><td>23.2</td><td>23.3</td><td>26.0</td><td>26.6</td><td>25.33</td><td>1.55</td><td>(+) 1.89</td></tr><tr><td>SFGC</td><td>26.9</td><td>24.2</td><td>27.2</td><td>25.3</td><td>25.5</td><td>26.6</td><td>25.95</td><td>1.04</td><td>(+) 1.27</td></tr><tr><td>SGDD</td><td>24.7</td><td>27.2</td><td>22.4</td><td>24.5</td><td>24.7</td><td>27.3</td><td>25.13</td><td>1.69</td><td>(+) 2.09</td></tr><tr><td>GDEM</td><td>28.2</td><td>28.0</td><td>25.4</td><td>26.1</td><td>28.2</td><td>27.4</td><td>27.22</td><td>1.09</td><td>-</td></tr><tr><td rowspan="4">Gamers (r = 0.25%)</td><td>GCOND</td><td>58.9</td><td>54.2</td><td>60.1</td><td>60.3</td><td>59.1</td><td>59.3</td><td>58.65</td><td>2.05</td><td>(+) 1.57</td></tr><tr><td>SFGC</td><td>58.8</td><td>55.0</td><td>56.3</td><td>57.2</td><td>57.5</td><td>59.8</td><td>57.43</td><td>1.57</td><td>(+) 2.79</td></tr><tr><td>SGDD</td><td>57.7</td><td>54.6</td><td>56.0</td><td>57.3</td><td>58.8</td><td>58.6</td><td>57.17</td><td>1.47</td><td>(+) 3.05</td></tr><tr><td>GDEM</td><td>60.8</td><td>59.5</td><td>61.0</td><td>59.9</td><td>59.8</td><td>60.3</td><td>60.22</td><td>0.54</td><td>-</td></tr></table>

Table 4. Optimal performance of different methods.   

<table><tr><td>Evaluation</td><td>GCN</td><td>SGC</td><td>PPNP</td><td>Cheb.</td><td>Bern.</td><td>GPR.</td></tr><tr><td>GCOND</td><td>77.7</td><td>77.6</td><td>77.9</td><td>77.3</td><td>78.2</td><td>78.3</td></tr><tr><td>SGDD</td><td>78.0</td><td>76.6</td><td>78.7</td><td>77.5</td><td>78.0</td><td>78.3</td></tr><tr><td>GDEM</td><td>78.4</td><td>76.1</td><td>78.1</td><td>78.1</td><td>78.2</td><td>78.6</td></tr></table>

Table 5. Time overhead (s) of different methods.   

<table><tr><td>Distillation</td><td>GCN</td><td>SGC</td><td>PPNP</td><td>Cheb.</td><td>Bern.</td><td>GPR.</td><td>Overall</td></tr><tr><td>GCOND</td><td>1.99</td><td>1.36</td><td>1.52</td><td>3.89</td><td>56.94</td><td>3.05</td><td>68.75</td></tr><tr><td>SGDD</td><td>2.95</td><td>2.18</td><td>2.33</td><td>4.95</td><td>58.07</td><td>4.28</td><td>74.76</td></tr><tr><td>GDEM</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>1.79</td></tr></table>

# 6.4. Optimal Performance and Time Overhead

We compare the optimal performance and time overhead of different GD methods by traversing various GNN architectures in the Pubmed dataset. Since GDEM does not use GNNs during distillation, we remove the inner- and outer-loop of GCOND and SGDD when calculating the time overhead for a fair comparison. Therefore, the running time is faster than the results in Yang et al. (2023).

In Table 4, we can find that both GCOND and SGDD improve their performance by traversing different GNNs compared to the results in Table 3. However, this strategy also introduces additional computation costs in the distillation stage. As shown in Table 5, the complexity of GCOND and SGDD is related to the complexity of distillation GNNs. Notably, when choosing GNNs with high complexity, e.g., BernNet, their time overhead will increase significantly. On the other hand, GDEM still exhibits remarkable performance compared to the traversal results of GCOND and SGDD. More importantly, the complexity of GDEM will not be affected by GNNs, which eliminates the traversal requirement of previous methods. As a result, the overall time overhead of GDEM is significantly smaller than GCOND and SGDD, which validates the efficiency of GDEM. See Appendix A.2 for more generalization results of GCOND and SGDD.

Table 6. Ablation studies on Pubmed / Gamers.   

<table><tr><td>Pubmed</td><td>GCN (↑)</td><td>GPR. (↑)</td><td>Avg. (↑)</td><td>Var. (↓)</td></tr><tr><td>GDEM</td><td>78.4 / 60.8</td><td>78.6 / 60.3</td><td>77.92 / 60.22</td><td>0.69 / 0.29</td></tr><tr><td>w/o Le</td><td>76.1 / 56.5</td><td>76.9 / 59.8</td><td>76.13 / 58.93</td><td>1.18 / 2.39</td></tr><tr><td>w/o Lo</td><td>77.9 / 59.0</td><td>76.4 / 58.9</td><td>77.07 / 58.85</td><td>2.15 / 2.34</td></tr><tr><td>w/o Ld</td><td>76.7 / 59.9</td><td>77.2 / 60.3</td><td>76.77 / 59.78</td><td>0.21 / 0.13</td></tr></table>

![](images/c89b37747ab825c38f71f07701020ca0ac112efc9286dc0e44a8e05bf1724c2a.jpg)  
Figure 3. Influence of $\mathcal { L } _ { e }$ and $\mathcal { L } _ { d }$ in GDEM.

# 6.5. Ablation Study

We perform ablation studies in the Pubmed and Gamers datasets to verify the effectiveness of different regularization terms, i.e., $\mathcal { L } _ { e }$ , $\mathcal { L } _ { o }$ , and $\mathcal { L } _ { d }$ .

Model Analysis. Table 6 shows the roles of different regularization terms. First, all of them contribute to both the effectiveness and generalization of GDEM. Specifically, $\mathcal { L } _ { e }$ and $\mathcal { L } _ { o }$ primarily govern the generalization ability of GDEM, as the variance of GNNs increases significantly when removing either of them. Second, we observe that $\mathcal { L } _ { d }$ hurts the generalization of GDEM. The reason is that the discrimination constraint uses the information of the graph spectrum and introduces the low-frequency preference. But it also improves the performance of GDEM. Therefore, GDEM needs to carefully balance these two loss functions.

Parameters Analysis. We conduct an additional parameter analysis to further demonstrate the influence of $\mathcal { L } _ { e }$ and $\mathcal { L } _ { d }$ , as illustrated in Figure 3. Specifically, we observe that with the increase in $\alpha$ , the variance of GDEM gradually decreases. However, a higher value of $\alpha$ also leads to performance degeneration. On the other hand, increasing the value of $\beta$ will continue to increase the variance of GDEM but the accuracy decreases when $\beta$ surpasses a specific threshold.

# 6.6. Visualization

We visualize the data distribution of synthetic graphs for a better understanding of our model. Specifically, Figure 4 illustrates the synthetic graphs distilled by GCOND, SGDD, and GDEM, from which we can observe that the value of TV in GDEM is the closest to the real graph. SGDD is closer to the distribution of the real graph than GCOND, implying that SGDD can better preserve the structural information. However, the performance is still not as good as GDEM, which validates the effectiveness of eigenbasis matching.

![](images/903f6a146a93e1afd47b0e5dd2e7dcb85ef3e083bbbb3a4b0a57fef3d7b9fe41.jpg)  
Figure 4. TVs of synthetic graphs distilled by different methods.

![](images/b538109c9655fcd026ffedcfa77379b12b7954ae972245d2451f286ee112754d.jpg)  
Figure 5. TVs of synthetic graphs at different epochs (GDEM).

Besides, we also visualize the synthetic graphs distilled by GDEM at different epochs in Figure 5. We can find that with the optimization of GDEM, the value of TV in the synthetic graphs is approaching the real graph $( 0 . 4 2  0 . 7 3  0 . 8 8 )$ ), which validates Proposition 5.3 that GDEM can preserve the spectral similarity of the real graph.

# 7. Related Work

Graph Neural Networks aim to design effective convolution operators to exploit the node features and topology structure information adequately. GNNs have achieved great success in graph learning and play a vital role in diverse realworld applications (Quan et al., 2023; Yang et al., 2017). Existing methods are roughly divided into spatial and spectral approaches. Spatial GNNs focus on neighbor aggregation strategies in the vertical domain (Kipf & Welling, 2017; Velickovic et al., 2018; Hamilton et al., 2017). Spectral GNNs aim to design filters in the spectral domain to extract certain frequencies for the downstream tasks (Chien et al., 2021; Defferrard et al., 2016; Bo et al., 2023; He et al., 2022; 2021).

Dataset Distillation (DD) has shown great potential in reducing data redundancy and accelerating model training (Sachdeva & McAuley, 2023; Lei & Tao, 2023; Geng et al., 2023; Yu et al., 2023). DD aims to generate small yet informative synthetic training data by matching the model gradient (Zhao et al., 2021; Liu et al., 2022b), data distribution (Zhao & Bilen, 2023; Wang et al., 2022), and training trajectory (Cazenavette et al., 2022; Guo et al., 2023) between the real and synthetic data. As a result, models trained on the real and synthetic data will have comparable performance.

DD has been widely used for graph data, including nodelevel tasks, e.g., GCond (Jin et al., 2022b), SFGC (Zheng et al., 2023), GCDM (Liu et al., 2022a) and MCond (Gao et al., 2023), and graph-level tasks, e.g., DosCond (Jin et al., 2022a) and KIDD $\mathrm { { X u } }$ et al., 2023). GCond is the first GD method based on gradient matching, which needs to optimize GNNs during the distillation procedure, resulting in inefficient computation. DosCond further provides one-step gradient matching to approximate gradient matching, thereby avoiding the bi-level optimization. GCDM proposes distribution matching for GD, which views the receptive fields of a graph as its distribution. Additionally, SFGC proposes structure-free GD to compress the structural information into the node features. KIDD utilizes the kernel ridge regression to further reduce the computational cost. However, all these methods do not consider the influence of GNNs, resulting in spectrum bias and traversal requirement.

# 8. Conclusion

In this paper, we propose eigenbasis matching for graph distillation, which only aligns the eigenbasis and node features of the real and synthetic graphs, thereby alleviating the spectrum bias and traversal requirement of the previous methods. Theoretically, GDEM preserves the restricted spectral similarity of the real graphs. Extensive experiments on both homophilic and heterophilic graphs validate the effectiveness, generalization, and efficiency of the proposed method. A promising future work is to explore eigenbasis matching without the need for explicit eigenvalue decomposition.

# Acknowledgements

This work is supported in part by the National Natural Science Foundation of China (No. U20B2045, 62192784, U22B2038, 62002029, 62172052).

# Impact Statement

This paper presents work aiming to advance the field of efficient graph learning and will save social resources by diminishing computation and storage energy consumption. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

# References

Ahmed, A. R., Bodwin, G., Sahneh, F. D., Hamm, K., Jebelli, M. J. L., Kobourov, S. G., and Spence, R. Graph spanners: A tutorial review. Comput. Sci. Rev., 37:100253, 2020.

Bo, D., Wang, X., Shi, C., and Shen, H. Beyond lowfrequency information in graph convolutional networks.

In AAAI, pp. 3950–3957. AAAI Press, 2021.

Bo, D., Shi, C., Wang, L., and Liao, R. Specformer: Spectral graph neural networks meet transformers. In ICLR, 2023.

Cazenavette, G., Wang, T., Torralba, A., Efros, A. A., and Zhu, J. Dataset distillation by matching training trajectories. In CVPR, pp. 10708–10717. IEEE, 2022.

Chien, E., Peng, J., Li, P., and Milenkovic, O. Adaptive universal generalized pagerank graph neural network. In ICLR. OpenReview.net, 2021.

Defferrard, M., Bresson, X., and Vandergheynst, P. Convolutional neural networks on graphs with fast localized spectral filtering. In NIPS, pp. 3837–3845, 2016.

Gao, X., Chen, T., Zang, Y., Zhang, W., Nguyen, Q. V. H., Zheng, K., and Yin, H. Graph condensation for inductive node representation learning. ArXiv, abs/2307.15967, 2023.

Gao, X., Yu, J., Jiang, W., Chen, T., Zhang, W., and Yin, H. Graph condensation: A survey. ArXiv, abs/2401.11720, 2024.

Geng, J., Chen, Z., Wang, Y., Woisetschlaeger, H., Schimmler, S., Mayer, R., Zhao, Z., and Rong, C. A survey on dataset distillation: Approaches, applications and future directions. In IJCAI, pp. 6610–6618. ijcai.org, 2023.

Guo, Z., Wang, K., Cazenavette, G., Li, H., Zhang, K., and You, Y. Towards lossless dataset distillation via difficulty-aligned trajectory matching. arXiv preprint arXiv:2310.05773, 2023.

Gutman, I. and Zhou, B. Laplacian energy of a graph. Linear Algebra and its applications, 414(1):29–37, 2006.

Hamilton, W. L., Ying, Z., and Leskovec, J. Inductive representation learning on large graphs. In NIPS, pp. 1024–1034, 2017.

He, M., Wei, Z., Huang, Z., and Xu, H. Bernnet: Learning arbitrary graph spectral filters via bernstein approximation. In NeurIPS, pp. 14239–14251, 2021.

He, M., Wei, Z., and Wen, J. Convolutional neural networks on graphs with chebyshev approximation, revisited. In NeurIPS, 2022.

Hu, W., Fey, M., Zitnik, M., Dong, Y., Ren, H., Liu, B., Catasta, M., and Leskovec, J. Open graph benchmark: Datasets for machine learning on graphs. In NeurIPS, 2020.

Jin, W., Tang, X., Jiang, H., Li, Z., Zhang, D., Tang, J., and Yin, B. Condensing graphs via one-step gradient matching. In KDD, pp. 720–730, 2022a.

Jin, W., Zhao, L., Zhang, S., Liu, Y., Tang, J., and Shah, N. Graph condensation for graph neural networks. In ICLR, 2022b.

Jin, Y., Loukas, A., and JaJ´ a, J. F. Graph coarsening with ´ preserved spectral properties. In AISTATS, volume 108, pp. 4452–4462. PMLR, 2020.

Kipf, T. N. and Welling, M. Semi-supervised classification with graph convolutional networks. In ICLR. OpenReview.net, 2017.

Klicpera, J., Bojchevski, A., and Gunnemann, S. Predict ¨ then propagate: Graph neural networks meet personalized pagerank. In ICLR. OpenReview.net, 2019.

Kumar, M., Sharma, A., Saxena, S., and Kumar, S. Featured graph coarsening with similarity guarantees. In ICML, volume 202 of Proceedings of Machine Learning Research, pp. 17953–17975. PMLR, 2023.

Lei, S. and Tao, D. A comprehensive survey of dataset distillation. ArXiv, abs/2301.05603, 2023.

Lim, D., Hohne, F., Li, X., Huang, S. L., Gupta, V., Bhalerao, O., and Lim, S. Large scale learning on nonhomophilous graphs: New benchmarks and strong simple methods. In NeurIPS, pp. 20887–20902, 2021.

Liu, M., Li, S., Chen, X., and Song, L. Graph condensation via receptive field distribution matching. ArXiv, abs/2206.13697, 2022a.

Liu, S., Wang, K., Yang, X., Ye, J., and Wang, X. Dataset distillation via factorization. In NeurIPS, 2022b.

Loukas, A. Graph reduction with spectral and cut guarantees. J. Mach. Learn. Res., 20:116:1–116:42, 2019.

Martinkus, K., Loukas, A., Perraudin, N., and Wattenhofer, R. SPECTRE: spectral conditioning helps to overcome the expressivity limits of one-shot graph generators. In ICML, volume 162, pp. 15159–15179. PMLR, 2022.

Quan, Y., Ding, J., Gao, C., Yi, L., Jin, D., and Li, Y. Robust preference-guided denoising for graph based social recommendation. In WWW, pp. 1097–1108. ACM, 2023.

Rozemberczki, B., Allen, C., and Sarkar, R. Multi-scale attributed node embedding. J. Complex Networks, 9(2), 2021.

Sachdeva, N. and McAuley, J. Data distillation: A survey. ArXiv, abs/2301.04272, 2023.

Sener, O. and Savarese, S. Active learning for convolutional neural networks: A core-set approach. In ICLR. OpenReview.net, 2018.

Spielman, D. A. and Srivastava, N. Graph sparsification by effective resistances. SIAM J. Comput., 40(6):1913–1926, 2011.

Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio,\` P., and Bengio, Y. Graph attention networks. In ICLR, 2018.

Wang, K., Zhao, B., Peng, X., Zhu, Z., Yang, S., Wang, S., Huang, G., Bilen, H., Wang, X., and You, Y. CAFE: learning to condense dataset by aligning features. In CVPR, pp. 12186–12195. IEEE, 2022.

Welling, M. Herding dynamical weights to learn. In ICML, volume 382, pp. 1121–1128. ACM, 2009.

Wu, F., Jr., A. H. S., Zhang, T., Fifty, C., Yu, T., and Weinberger, K. Q. Simplifying graph convolutional networks. In ICML, volume 97, pp. 6861–6871. PMLR, 2019.

Xu, Z., Chen, Y., Pan, M., Chen, H., Das, M., Yang, H., and Tong, H. Kernel ridge regression-based graph dataset distillation. In KDD, pp. 2850–2861, 2023.

Yang, B., Wang, K., Sun, Q., Ji, C., Fu, X., Tang, H., You, Y., and Li, J. Does graph distillation see like vision dataset counterpart? In NeurIPS, 2023.

Yang, C., Sun, M., Zhao, W. X., Liu, Z., and Chang, E. Y. A neural network approach to jointly modeling social networks and mobile trajectories. ACM Transactions on Information Systems (TOIS), 35(4):1–28, 2017.

Yu, R., Liu, S., and Wang, X. Dataset distillation: A comprehensive review. ArXiv, abs/2301.07014, 2023.

Yu, S., Alesiani, F., Yin, W., Jenssen, R., and Pr´ıncipe, J. C. Principle of relevant information for graph sparsification. In UAI, volume 180 of Proceedings of Machine Learning Research, pp. 2331–2341. PMLR, 2022.

Zeng, H., Zhou, H., Srivastava, A., Kannan, R., and Prasanna, V. K. Graphsaint: Graph sampling based inductive learning method. In ICLR. OpenReview.net, 2020.

Zhao, B. and Bilen, H. Dataset condensation with distribution matching. In WACV, pp. 6503–6512. IEEE, 2023.

Zhao, B., Mopuri, K. R., and Bilen, H. Dataset condensation with gradient matching. In ICLR. OpenReview.net, 2021.

Zhao, G., Li, G., Qin, Y., and Yu, Y. Improved distribution matching for dataset condensation. In CVPR, pp. 7856– 7865. IEEE, 2023.

Zheng, X., Zhang, M., Chen, C., Nguyen, Q. V. H., Zhu, X., and Pan, S. Structure-free graph condensation: From large-scale graphs to condensed graph-free data. ArXiv, abs/2306.02664, 2023.

# A. Experimental Details

# A.1. Visualization of Synthetic Graphs

Distillation Details with Low-pass and High-pass Filters. We use GCOND to distill two synthetic graphs on Pubmed by replacing SGC with a low-pass filter $\mathcal { F } _ { L } = \mathbf { A } \mathbf { X } \mathbf { W }$ and a high-pass filter $\mathcal { F } _ { H } = \mathbf { L X W }$ , respectively.

Visualization Details Once we generate the synthetic graphs, we calculate the value of total variation (TV) for each dimension. TV is a widely used metric to represent the distribution, i.e., smoothness, of a signal on the graph:

$$
\mathbf { x } ^ { \top } \mathbf { L } \mathbf { x } = \sum _ { ( i , j ) \in { \mathcal { E } } } ( x _ { i } - x _ { j } ) ^ { 2 } = \sum _ { i = 1 } ^ { n } \lambda _ { i } \mathbf { x } ^ { \top } \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } \mathbf { x } .
$$

Note that the edge number of synthetic graphs and the original graph is different, so we normalize node features and laplacian matrix first:

$$
\begin{array} { r l } & { \hat { \mathbf { x } } _ { i } = \frac { \mathbf { x } _ { i } } { \left\| \mathbf { x } _ { i } \right\| } , } \\ & { \hat { \mathbf { L } } = \mathbf { I } _ { N } - \mathbf { D } ^ { - \frac { 1 } { 2 } } \mathbf { A } \mathbf { D } ^ { - \frac { 1 } { 2 } } , } \end{array}
$$

where $\mathbf { x } _ { i }$ is the $i$ -th dimension node feature. Then we substitute $\hat { \mathbf { x } } _ { i }$ and $\hat { \bf L }$ into Equation 14 calculating the TV of the graph.   
Additionally, we report the average TV of all dimensions as reported in the legend of the visualization figures.

# A.2. Cross-architecture Performance of GCOND and SGDD

To verify the cross-architecture performance of the GCOND and SGDD, we generate six synthetic graphs on Pubmed under a $0 . 1 5 \%$ compression ratio, using six GNNs for the distillation procedure. Then we train these GNNs on the six synthetic graphs and evaluate their performance. Experimental settings are as follows.

Distillation Step. For spatial GNNs, i.e., GCN, SGC, and PPNP, we set the aggregation layers to 2. For GCN, we use 256 hidden units for each convolutional layer. For spectral GNNs, i.e., ChebyNet, BernNet, and GPR-GNN, we set the polynomial order to 10. The linear feature transformation layers of all GNNs are set to 1. For hyper-parameters tuning, we select training epochs from $\{ 4 0 0 , 5 0 0 , 6 0 0 \}$ , learning rates of node feature and topology structure from $\{ 0 . 0 0 0 1 , 0 . 0 0 0 5$ , $0 . 0 0 1 , 0 . 0 0 5 , 0 . 0 5 \}$ , outer loop from $\{ 2 5 , 2 0 , 1 5 , 1 0 \}$ , and inner loop from $\{ 1 5 , 1 0 , 5 , 1 \}$ .

Evaluation Step. For spatial GNNs, we use two aggregation layers. For spectral GNNs, we set the polynomial order to 10. The hidden units of convolutional layers and linear feature transformation layers are both set to 256. We train each GNN for 2000 epochs and select the model parameters with the best performance on validation sets for evaluation.

Table 7. GCOND with various distillation (D) and evaluation (E) GNNs in Pubmed dataset.   

<table><tr><td>D\E</td><td>GCN</td><td>SGC</td><td>PPNP</td><td>Cheb.</td><td>Bern.</td><td>GPR.</td></tr><tr><td>GCN</td><td>74.57</td><td>71.70</td><td>75.53</td><td>70.13</td><td>68.40</td><td>71.73</td></tr><tr><td>SGC</td><td>77.72</td><td>77.60</td><td>77.34</td><td>76.03</td><td>74.42</td><td>76.52</td></tr><tr><td>PPNP</td><td>72.70</td><td>70.40</td><td>77.46</td><td>73.38</td><td>70.56</td><td>74.02</td></tr><tr><td>Cheb.</td><td>73.60</td><td>70.62</td><td>75.10</td><td>77.30</td><td>77.62</td><td>78.10</td></tr><tr><td>Bern.</td><td>67.68</td><td>73.76</td><td>74.30</td><td>77.20</td><td>78.12</td><td>78.28</td></tr><tr><td>GPR.</td><td>76.04</td><td>72.20</td><td>77.94</td><td>75.92</td><td>77.12</td><td>77.96</td></tr><tr><td>Optimal</td><td>77.72</td><td>77.60</td><td>77.94</td><td>77.30</td><td>78.12</td><td>78.28</td></tr></table>

Table 8. SGDD with various distillation (D) and evaluation (E) GNNs in Pubmed dataset.   

<table><tr><td>D\E</td><td>GCN</td><td>SGC</td><td>PPNP</td><td>Cheb.</td><td>Bern.</td><td>GPR.</td></tr><tr><td>GCN</td><td>76.92</td><td>70.10</td><td>74.64</td><td>74.98</td><td>76.66</td><td>75.18</td></tr><tr><td>SGC</td><td>78.04</td><td>76.60</td><td>78.72</td><td>76.90</td><td>75.45</td><td>77.02</td></tr><tr><td>PPNP</td><td>76.44</td><td>74.34</td><td>76.28</td><td>73.70</td><td>74.94</td><td>75.98</td></tr><tr><td>Cheb.</td><td>77.42</td><td>73.66</td><td>75.40</td><td>77.50</td><td>77.96</td><td>77.12</td></tr><tr><td>Bern.</td><td>70.64</td><td>71.22</td><td>74.88</td><td>76.38</td><td>76.16</td><td>77.84</td></tr><tr><td>GPR.</td><td>63.76</td><td>61.24</td><td>76.32</td><td>71.40</td><td>71.70</td><td>78.30</td></tr><tr><td>Optimal</td><td>78.04</td><td>76.60</td><td>78.72</td><td>77.50</td><td>77.96</td><td>78.30</td></tr></table>

# A.3. Implementation Details of GDEM

Predefined Labels $\mathbf { Y } ^ { \prime }$ of Synthetic Graphs. The labels $\mathbf { Y } ^ { \prime }$ are predefined one-hot vectors, indicating the category to which the nodes belong. Specifically, given $N _ { l }$ labeled nodes in the real graph, we set the number of nodes of category $c$ in the synthetic graph as $\begin{array} { r } { \dot { N } _ { c } ^ { \prime } = N _ { c } \times \frac { \mathbf { \tilde { N } ^ { \prime } } } { N _ { l } } } \end{array}$ , where $N _ { c }$ is the number of nodes with label $c$ . The setting will make the label distribution of the synthetic graph consistent with the real graph.

Initialization of Synthetic Graphs. Different from previous GD methods that directly learn the adjacency matrix of the synthetic graph, GDEM aims to generate its eigenbasis. To ensure that the initialized eigenbasis is valid, we first use the stochastic block model (SBM) to randomly generate the adjacency matrix of the synthetic graph $\mathbf { A } ^ { \prime } \in \{ 0 , 1 \} ^ { N ^ { \prime } \times N ^ { \prime } }$ , and then decompose it to produce the top- $K$ eigenvectors as the initialized eigenbasis $\mathbf { U } _ { K } ^ { \prime } \in \mathbb { R } ^ { N ^ { \prime } \times K }$ . Moreover, to initialize the synthetic node features $\mathbf { X } ^ { \prime } \in \mathbb { R } ^ { N ^ { \prime } \times d }$ , we first train an MLP $\rho ( \cdot )$ in the real node features. Then we freeze the well-trained MLP and feed the synthetic node features into it to minimize the classification objective. This process can be formulated as:

$$
\underset { { \bf x } ^ { \prime } } { \operatorname* { m i n } } \sum _ { i = 1 } ^ { n ^ { \prime } } - y _ { i } ^ { \prime } \log \rho ( { \bf x } _ { i } ^ { \prime } , \theta ^ { * } ) , \ \mathrm { s . t . } \ \theta ^ { * } = \underset { \theta } { \arg \operatorname* { m i n } } \sum _ { i = 1 } ^ { n } - y _ { i } \log \rho ( { \bf x } _ { i } , \theta )
$$

where $\theta$ indicates the parameters of MLP.

# A.4. Complexity of Different Methods

We analyze the complexity of different methods and give the final complexity in Table 9. We use $E$ to present the number of edges. For simplicity, we use $d$ to denote both feature dimension and hidden units of GNNs. $t$ is the number of GNN layers and $r$ is the number of sampled neighbors per node. $\theta _ { t }$ denotes the model parameters of the GNNs. For SFGC, $M$ is the number of training trajectories and $S$ is the length of each trajectory.

# Complexity of GDEM.

(1) Pre-processing: The complexity of decomposition is $\mathcal { O } ( K N ^ { 2 } )$ . It’s noteworthy that the decomposition is performed once per graph and can be repeatedly used for subsequent training, inference, and hyperparameter tuning. Therefore, the time overhead of decomposition should be amortized by the entire experiment rather than simply summarized them. Additionally, we pre-process $\mathbf { u } _ { k } ^ { \top } \mathbf { X }$ in Equation 5 and $H$ in Equation 8, which cost $\mathcal { O } ( K N d )$ and $\mathcal { O } ( E d )$ .

(1) Complexity of $\mathcal { L } _ { e }$ : $\mathcal { O } ( K N ^ { \prime } d + K d ^ { 2 } )$ .   
(2) Complexity of $\mathcal { L } _ { d }$ : The complexity of calculating $H ^ { \prime }$ is $\mathcal { O } ( K N ^ { \prime } d ^ { \prime } )$ . The calculation of cosine similarity costs $\mathcal { O } ( C d ^ { 2 } )$ . (3) Complexity of $\mathcal { L } _ { o }$ : $\mathcal { O } ( K N ^ { \prime 2 } )$ .   
The final complexity can be simplified as $\mathcal { O } ( K N ^ { 2 } + K N d + E d ) + \mathcal { O } ( K N ^ { \prime 2 } + K N ^ { \prime } d + ( K + C ) d ^ { 2 } ) .$

# Complexity of GCOND.

(1) Pre-processing: GCOND doesn’t need special pre-processing.   
(2) Inference for $A ^ { \prime }$ : $\mathcal { O } ( N ^ { \prime 2 } d ^ { 2 } )$ .   
(3) Forward process of SGC on the original graph: $\mathcal { O } ( r ^ { t } N d ^ { 2 } )$ . That on the synthetic graph: $\mathcal { O } ( t N ^ { \prime 2 } d + t N ^ { \prime } d )$ .   
(4) Calculation of second-order derivatives in backward propagation: $\mathcal { O } ( | \theta _ { t } | + | A ^ { \prime } | + | X ^ { \prime } | )$ .   
The final complexity can be simplified as $\mathcal { O } ( r ^ { t } N d ^ { 2 } ) + \mathcal { O } ( N ^ { \prime 2 } d ^ { 2 } )$ .

# Complexity of SGDD.

(1) Pre-processing: SGDD doesn’t need special pre-processing.   
(2) Inference for $A ^ { \prime }$ : $\mathcal { O } ( N ^ { \prime 2 } d ^ { 2 } )$ .   
(3) Forward process of SGC on the original graph: $\mathcal { O } ( r ^ { t } N d ^ { 2 } )$ . That on the synthetic graph: $\mathcal { O } ( t N ^ { \prime 2 } d + t N ^ { \prime } d )$ .   
(4) Calculation of second-order derivatives in backward propagation: $\mathcal { O } ( | \theta _ { t } | + | A ^ { \prime } | + | X ^ { \prime } | )$ .   
(5) Structure optimization term: $\mathcal { O } ( N ^ { \prime 2 } k + N N ^ { \prime 2 } )$ .   
The final complexity can be simplified as $\mathcal { O } ( r ^ { t } N d ^ { 2 } ) + \mathcal { O } ( N ^ { \prime 2 } N )$ .

# Complexity of SFGC.

(1) Pre-processing: $\mathcal { O } ( M S ( t E d + t N d ^ { 2 } ) )$ . Note that $M S$ is usually very large, so it cannot be omitted.   
(2) Forward process of GCN on the synthetic graph: $\mathcal { O } ( t N ^ { \prime } d ^ { 2 } + t N ^ { \prime } d )$ . Note that SFGC pre-trains the trajectories on GCN, so there is no need to calculate the forward process on the original graph.   
(3) Backward propagation: SFGC uses a MTT(Cazenavette et al., 2022) method, which results in bi-level optimization(Yu et al., 2023) for the backward.   
The final complexity can be simplified as $\mathcal { O } ( M S ( t E d + t N d ^ { 2 } ) ) + \mathcal { O } ( t N ^ { \prime } d ^ { 2 } ) .$

Table 9. Complexity of different distillation methods.   

<table><tr><td>Method</td><td>Pre-processing</td><td>Training</td></tr><tr><td>GCOND</td><td></td><td>O(rLNd2) + O(N ′2d2)</td></tr><tr><td>SGDD</td><td></td><td>O(rLN d2) + O(N ′2N )</td></tr><tr><td>SFGC</td><td>O(MS(LEd + LNd2))</td><td>O(LN′d2)</td></tr><tr><td>GDEM</td><td>O(KN 2 + KNd + Ed)</td><td>O(KN ′2 + KN′d + (K + C)d2)</td></tr></table>

# A.5. Statistics of Datasets

In the experiments, we use seven graph datasets to validate the effectiveness of GDEM. For homophilic graphs, we use the public data splits. For heterophilic graphs, we use the splitting with training/validation/test sets accounting for $2 . 5 \% / 2 . 5 / \% 9 5 \%$ on Squirrel, and $5 0 \% / 2 5 \% 2 5 \%$ on Gamers. The detailed statistical information of each dataset is shown in Table 10.

Table 10. Statistics of datasets.   

<table><tr><td>Dataset</td><td>Nodes</td><td>Edges</td><td>Classes</td><td>Features</td><td>Training/Validation/Test</td><td>Edge hom.</td><td>LCC</td></tr><tr><td>Citeseer</td><td>3,327</td><td>4,732</td><td>6</td><td>3,703</td><td>120/500/1000</td><td>0.74</td><td>2,120</td></tr><tr><td>Pubmed</td><td>19,717</td><td>44,338</td><td>3</td><td>500</td><td>60/500/1,000</td><td>0.80</td><td>19,717</td></tr><tr><td>Ogbn-arxiv</td><td>169,343</td><td>1,166,243</td><td>40</td><td>128</td><td>90,941/29,799/48,603</td><td>0.66</td><td>169,343</td></tr><tr><td>Flickr</td><td>89,250</td><td>899,756</td><td>7</td><td>500</td><td>44,625/22,312/22,313</td><td>0.33</td><td>89,250</td></tr><tr><td>Reddit</td><td>232,965</td><td>57,307,946</td><td>41</td><td>602</td><td>153,932/23,699/55,334</td><td>0.78</td><td>231,371</td></tr><tr><td>Squirrel</td><td>5,201</td><td>396,846</td><td>5</td><td>2,089</td><td>130/130/4,941</td><td>0.22</td><td>5,201</td></tr><tr><td>Gamers</td><td>168,114</td><td>13,595,114</td><td>2</td><td>7</td><td>84,056/42,028/42,030</td><td>0.55</td><td>168,114</td></tr></table>

# A.6. Baselines

For a fair comparison of performance, we adopt the results of baselines reported in their papers, which are evaluated through meticulous experimental design and careful hyperparameter tuning. The experimental details are as follows: (1) GCOND employs a 2-layer SGC for distillation and a 2-layer GCN with 256 hidden units for evaluation.

(2) SGDD employs a 2-layer SGC for distillation and a 2-layer GCN with 256 hidden units for evaluatio (3) SFGC employs 2-layer GCNs with 256 hidden units both for distillation and evaluation.

# A.7. Evaluation Details

Performance Evaluation. For comparison with baselines, we report the performance of GDEM evaluated with a 2-layer GCN with 256 hidden units. Specifically, we generate 10 synthetic graphs with different seeds on the original graph. Then we train the GCN using these 10 synthetic graphs and report the average results of the best performance evaluated on test sets of the original graph.

Generalization Evaluation. For generalization evaluation, we train 6 GNNs using the synthetic graphs generated by different distillation methods. For SGC, GCN, and APPNP, we use 2-layer aggregations. For ChebyNet, we set the convolution layers to 2 with propagation steps from $\{ 2 , 3 , 5 \}$ . For BernNet and GPRGNN, we set the polynomial order to 10. The hidden units of both convolution layers and linear feature transformation are 256.

# A.8. Hyperparamters

Hyperparameter details are listed in Table 11. $\tau _ { 1 }$ and $\tau _ { 2 }$ are steps for alternating updates of node features and eigenvectors. $\alpha , \beta$ , and $\gamma$ denote the weights in Equation 9. lr feat and lr eigenvecs are the learning rates of node features and eigenvectors, respectively.

Table 11. Hyper-parameters of GDEM.   

<table><tr><td>Dataset</td><td>Ratio</td><td>epochs</td><td>K1</td><td>K2</td><td>τ1</td><td>τ2</td><td>α</td><td>β</td><td>γ</td><td>lr_feat</td><td>lr_eigenvecs</td></tr><tr><td rowspan="3">Citeseer</td><td>0.90%</td><td>500</td><td>30</td><td>0</td><td>5</td><td>1</td><td>1.0</td><td>1e-05</td><td>1.0</td><td>0.0001</td><td>0.01</td></tr><tr><td>1.80%</td><td>1500</td><td>48</td><td>12</td><td>10</td><td>15</td><td>0.05</td><td>1e-05</td><td>0.5</td><td>0.0005</td><td>0.0005</td></tr><tr><td>3.60%</td><td>500</td><td>114</td><td>6</td><td>1</td><td>10</td><td>0.01</td><td>1e-06</td><td>0.1</td><td>0.001</td><td>0.0001</td></tr><tr><td rowspan="3">Pubmed</td><td>0.08%</td><td>1000</td><td>15</td><td>0</td><td>15</td><td>5</td><td>0.0001</td><td>1e-07</td><td>0.01</td><td>0.0001</td><td>0.0005</td></tr><tr><td>0.15%</td><td>1500</td><td>30</td><td>0</td><td>5</td><td>5</td><td>1.0</td><td>1e-05</td><td>0.01</td><td>0.0005</td><td>0.01</td></tr><tr><td>0.30%</td><td>1500</td><td>57</td><td>3</td><td>20</td><td>1</td><td>0.01</td><td>1e-07</td><td>0.5</td><td>0.001</td><td>0.0001</td></tr><tr><td rowspan="3">Ogbn-arxiv</td><td>0.05%</td><td>500</td><td>86</td><td>4</td><td>1</td><td>5</td><td>0.0001</td><td>1e-02</td><td>0.01</td><td>0.0005</td><td>0.0005</td></tr><tr><td>0.25%</td><td>2000</td><td>409</td><td>45</td><td>10</td><td>5</td><td>0.01</td><td>1e-04</td><td>0.01</td><td>0.0001</td><td>0.0001</td></tr><tr><td>0.50%</td><td>1000</td><td>773</td><td>136</td><td>1</td><td>5</td><td>0.001</td><td>1e-04</td><td>1.0</td><td>0.0001</td><td>0.005</td></tr><tr><td rowspan="3">Flickr</td><td>0.10%</td><td>2000</td><td>44</td><td>0</td><td>5</td><td>10</td><td>0.01</td><td>1e-07</td><td>0.05</td><td>0.0001</td><td>0.05</td></tr><tr><td>0.50%</td><td>2000</td><td>223</td><td>0</td><td>5</td><td>10</td><td>0.01</td><td>1e-07</td><td>0.05</td><td>0.0001</td><td>0.05</td></tr><tr><td>1.00%</td><td>2000</td><td>446</td><td>0</td><td>5</td><td>10</td><td>0.01</td><td>1e-07</td><td>0.05</td><td>0.0001</td><td>0.05</td></tr><tr><td rowspan="3">Reddit</td><td>0.05%</td><td>1000</td><td>76</td><td>0</td><td>20</td><td>5</td><td>1.0</td><td>1e-06</td><td>0.01</td><td>0.0001</td><td>0.0001</td></tr><tr><td>0.10%</td><td>500</td><td>153</td><td>0</td><td>15</td><td>10</td><td>0.5</td><td>1e-06</td><td>05</td><td>0.0005</td><td>0.005</td></tr><tr><td>0.50%</td><td>1000</td><td>693</td><td>76</td><td>5</td><td>5</td><td>1.0</td><td>1e-06</td><td>0.5</td><td>0.0005</td><td>0.0001</td></tr><tr><td rowspan="3">Squirrel</td><td>0.60%</td><td>1000</td><td>31</td><td>1</td><td>5</td><td>1</td><td>1.0</td><td>1e-07</td><td>0.01</td><td>0.0001</td><td>0.005</td></tr><tr><td>1.20%</td><td>500</td><td>62</td><td>3</td><td>10</td><td>5</td><td>1.0</td><td>1e-07</td><td>0.01</td><td>0.0001</td><td>0.0001</td></tr><tr><td>2.05%</td><td>2000</td><td>104</td><td>26</td><td>5</td><td>1</td><td>0.0001</td><td>1e-05</td><td>0.05</td><td>0.0001</td><td>0.01</td></tr><tr><td rowspan="3">Gamers</td><td>0.05%</td><td>2000</td><td>80</td><td>4</td><td>15</td><td>1</td><td>0.0001</td><td>1e-07</td><td>0.05</td><td>0.0001</td><td>0.01</td></tr><tr><td>0.25%</td><td>2000</td><td>420</td><td>0</td><td>20</td><td>20</td><td>0.0001</td><td>1e-07</td><td>0.05</td><td>0.0001</td><td>0.005</td></tr><tr><td>0.50%</td><td>500</td><td>756</td><td>84</td><td>15</td><td>1</td><td>0.0001</td><td>1e-07</td><td>0.05</td><td>0.0001</td><td>0.0001</td></tr></table>

Table 12. The node classification performance of Ogbn-arxiv and Reddit on various truncated graph structures.   

<table><tr><td>Dataset</td><td>K = 500</td><td>K = 1000</td><td>K = 3000</td><td>K = 5000</td><td>Full Graph</td></tr><tr><td>Reddit</td><td>92.41±0.49</td><td>93.45±0.48</td><td>93.94±0.41</td><td>94.07±0.37</td><td>94.51±0.24</td></tr><tr><td>Ogbn-arxiv</td><td>61.87±0.89</td><td>64.65±1.20</td><td>67.32±1.11</td><td>69.22±0.93</td><td>70.02±1.19</td></tr></table>

# A.9. Analysis of the Worse Performance on Obgn-arxiv

To investigate the reason why GDEM performs slightly worse on Obgn-arxiv but achieves promising results on other large-scale graphs, we evaluate the number of useful eigenbasis in both Ogbn-arxiv and Reddit. Specifically, we first truncate the graph structures of Ogbn-arxiv and Reddit by:

$$
\mathbf { A } ^ { \prime } = \sum _ { k = 1 } ^ { K _ { 1 } } \lambda _ { k } \mathbf { u } _ { k } \mathbf { u } _ { k } ^ { \top } + \sum _ { k = N - K _ { 2 } + 1 } ^ { N } \lambda _ { k } \mathbf { u } _ { k } \mathbf { u } _ { k } ^ { \top }
$$

where $K _ { 1 } = r _ { k } K$ and $K _ { 2 } = ( 1 - r _ { k } ) K$ . We then gradually increase the value of $K$ and train a 2-layer SGC on each runcated graph structure. The results are shown in Table 12.

We can observe that in Reddit, only 1,000 eigenvectors are enough to match the performance of the full graph $_ { ( 9 3 . 4 5 \mathrm { ~ / ~ } }$ $9 4 . 5 1 \approx 9 8 . 9 \%$ , while in Ogbn-arxiv, a large number of eigenvectors (5,000) is required to approximate the full graph $( 6 9 . 2 2 / 7 0 . 0 2 \approx 9 8 . 9 \% )$ . Thus, we speculate that the structure information of Ogbn-arxiv is more widely distributed in the eigenbasis, making it challenging for GDEM to compress the entire distribution in synthetic data with an extremely small compression rate.

# B. Theoretical Analysis of RSS for Gradient Matching

We further theoretically analyze whether the gradient matching method can preserve the restricted spectral similarity. Given $x ^ { \prime }$ and $L ^ { \prime }$ learned by the gradient matching method, we have:

$$
\begin{array} { r l } & { | { \mathbf { x } } ^ { \top } { \mathbf { L } } { \mathbf { x } } - { \mathbf { x } } ^ { \prime ^ { \top } } { \mathbf { L } } ^ { \prime } { \mathbf { x } } ^ { \prime } | } \\ & { = | \displaystyle \sum _ { k = 0 } ^ { K } \lambda _ { k } { \mathbf { x } } ^ { \top } { \mathbf { u } } _ { k } { \mathbf { u } } _ { k } ^ { \top } { \mathbf { x } } - \displaystyle \sum _ { k = 0 } ^ { K } \lambda _ { k } ^ { \prime } { \mathbf { x } } ^ { \prime ^ { \top } } { \mathbf { u } } _ { k } ^ { \prime } { \mathbf { u } } _ { k } ^ { \prime ^ { \top } } { \mathbf { x } } ^ { \prime } | } \\ & { = \displaystyle | ( \displaystyle \sum _ { k = 0 } ^ { K } \lambda _ { k } { \mathbf { x } } ^ { \top } { \mathbf { u } } _ { k } { \mathbf { u } } _ { k } ^ { \top } { \mathbf { x } } - \displaystyle \sum _ { k = 0 } ^ { K } \lambda _ { k } { \mathbf { x } } ^ { \prime ^ { \top } } { \mathbf { u } } _ { k } ^ { \prime } { \mathbf { u } } _ { k } ^ { \prime ^ { \top } } { \mathbf { x } } ^ { \prime } ) + ( \displaystyle \sum _ { k = 0 } ^ { K } \lambda _ { k } { \mathbf { x } } ^ { \prime ^ { \top } } { \mathbf { u } } _ { k } ^ { \prime } { \mathbf { u } } _ { k } ^ { \prime ^ { \top } } { \mathbf { x } } ^ { \prime } - \displaystyle \sum _ { k = 0 } ^ { K } \lambda _ { k } ^ { \prime } { \mathbf { x } } ^ { \prime ^ { \top } } { \mathbf { u } } _ { k } ^ { \prime } { \mathbf { u } } _ { k } ^ { \prime ^ { \top } } { \mathbf { x } } ^ { \prime } ) | } \\ &  \leqslant \displaystyle \sum _ { k = 0 } ^ { K } \lambda _ { k } | { \mathbf { x } } ^ { \top } { \mathbf { u } } _ { k }  \mathbf { u }  \end{array}
$$

Combining with Lemma 3.1, when the number of GCN layers goes to infinity, the objective optimization based on gradient matching is dominated by $\left| \mathbf { x } ^ { \top } \mathbf { u } _ { 0 } \mathbf { u } _ { 0 } ^ { \top } \mathbf { x } - \mathbf { x } ^ { \prime } ^ { \top } \mathbf { u } _ { 0 } ^ { \prime } \mathbf { u } _ { 0 } ^ { \prime } ^ { \top } \mathbf { x } ^ { \prime } \right| ,$ , while paying less attention to the optimization of $\left| { \mathbf { x } } ^ { \top } { \mathbf u } _ { k } { \mathbf u } _ { k } ^ { \top } { \mathbf x } - { \mathbf { x } } ^ { \prime } ^ { \top } { \mathbf u } _ { k } ^ { \prime } { \mathbf u } _ { k } ^ { \prime } ^ { \top } { \mathbf x } ^ { \prime } \right| .$ , when $k \neq 0$ . Thus, gradient matching fails to constrain the first term of the upper bound of RSS. Moreover, gradient matching introduces spectrum bias causing $\lambda _ { k } ^ { \prime } \neq \lambda _ { k }$ , thus failing to constrain the second term of the upper bound. In summary, the gradient matching method is unable to preserve the restricted spectral similarity.

# C. Graph Distiilation

Gradient Matching (Jin et al., 2022b;a) generates the synthetic graph and node features by minimizing the differences between model gradients on $\mathcal { G }$ and $\mathcal { G } ^ { \prime }$ , which can be formulated as:

$$
\operatorname* { m i n } _ { \mathbf { A } ^ { \prime } , \mathbf { X } ^ { \prime } } \underset { \theta \sim P _ { \theta } } { \mathbb { E } } \left[ D \left( \nabla _ { \theta } \mathcal { L } \left( \Phi _ { \theta } \left( \mathbf { A } ^ { \prime } , \mathbf { X } ^ { \prime } \right) , \mathbf { Y } ^ { \prime } \right) , \nabla _ { \theta } \mathcal { L } \left( \Phi _ { \theta } \left( \mathbf { A } , \mathbf { X } \right) , \mathbf { Y } \right) \right) \right] ,
$$

where $\Phi _ { \theta }$ is the condensation GNNs with parameters $\theta$ , $\nabla _ { \theta }$ indicates the model gradients, $D$ is a metric to measure their differences, and $\mathcal { L }$ is the loss function. For clarity, we omit the subscript that indicates the training data.

Distribution Matching (Liu et al., 2022a) aims to align the distributions of node representations in each GNN layer to generate the synthetic graph, which can be expressed as:

$$
\operatorname* { m i n } _ { \mathbf { A } ^ { \prime } , \mathbf { X } ^ { \prime } } \underset { \theta \sim P _ { \theta } } { \mathbb { E } } \left[ \sum _ { t = 1 } ^ { L } D \left( \Phi _ { \theta } ^ { t } \left( \mathbf { A } ^ { \prime } , \mathbf { X } ^ { \prime } \right) , \Phi _ { \theta } ^ { t } \left( \mathbf { A } , \mathbf { X } \right) \right) \right] ,
$$

where $\Phi _ { \theta } ^ { t }$ is the $t$ -th layer in GNNs.

Trajectory Matching (Zheng et al., 2023) aligns the long-term GNN learning behaviors between the original graph and the synthetic graph:

$$
\operatorname* { m i n } _ { \mathbf { A } ^ { \prime } , \mathbf { X } ^ { \prime } } \underset { \theta _ { t } ^ { * , i } \sim P _ { \Theta } \mathcal { T } } { \mathbb { E } } \left[ \mathcal { L } _ { \mathrm { m e t a - t t } } \left( \theta _ { t } ^ { * } | _ { t = t _ { 0 } } ^ { p } , \tilde { \theta } _ { t } | _ { t = t _ { 0 } } ^ { q } \right) \right] .
$$

where $\theta _ { t } ^ { * } | _ { t = t _ { 0 } } ^ { p }$ and $\tilde { \theta } _ { t } | _ { t = t _ { 0 } } ^ { q }$ is the parameters of $\mathrm { G N N } _ { \mathcal { T } }$ and $\mathrm { G N N } _ { \mathcal { S } }$ , ${ \mathcal { L } } _ { \mathrm { m e t a - t t } }$ calculates certain parameter training intervals within $\left[ \theta _ { t _ { 0 } } ^ { * , i } , \theta _ { t _ { 0 } + p } ^ { * , i } \right]$ and $\left[ \tilde { \theta } _ { t _ { 0 } } , \tilde { \theta } _ { t _ { 0 } + q } \right]$ .

# D. General Settings

Optimizer. We use the Adam optimizer for all experiments.

Environment. The environment in which we run experiments is:

• Linux version: 5.15.0-91-generic   
• Operating system: Ubuntu 22.04.3 LTS   
• CPU information: Intel(R) Xeon(R) Platinum 8358 CPU $@$ 2.60GHz   
• GPU information: NVIDIA A800 80GB PCIe

Resources. The address and licenses of all datasets are as follows:

• Citeseer: https://github.com/kimiyoung/planetoid (MIT License) • Pubmed: https://github.com/kimiyoung/planetoid (MIT License) • Ogbn-arxiv: https://github.com/snap-stanford/ogb (MIT License) • Flickr: https://github.com/GraphSAINT/GraphSAINT (MIT License) • Reddit: https://github.com/williamleif/GraphSAGE (MIT License) • Squirrel: https://github.com/benedekrozemberczki/MUSAE (GPL-3.0 license) • Gamers: https://github.com/benedekrozemberczki/datasets (MIT License)