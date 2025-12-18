# EFFICIENT BACKPROPAGATION WITH VARIANCE-CONTROLLED ADAPTIVE SAMPLING

Ziteng Wang, Jianfei Chen1, Jun Zhu   
Dept. of Comp. Sci. and Tech., Institute for AI, BNRist Center, THBI Lab,   
Tsinghua-Bosch Joint ML Center, Tsinghua University   
wangzite23@mails.tsinghua.edu.cn; {jianfeic, dcszj}@tsinghua.edu.cn

# ABSTRACT

Sampling-based algorithms, which eliminate “unimportant” computations during forward and/or back propagation (BP), offer potential solutions to accelerate neural network training. However, since sampling introduces approximations to training, such algorithms may not consistently maintain accuracy across various tasks. In this work, we introduce a variance-controlled adaptive sampling (VCAS) method designed to accelerate BP. VCAS computes an unbiased stochastic gradient with fine-grained layerwise importance sampling in data dimension for activation gradient calculation and leverage score sampling in token dimension for weight gradient calculation. To preserve accuracy, we control the additional variance by learning the sample ratio jointly with model parameters during training. We assessed VCAS on multiple fine-tuning and pre-training tasks in both vision and natural language domains. On all the tasks, VCAS can preserve the original training loss trajectory and validation accuracy with an up to $7 3 . 8 7 \%$ FLOPs reduction of BP and $4 9 . 5 8 \%$ FLOPs reduction of the whole training process. The implementation is available at https://github.com/thu-ml/VCAS.

# 1 INTRODUCTION

Training neural networks can be computationally intensive. Contemporary networks typically employ stochastic gradient methods (Bottou et al., 2018) for training, which iteratively process batches of data to compute stochastic gradients through forward propagation (FP) and back propagation (BP) techniques (Rumelhart et al., 1986). $\mathrm { F P + B P }$ are costly, as they need to process every datum in the batch and every connection in the network, resulting in a multiplicative time complexity of batch size and model size. Such a time complexity becomes increasingly problematic in the era of big data and big models.

Data samples are not equally important. Some might be easy for the network to learn, while others might be extremely hard. Training can be accelerated by utilizing this disparity, focusing the available computational resources on more pivotal samples. At a high level, this can be achieved by further sampling the batch with higher keep probability of more important samples. The computational overhead is consequently diminished, in proportion to the quantity of retained samples. Various methods are proposed to assess the importance of samples, including meta-learning methods (Fan et al., 2017; Coleman et al., 2019; Mindermann et al., 2022), loss-based methods (Loshchilov & Hutter, 2015; Chang et al., 2017; Jiang et al., 2019; Ouyang et al., 2022),

![](images/92b5d0506f463a42b4277690ea9d6e2ff25c660d9da53c2f751e6fe27e220de9.jpg)  
Figure 1: VCAS mirrors the convergence trajectory with exact training with FLOPs redution of $4 1 . 5 6 \%$ . Other methods like SB (Jiang et al., 2019) and UB (Katharopoulos & Fleuret, 2018) fail with a similar FLOPs reduction.

and gradient norm based methods (Needell et al., 2014; Zhao & Zhang, 2015; Alain et al., 2015;   
Johnson & Guestrin, 2018; Katharopoulos & Fleuret, 2018).

While such methods seem promising, one core concern of sampling-based methods is their robustness. Misjudging the importance can hamper convergence, potentially leading to degraded accuracy and even longer training time than uniform sampling. Moreover, the optimal sample ratio is influenced by data distribution, which differs between tasks and is challenging to determine in advance. In general, there is a “no-free-lunch” phenomenon (Kaddour et al., 2023), where aggressive sampling often comes at the cost of reduced robustness.

In this work, we propose a robust variance-controlled adaptive sampling (VCAS) algorithm for deep learning under the stochastic optimization framework. VCAS computes a cost-effective approximated stochastic gradient (ASG) by partially conducting backpropagation for specific data and tokens. This ASG is unbiased, and we have developed an adaptive sampling method to meticulously control the variance of the ASG, aligning it with the original stochastic gradient’s variance. Consequently, convergence remains largely unaffected, with our method mirroring the progression of exact algorithms, as delineated in Fig. 1.

Unlike previous methods, VCAS construct the ASG in a fine-grained manner. Rather than dropping samples one-time in a whole, VCAS gradually drops more samples when backpropagating from topmost to bottommost network layers, as the gradient getting sparser. Furthermore, VCAS also more aggressively drops data in finer granularity of tokens rather than samples when computing the weight gradients. VCAS can achieve smaller variance under a given computational budget compared to coarse grained sampling on the data dimension.

We evaluate VCAS on multiple finetuning and pre-training tasks of language models and vision transformers. VCAS can preserve the original training loss trajectory and the validation accuracy on all tasks, while adaptively determining the computational saving depending on the difficulty of the task. VCAS can reduce the computational cost of backpropagation by up to $7 3 . 8 7 \%$ , and reduce the overall training computation by up to $4 9 . 5 8 \%$ .

# 2 RELATED WORK

Methods focusing on the difference of data, known as online batch selection (Loshchilov & Hutter, 2015), can be mainly categorized into three classes: meta learning methods, loss based methods and gradient norm based methods. In this section we will discuss these three ways separately and briefly introduce other orthogonal efficient training methods.

Meta Learning Methods. Some works formulate data sampling into an optimization problem and train a separate meta predictor to solve it. Fan et al. (2017) use deep reinforcement learning to train an agent for data selection. Coleman et al. (2019) and Mindermann et al. (2022) train a separate cheaper model with similar architecture for guidance. However, training a meta predictor will introduce further overhead and it’s a non-trivial learning task with more uncertainty introduced for weak theoretical guarantee.

Loss Based Methods. Loss is a natural indicator of the importance of different data. Loshchilov & Hutter (2015) maintains a history of losses and develops a sophisticated distribution based on the value or rank of loss. Jiang et al. (2019) and Ouyang et al. (2022) simplify it with sampling distribution proportion to the percentile of loss in the history. Chang et al. (2017) broadens the history to every datum and proposes to sample by the variance of prediction probability directly linked with previous losses. Dong et al. (2021) provides another method of minimizing the $L _ { 2 }$ norm between the sampled loss and the exact counterpart. Shah et al. (2020) samples the smallest loss for robustness to outliers. Zhang et al. (2023) ensembles several loss methods with a preset sample ratio and varies the weights assigned to these methods adaptively. Simple and effective as they may be, the loss based methods are heuristic and always need a hyperparameter of sample ratio to tune for different tasks, violating the goal of efficient training.

Gradient Norm Based Methods. Previous works have proved that the optimal data sampling distribution for SGD is proportional to the gradient norm(Needell et al., 2014; Zhao & Zhang, 2015). But calculating the gradient norm is prohibitive since it needs a full process of backpropagation. To solve this problem, Alain et al. (2015) applies distributed training with many workers calculating this importance score in parallel. Johnson & Guestrin (2018) uses a second-order approximation of gradient norm with history maintained. Closely related to our work, Katharopoulos & Fleuret (2018) develops a pure online algorithm by constructing an upper bound of gradient norm to sample with much cheaper computation. These methods are usually more expensive but have relatively strong theoretical guarantees. So we follow this way in our activation sampling.

![](images/9bcb51f4aa87d6ccb69a16561c6c3165f5202ecaba72ea99762c54d9b5b85cd2.jpg)  
Figure 2: The computing diagram of backpropagation with VCAS in every layer. We use light blue squares to represent small gradient entries and orange for large ones. White squares are discarded by sampling. The upper line calculates activation gradient and the lower for weight gradient. Please refer to Sec. 4 for notations.

Orthogonal Efficient Training Methods. Data pruning (Paul et al., 2021; Fayyaz et al., 2022) focuses on filtering less informative data before the whole training. Architecture pruning like layer dropping (Huang et al., 2016; Zhang & He, 2020) and token dropping (Hou et al., 2022; Yao et al., 2022; Li et al., 2022) modifies the architecture to make models faster to train with modest affect to performance. Mixed precision training and quantization (Micikevicius et al., 2018; Chen et al., 2021; Liu et al., 2022) change the training procedure to use low-precision in calculation for acceleration. Sparsity(Hoefler et al., 2021) focuses on pruning near-zero values in weights, activations, or gradients to achieve a low FLOPs(Raihan & Aamodt, 2020) and low memory footprint(Nikdan et al., 2023), yet is usually hard to bring a wall-clock time reduction like us due to the lack of hardware support(NVIDIA, 2021). All these works are orthogonal to our work since we focus on the computation approximation of a certain model architecture on a certain dataset with a certain training procedure to bring real training acceleration.

# 3 VARIANCE-CONTROLLED SAMPLING AS STOCHASTIC OPTIMIZATION

In this section, we present a high-level overview of our sampling algorithm as stochastic optimization. Consider the learning problem of a model $f ( X ; \theta )$ parameterized by $\theta$ on a dataset $\mathcal { D } = \{ ( X _ { i } , y _ { i } ) \} _ { i = 1 } ^ { | \mathcal { D } | }$ with a loss function $\ell ( \cdot , \cdot )$ . Define the learning objective as

$$
\mathcal { L } ( \boldsymbol { \theta } ) = \mathbb { E } _ { B } \left[ \ell ( \boldsymbol { f } ( \boldsymbol { X } ; \boldsymbol { \theta } ) , \boldsymbol { y } ) \right] ,
$$

where the expectation is taken over all possible batches $\boldsymbol { B } = ( \boldsymbol { X } , \boldsymbol { y } )$ from $\mathcal { D }$ . The model parameters can be learned by stochastic optimization algorithms (Bottou et al., 2018) with a stochastic gradient (SG) $g ( \theta ; B ) : = \mathrm { \bar { V } } _ { \theta } \ell ( f ( X ; \theta ) \bar { , } y )$ , which is an unbiased approximation of $\nabla _ { \boldsymbol { \theta } } \mathcal { L } ( \boldsymbol { \theta } )$ .

However, computing the stochastic gradient can be still too expensive, since it requires the full forward and back propagation, which iterate over all model parameters and all data in the batch. We build a cheap stochastic approximation $g ( \theta ; B , \epsilon )$ of the SG, which we refer as approximated stochastic gradient (ASG). ASG only computes the backpropagation partially, and is therefore cheaper than the SG. The randomness in the computing procedure of ASG is captured by $\epsilon$ . We ensure that ASG is unbiased: $\mathbb { E } _ { \epsilon } [ g ( \theta ; B , \epsilon ) ] = g ( \theta ; B )$ .

![](images/70493e3f1ddb4109d29b934940f93443cbfbd41770c0994ba694c2aa23e5d633.jpg)  
Figure 3: Gradient distribution over different layer and iterations of BERT-base finetuning on SST2 (6315 iterations in total). The normalized gradient norm of each datum is shown in the heatmaps. Black solid lines are the $9 5 \%$ percentile. Data above the lines are likely to be dicarded by VCAS.

With an unbiased SG, stochastic optimization algorithms are guaranteed to converge to a stationary point of Eq. (1), while the converge speed depends on the variance (cf. Bottou et al. (2018)). Therefore, if the variance of the ASG can be controlled to the similar variance level of SG, substituting the SG with ASG should have little impact to the convergence behavior. In fact, by the law of total variance (Chung, 2001), the variance of ASG can be decoupled as

$$
\operatorname { V a r } \left[ g ( \boldsymbol { \theta } ; \boldsymbol { B } , \boldsymbol { \epsilon } ) \right] = \operatorname { V a r } \left[ g ( \boldsymbol { \theta } ; \boldsymbol { B } ) \right] + \mathbb { E } _ { \boldsymbol { B } } \left[ \operatorname { V a r } _ { \boldsymbol { \epsilon } } \left[ g ( \boldsymbol { \theta } ; \boldsymbol { B } , \boldsymbol { \epsilon } ) \right] \right] ,
$$

where the first term is the intrinsic variance of SG caused by subsampling batches from the dataset, and the second term is the additional variance incurred by ASG. In the subsequent sections, we will discuss our constructions of the ASG, which incurs negligible additional variance compared to SG.

# 4 FINE-GRAINED SAMPLING

Here we present variance-controlled adaptive sampling (VCAS), a specific construction of the ASG. We compute ASG by approximating the backpropagation in a fine-grained manner, and speed up matrix multiplications with importance sampling on the data dimension.

Assume a batch $X$ of shape $N \times T \times K$ , where $N$ is the batch size, $T$ is the number of tokens of each datum, and $K$ is the dimensionality. For an $L$ -layer network , the model $f ( X ; \theta )$ can be described by the following forward propagation procedure: ${ \cal Z } ^ { ( 0 ) } = X , { \cal Z } ^ { ( l ) } =$ $f ^ { ( l ) } \left( Z ^ { ( l - 1 ) } ; \theta ^ { ( l ) } \right) , f ( X ; \theta ) = Z ^ { ( L ) }$ , where $Z ^ { ( l ) }$ and $\theta ^ { ( l ) }$ are the activation and parameters of the $l$ -th layer, and $\theta = ( \theta ^ { ( l ) } ) _ { l = 1 } ^ { L }$ . The SG can be computed by back-propagation in the following form: $\nabla _ { Z ^ { ( l - 1 ) } } = h ^ { ( l ) } \left( \nabla _ { Z ^ { ( l ) } } ; \dot { Z ^ { ( l - 1 ) } } , \theta ^ { ( l ) } \right) , \nabla _ { \theta ^ { ( l ) } } = g ^ { ( l ) } \left( \nabla _ { Z ^ { ( l ) } } ; Z ^ { ( l - 1 ) } , \theta ^ { ( l ) } \right)$ , where $\nabla _ { Z ^ { ( l ) } }$ and $\nabla _ { \theta ^ { ( l ) } }$ denote the activation / weight gradient, $\it { h ^ { ( l ) } }$ and $g ^ { ( l ) }$ denote the function that calculates input $/$ weight gradient of layer $l$ with the output gradient, layer input and weight. The SG $\mathrm { ~ \ i ~ } _ { g } ( \theta ; \mathbb { B } ) = \big ( \mathbf { \hat { V } } _ { \theta ^ { ( l ) } } \big ) _ { l = 1 } ^ { L }$ .

As illustrated by Fig. 3, the activation gradients $\nabla _ { Z ^ { ( l ) } }$ are sparse: the gradient $( \nabla _ { Z ^ { ( l ) } } ) _ { i }$ is close to zero for most sample $i$ , except for a few important samples. Such sparsity becomes more prominent as backpropagating to lower layers and as the training progresses. To speed up computation, we add samplers in the backpropagation graph:

$$
\begin{array} { r l } & { \hat { \nabla } _ { Z ^ { ( l ) } } = \mathrm { S a m p l e A } _ { \epsilon , \rho _ { l } } \left( \nabla _ { Z ^ { ( l ) } } \right) , \quad \nabla _ { Z ^ { ( l - 1 ) } } = h ^ { ( l ) } \left( \hat { \nabla } _ { Z ^ { ( l ) } } ; Z ^ { ( l - 1 ) } , \theta ^ { ( l ) } \right) , } \\ & { \tilde { \nabla } _ { Z ^ { ( l ) } } = \mathrm { S a m p l e W } _ { \xi _ { l } , \nu _ { l } } \left( \hat { \nabla } _ { Z ^ { ( l ) } } , Z ^ { ( l - 1 ) } \right) , \quad \nabla _ { \theta ^ { ( l ) } } = g ^ { ( l ) } \left( \tilde { \nabla } _ { Z ^ { ( l ) } } ; Z ^ { ( l - 1 ) } , \theta ^ { ( l ) } \right) . } \end{array}
$$

The sampler $\mathrm { S a m p l e A } _ { \epsilon , \rho _ { l } } \left( \cdot \right)$ randomly filter out unimportant data from the activation gradient, the keep ratio is $\rho _ { l }$ , with the randomness captured by $\epsilon$ . The sampler is applied for each layer, so the activation gradient becomes increasingly sparse when backpropagating from the $L$ -th layer to the first layer. The sampler $\mathrm { S a m p l e W } _ { \xi _ { l } , \nu _ { l } } \ : ( \cdot )$ filters (data, token) pairs specifically for weight gradient calculation, with a keep ratio $\nu _ { l }$ and the randomness $\xi _ { l }$ . With these samplers, we only need to compute backpropagation for the retained data $/$ token, so the computational cost is reduced. The sampling procedure is illustrated in Fig. 2, which constructs an unbiased ASG $g ( \theta ; B , \epsilon , \xi , \rho , \nu ) =$ $( \boldsymbol { \nabla } _ { \theta ^ { ( l ) } } ) _ { l = 1 } ^ { \bar { L } }$ , with $\nabla _ { \theta ^ { ( l ) } }$ defined as Eq. (2), and $\xi = ( \xi _ { l } ) _ { l = 1 } ^ { L }$ , $\rho = ( \rho ) _ { l = 1 } ^ { L }$ , $\nu = ( \nu _ { l } ) _ { l = 1 } ^ { \tt L }$ .

# 4.1 ACTIVATION GRADIENT

We apply unbiased low-variance approximation to the activation gradient to speed up subsequent computation. For an activation gradient tensor $G$ of shape $N \times T \times K$ , we sample

$$
\begin{array} { r } { \hat { G } = \mathrm { S a m p l e A } _ { \epsilon , \rho } \left( G \right) = G \circ ( m ( \epsilon , \rho ) \otimes \mathbf { 1 } \otimes \mathbf { 1 } ) , } \end{array}
$$

where Berno $\circ$ is elemenli vector: $\otimes$ nsor o, wh ee ask , and $\boldsymbol { m } \in \mathbb { R } ^ { N }$ $m ( \epsilon , \rho ) _ { i } = \mathrm { B e r n } ( p _ { i } ; \epsilon ) / p _ { i }$ $\begin{array} { r } { \sum _ { i = 1 } ^ { N } p _ { i } = N \rho } \end{array}$ $\operatorname { B e r n } ( p ; \epsilon )$ $p$ $\epsilon$ $\mathbb { E } [ m ( \epsilon , \rho ) _ { i } ] = 1 , \forall i$   
the approximation is unbiased: $\mathbb { E } [ \hat { G } ] = G$ . The sampler zeros out the gradient for all the data whose   
$m ( \epsilon , \bar { \rho } ) _ { i } = 0$ . The amount of retained data is $N \rho$ in expectation. With the sampler, we only need to   
compute backpropagation for retained data, so the cost is $\rho$ times lower.

The variance of the approximation is $\begin{array} { r } { \mathrm { V a r } \left[ \boldsymbol { \hat { G } } \right] = \sum _ { i = 1 } ^ { N } \frac { 1 - p _ { i } } { p _ { i } } \left. \boldsymbol { G } _ { i } \right. _ { F } ^ { 2 } } \end{array}$ , where we define the variance of a random tensor element-wise as $\begin{array} { r } { \mathrm { V a r } \left[ \hat { G } \right] = \sum _ { i j k } \mathrm { V a r } \left[ \hat { G } _ { i j k } \right] } \end{array}$ , and $G _ { i }$ denotes the $i$ -th matrix of $G$ in the $N$ dimension. We compute the keep probability $\bar { ( } p _ { i } )$ to minimize the variance, deriving a distribution proportional to the gradient norm of each datum: $\dot { p } _ { i } \propto \| G _ { i } \| _ { F }$ . Minimizing the variance of the activation gradient not necessarily minimize the variance of ASG, which is the gradient of parameters. Nevertheless, this is a useful heuristic which empirically achieves low variance as is revealed by Katharopoulos & Fleuret (2018), and the ASG variance will be carefully controlled by our adaptive algorithm, as we shall see soon in Sec. 5.

# 4.2 WEIGHT GRADIENT

We can accelerate the computation of weight gradient for linear layers by sampling in both data and token dimensions. Consider the approximate back propagation of a linear layer $Z ^ { ( l ) } = Z ^ { ( l - 1 ) } \theta ^ { ( l ) } { } ^ { \top }$ :

$$
\begin{array} { r } { \tilde { \boldsymbol { \gamma } } _ { Z ^ { ( l ) } } = \mathrm { S a m p l e A } _ { \epsilon , \rho _ { l } } \left( \nabla _ { Z ^ { ( l ) } } \right) , \quad \tilde { \nabla } _ { Z ^ { ( l ) } } = \mathrm { S a m p l e W } _ { \xi _ { l } , \nu _ { l } } \left( \hat { \nabla } _ { Z ^ { ( l ) } } , Z ^ { ( l - 1 ) } \right) , \quad \nabla _ { \theta ^ { ( l ) } } = \tilde { \nabla } _ { Z ^ { ( l ) } } ^ { \top } Z ^ { ( l - 1 ) } } \end{array}
$$

in matrix form, where we reshape the activation/gradients to $N T { \times } K$ , and $\hat { \nabla } _ { Z ^ { ( l ) } }$ is already a sampled matrix with only $N T \rho _ { l }$ non-zero rows in expectation. However, $\hat { \nabla } _ { Z ^ { ( l ) } }$ is only sampled in the data dimension. In fact, even $( \hat { \nabla } _ { Z ^ { ( l ) } } ) _ { i }$ is retained for some datum $i$ , it might still have some rows (i.e., tokens) which are close to zero. We can further sample

$$
\begin{array} { r } { \tilde { \nabla } _ { Z ^ { ( l ) } } = \mathrm { S a m p l e W } _ { \xi _ { l } , \nu _ { l } } \left( \hat { \nabla } _ { Z ^ { ( l ) } } , Z ^ { ( l - 1 ) } \right) = \hat { \nabla } _ { Z ^ { ( l ) } } \circ ( m ( \xi , \nu ) ^ { \top } \mathbf { 1 } ) , } \end{array}
$$

where the mask $\mathrm { B e r n } ( q _ { i } ; \epsilon ) / q _ { i }$ , where $\boldsymbol { m } \in \mathbb { R } ^ { N L }$ $\begin{array} { r } { \sum _ { i = 1 } ^ { N T } q _ { i } = N T \rho _ { l } \nu _ { l } } \end{array}$ is a random Bernoulli vector, and 1 is an all-one vector: . The variance is $m ( \xi , \nu ) _ { i } =$

$$
\mathrm { V a r } \left[ \tilde { \boldsymbol { \nabla } } _ { \boldsymbol { \theta } ^ { ( l ) } } \right] = \sum _ { i = 1 } ^ { N T } \frac { 1 - q _ { i } } { q _ { i } } \left. \hat { \boldsymbol { \nabla } } _ { Z ^ { ( l ) } { } _ { i } } \right. _ { 2 } ^ { 2 } \left. Z _ { i } ^ { ( l - 1 ) } \right. _ { 2 } ^ { 2 } .
$$

The minimal variance solution is $q _ { i } \propto \left. \hat { \nabla } _ { Z ^ { ( l ) } { i } } \right. _ { 2 } \left. Z _ { i } ^ { ( l - 1 ) } \right. _ { 2 }$ . This sampling method is also known as leverage score sampling in randomized numerical linear algebra (Drineas & Mahoney, 2018).

# 5 ADAPTING SAMPLE RATIOS

The question remained is how to set the sample ratios $( \rho _ { l } ) _ { l = 1 } ^ { L }$ and $( \nu _ { l } ) _ { l = 1 } ^ { L }$ . There is a tradeoff: lowering the sample ratio reduces the computational cost, but increases the variance. As discussed in Sec. 3, this ratio should be set to ensure that the additional variance of ASG is marginal compared to the original variance of SG. Adapting the sample ratio is nontrivial since the gradient sparsity pattern vary across layers and vary over time during training. In this section, we present an adaptation algorithm to control the variance during the entire training trajectory.

Intuitively, when the gradient norm First, we introduce a single hyperparameter $\left( \lVert G _ { i } \rVert _ { F } \right) _ { i = 1 } ^ { N }$ $s$ to control the sample ratios l=1 becomes sparser, we can more aggressively utilize $( \rho _ { l } ) _ { l = 1 } ^ { L }$ for all layers.

smaller keep ratio $\rho _ { l }$ to maximize speedup. Therefore, we compute $\rho _ { l }$ based on the sparsity $p _ { l }$ of the gradient norm sequence:

$$
p _ { l } ( s ) = \operatorname* { m i n } \{ n / N | \sum _ { i = 1 } ^ { n } \| G _ { i } \| _ { F } \geq s \sum _ { i = 1 } ^ { N } \| G _ { i } \| _ { F } \} , \quad \rho _ { l } ( s ) = \operatorname* { m a x } _ { j \leq l } p _ { j } ( s )
$$

where $s \in [ 0 , 1 ]$ is a hyperparameter on how much gradient norm is preserved. It’s shown in Fig. 3 that gradient norm grows sparser with layer, yielding a descending trend of $p _ { l }$ for $l$ from $L$ to 1. Thus it’s reasonable to construct a monotone increasing sequence of $\{ \rho _ { l } \} _ { l = 1 } ^ { L }$ based on $\{ p _ { l } \} _ { l = 1 } ^ { L }$ .

By law of total variance, we can decompose the variance of ASG as b $\begin{array} { r } { \lceil \operatorname { a r } [ g ( \theta ; \mathcal { B } , \epsilon , \xi , \rho , \nu ) ] = \operatorname { V a r } [ g ( \theta ; \mathcal { B } ) ] + \mathbb { E } _ { \mathcal { B } } [ \operatorname { V a r } _ { \epsilon } [ g ( \theta ; \mathcal { B } , \epsilon , \rho ( s ) ) ] ] + \mathbb { E } _ { \mathcal { B } , \epsilon } [ \operatorname { V a r } _ { \xi } [ g ( \theta ; \mathcal { B } , \epsilon , \xi , \rho , \nu ) ] , } \end{array}$ where we write $g ( \theta ; B , \epsilon , \rho ) : = \mathbb { E } _ { \xi } [ g ( \theta ; B , \epsilon , \xi , \rho , \nu ) ]$ to be the ASG without the sampler for weight gradient computation. The three variance terms are the SG variance, the variance introduced by approximately computing activation gradient, and the variance introduced by approximately computing weight gradient, respectively. Our algorithm adaptively tunes $s$ and $\nu$ during train to control the last two variance terms to be fractional comparing to the first variance term.

Controlling $\mathbb { E } _ { B } [ \operatorname { V a r } _ { \epsilon } \left[ g ( \theta ; B , \epsilon , \rho ( s ) ) \right] ]$ : We adopt a zeroth order method to adapt the hyperparameter $s$ to keep $\mathbb { E } _ { \boldsymbol { B } } [ \mathrm { V a r } _ { \epsilon } [ g ( \boldsymbol { \theta } ; \boldsymbol { B } , \epsilon , \boldsymbol { \rho } ( \boldsymbol { s } ) ) ] ] = \tau _ { a c t } \mathrm { V a r } [ g ( \boldsymbol { \theta } ; \boldsymbol { B } ) ]$ , where $\tau _ { a c t } \ll 1$ is a small constant. That is, the additional variance raised by approximately computing activation gradient is only $\tau _ { a c t }$ times the SG variance itself. Since larger $s$ increases the keep ratio and decreases the variance, we adopt the update:

$$
s \gets s + \alpha \mathrm { s i g n } \left( \mathbb { E } _ { \mathcal { B } } [ \mathrm { V a r } _ { \epsilon } \left[ g ( \theta ; \mathcal { B } , \epsilon , \rho ( s ) ) \right] ] - \tau _ { a c t } \mathrm { V a r } \left[ g ( \theta ; \mathcal { B } ) \right] \right) ,
$$

where $\mathrm { s i g n } ( x ) = + 1$ when $x \ge 0$ and $\mathrm { s i g n } ( x ) = - 1$ when $x \ < \ 0$ , and $\alpha$ is a step size. We approximate the expectation and variance with empirical ones with $M$ Monte Carlo repetitions. Therefore, each update requires $O ( M ^ { 2 } )$ $\mathrm { F P + B P s }$ , and we run the update every $F$ SGD iterations, where $F \gg M ^ { 2 }$ .

Controlling $\mathbb { E } _ { B , \epsilon } [ \mathrm { V a r } _ { \xi } [ g ( \theta ; B , \epsilon , \xi , \rho , \nu ] ]$ : As the variance sums up for each parameter $\theta ^ { ( l ) }$ , we can further decompose the variance as

$$
\mathbb { E } _ { \mathcal { B } , \epsilon } [ \mathrm { V a r } _ { \xi } \left[ g ( \theta ; \mathcal { B } , \epsilon , \xi , \rho , \nu ] \right] = \sum _ { l = 1 } ^ { L } \mathbb { E } _ { \mathcal { B } , \epsilon } \left[ \mathrm { V a r } _ { \xi } \left[ g ^ { ( l ) } ( \theta ; \mathcal { B } , \epsilon , \xi _ { l } , \rho , \nu _ { l } \right] \right] ,
$$

where $g ^ { ( l ) }$ is the gradient of the $l$ -th layer (i.e., $\nabla _ { \theta ^ { ( l ) } }$ ). We control the variance of each layer separately to keep $\begin{array} { r } { \mathbb { E } _ { \mathcal { B } , \epsilon } \left[ \operatorname { V a r } _ { \xi } \left[ g ^ { ( l ) } ( \theta ; \mathcal { B } , \epsilon , \xi _ { l } , \rho , \nu _ { l } ) \right] \right] = \tau _ { w } \operatorname { V a r } \left[ g ^ { ( l ) } ( \theta ; \mathcal { B } ) \right] } \end{array}$ . Again, this is achieved by a zeroth-order algorithm:

$$
\nu _ { l } \gets \nu _ { l } \beta ^ { \mathrm { s i g n } } \big ( \mathbb { E } _ { \mathcal { B } , \epsilon } \big [ \mathrm { V a r } _ { \xi } \big [ g ^ { ( l ) } ( \theta ; \mathcal { B } , \epsilon , \xi _ { l } , \rho , \nu _ { l } ) \big ] \big ] - \tau _ { w } \mathrm { V a r } \big [ g ^ { ( l ) } ( \theta ; \mathcal { B } ) \big ] \big ) ,
$$

where $\mathrm { V a r } _ { \xi } \left[ g ^ { ( l ) } \right]$ can be computed analytically by Eq. 3, and $\beta$ is a multiplier.

Now we are fully prepared to present the whole picture of VCAS in Alg. 1. Please refer to Appendix. D for more details about the algorithm.

# 6 EXPERIMENTS

# 6.1 TRAINING FLOPS REDUCTION

We assessed VCAS on multiple fine-tuning and pre-training tasks in both vision and natural language domains. We compare our algorithm with the exact training and two previous works in BP sampling: a loss based method SB(selective backprop) in Johnson & Guestrin (2018) and a gradient norm based method UB(upper bound) in Katharopoulos & Fleuret (2018). We choose these two methods since they are entirely online and need little modification to the original training pipeline like us. The results are shown in Tab. 1. All results are the average of 3 different seeds except for BERT-base pretraining and ViT finetuning on ImageNet-1k which we use 1.

<table><tr><td>Algorithm 1 Variance controlled adaptive sampling(VCAS) for backpropagation</td></tr><tr><td>Require:update frequency F, Monte-Carlo repetition number M, variance tolerant ratio for activation Tact, for weight τw, s step size α, weight ratio multiplier β</td></tr><tr><td>s ← 1, activation sample ratio schedule {ρl}l=1 ← 1, weight sample ratios {νl}l=1 ← 1 t ← 0</td></tr><tr><td>while not converge do if t mod F = 0 then</td></tr><tr><td>for i in 1, . . . , M do</td></tr><tr><td>(Xi, yi) ← batch selected randomly SGD gradient Gs,i ← exact backward using (Xi, yi)</td></tr><tr><td>for j in 1, . . . , M do</td></tr><tr><td>activation gradient Gact,i,j ← backward using (Xi, yi) with SampleA only calculate weight variance Vw,i,j analytically with Eq. 3 and Eq. 6</td></tr><tr><td>end for end for</td></tr><tr><td>SGD variance Vs ← 1 PM Gs,i| M −1 ∑i=1 F</td></tr><tr><td>1 ∑j=1 ∥Gact,i,j − Gs,ik2 )</td></tr><tr><td></td></tr><tr><td>M</td></tr><tr><td></td></tr><tr><td>update s with Vact and Vs according to Eq. 5 update {ρl}l=1 with new s according to Eq. 4</td></tr><tr><td>update {νl}=1 with Vw and Vs according to Eq. 7</td></tr><tr><td>end if</td></tr></table>

Note that to avoid falling into the pitfall of unfair comparison with baseline which is not tuned under efficient settings as is pointed out by Dehghani et al. (2021) and Kaddour et al. (2023), for all these experiments we use the same conservative setting of $\tau _ { a c t } = \tau _ { w } = 0 . 0 2 5 , \alpha = 0 . 0 1 , \beta =$ 0.95, $M = 2$ . We preset all these values heuristically without any tuning or prior knowledge. The only hyperpamater we modified among different tasks is the variance calculation frequency $F$ , which can be defined easily according to the total training steps.

In fact, all the hyperparameters introduced by VCAS have explicit meanings and are insensitive. We show experimentally that though extra tuning may achieve a slightly better result, overall VCAS is robust to these hyperparameters with reasonable values. Please refer to Appendix. A for details about ablation studies on these insensitive hyperparameters.

For SB and UB, we both adopt a sample ratio of $1 / 3$ , since it’s the recommended setting in the original papers and it can achieve a FLOPs reduction of $1 - ( 1 + 2 * 1 / 3 ) / 3 = 4 4 . 4 4 \%$ which is close to the results we get in most tasks. An exception is BERT-base pretraining task where we find the FLOPs reduction achievable is low so we manually set the sample ratio of SB and UB to get the same FLOPs reduction as VCAS, so that they can still give a decent result. Nevertheless we are indeed favoring these methods by helping them to define a reasonable sample ratio, which can not be done themselves.

From the table we can see that overall VCAS is better than SB and UB with the least impact on final train loss and final evaluation accuracy. With FLOPs reduction of up to $4 9 . 5 8 \%$ , VCAS can still achieve nearly the same results with the exact counterpart.

# 6.2 WALL-CLOCK TIME REDUCTION

We record the wall-clock time of BERT-large finetuning on MNLI and ViT-large finetuning on ImageNet-1k with NVIDIA 3090Ti, the results are depicted in Tab. 2 and Tab. 3.

From these tables, we can find that VCAS can translate FLOPs reduction into wall-clock time reduction as effectively as simpler online batch sampling methods like UB and SB that drop part of data one-time in a whole, while enjoying mirrored performance with the exact training under theoretical guarantee.

Table 1: Comparison of VCAS with other methods. Data format is Final Train Loss / Final Eval Acc. $( \% )$ for exact, SB and UB, and Final Train Loss / Final Eval Acc. $( \% ) / F L O P s$ reduction ratio $\% )$ for VCAS. The FLOPs reduction of SB and UB is $2 1 . 5 8 \%$ for BERT pretraining and $4 4 . 4 4 \%$ for other tasks. VCAS’s FLOPs take account of the adaptation overhead. For BERT pretraining, accuracy=average performance on GLUE. Bold indicates the best result of each metric except for exact. Underline means Eval Acc less than $0 . 1 \%$ off the exact training.   

<table><tr><td rowspan=1 colspan=1>Task</td><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>exact</td><td rowspan=1 colspan=1>SB</td><td rowspan=1 colspan=1>UB</td><td rowspan=1 colspan=1>VCAS</td></tr><tr><td rowspan=1 colspan=1>BERT-basepretraining</td><td rowspan=1 colspan=1>C4</td><td rowspan=1 colspan=1>2.099 / 78.37</td><td rowspan=1 colspan=1>2.133 / 77.53</td><td rowspan=1 colspan=1>2.106 / 77.96</td><td rowspan=1 colspan=1>2.134 / 78.36 / 21.58</td></tr><tr><td rowspan=1 colspan=1>BERT-basefinetuning</td><td rowspan=1 colspan=1>MNLI-mQQPQNLISST-2</td><td rowspan=1 colspan=1>0.2372 / 84.330.1143 / 91.000.1014 / 91.670.0559 / 92.59</td><td rowspan=1 colspan=1>0.3833 / 83.710.1441 / 90.760.2017 / 90.580.0727 / 92.63</td><td rowspan=1 colspan=1>0.2957 / 83.820.1964 / 89.530.1441 / 91.230.0743 / 92.82</td><td rowspan=1 colspan=1>0.2428 / 84.23 / 41.560.1189 / 90.92 / 47.100.1056 / 91.29 / 44.450.0600 / 93.04 / 48.28</td></tr><tr><td rowspan=1 colspan=1>BERT-largefinetuning</td><td rowspan=1 colspan=1>MNLI-mQQPQNLISST-2</td><td rowspan=1 colspan=1>0.1439 / 86.580.0885 / 91.640.0877 / 92.020.0537 / 93.60</td><td rowspan=1 colspan=1>0.2492 / 85.180.1308 / 91.200.1436 / 91.500.1136 / 91.81</td><td rowspan=1 colspan=1>0.2266 / 86.090.1751 / 90.510.1325 / 91.980.0838 / 93.40</td><td rowspan=1 colspan=1>0.1619 / 86.63 / 44.170.0962 / 91.57 / 49.500.0640 / 92.15 / 46.190.0593 / 93.67 / 49.24</td></tr><tr><td rowspan=1 colspan=1>ViT-basefinetuning</td><td rowspan=1 colspan=1>CIFAR10CIFAR100ImageNet-1k</td><td rowspan=1 colspan=1>0.1868 / 98.920.8760 / 91.190.6032 / 82.27</td><td rowspan=1 colspan=1>0.2367 / 98.822.248 / 89.600.6533 / 82.09</td><td rowspan=1 colspan=1>0.1923 / 98.941.175 / 89.680.6109 / 82.28</td><td rowspan=1 colspan=1>0.1873 / 98.90 / 45.900.8811 / 91.08 / 29.320.6089 / 82.27 / 45.29</td></tr><tr><td rowspan=1 colspan=1>ViT-largefinetuning</td><td rowspan=1 colspan=1>CIFAR10CIFAR100ImageNet-1k</td><td rowspan=1 colspan=1>0.1359 / 99.240.4590 / 93.560.4135 / 82.04</td><td rowspan=1 colspan=1>0.1439 / 99.210.5983 / 93.070.4637 / 82.21</td><td rowspan=1 colspan=1>0.1378 / 99.170.5170 / 93.360.4242 / 82.21</td><td rowspan=1 colspan=1>0.1393 / 99.28 / 48.370.4649 / 93.64 / 38.670.4228 / 82.27 / 49.58</td></tr></table>

Table 2: Wall-clock time of BERT-large finetuning on MNLI.   

<table><tr><td>Method</td><td>Train Loss</td><td>Eval Acc.(%)</td><td>Wall-clock Time(h) | FLOPs↓(%) | Time↓(%)</td><td></td><td></td></tr><tr><td>exact</td><td>0.1439</td><td>86.58</td><td>5.478</td><td></td><td>-</td></tr><tr><td>SB</td><td>0.2492</td><td>85.18</td><td>4.320</td><td>44.44</td><td>21.14</td></tr><tr><td>UB</td><td>0.2266</td><td>86.09</td><td>4.266</td><td>44.44</td><td>22.12</td></tr><tr><td>VCAS</td><td>0.1619</td><td>86.63</td><td>4.437</td><td>44.17</td><td>19.00</td></tr></table>

Table 3: Wall-clock time of ViT-large finetuning on ImageNet-1k.   

<table><tr><td>Method</td><td>Train Loss</td><td>Eval Acc.(%) |</td><td>| Wall-clock Time(h) | FLOPs↓(%) | Time↓(%)</td><td></td><td></td></tr><tr><td>exact</td><td>0.4135</td><td>82.04</td><td>52.29</td><td></td><td>-</td></tr><tr><td>SB</td><td>0.4637</td><td>82.21</td><td>42.56</td><td>44.44</td><td>18.61</td></tr><tr><td>UB</td><td>0.4242</td><td>82.21</td><td>41.92</td><td>44.44</td><td>19.83</td></tr><tr><td>VCAS</td><td>0.4228</td><td>82.27</td><td>41.28</td><td>49.58</td><td>21.06</td></tr></table>

The success of VCAS comes in two ways. One is the fine-grained sampling strategy that samples activation and weight jointly, which enables us to achieve much lower FLOPs given the variance budget. The other is the variance controlled framework combined with the self-adaptation algorithm, with which we are able to learn the proper sample ratios of different training phases. In the following two subsections, we will experimentally show the effectiveness of these two folds.

# 6.3 EFFECTIVENESS OF FINE-GRAINED SAMPLING

We compare VCAS that samples activation and weight jointly with strategies that solely sampling activation or weight. Specifically, we keep an equal extra variance for BERT-base finetuning on MNLI. We set $\tau _ { a c t } = \tau _ { w } = 0 . 0 2 5$ for VCAS, $\tau _ { a c t } = 0 . 0 5$ for activation sampling only and $\tau _ { w } =$ 0.05 for weight sampling only. We find that under the preliminary that $\tau _ { a c t } , \tau _ { w } \ll 1$ , the results of these sampling strategies show no significant difference due to controlled variance. While as is shown in Fig. 4, VCAS can achieve a much greater FLOPs reduction with the same total variance introduced. It’s reasonable since we can utilize more sparsity in both data and token dimensions with a fine-grained sampling strategy of VCAS.

![](images/7eba1cbd547fec1058e873b4e9aad2bac048cf4be53e12251e69e483e8e6e037.jpg)  
Figure 4: FLOPs reduction ratio of VCAS vs. sampling activation or weight solely with equal variance.

![](images/9c128f77d7756e326bcf9ca8df1f4c181ae37e78d1033942db5f0aa899c1c060.jpg)  
Figure 5: Gradient variance of different methods.

# 6.4 EFFECTIVENESS OF VARIANCE CONTROL AND SELF-ADAPTATION

In Fig. 5 we plot the variance of different methods during training process of BERT-base finetuning on MNLI. We can find that VCAS is able to control the extra sampling variance introduced to our preset threshold, while for other variance-unaware algorithms like UB and SB, the extra variance is out of control with a similar FLOPs reduction.

With carefully controlled variance, a similar convergence with exact training is guaranteed as we mentioned in the introduction. As is depicted in Fig. 1 and Fig. 6 for BERT-base finetuning on MNLI, VCAS shares nearly the same convergence trajectory with the exact training with reduced FLOPs, while UB converges slightly slower due to uncontrolled variance, and SB converges in an entirely different trajectory with variance introduced far larger than exact.

![](images/5e14cd22b8a791f9e162ce27fc563632826e65de1046c357f70253d8b707dbb7.jpg)  
Figure 6: Convergence comparison of different sampling methods. FLOPs is normalized by exact training.

# 7 CONCLUSION

We propose VCAS, a robust sampling method for back propagation with controlled variance and self-adaptive sample ratios. VCAS computes an approximate stochastic gradient by applying finegrained sampling to gradually remove samples and tokens during backpropagation. VCAS enjoys similar variance, convergence trajectory, and final accuracy with exact back propagation, while reduces the training cost by up to $4 9 . 5 8 \%$ .

# ACKNOWLEDGEMENTS

The authors would like to thank Bingrui Li and Weiyu Huang for their valuable discussions and help on algorithm design and implementation details. This work was supported by the National Key Research and Development Program of China (No. 2021ZD0110502), NSFC Projects (Nos. 62376131, 62061136001, 62106123, 62076147, U19A2081, 61972224), Tsinghua Institute for Guo Qiang, and the High Performance Computing Center, Tsinghua University. J.Z is also supported by the XPlorer Prize.

# REFERENCES

Guillaume Alain, Alex Lamb, Chinnadhurai Sankar, Aaron Courville, and Yoshua Bengio. Variance reduction in sgd by distributed importance sampling. arXiv preprint arXiv:1511.06481, 2015.

Leon Bottou, Frank E Curtis, and Jorge Nocedal. Optimization methods for large-scale machine ´ learning. SIAM Review, 60(2):223–311, 2018.

Haw-Shiuan Chang, Erik Learned-Miller, and Andrew McCallum. Active bias: Training more accurate neural networks by emphasizing high variance samples. Advances in Neural Information Processing Systems, 30, 2017.

Jianfei Chen, Lianmin Zheng, Zhewei Yao, Dequan Wang, Ion Stoica, Michael Mahoney, and Joseph Gonzalez. Actnn: Reducing training memory footprint via 2-bit activation compressed training. In International Conference on Machine Learning, pp. 1803–1813. PMLR, 2021.

Kai Lai Chung. A course in probability theory. Academic press, 2001.

Cody Coleman, Christopher Yeh, Stephen Mussmann, Baharan Mirzasoleiman, Peter Bailis, Percy Liang, Jure Leskovec, and Matei Zaharia. Selection via proxy: Efficient data selection for deep learning. arXiv preprint arXiv:1906.11829, 2019.

Mostafa Dehghani, Anurag Arnab, Lucas Beyer, Ashish Vaswani, and Yi Tay. The efficiency misnomer. arXiv preprint arXiv:2110.12894, 2021.

Chaosheng Dong, Xiaojie Jin, Weihao Gao, Yijia Wang, Hongyi Zhang, Xiang Wu, Jianchao Yang, and Xiaobing Liu. One backward from ten forward, subsampling for large-scale deep learning. arXiv preprint arXiv:2104.13114, 2021.

Petros Drineas and Michael W Mahoney. Lectures on randomized numerical linear algebra. The Mathematics of Data, 25(1), 2018.

Yang Fan, Fei Tian, Tao Qin, Jiang Bian, and Tie-Yan Liu. Learning what data to learn. arXiv preprint arXiv:1702.08635, 2017.

Mohsen Fayyaz, Ehsan Aghazadeh, Ali Modarressi, Mohammad Taher Pilehvar, Yadollah Yaghoobzadeh, and Samira Ebrahimi Kahou. Bert on a data diet: Finding important examples by gradient-based pruning. arXiv preprint arXiv:2211.05610, 2022.

Jonas Geiping and Tom Goldstein. Cramming: Training a language model on a single gpu in one day. arXiv preprint arXiv:2212.14034, 2022.

Torsten Hoefler, Dan Alistarh, Tal Ben-Nun, Nikoli Dryden, and Alexandra Peste. Sparsity in deep learning: Pruning and growth for efficient inference and training in neural networks. The Journal of Machine Learning Research, 22(1):10882–11005, 2021.

Le Hou, Richard Yuanzhe Pang, Tianyi Zhou, Yuexin Wu, Xinying Song, Xiaodan Song, and Denny Zhou. Token dropping for efficient bert pretraining. arXiv preprint arXiv:2203.13240, 2022.

Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q Weinberger. Deep networks with stochastic depth. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pp. 646–661. Springer, 2016.

Angela H Jiang, Daniel L-K Wong, Giulio Zhou, David G Andersen, Jeffrey Dean, Gregory R Ganger, Gauri Joshi, Michael Kaminksy, Michael Kozuch, Zachary C Lipton, et al. Accelerating deep learning by focusing on the biggest losers. arXiv preprint arXiv:1910.00762, 2019.

Tyler B Johnson and Carlos Guestrin. Training deep models faster with robust, approximate importance sampling. Advances in Neural Information Processing Systems, 31, 2018.

Jean Kaddour, Oscar Key, Piotr Nawrot, Pasquale Minervini, and Matt J Kusner. No train no gain: Revisiting efficient training algorithms for transformer-based language models. arXiv preprint arXiv:2307.06440, 2023.

Angelos Katharopoulos and Franc¸ois Fleuret. Not all samples are created equal: Deep learning with importance sampling. In International conference on machine learning, pp. 2525–2534. PMLR, 2018.

Conglong Li, Zhewei Yao, Xiaoxia Wu, Minjia Zhang, and Yuxiong He. Deepspeed data efficiency: Improving deep learning model quality and training efficiency via efficient data sampling and routing. arXiv preprint arXiv:2212.03597, 2022.

Xiaoxuan Liu, Lianmin Zheng, Dequan Wang, Yukuo Cen, Weize Chen, Xu Han, Jianfei Chen, Zhiyuan Liu, Jie Tang, Joey Gonzalez, et al. Gact: Activation compressed training for generic network architectures. In International Conference on Machine Learning, pp. 14139–14152. PMLR, 2022.

Ilya Loshchilov and Frank Hutter. Online batch selection for faster training of neural networks. arXiv preprint arXiv:1511.06343, 2015.

Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, and Hao Wu. Mixed precision training, 2018.

Soren Mindermann, Jan M Brauner, Muhammed T Razzak, Mrinank Sharma, Andreas Kirsch, Win-¨ nie Xu, Benedikt Holtgen, Aidan N Gomez, Adrien Morisot, Sebastian Farquhar, et al. Prioritized ¨ training on points that are learnable, worth learning, and not yet learnt. In International Conference on Machine Learning, pp. 15630–15649. PMLR, 2022.

Deanna Needell, Rachel Ward, and Nati Srebro. Stochastic gradient descent, weighted sampling, and the randomized kaczmarz algorithm. Advances in neural information processing systems, 27, 2014.

Mahdi Nikdan, Tommaso Pegolotti, Eugenia Iofinova, Eldar Kurtic, and Dan Alistarh. Sparseprop: Efficient sparse backpropagation for faster training of neural networks. arXiv preprint arXiv:2302.04852, 2023.

NVIDIA. Accelerating inference with sparsity using the nvidia ampere architecture and nvidia tensorrt. https://developer.nvidia.com/blog/ accelerating-inference-with-sparsity-using-ampere-and-tensorrt/, 2021.

Xu Ouyang, Shahina Mohd Azam Ansari, Felix Xiaozhu Lin, and Yangfeng Ji. Efficient model finetuning for text classification via data filtering. arXiv preprint arXiv:2207.14386, 2022.

Mansheej Paul, Surya Ganguli, and Gintare Karolina Dziugaite. Deep learning on a data diet: Finding important examples early in training. Advances in Neural Information Processing Systems, 34:20596–20607, 2021.

Md Aamir Raihan and Tor Aamodt. Sparse weight activation training. Advances in Neural Information Processing Systems, 33:15625–15638, 2020.

David E Rumelhart, Geoffrey E Hinton, and Ronald J Williams. Learning representations by backpropagating errors. nature, 323(6088):533–536, 1986.

Vatsal Shah, Xiaoxia Wu, and Sujay Sanghavi. Choosing the sample with lowest loss makes sgd robust. In International Conference on Artificial Intelligence and Statistics, pp. 2120–2130. PMLR, 2020.

Zhewei Yao, Xiaoxia Wu, Conglong Li, Connor Holmes, Minjia Zhang, Cheng Li, and Yuxiong He. Random-ltd: Random and layerwise token dropping brings efficient training for large-scale transformers. arXiv preprint arXiv:2211.11586, 2022.

Minghe Zhang, Chaosheng Dong, Jinmiao Fu, Tianchen Zhou, Jia Liang, Jia Liu, Bo Liu, Michinari Momma, Bryan Wang, Yan Gao, et al. Adaselection: Accelerating deep learning training through data subsampling. arXiv preprint arXiv:2306.10728, 2023.

Minjia Zhang and Yuxiong He. Accelerating training of transformer-based language models with progressive layer dropping. Advances in Neural Information Processing Systems, 33:14011– 14023, 2020.

Peilin Zhao and Tong Zhang. Stochastic optimization with importance sampling for regularized loss minimization. In international conference on machine learning, pp. 1–9. PMLR, 2015.

# A ABLATION ON HYPERPARAMETERS

There are a few hyperparameters in our self-adaptation algorithm, but all of them have explicit meaning. In this section we show that though extra tuning of these hyperparameters may achieve a slightly better result, overall VCAS is robust to these hyperparameters with reasonable values. We conduct ablation experiments on two tasks: BERT-base finetuning on SST-2 and MNLI. All the results are averaged over 3 different seeds.

# A.1 ACTIVATION AND WEIGHT VARIANCE THRESHOLDS $\tau _ { a c t } , \tau _ { w }$

The main hyperparameters in VCAS is the variance thresholds of activation $\tau _ { a c t }$ and weight $\tau _ { w }$ . For these two thresholds, how to split total variance among them is a big problem with optimal solution differing across models and tasks. So without prior knowledge introduced, we compromise by keeping $\tau _ { a c t } = \tau _ { w } = \tau \ll 1$ .

We further conduct an ablation on $\tau$ from 0.01 to 0.5 as is shown in Tab. 4 for SST-2 and Tab. 5 for MNLI. From the results we can find that a satisfactory outcome is assured regardless of the specific value of $\tau$ provided that $\tau \ll 1$ , which proves the robustness of VCAS.

Table 4: Ablation on different variance thresholds $\tau$ of BERT-base finetuning on SST-2   

<table><tr><td>τ</td><td>O(exact)</td><td>0.01</td><td>0.025</td><td>0.05</td><td>0.1</td><td>0.25</td><td>0.5</td></tr><tr><td>Final Train Loss</td><td>0.0559</td><td>0.0586</td><td>0.0600</td><td>0.0625</td><td>0.0642</td><td>0.0705</td><td>0.0761</td></tr><tr><td>Final Eval Acc(%)</td><td>92.59</td><td>93.07</td><td>93.04</td><td>93.25</td><td>92.81</td><td>92.79</td><td>92.18</td></tr><tr><td>FLOPs reduction(%)</td><td>-</td><td>45.92</td><td>48.28</td><td>49.82</td><td>50.05</td><td>51.57</td><td>52.71</td></tr></table>

Table 5: Ablation on different variance thresholds $\tau$ of BERT-base finetuning on MNLI   

<table><tr><td>τ</td><td>0(exact)</td><td>0.01</td><td>0.025</td><td>0.05</td><td>0.1</td><td>0.25</td><td>0.5</td></tr><tr><td>Final Train Loss</td><td>0.2372</td><td>0.2388</td><td>0.2428</td><td>0.2459</td><td>0.2552</td><td>0.2684</td><td>0.2805</td></tr><tr><td>Final Eval Acc(%)</td><td>84.33</td><td>84.31</td><td>84.23</td><td>84.33</td><td>84.07</td><td>84.13</td><td>84.08</td></tr><tr><td>FLOPs reduction(%)</td><td>-</td><td>38.59</td><td>41.56</td><td>43.49</td><td>45.37</td><td>47.53</td><td>48.92</td></tr></table>

# A.2 MONTE-CARLO REPETITIONS M

To calculate variances, VCAS introduces an overhead of extra iterations quadratic with Monte-Carlo repetitions $M$ .

![](images/2a6baa6abf0c0bb40148050c0854df0a6024e2fad3c370374aff938f378d03ca.jpg)  
Figure 7: Variance calculated with different Monte-Carlo samples $M$ of BERT-base finetuning on SST-2.

![](images/b0496cd73ec89d335740dfef9c7215bc22d9e89776b9e6318899334ab034d412.jpg)  
Figure 8: Variance calculated with different Monte-Carlo samples $M$ of BERT-base finetuning on MNLI.

Obviously bigger $M$ will bring more precise empirical variance, yet the cost is prohibitive.

We experiment on different $M$ from 2 to 10 and find no significant difference in the empirical variance as is shown in Fig. 7 for SST-2 and Fig. 8 for MNLI. Therefore, we adopted the setting of $M = 2$ , with which we only need to perform 6 extra iterations that is negligible if the variance calculation frequency is large enough like 100 in SST-2 and 500 in MNLI.

# A.3 VARIANCE CALCULATION FREQUENCY $F$

Similar to $M$ , the variance calculation frequency $F$ is also a trade-off between better empirical approximation and less overhead introduced. We experimented on $F = 5 0$ , 100, 200, 500, 1000 in Tab. 6 for SST-2 and Tab. 7 for MNLI. We can see that although as $F$ grows larger the overhead of VCAS is gradually relieved, with a too large $F$ , like $F = 1 0 0 0$ in SST-2 that leads to only 6 times of self-adaptation update, the sample ratio schedule is not fully explored and the final FLOPs reduction is even smaller. Therefore, for all these tasks we set $F$ to be at least $1 / 5 0$ of total training steps and no more than 500 due to slight marginal gains.

Table 6: Ablation on different adaptation frequency $F$ of BERT-base finetuning on SST-2, the number of training steps is 6315.   

<table><tr><td>F</td><td>0(exact)</td><td>50</td><td>100</td><td>200</td><td>500</td><td>1000</td></tr><tr><td>Final Train Loss</td><td>0.0559</td><td>0.0589</td><td>0.0600</td><td>0.0587</td><td>0.0577</td><td>0.0562</td></tr><tr><td>Final Eval Acc(%)</td><td>92.59</td><td>92.71</td><td>93.04</td><td>92.56</td><td>93.15</td><td>93.19</td></tr><tr><td>FLOPs reduction(%)</td><td>-</td><td>47.33</td><td>48.28</td><td>46.06</td><td>39.43</td><td>31.03</td></tr></table>

Table 7: Ablation on different adaptation frequency $F$ of BERT-base finetuning on MNLI, the number of training steps is 36816.   

<table><tr><td>F</td><td>O(exact)</td><td>50</td><td>100</td><td>200</td><td>500</td><td>1000</td></tr><tr><td>Final Train Loss</td><td>0.2372</td><td>0.2460</td><td>0.2461</td><td>0.2440</td><td>0.2428</td><td>0.2428</td></tr><tr><td>Final Eval Acc(%)</td><td>84.33</td><td>84.20</td><td>84.23</td><td>84.12</td><td>84.23</td><td>84.21</td></tr><tr><td>FLOPs reduction(%)</td><td>-</td><td>35.16</td><td>39.58</td><td>41.31</td><td>41.56</td><td>39.43</td></tr></table>

# A.4 $s$ UPDATE STEP $\alpha$ AND WEIGHT RATIO MULTIPLIER $\beta$

A simple grid search is conducted for $\alpha \in \{ 0 . 0 0 5 , 0 . 0 1 , 0 . 0 2 \}$ and $\beta \in \{ 0 . 9 5 , 0 . 9 , 0 . 8 \}$ in Fig. 9 for SST-2 and Fig. 10 for MNLI. From the figures, we can find that we are able to trade convergence for efficiency with a more aggressive setting of larger $\alpha$ and smaller $\beta$ , yet all results here are decent

with a final accuracy drop of no more than $0 . 3 \%$ for both tasks. Thus, VCAS is robust to different $\alpha$ and $\beta$ .

![](images/e275b2a12d42f013667dae5ab00c43c90d57ba15d67f3e9aea4dc3960d966ea4.jpg)  
Figure 9: Grid search of $s$ update step $\alpha$ and weight ratio multiplier $\beta$ of BERT-base finetuning on SST-2. The darker color the better.

![](images/f5e52be86d5b418e192905b45c98d6d0498347c1963897f259aedbaa80c08976.jpg)  
Figure 10: Grid search of $s$ update step $\alpha$ and weight ratio multiplier $\beta$ of BERT-base finetuning on MNLI. The darker color the better.

From all the ablation results above, we can see that VCAS is robust to all these hyperparameters with reasonable values, proving the insensitiveness.

# B INSIGHTS ON UPDATE OF $s$ , $\{ \rho _ { l } \}$ AND $\{ \nu _ { l } \}$

In this section, we will show how the gradient norm preserving ratio $s$ as well as all the sample ratios $\{ \rho _ { l } \}$ and $\{ \nu _ { l } \}$ update across the training.

We record the update process of BERT-base finetuning on MNLI with different variance tolerance thresholds $\tau$ as in Appendix. A.1. All results are averaged on three different seeds.

Fig. 11a depicts the update of $s$ . For non-decreasing $\{ \rho _ { l } \}$ , we plot the update of the first and the last values $\rho _ { 1 } , \rho _ { L }$ in Fig. 11b, with other values lying between. For $\{ \nu _ { l } \}$ , we show the update of the first three ones $\nu _ { 1 } , \nu _ { 2 } , \nu _ { 3 }$ in Fig. 11c and observe similar behavior of other weights.

It is seen in Fig. 11 that during training of BERT-base on MNLI, the gradient norm preserving ratio $s$ first decreases and then shows a slight downward trend. The activation sample ratios $\{ \rho _ { l } \}$ gradually decrease with an abrupt change between epochs due to the rapid decline of train loss caused by the lowered learning rate in the linear learning rate scheduler. The weight sample ratios $\{ \nu _ { l } \}$ first decrease and then fluctuate to match the change of activation sample ratios.

# C PERFORMANCE ON CNN

In Sec. 6, we mainly experiment with Transformer-based models and Adam optimizers. But the variance controlled adaptation depicted in Sec. 5 holds universally for any DNNs with SGD-based optimizers, since it just provides an approximated stochastic gradient with controlled variance to estimate the full gradient. In this section, we employ VCAS on other architectures and other optimizers to prove its versatility.

![](images/4ead9a7043eaa125bb0c9f9254f1b76b170ab04319d94bcc3c2762dd4389adac.jpg)  
Figure 11: VCAS update process with different $\tau$ for BERT-base finetuning on MNLI.

For CNN, it is noted that the weight sampler SampleW in Sec. 4 designed for linear layers is not usable for convolution layers. Thus we employ VCAS with a degraded version of activation sampling only.

We experiment with WideResNet-18 with widen factor $w = 4$ pretraining on ImageNet. We use eight NVIDIA 3090Ti to parallel the training with Distributed Data Parallel(DDP). We employ SGDM optimizer with momentum $m = 0 . 9$ . The results are in Tab. 8.

Table 8: Training results of WideResNet-18 pretraining on ImageNet with 8 NVIDIA 3090Ti.   

<table><tr><td>Method</td><td>Train Loss</td><td>Eval Acc(%) |</td><td>Train Time(h)</td><td>FLOPs↓(%)</td><td>| Time↓(%)</td></tr><tr><td>exact VCAS</td><td>1.474 1.479</td><td>75.96 75.86</td><td>21.31 20.20</td><td>17.47</td><td>- 5.21</td></tr></table>

From the table we can see VCAS is also capable of accelerating the training of CNN. Besides, the parallel setting also proves the parallelizability of VCAS. The relatively low but still decent time reduction can be explained with Amdahl’s Law since VCAS only accelerate the calculation part and is not able to accelerate other parts like communication cost during parallel training.

# D DETAILS ABOUT ALGORITHM. 1

It should be noted that some parts of Alg. 1 are simplified for clarity and we list the implementation details below:

In the algorithm table, we put the calculation of empirical variances out of the two Monte-Carlo loops for simplicity. Yet practically we can calculate $V _ { a c t }$ and $V _ { w }$ inside the loops and average the variance scalars outside. Therefore, we only need to store three tensors additionally regardless of $M$ : SGD gradient $G _ { s , i }$ to calculate $V _ { a c t }$ , and its running mean and running square mean to calculate $V _ { s }$ . By sampling only part of parameters to keep gradients, like $1 \%$ in our experiments, the memory overhead can be neglected.

Besides, since weight sample ratios $\{ \nu _ { l } \}$ are updated parameter-wise according to Eq. 7, the empirical weight variances and SGD variances are also stored parameter-wise when implemented.

Update of activation sample ratios $\{ \rho _ { l } \}$ requires finding out gradient sparsity $\{ p _ { l } \}$ with the new $s$ according to Eq. 4. In implementation, this is achieved by calculating possible new $\{ \rho _ { l } \}$ with both $s + \alpha$ and $s - \alpha$ inside the Monte-Carlo loops and averaging them outside. Then just choose the proper one with new $s$ .

# E PROOF

# E.1 PROOF TO UNBIASEDNESS OF VCAS

Let’s first consider a $L$ -layer MLP. (Note: for simplicity we mildly abuse the term ”layer” here, representing a single operation like matrix multiplication and ReLU)

For the last layer $L$ , the output gradient $\nabla _ { Z ^ { ( L ) } }$ is calculated from the loss directly, the same as the Exact BP. Since activation sampler $\hat { \nabla } _ { Z ^ { ( L ) } } = \mathrm { S a m p l e A } _ { \epsilon , \rho _ { L } } \left( \nabla _ { Z ^ { ( L ) } } \right)$ is unbiased, we have:

$$
\mathbb { E } \left[ \hat { \nabla } _ { Z ^ { ( L ) } } \right] = \nabla _ { Z ^ { ( L ) } }
$$

When back propagation proceeds, we may encounter two types of layers: linear and non-linear. For the linear layer, we have:

$$
\nabla _ { Z ^ { ( L - 1 ) } } = \hat { \nabla } _ { Z ^ { ( L ) } } \theta ^ { ( L ) }
$$

Thus unbiasedness is preserved with the output gradient of the $( L - 1 )$ -th layer:

$$
\begin{array} { r } { \mathbb { E } \left[ \nabla _ { Z ^ { ( L - 1 ) } } \right] = \mathbb { E } \left[ \hat { \nabla } _ { Z ^ { ( L ) } } \right] \theta ^ { ( L ) } = \nabla _ { Z ^ { ( L ) } } \theta ^ { ( L ) } = \mathrm { E x a c t ~ B P ~ r e s u l t } } \end{array}
$$

While for the non-linear layer like ReLU, we have:

$$
\nabla _ { Z ^ { ( L - 1 ) } } = \hat { \nabla } _ { Z ^ { ( L ) } } \odot J _ { Z ^ { ( L ) } }
$$

where $\odot$ is the Hadamard product and $J _ { Z ^ { ( L ) } }$ is the Jacobbi matrix determined by $Z ^ { ( L ) }$ which is saved in forward pass and is exact. Thus again we derive the the output gradient of the $( L - 1 )$ -th layer being unbiased:

$$
\mathbb { E } \left[ \nabla _ { Z ^ { ( L - 1 ) } } \right] = \mathbb { E } \left[ \hat { \nabla } _ { Z ^ { ( L ) } } \right] \odot J _ { Z ^ { ( L ) } } = \nabla _ { Z ^ { ( L ) } } \odot J _ { Z ^ { ( L ) } } = \operatorname { E x a c t } \operatorname { B P } \operatorname { r e s u l t }
$$

Thus by induction, VCAS assures all activation gradients $\hat { \nabla } _ { Z ^ { ( l ) } } , l = 1 \ldots L$ being unbiased.

Then for weight gradients, since weight sampler $\tilde { \nabla } _ { Z ^ { ( l ) } } = \mathrm { S a m p l e W } _ { \xi _ { l } , \nu _ { l } } \left( \hat { \nabla } _ { Z ^ { ( l ) } } , Z ^ { ( l - 1 ) } \right)$ is unbiased, we have:

$$
\mathbb { E } \left[ \tilde { \nabla } _ { Z ^ { ( l ) } } \right] = \mathbb { E } \left[ \hat { \nabla } _ { Z ^ { ( l ) } } \right] = \nabla _ { Z ^ { ( l ) } }
$$

Finally, we derive all weight gradients being unbiased:

$$
\mathbb { E } \left[ \nabla _ { \theta ^ { ( l ) } } \right] = \mathbb { E } \left[ \tilde { \nabla } _ { Z ^ { ( l ) } } \right] ^ { \top } Z ^ { ( l - 1 ) } = \nabla _ { Z ^ { ( l ) } } ^ { \top } Z ^ { ( l - 1 ) } = \mathrm { E x a c t ~ B P ~ r e s u l t }
$$

For more complicated neural networks like CNN and Transformer, since operations like convolutions and layernorm are all linear transforms, by similar reasoning the unbiasedness still holds.

# F EXPERIMENT DETAILS

# F.1 BERT-BASE PRETRAINING

For BERT-base pretraining we use a crammed BERT in Geiping & Goldstein (2022) with the recipe same as the original settings of 1 day training on a single NVIDIA 2080Ti. The full results are as follows in Tab. 9

From the table we can find that although VCAS achieves a relatively high train loss, the downstream task performance is still competent with exact training. While SB and UB both perform worse on CoLA, which is a vulnerable task, reflecting that they have changed the original convergence trajectory of SGD.

Table 9: Full results on BERT-base pretraining   

<table><tr><td>Methods</td><td>Loss</td><td>MNLI-m</td><td>MNLI-mm</td><td>QQP</td><td>QNLI</td><td>SST2</td><td>CoLA</td><td>STSB</td><td>MRPC</td><td>RTE</td><td>Avg.</td></tr><tr><td>exact</td><td>2.099</td><td>82.28</td><td>82.68</td><td>87.08</td><td>88.85</td><td>91.28</td><td>48.07</td><td>83.26</td><td>86.98</td><td>54.87</td><td>78.37</td></tr><tr><td>SB</td><td>2.133</td><td>82.34</td><td>82.86</td><td>87.27</td><td>88.63</td><td>91.28</td><td>41.82</td><td>82.86</td><td>85.53</td><td>55.23</td><td>77.53</td></tr><tr><td>UB</td><td>2.106</td><td>82.95</td><td>83.46</td><td>87.27</td><td>88.66</td><td>91.05</td><td>42.80</td><td>83.68</td><td>85.90</td><td>55.95</td><td>77.96</td></tr><tr><td>VCAS</td><td>2.134</td><td>82.03</td><td>82.82</td><td>86.92</td><td>89.23</td><td>91.62</td><td>48.36</td><td>83.02</td><td>86.03</td><td>55.23</td><td>78.36</td></tr></table>

# F.2 RECIPE OF OTHER TASKS

For BERT finetuning, we use AdamW optimizer with $l r = 2 e ^ { - 5 }$ and $w d = 0 . 0 1$ . The learning rate scheduler is a linear one with warmup ratio $= 0 . 1$ . We set epoch numbers $N = 3$ and a batch size of b $\substack { \scriptscriptstyle { 1 } t c h . s i z e = 3 2 }$ .

For ViT finetuning, we use Adam optimizer with $l r = 2 e ^ { - 5 }$ . A linear lr scheduler with no warmup employed. We run $N = 5$ epochs with batch size batch $_ { - s i z e } = 3 2$

# G LIMITATIONS

VCAS is designed for adaptively learning the proper sample ratios of large model training on large datasets. It is not suitable for small models with low gradient variances resulting in increased numerical errors, or small datasets with few training steps that is insufficient for the update process in VCAS.

The weight sampler SampleW in VCAS is specially designed for linear layers and is not usable for other operations like convolution. But the activation sampler SampleA can be applied to all mainstream architectures with deep layers. So for CNN or RNN, we need to employ a degraded version of VCAS with activation sampling only, as shown in Appendix. C.

VCAS focuses on mirroring the exact training with theoretical guarantee and is lack of exploration of other possible convergence trajectories that may bring a better result. Thus it is not recommended when the original training recipe is under-optimized.