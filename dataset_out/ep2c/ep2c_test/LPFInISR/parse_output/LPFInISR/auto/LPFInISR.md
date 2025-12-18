# Exploring the Low-Pass Filtering Behavior in Image Super-Resolution

Haoyu Deng 1 Zijing $\mathbf { X } \mathbf { u } ^ { 1 }$ Yule Duan 1 Xiao Wu 1 Wenjie Shu 1 Liang-Jian Deng 1

# Abstract

Deep neural networks for image super-resolution (ISR) have shown significant advantages over traditional approaches like the interpolation. However, they are often criticized as ‘black boxes’ compared to traditional approaches with solid mathematical foundations. In this paper, we attempt to interpret the behavior of deep neural networks in ISR using theories from the field of signal processing. First, we report an intriguing phenomenon, referred to as ‘the sinc phenomenon.’ It occurs when an impulse input is fed to a neural network. Then, building on this observation, we propose a method named Hybrid Response Analysis (HyRA) to analyze the behavior of neural networks in ISR tasks. Specifically, HyRA decomposes a neural network into a parallel connection of a linear system and a non-linear system and demonstrates that the linear system functions as a low-pass filter while the non-linear system injects high-frequency information. Finally, to quantify the injected highfrequency information, we introduce a metric for image-to-image tasks called Frequency Spectrum Distribution Similarity (FSDS). FSDS reflects the distribution similarity of different frequency components and can capture nuances that traditional metrics may overlook. Code, videos and raw experimental results for this paper can be found in: https://github.com/RisingEntropy/LPFInISR.

Please refer to Appx. A for notation conventions.

![](images/602cfc3bd844d288304344a35e7548a5b4ce3f04607892effc5a8519c72c1d4f.jpg)  
Figure 1. $I$ is an image in which only the central pixel is 1 and the other pixels are 0. What would the result look like if image I is super-resolved using a neural network, A, B, C, or D? Surprisingly, the answer is A. We name this phenomenon as the sinc phenomenon. In this paper, we give a possible explanation for this phenomenon.

ages through various techniques. In recent years, with advances in deep learning, growing ISR methods using neural networks are proposed, bringing the development of ISR into a new level. While impressive results persistently arise, the mechanism under ISR networks remain largely unexplored, leading to criticism that they are considered black boxes. In comparison, traditional methods, such as interpolation or filtering, have strong interpretability. Despite the principles of traditional methods and neural networks are different, we can still attempt to explain the behavior of ISR networks using theories from traditional methods. In this paper, following this line of thought, we successfully utilize theories from the field of signal processing techniques to explain the performance of neural networks in the ISR task.

The goal of image super-resolution (ISR) is to reconstruct low-resolution (LR) images into high-resolution (HR) im

# 1. Introduction

The target of the ISR task is to upsample a two-dimensional signal. In traditional signal processing theory (Oppenheim & Schafer, 2009; Oppenheim et al., 1996), a feasible method for upsample involves restoring a discrete low-samplingrate signal to a continuous signal using a low-pass filter, and then sampling the continuous signal at a higher rate to obtain a high-sampling-rate signal. An intriguing aspect of this process is that when we try to upsample a Dirac $\delta$ signal, we will finally get a sinc signal since the sinc signal is the time-domain waveform of a low-pass filter, (for details about this, please refer to Sec. 3 ). Given this, we can conjecture: if neural networks exhibit similar behavior, then when attempt to super-resolve a Dirac $\delta$ signal, the resultant outcome would also be a sinc signal. As shown in Fig. 1, we indeed observe this phenomenon, and we name it as ‘the sinc phenomenon’. This phenomenon establishes a connection between traditional signal processing theory and the interpretability of neural networks, thus helping us form a deeper understanding of ISR networks.

Building upon the sinc phenomenon, we further propose a method named HyRA1, which stands for Hybrid Response Analysis. HyRA considers the neural network as a parallel combination of a linear system and a non-linear system with a zero impulse response. It further indicates that this linear system functions as a low-pass filter, while the nonlinear system utilizes the learned prior knowledge to inject high-frequency information. By employing HyRA, we can analyze performance bottlenecks in neural networks, discerning whether the issue lies in inadequate preservation of low-frequency components or insufficient injection of highfrequency components. This analysis facilitates the proposal of targeted improvements for enhanced adaptability.

Given that the non-linear component is injecting highfrequency information, there is a pressing need for a metric to quantitatively describe the extent of the injected high frequencies. Previous metrics, like PSNR, SSIM (Wang et al., 2004) and LPIPS (Zhang et al., 2018a), have not approached the evaluation of images from a frequency perspective. Therefore, we propose the frequency spectrum distribution similarity (FSDS), a metric that evaluates image quality based on the power distribution in the frequency spectrum.

In summary, our contribution can be concluded as:

• We report an intriguing phenomenon: the impulse responses of image super-resolution (ISR) networks are sinc functions, representing the temporal waveform of a low-pass filter. We name it the ’sinc phenomenon’. This observation helps to establish a connection between signal processing theory and neural networks. Moreover, we find that for a network, the more similar the impulse response is to the sinc function, the better performance it produces.

• In order to further explain the performance of neural networks in the ISR task through this phenomenon, we introduce HyRA. HyRA considers the neural network as a parallel combination of a linear system and a nonlinear system with a zero impulse response. It points out that the linear system operates as a low-pass filter, while the non-linear system injects high-frequency

information.

• To quantitatively describe the injection of high frequencies, we introduce the FSDS metric. FSDS measures image quality using frequency spectrum produced by FFT and can reflect high-frequency distortions that previous metrics fail to capture.

# 2. Related works

# 2.1. Super Resolution Using Neural Networks

Recent review articles in ISR include fixed-scale superresolution (Yang et al., 2019) and arbitrary-scale superresolution review (Liu et al., 2023). There are various architectures of mainstream ISR backbone networks, including CNN-style backbones (Ahn et al., 2018; Hui et al., 2019; Lim et al., 2017; Zhang et al., 2018b;c), transformer-style backbones (Liang et al., 2021; Wang et al., 2023) and GANstyle backbone networks (Wang et al., 2018), etc. Based on these backbones, researchers have proposed quantitative modules with various functions. For example, ArbSR (Wang et al., 2021) can expand a fixed-scale super-resolution network to an arbitrary-scale ISR network, LTE (Lee & Jin, 2022) can enhance local textures, etc. What worth mentioning is that LIIF (Chen et al., 2021) introduces implicit neural representation into ISR for the first time, bringing a new approach for ISR. This paper mainly focuses on approaches that utilize CNN-style or transformer-style backbones. Except for network architectures, numerous datasets have been proposed to facilitate further research. Commonly used datasets for ISR includes Set5 (Bevilacqua et al., 2012), Urban100 (Huang et al., 2015), Flickr2K (Young et al., 2014), SCI1K (Yang et al., 2021), DIV2K (Agustsson & Timofte, 2017), etc. We evaluate the effectiveness of our proposed FSDS metric on DIV2K dataset. The large size of the DIV2K dataset contributes to increased reliability in our conclusions.

# 2.2. Explaining the Behavior of Neural Networks

Despite neural networks are often criticized as ‘black boxes,’ predecessors have made remarkable efforts to mitigate this situation. Various previous researches have proposed plenty of methods to analyze the behavior of neural networks. Since Sundararajan et al. (Sundararajan et al., 2017) introduce the integrated gradients (IG) for attribution in classification tasks, numerous researchers have expanded this method to various domains, broadening the scope of attribution beyond classification tasks. Based on IG, Gu & Dong (Gu & Dong, 2021) propose LAM to analyze the impact of the local patch on the entire ISR outcome. However, such a method requires manually determined hyper-parameters and baselines, thus introducing subjectivity. Several notable analysis methods utilizing the Fourier transform have been explored in the literature (Xu, 2018; 2020; Xu et al., 2019; Zhang et al., 2019). Notably, Xu et al. (Xu et al., 2019) propose the Frequency-Principle, claiming its relevance to both convolutional neural networks (CNNs) and fully-connected deep neural networks. According to their proposition, these networks inherently adhere to the Frequency-Principle, wherein training data is systematically acquired in a sequential manner, progressing from low to high frequency. Unlike previous approaches these approaches, HyRA distinguishes itself by employing impulse response to probe the potential mechanisms of deep neural networks in the context of the ISR task.

# 3. Preliminaries

Appx. B.1-Appx. B.3 provide a brief overview of signal processing concepts for readers who are not familiar with it. Appx. B.1 introduces the concepts of signals and systems, along with the computation of responses in Linear TimeInvariant (LTI) systems. Appx. B.2 covers the processes of signal sampling and reconstruction. Appx. B.3 delves into the phenomenon of spectrum aliasing, a factor contributing to the ill-posed nature of the ISR task. And we will introduce applying low-pass filter for ISR here.

We can employ signal recovery methods to achieve image super-resolution (ISR). Initially, we conceptualize an image as a series of impulse trains in a two-dimensional continuous space, with varying densities representing different resolutions. Then, for the low-resolution image, we begin by implementing low-pass filtering, following the procedure outlined in Appx. B.2, to obtain the continuous image $I ^ { \mathrm { c o n t } }$ , This process can be mathematically described as:

$$
I _ { x , y } ^ { \mathrm { c o n t } } = s i n c _ { x , y } ^ { \omega } * I _ { x , y } ^ { \mathrm { L R } } ,
$$

where $^ *$ denotes convolution, $I _ { x , y } ^ { L R }$ is the low resolution with variant is a two-di $x , y$ and iona $I _ { x , y } ^ { \mathrm { c o n t } }$ is the continuous signal. function with parameter $s i n c _ { x , y } ^ { \omega }$   
$\omega ^ { 2 }$ , whose frequency spectrum is an ideal low-pass filter with a passband of $0 \sim \omega$ . Subsequently, we sample the ‘conceptually continuous signal’ at an elevated sampling rate to acquire a more densely populated two-dimensional sequence of impulse trains, i.e., an image with higher resolution denoted as ISR:

$$
I _ { x , y } ^ { S R } = I _ { x , y } ^ { \mathrm { c o n t } } \cdot s _ { x , y } ^ { \Delta X , \Delta Y } .
$$

In the equation, pulse trains wit $s _ { x , y } ^ { \Delta X , \Delta Y }$ denls of hein o-dimenaxis and l iin $\Delta X$ $x$ $\Delta Y$ $y$ axis.

In fact, commonly used interpolation kernels for ISR, such as nearest-neighbor interpolation, linear interpolation, cubic interpolation, etc., can be seen as approximations of the sinc function considering a balance between computational complexity and effectiveness, as illustrated in Fig. 2. Taking into account the similarity of these interpolation kernels, in this paper, we collectively refer to these parameter-free methods as low-pass filter-based super-resolution methods.

![](images/604486344468fb162c17394e7f75db068255b85b39ab6e50d15527c7b512291e.jpg)  
Figure 2. Various interpolation kernels for ISR. They can all be seen as an approximation of sinc function.

# 4. Method

# 4.1. Hybrid Response Analysis (HyRA)

In this section, we describe the proposed Hybrid Response Analysis (HyRA), which treats the neural network as a combination of a linear system and a non-linear system. Through the impulse response, we can calculate a linear time invariant (LTI) system’s output from any input using the convolution operation (see Appx. B.1). However, since neural networks are nonlinear systems, we cannot apply convolution to analyze them. To further explore the network features, we need to split it into a linear system and a non-linear system, i.e., HyRA. The core concept HyRA is illustrated in Fig. 3. We denote an ISR network as $N ( I )$ , where $I$ is the input image. $N ( I )$ is a non-linear system that can be expressed as the sum of a linear system and a non-linear system:

$$
N ( I ) = H ( I ) + G ( I ) .
$$

![](images/da4699e9c5ecf2574bb03cad46d4ca9169b209124ff455c8047334f17207b17d.jpg)  
Figure 3. Conceptual diagram of HyRA’s core idea.

In the equation, $H ( I )$ represents a linear system, and $G ( I )$ represents a non-linear system. Without constraints, such a representation is meaningless because $H ( I )$ can be arbitrarily chosen, leading to an infinite variety of representations with the same form but different meanings. To give meaning to this representation, we introduce a constraint: the impulse response of $G ( I )$ is zero. With this constraint, both $H ( I )$ and $G ( I )$ can be uniquely determined. Lemma 4.1 demonstrates that under this constraint, $N ( I )$ can still be expressed in the form of Eq. 3. This straightforward method is the essence of HyRA.

For the ISR task, there is a distinctive property known as a ‘spatially invariant system’ (Miller et al., 1992) associated with it. Consider the definition of time-invariant systems as mentioned in Appx. B.1, we can naturally extend the concept of in-variance from one-dimensional to twodimensional space and the definition of spatially invariant systems is: when the input is $I _ { x , y }$ , the output is $G ( I _ { x , y } ) =$ $O ( x , y )$ ; when the input becomes $I ^ { \prime } = I _ { x - x _ { 0 } , y - y _ { 0 } }$ , the output should be $G ( I ^ { \prime } ) = O ( x - x _ { 0 } , y - y _ { 0 } )$ . For convolution based architectures, we can easily prove its spatial invariance (see the proof below). For transformer-based architectures, we can still use experiments to prove the spatial invariance (see Fig. 15).

Proof. A convolution operation can be defined as:

$$
C o n v _ { i , j } = \sum _ { p , q } I _ { i - p , j - 1 } K _ { p , q } .
$$

Then, the shifting operation can be defined as:

$$
\mathrm { S h } ( i , j )  ( i + k , j + l ) .
$$

Combine these two, we then have:

$$
\begin{array} { l } { { \displaystyle C o n v _ { \mathrm { S h } ( i , j ) } = C o n v _ { i + k , j + l } } } \\ { ~ = \displaystyle \sum _ { p , q } I _ { i + k - p , j + l - q } K _ { p , q } } \\ { ~ = \mathrm { S h } ( \displaystyle \sum _ { i - p , j - q } K _ { p , q } ) } \\ { ~ = \mathrm { S h } ( C o n v _ { i , j } ) . } \end{array}
$$

This is the invariance of a single convolution layer, and still holds for more layers.

According to HyRA, when we input a Dirac $\delta$ signal to the neural network, we can get the impulse response of the linear system (please recall Appx. B.1), denoted as $H ( \delta )$ For any input $I$ , the response of the linear space invariant system can be obtained by convolving the input with the obtained impulse response, which can be expressed as:

$$
H ( I ) = I * H ( \delta ) ,
$$

where $^ *$ means the convolution operation. Although the response of the non-linear component cannot be directly computed, if we obtain the final output of the neural network, the non-linear part can be deduced by subtracting the response of the linear component from the final output, namely the non-linear response can be computed as:

$$
\begin{array} { c } { { G ( I ) = N ( I ) - H ( I ) } } \\ { { = N ( I ) - I * H ( \delta ) . } } \end{array}
$$

Lemma 4.1. A neural network $N ( I )$ can be expressed as a combination of a linear system $H ( I )$ and a non-linear system with an impulse response of zero, i.e., $N ( I ) \ =$ $H ( I ) + G ( I )$ , where $G ( \delta ) = 0$ . Here, $\delta$ represents the Dirac delta function.

1) When $G ( \delta ) = 0$ , the conclusion holds.

2) When $G ( \delta ) \neq 0$ , Let $H _ { 1 } ( I ) = H ( I ) + G ( \delta ) * I$ and $G _ { 1 } ( I ) = G ( I ) - G ( \delta ) * I$ . In this case, $H _ { 1 } ( I )$ remains a linear system and $G ^ { \prime } ( I )$ remains a non-linear system. The equation $N ( I ) = H _ { 1 } ( I ) + G _ { 1 } ( I )$ holds, and it satisfies $G _ { 1 } ( \delta ) = 0$ . □

# 4.1.1. $H ( I )$ IS A LOW-PASS FILTER

In Sec. 3, we mention that a simple low-pass filter achieves ISR functionality. Do neural networks possess low-pass filters internally? If this hypothesis is valid, according to the principle of HyRA, when we input a Dirac $\delta$ signal into the neural network $N ( I )$ , the output should be the impulse response of the low-pass filter, i.e., the sinc function (please recall Appx. B.1 and Tab. 3). In the experiment section (Sec. 5.2), we conduct tests on three mainstream ISR backbones and some derived methods. We find that their impulse responses are sinc functions3. Now, with both the impulse response and spatial invariance property, we can compute the response of the linear system $H ( I )$ to any input through convolution:

$$
\begin{array} { l } { \displaystyle H ( I ) _ { x , y } = I _ { x , y } * H ( \delta ) } \\ { \displaystyle \qquad = \int \int _ { ( \tau , u ) \in \mathbb { R } ^ { 2 } } I _ { \tau , u } H ( \delta ) _ { x - \tau , y - u } \mathrm { d } \tau \mathrm { d } u . } \end{array}
$$

In a practical scenario, when dealing with a two-dimensional impulse array represented by $I$ , the integration process can be effectively substituted with summation, incorporating appropriate padding. Despite the convolution operator in PyTorch (Paszke et al., 2019) being inherently a correlation operator, the symmetric nature of the sinc function allows for its seamless utilization within such an operator. We present a toy example in Fig. 4 in which we compute the response of the linear component of the EDSR network (Lim et al., 2017) during ISR. Observing the experimental results, we notice that the linear function $H ( I )$ essentially achieves super-resolution, but there are some issues: edge blurring and the presence of grid-like distortions.

![](images/9fd816b806fc54baa7934d7d4b9da2a30f580e26ddadf46ab3496c3b338f224c.jpg)  
Figure 4. Top row: a super-resolved image by (Lim et al., 2017) can be viewed as the summation of a linear response obtained by convolving impulse response with the input and the non-linear response gained by subtracting linear-part from the ISR result. Second row: the corresponding frequency spectrum amplitude of the top row. Third row: the corresponding frequency spectrum phase of the top row. The phase compensation indicates that the non-linear part is compensating distortion.

The edge is blurred because the low-pass filter removes some high-frequency details. In the frequency spectrum, it is manifested as a relatively small range of diffusion of the central bright spot towards the surroundings. This implies that the image has more low-frequency components and fewer high-frequency components. Such an outcome is the inevitable consequence of applying the low-pass filter.

When computing the response of the linear system, we first perform zero-interpolation on the low-resolution image to achieve the target spatial size. This operation leads to periodic extension in the frequency spectrum4. Since this low-pass filter is not a complete ideal filter, but an ideal filter truncated by a certain window function, its filtering performance is weakened by the window function. The weakened filter cannot completely eliminate the extended spectrum, meaning the attenuation in the stopband is insufficient, as referred to in signal processing, thus causing such gird-like distortions.

![](images/51f9f7cc1d89924608056bb60c155db262f4f55a4cff2e28ef1d0038eab17430.jpg)  
Figure 5. An illustration of how the passband width of a low-pass filter affects its ISR results. A too wide passband or a too narrow passband can result in a decline in performance.

In summary, the linear system $H ( I )$ (the low-pass filter approximated by the neural network) can achieve superresolution functionality, but it is not perfect. On one hand, the low-pass filter determines that the image is blurred, lacking high frequencies. On the other hand, the filter is windowed, leading to a weakened filtering performance and resulting in grid-like distortions. These issues will be compensated for by the nonlinear system $G ( I )$ .

# 4.1.2. $G ( I )$ INJECTS HIGH-FREQUENCY INFORMATION

Though a low-pass filter can achieve ISR (please refer to Sec. 3), its performance can never surpass a well-trained neural network. The outcome of a low-pass filter varies with respect to the passband width, as depicted in Fig. 5. However, information outside the passband will be completely wiped out, causing an observable detail loss in high-frequency components. On the contrary, the non-linear part of neural networks is able to inject information in high-frequency domain based on learned or structural priors. Moreover, it can compensate the grid-like distortions brought by the windowed low-pass filter. Together with the linear part, neural networks function as the superset of low-pass filter, retaining both high and low frequency information.

We compute the non-linear response and its frequency spectrum of the neural network using the proposed HyRA paradigm. In the toy example presented in Fig. 4, it can be noticed that the response of the non-linear component exhibits sharper edges. Compared with the frequency spectrum of the ISR results, the central bright spot in the response of $G ( I )$ spreads to a larger range, indicating that more power is distributed into the high-frequency domain. Almost all the components of the high-frequency part in the final ISR result are contributed by the non-linear component.

As mentioned in Sec. 4.1.1, the non-linear component also plays a crucial role in compensating for the distortion introduced by $H ( I )$ . Examining the response of $G ( I )$ , we note that it also exhibits grid-like distortions, matching those in the response of $H ( I )$ . This allows for the cancellation of the grid-like distortions, achieving the final goal of ISR. As shown in Fig. 4, upon observing the frequency spectrum, bright spots corresponding to the amplitude spectrum of $H ( I )$ exist in all four corners of the amplitude spectrum of $G ( I )$ . However, the phase spectrum of $G ( I )$ is in compensation of the phase spectrum of $H ( I )$ , indicating that the grid-like distortion is ‘erased’ here.

In summary, the non-linear component $G ( I )$ serves to inject high-frequency details learned during training to compensate for the loss of high frequencies introduced by the lowpass filter. Simultaneously, it addresses distortions arising from the imperfect performance of the low-pass filter.

# 4.2. Frequency Spectrum Distribution Similarity (FSDS)

In this section, we introduce the FSDS metric to quantitatively describe the so called the ‘injected high frequencies’ as discussed in Sec. 4.1.2.

# 4.2.1. MOTIVATION AND METHOD

![](images/f95ffbb7fd0b495af6b408f35122b336eef750aec098c6065fae0f0b69e07e9f.jpg)  
Figure 6. X-FFT- $\Sigma$ denotes the integrated frequency spectrum, the integration path is from origin to infinty in every quadrant. Columns 1 and 2 in the figure respectively show that the differences in the results of different ISR methods can be reflected in the frequency spectrum. Column 3 presents the integral of the spectrum from low to high frequencies in a contour plot. The distribution of contours visually represents the distinct distribution of different frequency components.

Since we need to measure the components of injected high frequencies, we must delve into the issue from a frequency spectrum perspective. However, commonly used metrics such as PSNR, SSIM (Wang et al., 2004), and LPIPS (Zhang et al., 2018a) do not measure the quality of an image from a spectral perspective.

Additionally, we’ve noted that the frequency domain distribution in the ISR field can significantly impact downstream applications $\mathrm { X u }$ et al., 2020; Yu et al., 2023). Consequently, we propose that evaluating the ISR effectiveness of a network requires a thorough assessment of its performance in the frequency spectrum. This involves examining the similarity in frequency spectrum between the low-resolution image and the high-resolution image. The Frequency Spectrum Distribution Similarity (FSDS) metric integrates the power distribution maps of the spectrum for both images. The difference is then calculated to generate an error map, and the total sum of its absolute values is computed.

For an image $I _ { x , y } ^ { \mathrm { H R } }$ , to minimize the impact of the data input range on the results, we normalize the input data and then perform a two-dimensional Fourier transform to obtain $I _ { j \omega _ { 1 } , j \omega _ { 2 } } ^ { \mathrm { H R } }$ , which can be mathematically described as:

$$
I _ { j \omega _ { 1 } , j \omega _ { 2 } } ^ { \mathrm { H R } } = \mathcal { F } \left[ \frac { I ^ { \mathrm { H R } } - E ( I ^ { \mathrm { H R } } ) } { \sigma ( I ^ { \mathrm { H R } } ) } \right] ,
$$

where $E ( I ^ { \mathrm { H R } } )$ and $\sigma ( I ^ { \mathrm { H R } } )$ are the mean value and variance of $I ^ { \mathrm { H R } }$ respectively. Similarly, we perform a Fourier transform on the ISR image to obtain $I _ { j \omega _ { 1 } , j \omega _ { 2 } } ^ { \mathrm { S R } }$ . It is worth noting that unlike other metrics, such as PSNR and SSIM(Wang et al., 2004), which do not incorporate normalization, FSDS is specifically designed to accentuate numerical variations due to its emphasis on numerical changes rather than absolute numerical values. Then, the complex integration of the two spectrum is performed, providing the power distribution map $\bar { D } ^ { \mathrm { H R } }$ , which is defined as:

$$
D ^ { \mathrm { H R } } = \iint _ { ( \omega _ { 1 } , \omega _ { 2 } ) \in \mathbb { R } ^ { 2 } } I ^ { \mathrm { H R } } \mathrm { d } \omega _ { 1 } \mathrm { d } \omega _ { 2 } .
$$

Similarly, we can obtain $D ^ { S R }$ . Subsequently, the difference between $D ^ { H R }$ and $D ^ { S R }$ is calculated, providing a difference map $D ^ { \mathrm { d i f f } }$ of their power distribution:

$$
\begin{array} { r } { D ^ { \mathrm { d i f f } } = D ^ { \mathrm { H R } } - D ^ { \mathrm { S R } } . } \end{array}
$$

Finally, we define the frequency spectrum distribution similarity (FSDS) as:

$$
\mathrm { F S D S } = - 1 0 \log _ { 1 0 } \frac { \iint _ { ( \omega _ { 1 } , \omega _ { 2 } ) \in \mathbb { R } ^ { 2 } } | D ^ { \mathrm { d i f f } } | ^ { 2 } \mathrm { d } \omega _ { 1 } \mathrm { d } \omega _ { 2 } } { \iint _ { ( \omega _ { 1 } , \omega _ { 2 } ) \in \mathbb { R } ^ { 2 } } | D ^ { \mathrm { H R } } | ^ { 2 } \mathrm { d } \omega _ { 1 } \mathrm { d } \omega _ { 2 } } ,
$$

where $| \cdot |$ represents taking the magnitude of a complex number. Considering a more concise description of a larger dynamic range, logarithm is taken. A larger FSDS value indicates that the two images are closer, thereby suggesting better ISR results.

![](images/fd0b48a695c1f9db92a4ad956e14f055fa828e278505de3d3276386742b35a20.jpg)  
Figure 7. A comparison of SSIM and FSDS in JPEG compression and steganography. As can be seen, SSIM fails to reflect distortion brought by steganography, while FSDS captures both cases of distortion.

# 4.2.2. THE MERITS OF FSDS

Previous image evaluation metrics, such as PSNR, SSIM (Wang et al., 2004), have focused on statistical or structural features of images, but no work has evaluated images from the perspective of their frequency spectrum. The spectrum is the concentrated expression of components with different changing rates in a signal or image. It is crucial for capturing details, eliminating noise, and comprehensively understanding image features. In image processing, spectrum analysis provides a more accurate evaluation, particularly playing a key role in applications sensitive to details. Due to the nature of Fourier transformation, which involves every pixel of the image in the computation, it encompasses not only information such as signal-to-noise ratio and structural similarity but also the overall similarity of the entire image. Therefore, evaluating image quality from the perspective of the spectrum is highly reasonable and necessary. Our FSDS metric can reflect distribution differences by employing a paradigm of integrating first in the frequency spectrum and then comparing. In other words, FSDS not only reflects the signal-to-noise ratio captured by the PSNR metric and the structural similarity indicated by the SSIM metric, but also captures features that these two metrics cannot represent. In the next paragraph, we will use two toy examples to demonstrate the rationale and advantages of FSDS.

From Fig. 6, it can be observed that images obtained by different ISR methods have different proportions of highfrequency components (the center of the spectrum figure represents low frequency, while higher frequencies extend outward). After integration, this is reflected in the varying widths of the dark cross-shaped patterns in the center. A narrower width indicates a higher proportion of low-frequency components in the spectrum, and vice versa. Existing methods may not effectively capture the loss of high-frequency components with low power in the frequency spectrum. Performing information steganography in the frequency spectrum can effectively highlight this aspect. As shown in Fig. 7, we embed some content in the frequency spectrum of the image. Such steganography causes our FSDS metric to drop to $2 6 . 3 7 \mathrm { d B }$ while the SSIM metric remains in a high level of 0.995. we can observe that after applying specific steganography to the spectrum of an image, the image exhibits some blurring and oscillation. Such oscillations are actually the Gibbs phenomenon, a typical oscillation phenomenon caused by the loss of high-frequency information. Meanwhile, when we apply JEPG compression to the im$\mathsf { a g e } ^ { 5 }$ , when FSDS drops to 26.39dB, SSIM together drops to 0.842. This toy example demonstrates that there indeed exists some feature SSIM cannot reflect while that can be reflected by FSDS.

In summary, previous methods may not effectively reflect the situation in the image frequency spectrum, while our proposed FSDS metric can sensitively detect distortions in the frequency spectrum.

# 5. Experiments

Due to the page limitation, we can only present three of the most crucial experiments in this section, namely: 1) the relationship between the low-pass filter passband width and ISR performance; 2) various network impulse responses; 3) a comparison of FSDS metrics with PSNR, SSIM and LPIPS (Zhang et al., 2018a) metrics on the DIV2K dataset. For more experiments, please refer to Appx. C.

![](images/66495115fa4d1483441ed94771778112b8167fa9a31da1a8b1d13a32b6348f44.jpg)  
Figure 8. The ISR performance using a low-pass filter shows variations with the cutoff frequency $\omega$ . This figure illustrates the results obtained from the $\times 2$ ISR task conducted on the DIV2K dataset. To enhance the clarity of the visualization, the curve has been smoothed using a moving average with a window length of 10.

# 5.1. Experiment on Low-pass Filtering Super-resolution Performance

In Sec. 4.1.2, we mention that a vanilla low-pass filter can achieve ISR, we now present an experiment on the relationship between the low-pass filter passband width and ISR performance. As shown in Fig. 8, we utilized various lowpass filters to perform $\times 2$ ISR on the validation set from of DIV2K dataset. Subsequently, we evaluated the ISR results using the PSNR and SSIM metrics. When $\omega = 4 8$ , PSNR reaches its maximum value of 31.40. When $\omega \ = \ 4 5 . 8$ SSIM reaches its maximum value of 0.87. We assert that, in terms of neural network performance, for $\times 2$ ISR, the PSNR should not fall below 31.40, and the SSIM should not be lower than 0.87. Otherwise, it can be considered that the neural network may not effectively capture both low-frequency and high-frequency information.

# 5.2. Experiment on Impulse Response

We select several mainstream backbones and their derivatives commonly used for the ISR task (Chen et al., 2021; Hu et al., 2019; Lee & Jin, 2022; Liang et al., 2021; Lim et al., 2017; Song et al., 2023; Wei & Zhang, 2023; Zhang et al., 2018b;c) and conduct impulse response tests. The experimental results are compared with the sinc function and depicted in Fig. 9. The input image is an $1 1 \times 1 1$ image where only the pixel at position $( 5 , 5 )$ is white (the values for all three channels at this position are 255, with indices starting from 0), and the rest of the image is black (with values of 0). According to Tab. 3, it can be observed that as the ISR factor increases, the central peak of the output sinc function becomes wider and more pronounced. To balance visual saliency and the maximum ISR factor achievable by certain networks, we opted for a $4 \mathbf { x }$ ISR factor. Observing the experimental results, we can notice that regardless of the neural network structure used for ISR, whether it’s a CNN or a transformer, the impulse response exhibits some degree of similarity to the two-dimensional sinc function. This similarity is particularly pronounced in networks like RDN (Zhang et al., 2018c) and RCAN (Zhang et al., 2018b). Despite some distortion in comparison to the sinc function, EDSR (Lim et al., 2017), EQSR (Wang et al., 2023), and their derivatives still exhibit significant features of the sinc function, including the central bright spot and elongated bright patches in the cardinal directions. From Tab. 1, we observe that networks exhibiting superior performance tend to generate impulse responses that closely resemble the sinc function. This observation suggests that preserving lowfrequency information more effectively can also enhance performance. However, few previous works has focus on low-frequency, giving us a new idea for furture ISR networks.

# 5.3. Experiment on FSDS Metric

We conducted tests on the validation set of the DIV2K dataset (Agustsson & Timofte, 2017) using several methods (Chen et al., 2021; Lee & Jin, 2022; Liang et al., 2021; Lim et al., 2017; Song et al., 2023; Wei & Zhang, 2023; Yang et al., 2021; Zhang et al., 2018c), depicted in Tab. 1. The evaluation metrics include PSNR, SSIM (Wang et al., 2004), LPIPS (Zhang et al., 2018a), and our FSDS. All tests are performed using code and weights available in open-source official repositories. For all methods, we conduct experiment for $\times 2$ to $\times 4$ . For methods that support arbitrary-scale ISR, we test for $\times 6$ and $\times 1 2$ as well. In the $\times 2$ to $\times 4$ range, GRLBase(Li et al., 2023) consistently achieves the best performance across PSNR, SSIM and LPIPS metrics, and FSDS shows that SwinIR (Liang et al., 2021) achieves the best performance. For $\times 6$ and $\times 1 2$ , RDN-LTE (Lee & Jin, 2022) exhibits the best PSNR and SSIM metrics, while RDN-LIIF performs best on LPIPS and the FSDS metric.

From Sec. 4.2, we claim that previous metrics are not sensitive to high-frequency information, while FSDS does. This can be proven by Tab. 1. In the case of slight high-frequency loss, such as on scales $\times 2$ to $\times 4$ , FSDS responds differently compared to previous metrics. In cases that suffer from severe high-frequency loss, such as on $\times ~ 6$ and $\times$ 12 scales, FSDS shows consistency with previous metrics. This is because when high-frequency loss is slight, previous metrics fail to reflect such high-frequency loss and while the loss becomes more severe, they start to capture such loss.

![](images/4de26d544961c741f7ac7987c5df6dd8036f100ef81e1b701f0f5b806393310a.jpg)  
Figure 9. Comparison of impulse responses and the sinc function for several mainstream backbone networks and their derivatives. The impulse response of the bicubic interpolation result is presented as a reference.

<table><tr><td rowspan="2">Method</td><td colspan="4">PSNR</td><td colspan="4">SSIM</td><td colspan="4"></td><td colspan="4"></td><td colspan="4">FSDS (Ours)</td></tr><tr><td>×2</td><td>×3</td><td>×4</td><td>×6</td><td>×12</td><td>×2</td><td>×3</td><td>×4</td><td>×6</td><td>×12</td><td>×2</td><td>×3</td><td>×4</td><td>×6</td><td>×12</td><td>×2</td><td>×3</td><td>×4</td><td>×6</td><td>×12</td></tr><tr><td>EDSR(Lim et al., 2017)</td><td>34.6312</td><td>30.9514</td><td>28.8716</td><td></td><td></td><td>0.93712</td><td>0.87413</td><td>0.81616</td><td></td><td></td><td>0.04212</td><td>0.10114</td><td>0.15516</td><td></td><td></td><td>39.2115</td><td>34.1510</td><td>31.387</td><td></td><td></td></tr><tr><td>EDSR-LIIF(Lim et al., 2017)</td><td>34.5514</td><td>30.9215</td><td>28.9815</td><td>26.764</td><td>23.754</td><td>0.93715</td><td>0.87414</td><td>0.81914</td><td>0.7414</td><td>0.6334</td><td>0.04315</td><td>0.10013</td><td>0.15313</td><td>0.2434</td><td>0.4282</td><td>39.3713</td><td>34.536</td><td>31.329</td><td>28.454</td><td>23.153</td></tr><tr><td>EDSR-OPESR(Lim et al., 2017)</td><td>34.3416</td><td>30.9612</td><td>29.0412</td><td></td><td></td><td>0.93616</td><td>0.87511</td><td>0.82013</td><td></td><td></td><td>0.04313</td><td>0.10011</td><td>0.15314</td><td></td><td></td><td>39.806</td><td>34.635</td><td>31.2911</td><td></td><td></td></tr><tr><td>EDSR-SRNO(Lim et al., 2017)</td><td>34.729</td><td>31.0510</td><td>29.1310</td><td>26.903</td><td>23.873</td><td>0.9399</td><td>0.87610</td><td>0.82211</td><td>0.7463</td><td>0.6383</td><td>0.0418</td><td>0.09810</td><td>0.14910</td><td>0.2413</td><td>0.4374</td><td>39.5311</td><td>34.537</td><td>31.456</td><td>28.463</td><td>22.784</td></tr><tr><td>EDSR-LTE(Lim et al., 2017)</td><td>34.6113</td><td>30.9711</td><td>29.0314</td><td></td><td></td><td>0.93714</td><td>0.87412</td><td>0.82012</td><td></td><td></td><td>0.04314</td><td>0.10012</td><td>0.15212</td><td></td><td></td><td>39.2914</td><td>34.338</td><td>31.3010</td><td></td><td></td></tr><tr><td>RDN(Zhang et al., 2018c)</td><td>34.6910</td><td>30.5816</td><td>29.1211</td><td></td><td></td><td>0.93810</td><td>0.86716</td><td>0.82310</td><td></td><td></td><td>0.0419</td><td>0.10616</td><td>0.15011</td><td></td><td></td><td>40.023</td><td>32.9516</td><td>31.654</td><td></td><td></td></tr><tr><td>RDN-LIIF(Zhang et al., 2018c)</td><td>34.868</td><td>31.218</td><td>29.269</td><td>26.992</td><td>23.932</td><td>0.9398</td><td>0.8798</td><td>0.8269</td><td>0.7492</td><td>0.6392</td><td>0.04110</td><td>0.0968</td><td>0.1478</td><td>0.2311</td><td>0.4061</td><td>39.699</td><td>34.833</td><td>31.833</td><td>28.891</td><td>23.781</td></tr><tr><td>RDN-OPESR(Zhang et al., 2018c)</td><td>34.5215</td><td>31.199</td><td>29.288</td><td></td><td></td><td>0.93811</td><td>0.8799</td><td>0.8268</td><td></td><td></td><td>0.04211</td><td>0.0967</td><td>0.1489</td><td></td><td></td><td>40.192</td><td>34.962</td><td>31.485</td><td></td><td></td></tr><tr><td>RDN-LTE(Zhang et al., 2018c)</td><td>34.917</td><td>31.267</td><td>29.317</td><td>27.051</td><td>23.991</td><td>0.9397</td><td>0.8797</td><td>0.8277</td><td>0.7501</td><td>0.6411</td><td>0.0417</td><td>0.0956</td><td>0.1446</td><td>0.2332</td><td>0.4313</td><td>39.825</td><td>34.754</td><td>31.842</td><td>28.652</td><td>23.352</td></tr><tr><td>SwinIR-classical(Liang et al., 2021)</td><td>35.345</td><td>31.645</td><td>29.634</td><td></td><td></td><td>0.9435</td><td>0.8855</td><td>0.8355</td><td></td><td></td><td>0.0385</td><td>0.0924</td><td>0.1405</td><td></td><td></td><td>40.371</td><td>35.131</td><td>32.371</td><td></td><td></td></tr><tr><td>ITSRN(Yang et al., 2021)</td><td>32.6717</td><td>30.4917</td><td>28.7317</td><td>26.645</td><td>23.725</td><td>0.92217</td><td>0.86617</td><td>0.81317</td><td>0.7365</td><td>0.6305</td><td>0.05217</td><td>0.11317</td><td>0.16717</td><td>0.2715</td><td>0.4695</td><td>31.2518</td><td>26.1818</td><td>25.8818</td><td>25.625</td><td>21.575</td></tr><tr><td>HAT-S(Chen et al., 2023)</td><td>35.462</td><td>31.723</td><td>29.723</td><td></td><td></td><td>0.944²2</td><td>0.8873</td><td>0.8373</td><td></td><td></td><td>0.0382</td><td>0.0923</td><td>0.1393</td><td></td><td></td><td>39.787</td><td>33.8013</td><td>31.0614</td><td></td><td></td></tr><tr><td>HAT(Chen et al., 2023)</td><td>35.462</td><td>31.772</td><td>29.752</td><td></td><td></td><td>0.9442</td><td>0.8872</td><td>0.837²2</td><td></td><td></td><td>0.0382</td><td>0.0902</td><td>0.1382</td><td></td><td></td><td>39.787</td><td>33.911</td><td>31.2012</td><td></td><td></td></tr><tr><td>HDSRNet(Tian et al., 2024)</td><td>34.6411</td><td>30.9513</td><td>29.0413</td><td></td><td></td><td>0.93713</td><td>0.87315</td><td>0.81915</td><td></td><td></td><td>0.04316</td><td>0.10315</td><td>0.15415</td><td></td><td></td><td>39.4612</td><td>34.209</td><td>31.338</td><td></td><td></td></tr><tr><td>GRLBase(Li et al., 2023)</td><td>35.661</td><td>31.931</td><td>29.911</td><td></td><td></td><td>0.9451</td><td>0.8891</td><td>0.8411</td><td></td><td></td><td>0.0371</td><td>0.0891</td><td>0.1351</td><td></td><td></td><td>39.994</td><td>33.8412</td><td>31.1213</td><td></td><td></td></tr><tr><td>GRLSmall(Li et al., 2023)</td><td>35.394</td><td>31.654</td><td>29.635</td><td></td><td></td><td>0.9434</td><td>0.8864</td><td>0.8354</td><td></td><td></td><td>0.0384</td><td>0.0925</td><td>0.1404</td><td></td><td></td><td>39.5410</td><td>33.6814</td><td>31.0315</td><td></td><td></td></tr><tr><td>GRLTiny(Li et al., 2023)</td><td>35.176 31.0418</td><td>31.416</td><td>29.406 26.6918</td><td></td><td></td><td>0.9426</td><td>0.8826</td><td>0.8306</td><td></td><td></td><td>0.0396</td><td>0.0969</td><td>0.1467</td><td></td><td></td><td>39.2016</td><td>33.2015 28.9017</td><td>30.5616</td><td></td><td></td></tr><tr><td>Bicubic</td><td></td><td>28.2518</td><td></td><td>24.876</td><td>22.346</td><td>0.89318</td><td>0.81318</td><td>0.75218</td><td>0.6756</td><td>0.5876</td><td>0.09618</td><td>0.19118</td><td>0.29118</td><td>0.4396</td><td>0.6136</td><td>32.7917</td><td></td><td>26.5717</td><td>23.456</td><td>18.746</td></tr></table>

Table 1. Comparison of PSNR, SSIM (Wang et al., 2004), LPIPS (Zhang et al., 2018a) and FSDS metrics for different methods on the DIV2K dataset (Agustsson & Timofte, 2017). Items with the highest and the second-highest mean values are highlighted in red and blue, respectively. The gray superscripts are the order of each method.

This observation shows the necessity of applying the FSDS metrics to assess image quality objectively.

# 5.4. Some Exceptions to Impulse Responses

![](images/d747d66fe3041f34cfa08810b2e9e86bce0058fbb1f668d435be5313ec2fd602.jpg)  
Figure 10. The impulse response of SwinIR-Real (Liang et al., 2021) and ESRGAN (Wang et al., 2018) is not an obvious sinc function.

We observe that not all impulse response of networks is ‘sinc’ function, as shown in Fig. 10. SwinIR-Real (Liang et al., 2021) and ESRGAN (Wang et al., 2018) are trained using adversarial loss, while methods in Fig. 9 uses loss like $\ell _ { 1 }$ or $\ell _ { 2 }$ loss. Therefore, we believe the ‘sinc’ impulse response is related to the loss function.

# 6. Conclusion

In this paper, we report an intriguing observation. i.e., the sinc phenomenon, which reveals that the impulse response of ISR networks act as low-pass filters. Building on this observation, we introduce a novel approach called Hybrid Response Analysis (HyRA) to explore the hidden behavior of ISR networks. HyRA treats a neural network as a combination of a linear system and a non-linear system with a zero impulse response. The linear system functions as a low-pass filter, while the non-linear system utilizes prior knowledge to inject high-frequency details. To assess the neural network’s information recovery across the frequency spectrum, we propose the Frequency Spectrum Distribution Similarity (FSDS) metric. FSDS uncovers properties overlooked by previous metrics, and experiments validate the rationality and necessity of it.

# Acknowledgements

We appreciate anonymous reviewers for their previous suggestions to help this paper better. Moreover, we would like to express our sincere gratitude to Ruijie Zhu (rzhu48@ucsc.edu) for his generous support in GPUs. Without his support, it is hard for us to do experiments using full-scale DIV2K dataset. This work is supported by NSFC (12271083).

# Impact Statement

This paper presents work whose goal is to advance the interpretability of neural networks in Image super-resolution. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

# References

Agustsson, E. and Timofte, R. Ntire 2017 challenge on single image super-resolution: Dataset and study. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, pp. 126–135, 2017.

Ahn, N., Kang, B., and Sohn, K.-A. Fast, accurate, and lightweight super-resolution with cascading residual network. In Proceedings of the European conference on computer vision (ECCV), pp. 252–268, 2018.

Bevilacqua, M., Roumy, A., Guillemot, C., and AlberiMorel, M. L. Low-complexity single-image superresolution based on nonnegative neighbor embedding. 2012.

Chen, X., Wang, X., Zhou, J., Qiao, Y., and Dong, C. Activating more pixels in image super-resolution transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 22367–22377, 2023.

Chen, Y., Liu, S., and Wang, X. Learning continuous image representation with local implicit image function. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 8628–8638, 2021.

Gu, J. and Dong, C. Interpreting super-resolution networks with local attribution maps. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9199–9208, 2021.

Hu, X., Mu, H., Zhang, X., Wang, Z., Tan, T., and Sun, J. Meta-sr: A magnification-arbitrary network for superresolution. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 1575– 1584, 2019.

Huang, J.-B., Singh, A., and Ahuja, N. Single image superresolution from transformed self-exemplars. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 5197–5206, 2015.

Hui, Z., Gao, X., Yang, Y., and Wang, X. Lightweight image super-resolution with information multi-distillation network. In Proceedings of the 27th acm international conference on multimedia, pp. 2024–2032, 2019.

Lee, J. and Jin, K. H. Local texture estimator for implicit representation function. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 1929–1938, 2022.

Li, Y., Fan, Y., Xiang, X., Demandolx, D., Ranjan, R., Timofte, R., and Van Gool, L. Efficient and explicit modelling of image hierarchies for image restoration. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18278–18289, 2023.

Liang, J., Cao, J., Sun, G., Zhang, K., Van Gool, L., and Timofte, R. Swinir: Image restoration using swin transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 1833–1844, 2021.

Lim, B., Son, S., Kim, H., Nah, S., and Mu Lee, K. Enhanced deep residual networks for single image superresolution. In Proceedings of the IEEE conference on computer vision and pattern recognition workshops, pp. 136–144, 2017.

Liu, H., Li, Z., Shang, F., Liu, Y., Wan, L., Feng, W., and Timofte, R. Arbitrary-scale super-resolution via deep learning: A comprehensive survey. Information Fusion, pp. 102015, 2023.

Miller, J. W., Farison, J. B., and Shin, Y. Spatially invariant image sequences. IEEE Transactions on Image Processing, 1(2):148–161, 1992.

Nyquist, H. Certain topics in telegraph transmission theory. Transactions of the American Institute of Electrical Engineers, 1928.

Oppenheim, A. V. and Schafer, R. W. Discrete-Time Signal Processing. Prentice Hall Press, USA, 2009.

Oppenheim, A. V., Willsky, A. S., and Nawab, S. H. Signals & Systems (2nd Ed.). Prentice-Hall, Inc., USA, 1996.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019.

Song, G., Sun, Q., Zhang, L., Su, R., Shi, J., and He, Y. Ope-sr: Orthogonal position encoding for designing a parameter-free upsampling module in arbitrary-scale image super-resolution. In Proceedings of the IEEE/CVF

Conference on Computer Vision and Pattern Recognition, pp. 10009–10020, 2023.

Sundararajan, M., Taly, A., and Yan, Q. Axiomatic attribution for deep networks. In International conference on machine learning, pp. 3319–3328. PMLR, 2017.

Tian, C., Zhang, X., Ren, J., Zuo, W., Zhang, Y., and Lin, C.-W. A heterogeneous dynamic convolutional neural network for image super-resolution. arXiv preprint arXiv:2402.15704, 2024.

Wang, L., Wang, Y., Lin, Z., Yang, J., An, W., and Guo, Y. Learning a single network for scale-arbitrary superresolution. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4801–4810, 2021.

Wang, X., Yu, K., Wu, S., Gu, J., Liu, Y., Dong, C., Qiao, Y., and Change Loy, C. Esrgan: Enhanced super-resolution generative adversarial networks. In Proceedings of the European conference on computer vision (ECCV) workshops, pp. 0–0, 2018.

Wang, X., Chen, X., Ni, B., Wang, H., Tong, Z., and Liu, Y. Deep arbitrary-scale image super-resolution via scaleequivariance pursuit. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1786–1795, 2023.

Wang, Z., Bovik, A. C., Sheikh, H. R., and Simoncelli, E. P. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612, 2004.

Wei, M. and Zhang, X. Super-resolution neural operator. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18247–18256, 2023.

Xu, K., Qin, M., Sun, F., Wang, Y., Chen, Y.-K., and Ren, F. Learning in the frequency domain. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 1740–1749, 2020.

Xu, Z. J. Understanding training and generalization in deep learning by fourier analysis. arXiv preprint arXiv:1808.04295, 2018.

Xu, Z.-Q. J. Frequency principle: Fourier analysis sheds light on deep neural networks. Communications in Computational Physics, 28(5):1746–1767, 2020.

Xu, Z.-Q. J., Zhang, Y., and Xiao, Y. Training behavior of deep neural network in frequency domain. In Neural Information Processing: 26th International Conference, ICONIP 2019, Sydney, NSW, Australia, December 12– 15, 2019, Proceedings, Part I 26, pp. 264–274. Springer, 2019.

Yang, J., Shen, S., Yue, H., and Li, K. Implicit transformer network for screen content image continuous super-resolution. Advances in Neural Information Processing Systems, 34:13304–13315, 2021.

Yang, W., Zhang, X., Tian, Y., Wang, W., Xue, J.-H., and Liao, Q. Deep learning for single image super-resolution: A brief review. IEEE Transactions on Multimedia, 21 (12):3106–3121, 2019.

Young, P., Lai, A., Hodosh, M., and Hockenmaier, J. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. Transactions of the Association for Computational Linguistics, 2:67–78, 2014.

Yu, Y., She, K., Liu, J., Cai, X., Shi, K., and Kwon, O. A super-resolution network for medical imaging via transformation analysis of wavelet multi-resolution. Neural Networks, 166:162–173, 2023.

Zhang, R., Isola, P., Efros, A. A., Shechtman, E., and Wang, O. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 586–595, 2018a.

Zhang, Y., Li, K., Li, K., Wang, L., Zhong, B., and Fu, Y. Image super-resolution using very deep residual channel attention networks. In Proceedings of the European conference on computer vision (ECCV), pp. 286–301, 2018b.

Zhang, Y., Tian, Y., Kong, Y., Zhong, B., and Fu, Y. Residual dense network for image super-resolution. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2472–2481, 2018c.

Zhang, Y., Xu, Z.-Q. J., Luo, T., and Ma, Z. Explicitizing an implicit bias of the frequency principle in two-layer neural networks. arXiv preprint arXiv:1905.10264, 2019.

A. Notation Conventions   
Table 2. Notation Conventions   

<table><tr><td>Symbols</td><td></td></tr><tr><td>j</td><td>Imaginary number unit</td></tr><tr><td>*</td><td>Convolution operator</td></tr><tr><td></td><td>2-D signal with variant x, y</td></tr><tr><td>rnt</td><td>Fourier transform of Ix,y</td></tr><tr><td>x(t)</td><td>1-D signal with variant t</td></tr><tr><td>X(jω)</td><td>Fourier transform of x(t), jω is a notation, ω is the variant</td></tr><tr><td>x[n]</td><td>Discrete signal with index n</td></tr><tr><td>X [k]</td><td>DFT of x[n]</td></tr><tr><td>F[x(t)]</td><td>Fourier transform operator, X (jω) = F[x(t)]</td></tr><tr><td>F−1[X(jω)]</td><td>Inverse Fourier transform, x(t) = F−1[X (jω)]</td></tr><tr><td>Signals</td><td></td></tr><tr><td>δ(t)</td><td>Dirac δ function The sinc funtion with parameter ω, sincω(t) = sin(ωt)</td></tr><tr><td>sincω(t)</td><td>The sinc function is the time-domain πt waveform of an ideal low-pass filter.</td></tr><tr><td></td><td>sin(ωx) . sin(ωy) πy πx</td></tr><tr><td>S∆T(t)</td><td>1-D sample signal with a sample interval of ∆T, s∆T (t) = ∑ δ(t − nT )</td></tr></table>

# B. Signal Processing Theories

We briefly introduce some related concepts and methods used in this paper in this section.

# B.1. System and Response

The word ‘system’ has many meanings and interpretations. This paper views a system as a process in which input signals are transformed by the system or cause the system to respond in some way, resulting in other signals as output (Oppenheim et al., 1996). Systems can be divided into linear systems and nonlinear systems according to their mathematical properties. A linear system refers to a system with such a property: the response of the system to the input $x _ { 1 } ( t ) , x _ { 2 } ( t )$ is $y _ { 1 } ( t )$ , $y _ { 2 } ( t )$ respectively, then when the input is $x _ { 1 } ( t ) + x _ { 2 } ( t )$ , the response of the system is $y _ { 1 } ( t ) + y _ { 2 } ( t )$ .

Systems can also be divided into time-variant ones and time-invariant ones according to their temporal properties. A time-invariant system refers to that the properties of the system do not change with time, that is, the system has the same impulse response at any time. It satisfies such a relationship: when the input is $x ( t )$ , the output is $y ( t )$ , and when the input is $x ( t - t _ { 0 } )$ , the output is $y ( t - t _ { 0 } )$ .

A system with both linear and time-invariant properties is a linear time-invariant (LTI) system. For an LTI system, we can use ‘impulse response’ to uniquely describe it: systems with the same impulse response are the same system, vice versa. The impulse response $h ( t )$ is defined as the output of the system when the input signal is $\delta ( t )$ (Dirac delta function). The response of a linear system to an arbitrary input signal can be computed through the convolution operation of its impulse response and the input signal, namely:

$$
y ( t ) = x ( t ) * h ( t ) = \int _ { - \infty } ^ { + \infty } x ( \tau ) h ( t - \tau ) d \tau .
$$

In the equation, $^ *$ is the convolution operator, $y ( t )$ is the system output and $x ( t )$ is the input signal. When we apply Fourier transform to the impulse response $h ( t )$ , then we can obtain the transfer function $H ( j \omega )$ of the system. The transfer function describes the frequency domain waveform of the impulse response. According to the convolution theorem, the response of a linear system can also be obtained by multiplying the Fourier transform of the input signal by the transfer function of the system and then performing the inverse Fourier transform. In summary, given the impulse response of an LTI system, we can calculate the system’s response to any output.

# B.2. Signal Sampling and Recovery

Time domain:

![](images/ef3a6e1b3a404402c1a8dd362d2650bb6190dcb17904c24f5a1c89a87e9e962e.jpg)  
Figure 11. Time-domain to frequency-domain waveform variation of the continuous signal sampling process. The sampling function $s _ { \delta T }$ is an impulse train sequence with an interval of $T$ , and $S _ { \delta T } ( t )$ is its frequency domain waveform, which is also an impulse train sequence. Sampling a signal causes duplication in the frequency domain.

![](images/73b706a6728e85653db2a520f63ee47e5433f9edb96f27f64077cf7ec56fd4a3.jpg)  
Figure 12. Time-domain to frequency-domain waveform variation in the process of sampling signal recovery. $h _ { L P } ( t )$ is the time-domain impulse response of a low-pass filter, and $H _ { L P } ( j \omega )$ is its frequency-domain waveform.

Signal sampling and sample recovery are very common operations, and in this section, we will briefly analyze this process from the perspective of both the time-domain and frequency-domain. The upper part of Fig. 11 shows the time domain waveform variation of signal sampling process, and the lower part shows the frequency domain waveform variation of signal sampling process. To sample a continuous signal, the sampling process can be regarded as the multiplication of the original signal $x ( t )$ and an impulse train signal $s _ { \Delta T } ( t )$ . It can be described as:

$$
\left\{ \begin{array} { c } { { \displaystyle x ^ { \prime } ( t ) = x ( t ) \cdot s _ { \Delta T } ( t ) , } } \\ { { \displaystyle s _ { \Delta T } ( t ) = \sum _ { n = - \infty } ^ { \infty } \delta ( t - n T ) , } } \end{array} \right.
$$

where $T$ denotes the sampling interval, $x ^ { \prime } ( t )$ denotes the sampled signal. According to the convolution theorem, the frequency domain change of the sampling process can be described in the following way:

$$
\begin{array} { l } { { X ^ { \prime } ( j \omega ) = X ( j \omega ) * S _ { \delta T } ( j \omega ) } } \\ { { \displaystyle = \sum _ { k = - \infty } ^ { \infty } X [ j ( \omega - k \frac { 2 \pi } { T } ) ] . } } \end{array}
$$

That is, the sampling process is reflected in the frequency spectrum as a periodic extension of the frequency spectrum of the original signal $x ( t )$ .

Fig. 12 shows the time domain and frequency domain waveform variation during the recovery process. For sampling recovery, in order to restore the sampled signal $x ^ { \prime } ( t )$ to the original signal $x ( t )$ , from the perspective of frequency domain, a low-pass filter is all we needed, that is, convolving the sampled signal with a low-pass filter $h _ { L P } ( t )$ . This process can be expressed as:

$$
\begin{array} { c } { { x ( t ) = x ^ { \prime } ( t ) * h _ { L P } ( t ) } } \\ { { = x ^ { \prime } ( t ) * s i n c _ { \omega } ( t ) , } } \end{array}
$$

where in the equation, $\begin{array} { r } { h _ { L P } ( t ) = s i n c _ { \omega _ { 0 } } ( t ) = \frac { s i n ( \omega _ { 0 } t ) } { \pi t } } \end{array}$ is the time domain response of the ideal low-pass filter, and its frequency-domain waveform $H _ { L P } ( j \omega )$ is a rectangular window.

# B.3. Spectrum Aliasing

![](images/8dbbdd780e7fff5d2b1b3e3be0999e7acdde0d63600619cee142f74b1e15fd2a.jpg)  
Figure 13. The illustration of spectrum aliasing. On the left, there is no aliasing as the sampling rate is sufficiently high. On the right, aliasing occurs due to an insufficient sampling rate.

Spectrum aliasing is a manifestation of information loss. Fig. 13 depicts the time-domain and frequency-domain scenarios of no frequency overlapping and frequency overlapping, respectively. When the sampling rate is lower than the Nyquist sampling rate6 (Nyquist, 1928). When the sampling rate is below the Nyquist sampling rate, the approach mentioned in Appx. B.2 cannot completely restore the original signal $x ( t )$ . From Tab. 3, we can see that for the sample signal $s _ { \Delta T } ( t )$ , the larger $T$ is, the sparser its time domain impulse train gets, while in the frequency spectrum the impulse trains gets denser. When the impulse trains in the frequency domain become sufficiently dense, and the spectrum of the original signal is periodically extended, overlapping occurs, preventing the complete recovery of the original signal. In ISR tasks, spectrum aliasing is manifested when restoring a low-resolution image to a high-resolution image, resulting in the loss of high-frequency information such as details and textures.

# C. Extra Experiments

# C.1. Linear and Non-linear Responses

In Fig. 14, we present the linear and nonlinear responses of various ISR networks along with their corresponding spectrums. From the figure, it is evident that different networks exhibit varying filtering effects in their linear components. EDSR demonstrates a pronounced removal of high-frequency components, and compared to other methods, it exhibits the smallest area of brightness diffusion around the central bright spot in its spectrum. From the nonlinear responses, it can be observed that the nonlinear components of the networks are all involved in supplementing high-frequency information and correcting distortions.

![](images/25f2d16a14f6af7719a56a8066f03db15d2cd9ce41c0c1ad237243eee3afae70.jpg)  
Figure 14. Linear and non-linear responses and their corresponding frequency spectrum of various ISR methods.

# C.2. Space Invariance

We conducted spatial invariance testing on RDN (Zhang et al., 2018c) (For the concept of spatial invariance, please refer to Sec. 4.1). The input data consists of an image where only one pixel is white (the pixel value is 1), and all other pixels are black (the pixel value is 0). By shifting the position of this white pixel, we obtain $I ( x - \Delta x , y - \Delta y )$ . This shifted input $I ( x - \Delta x , y - \Delta y )$ is then fed into the neural network, and we obtain its shifted impulse response, as illustrated in Fig. 15. Observing the experimental results, we find that the responses to different $I ( x - \Delta x , y - \Delta y )$ are consistent, with the only difference being their position. This demonstrates that, for ISR networks, the linear component in HyRA exhibits spatial invariance.

# C.3. Exploration of the Positional Origin of Sinc-like Patterns

As shown in Fig. 16, we visualize the output features of different components in the EDSR (Lim et al., 2017) network for analysis. We observe that the approximate shape of the sinc function begins to take form after the Upsampler module, and after a convolution, it essentially forms the shape of a sinc function. Interestingly, in the EDSR network, the Upsample

![](images/d56633b381dbd46bee82551a62fc577197d5848cd0f651e7e3cb4e117e7b8601.jpg)  
Figure 15. Spatial invariance experiment conducted on SwinIR (Liang et al., 2021). When we feed the SwinIR network with impulses at various positions, the ISR results demonstrate that the RDN exhibits spatial invariance.

![](images/f059066fcf16ca0cf6a77a89f580dfc319d49634e205d85784f15991caa28a03.jpg)  
Figure 16. Visualization of feature maps in different EDSR (Lim et al., 2017) network layers. The sinc-like pattern start to take shapes after the sub-pixel convolution, before the last convolution layer.

module uses sub-pixel convolution (convolution $^ +$ pixel shuffle) for upsampling without any interpolation. This indicates that the low-pass filter present in the network is learned by the network itself and not introduced by interpolation kernels.

# D. The Fourier Transform Pairs Involved in This Paper

Table 3. Fourier transform pairs   

<table><tr><td>Symbol/Name</td><td>Section(s)</td><td>Time domain</td><td>Frequency Domain</td></tr><tr><td>S∆T (t), ∆T is the samping in- Appx. B.2, Appx. B.3 terval</td><td></td><td>∞ ∑ δ(t − nT ) n=−∞</td><td>$\fa$ ∞ ∑ δ(ω − k2π) k=−∞</td></tr><tr><td>Ideal Low-pass filter</td><td>Sec. 4.1.1</td><td>x(t) = sincω (t) = sin(ωt), πt ω0 is called the cut-off fre- quency</td><td>1, |ω&lt;ωo X(jω) 0, |ω&gt;wo</td></tr><tr><td>δ(t)</td><td>Sec. 4.1.1</td><td>T→0 f+τ  (t) = 1</td><td></td></tr></table>

# E. The Windowing Operation

![](images/4b9c34538d0bcd84135283ad62b85588f5dc89e6f74af185666f77d4c454f459.jpg)  
Figure 17. Various window functions.

The time-domain waveform of an ideal low-pass filter is a sinc function. The sinc function is defined over $[ - \infty , \infty ]$ , and the number of zero crossings is countable. This implies that in reality, an ideal low-pass filter does not exist. In discrete-time signal processing, truncating a designed filter using a window function is common. There are many window functions, such as the rectangular window, Hanning window, Blackman window, and so on. Fig. 17 illustrates some commonly used window functions. Observing the experimental results and analyzing the relationship between the peak values of the main lobe and the first side lobe, we find that the impulse response of the neural network seems to undergo windowing. However, different networks appear to adopt different window functions.

# F. Frequency Spectrum Period Extension Caused by Zero Padding

When considering integer factor ISR, our approach to computing the linear component response is as follows: first, upsample the low-resolution image to the high-resolution image through zero-padding, and then convolve it with the impulse response to obtain the response. During the zero-padding process, it leads to period extension in the frequency spectrum. For a signal

![](images/1f47b80c6b6ef760da964ee3f56ce3c7614d56e7bde7c7e5485de1411e6361f9.jpg)  
Figure 18. Performing zero-padding on an image to reach the target size will result in periodic extension in the frequency spectrum obtained through its Discrete Fourier Transform.

$x [ n ]$ of length $N$ undergoing DFT to obtain $X [ k ]$ , we have:

$$
X [ k ] = \sum _ { n = 0 } ^ { N - 1 } x [ n ] e ^ { - j { \frac { 2 \pi } { N } } k n } .
$$

Then, zero-padding is applied to $x [ n ]$ , producing in a new signal $x _ { 2 } [ n ]$ of length $3 N$ :

$$
x _ { 2 } [ n ] = \left\{ \begin{array} { l l } { x [ \frac { n } { 3 } ] , } & { n = 0 , 3 , \cdots , 3 N - 3 } \\ { 0 , } & { o t h e r w i s e . } \end{array} \right.
$$

Perform DFT to $x _ { 2 } [ n ]$ to obtain $X ^ { \prime } [ k ]$ , then we have:

$$
\begin{array} { l } { { \displaystyle X ^ { \prime } [ k ] = \sum _ { n = 0 } ^ { 3 N - 1 } x _ { 2 } [ n ] e ^ { - j \frac { 2 \pi } { 3 N } k \cdot n } } \ ~ } \\ { { \displaystyle ~ = \sum _ { n = 0 } ^ { N - 1 } x [ n ] e ^ { - j \frac { 2 \pi } { 3 N } k \cdot 3 n } ~ . } } \\ { { \displaystyle ~ = \sum _ { n = 0 } ^ { N - 1 } x [ n ] e ^ { - j \frac { 2 \pi } { N } k n } } } \end{array}
$$

When $k < N$ , there exists:

$$
e ^ { - j { \frac { 2 \pi } { N } } k n } = e ^ { - j { \frac { 2 \pi } { N } } ( k + N ) n } = e ^ { - j { \frac { 2 \pi } { N } } ( k + 2 N ) n } = \cdot \cdot \cdot .
$$

Therefore,

$$
X ^ { \prime } [ k ] = { \left\{ \begin{array} { l l } { X [ k ] } & { 0 \leq k < N } \\ { X [ k m o d N ] } & { N \leq k < 3 N - 1 } \end{array} \right. }
$$

Thus, zero-padding causes period extension in the frequency spectrum. Ideally, the extended spectrum would be filtered out by the low-pass filter used in the ISR process. However, due to the limited filtering capability of the potential filters within the neural network, the stopband attenuation is low, and the extended spectrum cannot be completely filtered out.

# G. Comparison Between FSDS and $\ell _ { 1 } , \ell _ { 2 }$ norms

We compare our proposed FSDS metric with $\ell _ { 1 }$ norm, and $\ell _ { 2 }$ norm on both frequency domain and image domain as depicted in Tab. 4 and Tab. 5. From these two figures, we can observe that $\ell _ { 1 }$ norm and $\ell _ { 2 }$ norm produces similar ranking orders

when they are calculated on the same domain, indicating that $\ell _ { 1 }$ norm is equivalent to $\ell _ { 2 }$ norm when assessing image quality. However, the ranking orders produced by FSDS is distinctive to that of $\ell _ { 1 }$ and $\ell _ { 2 }$ . This means our FSDS metric reflects image quality in a unique way.

<table><tr><td rowspan="2">Method</td><td colspan="5">FSDS (Ours)</td><td colspan="5">1 Norm in Frequency Domain</td><td colspan="5">1 Norm</td></tr><tr><td>×2</td><td>×3</td><td>×4</td><td>×6</td><td>×12</td><td>×2</td><td>×3</td><td>×4</td><td>×6</td><td>×12</td><td>×2</td><td>×3</td><td>×4</td><td>×6</td><td>×12</td></tr><tr><td>EDSR(Lim et al., 2017)</td><td>39.21015</td><td>34.14810</td><td>31.3807</td><td></td><td></td><td>35.19012</td><td>51.67612</td><td>62.47816</td><td></td><td></td><td>0.01210</td><td>0.01913</td><td>0.02316</td><td></td><td></td></tr><tr><td>EDSR-LIIF(Lim et al., 2017)</td><td>39.37113</td><td>34.5356</td><td>31.3219</td><td>28.4464</td><td>23.1473</td><td>35.36914</td><td>51.88614</td><td>61.99615</td><td>73.9634</td><td>88.2035</td><td>0.01315</td><td>0.01915</td><td>0.02315</td><td>0.0294</td><td>0.0424</td></tr><tr><td>EDSR-OPESR(Lim et al., 2017)</td><td>39.7986</td><td>34.6305</td><td>31.28611</td><td></td><td></td><td>36.79616</td><td>51.94815</td><td>61.71114</td><td></td><td></td><td>0.01316</td><td>0.01811</td><td>0.02312</td><td></td><td></td></tr><tr><td>EDSR-SRNO(Lim et al., 2017)</td><td>39.53311</td><td>34.5277</td><td>31.4486</td><td>28.4583</td><td>22.7784</td><td>34.68810</td><td>51.11310</td><td>61.14011</td><td>73.1363</td><td>87.7043</td><td>0.01211</td><td>0.01810</td><td>0.02310</td><td>0.0293</td><td>0.0413</td></tr><tr><td>EDSR-LTE(Lim et al., 2017)</td><td>39.29014</td><td>34.3288</td><td>31.30310</td><td></td><td></td><td>35.11011</td><td>51.55611</td><td>61.70612</td><td></td><td></td><td>0.01314</td><td>0.01914</td><td>0.02314</td><td>-</td><td></td></tr><tr><td>RDN(Zhang et al., 2018c)</td><td>40.0223</td><td>32.94616</td><td>31.6464</td><td></td><td></td><td>34.5959</td><td>53.43116</td><td>61.06910</td><td></td><td></td><td>0.013|2</td><td>0.01916</td><td>0.02311</td><td></td><td></td></tr><tr><td>RDN-LIIF(Zhang et al., 2018c)</td><td>39.6909</td><td>34.8313</td><td>31.8323</td><td>28.8941</td><td>23.7781</td><td>34.2248</td><td>50.4348</td><td>60.5639</td><td>72.7532</td><td>87.4882</td><td>0.0128</td><td>0.0189</td><td>0.0229</td><td>0.0292</td><td>0.0412</td></tr><tr><td>RDN-OPESR(Zhang et al., 2018c)</td><td>40.1882</td><td>34.9592</td><td>31.4755</td><td></td><td></td><td>36.12615</td><td>50.7929</td><td>60.3888</td><td></td><td></td><td>0.01313</td><td>0.0187</td><td>0.0227</td><td></td><td></td></tr><tr><td>RDN-LTE(Zhang et al., 2018c)</td><td>39.8255</td><td>34.7494</td><td>31.8372</td><td>28.6542</td><td>23.3462</td><td>33.997</td><td>50.1067</td><td>60.2297</td><td>72.3011</td><td>87.0901</td><td>0.0127</td><td>0.0188</td><td>0.0228</td><td>0.0291</td><td>0.0411</td></tr><tr><td>SwinIR-classical(Liang et al., 2021)</td><td>40.3721</td><td>35.1251</td><td>32.3701</td><td></td><td></td><td>32.7505</td><td>48.3804</td><td>58.5794</td><td></td><td></td><td>0.0125</td><td>0.0175</td><td>0.0215</td><td></td><td></td></tr><tr><td>ITSRN(Yang et al., 2021)</td><td>31.25418</td><td>26.17818</td><td>25.87618</td><td>25.6195</td><td>21.5665</td><td>41.92817</td><td>53.45917</td><td>62.92717</td><td>74.2605</td><td>88.1604</td><td>0.01517</td><td>0.02017</td><td>0.02417</td><td>0.0305</td><td>0.0425</td></tr><tr><td>Bicubic</td><td>32.79017</td><td>28.89617</td><td>26.56817</td><td>23.4506</td><td>18.7366</td><td>50.57118</td><td>64.94918</td><td>73.41218</td><td>82.5676</td><td>93.0126</td><td>0.01818</td><td>0.02418</td><td>0.02918</td><td>0.0366</td><td>0.0506</td></tr><tr><td>HAT-S(Chen et al., 2023)</td><td>39.7847</td><td>33.79913</td><td>31.05714</td><td></td><td></td><td>32.3872</td><td>48.0473</td><td>58.2323</td><td></td><td></td><td>0.0112</td><td>0.0173</td><td>0.0213</td><td></td><td>-</td></tr><tr><td>HAT(Chen et al., 2023)</td><td>39.7847</td><td>33.91311</td><td>31.19612</td><td></td><td></td><td>32.3872</td><td>47.8152</td><td>58.0522</td><td></td><td></td><td>0.0112</td><td>0.0172</td><td>0.0212</td><td></td><td></td></tr><tr><td>HDSRNet(Tian et al., 2024)</td><td>39.45812</td><td>34.2039</td><td>31.3258</td><td></td><td></td><td>35.24513</td><td>51.69013</td><td>61.0913</td><td></td><td></td><td>0.0129</td><td>0.01912</td><td>0.02313</td><td></td><td></td></tr><tr><td>GRLBase(Li et al., 2023)</td><td>39.9904</td><td>33.84012</td><td>31.12113</td><td></td><td></td><td>31.8071</td><td>47.2631</td><td>57.2961</td><td></td><td></td><td>0.0111</td><td>0.0171</td><td>0.0211</td><td></td><td></td></tr><tr><td>GRLSmall(Li et al., 2023)</td><td>39.54410</td><td>33.67914</td><td>31.02715</td><td></td><td></td><td>32.7144</td><td>48.5935</td><td>58.8055</td><td></td><td></td><td>0.0124</td><td>0.0174</td><td>0.0214</td><td></td><td></td></tr><tr><td>GRLTiny(Li et al., 2023)</td><td>39.20516</td><td>33.19715</td><td>30.55616</td><td></td><td></td><td>33.3946</td><td>49.5506</td><td>59.8356</td><td></td><td></td><td>0.0126</td><td>0.0186</td><td>0.0226</td><td></td><td></td></tr></table>

Table 4. Comparison between our proposed FSDS metric and $\ell _ { 1 }$ norm in both frequency domain and image domain. Items with the highest mean values are highlighted in red and secondary mean values in blue. The gray superscripts denote the ranking order.

<table><tr><td rowspan="2">Method</td><td colspan="5">FSDS (Ours)</td><td colspan="5">2 Norm in Frequency Domain</td><td colspan="5">2 Norm</td></tr><tr><td>×2</td><td>×3</td><td>×4</td><td>×6</td><td>×12</td><td>×2</td><td>×3</td><td>×4</td><td>×6</td><td>×12</td><td>×2</td><td>×3</td><td>×4</td><td>×6</td><td>×12</td></tr><tr><td>EDSR(Lim et al., 2017)</td><td>39.21015</td><td>34.14810</td><td>31.3807</td><td></td><td></td><td>4966.90513</td><td>11163.14815</td><td>17206.00916</td><td></td><td></td><td>4966.90613</td><td>11163.14915</td><td>17206.01016</td><td></td><td></td></tr><tr><td>EDSR-LIIF(Lim et al., 2017)</td><td>39.3713</td><td>34.5356</td><td>31.3219</td><td>28.4464</td><td>23.1473</td><td>4967.72714</td><td>11163.1124</td><td>16947.23515</td><td>27007.7484</td><td>49280.8834</td><td>4967.72714</td><td>11163.11314</td><td>16947.23615</td><td>27007.7494</td><td>49280.8864</td></tr><tr><td>EDSR-OPESR(Lim et al., 2017)</td><td>39.7986</td><td>34.630</td><td>31.28611</td><td></td><td></td><td>5247.22916</td><td>11117.69412</td><td>16818.59814</td><td></td><td></td><td>5247.22916</td><td>1111769512</td><td>16818.59914</td><td></td><td></td></tr><tr><td>EDSR-SRNO(Lim et al., 2017)</td><td>39.53311</td><td>34.5277</td><td>31.4486</td><td>28.4583</td><td>22.7784</td><td>4794.9039</td><td>10845.39010</td><td>16453.70110</td><td>26236.2463</td><td>48097.2463</td><td>4794.9039</td><td>10845.39110</td><td>16453.70210</td><td>26236.2473</td><td>48097.2493</td></tr><tr><td>EDSR-LTE(Lim et al., 2017)</td><td>39.29014</td><td>34.3288</td><td>31.30310</td><td></td><td></td><td>490785511</td><td>11032.25211</td><td>16781.6213</td><td></td><td></td><td>4907.85511</td><td>11032.25211</td><td>16781.62213</td><td></td><td></td></tr><tr><td>RDN(Zhang et al., 2018c)</td><td>40.0223</td><td>32.94616</td><td>31.6464</td><td></td><td></td><td>4820.54010</td><td>12055.74516</td><td>16462.59411</td><td></td><td></td><td>4820.54110</td><td>12055.74616</td><td>16462.59511</td><td></td><td></td></tr><tr><td>RDN-LIIF(Zhang et al., 2018c)</td><td>39.6909</td><td>34.8313</td><td>31.8323</td><td>28.8941</td><td>23.7781</td><td>4627.8068</td><td>10477.3038</td><td>15993.7978</td><td>25720.7662</td><td>47650.6812</td><td>4627.8068</td><td>10477.3048</td><td>15993.7988</td><td>25720.7682</td><td>47650.6832</td></tr><tr><td>RDN-OPESR(Zhang et al., 2018c)</td><td>40.1882</td><td>34.9592</td><td>31.475</td><td></td><td></td><td>5061.53515</td><td>10609.1689</td><td>16034.9259</td><td></td><td></td><td>5061.53515</td><td>10609.1699</td><td>16034.9269</td><td></td><td></td></tr><tr><td>RDN-LTE(Zhang et al., 2018c)</td><td>39.8255</td><td>34.7494</td><td>31.8372</td><td>28.6542</td><td>23.3462</td><td>4576.6527</td><td>10349.2737</td><td>15808.8907</td><td>25384.5971</td><td>46955.7291</td><td>4576.6527</td><td>10349.2747</td><td>15808.8917</td><td>25384.5981</td><td>46955.7311</td></tr><tr><td>SwinIR-classical(Liang et al., 2021)</td><td>40.3721</td><td>35.125</td><td>32.3701</td><td></td><td></td><td>4222.2263</td><td>9617.2125</td><td>14804.5335</td><td></td><td></td><td>4222.226</td><td>9617.2125</td><td>14804.5345</td><td></td><td></td></tr><tr><td>ITSRN(Yang et al., 2021)</td><td>31.25418</td><td>26.17818</td><td>25.87618</td><td>25.6195</td><td>21.5665</td><td>7483.58217</td><td>12244.89717</td><td>17878.39917</td><td>27664.0545</td><td>49417.0615</td><td>7483.58217</td><td>12244.89817</td><td>17878.40017</td><td>27664.0563</td><td>49417.0645</td></tr><tr><td>Bicubic</td><td>32.79017</td><td>28.89617</td><td>26.56817</td><td>23.4506</td><td>18.7366</td><td>10873.90018</td><td>1956447318</td><td>26817.58018</td><td>38541.6756</td><td>64011.6986</td><td>10873.9018</td><td>19564.4418</td><td>268175818</td><td>38541.6776</td><td>64011.7026</td></tr><tr><td>HAT-S(Chen et al., 2023)</td><td>39.7847</td><td>33.79913 33.91311</td><td>31.05714</td><td></td><td></td><td>4107.3822</td><td>9423.4473</td><td>14498.5093</td><td></td><td></td><td>4107.383²2</td><td>9423.4483</td><td>14498.5103</td><td></td><td></td></tr><tr><td>HAT(Chen et al., 2023)</td><td>39.7847 39.45812</td><td>34.203</td><td>31.19612</td><td></td><td></td><td>4107.322</td><td>9326.632</td><td>14407.4572</td><td></td><td></td><td>4107.3832</td><td>9326.6132</td><td>14407.4582</td><td></td><td></td></tr><tr><td>HDSRNet(Tian et al., 2024)</td><td>39.9904</td><td>33.84012</td><td>31.3258 31.12113</td><td></td><td></td><td>4948.7/2</td><td>11121. 91213</td><td>1697. 18512</td><td></td><td></td><td>4948.23812</td><td>111121.923</td><td>1667.18612</td><td></td><td></td></tr><tr><td>GRLBase(Li et al., 2023)</td><td>39.54410</td><td>33.67914</td><td></td><td></td><td></td><td>3937.0131</td><td>903.913</td><td>13959.0781</td><td></td><td></td><td>3937.0131</td><td>9073.913</td><td>13959.0791</td><td></td><td></td></tr><tr><td>GRLSmall(Li et al., 2023)</td><td></td><td>33.19715</td><td>31.02715</td><td></td><td></td><td>4170.853</td><td>9585.3264</td><td>14742.7874</td><td></td><td></td><td>4170.853</td><td>9585.327</td><td>14742.7884</td><td></td><td></td></tr><tr><td>GRLTiny(Li et al., 2023)</td><td>39.20516</td><td></td><td>30.55616</td><td></td><td></td><td>4384.3646</td><td>10079.0176</td><td>15488.316</td><td></td><td></td><td>4384.3646</td><td>10079.0186</td><td>15488.3176</td><td></td><td></td></tr></table>

Table 5. Comparison between our proposed FSDS metric and $\ell _ { 2 }$ norm in both frequency domain and image domain. Items with the highest mean values are highlighted in red and secondary mean values in blue. The gray superscripts denote the ranking order. To demonstrate Parseval’s theorem, we omit the mean operation when calculating $\ell _ { 2 }$ norm.

# H. How G(I) is Trained and How G(I) and H(I) Vary Dependently

To better explain how $G ( I )$ is trained and how $G ( I )$ and $H ( I )$ vary dependently, we conduct a new experiment on observing their varying progress during training. We train a vanilla RDN (Zhang et al., 2018c) network from scratch and obtain the impulse response, $H ( I )$ , $G ( I )$ , and $N ( I )$ from each epoch (see Fig. 19). As shown in the figure, in the second row, the sinc phenomenon becomes clearer along with the training process, this indicates that the network is gradually learning the low-pass filter. The same conclusion can be further supported by observing the variation in $H ( I )$ as shown in the third row. In this row, $H ( I )$ illustrates the phenomenon of the grid-like distortion vanishing while low-frequency areas getting smoother. The fourth row depict the plot of $G ( I )$ . This row demonstrates that $G ( I )$ is capturing more and more high-frequency information, such as edges. This is very significant especially when comparing the result from Epoch 1 and Epoch 50. Such increase in high-frequncy information can also be found in the frequency spectrum. During the entire training process, the network fails to recover the words on the wall (please see the magnified area, there are words on the wall next to the blue awning). Observing $G ( I )$ , we can find no sign of words as well. This indicates that the network treats it as low-frequency information and ignores it, pointing the way for future network improvements.

![](images/746078e983b99b5ca52e1bb3d96df92463017bc3a3b6bcec83ea3f163f861992.jpg)  
Figure 19. $N ( I )$ , $H ( I )$ and $G ( I )$ from different epoches during training.

# I. Code Repositories

Table 6. The papers and repository links used in this paper.   

<table><tr><td>Abbreviate</td><td>Title</td><td>Publication</td><td>Year</td><td>Code Link</td></tr><tr><td>EDSR (Lim et al., 2017)</td><td>Enhanced Deep Residual Networks for Single Image Super-Resolution</td><td>CVPRW</td><td>2017</td><td>Github</td></tr><tr><td>LIIF(Chen et al., 2021)</td><td>Learning Continuous Image Representation with Local Implicit Image Function</td><td>CVPR</td><td>2021</td><td>Github</td></tr><tr><td>OPE-SR(Song et al., 2023)</td><td>OPE-SR: Orthogonal Position Encoding for Designing a Parameter-Free Upsampling Module in Arbitrary-Scale</td><td>CVPR</td><td>2023</td><td>Github</td></tr><tr><td>SRNO(Wei &amp; Zhang, 2023)</td><td>Image Super-Resolution Super-Resolution Neural Operator</td><td>CVPR</td><td>2023</td><td>Github</td></tr><tr><td>LTE (Lee &amp; Jin, 2022)</td><td>Local Texture Estimator for Implicit Representation Func- tion</td><td>CVPR</td><td>2022</td><td>Github</td></tr><tr><td>RDN (Zhang et al., 2018c)</td><td>Residual Dense Network for Image Super-Resolution</td><td>CVPR</td><td>2018</td><td>Github</td></tr><tr><td>SwinIR (Liang et al., 2021)</td><td>SwinIR: Image Restoration Using Swin Transformer</td><td>ICCV</td><td>2021</td><td>Github</td></tr><tr><td>ITSRN (Yang et al., 2021)</td><td>Implicit Transformer Network for Screen Content Image</td><td>NeurIPS</td><td>2021</td><td>Github</td></tr><tr><td>RCAN (Zhang et al., 2018b)</td><td>Continuous Super-Resolution Image Super-Resolution Using Very Deep Residual Chan-</td><td>ECCV</td><td>2018</td><td>Github</td></tr><tr><td>HAT (Chen et al., 2023)</td><td>nel Attention Networks Activating More Pixels in Image Super-Resolution Trans-</td><td>CVPR</td><td>2023</td><td>Github</td></tr><tr><td>HDSRNet (Tian et al., 2024)</td><td>former Heterogeneous Dynamic Convolutional Network in Image</td><td>Arxiv</td><td>2024</td><td>Github</td></tr><tr><td>GRL (Li et al., 2023)</td><td>Super-Resolution Efficient and Explicit Modelling of Image Hierarchies for Image Restoration</td><td>CVPR</td><td>2023</td><td>Github</td></tr></table>