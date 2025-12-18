# VOCOS: CLOSING THE GAP BETWEEN TIME-DOMAIN AND FOURIER-BASED NEURAL VOCODERS FOR HIGHQUALITY AUDIO SYNTHESIS

Hubert Siuzdak∗

# ABSTRACT

Recent advancements in neural vocoding are predominantly driven by Generative Adversarial Networks (GANs) operating in the time-domain. While effective, this approach neglects the inductive bias offered by time-frequency representations, resulting in reduntant and computionally-intensive upsampling operations. Fourierbased time-frequency representation is an appealing alternative, aligning more accurately with human auditory perception, and benefitting from well-established fast algorithms for its computation. Nevertheless, direct reconstruction of complexvalued spectrograms has been historically problematic, primarily due to phase recovery issues. This study seeks to close this gap by presenting Vocos, a new model that directly generates Fourier spectral coefficients. Vocos not only matches the state-of-the-art in audio quality, as demonstrated in our evaluations, but it also substantially improves computational efficiency, achieving an order of magnitude increase in speed compared to prevailing time-domain neural vocoding approaches. The source code and model weights have been open-sourced at https://github.com/gemelo-ai/vocos.

# 1 INTRODUCTION

Sound synthesis, the process of generating audio signals through electronic and computational means, has a long and rich history of innovation . Within the scope of text-to-speech (TTS), concatenative synthesis (Moulines & Charpentier, 1990; Hunt & Black, 1996) and statistical parametric synthesis (Yoshimura et al., 1999) were the prevailing approaches. The latter strategy relied on a source-filter theory of speech production, where the speech signal was seen as being produced by a source (the vocal cords) and then shaped by a filter (the vocal tract). In this framework, various parameters such as pitch, vocal tract shape, and voicing were estimated and then used to control a vocoder (Dudley, 1939) which would reconstruct the final audio signal. While vocoders evolved significantly (Kawahara et al., 1999; Morise et al., 2016), they tended to oversimplify speech production, generating a distinctive ”buzzy” sound and thus compromising the naturalness of the speech.

A significant breakthrough in speech synthesis was achieved with the introduction of WaveNet (Oord et al., 2016), a deep generative model for raw audio waveforms. WaveNet proposed a novel approach to handle audio signals by modeling them autoregressively in the time-domain, using dilated convolutions to broaden receptive fields and consequently capture long-range temporal dependencies. In contrast to the traditional parametric vocoders which incorporate prior knowledge about audio signals, WaveNet solely depends on end-to-end learning.

Since the advent of WaveNet, modeling distribution of audio samples in the time-domain has become the most popular approach in the field of audio synthesis. The primary methods have fallen into two major categories: autoregressive models and non-autoregressive models. Autoregressive models, like WaveNet, generate audio samples sequentially, conditioning each new sample on all previously generated ones (Mehri et al., 2016; Kalchbrenner et al., 2018; Valin & Skoglund, 2019). On the other hand, nonautoregressive models generate all samples independently, parallelizing the process and making it more computationally efficient (Oord et al., 2018; Prenger et al., 2019; Donahue et al., 2018).

![](images/d117bf6163ca92cd70cab72b931cd0c8f31ca5341c65c6396d3f7ea0c4ece67b.jpg)  
Figure 1: This illustrates the phase wrapping using an example sinusoidal signal (b) generated with a time-varying frequency (a). The instantaneous phase, $\varphi ( t )$ , is shown in (c). The apparent discontinuities observed around $- \pi$ and $\pi$ are the result of phase wrapping. Nevertheless, when viewed on the complex plane, these discontinuities represent continuous rotations. The instantaneous phase is computed as $\varphi ( t ) = \arg \left\{ \hat { s } ( t ) \right\}$ , where $\hat { s } ( t )$ denotes the Hilbert transform of $s ( t ) = \sin ( \omega t )$ .

# 1.1 CHALLENGES OF MODELING PHASE SPECTRUM

Despite considerable advancements in time-domain audio synthesis, efforts to generate spectral representations of signals have been relatively limited. While it’s possible to perfectly reconstruct the original signal from its Short-Time Fourier Transform (STFT), in many applications, only the magnitude of the STFT is utilized, leading to inherent information loss. The magnitude of the STFT provides a clear understanding of the signal by indicating the amplitude of different frequency components throughout its duration. In contrast, phase information is less intuitive and its manipulation can often yield unpredictable results.

Modeling the phase distribution presents challenges due to its intricate nature in the time-frequency domain. Phase spectrum exhibits a periodic structure causing wrapping around the principal values within the range of $( - \pi , \pi ]$ (Figure 1). Furthermore, the literature does not provide a definitive answer regarding the perceptual importance of phase-related information in speech (Wang & Lim, 1982; Paliwal et al., 2011). However, improved phase spectrum estimates have been found to minimize perceptual impairments (Saratxaga et al., 2012). Researchers have explored the use of deep learning for directly modeling the phase spectrum, but this remains a challenging area (Williamson et al., 2015).

# 1.2 CONTRIBUTION

Attempts to model Fourier-related coefficients with generative models have not achieved the same level of success as has been seen with modeling audio in the time-domain. This study focuses on bridging that gap with the following contributions:

• We propose Vocos – a GAN-based vocoder, trained to produce complex STFT coefficients of an audio clip. Unlike conventional neural vocoder architectures that rely on transposed convolutions for upsampling, this work proposes maintaining the same feature temporal resolution across all layers. The upsampling to waveform is realized through the Inverse Fast Fourier Transform.   
• To estimate phase angles, we propose a simple activation function defined in terms of a unit circle. This approach naturally incorporates implicit phase wrapping, ensuring meaningful values across all phase angles.   
• As Vocos maintains a low temporal resolution throughout the network, we revisited the need to use dilated convolutions, typical to time-domain vocoders. Our results indicate that integrating ConvNeXt (Liu et al., 2022) blocks contributes to better performance.   
• Our extensive evaluation shows that Vocos matches the state-of-the-art in audio quality while demonstrating over an order of magnitude increase in speed compared to time-domain counterparts. The source code and model weights have been made open-source, enabling further exploration and potential advancements in the field of neural vocoding.

# 2 RELATED WORK

GAN-based vocoders Generative Adversarial Networks (GANs) (Goodfellow et al., 2014), have achieved significant success in image generation, sparking interest from audio researchers due to their ability for fast and parallel waveform generation (Donahue et al., 2018; Engel et al., 2018). Progress was made with the introduction of advanced critics, such as the multi-scale discriminator (MSD) (Kumar et al., 2019) and the multi-period discriminator (MPD) (Kong et al., 2020). These works also adopted a feature-matching loss to minimize the distance between the discriminator feature maps of real and synthetic audio. To discriminate between real and generated samples, also multi-resolution spectrograms (MRD) were employed (Jang et al., 2021).

At this point the standard practice involves using a stack of dilated convolutions to increase the receptive field, and transposed convolutions to sequentially upsample the feature sequence to the waveform. However, this design is known to be susceptible to aliasing artifacts, and there are works suggesting more specialized modules for both the discriminator (Bak et al., 2022) and generator (Lee et al., 2022). The historical jump in quality is largely attributed to discriminators that are able to capture implicit structures by examining input audio signal at various periods or scales. It has been argued (You et al., 2021) that the architectural details of the generators do not significantly affect the vocoded outcome, given a well-established multi-resolution discriminating framework. Contrary to these methods, Vocos presents a carefully designed, frequency-aware generator that models the distribution of Fourier spectral coefficients, rather than modeling waveforms in the time domain.

Phase and magnitude estimation Historically, the phase estimation problem has been at the core of audio signal reconstruction. Traditional methods usually rely on the Griffin-Lim algorithm (Griffin & Lim, 1984), which iteratively estimate the phase by enforcing spectrogram consistency. However, the Griffin-Lim method introduces unnatural artifacts into synthesized speech. Several methods have been proposed for reconstructing phase using deep neural networks, including likelihood-based approaches (Takamichi et al., 2018) and GANs (Oyamada et al., 2018). Another line of work suggests perceptual phase quantization (Kim, 2003), which has proven promising in deep learning by treating the phase estimation problem as a classification problem (Takahashi et al., 2018).

Despite their effectiveness, these models assume the availability of a full-scale magnitude spectrogram, while modern audio synthesis pipelines often employ more compact representations, such as melspectrograms (Shen et al., 2018). Furthermore, recent research is focusing on leveraging latent features extracted by pretrained deep learning models (Polyak et al., 2021; Siuzdak et al., 2022).

Closer to this paper are studies that estimate both the magnitude and phase spectrum. This can be done either implicitly, by predicting the real and imaginary parts of the STFT, or explicitly, by parameterizing the model to generate the phase and magnitude components. In the former category, Gritsenko et al. (2020) presents a variant of a model trained to produce STFT coefficients. They recognized the significance of adversarial objective in preventing robotic sound quality, however they were unable to train it successfully due to its inherent instability. On the other hand, iSTFTNet (Kaneko et al., 2022) proposes modifications to HiFi-GAN, enabling it to return magnitude and phase spectrum. However, their optimal model only replaces the last two upsample blocks with inverse STFT, leaving the majority of the upsampling to be realized with transposed convolutions. They find that replacing more upsampling layers drastically degrades the quality. Pasini & Schluter ¨ (2022) were able to successfully model the magnitude and phase spectrum of audio with higher frequency resolution, although it required multi-step training (Caillon & Esling, 2021), because of the adversarial objective instability. Also, the initial studies using GANs to generate invertible spectrograms involved estimating instantaneous frequency (Engel et al., 2018). However, these were limited to a single dataset containing only individual musical instrument notes, with the assumption of a constant instantaneous frequency.

# 3 VOCOS

# 3.1 OVERVIEW

At its core, the proposed GAN model uses Fourier-based time-frequency representation as the target data distribution for the generator. Vocos is constructed without any transposed convolutions; instead, the upsample operation is realized solely through the fast inverse STFT. This approach permits a unique model design compared to time-domain vocoders, which typically employ a series of upsampling layers to inflate input features to the target waveform’s resolution, often necessitating upscaling by several hundred times. In contrast, Vocos maintains the same temporal resolution throughout the network (Figure 2). This design, known as an isotropic architecture, has been found to work well in various settings, including Transformer (Vaswani et al., 2017). This approach can also be particularly beneficial for audio synthesis. Traditional methods often use transposed convolutions that can introduce aliasing artifacts, necessitating additional measures to mitigate the issue (Karras et al., 2021; Lee et al., 2022). Vocos eliminates learnable upsampling layers, and instead employs the well-establish inverse Fourier transform to reconstruct the original-scale waveform. In the context of converting mel-spectrograms into audio signal, the temporal resolution is dictated by the hop size of the STFT.

![](images/87756abc5e3777ea32f4512a23f56f3a89a7427689bd987d4d0c944420ea5794.jpg)  
Figure 2: Comparison of a typical time-domain GAN vocoder (a), with the proposed Vocos architecture (b) that maintains the same temporal resolution across all layers. Time-domain vocoders use transposed convolutions to sequentially upsample the signal to the desired sample rate. In contrast, Vocos achieves this by using a computationally efficient inverse Fourier transform.

Vocos uses the Short-Time Fourier Transform (STFT) to represent audio signals in the time-frequency domain:

$$
\mathrm { S T F T } _ { x } [ m , k ] = \sum _ { n = 0 } ^ { N - 1 } x [ n ] w [ n - m ] e ^ { - j 2 \pi k n / N }
$$

The STFT applies the Fourier transform to successive windowed sections of the signal. In practice, the STFT is computed by taking a sequence of Fast Fourier Transforms (FFTs) on overlapping, windowed frames of data, which are created as the window function advances or “hops” through time.

# 3.2 MODEL

Backbone Vocos adapts ConvNeXt (Liu et al., 2022) as the foundational backbone for the generator. It first embeds the input features into a hidden dimensionality and then applies a stack of 1D convolutional blocks. Each block consists of a depthwise convolution, followed by an inverted bottleneck that projects features into a higher dimensionality using pointwise convolution. GELU (Gaussian Error Linear Unit) activations are used within the bottleneck, and Layer Normalization is employed between the blocks.

Head Fourier transform of real-valued signals is conjugate symmetric, so we use only a single side band spectrum, resulting in $n _ { f f t } / 2 + 1$ coefficients per frame. As we parameterize the model to output phase and magnitude values, hidden-dim activations are projected into a tensor $\mathbf { h }$ with $n _ { f f t } + 2$ channels and splitted into:

$$
\mathbf { m } , \mathbf { p } = \mathbf { h } [ 1 : ( n _ { f f t } / 2 + 1 ) ] , \mathbf { h } [ ( n _ { f f t } / 2 + 2 ) : n ]
$$

To represent the magnitude, we apply the exponential function to m: $\mathbf { M } = \exp ( \mathbf { m } )$ .

We map $\mathbf { p }$ onto the unit circle by calculating the cosine and sine of $\mathbf { p }$ to obtain $\mathbf { x }$ and $\mathbf { y }$ , respectively

$$
\begin{array} { c } { \mathbf { x } = \cos ( \mathbf { p } ) } \\ { \mathbf { y } = \sin ( \mathbf { p } ) } \end{array}
$$

Finally, we represent complex-valued coefficients as: $\mathbf { S T F T } = \mathbf { M } \cdot \left( \mathbf { x } + j \mathbf { y } \right)$ .

Importantly, this simple formulation allows to express phase angle $\varphi = \mathrm { a t a n } 2 ( \mathbf { y } , \mathbf { x } )$ for any real argument p, and it ensures that $\varphi$ is correctly wrapped into the desired range $( - \pi , \pi ]$ .

Discriminator We employ the multi-period discriminator (MPD) as defined by Kong et al. (2020), and multi-resolution discriminator (MRD) (Jang et al., 2021).

# 3.3 LOSS

Following the approach proposed by Kong et al. (2020), the training objective of Vocos consists of reconstruction loss, adversarial loss and feature matching loss. However, we adopt a hinge loss formulation instead of the least squares GAN objective, as suggested by Zeghidour et al. (2021):

$$
\ell _ { G } ( \hat { \pmb x } ) = \frac { 1 } { K } \sum _ { k } \operatorname* { m a x } \left( 0 , 1 - D _ { k } ( \hat { \pmb x } ) \right)
$$

$$
\ell _ { D } ( { \pmb x } , \hat { \pmb x } ) = \frac { 1 } { K } \sum _ { k } \operatorname* { m a x } \left( 0 , 1 - D _ { k } ( { \pmb x } ) \right) + \operatorname* { m a x } \left( 0 , 1 + D _ { k } ( \hat { \pmb x } ) \right)
$$

where $D _ { k }$ is the kth subdiscriminator. The reconstruction loss, denoted as $L _ { m e l }$ , is defined as the L1 distance between the mel-scaled magnitude spectrograms of the ground truth sample $_ { \textbf { \em x } }$ and the synthesized sample: $\hat { \textbf { \textit { x } } }$ : $L _ { m e l } = \| \mathcal { M } ( \pmb { x } ) - \pmb { \mathcal { M } } ( \hat { \pmb { x } } ) \| _ { 1 }$ . The feature matching loss, denoted as $L _ { f e a t }$ is calculated as the mean of the distances between the lth feature maps of the $k$ th subdistriminator: $\begin{array} { r } { L _ { f e a t } = \frac { 1 } { K L } \sum _ { k } \sum _ { l } \big \| D _ { k } ^ { l } ( \pmb { x } ) - D _ { k } ^ { l } ( \pmb { \hat { x } } ) \big \| _ { 1 } . } \end{array}$ .

# 4 RESULTS

# 4.1 MEL-SPECTROGRAMS

Reconstructing audio waveforms from mel-spectrograms has become a fundamental task for vocoders in contemporary speech synthesis pipelines. In this section, we assess the performance of Vocos relative to established baseline methods.

Data The models are trained on the LibriTTS dataset (Zen et al., 2019), from which we use the entire training subset (both train-clean and train-other). We maintain the original sampling rate of $2 4 \mathrm { k H z }$ for the audio files. For each audio sample, we compute mel-scaled spectrograms using parameters: $n _ { f f t } = 1 0 2 4$ , $h o p _ { n } = 2 5 6$ , and the number of Mel bins is set to 100. A random gain is applied to the audio samples, resulting in a maximum level between -1 and -6 dBFS.

Training Details We train our models up to 2 million iterations, with 1 million iterations per generator and discriminator. During training, we randomly crop the audio samples to 16384 samples and use a batch size of 16. The model is optimized using the AdamW optimizer with an initial learning rate of 2e-4 and betas set to (0.9, 0.999). The learning rate is decayed following a cosine schedule.

Table 1: Objective evaluation metrics for various models, including baseline models (HiFi-GAN, iSTFTNet, BigVGAN) and Vocos.   

<table><tr><td></td><td>UTMOS (↑)</td><td>VISQOL (↑)</td><td>PESQ (↑)</td><td>V/UV F1 (↑)</td><td>Periodicity ()</td></tr><tr><td>Ground truth</td><td>4.058</td><td></td><td></td><td>−</td><td></td></tr><tr><td>HiFi-GAN</td><td>3.669</td><td>4.57</td><td>3.093</td><td>0.9457</td><td>0.129</td></tr><tr><td>iSTFTNet</td><td>3.564</td><td>4.56</td><td>2.942</td><td>0.9372</td><td>0.141</td></tr><tr><td>BigVGAN</td><td>3.749</td><td>4.65</td><td>3.693</td><td>0.9557</td><td>0.108</td></tr><tr><td>Vocos</td><td>3.734</td><td>4.66</td><td>3.70</td><td>0.9582</td><td>0.101</td></tr><tr><td>w/ absolute phase</td><td>3.590</td><td>4.65</td><td>3.565</td><td>0.9556</td><td>0.108</td></tr><tr><td>w/ Snake</td><td>3.699</td><td>4.66</td><td>3.629</td><td>0.9579</td><td>0.102</td></tr><tr><td>w/o ConvNeXt</td><td>3.658</td><td>4.65</td><td>3.528</td><td>0.9534</td><td>0.109</td></tr></table>

Baseline Methods Our proposed model, Vocos, is compared to: iSTFTNet (Kaneko et al., 2022), BigVGAN (Lee et al., 2022), and HiFi-GAN (Kong et al., 2020). These models are retrained on the same LibriTTS subset for up to 2 million iterations, following the original training details recommended by the authors. We use the official implementations of $\mathbf { \bar { B i g V G A N ^ { l } } }$ and HiFi-GAN2, and a community open-sourced version of iSTFTNet3.

# 4.1.1 EVALUATION

Objective Evaluation For objective evaluation of our models, we employ the UTMOS (Saeki et al., 2022) automatic Mean Opinion Score (MOS) prediction system. Although UTMOS can yield scores highly correlated with human evaluations, it is restricted to $1 6 \mathrm { k H z }$ sample rate. To assess perceptual quality, we also utilize ViSQOL (Chinen et al., 2020) in audio-mode, which operates in the full band. Our evaluation process also encompasses several other metrics, including the Perceptual Evaluation of Speech Quality (PESQ) (Rix et al., 2001), periodicity error, and the F1 score for voiced/unvoiced classification (V/UV F1), following the methodology proposed by Morrison et al. (2021). The results are presented in Table 1. Vocos achieves superior performance in most of the metrics compared to the other models. It obtains the highest scores in VISQOL and PESQ. Importantly, it also effectively mitigates the periodicity issues frequently associated with time-domain GANs. BigVGAN stands out as the closest competitor, especially in the UTMOS metric, where it slightly outperforms Vocos.

In our ablation study, we examined the impact of specific design decisions on Vocos’s performance:

• Vocos with absolute phase: In this variant, we predict phase angles using a tanh nonlinearity, scaled to fit within the range of $[ - \pi , \pi ]$ . This formulation does not give the model an inductive bias regarding the periodic nature of phase, and the results show it leads to degraded quality. This finding emphasizes the importance of implicit phase wrapping in the effectiveness of Vocos.

• Vocos with Snake activation: Although Snake (Ziyin et al., 2020) has been shown to enhance time-domain vocoders such as BigVGAN, in our case, it did not result in performance gains; in fact, it showed a slight decline. The primary purpose of the Snake function is to induce periodicity, addressing the limitations of time-domain vocoders. Vocos, on the other hand, explicitly incorporates periodicity through the use of Fourier basis functions, eliminating the need for specialized modules like Snake.

• Vocos without ConvNeXt: Replacing ConvNeXt blocks with traditional ResBlocks with dilated convolutions, slightly lowers scores across all evaluated metrics. This finding highlights the integral role of ConvNeXt blocks in Vocos, contributing significantly to its overall success.

Table 2: Subjective evaluation metrics – 5-scale Mean Opinion Score (MOS) and Similarity Mean Opinion Score (SMOS) with $9 5 \%$ confidence interval.   

<table><tr><td></td><td>MOS (↑)</td><td>SMOS (↑)</td></tr><tr><td>Ground truth</td><td>3.81±0.16</td><td>4.70±0.11</td></tr><tr><td>HiFi-GAN</td><td>3.54±0.16</td><td>4.49±0.14</td></tr><tr><td>iSTFTNet</td><td>3.57±0.16</td><td>4.42±0.16</td></tr><tr><td>BigVGAN</td><td>3.64±0.15</td><td>4.54±0.14</td></tr><tr><td>Vocos</td><td>3.62±0.15</td><td>4.55±0.15</td></tr></table>

Subjective Evaluation We conducted crowd-sourced subjective assessments, using a 5-point Mean Opinion Score (MOS) to evaluate the naturalness of the presented recordings. Participants rated speech samples on a scale from 1 (’poor - completely unnatural speech’) to 5 (’excellent - completely natural speech’). Following (Lee et al., 2022), we also conducted a 5-point Similarity Mean Opinion Score (SMOS) between the reproduced and ground-truth recordings. Participants were asked to assign a similarity score to pairs of audio files, with a rating of 5 indicating ’Extremely similar’ and a rating of 1 representing ’Not at all similar’.

To ensure the quality of responses, we carefully selected participants through a third-party crowdsourcing platform. Our criteria included the use of headphones, fluent English proficiency, and a declared interest in music listening as a hobby. A total of 1560 ratings were collected from 39 participants.

The results are detailed in Table 2. Vocos performs on par with the state-of-the-art in both perceived quality and similarity. Statistical tests show no significant differences between Vocos and BigVGAN in MOS and SMOS scores, with p-values greater than 0.05 from the Wilcoxon signed-rank test.

Table 3: VISQOL scores of various models tested on the MUSDB18 dataset. A higher VISQOL score indicates better perceptual audio quality.   

<table><tr><td></td><td>Mixture</td><td>Drums</td><td>Bass</td><td>Other</td><td>Vocals</td><td>Average</td></tr><tr><td>HiFi-GAN</td><td>4.46</td><td>4.40</td><td>4.12</td><td>4.44</td><td>4.54</td><td>4.39</td></tr><tr><td>iSTFTNet</td><td>4.47</td><td>4.48</td><td>3.80</td><td>4.40</td><td>4.53</td><td>4.34</td></tr><tr><td>BigVGAN</td><td>4.60</td><td>4.60</td><td>4.29</td><td>4.58</td><td>4.64</td><td>4.54</td></tr><tr><td>Vocos</td><td>4.61</td><td>4.61</td><td>4.31</td><td>4.58</td><td>4.66</td><td>4.55</td></tr></table>

Out-of-distribution data A crucial aspect of a vocoder is its ability to generalize to unseen acoustic conditions. In this context, we evaluate the performance of Vocos with out-of-distribution audio using the MUSDB18 dataset (Rafii et al., 2017), which includes a variety of multi-track music audio like vocals, drums, bass, and other instruments, along with the original mixture. The VISQOL scores for this evaluation are provided in Table 3. From the table, Vocos consistently outperforms the other models, achieving the highest scores across all categories.

Figure 3 presents spectrogram visualization of an out-of-distribution singing voice sample, as reproduced by different models. Periodicity artifacts are commonly observed when employing time-domain GANs. BigVGAN, with its anti-aliasing filters, is able to recover some of the harmonics in the upper frequency ranges, marking an improvement over HiFi-GAN. Nonetheless, Vocos appears to provide a more accurate reconstruction of these harmonics, without the need for additional modules.

# 4.2 NEURAL AUDIO CODEC

While traditionally, neural vocoders reconstruct the audio waveform from a mel-scaled spectrogram – an approach widely adopted in many speech synthesis pipelines – recent research has started to utilize learnt features (Siuzdak et al., 2022), often in a quantized form (Borsos et al., 2022).

In this section, we draw a comparison with EnCodec (Defossez et al., 2022), an open-source neural ´ audio codec, which follows a typical time-domain GAN vocoder architecture and uses Residual Vector Quantization (RVQ) (Zeghidour et al., 2021) to compress the latent space. RVQ cascades multiple layers of Vector Quantization, iteratively quantizing the residuals from the previous stage to form a multi-stage structure, thereby enabling support for multiple bandwidth targets. In EnCodec, dedicated discriminators are trained for each bandwidth. In contrast, we have adapted Vocos to be a conditional GAN with a projection discriminator (Miyato & Koyama, 2018), and have incorporated adaptive layer normalization (Huang & Belongie, 2017) into the generator.

Audio reconstruction We utilize the open-source model checkpoint of EnCodec operating at 24 kHz. To align with EnCodec, we scale down Vocos to match its parameter count (7.9M) and train it on clean speech segments sourced from the DNS Challenge (Dubey et al., 2022). Our evaluation, conducted on the DAPS dataset (Mysore, 2014) and detailed in Table 4, reveals that despite EnCodec’s reconstruction artifacts not significantly impacting PESQ and Periodicity scores, they are considerably reflected in the perceptual score, as denoted by UTMOS. In this regard, Vocos notably outperforms EnCodec. We also performed a crowd-sourced subjective assessment to evaluate the naturalness of these samples. The results, as shown in Table 5, indicate that Vocos consistently achieves better performance across a range of bandwidths, based on evaluations by human listeners.

Table 4: Objective evaluation metric calculated for various bandwidths.   

<table><tr><td></td><td>Bandwidth</td><td>UTMOS (↑)</td><td>VISQOL (↑)</td><td>PESQ (↑)</td><td>V/UV F1 (↑)</td><td>Periodicity (</td></tr><tr><td rowspan="4">EnCodec</td><td>1.5 kbps</td><td>1.527</td><td>3.74</td><td>1.508</td><td>0.8826</td><td>0.215</td></tr><tr><td>3.0 kbps</td><td>2.522</td><td>3.93</td><td>2.006</td><td>0.9347</td><td>0.141</td></tr><tr><td>6.0 kbps</td><td>3.262</td><td>4.13</td><td>2.665</td><td>0.9625</td><td>0.090</td></tr><tr><td>12.0 kbps</td><td>3.765</td><td>4.25</td><td>3.283</td><td>0.9766</td><td>0.062</td></tr><tr><td rowspan="4">Vocos</td><td>1.5 kbps</td><td>3.210</td><td>3.88</td><td>1.845</td><td>0.9238</td><td>0.160</td></tr><tr><td>3.0 kbps</td><td>3.688</td><td>4.06</td><td>2.317</td><td>0.9380</td><td>0.135</td></tr><tr><td>6.0 kbps</td><td>3.822</td><td>4.22</td><td>2.650</td><td>0.9439</td><td>0.124</td></tr><tr><td>12.0 kbps</td><td>3.882</td><td>4.34</td><td>2.874</td><td>0.9482</td><td>0.116</td></tr></table>

Table 5: Subjective evaluation metrics – 5-scale Mean Opinion Score (MOS) with $9 5 \%$ confidence interval for various bandwidths.   

<table><tr><td>Bandwidth</td><td>Vocos</td><td>EnCodec</td></tr><tr><td>1.5 kbps</td><td>2.73±0.20</td><td>1.09±0.05</td></tr><tr><td>3 kbps</td><td>3.50±0.18</td><td>1.71±0.21</td></tr><tr><td>6 kbps</td><td>3.84±0.16</td><td>2.41±0.15</td></tr><tr><td>12 kbps</td><td>4.00±0.16</td><td>3.08±0.19</td></tr><tr><td>Ground truth</td><td>4.09±0.16</td><td></td></tr></table>

End-to-end text-to-speech Recent progress in text-to-speech (TTS) has been notably driven by language modeling architectures employing discrete audio tokens. Bark (Suno AI, 2023), a widely recognized open-source model, leverages a GPT-style, decoder-only architecture, with EnCodec’s 6kbps audio tokens serving as its vocabulary. Vocos trained to reconstruct EnCodec tokens can effectively serve as a drop-in replacement vocoder for Bark. We have provided text-to-speech samples from Bark and Vocos on our website and encourage readers to listen to them for a direct comparison.4.

# 4.3 INFERENCE SPEED

Our inference speed benchmarks were conducted using an Nvidia Tesla A100 GPU and an AMD EPYC 7542 CPU. The code was implemented in Pytorch, with no hardware-specific optimizations.

![](images/1bcfa19c53679dbec68640b3d6e647db5c39fd72d0b0433c619f200c291d4b58.jpg)  
Figure 3: Spectrogram visualization of an out-of-distribution singing voice sample reproduced by different models. The bottom row presents a zoomed-in view of the upper midrange frequency range.

The forward pass was computed using a batch of 16 samples, each one second long. Table 6 presents the synthesis speed and model footprint of Vocos in comparison to other models.

Vocos showcases notable speed advantages compared to other models, operating approximately 13 times faster than HiFi-GAN and nearly 70 times faster than BigVGAN. This speed advantage is particularly pronounced when running without GPU acceleration. This is primarily due to the use of the Inverse Short-Time Fourier Transform (ISTFT) algorithm instead of transposed convolutions. We also evaluate a variant of Vocos that utilizes ResBlock’s dilated convolutions instead of ConvNeXt blocks. Depthwise separable convolutions offer an additional speedup when executed on a GPU.

Table 6: Model footprint and synthesis speed. xRT denotes the speed factor relative to real-time. A higher xRT value means the model can generate speech faster than real-time, with a value of 1.0 denoting real-time speed.   

<table><tr><td rowspan="2">Model</td><td colspan="2">xRT (↑)</td><td rowspan="2">Parameters</td></tr><tr><td>GPU</td><td>CPU</td></tr><tr><td>HiFi-GAN</td><td>495.54</td><td>5.84</td><td>14.0 M</td></tr><tr><td>BigVGAN</td><td>98.61</td><td>0.40</td><td>14.0 M</td></tr><tr><td>ISTFTNet</td><td>1045.94</td><td>14.44</td><td>13.3 M</td></tr><tr><td>Vocos</td><td>6696.52</td><td>169.63</td><td>13.5 M</td></tr><tr><td>w/o ConvNeXt</td><td>4565.71</td><td>193.56</td><td>14.9 M</td></tr></table>

# 5 CONCLUSIONS

This paper introduces Vocos, a novel neural vocoder that bridges the gap between time-domain and Fourier-based approaches. Vocos tackles the challenges associated with direct reconstruction of complex-valued spectrograms, with careful design of generator that correctly handle phase wrapping. It achieves accurate reconstruction of the coefficients in Fourier-based time-frequency representations.

The results demonstrate that the proposed vocoder matches state-of-the-art audio quality while effectively mitigating periodicity issues commonly observed in time-domain GANs. Importantly, Vocos provides a significant computational efficiency advantage over traditional time-domain methods by utilizing inverse fast Fourier transform for upsampling.

Overall, the findings of this study contribute to the advancement of neural vocoding techniques by incorporating the benefits of Fourier-based time-frequency representations. The open-sourcing of the source code and model weights allows for further exploration and application of the proposed vocoder in various audio processing tasks.

# REFERENCES

Taejun Bak, Junmo Lee, Hanbin Bae, Jinhyeok Yang, Jae-Sung Bae, and Young-Sun Joo. Avocodo: Generative adversarial network for artifact-free vocoder. arXiv preprint arXiv:2206.13404, 2022.

Zalan Borsos, Rapha ´ el Marinier, Damien Vincent, Eugene Kharitonov, Olivier Pietquin, Matt Sharifi, ¨ Olivier Teboul, David Grangier, Marco Tagliasacchi, and Neil Zeghidour. Audiolm: a language modeling approach to audio generation. arXiv preprint arXiv:2209.03143, 2022.

Marina Bosi and Richard E Goldberg. Introduction to digital audio coding and standards, volume 721. Springer Science & Business Media, 2002.

Antoine Caillon and Philippe Esling. Rave: A variational autoencoder for fast and high-quality neural audio synthesis. arXiv preprint arXiv:2111.05011, 2021.

Michael Chinen, Felicia SC Lim, Jan Skoglund, Nikita Gureev, Feargus O’Gorman, and Andrew Hines. Visqol v3: An open source production ready objective speech and audio metric. In 2020 twelfth international conference on quality of multimedia experience (QoMEX), pp. 1–6. IEEE, 2020.

Alexandre Defossez, Jade Copet, Gabriel Synnaeve, and Yossi Adi. High fidelity neural audio ´ compression. arXiv preprint arXiv:2210.13438, 2022.

Chris Donahue, Julian McAuley, and Miller Puckette. Adversarial audio synthesis. arXiv preprint arXiv:1802.04208, 2018.

Harishchandra Dubey, Vishak Gopal, Ross Cutler, Ashkan Aazami, Sergiy Matusevych, Sebastian Braun, Sefik Emre Eskimez, Manthan Thakker, Takuya Yoshioka, Hannes Gamper, et al. Icassp 2022 deep noise suppression challenge. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 9271–9275. IEEE, 2022.

Homer Dudley. Remaking speech. The Journal of the Acoustical Society of America, 11(2):169–177, 1939.

Jesse Engel, Kumar Krishna Agrawal, Shuo Chen, Ishaan Gulrajani, Chris Donahue, and Adam Roberts. Gansynth: Adversarial neural audio synthesis. In International Conference on Learning Representations, 2018.

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, and K.Q. Weinberger (eds.), Advances in Neural Information Processing Systems, volume 27. Curran Associates, Inc., 2014.

Daniel Griffin and Jae Lim. Signal estimation from modified short-time fourier transform. IEEE Transactions on acoustics, speech, and signal processing, 32(2):236–243, 1984.

Alexey Gritsenko, Tim Salimans, Rianne van den Berg, Jasper Snoek, and Nal Kalchbrenner. A spectral energy distance for parallel speech synthesis. Advances in Neural Information Processing Systems, 33:13062–13072, 2020.

Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104, 2023.

Xun Huang and Serge Belongie. Arbitrary style transfer in real-time with adaptive instance normalization. In Proceedings of the IEEE international conference on computer vision, pp. 1501–1510, 2017.

Andrew J Hunt and Alan W Black. Unit selection in a concatenative speech synthesis system using a large speech database. In 1996 IEEE International Conference on Acoustics, Speech, and Signal Processing Conference Proceedings, volume 1, pp. 373–376. IEEE, 1996.

Won Jang, Dan Lim, Jaesam Yoon, Bongwan Kim, and Juntae Kim. Univnet: A neural vocoder with multi-resolution spectrogram discriminators for high-fidelity waveform generation. arXiv preprint arXiv:2106.07889, 2021.

Nal Kalchbrenner, Erich Elsen, Karen Simonyan, Seb Noury, Norman Casagrande, Edward Lockhart, Florian Stimberg, Aaron Oord, Sander Dieleman, and Koray Kavukcuoglu. Efficient neural audio synthesis. In International Conference on Machine Learning, pp. 2410–2419. PMLR, 2018.

Takuhiro Kaneko, Kou Tanaka, Hirokazu Kameoka, and Shogo Seki. istftnet: Fast and lightweight mel-spectrogram vocoder incorporating inverse short-time fourier transform. In ICASSP 2022- 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 6207–6211. IEEE, 2022.

Tero Karras, Miika Aittala, Samuli Laine, Erik Hark ¨ onen, Janne Hellsten, Jaakko Lehtinen, and Timo ¨ Aila. Alias-free generative adversarial networks. Advances in Neural Information Processing Systems, 34:852–863, 2021.

Hideki Kawahara, Ikuyo Masuda-Katsuse, and Alain De Cheveigne. Restructuring speech representations using a pitch-adaptive time–frequency smoothing and an instantaneous-frequency-based f0 extraction: Possible role of a repetitive structure in sounds. Speech communication, 27(3-4): 187–207, 1999.

Doh-Suk Kim. Perceptual phase quantization of speech. IEEE transactions on speech and audio processing, 11(4):355–364, 2003.

Jungil Kong, Jaehyeon Kim, and Jaekyoung Bae. Hifi-gan: Generative adversarial networks for efficient and high fidelity speech synthesis. Advances in Neural Information Processing Systems, 33:17022–17033, 2020.

Kundan Kumar, Rithesh Kumar, Thibault De Boissiere, Lucas Gestin, Wei Zhen Teoh, Jose Sotelo, Alexandre de Brebisson, Yoshua Bengio, and Aaron C Courville. Melgan: Generative adversarial ´ networks for conditional waveform synthesis. Advances in neural information processing systems, 32, 2019.

Sang-gil Lee, Wei Ping, Boris Ginsburg, Bryan Catanzaro, and Sungroh Yoon. Bigvgan: A universal neural vocoder with large-scale training. arXiv preprint arXiv:2206.04658, 2022.

Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 11976–11986, 2022.

Soroush Mehri, Kundan Kumar, Ishaan Gulrajani, Rithesh Kumar, Shubham Jain, Jose Sotelo, Aaron Courville, and Yoshua Bengio. Samplernn: An unconditional end-to-end neural audio generation model. arXiv preprint arXiv:1612.07837, 2016.

Takeru Miyato and Masanori Koyama. cgans with projection discriminator. arXiv preprint arXiv:1802.05637, 2018.

Masanori Morise, Fumiya Yokomori, and Kenji Ozawa. World: a vocoder-based high-quality speech synthesis system for real-time applications. IEICE TRANSACTIONS on Information and Systems, 99(7):1877–1884, 2016.

Max Morrison, Rithesh Kumar, Kundan Kumar, Prem Seetharaman, Aaron Courville, and Yoshua Bengio. Chunked autoregressive gan for conditional waveform synthesis. arXiv preprint arXiv:2110.10139, 2021.

Eric Moulines and Francis Charpentier. Pitch-synchronous waveform processing techniques for text-to-speech synthesis using diphones. Speech communication, 9(5-6):453–467, 1990.

Gautham J. Mysore. Daps (device and produced speech) dataset, May 2014. URL https://doi. org/10.5281/zenodo.4660670.

Aaron Oord, Yazhe Li, Igor Babuschkin, Karen Simonyan, Oriol Vinyals, Koray Kavukcuoglu, George Driessche, Edward Lockhart, Luis Cobo, Florian Stimberg, et al. Parallel wavenet: Fast high-fidelity speech synthesis. In International conference on machine learning, pp. 3918–3926. PMLR, 2018.

Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, and Koray Kavukcuoglu. Wavenet: A generative model for raw audio. arXiv preprint arXiv:1609.03499, 2016.

Keisuke Oyamada, Hirokazu Kameoka, Takuhiro Kaneko, Kou Tanaka, Nobukatsu Hojo, and Hiroyasu Ando. Generative adversarial network-based approach to signal reconstruction from magnitude spectrogram. In 2018 26th European Signal Processing Conference (EUSIPCO), pp. 2514–2518. IEEE, 2018.

Kuldip Paliwal, Kamil Wojcicki, and Benjamin Shannon. The importance of phase in speech ´ enhancement. speech communication, 53(4):465–494, 2011.

Marco Pasini and Jan Schluter. Musika! fast infinite waveform music generation. ¨ arXiv preprint arXiv:2208.08706, 2022.

Adam Polyak, Yossi Adi, Jade Copet, Eugene Kharitonov, Kushal Lakhotia, Wei-Ning Hsu, Abdelrahman Mohamed, and Emmanuel Dupoux. Speech resynthesis from discrete disentangled self-supervised representations. arXiv preprint arXiv:2104.00355, 2021.

Ryan Prenger, Rafael Valle, and Bryan Catanzaro. Waveglow: A flow-based generative network for speech synthesis. In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 3617–3621. IEEE, 2019.

Zafar Rafii, Antoine Liutkus, Fabian-Robert Stoter, Stylianos Ioannis Mimilakis, and Rachel Bittner. ¨ Musdb18-a corpus for music separation. 2017.

Antony W Rix, John G Beerends, Michael P Hollier, and Andries P Hekstra. Perceptual evaluation of speech quality (pesq)-a new method for speech quality assessment of telephone networks and codecs. In 2001 IEEE international conference on acoustics, speech, and signal processing. Proceedings (Cat. No. 01CH37221), volume 2, pp. 749–752. IEEE, 2001.

Takaaki Saeki, Detai Xin, Wataru Nakata, Tomoki Koriyama, Shinnosuke Takamichi, and Hiroshi Saruwatari. Utmos: Utokyo-sarulab system for voicemos challenge 2022. arXiv preprint arXiv:2204.02152, 2022.

Ibon Saratxaga, Inma Hernaez, Michael Pucher, Eva Navas, and Inaki Sainz. Perceptual importance ˜ of the phase related information in speech. In Thirteenth Annual Conference of the International Speech Communication Association, 2012.

Jonathan Shen, Ruoming Pang, Ron J Weiss, Mike Schuster, Navdeep Jaitly, Zongheng Yang, Zhifeng Chen, Yu Zhang, Yuxuan Wang, Rj Skerrv-Ryan, et al. Natural tts synthesis by conditioning wavenet on mel spectrogram predictions. In 2018 IEEE international conference on acoustics, speech and signal processing (ICASSP), pp. 4779–4783. IEEE, 2018.

Hubert Siuzdak, Piotr Dura, Pol van Rijn, and Nori Jacoby. WavThruVec: Latent speech representation as intermediate features for neural speech synthesis. In Proc. Interspeech 2022, pp. 833–837, 2022. doi: 10.21437/Interspeech.2022-10797.

Suno AI. Bark: Text-prompted generative audio model. https://github.com/suno-ai/ bark, 2023. GitHub repository.

Naoya Takahashi, Purvi Agrawal, Nabarun Goswami, and Yuki Mitsufuji. Phasenet: Discretized phase modeling with deep neural networks for audio source separation. In Interspeech, pp. 2713–2717, 2018.

Shinnosuke Takamichi, Yuki Saito, Norihiro Takamune, Daichi Kitamura, and Hiroshi Saruwatari. Phase reconstruction from amplitude spectrograms based on von-mises-distribution deep neural network. In 2018 16th International Workshop on Acoustic Signal Enhancement (IWAENC), pp. 286–290. IEEE, 2018.

Jean-Marc Valin and Jan Skoglund. Lpcnet: Improving neural speech synthesis through linear prediction. In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 5891–5895. IEEE, 2019.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Dequan Wang and Jae Lim. The unimportance of phase in speech enhancement. IEEE Transactions on Acoustics, Speech, and Signal Processing, 30(4):679–681, 1982.

Ye Wang and Mikka Vilermo. Modified discrete cosine transform: Its implications for audio coding and error concealment. Journal of the Audio Engineering Society, 51(1/2):52–61, 2003.

Donald S Williamson, Yuxuan Wang, and DeLiang Wang. Complex ratio masking for monaural speech separation. IEEE/ACM transactions on audio, speech, and language processing, 24(3): 483–492, 2015.

Takayoshi Yoshimura, Keiichi Tokuda, Takashi Masuko, Takao Kobayashi, and Tadashi Kitamura. Simultaneous modeling of spectrum, pitch and duration in hmm-based speech synthesis. In Sixth European Conference on Speech Communication and Technology, 1999.

Jaeseong You, Dalhyun Kim, Gyuhyeon Nam, Geumbyeol Hwang, and Gyeongsu Chae. Gan vocoder: Multi-resolution discriminator is all you need. arXiv preprint arXiv:2103.05236, 2021.

Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund, and Marco Tagliasacchi. Soundstream: An end-to-end neural audio codec. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 30:495–507, 2021.

Heiga Zen, Viet Dang, Rob Clark, Yu Zhang, Ron J Weiss, Ye Jia, Zhifeng Chen, and Yonghui Wu. Libritts: A corpus derived from librispeech for text-to-speech. arXiv preprint arXiv:1904.02882, 2019.

Liu Ziyin, Tilman Hartwig, and Masahito Ueda. Neural networks fail to learn periodic functions and how to fix it. Advances in Neural Information Processing Systems, 33:1583–1594, 2020.

# A MODIFIED DISCRETE COSINE TRANSFORM (MDCT)

While STFT is widely used in audio processing, there are other time-frequency representations with different properties. In audio coding applications, it is desirable to design the analysis/synthesis system in such a way that the overall rate at the output of the analysis stage equals the rate of the input signal. Such systems are described as being critically sampled. When we transform the signal via the DFT, even a slight overlap between adjacent blocks increases the data rate of the spectral representation of the signal. With $50 \%$ overlap between adjoining blocks, we end up doubling our data rate.

The Modified Discrete Cosine Transform (MDCT) with its corresponding Inverse Transform (IMDCT) have become a crucial tool in high-quality audio coding as they enable the implementation of a critically sampled analysis/synthesis filter bank. A key feature of these transforms is the Time-Domain Aliasing Cancellation (TDAC) property, which allows for the perfect reconstruction of overlapping segments from a source signal.

The MDCT is defined as follows:

$$
X [ k ] = \sum _ { n = 0 } ^ { 2 N - 1 } x [ n ] \cos \left[ { \frac { \pi } { N } } \left( n + { \frac { 1 } { 2 } } + { \frac { N } { 2 } } \right) \left( k + { \frac { 1 } { 2 } } \right) \right]
$$

for $k = 0 , 1 , \ldots , N - 1$ and $N$ is the length of the window.

The MDCT is a lapped transform and thus produces $N$ output coefficients from $2 N$ input samples, allowing for a $50 \%$ overlap between blocks without increasing the data rate.

There is a relationship between the MDCT and the DFT through the Shifted Discrete Fourier Transform (SDFT) (Wang & Vilermo, 2003). It can be leveraged to implement a fast version of the MDCT using FFT (Bosi & Goldberg, 2002). See Appendix A.3.

# A.1 VOCOS AND MDCT

MDCT is attractive in audio coding because of its its efficiency and compact representation of audio signals. In the context of deep learning, this might be seen as reduced dimensionality, potentially advantageous as it requires fewer data points during generation.

While STFT coefficients can be conveniently expressed in polar form, providing a clear interpretation of both magnitude and phase, MDCT represents the signal only in a real subspace of the complex space needed to accurately convey spectral magnitude and phase. Naive approach would be to treat raw unnormalized hidden outputs of the network as MDCT coefficients and convert it back to time-domain with IMDCT. In our preliminary experiments we found that it led to slower convergence. However we can easily observe that the MDCT spectrum, similarly to the STFT, can be more perceptually meaningful on the logarithmic scale, which reflects the logarithmic nature of human auditory perception of sound intensity. But as the MDCT can take also negative values, they cannot be represented using the conventional logarithmic transformation.

One solution is to utilize a symmetric logarithmic function. In the context of deep learning, Hafner et al. (2023) introduces such function and its inverse, referred to as symlog and symexp respectively:

$$
\operatorname { s y m l o g } ( x ) = \operatorname { s i g n } ( x ) \ln ( \left| x \right| + 1 ) \qquad \operatorname { s y m e x p } ( x ) = \operatorname { s i g n } ( x ) ( \exp ( \left| x \right| ) - 1 )
$$

The symlog function compresses the magnitudes of large values, irrespective of their sign. Unlike the conventional logarithm, it is symmetric around the origin and retains the input sign. We note the correspondence with the $\mu$ -law companding algorithm, a well-established method in telecommunication and signal processing.

An alternative approach involves parametrizing the model to output the absolute value of the MDCT coefficients and its corresponding sign. While the MDCT does not directly convey information about phase relationships, this strategy may offer advantages as the sign of the MDCT can potentially provide additional insights indirectly. For example, an opposite sign could imply a phase difference of 180 degrees. In practice, we compute a ”soft” sign using the cosine activation function, which supposedly provides a periodic inductive bias. Hence, similar to the ISTFT head, this approach projects the hidden activations into two values for each frequency bin, representing the final coefficients as $\mathbf { M D C T } = \exp ( \mathbf { m } ) \cdot \cos ( \mathbf { p } )$ .

Table 7: Objective evaluation metrics for MDCT variant of Vocos compared to the ISTFT baseline.   

<table><tr><td></td><td>UTMOS (↑)</td><td>PESQ (↑)</td><td>V/UV F1 (↑)</td><td>Periodicity (↓)</td></tr><tr><td>Ground truth</td><td>4.058</td><td></td><td></td><td>−</td></tr><tr><td>Baseline (ISTFT)</td><td>3.734</td><td>3.70</td><td>0.9582</td><td>0.101</td></tr><tr><td>IMDCT (symexp)</td><td>3.498</td><td>3.648</td><td>0.9569</td><td>0.106</td></tr><tr><td>IMDCT (sign)</td><td>3.536</td><td>3.565</td><td>0.9547</td><td>0.109</td></tr></table>

# A.2 RESULTS

Table 7 presents objective evaluation metrics for a variant of Vocos that represents audio samples with MDCT coefficients. Both ’symexp’ and ’sign’ demonstrate significantly weaker performance compared to their STFT-based counterpart. This suggests that while MDCT may be attractive in audio coding applications, its properties may not be as favorable in the context of generative modeling with GANs. The redundancy inherent in the STFT representation appears to be beneficial for generative tasks. This finding aligns with the work of Gritsenko et al. (2020), who discovered that an overcomplete Fourier basis contributed to improved training stability. Furthermore, it is worth noting that the MDCT, being a lapped transform, incorporates information from surrounding windows, which effectively act as aliases of the signal. To ensure Time Domain Alias Cancellation (TDAC), the prediction of the coefficients has to be accurate and consistent over the frames.

# A.3 FORWARD MDCT ALGORITHM

<table><tr><td colspan="3">Algorithm 1 Fast MDCT Algorithm realized with FFT</td></tr><tr><td>1: Input: Audio signal x with frame length N</td><td></td><td></td></tr><tr><td>2: 3:</td><td>Output: MDCT coefficients X procedure MDCT(x)</td><td></td></tr><tr><td>4:</td><td>for each frame f in x with overlap of N/2 do</td><td></td></tr><tr><td>5:</td><td>f ← f × window function</td><td></td></tr><tr><td>6:</td><td>f ← f × e−j 2πn</td><td> Pre-twiddle</td></tr><tr><td>7:</td><td>f ←FFT(f)</td><td> N-point FFT</td></tr><tr><td>8:</td><td>f ← f × e−j π no(k+ 1 )</td><td> Post-twiddle</td></tr><tr><td>9:</td><td>f ← f √\fr  }$</td><td></td></tr><tr><td>10:</td><td>Xk ← R(f) × √2</td><td></td></tr><tr><td>11:</td><td>end for</td><td></td></tr><tr><td>12:</td><td>return X</td><td></td></tr><tr><td>13:</td><td></td><td></td></tr><tr><td></td><td>end procedure</td><td></td></tr></table>