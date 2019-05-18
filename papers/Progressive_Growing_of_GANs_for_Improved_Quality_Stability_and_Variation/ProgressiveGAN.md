# ■ 論文
- 論文タイトル："Progressive Growing of GANs for Improved Quality, Stability, and Variation"
- 論文リンク：https://arxiv.org/abs/1710.10196
- 論文投稿日付：2017/10/27(v1), 2018/02/26(v3)
- 著者（組織）：(NVIDIA)
- categories：

# ■ 概要（何をしたか？）

## ABSTRACT

- We describe a new training methodology for generative adversarial networks.
    - 我々は、GAN のための新しいトレーニング手法を説明する。

- The key idea is to grow both the generator and discriminator progressively: starting from a low resolution, we add new layers that model increasingly fine details as training progresses.
    - **キーとなるアイデアは、生成器と識別器の両方を、次第に [progressively] 成長させるといことである。**
    - **即ち、低解像度から開始して、学習が進むにつれて、細かい詳細を増加させるモデル化する [model] ような新しい層を追加する。**

- This both speeds the training up and greatly stabilizes it, allowing us to produce images of unprecedented quality, e.g., CELEBA images at 1024^2.
    - この両方は、学習をスピードアップし、それを非常に安定化させ、
    - **前例のない [unprecedented] 品質の画像を生成することを許容する。**

- We also propose a simple way to increase the variation in generated images, and achieve a record inception score of 8.80 in unsupervised CIFAR10.
    - 我々はまた、生成された画像において、変動を増加させるための単純な方法を提案する。
    - そして、教師なしの CIFAR10 において、8.80 の inception score の記録を達成する。

- Additionally, we describe several implementation details that are important for discouraging unhealthy competition between the generator and discriminator.
    - 加えて、生成器と識別器の間の、不健全な [unhealthy] 競争を妨げる [discouraging] ために重要となる、いくつかの実装の詳細を説明する。

- Finally, we suggest a new metric for evaluating GAN results, both in terms of image quality and variation.
    - 最後に、我々は、画像の品質と変動に関して、GAN を評価するための新しい距離を提案する。

- As an additional contribution, we construct a higher-quality version of the CELEBA dataset.
    - 追加の貢献として、CELEBA データセットのより高い品質のバージョンを構築する。


# ■ イントロダクション（何をしたいか？）

## 1. INTRODUCTION

- Generative methods that produce novel samples from high-dimensional data distributions, such as images, are finding widespread use, for example in speech synthesis (van den Oord et al., 2016a), image-to-image translation (Zhu et al., 2017; Liu et al., 2017; Wang et al., 2017), and image inpainting (Iizuka et al., 2017).
    - 高次元のデータ分布からの新しいサンプル（例えば画像）を生成するような生成手法は、広く使われている。
    - 例えば、音声合成 [speech synthesis]、image-to-image 変換、画像修復 [image inpainting]

- Currently the most prominent approaches are autoregressive models (van den Oord et al., 2016b;c), variational autoencoders (VAE) (Kingma & Welling, 2014), and generative adversarial networks (GAN) (Goodfellow et al., 2014).
    - 現在では、最も約束されたアプローチは、自己回帰モデル(autoregressive models), VAE, GAN である。

- Currently they all have significant strengths and weaknesses. 
    - 現在のところ、それらは全て重要な強みと弱みをもっている。

- Autoregressive models – such as PixelCNN – produce sharp images but are slow to evaluate and do not have a latent representation as they directly model the conditional distribution over pixels, potentially limiting their applicability.
    - PixelCNN のような自己回帰モデルは、鮮明な画像を生成する。
    - しかし、評価が遅く、
    - ピクセル上の条件付き確率分布を直接モデル化するような潜在表現を持たず、
    - 適用性が制限される可能性がある。

- VAEs are easy to train but tend to produce blurry results due to restrictions in the model, although recent work is improving this (Kingma et al., 2016).
    - VAE は、学習が容易であるが、
    - モデルの制約のために、ぼやけた結果になる傾向がある。
    - 最近の研究では、改善されてはいるが、

- GANs produce sharp images, albeit only in fairly small resolutions and with somewhat limited variation, and the training continues to be unstable despite recent progress (Salimans et al., 2016; Gulrajani et al., 2017; Berthelot et al., 2017; Kodali et al., 2017).
    - GAN は、鮮明な画像を生成する。
    - とはいえ [albeit]、かなり [fairly] 小さい解像度であり、
    - いくらか変動が制限され、
    - 最近の進歩にも関わらず学習が不安定である。

- Hybrid methods combine various strengths of the three, but so far lag behind GANs in image quality (Makhzani & Frey, 2017; Ulyanov et al., 2017; Dumoulin et al., 2016).
    - ハイブリッド手法は、３つの手法の強みを様々に組み合わせたものであるが、
    - これまでのところ、画像品質において、GAN に比べて劣っている。

---

- Typically, a GAN consists of two networks: generator and discriminator (aka critic).
    - 典型的には、GAN は２つのネットワークから構成される。
    - 即ち、生成器と識別器（別名 [aka]、クリティック）

- The generator produces a sample, e.g., an image, from a latent code, and the distribution of these images should ideally be indistinguishable from the training distribution.
    - 生成器は、１つのサンプル（例えば画像）を、潜在変数から生成する。
    - そして、これらの（生成）画像の分布は、理想的には、学習データの分布（真の分布）と区別がつかないべきである。

- Since it is generally infeasible to engineer a function that tells whether that is the case, a discriminator network is trained to do the assessment, and since networks are differentiable, we also get a gradient we can use to steer both networks to the right direction.
    - このような状況（分布の区別がつかない状況）かになっているかを教えるような関数を設計する [engineer] ことは、実現不可能 [infeasible] であるため、
    - 識別器は、評価 [assessment] をすることを学習し、
    - ネットワークが微分可能であるので、両方のネットワークを正しい方向に導く [steer] ために使用できる勾配を得ることも出来る。

- Typically, the generator is of main interest – the discriminator is an adaptive loss function that gets discarded once the generator has been trained.
    - 典型的には、生成器は、主に重要である？
    - つまり、識別器は、生成器が一度学習されると捨てられる [discarded] ような adaptive 損失関数をもつ。

---

- There are multiple potential problems with this formulation.
    - この定式化には、複数の潜在的な問題がある。

- When we measure the distance between the training distribution and the generated distribution, the gradients can point to more or less random directions if the distributions do not have substantial overlap, i.e., are too easy to tell apart (Arjovsky & Bottou, 2017).
    - 学習データの分布と生成画像の分布との間の距離を測るとき、
    - もし、分布が実質的な重なりあいを持っていない（即ち、区別が容易すぎる）ならば、
    - 勾配は、多かれ少なかれランダムな方向になる。

- Originally, Jensen-Shannon divergence was used as a distance metric (Goodfellow et al., 2014), and recently that formulation has been improved (Hjelm et al., 2017) and a number of more stable alternatives have been proposed, including least squares (Mao et al., 2016b), absolute deviation with margin (Zhao et al., 2017), and Wasserstein distance (Arjovsky et al., 2017; Gulrajani et al., 2017).
    - 元々 [Originally] には、JSダイバージェンスが距離の指標として使用され、
    - 最近にその定式化が改善され、
    - 最小２乗法を含むような、いくつのより安定化した代替案が提案された。
    - マージン付き絶対偏差、Wasserstein GAN

- Our contributions are largely orthogonal to this ongoing discussion, and we primarily use the improved Wasserstein loss, but also experiment with least-squares loss.
    - 我々の貢献は、この進行中の議論と、大部分で [largely ] で直交しており、（＝別方向で）、
    - 主として [primarily]、改善された Wasserstein loss を使用するが、
    - 最小２乗法誤差での実験も行う。

---

- The generation of high-resolution images is difficult because higher resolution makes it easier to tell the generated images apart from training images (Odena et al., 2017), thus drastically amplifying the gradient problem.
    - **高解像度の画像の生成は、困難である。**
    - **なぜならば、より高い解像度は、生成画像と学習画像の識別をより簡単にするからである。**

- Large resolutions also necessitate using smaller minibatches due to memory constraints, further compromising training stability.
    - **大きな解像度ではまた、メモリ制限のために、より小さなミニバッチを使用する必要があり、[necessitate]**
    - **更に、学習性を落とす。[compromising]**

- Our key insight is that we can grow both the generator and discriminator progressively, starting from easier low-resolution images, and add new layers that introduce higher-resolution details as the training progresses.
    - **我々のキーとなる洞察は、生成器と識別器を、より簡単な低解像度の画像から開始して、段階的に成長させ、**
    - **学習が進むにつれて、高解像の詳細情報をもつよぅな新しい層を追加するというものである。**

- This greatly speeds up training and improves stability in high resolutions, as we will discuss in Section 2.
    - セクション２で述べるように、
    - これにより、学習が劇的にスピードアップし、高解像度においての安定性が向上する。

---

- The GAN formulation does not explicitly require the entire training data distribution to be represented by the resulting generative model.
    - GAN の定式化は、得られた生成モデルによって表現された？、
    - 学習データ分布の全体を、明確に [explicitly] 要求しない。

- The conventional wisdom has been that there is a tradeoff between image quality and variation, but that view has been recently challenged (Odena et al., 2017).
    - 従来の知見は、画像品質と変動との間にトレードオフがあるというものであった。
    - しかし、この見方は最近変わった。

- The degree of preserved variation is currently receiving attention and various methods have been suggested for measuring it, including inception score (Salimans et al., 2016), multi-scale structural similarity (MS-SSIM) (Odena et al., 2017; Wang et al., 2003), birthday paradox (Arora & Zhang, 2017), and explicit tests for the number of discrete modes discovered (Metz et al., 2016).
    - 保存された変動の程度は、最近注目を集めており、
    - それを計測するための様々な手法が提案されている。
    - inception score, multi-scale structural similarity (MS-SSIM), birthday paradox, 

- We will describe our method for encouraging variation in Section 3, and propose a new metric for evaluating the quality and variation in Section 5.
    - セクション３で、変動を促進する [encouraging] ための我々の手法を説明する。
    - そして、セクション５で、品質と変動を評価するための新しい手法を提案する。

---

- Section 4.1 discusses a subtle modification to the initialization of networks, leading to a more balanced learning speed for different layers.
    - セクション4.1では、ネットワークの初期化の繊細な [subtle] 修正について議論し、
    - 異なる層に対して、よりバランスのとれた学習速度を導く。

- Furthermore, we observe that mode collapses traditionally plaguing GANs tend to happen very quickly, over the course of a dozen minibatches.
    - 更には、伝統的に厄介な [plaguing] モード崩壊が、GANに、ミニバッチの過程で、とても早く発生する傾向があるということを観測する。

- Commonly they start when the discriminator overshoots, leading to exaggerated gradients, and an unhealthy competition follows where the signal magnitudes escalate in both networks.
    - 一般的に、それらは識別器が行き過ぎる [overshoots] とき、誇大な [exaggerated] 勾配を導くとき、
    - そして、両方のネットワークにおいて、信号の値 [magnitude] がエスカレート [escalate] するような場所で、不健全な競争が発生する。

- We propose a mechanism to stop the generator from participating in such escalation, overcoming the issue (Section 4.2).
    - 我々は、このようなエスカレートへの生成器の参加を止めるためのメカニズムを提供し、
    - この問題を解決する。（セクション4.2）

---

- We evaluate our contributions using the CELEBA, LSUN, CIFAR10 datasets.

- We improve the best published inception score for CIFAR10.

- Since the datasets commonly used in benchmarking generative methods are limited to a fairly low resolution, we have also created a higher quality version of the CELEBA dataset that allows experimentation with output resolutions up to 1024 × 1024 pixels.
    - 生成モデルのベンチマークで一般的に使用されるデータセットは、かなり低い解像度に制限されているので、私達はまた最大1024×1024ピクセルまでの出力解像度での実験を可能にするCELEBAデータセットの高品質バージョンを作成しました。

- This dataset and our full implementation are available at https://github.com/tkarras/progressive_growing_of_gans, trained networks can be found at https://drive.google.com/open?id=0B4qLcYyJmiz0NHFULTdYc05lX0U along with result images, and a supplementary video illustrating the datasets, additional results, and latent space interpolations is at https://youtu.be/G06dEcZ-QTg.

# ■ 結論

## x. 論文の項目名 (Conclusion)


# ■ 何をしたか？詳細

## 2. PROGRESSIVE GROWING OF GANS

- Our primary contribution is a training methodology for GANs where we start with low-resolution images, and then progressively increase the resolution by adding layers to the networks as visualized in Figure 1.
    - 我々の貢献は、GAN への学習の方法論 [methodology] である。
    - （この方法論というのは、）図１に示したように、低解像度の画像から開始して、ネットワークに層を追加することによって、段階的に解像度を増加させるものである。

- This incremental nature allows the training to first discover large-scale structure of the image distribution and then shift attention to increasingly finer scale detail, instead of having to learn all scales simultaneously.
    - この漸近的な [incremental] 性質 [nature] は、
    - 学習に、画像の分布の大きなスケールでの構造を、最初に発見することを許容する。
    - （＝学習は、最初に、大規模な画像分布の構造を発見する。）
    - そして次に、同時に全てのスケール学習する代わりに、ますます [increasingly] 細かいスケールでの詳細に注意を向ける。

![image](https://user-images.githubusercontent.com/25688193/57967249-61d6cb00-7997-11e9-96cc-927ebf85f4aa.png)

- > Figure 1: Our training starts with both the generator (G) and discriminator (D) having a low spatial resolution of 4×4 pixels.
    - > 図１：我々の学習は、4×4 ピクセルの低い空間的な [spatial] 解像度を持つような、生成器（G）と識別器（D）の両方からスタートする。

- > As the training advances, we incrementally add layers to G and D, thus increasing the spatial resolution of the generated images.
    - > 学習が進むにつれて、G と D に段階的に層を追加する。それ故、生成画像の空間的な解像度が増加する。

- > All existing layers remain trainable throughout the process.
    - > 全ての存在する層は、プロセスを通じて、学習可能なままである。

- > Here N × N refers to convolutional layers operating on N × N spatial resolution.
    - > ここで、N × N は、N × N の空間的な解像度で動作する畳み込み層を示している。

- > This allows stable synthesis in high resolutions and also speeds up training considerably.
    - > このことは、高解像度での安定した合成 [synthesis] が可能になり、学習を大幅に高速化する。

- > One the right we show six example images generated using progressive growing at 1024 × 1024.
    - > 右側に、1024×1024 での進歩的な [progressive] 成長を使用して生成された、６つのサンプルの画像を示す。

---

- We use generator and discriminator networks that are mirror images of each other and always grow in synchrony.
    - 互いにミラー画像で、常に同調して [synchrony] 成長するような、生成器と識別器を使用する。

- All existing layers in both networks remain trainable throughout the training process.
    - **両方のネットワークの中で全ての存在する層は、学習プロセスを通じて、学習可能なままである。**

- When new layers are added to the networks, we fade them in smoothly, as illustrated in Figure 2.
    - **新しい層がネットワークに追加されるとき、図２に示したように、それら滑らかに減衰 [fade] させる。**

- This avoids sudden shocks to the already well-trained, smaller-resolution layers. 
    - **このことは、既にうまく学習されている、より小さい解像度での層への、突然の衝撃を回避する。**

- Appendix A describes structure of the generator and discriminator in detail, along with other training parameters.
    - 補足 A は、他の学習パラメーターと共に、生成器と識別器の構造を詳細に説明している。

![image](https://user-images.githubusercontent.com/25688193/57967792-dca2e480-799d-11e9-9347-e64b4ec4f790.png)

- > Figure 2: When doubling the resolution of the generator (G) and discriminator (D) we fade in the new layers smoothly.
    - > 図２：生成器と識別器の解像度を２倍にするとき、新しい層にスムーズに fade する。

- > This example illustrates the transition from 16 × 16 images (a) to 32 × 32 images (c). 
    - > この例は、16 × 16 の画像 (a) から、32 × 32 の画像 (c) への変換を図示している。

- > During the transition (b) we treat the layers that operate on the higher resolution like a residual block, whose weight α increases linearly from 0 to 1.
    - > **変換 (b) の間に、重み α が、0 から 1 へ線形的に増加する残差ブロック [residual block] のような、より高い解解像度への演算するような層を扱う。**

> residual block : ResNet で使われている block

- > Here 2× and 0.5× refer to doubling and halving the image resolution using nearest neighbor filtering and average pooling, respectively.
    - > <font color="Pink">**ここで、2× と 0.5× は、それぞれ、nearest neighbor filtering と average pooling を使用して、画像解像度を２倍と半分にすることを示している。**</font>

> nearest neighbor filtering ? ：

> average pooling : golobal average pooling のこと？

- > The toRGB represents a layer that projects feature vectors to RGB colors and fromRGB does the reverse; both use 1 × 1 convolutions.
    - > toRGB は、特徴ベクトルをRGB色へ投影する層を表現しており、fromRGB はその逆wお行う層を表している。即ち、両方共 1 × 1 の畳み込みを使用する。

- > When training the discriminator, we feed in real images that are downscaled to match the current resolution of the network.
    - > 識別器を学習するとき、ネットワークの現在の解像度にマッチするようにダウンスケールされた本物画像を供給する。

- > During a resolution transition, we interpolate between two resolutions of the real images, similarly to how the generator output combines two resolutions.
    - > 解像度の変換をしている間に、本物画像の２つの解像度の間を補完する。
    - > 生成器が、どのようにして２つの解像度を組み合わせて出力するのかと同様にして、

---

- We observe that the progressive training has several benefits.

- Early on, the generation of smaller images is substantially more stable because there is less class information and fewer modes (Odena et al., 2017).

- By increasing the resolution little by little we are continuously asking a much simpler question compared to the end goal of discovering a mapping from latent vectors to e.g. 1024^2 images. 

- This approach has conceptual similarity to recent work by Chen & Koltun (2017).

- In practice it stabilizes the training sufficiently for us to reliably synthesize megapixel-scale images using WGAN-GP loss (Gulrajani et al., 2017) and even LSGAN loss (Mao et al., 2016b).

---

- Another benefit is the reduced training time.

- With progressively growing GANs most of the iterations are done at lower resolutions, and comparable result quality is often obtained up to 2–6 times faster, depending on the final output resolution.

---


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名

## 7. DISCUSSION

- While the quality of our results is generally high compared to earlier work on GANs, and the training is stable in large resolutions, there is a long way to true photorealism. 
    - 我々の結果の

- Semantic sensibility and understanding dataset-dependent constraints, such as certain objects being straight rather than curved, leaves a lot to be desired.

- There is also room for improvement in the micro-structure of the images.

- That said, we feel that convincing realism may now be within reach, especially in CELEBA-HQ.


# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


