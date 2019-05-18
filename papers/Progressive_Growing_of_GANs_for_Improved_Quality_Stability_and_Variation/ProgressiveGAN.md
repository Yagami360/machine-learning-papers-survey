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



# ■ 結論

## x. 論文の項目名 (Conclusion)


# ■ 何をしたか？詳細

## x. 論文の項目名


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


