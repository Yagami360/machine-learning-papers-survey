> 論文まとめノート：https://github.com/Yagami360/My_NoteBook/blob/master/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6_%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_%E7%94%9F%E6%88%90%E3%83%A2%E3%83%87%E3%83%AB.md#ProgressiveGAN%EF%BC%88PGGAN%EF%BC%89

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
    - 我々はまた、生成された画像において、変動（多様性）を増加させるための単純な方法を提案する。
    - そして、教師なしの CIFAR10 において、8.80 の inception score の記録を達成する。

- Additionally, we describe several implementation details that are important for discouraging unhealthy competition between the generator and discriminator.
    - 加えて、生成器と識別器の間の、不健全な [unhealthy] 競争を妨げる [discouraging] ために重要となる、いくつかの実装の詳細を説明する。

- Finally, we suggest a new metric for evaluating GAN results, both in terms of image quality and variation.
    - 最後に、我々は、画像の品質と変動（多様性）に関して、GAN を評価するための新しい距離を提案する。

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
    - いくらか変動（多様性）が制限され、
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
    - 従来の知見は、画像品質と変動（多様性）との間にトレードオフがあるというものであった。
    - しかし、この見方は最近変わった。

- The degree of preserved variation is currently receiving attention and various methods have been suggested for measuring it, including inception score (Salimans et al., 2016), multi-scale structural similarity (MS-SSIM) (Odena et al., 2017; Wang et al., 2003), birthday paradox (Arora & Zhang, 2017), and explicit tests for the number of discrete modes discovered (Metz et al., 2016).
    - 保存された変動（多様性）の程度は、最近注目を集めており、
    - それを計測するための様々な手法が提案されている。
    - inception score, multi-scale structural similarity (MS-SSIM), birthday paradox, 

- We will describe our method for encouraging variation in Section 3, and propose a new metric for evaluating the quality and variation in Section 5.
    - セクション３で、変動（多様性）を促進する [encouraging] ための我々の手法を説明する。
    - そして、セクション５で、品質と変動（多様性）を評価するための新しい手法を提案する。

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

## 7. DISCUSSION

- While the quality of our results is generally high compared to earlier work on GANs, and the training is stable in large resolutions, there is a long way to true photorealism. 
    - 我々の結果の質はGANに関する以前の研究と比較して一般的に高く、そして訓練は大きな解像度で安定していますが、真のフォトリアリズムへの長い道のりがあります。

- Semantic sensibility and understanding dataset-dependent constraints, such as certain objects being straight rather than curved, leaves a lot to be desired.
    - あるオブジェクトが曲線ではなくまっすぐであるなどの意味的な感度とデータセットに依存する制約を理解することは、望まれるべき多くのことを残します。

- There is also room for improvement in the micro-structure of the images.
    - 画像の微細構造にも改良の余地がある。

- That said, we feel that convincing realism may now be within reach, especially in CELEBA-HQ.
    - とは言っても、説得力のあるリアリズムは、特にCELEBA-HQにおいて手の届くところにあるかもしれないと我々は感じています。


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
    - 進行的な学習は、いくつかの利点がある。

- Early on, the generation of smaller images is substantially more stable because there is less class information and fewer modes (Odena et al., 2017).
    - 初期段階では [Early on]、より小さな画像の生成は、より安定化する。
    - なぜならば、より小さな情報とより少ないモードが存在するためである。

- By increasing the resolution little by little we are continuously asking a much simpler question compared to the end goal of discovering a mapping from latent vectors to e.g. 1024^2 images. 
    - **解像度を少しずつ増加させることにより、**
    - **1024×1024 の画像の潜在ベクトルからの写像を発見するという最終ゴールと比較して、（我々の手法は、）もっとシンプルな質問を問っている。**

- This approach has conceptual similarity to recent work by Chen & Koltun (2017).
    - このアプローチは、Chen & Koltun による最近の研究に、概念的に [conceptual] 似ている。

- In practice it stabilizes the training sufficiently for us to reliably synthesize megapixel-scale images using WGAN-GP loss (Gulrajani et al., 2017) and even LSGAN loss (Mao et al., 2016b).
    - 実用的には、これは、
    - WGAN-GP loss と更には LSGAN loss を使用したメガピクセルのスケールでの画像を確実に合成するための、
    - 学習を十分に安定化させる。

---

- Another benefit is the reduced training time.
    - 他の利点は、学習時間を減らすことである。

- With progressively growing GANs most of the iterations are done at lower resolutions, and comparable result quality is often obtained up to 2–6 times faster, depending on the final output resolution.
    - 進歩的に成長する GAN では、ほとんどのイテレーションが、低解像度で行われる。
    - 同程度の [comparable] 結果の品質が、最終的な出力の解像度に依存して、2-6 倍速く手に入る。

---

- The idea of growing GANs progressively is related to the work of Wang et al. (2017), who use multiple discriminators that operate on different spatial resolutions.
    - 進歩的に成長する GAN のアイデアは、Wang の研究に関連がある。
    - （この研究というのは、）異なる空間的な解像度を演算するような複数の識別器を使用するものである。

- That work in turn is motivated by Durugkar et al. (2016) who use one generator and multiple discriminators concurrently, and Ghosh et al. (2017) who do the opposite with multiple generators and one discriminator.
    - この研究は、次には、１つの生成器と複数の識別器を同時に [concurrently] 使用するような、Durugkar による（研究）で動機づけられている。
    - そして、それとは反対に、複数の生成器と１つの識別器を使用するような、Ghosh による（研究）で動機づけられている。

- Hierarchical GANs (Denton et al., 2015; Huang et al., 2016; Zhang et al., 2017) define a generator and discriminator for each level of an image pyramid.
    - 階層的 GAN は、１つの画像ピラミッド [pyramid] の各レベルに対して、生成器と識別器を定義する。

- These methods build on the same observation as our work – that the complex mapping from latents to high-resolution images is easier to learn in steps – but the crucial difference is that we have only a single GAN instead of a hierarchy of them.
    - これらの手法は、我々の研究のように、同じ観測 [observation] に基づいている [build on]。
    - 即ち、潜在変数から高解像度の画像への複雑な写像は、段階的に [in step] 学習することがより簡単である。
    - しかし、極めて重要な [crucial] 違いは、それらの階層性の代わりに、我々の GAN は、単一の GAN のみであることである。

- In contrast to early work on adaptively growing networks, e.g., growing neural gas (Fritzke, 1995) and neuro evolution of augmenting topologies (Stanley & Miikkulainen, 2002) that grow networks greedily, we simply defer the introduction of pre-configured layers.
    - 適合的に成長するネットワークの初期の研究（例えば、growing neural gas や 貪欲に成長するネットワークである neuro evolution of augmenting topologies）とは対称的に、
    - <font color="Pink">**我々の手法は、単純に、pre-configured layers（事前設定層？）を、単純に延期する [defer]。**</font>

- In that sense our approach resembles layer-wise training of autoencoders (Bengio et al., 2007).
    - そういった意味では [In that sense]、我々のアプローチは、オートエンコーダーの層単位での学習に似ている。


## 3 INCREASING VARIATION USING MINIBATCH STANDARD DEVIATION

- GANs have a tendency to capture only a subset of the variation found in training data, and Salimans et al. (2016) suggest “minibatch discrimination” as a solution.
    - GAN は、学習データの中で見つかるような、変動（多様性）のサブセット（部分集合）のみを抽出する傾向 [tendency] がある。
    - そして、Salimans は、解決策として、“minibatch discrimination” を提案している。

> モード崩壊のこと

- They compute feature statistics not only from individual images but also across the minibatch, thus encouraging the minibatches of generated and training images to show similar statistics.
    - それらの手法は、個々の画像だけでなく、ミニバッチを渡って、特徴量の統計量を計算する。
    - それ故、生成された画像や学習した画像が、よく似た統計量を表示するように促進する [encouraging]。

- This is implemented by adding a minibatch layer towards the end of the discriminator, where the layer learns a large tensor that projects the input activation to an array of statistics.
    - この手法は、識別器の終端に、ミニバッチ層を追加することによって、実装される。
    - ここでミニバッチ層は、入力活性化関数を、統計量の配列へ投影するような、大きなテンソルを学習する。

- A separate set of statistics is produced for each example in a minibatch and it is concatenated to the layer’s output, so that the discriminator can use the statistics internally.
    - 統計量の個別の [separate] 集合は、ミニバッチデータの各サンプルに対して、生成される。
    - そして、それは、識別器が内部で統計量を使用出来るように、層の出力に結合される。

- We simplify this approach drastically while also improving the variation.
    - 我々は、多様性を改善しながら [while]、このアプローチを劇的に単純化する。

---

- Our simplified solution has neither learnable parameters nor new hyperparameters.
    - 我々の単純化された解決法は、学習可能なパラメーターだけでなく、新しいハイパーパラメータも持っていない。

- We first compute the standard deviation for each feature in each spatial location over the minibatch.
    - 我々の手法では、初めに、ミニバッチに渡って、各空間的な位置で、各特徴量に対して標準偏差を計算する。

- We then average these estimates over all features and spatial locations to arrive at a single value.
    - 全ての特徴量と空間的な位置に渡って、１つの値に到着するために、これらの推定値を平均化する。

- We replicate the value and concatenate it to all spatial locations and over the minibatch, yielding one additional (constant) feature map.
    - その値を複製 [replicate] し、それを、ミニバッチに渡っての全ての空間的な位置で結合し、
    - １つの追加の(定数の）特徴マップを生み出す。

- This layer could be inserted anywhere in the discriminator, but we have found it best to insert it towards the end (see Appendix A.1 for details).
    - この層は、識別器のどの場所にでも挿入することが出来る。
    - しかし、我々は、末端に挿入することがベストであることを見つけ出した。（詳細は、補足 A.1 参照）

- We experimented with a richer set of statistics, but were not able to improve the variation further.
    - 我々は、豊富な統計量のセットで実験した。
    - しかし、変動をこれ以上に改善することは出来なかった。

- In parallel work, Lin et al. (2017) provide theoretical insights about the benefits of showing multiple images to the discriminator.
    - 並行的な研究では、Lin は、識別機に複数の画像を見せることの利点についての、理論的な洞察を提供する。

---

- Alternative solutions to the variation problem include unrolling the discriminator (Metz et al., 2016) to regularize its updates, and a “repelling regularizer” (Zhao et al., 2017) that adds a new loss term to the generator, trying to encourage it to orthogonalize the feature vectors in a minibatch.
    - 多様性問題の代替の解決法は、
    - 更新を規則化（正則化）する unrolling the discriminator や、
    - 生成器に新しい損失項を追加する “repelling regularizer” を含んでおり、
    - ミニバッチにおいて、特徴ベクトルを直交化するために、それを促進する。

- The multiple generators of Ghosh et al. (2017) also serve a similar goal.
    - Ghosh の複数の生成器もまた、似たゴールの目的を果たしている [serve]。

- We acknowledge that these solutions may increase the variation even more than our solution – or possibly be orthogonal to it – but leave a detailed comparison to a later time.
    - 我々は、これらの解決法が、我々の解決法よりも、多様性を増加させるだろうこと、は認識している [acknowledge]。
    - 或いは、それを直交化させる可能性については認識している。
    - しかし、詳細な比較は、後に残す。


## 4 NORMALIZATION IN GENERATOR AND DISCRIMINATOR

- GANs are prone to the escalation of signal magnitudes as a result of unhealthy competition between the two networks.
    - GAN は、２つのネットワークの間の不健全な競争の結果として、単一の大きさ [magnitudes] が増大する傾向がある [prone]。

> 生成器と識別器のどちらかが強くなりすぎて、単一の大きさ（偽物画像 or 識別結果）が増大するという意味

- Most if not all earlier solutions discourage this by using a variant of batch normalization (Ioffe & Szegedy, 2015; Salimans & Kingma, 2016; Ba et al., 2016) in the generator, and often also in the discriminator.
    - 全てではないにしろほとんどの [Most if not all]、初期の解決法では、- 生成器において、そしてしばしば識別器においても、
    - batch norm の変動を使用することによって、これを阻止する [discourage]。

- These normalization methods were originally introduced to eliminate covariate shift.
    - これらの正規化手法は、元々は、共変量シフト [covariate shift] を除外する [eliminate] ために紹介されている。

- However, we have not observed that to be an issue in GANs, and thus believe that the actual need in GANs is constraining signal magnitudes and competition.
    - しかしながら、GAN において、（共変量シフトが）１つの問題となることを観測しない。
    - そしてそれ故に、GAN において本当に必要なのは、単一の大きさ [magnitudes] と競争を、抑制すること [constraining] であると信じている。

- We use a different approach that consists of two ingredients, neither of which include learnable parameters.
    - 我々は、２つの要素 [ingredients] から構成される異なるアプローチを使用する。
    - その両方共、学習可能なパラメーターを含んでいない。

### 4.1 EQUALIZED LEARNING RATE

- We deviate from the current trend of careful weight initialization, and instead use a trivial N(0,1) initialization and then explicitly scale the weights at runtime.
    - 現在の注意深い重み初期化の傾向から脱却し [deviate from]、
    - 代わりに、自明な [trivial] N(0,1) での初期化を使用し、
    - そして次に、実行時に [at runtime]、明示的に [explicitly] 重みスケールを使用する。

- To be precise, we set ![image](https://user-images.githubusercontent.com/25688193/57992711-a3c75480-7af0-11e9-877e-72853fdfa0ff.png), where w_i are the weights and c is the per-layer normalization constant from He’s initializer (He et al., 2015).
    - 正確に言えば [To be precise]、![image](https://user-images.githubusercontent.com/25688193/57992711-a3c75480-7af0-11e9-877e-72853fdfa0ff.png) を設定する。
    - ここで、w_i は、重みで、
    - c は、その初期化からの層単位での正規化定数である。

- The benefit of doing this dynamically instead of during initialization is somewhat subtle, and relates to the scale-invariance in commonly used adaptive stochastic gradient descent methods such as RMSProp (Tieleman & Hinton, 2012) and Adam (Kingma & Ba, 2015).
    - 初期化中の代わりに、この動的な処理を行うことの利点は、いくらか微妙 [subtle] であり、
    - RMSProp や Adam のような、一般的に使用されている適合的確率的勾配法を使用したスケール不変性に関連がある。

- These methods normalize a gradient update by its estimated standard deviation, thus making the update independent of the scale of the parameter.
    - これらの手法は、その推定された標準偏差によって、勾配の更新を正規化する。
    - それ故に、更新を、パラメーターのスケールとは無関係にする。

- As a result, if some parameters have a larger dynamic range than others, they will take longer to adjust.
    - 結果として、いくつかのパラメーターが他のものよりも大きいダイナミックレンジ [dynamic range] を持つならば、
    - 適合するのに、より時間がかかるだろう。

> ダイナミックレンジ（英: dynamic range）とは、識別可能な信号の最小値と最大値の比率をいう。 信号の情報量を表すアナログ指標のひとつ。

- This is a scenario modern initializers cause, and thus it is possible that a learning rate is both too large and too small at the same time.
    - これは、最新の [modern] 初期化が引き起こすシナリオである。
    - そしてそれ故に、同時に、学習率を大きすぎたり、小さ過ぎたりすることが可能となる。

- Our approach ensures that the dynamic range, and thus the learning speed, is the same for all weights.
    - 我々のアプローチは、ダイナミックレンジを保証する [ensures]。
    - そしてそれ故に、学習スピードは、全ての重みで同じである。

- A similar reasoning was independently used by van Laarhoven (2017).
    - よく似た推論 [reasoning] は、van Laarhoven によって、独自に使用されている。


### 4.2 PIXELWISE FEATURE VECTOR NORMALIZATION IN GENERATOR

- To disallow the scenario where the magnitudes in the generator and discriminator spiral out of control as a result of competition, we normalize the feature vector in each pixel to unit length in the　generator after each convolutional layer.
    - 生成器と識別器の大きさ [magnitudes] が、競争の結果として、手に負えないような状況に陥る [spiral out of control] シナリオが許容しないために、
    - 各畳み込み層の後の生成器において、各ピクセルの中の特徴ベクトルを、ユニット長へ正規化する。

- We do this using a variant of “local response normalization” (Krizhevsky et al., 2012), configured as $b_{x,y}=a_{x,y}/\sqrt{ \frac{1}{N} \sum_{j=0}^{N-1} (a_{x,y}^j )^2+\varepsilon }$
    - 我々は、
    - $b_{x,y}=a_{x,y}/\sqrt{ \frac{1}{N} \sum_{j=0}^{N-1} (a_{x,y}^j )^2+\varepsilon }$
    - として設定される “local response normalization” の変種を使用して、これを行う。

- where $\varepsilon = 10^{-8}$, N is the number of feature maps, and $a_{x,y}$ and $b_{x,y}$ are the original and normalized feature vector in pixel (x,y), respectively.
    - ここで、$\varepsilon = 10^{-8}$
    - N は、特徴マップの数
    - $a_{x,y}$ と $b_{x,y}$ は、ピクセル (x,y) での、直交化され正規化された特徴ベクトルである。

- We find it surprising that this heavy-handed constraint does not seem to harm the generator in any way, and indeed with most datasets it does not change the results much, but it prevents the escalation of signal magnitudes very effectively when needed.
    - 我々は、この強引な [heavy-handed] 制約 [constraint] が、決して [in any way] 生成器に害を与えないようであること、
    - そして、実際には [indeed]、殆どのデータセットで、結果を大きくは変えないが、
    - それは必要なときに、とても効率的に、信号の大きさの過剰な増大を防ぐ。
    - ということを、驚きを持って見つけ出した。

## 5. MULTI-SCALE STATISTICAL SIMILARITY FOR ASSESSING GAN RESULTS

- In order to compare the results of one GAN to another, one needs to investigate a large number of images, which can be tedious, difficult, and subjective.
    - ある GAN と別のもの（＝別のGAN）の結果を比較するために、
    - １つには、退屈で [tedious]、困難で、主観的 [subjective] であり得るような、巨大な枚数の画像を調査する必要がある。

- Thus it is desirable to rely on automated methods that compute some indicative metric from large image collections.
    - そういうわけで、巨大な画像のコレクションから、いくつかの指標 [indicative] 計量 [metric] を計算するような自動化手法に頼ることが望ましい。

- We noticed that existing methods such as MS-SSIM (Odena et al., 2017) find large-scale mode collapses reliably but fail to react to smaller effects such as loss of variation in colors or textures, and they also do not directly assess image quality in terms of similarity to the training set.
    - 我々は、MS-SSIM のような既に存在する手法が、大きなスケールでのモード崩壊を、確実に [reliably] 見つけ出すが、
    - 色やテクスチャーの変動の損失値のような、小さな効果に反応せず、
    - そして、学習データセットの類似性に関して、画像の品質を直接評価 [assess] もしない。
    - ということに気がついた。

---

- We build on the intuition that a successful generator will produce samples whose local image structure is similar to the training set over all scales.
    - 成功する生成器は、局所的な画像の構造が、全てのスケールに渡って学習データによく似ているような、サンプルを生成するだろうというような、直感 [intuition] に基づいている [build on]。

- We propose to study this by considering the multiscale statistical similarity between distributions of local image patches drawn from Laplacian pyramid (Burt & Adelson, 1987) representations of generated and target images, starting at a low-pass resolution of 16 × 16 pixels.
    - 我々は、
    - 16 × 16 ピクセルのローパス解像度で開始して、
    - 生成画像と教師画像のラプラシアンピラミッド表現から描写されるような、局所的な画像パッチの分布との間の、
    - マルチスケールの統計的な類似性を考慮することによって、これを研究することを提案する。

- As per standard practice, the pyramid progressively doubles until the full resolution is reached, each successive level encoding the difference to an up-sampled version of the previous level.
    - 標準的な技法 [standard practice] によると [as per]、ラプラシアンは、フル解像度に到達するまで、進歩的に２倍にする。
    - 連続する [successive] 各レベルは、アップサンプリングされた以前のレベルのバージョンとの差異を符号化する。

---

![image](https://user-images.githubusercontent.com/25688193/59900287-b35cf480-9432-11e9-881e-725356514499.png)

---

- A single Laplacian pyramid level corresponds to a specific spatial frequency band.
    - １つのラプラシアンのレベルは、特定の空間的な周波数バンドに一致する。

- We randomly sample 16384 images and extract 128 descriptors from each level in the Laplacian pyramid, giving us 221 (2.1M) descriptors per level.
    - 16384 枚の画像をランダムにサンプルし、ラプラシアンの各レベルから、128 個の記述子 [descriptors] を展開し、
    - 各レベルに対して、221（2.1M）個の記述子が得られる。

- Each descriptor is a 7 × 7 pixel neighborhood with 3 color channels, denoted by ![image](https://user-images.githubusercontent.com/25688193/59900255-890b3700-9432-11e9-899d-802904d69350.png).
    - 各記述子は、３つのカラーチャンネルで、7 × 7 ピクセルの近傍であり、
    - ![image](https://user-images.githubusercontent.com/25688193/59900255-890b3700-9432-11e9-899d-802904d69350.png) で示される。

- We denote the patches from level l of the training set and generated set as ![image](https://user-images.githubusercontent.com/25688193/59900341-e8694700-9432-11e9-9b78-e008e7040b56.png) and ![image](https://user-images.githubusercontent.com/25688193/59900383-0fc01400-9433-11e9-9079-e7490b00e355.png), respectively.
    - 学習データセットと生成されたセットのレベル l からのパッチを、それぞれ [respectively]、![image](https://user-images.githubusercontent.com/25688193/59900341-e8694700-9432-11e9-9b78-e008e7040b56.png) and ![image](https://user-images.githubusercontent.com/25688193/59900383-0fc01400-9433-11e9-9079-e7490b00e355.png) として示す。

- We first normalize ![image](https://user-images.githubusercontent.com/25688193/59900469-63caf880-9433-11e9-92c8-1b75c9c34c7c.png) and ![image](https://user-images.githubusercontent.com/25688193/59900637-d340e800-9433-11e9-9ea1-90af2f793278.png) w.r.t. the mean and standard deviation of each color channel, and then estimate the statistical similarity by computing their sliced Wasserstein distance SWD ![image](https://user-images.githubusercontent.com/25688193/59900434-3b42fe80-9433-11e9-9189-2f9316411156.png), an efficiently computable randomized approximation to earthmovers distance, using 512 projections (Rabin et al., 2011).
    - 我々は最初に、
    - 各カラーチャンネルの平均と標準偏差に関して [w.r.t. : with reference to]、
    - ![image](https://user-images.githubusercontent.com/25688193/59900469-63caf880-9433-11e9-92c8-1b75c9c34c7c.png) と ![image](https://user-images.githubusercontent.com/25688193/59900637-d340e800-9433-11e9-9ea1-90af2f793278.png) を正規化する。
    - そして次に、
    - 512 個の射影を使用した、効率的な計算可能な earthmovers 距離へのランダム化された近似し、
    - スライスされた Wasserstein 距離 SWD ![image](https://user-images.githubusercontent.com/25688193/59900434-3b42fe80-9433-11e9-9189-2f9316411156.png) を計算することによって、統計的な類似性を推定する。

---

- Intuitively a smallWasserstein distance indicates that the distribution of the patches is similar, meaning that the training images and generator samples appear similar in both appearance and variation at this spatial resolution.
    - 直感的には [Intuitively]、小さな Wasserstein 距離は、パッチの分布が似ていることを示しており、
    - 学習画像と生成されたサンプルが、この空間的な解像度において、外観 [appearance ] と多様性の両方で、よく似て現れることを意味している。

- In particular, the distance between the patch sets extracted from the lowestresolution 16 × 16 images indicate similarity in large-scale image structures, while the finest-level patches encode information about pixel-level attributes such as sharpness of edges and noise.
    - 特に、最も低い解像度 16 × 16 から抽出されたパッチのセットの間の距離は、大きなスケールでの画像の構造において、似ていることを示している。
    - 一方で、最も細かい [finest] レベルのパッチは、辺の輪郭やノイズのような、ピクセルレベルでの情報を符号化する。


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## 6 EXPERIMENTS

- In this section we discuss a set of experiments that we conducted to evaluate the quality of our results.

- Please refer to Appendix A for detailed description of our network structures and training configurations.

- We also invite the reader to consult the accompanying video (https://youtu.be/G06dEcZ-QTg) for additional result images and latent space interpolations.

- In this section we will distinguish between the network structure (e.g., convolutional layers, resizing), training configuration (various normalization layers, minibatch-related operations), and training loss (WGAN-GP, LSGAN).
    - このセクションでは、ネットワーク構造（例えば、畳み込み層、リサイズ）、学習設定（様々な正規化層、ミニバッチ関連の演算）、学習損失（WGAN-GP, LSGAN）を区別する。

### 6.1 IMPORTANCE OF INDIVIDUAL CONTRIBUTIONS IN TERMS OF STATISTICAL SIMILARITY

- We will first use the sliced Wasserstein distance (SWD) and multi-scale structural similarity (MSSSIM) (Odena et al., 2017) to evaluate the importance our individual contributions, and also perceptually validate the metrics themselves.
    - 我々はまず、我々の個々の貢献の重要性を評価するために、sliced Wasserstein distance (SWD) と multi-scale structural similarity (MSSSIM) を使用し、
    - 指標自体を知覚的に検証する。

- We will do this by building on top of a previous state-of-the-art loss function (WGAN-GP) and training configuration (Gulrajani et al., 2017) in an unsupervised setting using CELEBA (Liu et al., 2015) and LSUN BEDROOM (Yu et al., 2015) datasets in 128^2 resolution.
    - 我々は、これを以前の SOTA 損失関数（WGAN-GP）と、
    - CELEBA と 128 * 128 の解像度のデータセットである LSUN BEAROOM を用いた教師なし設定における学習設定とを、
    - 構築することによって行う。

> training configuration 学習設定：様々な正規化層、ミニバッチ関連の演算など

- CELEBA is particularly well suited for such comparison because the training images contain noticeable artifacts (aliasing, compression, blur) that are difficult for the generator to reproduce faithfully.
    - CELEBA は、このような比較に特に適している。
    - なぜならば、学習画像が、生成器にとって忠実な [faithfully] 再現 [reproduce] が困難であるような注目に値する（＝目立つ） [noticeable] 人工物（エイリアシング、圧縮、ぼやけ）を含むためである。

- In this test we amplify the differences between training configurations by choosing a relatively low-capacity network structure (Appendix A.2) and terminating the training once the discriminator has been shown a total of 10M real images.
    - このテストでは、比較的低い容量のネットワーク構造を選択し、
    - いったん識別器が、トータル 10M の本物画像が表示されれば、学習を終了させる [terminating] ことによって 、
    - 学習設定との間の違いを増幅 [amplify] する。

- As such the results are not fully converged.
    - このような結果は、完全に収束しない。

---

![image](https://user-images.githubusercontent.com/25688193/59899201-d71e3b80-942e-11e9-9445-786ce77f0782.png)

- > Table 1: Sliced Wasserstein distance (SWD) between the generated and training images (Section 5) and multi-scale structural similarity (MS-SSIM) among the generated images for several training setups at 128 × 128.
    - > 表１：128 × 128 での、生成された画像と学習画像の間の Sliced Wasserstein distance (SWD) と
    - > いくつかの学習設定に対して生成された画像の間の multi-scale structural similarity (MS-SSIM)

- > For SWD, each column represents one level of the Laplacian pyramid, and the last one gives an average of the four distances.
    - > SWD に対しては、各列はラプラシアンピラミッドのレベルを表し、最後の列は４つの距離の平均を与えている

---

![image](https://user-images.githubusercontent.com/25688193/59902350-f02bea00-9438-11e9-9f6c-b845f84d5e85.png)

- > Figure 3: (a) – (g) CELEBA examples corresponding to rows in Table 1. 

- > These are intentionally non-converged.

- > (h) Our converged result.

- > Notice that some images show aliasing and some are not sharp – this is a flaw of the dataset, which the model learns to replicate faithfully.
    - > いくつかの画像は、エイリアシングを示し、いくつかの画像はシャープでないことに注意。即ち、これはデータセットの欠陥 [flaw] であり、モデルは忠実に [faithfully] 再現することを学習する。

---

- Table 1 lists the numerical values for SWD and MS-SSIM in several training configurations, where our individual contributions are cumulatively enabled one by one on top of the baseline (Gulrajani et al., 2017).
    - 表１は、いくつかの学習設定における SWD と MS-SSIM に対しての数値をリストしている。
    - ここで、我々の個々の貢献が、ベースラインの上に１つづつ累積的に [cumulatively] 有効になっている。

> baseline (Gulrajani et al., 2017): WGAN-GP

- The MS-SSIM numbers were averaged from 10000 pairs of generated images, and SWD was calculated as described in Section 5.
    - MS-SSIM 数は、生成画像の 10000 個のペアから平均化され、SWD はセクション５において記述されたように計算される。

- Generated CELEBA images from these configurations are shown in Figure 3.
    - これらの設定から生成された CELEBA 画像は、図３に示される。

- Due to space constraints, the figure shows only a small number of examples for each row of the table, but a significantly broader set is available in Appendix H.
    - スペースの制約のため、図は、表の各行に対しての小さな数の例のみを示すが、かなり [significantly] 広い [broader] セットが、補足 H で利用可能である。

- Intuitively, a good evaluation metric should reward plausible images that exhibit plenty of variation in colors, textures, and viewpoints.
    - 直感的には [Intuitively]、良い評価指標は、色やテクスチャー、視点の沢山の変動を展開するような、もっともらしい [plausible] 画像に報いるべきである。

- However, this is not captured by MS-SSIM: we can immediately see that configuration (h) generates significantly better images than configuration (a), but MS-SSIM remains approximately unchanged because it measures only the variation between outputs, not similarity to the training set.
    - しかしながら、これは MS-SSIMによって抽出されない。
    - 即ち、設定 (h) は設定 (a) よりかなり良い画像を生成しているが、MS-SSIM は、学習データセットの類似性ではなく、出力の間の変動のみを計測するため、おおよそ変化していないとうことを即座に見ることが出来る。

- SWD, on the other hand, does indicate a clear improvement.
    - 一方、SWD は、明らかな改善を示している。

---

- The first training configuration (a) corresponds to Gulrajani et al. (2017), featuring batch normalization in the generator, layer normalization in the discriminator, and minibatch size of 64.
    - 最初の学習設定 (a) は、Gulrajaniらのものと一致する。
    - （これは、）生成器における featuring batch normalization、識別器における layer normalization、ミニバッチサイズ 64（をもつ）

- (b) enables progressive growing of the networks, which results in sharper and more believable output images.
    - (b) は、ネットワークの progressive growing を有効にしており、しょりシャープで信頼性のある出力画像という結果になっている。

- SWD correctly finds the distribution of generated images to be more similar to the training set.
    - SWD は、生成画像の分布が、学習データセットのものとより似ているようになることを正確に見つけ出す。

---

- Our primary goal is to enable high output resolutions, and this requires reducing the size of minibatches in order to stay within the available memory budget.
    - 我々の主要なゴールは、高解像度の出力を可能にすることであり、これは利用可能なメモリの予算の中に留まるために、ミニバッチのサイズを減らすことを要求する。

> 高解像度での学習フェイズにて、ミニバッチサイズが大きいと大量のメモリを消費し、メモリアロケーションエラーとなるので、高解像度での学習フェイズではミニバッチサイズを減らす必要がある。

- We illustrate the ensuing challenges in (c) where we decrease the minibatch size from 64 to 16.
    - 我々は (c) においてその後の [ensuing] 変化を図示している。ここで、ミニバッチサイズを 64 から 16 に減らしている。

- The generated images are unnatural, which is clearly visible in both metrics.
    - 生成画像は、不自然になり、これは両方の指標（＝SWD, MS-SSIM）で明らかに見られる。

- In (d), we stabilize the training process by adjusting the hyperpa- rameters as well as by removing batch normalization and layer normalization (Appendix A.2).
    - (d) では、ハイパーパラメーターを適合することによって、同様にして、batch norm を除外し layer norm を加えることによって、学習プロセスを安定化している。

- As an intermediate test (e∗), we enable minibatch discrimination (Salimans et al., 2016), which somewhat surprisingly fails to improve any of the metrics, including MS-SSIM that measures output variation.
    - 中間のテスト (e*) では、我々は、minibatch discrimination を有効にしており、
    - これは、いくらかの驚くことに、出力変動を計算する MS-SSIM を含む全ての指標を改善しない。

- In contrast, our minibatch standard deviation (e) improves the average SWD scores and images.
    - 反対に、我々の minibatch standard deviation (e) は、平均の SWD スコアと画像を改善する。

- We then enable our remaining contributions in (f) and (g), leading to an overall improvement in SWD and subjective visual quality.
    - 我々は次に、(f), (g) において、全体に渡っての SWD と主観的な視覚品質での改善を導く、残りの貢献を可能にする。

- Finally, in (h) we use a non-crippled network and longer training – we feel the quality of the generated images is at least comparable to the best published results so far.
    - 最後に、(h) では、我々はクリッピングされていいないネットワークとより長い時間での学習を使用する。
    - 即ち、我々は、生成画像の品質が、以前に公開されたベストの結果と少なくとも比較できると感じている。

### 6.2 CONVERGENCE AND TRAINING SPEED

![image](https://user-images.githubusercontent.com/25688193/59905212-7ac41780-9440-11e9-85f1-df9095eaa5be.png)

- > Figure 4: Effect of progressive growing on training speed and convergence.
    - > 図４：学習スピートと収束性での progressive growing の効果

- > The timings were measured on a single-GPU setup using NVIDIA Tesla P100.
    - > タイミングは、NVIDIA Tesla P100 を使用して計測された。

- > (a) Statistical similarity with respect to wall clock time for Gulrajani et al. (2017) using CELEBA at 128 × 128 resolution.
    - > (a) 128 × 128 の解像度での CELEBA を使用した、Gulrajani らの壁時計に関しての統計的な類似度

- > Each graph represents sliced Wasserstein distance on one level of the Laplacian pyramid, and the vertical line indicates the point where we stop the training in Table 1.
    - > 各グラフは、ラプラシアンピラミッドのレベルにおける、SWD を表現しており、
    - > 垂線 [vertical line] は表１において学習を停止する場所を示している

- > (b) Same graph with progressive growing enabled.
    - > (b) progressive growing が有効である同じグラフ

- > The dashed vertical lines indicate points where we double the resolution of G and D.
    - > 点垂線は、G と D の解像度が２倍である場所を示している。

- > (c) Effect of progressive growing on the raw training speed in 1024 × 1024 resolution.
    - > (c) 1024 × 1024 の解像度での、生の [on the row] 学習スピードにおける progressive growing の影響

---

- Figure 4 illustrates the effect of progressive growing in terms of the SWD metric and raw image throughput.
    - 図４は、SWD 指標と生の画像のスループット（＝単位時間あたりの処理能力） [throughput] の観点で、progressive growing の効果を図示している。

- The first two plots correspond to the training configuration of Gulrajani et al. (2017) without and with progressive growing.
    - 最初の２つのプロットは、progressive growing 有り無しでの Gulrajani らの学習設定に一致する。

- We observe that the progressive variant offers two main benefits: it converges to a considerably better optimum and also reduces the total training time by about a factor of two.
    - 我々は、progressive の変種は２つの利点をもたらすと観測する。
    - 即ち、それはかなりよい最適点に収束し、約 1/2 に全体の学習時間をへらす。[by a factor of ~ : ~ 倍で]

- The improved convergence is explained by an implicit form of curriculum learning that is imposed by the gradually increasing network capacity. 
    - <font color="Pink">改善された収束性は、徐々に増加するネットワーク容量によって、課せられる [imposed] ような、カリキュラム [curriculum] 学習の暗黙の [implicit] 形式によって、説明される。</font>

- Without progressive growing, all layers of the generator and discriminator are tasked with simultaneously finding succinct intermediate representations for both the large-scale variation and the small-scale detail.
    - progressive growing なしでは、生成器と識別器の全ての層は、大きなスケールでの変動と小さなスケールでの詳細の両方に対して、簡潔な [succinct] な中間表現を同時に見つけることを任されている [tasked]。

- With progressive growing, however, the existing low-resolution layers are likely to have already converged early on, so the networks are only tasked with refining the representations by increasingly smaller-scale effects as new layers are introduced.
    - progressive growing 有りでは、しかしながら、存在する低解像度の層は、既に初期段階で収束しそうである。
    - なので、ネットワークは、新しい層が導入されるにつれて、ますます [increasingly] より小さいスケールでの効果によって、表現を洗練する [refining] ように任されているのみである。

- Indeed, we see in Figure 4(b) that the largest-scale statistical similarity curve (16) reaches its optimal value very quickly and remains consistent throughout the rest of the training.
    - 実際に [Indeed]、図 4 (b) において、最も大きいスケールでの統計的類似性の曲線 (16) は、とても素早くその最適点に到達し、学習の残りを通じて一定のままである。

- The smaller-scale curves (32, 64, 128) level off one by one as the resolution is increased, but the convergence of each curve is equally consistent.
    - より小さいスケールでの曲線 (32, 64, 128) は、解像度が増加するにつれて、１つづつ平らになる [level off] が、各曲線の収束性は、等しく一定である。

- With non-progressive training in Figure 4(a), each scale of the SWD metric converges roughly in unison, as could be expected.
    - 図 4 (a) における非 progressive 学習では、SWD 指標の各スケールは、予想できるように、一致して [in unison] 大まかに収束する。

---

- The speedup from progressive growing increases as the output resolution grows.
    - 出力解像度の成長するにつれて、progressive growing からのスピードは増加する。

- Figure 4(c) shows training progress, measured in number of real images shown to the discriminator, as a function of training time when the training progresses all the way to 1024^2 resolution.
    - 図 4 (c) は、
    - 学習段階が、1024 * 1024 までずっと [all the way] 進むときの、学習時間の関数として識別器へ見せる本物画像の数を計測されるような、学習段階を示している。

- We see that progressive growing gains a significant head start because the networks are shallow and quick to evaluate at the beginning.
    - progressive gan が重要な優先スタート [head start] を得ることが見て取れる。
    - なぜならば、ネットワークが浅く、最初は [at the beginning] すばやく評価するので、

- Once the full resolution is reached, the image throughput is equal between the two methods.
    - いったんフルの解像度に到達すれば、画像の処理能力は２つの手法で等しい。

- The plot shows that the progressive variant reaches approximately 6.4 million images in 96 hours, whereas it can be extrapolated that the non-progressive variant would take about 520 hours to reach the same point.
    - プロットは、progressive 変種が、96 時間でおよそ 6.4 万画像に到達することを示している。
    - 一方で [whereas]、非 progressive 変種が、同じポイントに到達するのに、約 520 時間かかっていることを推定 [extrapolated] する。

- In this case, the progressive growing offers roughly a 5.4× speedup.
    - このケースでは、progressive growing は大まかに 5.4 倍のスピードアップを提供している。

###  6.3 HIGH-RESOLUTION IMAGE GENERATION USING CELEBA-HQ DATASET

- To meaningfully demonstrate our results at high output resolutions, we need a sufficiently varied high-quality dataset.
    - 高い解像度での我々の結果を、意味をもつように実証するために、我々は様々な高品質のデータセットを必要とする。

- However, virtually all publicly available datasets previously used in GAN literature are limited to relatively low resolutions ranging from 32^2 to 480^2.
    - しかしながら、GANの文脈で以前に使用されていた実質的に [virtually] 全ての公開されている利用可能なデータセットは、32 * 32 から 480 * 480 の範囲でという、比較的低い解像度で制限されている。

- To this end, we created a high-quality version of the CELEBA dataset consisting of 30000 of the images at 1024 × 1024 resolution.
    - この目的で [to this end]、我々は、1024 * 1024 の解像度での 30000 枚の画像を含む CELEBA データセットの高解像度バージョンを作成した。

- We refer to Appendix C for further details about the generation of this dataset.
    - このデータセットの生成についてのさらなる詳細は、補足 C を参照

---

- Our contributions allow us to deal with high output resolutions in a robust and efficient fashion.
    - 我々の貢献は、ロバストで効果的な流儀 [fashion] において、高解像度の出力に対処する [deal with] ことである。

- Figure 5 shows selected 1024 × 1024 images produced by our network. 
    - 図５は、我々のネットワークによって生成された選択された 1024 * 1024 画像を示している。

- While megapixel GAN results have been shown before in another dataset (Marchesi, 2017), our results are vastly more varied and of higher perceptual quality.
    - メガピクセルの GAN の結果が、以前に、他のデータセットで見られた一方で、
    - 我々の結果は、非常に [vastly] より多様 [varied] で、知覚的な品質がより高い。

- Please refer to Appendix F for a larger set of result images as well as the nearest neighbors found from the training data.
    - 結果の画像のより大きなセットと同様にして、学習データセットから見つかった 最近傍画像も、
    - 補足 F を参照してください。

- The accompanying video shows latent space interpolations and visualizes the progressive training.
    - 添付のビデオは、潜在空間の補間と段階的な学習を視覚化している。

- The interpolation works so that we first randomize a latent code for each frame (512 components sampled individually from N (0, 1)), then blur the latents across time with a Gaussian (σ = 45 frames @ 60Hz), and finally normalize each vector to lie on a hypersphere.
    - この補間は、最初に各フレーム（ N(0,1) からサンプリングされた 512 個のコンポーネント）に対して潜在変数をランダム化し、次にガウジアンでの時間に渡って潜在変数をぼかし、最後に超球面に横たえるように（＝超球面上で）各ベクトルを正規化する。

---

- We trained the network on 8 Tesla V100 GPUs for 4 days, after which we no longer observed qualitative differences between the results of consecutive training iterations.
    - 我々はネットワークを、8 個の Tesla V100 GPU で 4 日間で学習した。
    - その後、連続的な [consecutive] 学習イテレーションの結果との間の定性的な違いがもはや [no longer] 観測されなくなった。

- Our implementation used an adaptive minibatch size depending on the current output resolution so that the available memory budget was optimally utilized.
    - 我々の実装では、利用可能なメモリの予算が最適に使用されるように、現在の出力解像度度に依存した適合的なミニバッチサイズを使用した

---

- In order to demonstrate that our contributions are largely orthogonal to the choice of a loss function, we also trained the same network using LSGAN loss instead of WGAN-GP loss.
    - 我々の貢献が、損失関数の選択に大きく直交していることを実証するために、我々はまた、WGAN-GP の損失関数の代わりに LSGAN の損失関数を使用した同じネットワークを学習した。

- Figure 1 shows six examples of 1024^2 images produced using our method using LSGAN. Further details of this setup are given in Appendix B.
    - 図１は、LSGAN を用いた我々の手法使用して生成された 1024 * 1024 の６つの画像の例を示している。

### 6.4 LSUN RESULTS

![image](https://user-images.githubusercontent.com/25688193/59957901-2456e780-94d9-11e9-95ad-407d3d6f4d44.png)

![image](https://user-images.githubusercontent.com/25688193/59957933-916a7d00-94d9-11e9-93c7-d73919baaa0f.png)

- Figure 6 shows a purely visual comparison between our solution and earlier results in LSUN BEDROOM. 
    - 図６は、LSUN BEDROOM における我々の解決法と初期の結果との間の純粋な視覚的な比較を示している。

- Figure 7 gives selected examples from seven very different LSUN categories at 256^2.
    - 図７は、256 * 256 での７つの異なる LSUN カテゴリから選択された例を与えている。

- A larger, non-curated set of results from all 30 LSUN categories is available in Appendix G, and the video demonstrates interpolations.
    - より大きく、30 個の LSUN カテゴリ全てからの結果のキューレート（＝収集）されていない [non-curated] セットは補足 G で利用可能である。

- We are not aware of earlier results in most of these categories, and while some categories work better than others, we feel that the overall quality is high.
    - これらのカテゴリの大部分において、初期の結果に気づかず、
    - いくつかのカテゴリがよりよく動作している一方で、全体の品質が高いと感じる。

### 6.5 CIFAR10 INCEPTION SCORES

![image](https://user-images.githubusercontent.com/25688193/59958143-cb894e00-94dc-11e9-9109-d6b7f41deb85.png)

![image](https://user-images.githubusercontent.com/25688193/59958134-a4328100-94dc-11e9-849e-80d15289745c.png)

---

- The best inception scores for CIFAR10 (10 categories of 32 × 32 RGB images) we are aware of are 7.90 for unsupervised and 8.87 for label conditioned setups (Grinblat et al., 2017).
    - 我々が気づいている CIFAR-10 に対しての最良の inseption score (IS) は、教師なし設定に対して 7.90 であり、ラベルで条件づけされた設定に対して 8.87 である。

- The large difference between the two numbers is primarily caused by “ghosts” that necessarily appear between classes in the unsupervised setting, while label conditioning can remove many such transitions.
    - ２つの数字の間の大きな違いは、教師なし設定のクラスとの間に必ず [necessarily] 現れる "ghosts" によって主に引き起こされる。
    - 一方で、ラベル条件づけは、このような多くの変換を除外することが出来る。

---

- When all of our contributions are enabled, we get 8.80 in the unsupervised setting.
    - 我々の貢献が全て有効であるとき、教師なし設定において 8.80 の inception score を手に入れる。

- Appendix D shows a representative set of generated images along with a more comprehensive list of results from earlier methods.
    - 補足 D では、以前の手法からの結果のより包括的な [comprehensive] リストとともに、生成画像の代表的な [representative] セットを示している。

- The network and training setup were the same as for CELEBA, progres- sion limited to 32 × 32 of course.
    - ネットワークと学習設定は、CELEBA と同じであり、段階学習は、32 * 32 にもちろん [of course] 制限されている。

- The only customization was to the WGAN-GP’s regularization term ![image](https://user-images.githubusercontent.com/25688193/59957987-8e23c100-94da-11e9-8923-d32a1749cdbf.png).
    - 唯一のカスタムは、WGAN-GP の正則化項 ![image](https://user-images.githubusercontent.com/25688193/59957987-8e23c100-94da-11e9-8923-d32a1749cdbf.png) のみである。

- Gulrajani et al. (2017) used γ = 1.0, which corresponds to 1-Lipschitz, but we noticed that it is in fact significantly better to prefer fast transitions (γ = 750) to minimize the ghosts.
    - Gulrajani らは γ = 1.0 を使用し、これは 1-リプシッツ連続に一致する。
    - しかし、我々は、ghosts を最小化するためには、早い変換 (γ = 750) のほうが遥かに望ましいという事実に気がついた。

- We have not tried this trick with other datasets.
    - 我々は、このトリックを他のデータセットに対しては試していない。


# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


