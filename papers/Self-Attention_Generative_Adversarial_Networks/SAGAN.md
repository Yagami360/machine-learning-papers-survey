# ■ 論文
- 論文タイトル："Self-Attention Generative Adversarial Networks"
- 論文リンク：https://arxiv.org/abs/1805.08318
- 論文投稿日付：2018/03/21
- 著者（組織）：Han Zhang, Ian Goodfellow, Dimitris Metaxas, Augustus Odena
- categories：

# ■ 概要（何をしたか？）

## Abstract

- In this paper, we propose the Self-Attention Generative Adversarial Network (SAGAN) which allows attention-driven, long-range dependency modeling for image generation tasks.
    - 本論文では、画像生成タスクにおいて、注意喚起型 [attention-driven] の長距離依存性モデリングを可能にする自己注意生成型敵対的ネットワーク（SAGAN）を提案する。

- Traditional convolutional GANs generate high-resolution details as a function of only spatially local points in lower-resolution feature maps. In SAGAN, details can be generated using cues from all feature locations.
    - 伝統的な畳み込みGANは、低解像度の特徴マップ内の空間的に局所的な点のみの関数として高解像度の詳細を生成します。 SAGANでは、すべての特徴位置からの手がかり [cues] を使用して詳細を生成できます。

- Moreover, the discriminator can check that highly detailed features in distant portions of the image are consistent with each other.
    - **さらに、識別器は、画像の離れた部分 [portions] において、非常に詳細な特徴が互いに一致していることを確認することができる。**

- Furthermore, recent work has shown that generator conditioning affects GAN performance.
    - さらに、最近の研究は、生成器の条件付がGANの性能に影響を与えることを示しています。

- Leveraging this insight, we apply spectral normalization to the GAN generator and find that this improves training dynamics.
    - この洞察を利用して、我々はスペクトル正規化をGAN生成器に適用し、そしてこれが学習ダイナミクスを改善することを発見する。

- The proposed SAGAN performs better than prior work, boosting the best published Inception score from 36.8 to 52.52 and reducing Fre ́chet Inception distance from 27.62 to 18.65 on the challenging ImageNet dataset.
    - 提案されているSAGANは、以前の研究よりも優れたパフォーマンスを発揮し、挑戦的なImageNetデータセットで、公表されている最良のInceptionスコアを36.8から52.52に向上させ、Frechet Inceptionの距離を27.62から18.65に短縮します。

- Visualization of the attention layers shows that the generator leverages neighborhoods that correspond to object shapes rather than local regions of fixed shape.
    - **注意層の視覚化は、生成器が固定形状の局所領域ではなく、オブジェクト形状に対応する近傍を活用する [leverages] ことを示しています。**


# ■ イントロダクション（何をしたいか？）

## x. Introduction

- Image synthesis is an important problem in computer vision. There has been remarkable progress in this direction with the emergence of Generative Adversarial Networks (GANs) (Goodfellow et al., 2014), though many open problems remain (Odena, 2019). GANs based on deep convolutional networks (Radford et al., 2016; Karras et al., 2018; Zhang et al.) have been especially successful. 
    - 画像合成は、コンピュータビジョンにおける重要な問題です。 多くの未解決の問題が残っているが（Odena、2019）、生成的敵対的ネットワーク（GAN）の出現により、この方向への著しい進歩があった（Goodfellow et al。、2014）。 ディープコンボリューションネットワークに基づくGAN（Radfordら、2016年; Karrasら、2018年; Zhangら）は特に成功している。

- However, by carefully examining the generated samples from these　models, we can observe that convolutional GANs (Odena et al., 2017; Miyato et al., 2018; Miyato & Koyama, 2018) have much more difficulty in modeling some image classes than others when trained on multi-class datasets (e.g., ImageNet (Russakovsky et al., 2015)). 
    - しかしながら、これらのモデルから生成されたサンプルを注意深く調べることによって、畳み込みGAN（Odena et al。、2017; Miyato et al。、2018; Miyato＆Koyama、2018）は、マルチクラスデータセットについて学習されたとき（例：ImageNet（Russakovsky et al。、2015））に、他のものよりはるかに複数の画像クラスをモデリングすることが困難であることが観測出来る。

- For example, while the state-of-the-art ImageNet GAN model (Miyato & Koyama, 2018) excels at synthesizing image classes with few structural constraints (e.g., ocean, sky and landscape classes, which are distinguished more by texture than by geometry), it fails to capture geometric or structural patterns that occur consistently in some classes (for example, dogs are often drawn with realistic fur texture but without clearly defined separate feet).
    - 例えば、最先端のImageNet GANモデル（Miyato＆Koyama、2018）は、構造的制約がほとんどない画像クラス（例えば、幾何学よりもテクスチャーによってより多く識別出来るような、海洋、空、景観クラスなど）の合成に優れて [excels] いる一方で、 
    - いくつかのクラスで一貫して発生する幾何学的または構造的パターンを捉えることができない（例えば、犬はしばしば現実的な毛皮の質感で描かれているが、明確に定義された別々の足はない）。

- One possible explanation for this is that previous models rely heavily on convolution to model the dependencies across different image regions. Since the convolution operator has a local receptive field, long range dependencies can only be processed after passing through several convolutional layers.
    - これに対する１つの可能な説明は、以前のモデルは異なる画像領域にわたる依存性をモデル化するために畳み込みに大きく依存しているということである。 
    - 畳み込み演算子には局所的な受容野があるので、長距離依存性はいくつかの畳み込み層を通過した後にのみ処理することができます。

- This could prevent learning about long-term dependencies for a variety of reasons: a small model may not be able to represent them, optimization algorithms may have trouble discovering parameter values that carefully coordinate multiple layers to capture these dependencies, and these parameterizations may be statistically brittle and prone to failure when applied to previously unseen inputs.
    - これは、さまざまな理由で長期の依存関係について学ぶのを妨げる可能性があります
    - 小さなモデルではそれらを表現できない、
    - 最適化アルゴリズムでは、これらの依存関係を捉えるために、複数のレイヤーを慎重に調整して [coordinate] パラメーター値を見つけるのは困難です。
    - そして、これらのパラメーター化は、これまで目に見えなかった入力に適用するとき、統計的に [statistically]、脆くなり [brittle]、失敗する傾向があり [prone] ます。

- Increasing the size of the convolution kernels can increase the representational capacity of the network but doing so also loses the computational and statistical efficiency obtained by using local convolutional structure.

- Self-attention (Cheng et al., 2016; Parikh et al., 2016; Vaswani et al., 2017), on the other hand, exhibits a better balance between the ability to model long-range dependencies and the computational and statistical efficiency.

- The self-attention module calculates response at a position as a weighted sum of the features at all positions, where the weights – or attention vectors – are calculated with only a small computational cost.

---

- In this work, we propose Self-Attention Generative Adver- sarial Networks (SAGANs), which introduce a self-attention mechanism into convolutional GANs. The self-attention module is complementary to convolutions and helps with modeling long range, multi-level dependencies across image regions. Armed with self-attention, the generator can draw images in which fine details at every location are carefully coordinated with fine details in distant portions of the image. Moreover, the discriminator can also more accurately enforce complicated geometric constraints on the global image structure.

---

- In addition to self-attention, we also incorporate recent insights relating network conditioning to GAN performance. The work by (Odena et al., 2018) showed that well-conditioned generators tend to perform better. We propose enforcing good conditioning of GAN generators using the spectral normalization technique that has previously been applied only to the discriminator (Miyato et al., 2018).

---

- We have conducted extensive experiments on the ImageNet dataset to validate the effectiveness of the proposed self- attention mechanism and stabilization techniques. SAGAN significantly outperforms prior work in image synthe- sis by boosting the best reported Inception score from 36.8 to 52.52 and reducing Fre ́chet Inception distance from 27.62 to 18.65. Visualization of the attention layers shows that the generator leverages neighborhoods that correspond to object shapes rather than local regions of fixed shape. Our code is available at https://github.com/ brain-research/self-attention-gan.


# ■ 結論

## x. 論文の項目名 (Conclusion)


# ■ 何をしたか？詳細

## x. 論文の項目名


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


