# ■ 論文
- 論文タイトル："A Style-Based Generator Architecture for Generative Adversarial Networks"
- 論文リンク：
- 論文投稿日付：2018/11/12(v1), 2019/03/29(v2)
- 著者：Tero Karras(NVIDIA), Samuli Laine(NVIDIA), Timo Aila(NVIDIA)
- categories：

# ■ 概要（何をしたか？）

## Abstract

- We propose an alternative generator architecture for generative adversarial networks, borrowing from style transfer literature.
    - 我々は、style transfer の文献から拝借して、GAN に対しての代わりとなるアーキテクチャを提供する。

- The new architecture leads to an automatically learned, unsupervised separation of high-level attributes (e.g., pose and identity when trained on human faces) and stochastic variation in the generated images (e.g., freckles, hair), and it enables intuitive, scale-specific control of the synthesis.
    - 新しいアーキテクチャは、自動的に学習され、
    - <font color="Pink">高レベルの特徴（例えば、人間の顔を学習するときの顔の向きと特徴）の教師無しでの分割と、
    - そして、生成された画像における、確率的な多様性 [variation]（例えば、そばかす[freckles]、髪）
    - そして、これは、直感的な [intuitive] で scale-specific な合成 [synthesis] のコントロールを可能にする。</font>


- The new generator improves the state-of-the-art in terms of traditional distribution quality metrics, leads to demonstrably better interpolation properties, and also better disentangles the latent factors of variation.
    - 新しい生成器は、伝統的な分布のクオリティ指標に関して [in terms of]、SOTA を改善する。
    - 明らかに [demonstrably] より良い、補間 [interpolation] 性質 [demonstrably] に導き、
    - そしてまた、多様性の潜在的な [demonstrably] 要因を、より良く解きほぐす [disentangles]。

- To quantify interpolation quality and disentanglement, we propose two new, automated methods that are applicable to any generator architecture.
    - 補間クオリティと（絡み合った謎の）解きほぐし [disentanglement] を定量化 [quantify] するために、
    - 我々は、どんな生成器のアーキテクチャにも適用できる２つの新しい自動化手法を提案する。

- Finally, we introduce a new, highly varied and high-quality dataset of human faces.
    - 最後に、我々は、高い多様性と品質の、人間の顔の新しいデータセットを紹介する。

> ・特徴が絡み合っている<br>
> →この軸を調整すると、目が大きくなって髪が長くなって輪郭が丸くなるみたいになると各軸が複雑に影響し合って生成しづらい<br>

> ・特徴がしっかり分解されている<br>
> →この軸は目の大きさ、この軸は髪の長さ、この軸は輪郭みたいに特徴が独立していると生成がしやすい<br>

> StyleGAN では、この絡み合った [entanglement] 特徴を解きほぐし、[disentanglement]、特徴を分割する。[separation] <br>


# ■ イントロダクション（何をしたいか？）

## 1. Introduction

- The resolution and quality of images produced by generative methods—especially generative adversarial networks (GAN) [22]—have seen rapid improvement recently [30, 45, 5].
    - 生成モデル、とりわけ GAN によって生成された画像の解像度と品質は、最近急速に改善している。

- Yet the generators continue to operate as black boxes, and despite recent efforts [3], the understanding of various aspects of the image synthesis process, e.g., the origin of stochastic features, is still lacking.
    - まだ、生成器がブラックボックスとして、動作し続け、
    - そして最近の努力にも関わらず、
    - 画像合成 [synthesis] 処理の様々な側面の理解（例えば、確率的な特徴の起源）は、まだ欠けている。

- The properties of the latent space are also poorly understood, and the commonly demonstrated latent space interpolations [13, 52, 37] provide no quantitative way to compare different generators against each other.
    - 潜在空間の性質もまた、よく理解されていない。
    - そして、一般に [commonly] 実証されている潜在空間の補間は、互いに異なる生成器を比較するための定量的な [quantitative] 方法を提供していない。

---

- Motivated by style transfer literature [27], we re-design the generator architecture in a way that exposes novel ways to control the image synthesis process.
    - style transfer の文献 [27] によって動機づけされ、
    - 我々は、画像合成処理をコントロールするための目新しい [novel] 方法を
    公開するような方法で、生成器のアーキテクチャを再設計する。

- Our generator starts from a learned constant input and adjusts the “style” of the image at each convolution layer based on the latent code, therefore directly controlling the strength of image features at different scales.
    - 我々の生成器は、学習された一定の入力からスタートする。
    - そして、潜在コード？において、各畳み込み層ベースでの、画像の ”スタイル” を調整する。
    - それ故、異なるスケールで、画像の特徴の強度を直接的にコントロールする。

- Combined with noise injected directly into the network, this architectural change leads to automatic, unsupervised separation of high-level attributes (e.g., pose, identity) from stochastic variation (e.g., freckles,hair) in the generated images, and enables intuitive scale-specific mixing and interpolation operations.
    - ネットワークの中に、直接的に注入されたノイズを組み合わせることで、
    - <font color="Pink">このアーキテクチャの変更は、生成された画像において、確率的な多様性（例えば、そばかすや髪）から、高レベルの特徴（例えば、顔の向きや特徴）の、自動的で、教師なしの分割？に導く。</font>

- We do not modify the discriminator or the loss function in any way, and our work is thus orthogonal to the ongoing discussion about GAN loss functions, regularization, and hyperparameters [24, 45, 5, 40, 44, 36].
    - 我々は、あらゆる方法で、識別器や損失関数を修正しない。
    - そして、我々の研究は、そういうわけで、GAN の損失関数、正則化、ハイパーパラメータについて、進行中の議論と直交している。（＝別方向である）

---

- Our generator embeds the input latent code into an intermediate latent space, which has a profound effect on how the factors of variation are represented in the network.
    - **我々の生成器は、入力潜在コードを、中間の [intermediate] 潜在空間へ埋め込む。**
    - **（この潜在空間というのは、）多様性の要因が、どのようにして、ネットワークの中で表されるのかということに、深い [profound] 影響を与えるような（潜在空間）**

- The input latent space must follow the probability density of the training data, and we argue that this leads to some degree of unavoidable entanglement.
    - 入力潜在空間は、学習データの確率分布に従わなければならない。
    - そして、我々は、このことが、ある程度 [some degree of] 避けられられない絡み合い [entanglement] に至る [leads to] ということを主張する。

- Our intermediate latent space is free from that restriction and is therefore allowed to be disentangled.
    - 我々の中間の潜在空間は、このような制限から自由であり、
    - それ故に、絡み合った謎を解きほぐす [disentangled] ことを許容する。

- As previous methods for estimating the degree of latent space disentanglement are not directly applicable in our case, we propose two new automated metrics—perceptual path length and linear separability—for quantifying these aspects of the generator.
    - 潜在空間の解きほぐしの程度を推定するための以前の方法は、我々のケースに、直接的に応用出来ないので、
    - **我々は、生成器のこれらの側面を定量化するたの、２つの新しい自動的な指標を提案する。**
    - **（この新しい２つの指標というのは、perceptual path length と linear separability ）**

- Using these metrics, we show that compared to a traditional generator architecture, our generator admits a more linear, less entangled representation of different factors of variation.
    - これらの指標を使用することで、
    - 我々は、伝統的な生成器のアーキテクチャとの比較して、
    - 我々の生成器は、多様性の異なる要因の、より線形で、よりもつれ [entangled] の少ない表現になっていることを見せる。

> ・特徴が絡み合っている<br>
> →この軸を調整すると、目が大きくなって髪が長くなって輪郭が丸くなるみたいになると各軸が複雑に影響し合って生成しづらい<br>

> ・特徴がしっかり分解されている<br>
> →この軸は目の大きさ、この軸は髪の長さ、この軸は輪郭みたいに特徴が独立していると生成がしやすい<br>

> StyleGAN では、この絡み合った [entanglement] 特徴を解きほぐし、[disentanglement]、特徴を分割する。[separation] <br>

---

- Finally, we present a new dataset of human faces (Flickr-Faces-HQ, FFHQ) that offers much higher quality and covers considerably wider variation than existing high-resolution datasets (Appendix A).
    - 最後に、我々は、人間の顔の新しいデータセット（Flickr-Faces-HQ, FFHQ）を提供する。
    - （このデータセットというのは、）今存在している高解像度のデータセットより、より高い品質で、考えられる広い多様性をカバーするような（データセット）

- We have made this dataset publicly available, along with our source code and pre-trained networks.(1) 
    - 我々は、このデータセットを公共に利用可能にした。
    - 我々のソースコードと、事前学習されたネットワーク (1) と共に、

- The accompanying video can be found under the same link.
    
(1) : https://github.com/NVlabs/stylegan


# ■ 結論

## 5. Conclusion

- Based on both our results and parallel work by Chen et al.[6], it is becoming clear that the traditional GAN generator architecture is in every way inferior to a style-based design.
    - **我々の結果と、Chen による [6] との並行した研究の両方に基づいて、**
    - **伝統的な GAN のアーキテクチャが、全ての方法において、style-based の設計より劣っていることが明らかになっている。**

- This is true in terms of established quality metrics, and we further believe that our investigations to the separation of high-level attributes and stochastic effects, as well as the linearity of the intermediate latent space will prove fruitful in improving the understanding and controllability of GAN synthesis.
    - このことは、定量的な指標に関して、真実であり、
    - そして、我々は更に信じている。
    - 高レベルの特徴と確率的な効果の分離への、我々の調査。
    - 中間の潜在空間の線形性が、GAN での画像合成の理解と操作性を改善することにおいて、実りのある証明であるということを。

- We note that our average path length metric could easily be used as a regularizer during training, and perhaps some variant of the linear separability metric could act as one, too.
    - 我々は、平均 path length 指標が、学習中の正則化として、簡単に使用lできることと、
    - そして、もしかすると、線形分離指標のいくつかの変形が、それと同じように動作するということを記す。

- In general, we expect that methods for directly shaping the intermediate latent space during training will provide interesting avenues for future work.
    - 一般的に、我々は、学習中に中間潜在空間を直接的に形づくる [shaping] ための方法が、将来の研究のために、興味深い手段 [avenues] を提供することを期待する。


# ■ 何をしたか？詳細

## 2. Style-based generator

- Traditionally the latent code is provided to the generator through an input layer, i.e., the first layer of a feedforward network (Figure 1a). 
    - 伝統的に、潜在コードは、生成器の１つの入力層を通って、提供される。
    - 例えば、順伝搬ネットワークの最初の層。（図1a）

- We depart from this design by omitting the input layer altogether and starting from a learned constant instead (Figure 1b, right). 
    - 我々は、入力層を完全に省略 [omitting] し 、代わりに学習済み定数から開始することによって 、このデザインから出発する。

- Given a latent code z in the input latent space Z, a non-linear mapping network f : Z → W first produces w ∈ W (Figure 1b, left).
    - 入力潜在空間 Z の中の潜在コード z を与えるような、
    - 非線形写像 f : Z → W は、最初に w ∈ W を生成する。（図1bの左）

- For simplicity, we set the dimensionality of both spaces to 512, and the mapping f is implemented using an 8-layer MLP, a decision we will analyze in Section 4.1.
    - 単純に、我々は、両方の空間の次元を、512 に設定する。
    - そして、写像 f は、8層の MLP を使用して実装される。
    - この決定については、セクション 4.1 で分析する。

- Learned affine transformations then specialize w to styles y = (y_s,y_b) that control adaptive instance normalization (AdaIN) [27, 17, 21, 16] operations after each convolution layer of the synthesis network g.
    - w を スタイル y = (y_s,y_b) に特定するように学習されたアフィン変換
    - （このスタイル y というのは、）合成されたネットワーク G の各畳み込みの後で、adaptive instance normalization (AdaIN) 演算子を制御するようなもの

> A：w を style=(y_s,y_b) に変換するためののアフィン変換

> 潜在変数wはアフィン変換されることでスタイルyになる

- The AdaIN operation is defined as

![image](https://user-images.githubusercontent.com/25688193/57177273-b8bfa900-6e9d-11e9-829d-8db08864a8fb.png)<br>

- where each feature map x_i is normalized separately, and then scaled and biased using the corresponding scalar components from style y.
    - ここで、各特徴マップ x_i は、別々に [separately] 正規化されており、
    - そして、次に、スタイル y からの一致するスカラー成分 [components] を使用して、スケーリングされ、バイアス化される。

- Thus the dimensionality of y is twice the number of feature maps on that layer.
    - それ故、y の次元は、この層において、２倍の特徴マップの枚数である。

---

![image](https://user-images.githubusercontent.com/25688193/57177089-68474c00-6e9b-11e9-8568-d038a8e7e9ec.png)<br>

- > Figure 1.

- > While a traditional generator [30] feeds the latent code though the input layer only, we first map the input to an intermediate latent space W, which then controls the generator through adaptive instance normalization (AdaIN) at each convolution layer.
    - > 伝統的な生成器が、潜在コードを、入力層のみ通す一方で、
    - > 我々は、最初に、入力を、中間潜在空間 W へ写像する。
    - > （この中間潜在空間というのは、）各畳み込み層で、adaptive instance normalization (AdaIN) を通じて、生成器を制御するようなもの

- > Gaussian noise is added after each convolution, before evaluating the nonlinearity.
    - > 非線形を評価する前に、ガウスノイズは、各畳み込み層の後に追加される。

- > Here “A” stands for a learned affine transform, and “B” applies learned per-channel scaling factors to the noise input.
    - > ここの "A" は、学習されたアフィン変換を表現している。
    - > そして、"B" は、学習されたチャンネル単位のスケーリング要素に、ノイズ入力を適用すること（を表している。）

- > The mapping network f consists of 8 layers and the synthesis network g consists of 18 layers—two for each resolution (4^2 - 1024^2).
    - > 写像ネットワーク f は、8 個の層から構成され、合成ネットワーク g は、各解像度に対して、18 層から構成される。

- > The output of the last layer is converted to RGB using a separate 1 × 1 convolution, similar to Karras et al. [30].
    - > 最後の層の出力は、別々の 1 × 1 の畳み込みを使用して、RGB に変換される。

- > Our generator has a total of 26.2M trainable parameters, compared to 23.1M in the traditional generator.
    - > 我々の生成器は、全部で、26.2 M の学習可能なパラメーターを持ち、
    - > これは、伝統的な生成器の 23.1 M と比較される。

---

- the spatially invariant style y from vector w instead of an example image.
    - ベクトル w からの、空間的に [spatially ] 不変な [invariant] スタイル y は、１つの画像の例の代わりとなる。

- We choose to reuse the word “style” for y because similar network architectures are already used for feedforward style transfer [27], unsupervised image-toimage translation [28], and domain mixtures [23].
    - 我々は、y に対しての、"style" という用語を再利用することを選択する。
    - なぜならば、似たようなネットワークが、教師なしの image-to-image 変換である順伝搬型 style transfer と、領域を混合で、既に使用されているからである。

- Compared to more general feature transforms [38, 57], AdaIN is particularly well suited for our purposes due to its efficiency and compact representation.
    - より一般的な特徴変換と比較すると、AdaIN は、その効率性とコンパクトな表現のために、我々の目的に、特にうまく適合できる。

---

- Finally, we provide our generator with a direct means to generate stochastic detail by introducing explicit noise inputs.
    - <font color="Pink">最後に、我々は、明確な [explicit ] ノイズ入力を紹介することによって、確率的な詳細を生成するための直接的な手段 [mean] で、我々の生成器に、提供する。</font>

- These are single-channel images consisting of uncorrelated Gaussian noise, and we feed a dedicated noise image to each layer of the synthesis network.
    - これらは、無相関な [uncorrelated] ガウスノイズを含んでいる１チャンネルの画像である。
    - そして、我々は、合成ネットワークの各層へ、専用の [dedicated] ノイズ画像を供給する。

- The noise image is broadcasted to all feature maps using learned perfeature scaling factors and then added to the output of the corresponding convolution, as illustrated in Figure 1b.
    - 図１-b に示されているように、
    - ノイズ画像は、学習された　perfeature スケーリングファクターを使用して、全ての特徴マップへ、送信される [broadcasted]。
    - そして次に、一致する畳み込み層の出力に追加される。

- The implications of adding the noise inputs are discussed in Sections 3.2 and 3.3.
    - ノイズ入力を追加することでの影響 [implications] は、セクション 3.2 と 3.3 で議論される。


## 3. Properties of the style-based generator

- Our generator architecture makes it possible to control the image synthesis via scale-specific modifications to the styles.
    - 我々の生成器のアーキテクチャは、スタイルへの、スケール特化の修正を経由して、画像合成を制御することを可能にする。

- We can view the mapping network and affine transformations as a way to draw samples for each style from a learned distribution, and the synthesis network as a way to generate a novel image based on a collection of styles.
    - <font color="Pink">我々は、学習された分布からの、各スタイル変換に対してのサンプルを引き出す [drqw] 方法として、写像ネットワークとアフィン変換を見ることが出来る。</font>
    - そして、スタイルのコレクションに基づいた新しい画像を生成する方法として、合成ネットワークを見ることが出来る。

- The effects of each style are localized in the network, i.e., modifying a specific subset of the styles can be expected to affect only certain aspects of the image.
    - 各スタイルの効果は、ネットワークの中に、局在する。
    - 例えば、スタイルの特定のサブセットを修正することは、画像の正確なアスペクトのみに、影響を及ぼす [affect] ことが予想される。

---

- To see the reason for this localization, let us consider how the AdaIN operation (Eq. 1) first normalizes each channel to zero mean and unit variance, and only then applies scales and biases based on the style.

- The new per-channel statistics, as dictated by the style, modify the relative importance of features for the subsequent convolution operation, but they do not depend on the original statistics because of the normalization.

- Thus each style controls only one convolution before being overridden by the next AdaIN operation.


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

### 2.1 Quality of generated images

![image](https://user-images.githubusercontent.com/25688193/57187291-28c64180-6f27-11e9-8e27-ca4f26150dd0.png)<br>

- > Table 1. Fr´echet inception distance (FID) for various generator designs (lower is better).

- > In this paper we calculate the FIDs using 50,000 images drawn randomly from the training set, and report the lowest distance encountered over the course of training.

> これまで様々な改良をすることで画像のクオリティを上げてきた。評価指標はFID

> FIDはGANの生成分布と真のデータ分布の距離を表す指標。数値は低いほど良いことを意味する

> **（C) : 更にベースラインの改良として、マッピングネットワークとAdaIN処理を加えた。この時点で、従来通り最初のconvolution層に潜在変数を与えるネットワークでは改善の余地がないという見解に至った**

> （F) : mixing regularizationを導入により、隣接したスタイルの相互関係をなくし、より洗練された画像を生成することができる



# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


