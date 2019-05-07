# ■ 論文
- 論文タイトル："A Style-Based Generator Architecture for Generative Adversarial Networks"
- 論文リンク：https://arxiv.org/abs/1812.04948
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
    - 補間クオリティと disentanglement を定量化 [quantify] するために、
    - 我々は、どんな生成器のアーキテクチャにも適用できる２つの新しい自動化手法を提案する。

> disentanglement : <br>
> 直訳すると、（複雑に絡み合った問題の）”解きほぐし” の意味であるが、ここでの意味は、データのもつれを解く最適な表現方法を獲得す（disentanglement）こと。

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
    - そして、潜在変数において、各畳み込み層ベースでの、画像の ”スタイル” を調整する。
    - それ故、異なるスケールで、画像の特徴の強度を直接的にコントロールする。

- Combined with noise injected directly into the network, this architectural change leads to automatic, unsupervised separation of high-level attributes (e.g., pose, identity) from stochastic variation (e.g., freckles,hair) in the generated images, and enables intuitive scale-specific mixing and interpolation operations.
    - ネットワークの中に、直接的に注入されたノイズを組み合わせることで、
    - <font color="Pink">このアーキテクチャの変更は、生成された画像において、確率的な多様性（例えば、そばかすや髪）から、高レベルの特徴（例えば、顔の向きや特徴）の、自動的で、教師なしの分割？に導く。</font>

- We do not modify the discriminator or the loss function in any way, and our work is thus orthogonal to the ongoing discussion about GAN loss functions, regularization, and hyperparameters [24, 45, 5, 40, 44, 36].
    - 我々は、あらゆる方法で、識別器や損失関数を修正しない。
    - そして、我々の研究は、そういうわけで、GAN の損失関数、正則化、ハイパーパラメータについて、進行中の議論と直交している。（＝別方向である）

---

- Our generator embeds the input latent code into an intermediate latent space, which has a profound effect on how the factors of variation are represented in the network.
    - **我々の生成器は、入力潜在変数を、中間の [intermediate] 潜在空間へ埋め込む。**
    - **（この潜在空間というのは、）多様性の要因が、どのようにして、ネットワークの中で表されるのかということに、深い [profound] 影響を与えるような（潜在空間）**

- The input latent space must follow the probability density of the training data, and we argue that this leads to some degree of unavoidable entanglement.
    - 入力潜在空間は、学習データの確率分布に従わなければならない。
    - そして、我々は、このことが、ある程度 [some degree of] 避けられられない絡み合い [entanglement] に至る [leads to] ということを主張する。

- Our intermediate latent space is free from that restriction and is therefore allowed to be disentangled.
    - 我々の中間の潜在空間は、このような制限から自由であり、
    - それ故に、絡み合った謎を解きほぐす [disentangled] ことを許容する。

- As previous methods for estimating the degree of latent space disentanglement are not directly applicable in our case, we propose two new automated metrics—perceptual path length and linear separability—for quantifying these aspects of the generator.
    - 潜在空間の disentanglement（解きほぐし）の程度を推定するための以前の方法は、我々のケースに、直接的に応用出来ないので、
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
    - 伝統的に、潜在変数は、生成器の１つの入力層を通って、提供される。
    - 例えば、順伝搬ネットワークの最初の層。（図1a）

- We depart from this design by omitting the input layer altogether and starting from a learned constant instead (Figure 1b, right). 
    - 我々は、入力層を完全に省略 [omitting] し 、代わりに学習済み定数から開始することによって 、このデザインから出発する。

- Given a latent code z in the input latent space Z, a non-linear mapping network f : Z → W first produces w ∈ W (Figure 1b, left).
    - 入力潜在空間 Z の中の潜在変数 z を与えるような、
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
    - > 伝統的な生成器が、潜在変数を、入力層のみ通す一方で、
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
    - この局在化の理由を見るために、AdaIN 演算が、どのよにして、最初に各チャンネルを、平均値０や分散値１に、正規化するのかを、考察する。
    - そして、そのときのみ、スタイルに基づいた、スケールとバイアスを適用する。

- The new per-channel statistics, as dictated by the style, modify the relative importance of features for the subsequent convolution operation, but they do not depend on the original statistics because of the normalization.
    - <font color="Pink">スタイルによって決定づけられる [dictated] ような、チャンネル単位の新しい統計では、それに続く [subsequent] 畳み込み演算に対して、特徴の相対的な [relative] 重要性を修正する。
    - しかし、正規化のため、オリジナルの統計に依存しない。</font>

- Thus each style controls only one convolution before being overridden by the next AdaIN operation.
    - それ故、次の AdaIN 演算によって、上書きされる前に、各スタイルは、１つの畳み込みのみ制御する。

### 3.1. Style mixing

- To further encourage the styles to localize, we employ mixing regularization, where a given percentage of images are generated using two random latent codes instead of one during training.
    - スタイルを局在化することを、更に促す [encourage] ために、
    - 我々は、mixing regularization を使用する。
    - （この mixing regularization というのは、）与えられた画像のパーセンテージが、学習の間に、１つ潜在変数の代わりに、の２つの潜在変数を使用して、生成される。

- When generating such an image, we simply switch from one latent code to another—an operation we refer to as style mixing—at a randomly selected point in the synthesis network.
    - そのような画像を生成するとき、
    - 合成ネットワークにおいて、ランダムに選択された点で、
    - 我々は、単純に、１つの潜在変数から、別の１つの演算（これをスタイル混合 [style mixing]という）に、交換する。

- To be specific, we run two latent codes z_1,z_2 through the mapping network, and have the corresponding w_1,w_2 control the styles so that w_1 applies before the crossover point and w2 after it.
    - 具体的に言うと、我々は、写像ネットワークを通じて、２つの潜在変数 z_1,z_2 を動作する。
    - そして、対応している w_1 と w_2 で、スタイルを制御する。
    - w_1 交差している点の前に適用し、
    - そして、w_2 は、それの後に、（適用する。）

- This regularization technique prevents the network from assuming that adjacent styles are correlated.
    - **この正則化テクニックは、ネットワークが、隣接する [adjacent] スタイルが相関しているという仮定することを防ぐ。**

> Synthesis networkのランダムに選ばれたところで潜在変数を他のものに切り替える(これをstyle mixingと呼ぶ）

> 具体的には、Mapping networkではz1とz2という２つの潜在変数を実行する Synthesis networkではw1を適用して、その後ランダムに選ばれたところからw2を適用する

> この正則化手法により、隣り合ったスタイルは相互に関連しているということをネットワークが仮定せずに生成することができる

### 3.2 Stochastic variation

- There are many aspects in human portraits that can be regarded as stochastic, such as the exact placement of hairs, stubble, freckles, or skin pores.
    - 髪の毛、ひげ [stubble]、そばかす [freckles]、肌質 [skin pores] の詳細な配置 [placement] などのように、
    - 人間の肖像画 [portraits] において、確率的とみなせるような、多くの側面が存在する。

- Any of these can be randomized without affecting our perception of the image as long as they follow the correct distribution.
    - これらのいくつかは、それらが正しい分布に従っている限りは、我々の感覚 [perception] に影響を与えることなしに、ランダム化出来る。

---

- Let us consider how a traditional generator implements stochastic variation.
    - 従来の生成器が、どのようにして、確率的変動を実装しているのかを考える。

- Given that the only input to the network is through the input layer, the network needs to invent a way to generate spatially-varying pseudorandom numbers from earlier activations whenever they are needed.
    - <font color="Pink">ネットワークへの唯一の入力が、入力層を通ると仮定すれば [Given that]、
    - ネットワークは、必要なときはいつでも、初期の活性化？から、空間的に [spatially] 変化する [varying] 疑似乱数 [pseudorandom] を生成する方法を、発明する [invent] 必要がある。</font>

- This consumes network capacity and hiding the periodicity of generated signal is difficult—and not always successful, as evidenced by commonly seen repetitive patterns in generated images. 
    - これは、ネットワークの容量を消費し、
    - 生成された信号の周期性を隠すことは、困難であり、
    - いつもは成功しない。
    - 生成された画像において、共通に見られる繰り返しの [repetitive] パターン、からも明らかなように [as evidenced by]、

- Our architecture sidesteps these issues altogether by adding per-pixel noise after each convolution.
    - **我々のアーキテクチャは、各畳み込み層の後に、ピクセル単位のノイズを加えることにより、これらの問題を回避 [sidestep] する。**


### 3.3. Separation of global effects from stochasticity

- The previous sections as well as the accompanying video demonstrate that while changes to the style have global effects (changing pose, identity, etc.), the noise affects only inconsequential stochastic variation (differently combed hair, beard, etc.).
    - 前のセクションとそれに付随するビデオでは、スタイルの変更が全体的な効果（ポーズ、アイデンティティなどの変更）を持つのに対して、ノイズは重要でない確率的変動（異なる髪の毛、ひげなど）にのみ影響を与えることを示しています。

- This observation is in line with style transfer literature, where it has been established that spatially invariant statistics (Gram matrix, channel-wise mean, variance, etc.) reliably encode the style of an image [20, 39] while spatially varying features encode a specific instance.
    - この観察は、空間的に不変な統計量（グラム行列、チャネルワイズ平均、分散など）が画像のスタイルを確実に符号化し、空間的に変化する特徴が符号化することが確立されている。 特定のインスタンス

---

- In our style-based generator, the style affects the entire image because complete feature maps are scaled and biased with the same values.
    - **我々の style-based 生成器においては、スタイルは画像全体に影響を及ぼす。**
    - **なぜならば、完全な特徴マップが、同じ値でスケール化され、バイアス化されているためである。**

- Therefore, global effects such as pose, lighting, or background style can be controlled coherently.
    - それ故に、姿勢、明暗、背景などのグローバルなスタイルの影響は、首尾一貫して [coherently] 制御される。

- Meanwhile, the noise is added independently to each pixel and is thus ideally suited for controlling stochastic variation.
    - その一方で [Meanwhile]、ノイズは、各ピクセルに独立して追加される。
    - そしてそれ故に、確率的変動を制御することに理想的である。

- If the network tried to control, e.g., pose using the noise, that would lead to spatially inconsistent decisions that would then be penalized by the discriminator.
    - ネットワークが、例えばノイズを使用してポーズを制御しようとした場合、それは空間的に矛盾する決定を導き、それは識別器によって不利になる。

- Thus the network learns to use the global and local channels appropriately, without explicit guidance.
    - したがって、ネットワークは、明示的なガイダンスなしに、グローバルチャネルとローカルチャネルを適切に使用することを学習します。


## 4. Disentanglement studies

- There are various definitions for disentanglement [54, 50, 2, 7, 19], but a common goal is a latent space that consists of linear subspaces, each of which controls one factor of variation.
    - disentanglement（データのもつれの解きほぐし） に対しての、様々な定義が存在する。[54, 50, 2, 7]
    - しかし、共通のゴールは、各部分空間が、１つの変動要因を制御するような、線形部分空間で構成される潜在空間となる。

> disentanglement:<br>
> 直訳すると、（複雑に絡み合った問題の）”解きほぐし” の意味であるが、ここでの意味は、データのもつれを解く最適な表現方法を獲得する（disentanglement）こと。

- However, the sampling probability of each combination of factors in Z needs to match the corresponding density in the training data. 
    - しかしながら、潜在空間 Z の中の、各要因の各組み込せのサンプリング確率は、学習データにおける確率密度と一致する必要がある。

- As illustrated in Figure 6, this precludes the factors from being fully disentangled with typical datasets and input latent distributions.(2)
    - 図６で図示したように、これは、典型的な [typical] データセットや入力潜在分布と共に、完全な絡み合いからの要因を、排除する [precludes]。

![image](https://user-images.githubusercontent.com/25688193/57224830-8486ec80-7045-11e9-952b-05814acf0223.png)<br>

- > Figure 6. Illustrative example with two factors of variation (image features, e.g., masculinity and hair length).
    - > 図６：２つの変動の要因（画像の特徴、例えば、男らしさや髪の長さ）での実例 [Illustrative]

- > (a) An example training set where some combination (e.g., long haired males) is missing.
    - > (a) いくつかの組み合わせ（例えば、長髪の男性）が欠けている学習セットの例

- > (b) This forces the mapping from Z to image features to become curved so that the forbidden combination disappears in Z to prevent the sampling of invalid combinations.
    - > (b) これは、Z から画像特徴への写像が、曲線なることを、強制する。
    - > 禁止された組み合わせが、Z において、消えるようなるように、
    - > 無効な組み合わせのサンプリングを防止するために、

- > (c) The learned mapping from Z to W is able to “undo” much of the warping.
    - > (c) Z から W への学習された写像は、歪み [warping] の多くを、元に戻す [undo] ことが出来る。

---

- A major benefit of our generator architecture is that the intermediate latent space W does not have to support sampling according to any fixed distribution; its sampling density is induced by the learned piecewise continuous mapping f(z).
    - 我々の生成器のアーキテクチャの主な利点は、中間潜在空間 W が、いかなる固定された確率分布に従うサンプリングを、サポートする必要がないといことである。
    - 即ち、そのサンプリング密度は、学習されたピクセル単位での連続写像 f(z) によって、誘発される。

- This mapping can be adapted to “unwarp”W so that the factors of variation become more linear.
    - この写像は、変動の要因がより線形になるように、”歪み” W に、適用させることが出来る。

- We posit that there is pressure for the generator to do so, as it should be easier to generate realistic images based on a disentangled representation than based on an entangled representation.
    - 我々は、生成器に対して、そうするようにする圧力があるということを、仮定する [posit]。
    - entangled（絡み合った）表現よりも、disentangled （解きほぐした）表現のほうで、リアルな画像を生成することが、より簡単であるので、

- As such, we expect the training to yield a less entangled W in an unsupervised setting, i.e., when the factors of variation are not known in advance [10, 35, 49, 8, 26, 32, 7].
    - このようにして、我々は、教師なし設定において、より少なく絡み合った中間潜在空間 W を生み出すための学習を期待する。
    - （この教師なし設定というのは、）例えば、変動の要因が前もって知らないようなもの。

---

- Unfortunately the metrics recently proposed for quantifying disentanglement [26, 32, 7, 19] require an encoder network that maps input images to latent codes.
    - 残念なことに、disentanglement（解きほぐし）を定量化するために、提案された最近の指標は、
    - 入力画像を潜在変数へ写像するような、encoder ネットワークを要求している。

- These metrics are ill-suited for our purposes since our baseline GAN lacks such an encoder.
    - 我々のベースラインの GAN が、このような enocoder を欠いているので、
    - これらの指標は、我々の目的に対して、不適当である [ill-suited]。

- While it is possible to add an extra network for this purpose [8, 12, 15], we want to avoid investing effort into a component that is not a part of the actual solution.
    - この目的に対して、追加のネットワークを加えることが可能であるが、
    - 我々は、実際の解の一部ではないような、構成への発明努力を避けたい。

- To this end, we describe two new ways of quantifying disentanglement, neither of which requires an encoder or known factors of variation, and are therefore computable for any image dataset and generator.
    - この目標を達成するために [To this end]、我々は、disentanglement を定量化するための、２つの新しい方法を記述する。
    - 両方とも、エンコーダーや変動の要因を知る必要がない。
    - そしてそれ故に、いかなる画像データセットや生成器に対して、計算可能である。

### 4.1. Perceptual path length

- As noted by Laine [37], interpolation of latent-space vectors may yield surprisingly non-linear changes in the image.
    - Laine によって指摘されたように、潜在空間のベクトルの補間は、画像において、驚くほどの非線形な変化を生み出すかもしれない。

- For example, features that are absent in either endpoint may appear in the middle of a linear interpolation path.
    - 例えば、終端の両端にも存在しない特徴は、線形補間経路 [linear interpolation path] の中間において、現れるかもしれない。

- This is a sign that the latent space is entangled and the factors of variation are not properly separated.
    - これは、潜在空間が、絡み合っており、
    - そして、変動の要因が、適切に [properly ] 分解されていないサインである。

- To quantify this effect, we can measure how drastic changes the image undergoes as we perform interpolation in the latent space. 
    - この効果を定量化するために、
    - 潜在空間において、補間を実行するときに、
    - 画像が、どのようにして、劇的に変化しているのかを、測定することが出来る。

- Intuitively, a less curved latent space should result in perceptually smoother transition than a highly curved latent space.
    - 直感的には [Intuitively]、より曲がりの少ない潜在空間は、より曲がりの強い潜在空間よりも、知覚的に [perceptually] 滑らかな変換という結果になるべきである。

![image](https://user-images.githubusercontent.com/25688193/57267615-aae76f00-70bb-11e9-9709-eb4b95075b2e.png)<br>

---

- As a basis for our metric, we use a perceptually-based pairwise image distance [65] that is calculated as a weighted difference between two VGG16 [58] embeddings, where the weights are fit so that the metric agrees with human perceptual similarity judgments. 
    - 我々の指標のための基礎として、知覚ベースのピクセル単位の画像距離を使用する。
    - （この画像距離というのは、）２つの VGG16 での埋めこみ間の加重和として、計算されたものである。
    - ここで、重みは、指標が人間の知覚での判断に近くなるように、適合される。

- If we subdivide a latent space interpolation path into linear segments, we can define the total perceptual length of this segmented path as the sum of perceptual differences over each segment, as reported by the image distance metric.
    - もし、潜在空間の補間経路を、線形セグメントに、細分する [subdivide] ならば、
    - 我々は、このセグメント化された経路の全ての perceptual path length を、
    - 各セグメントに渡っての知覚的差異の合計として、定義することが出来る。
    - 画像距離の指標によって、報告されるように、

- A natural definition for the perceptual path length would be the limit of this sum under infinitely fine subdivision, but in practice we approximate it using a small subdivision epsilon ε = 10^-4.
    - perceptual path length に対しての自然な定義は、無限に細かい分割の元で、この和の極限となる。
    - しかし、実際には、小さな分割 ε = 10^-4 を使用して、近似する。

- The average perceptual path length in latent space Z, over all possible endpoints, is therefore
    - それ故、潜在空間 Z において、可能な全ての端点に渡っての、perceptual path length の平均は、

![image](https://user-images.githubusercontent.com/25688193/57271133-492e0180-70c9-11e9-9752-8fa98466937c.png)<br>

- where z_1,z_2 ~ P(z) ; t ~ U(0; 1), G is the generator (i.e., g ○f for style-based networks), and d(・,・) evaluates the perceptual distance between the resulting images.
    - ここで、z_1,z_2 ~ P(z) ; t ~ U(0; 1),
    - G は生成器
    - d(・,・) は、画像の間の perceptual distance を評価する（関数）

- Here slerp denotes spherical interpolation [56], which is the most appropriate way of interpolating in our normalized input latent space [61].
    - ここで、slerp は、球面上の補間を示している。
    - これは、正規化された入力潜在空間において、最もより補間の方法である。

- To concentrate on the facial features instead of background, we crop the generated images to contain only the face prior to evaluating the pairwise image metric.
    - 背景の代わりに顔の特徴に集中するために、ペア単位の画像指標を評価する前に、生成された画像を、顔だけを含むように切り取る [crop]。

- As the metric d is quadratic [65], we divide by ε^2.
    - 指標 d は２次 [quadratic] になるように、ε^2 で割る。

- We compute the expectation by taking 100,000 samples.
    - 我々は、100,000 個のサンプルを取ることによって、期待値 [expectation] を計算する。

---

- Computing the average perceptual path length in W is carried out in a similar fashion:
    - 中間潜在空間 W において、平均 perceptual path length を計算することは、よく似た形式で実行される。

![image](https://user-images.githubusercontent.com/25688193/57272084-09691900-70cd-11e9-8f99-988b227a75d4.png)<br>

- where the only difference is that interpolation happens in W space.
    - ここで、唯一の違いは、この補間が、中間潜在空間 W において発生することである。

- Because vectors in W are not normalized in any fashion, we use linear interpolation (lerp).
    - 中間潜在空間 W のベクトルは、いずれの形式でも正規化されていないので、線形補間（lerp）を使用する。

### 4.2. Linear separability

- If a latent space is sufficiently disentangled, it should be possible to find direction vectors that consistently correspond to individual factors of variation.
    - もし潜在空間が十分に分解可能であるならば、変動の各々の要因に、連続的に一致するような、方向ベクトルを見つけることが可能であるべきである。

- We propose another metric that quantifies this effect by measuring how well the latent-space points can be separated into two distinct sets via a linear hyperplane, so that each set corresponds to a specific binary attribute of the image.
    - 潜在空間の点が、線形な超平面を介して、どの程度うまく２つの集合に分離可能であるのかを計測することによって、
    - この効果を定量化するような、他の指標を提案する。
    - 各集合が、画像の特定のバイナリ属性に一致するように、

---

- In order to label the generated images, we train auxiliary classification networks for a number of binary attributes, e.g., to distinguish male and female faces.
    - 生成された画像をラベル付けするために、バイナリー属性（例えば、男性の顔と女性の顔の区別）の数に対して、補助の [auxiliary] 分類器ネットワークを学習する。

- In our tests, the classifiers had the same architecture as the discriminator we use (i.e., same as in [30]), and were trained using the CELEBA-HQ dataset that retains the 40 attributes available in the original CelebA dataset. 
    - 我々のテストにおいては、この分類機は、識別器と同じアーキテクチャでを持つ。
    - そして、オリジナルの CelebA データセットの中の 40 個の属性を保持するような、CELEBA-HQ データセットを用いて学習された。

- To measure the separability of one attribute, we generate 200,000 images with z ~ P(z) and classify them using the auxiliary classification network.
    - １つの属性の分離性を測定するために、ノイズ z ~ P(z) で、200,000 枚の画像を生成する。
    - そして、補助的な分類ネットワークを用いて、それらを分類する。

- We then sort the samples according to classifier confidence and remove the least confident half, yielding 100,000 labeled latent-space vectors.
    - 次に、分類器の信用性に基づき、サンプルをソートする。
    - そして、最も小さい信用度の半分（のサンプル）を除外し、
    - （信用度の大きいもう半分の）100,00 個のラベリングされた潜在空間のベクトルを生成する。

---

- For each attribute, we fit a linear SVM to predict the label based on the latent-space point—z for traditional and w for style-based—and classify the points by this plane.
    - 各属性に対して、線形 SVM を適合する。
    - 潜在空間の点（従来では、z、スタイムベース手法では w）に基づいて、ラベルを予想し、
    - この（超）平面によって、点を分類するために、

- We then compute the conditional entropy H(Y|X) where X are the classes predicted by the SVM and Y are the classes determined by the pre-trained classifier.
    - 次に、条件付きエントロピー H(Y|X) を計算する。
    - ここで、X は、SVM によって予想されたクラスであり、
    - Y は、事前学習された分類器によって、決定されたクラスである。

- This tells how much additional information is required to determine the true class of a sample, given that we know on which side of the hyperplane it lies.
    - それ（＝１つのサンプル）が、超平面のどちらの側に横たわっていることを知っていることを考慮すれば、
    - これ（＝条件付きエントロピー）は、１つのサンプルの真のクラスを決定するために、どのくらい多くの追加情報が、必要であるかを教える。

- A low value suggests consistent latent space directions for the corresponding factor(s) of variation.
    - 小さな値は、対応する [corresponding] 変動の要因に対して、潜在空間の方向が、一致する [consistent] ということを提案する。

---

- We calculate the final separability score as
    - 最終的な分離スコアは、以下で計算される。

![image](https://user-images.githubusercontent.com/25688193/57273099-78487100-70d1-11e9-9559-d04b1e7395a3.png), 

- where i enumerates the 40 attributes.
    - ここで、i は属性の数

- Similar to the inception score [53], the exponentiation brings the values from logarithmic to linear domain so that they are easier to compare.
    - inception score [53] によく似ている。
    - xxx

# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

### 2.1 Quality of generated images

![image](https://user-images.githubusercontent.com/25688193/57187291-28c64180-6f27-11e9-8e27-ca4f26150dd0.png)<br>

- > Table 1. Fr´echet inception distance (FID) for various generator designs (lower is better).

- > In this paper we calculate the FIDs using 50,000 images drawn randomly from the training set, and report the lowest distance encountered over the course of training.

> これまで様々な改良をすることで画像のクオリティを上げてきた。評価指標はFID

> FIDはGANの生成分布と真のデータ分布の距離を表す指標。数値は低いほど良いことを意味する

> **（C) : 更にベースラインの改良として、マッピングネットワークとAdaIN処理を加えた。この時点で、従来通り最初のconvolution層に潜在変数を与えるネットワークでは改善の余地がないという見解に至った**

> （F) : mixing regularizationを導入により、隣接したスタイルの相互関係をなくし、より洗練された画像を生成することができる


### 3.1. Style mixing

- Table 2 shows how enabling mixing regularization during training improves the localization considerably, indicated by improved FIDs in scenarios where multiple latents are mixed at test time.
    - 表２は、学習中に、有効な mixing regularization が、局在化を大幅に [considerably] 改善することを示している。
    - （このことは、）テスト時に複数の潜在変数が混在するようなシナリオにおいて、改善された FID によって、示される [indicated]

![image](https://user-images.githubusercontent.com/25688193/57204798-da7f7400-6ff4-11e9-9ccb-43599c5fb9d2.png)<br>

- > Table 2.

- > FIDs in FFHQ for networks trained by enabling the mixing regularization for different percentage of training examples.
    - > 有効な mixing regularization によって学習されたネットワークに対してと、学習サンプルの異なるパーセンテージに対しての、FFHQ データセットでの FIDs。

- > Here we stress test the trained networks by randomizing 1....4 latents and the crossover points between them.
    - > ここで、ランダムな 1~4 個の潜在変数とそれらの間の交差点によって、学習されたネットワークを、ストレステストする。

- > Mixing regularization improves the tolerance to these adverse operations significantly.
    - > Mixing regularization は、これら敵対的な演算への耐性 [tolerance] を大幅に改善する。

- > Labels E and F refer to the configurations in Table 1.
    - > ラベル E と F は、表１の設定を参照している。

> 潜在変数の数(Number of latents during testing)と潜在変数を混ぜて生成する比率(Mixing regularization)を変えて結果を表にして示している。<br>
> 評価指標はFID<br>
> FIDはGANの生成分布と真のデータ分布の距離を表す指標。数値は低い方が良い<br>
> Mixing regularizationの比率を大きくすることで結果が良くなっているのがわかる<br>

---

- Figure 3 presents examples of images synthesized by mixing two latent codes at various scales.
    - 図３は、様々なスケールでの、２つの潜在変数の混合によって、合成された画像の例を提示している。

- We can see that each subset of styles controls meaningful high-level attributes of the image.
    - 我々は、スタイルの各サブセットが、意味のある画像の高レベルの特徴を制御していることが、見てとれる。

![image](https://user-images.githubusercontent.com/25688193/57206214-0f90c400-6fff-11e9-906f-289ff848a498.png)<br>

- > Figure 3.

- > Two sets of images were generated from their respective latent codes (sources A and B); the rest of the images were generated by copying a specified subset of styles from source B and taking the rest from source A.
    - > 画像の２つのセットは、各々の [respective] 潜在変数から生成された。（ソースA,ソースB）
    - > 残りの [the rest of] 画像は、ソースBとからの、特定のスタイルのサブセットのコピーによって、
    - > 残りを、ソース A からの取得で生成された。

- > Copying the styles corresponding to coarse spatial resolutions (4^2 – 8^2) brings high-level aspects such as pose, general hair style, face shape, and eyeglasses from source B, while all colors (eyes, hair, lighting) and finer facial features resemble A.
    - > **粗い [coarse] 空間的な [spatial] 解像度 (4*4 ~ 8*8pixel) での一致のコピー、ソースBからの姿勢、髪型、顔の輪郭、メガネ [eyeglasses] などの高レベルの側面をもたらす。**
    - > **目や髪の明暗などの全ての色や、より細かい [finer] 顔の特徴が、Aに似ている一方で、**

> 解像度が粗い段階で合わせた場合

- > If we instead copy the styles of middle resolutions (16^2 – 32^2) from B, we inherit smaller scale facial features, hair style, eyes open/closed from B, while the pose, general face shape, and eyeglasses from A are preserved.
    - > もしも、ソース B からの中間の解像度のスタイルを代わりにコピーすると、
    - > ソースBから、より小さいスケースでの顔の特徴、髪型、開いてる目、閉じてる目を継承する。
    - > ソース A から、顔の姿勢や輪郭、メガネが保存される一方で、

- > Finally, copying the fine styles (64^2 – 1024^2) from B brings mainly the color scheme and microstructure.

> 1つの潜在変数で生成されたスタイル（ソースA：Destination）に、同じく1つの潜在変数から生成されたスタイル（ソースB:Source)を合わせると画像はどのように変化していくのかを可視化している。


### 3.2 Stochastic variation

- Figure 4 shows stochastic realizations of the same underlying image, produced using our generator with different noise realizations.
    - 図４は、同じ根本的な [underlying] 画像の、確率的な具現化 [realizations] を示している。
    - （この根本的な画像というのは、）異なるノイズの具現化 [realization] で、我々の生成器を使用して生成された（画像）

- We can see that the noise affects only the stochastic aspects, leaving the overall composition and high-level aspects such as identity intact.
    - ノイズは、確率的な側面のみに影響を与えることが見てとれる。
    - 完全なままの [intact] アイデンティティーのように、全体的な構成 [composition] と高レベルの側面は残る。[leave]

![image](https://user-images.githubusercontent.com/25688193/57207929-8fbd2680-700b-11e9-9198-90545e2a00d0.png)<br>

- > Figure 4. Examples of stochastic variation.

- > (a) Two generated images.

- > (b) Zoom-in with different realizations of input noise.

- > While the overall appearance is almost identical, individual hairs are placed very differently.
    - > 全体的な外面 [appearance] は、とぼ同じである [identical] が、各々の髪の毛は、非常に異なって、配置されている。

- > (c) Standard deviation of each pixel over 100 different realizations, highlighting which parts of the images are affected by the noise.
    - > 100 個の異なる具現化 [realizations] の標準偏差。
    - > 画像の一部のハイライトは、ノイズによって影響を受ける場所である。

- > The main areas are the hair, silhouettes, and parts of background, but there is also interesting stochastic variation in the eye reflections.
    - > 主要な場所は、髪、シルエット、背景の一部である。
    - > しかし、目の反射の中の、確率的変動にも興味がある。

- > Global aspects such as identity and pose are unaffected by stochastic variation.
    - > アイデンティティーや姿勢といった大域的な側面は、確率的変動 [stochastic variation] によって、影響を受けない。

---

- Figure 5 further illustrates the effect of applying stochastic variation to different subsets of layers.
    - 図５は、

- Since these effects are best seen in animation, please consult the accompanying video for a demonstration of how changing the noise input of one layer leads to stochastic variation at a matching scale.

![image](https://user-images.githubusercontent.com/25688193/57208427-b3ce3700-700e-11e9-9bf7-78f98d21bff7.png)<br>

- > Figure 5. Effect of noise inputs at different layers of our generator.

- > (a) Noise is applied to all layers.

- > (b) No noise.

- > (c) Noise in fine layers only (64^2 – 1024^2).

- > (d) Noise in coarse layers only (4^2 – 32^2).

- > We can see that the artificial omission of noise leads to featureless “painterly” look.

- > Coarse noise causes large-scale curling of hair and appearance of larger background features, while the fine noise brings out the finer curls of hair, finer background detail, and skin pores.

---

- We find it interesting that the effect of noise appears tightly localized in the network.
    - 我々は、ノイズの影響が、ネットワークの中にびっしりと [tightly ] 局在化しているということが興味深いことを見出した。

- We hypothesize that at any point in the generator, there is pressure to introduce new content as soon as possible, and the easiest way for our network to create stochastic variation is to rely on the noise provided.
    - 我々は、生成器のいかなる点においても、仮定する。
    - 出来るだけ早く、新しい content を、紹介するプレッシャーが存在する。
    - そして、我々のネットワークに対して、stochastic variation を作るための最も早い方法は、ノイズ供給に頼ることである。

- A fresh set of noise is available for every layer, and thus there is no incentive to generate the stochastic effects from earlier activations, leading to a localized effect.
    - ノイズの新しいセットは、全ての層に対して適用可能であり、
    - <font color="Pink">それ故に、局在化な影響に導くような、初期の活性化から、確率的な影響を生成するための動機 [incentive ] は存在しない。</font>

### 4.1. Perceptual path length

- Table 3 shows that this full-path length is substantially shorter for our style-based generator with noise inputs, indicating thatW is perceptually more linear than Z.

- Yet, this measurement is in fact slightly biased in favor of the input latent space Z.

- If W is indeed a disentangled and “flattened” mapping of Z, it may contain regions that are not on the input manifold—and are thus badly reconstructed by the generator—even between points that are mapped from the input manifold, whereas the input latent space Z has no such regions by definition.

- It is therefore to be expected that if we restrict our measure to path endpoints, i.e., t 2 f0; 1g, we should obtain a smaller lW while lZ is not affected.

- This is indeed what we observe in Table 3.

---

- Table 4 shows how path lengths are affected by the mapping network.

- We see that both traditional and style-based generators benefit from having a mapping network, and additional depth generally improves the perceptual path length as well as FIDs.

- It is interesting that while lW improves in the traditional generator, lZ becomes considerably worse, illustrating our claim that the input latent space can indeed be arbitrarily entangled in GANs.


# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


