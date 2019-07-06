# ■ 論文
- 論文タイトル："Semantic Image Synthesis with Spatially-Adaptive Normalization"
- 論文リンク：https://arxiv.org/abs/1903.07291
- 論文投稿日付：
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- We propose spatially-adaptive normalization, a simple but effective layer for synthesizing photorealistic images given an input semantic layout.
    - 入力セマンティックレイアウトが与えられたフォトリアリスティックな画像を合成するための単純だが効果的な層である spatially-adaptive normalization を提案する。

- Previous methods directly feed the semantic layout as input to the deep network, which is then processed through stacks of convolution, normalization, and nonlinearity layers.
    - 以前の方法は、セマンティックレイアウトをディープネットワークへの入力として直接供給し、
    - それ（＝セマンティックレイアウト）を次に、畳み込み層・正規化層・および非線形層の重ね合わせを通して処理される。

- We show that this is suboptimal as the normalization layers tend to “wash away” semantic information.
    - 正規化層が、セマンティック情報を洗い流す [“wash away”] 傾向があるので、我々はこれが準最適解 [suboptimal] であることを示す。

- To address the issue, we propose using the input layout for modulating the activations in normalization layers through a spatially- adaptive, learned transformation. 
    - この問題を解決するために、
    - 我々は、spatially- adaptive（空間的に適合して）学習された変換を介して、正規化レイヤにおいて、活性化値を調整する [modulating] ために入力レイアウトを使用することを提案する。

- Experiments on several challenging datasets demonstrate the advantage of the proposed method over existing approaches, regarding both visual fidelity and alignment with input layouts.
    - いくつかの挑戦的なデータセットに関する実験は、視覚的な忠実度 [fidelity] と入力レイアウトの整列 [alignment] の両方に関して、既存のアプローチに対する提案された方法の利点を実証している。

- Finally, our model allows user control over both semantic and style as synthesizing images.
    - 最後に、私たちのモデルは、画像を合成するときに意味（＝セマンティック）とスタイルの両方をユーザーが制御できるようにします。

- Code will be available at https: //github.com/NVlabs/SPADE .


# ■ イントロダクション（何をしたいか？）

## 1. Introduction

- Conditional image synthesis refers to the task of gen- erating photorealistic images conditioning on some input data.
    - 条件付き画像合成は、いくつかの入力データにおいて、フォトリアリスティックな画像条件を生成するタスクのことをを指す

- Earlier methods compute the output image by stitching pieces from a database of images [3, 13].
    - 以前の方法は、画像のデータベースから断片をつなぎ合わせること [stitching] によって出力画像を計算する[3、13]。

- Recent methods directly learn the mapping using neural net- works[4,7,20,39,40,45,46,47].

- The latter methods are generally faster and require no external database of images.
    - 後者の方法は一般的に速く、そして画像の外部データベースを必要としない。

---

- We are interested in a specific form of conditional image synthesis, which is converting a semantic segmentation mask to a photorealistic image.
    - セマンティックセグメンテーションマスクをフォトリアリスティックな画像に変換するような、特定の形式の条件付き画像合成に興味があります。

- This form has a wide range of applications such as content generation and image editing [7, 20, 40].
    - この形式には、コンテンツの生成や画像編集など、幅広い用途があります[7、20、40]。

- We will refer to this form as semantic image synthesis. 
    - 我々は、この形式を、セマンティック画像合成としてみなす。

- In this paper, we show that the conventional network architecture [20, 40], which is built by stacking convolutional, normalization, and nonlinearity layers, is at best sub-optimal, because their normalization layers tend to “wash away” information in input semantic masks. 
    - **この論文では、**
    - **畳み込み層・正規化層。非線形層を積み重ねた伝統的な [conventional] ネットワークが、ベストでも準最適解であることを示す。**
    - **それらの正規化層が、入力セマンティックマスクにおいて、情報を洗い流す傾向があるため、**

- To address the issue, we propose spatially-adaptive normalization, a conditional normalization layer that modulates the activations using input semantic layouts through a spatially-adaptive, learned transformation and can effectively propagate the semantic information throughout the network.
    - **この問題に対処するために、我々は、spatially-adaptive normalization を提案する。**
    - **これは、空間的に適応して学習された変換を介して、入力セマンティックレイアウトを使用して活性化値を調整し [modulates]、ネットワーク全体にセマンティック情報を効果的に伝播 [propagate] できる条件付き正規化層である。**

---

- We conduct experiments on several challenging datasets including the COCO-Stuff [5, 26], the ADE20K [48], and the Cityscapes [8]. 

- We show that with the help of our spatially-adaptive normalization layer, a compact network can synthesize significantly better results compared to several state-of-the-art methods.
    - 我々の spatially-adaptive normalization 層の助けを借りて、コンパクトなネットワークがいくつかの SOTA 手法と比較して非常に良い結果を合成できることを示す。

- Additionally, an extensive ablation study demonstrates the effectiveness of the proposed normalization layer against several variants for the semantic image synthesis task.
    - さらに、広範囲の切除研究 [ablation study] は、セマンティック画像合成タスクのためのいくつかの変種に対する提案された正規化層の有効性を実証する。

- Finally, our method supports multi- modal and style-guided image synthesis, enabling controllable, diverse outputs as shown in Figure 1.
    - 最後に、私たちの方法は、図1に示すように、制御可能で多様な出力を可能にする、マルチモーダルおよびスタイルガイド画像合成をサポートします。

---

![image](https://user-images.githubusercontent.com/25688193/60512164-d5962280-9d0e-11e9-8b09-998c251589ad.png)

- > Figure 1: Our model allows user control over both semantic and style as synthesizing an image.
    - > 私たちのモデルは、画像を合成するときに、意味（セマンティック）とスタイルの両方をユーザーが制御することを可能にします。

- > The semantic (e.g., existence of a tree) is controlled via a label map (visualized in the top row), while the style is controlled via the reference style image (visualized in the leftmost column).
    - > セマンティック（例えば、木の存在）は、ラベルマップ（上段の列に視覚化）経由で制御される。
    - > 一方で、スタイル（画風）は、参照スタイル画像（左行）経由で制御される。

- > Please visit our website for interactive image synthesis demos.

# ■ 結論

## 5. Conclusion

- We have proposed the spatially-adaptive normalization, which utilizes the input semantic layout while performing the affine transformation in the normalization layers.
    - 我々は、正規化レイヤにおいてアフィン変換を実行しながら入力セマンティックレイアウトを利用する spatially-adaptive normalization を提案した。

- The proposed normalization leads to the first semantic image synthesis model that can produce photorealistic outputs for diverse scenes including indoor, outdoor, landscape, and street scenes.
    - 提案された正規化は、屋内、屋外、風景、および街路のシーンを含む多様なシーンのためのフォトリアリスティックな出力を生成することができる最初の意味的画像合成モデルにつながる。

- We further demonstrate its application for multi-modal synthesis and guided image synthesis.
    - 我々はさらに、マルチモーダル合成および誘導画像合成へのその適用を実証する。

# ■ 何をしたか？詳細

## 3. Semantic Image Synthesis

- Let ![image](https://user-images.githubusercontent.com/25688193/60588016-d7c0b580-9dd0-11e9-8e01-1241ba0f593c.png) be a semantic segmentation mask where L is a set of integers denoting the semantic labels, and H and W are the image height and width.
    - ![image](https://user-images.githubusercontent.com/25688193/60588016-d7c0b580-9dd0-11e9-8e01-1241ba0f593c.png) をセマンティックセグメンテーションマスクとする。
    - ここで、Ｌはセマンティックラベルを表す整数の集合であり、ＨおよびＷは画像の高さおよび幅である。

- Each entry in m denotes the semantic label of a pixel.
    - m の各エントリは、ピクセルの意味ラベルを表します。

- We aim to learn a mapping function that can convert an input segmentation mask m to a photorealistic image.
    - **入力セグメンテーションマスクｍを写実的な画像に変換することができる写像関数を学ぶことを目的とする。**

### Spatially-adaptive denormalization

![image](https://user-images.githubusercontent.com/25688193/60586454-12c0ea00-9dcd-11e9-9890-9393d5345f6e.png)

- > Figure 2: In SPADE, the mask is first projected onto an embedding space, and then convolved to produce the modulation parameters γ and β. 
    - > **SPADEでは、マスクはまず埋め込み空間に投影され、次に畳み込まれて調整パラメータγとβが生成されます。**

- > Unlike prior conditional normalization methods, γ and β are not vectors, but tensors with spatial dimensions.
    - > 従来の条件付き正規化法とは異なり、γとβはベクトルではなく空間次元を持つテンソルです。

- > The produced γ and β are multiplied and added to the normalized activation element-wise.
    - > 生成されたγおよびβは乗算され、正規化された活性化要素に要素ごとに加算される。
    
---

- Let ![image](https://user-images.githubusercontent.com/25688193/60588648-71d52d80-9dd2-11e9-8275-4aeb729f92bf.png) denote the activations of the i-th layer of a deep convolutional network given a batch of N samples.
    - Ｎ個のバッチサンプルが与えられた場合、![image](https://user-images.githubusercontent.com/25688193/60588648-71d52d80-9dd2-11e9-8275-4aeb729f92bf.png) は、深層畳み込みネットワークのｉ番目の層の活性化を示すとする。

- Let Ci be the number of channels in the layer.

- Let Hi and Wi be the height and width of the activation map in the layer.

- We propose a new conditional normalization method called SPatially-Adaptive (DE)normalization1 (SPADE).
    - SPatially-Adaptive（DE）正規化1（SPADE）と呼ばれる新しい条件付き正規化方法を提案します。

- Similar to Batch Normaliza- tion [19], the activation is normalized in the channel-wise manner, and then modulated with learned scale and bias.
    - バッチ正規化[19]と同様に、活性化はチャネルごとに正規化され、その後学習されたスケールとバイアスで変調されます。 

- Figure 2 illustrates the SPADE design.
    - 図2は、SPADEデザインを示しています。 

- The activation value at site (n ∈ N,c ∈ Ci,y ∈ Hi,x ∈ Wi) is given by   
    - 場所（n∈N、c∈Ci、y∈Hi、x∈Wi）での活性化値は次式で与えられます。

![image](https://user-images.githubusercontent.com/25688193/60588806-e4460d80-9dd2-11e9-9d8d-c550d1868f67.png)

- where h_{n,c,y,x}^i is the activation at the site before normalization, μ_c^i and σ_c^i are the mean and standard deviation of the activation in channel c:
    - ここで、h_ {n、c、y、x} ^ i は正規化前のサイトでの活性化であり、μ_c^ i とσ_c^ i はチャネルcでの活性化の平均と標準偏差です。

![image](https://user-images.githubusercontent.com/25688193/60589794-4f90df00-9dd5-11e9-96fd-6217e558e42d.png)

---

- The variables γci,y,x(m) and βci,y,x(m) in (1) are the learned modulation parameters of the normalization layer.
    - 式（１）の変数γｃｉ、ｙ、ｘ（ｍ）およびβｃｉ、ｙ、ｘ（ｍ）は、正規化レイヤの学習変調パラメータである。

- In contrast to BatchNorm [19], they depend on the input segmentation mask and vary with respect to the location (y, x).
    - BatchNorm [19]とは対照的に、それらは入力セグメンテーションマスクに依存し、位置（y、x）に関して変化します。

- We use the symbol γci,y,x and βci,y,x to denote the functions that convert the input segmentation mask m to the scaling and bias values at the site (c, y, x) in the i-th activation map.
    - 入力セグメンテーションマスクmをi番目の活性化マップの場所（c、y、x）でのスケーリング値とバイアス値に変換する関数を表すために、シンボルγci、y、xとβci、y、xを使用します。

- We implement the functions γci,y,x and βci,y,x using a simple two-layer convolutional network, whose detail design can be found in the appendix.
    - 関数γci、y、xとβci、y、xを単純な二層たたみ込みネットワークを使って実装します。詳細な設計は付録にあります。

---

- In fact, SPADE is related to, and is a generalization of several existing normalization layers.
    - 実際、SPADEはいくつかの既存の正規化層に関連しており、それを一般化したものです。

- First, replacing the segmentation mask m with the image class label and making the modulation parameters spatially-invariant (i.e.,γci,y1,x1 ≡ γci,y2,x2 and βci,y1,x1 ≡ βci,y2,x2 for any y1, y2 ∈ {1,2,...,Hi} and x1,x2 ∈ {1,2,...,Wi}), we arrive at the form of Conditional Batch Normalization layer [10].
    - まず、セグメンテーションマスクｍを画像クラスラベルで置き換え、変調パラメータを空間的に不変にする（すなわち、任意のｙ１に対してγｃｉ、ｙ１、ｘ１≡γｃｉ、ｙ２、ｘ２およびβｃｉ、ｙ１、ｘ１ ≡βｃｉ、ｙ２、ｘ２、 y2∈{1,2、...、Hi}とx1、x2∈{1,2、...、Wi}）とすると、条件付きバッチ正規化レイヤ[10]の形にたどり着きます。

- Indeed, for any spatially-invariant conditional data, our method reduces to Conditional BN.
    - 確かに、空間的に不変な条件付きデータについては、我々の方法は条件付きBNに帰着する。

- Similarly, we can arrive at AdaIN [17] by replacing the segmentation mask with an- other image, making the modulation parameters spatially- invariant and setting N = 1.
    - 同様に、セグメンテーションマスクを他の画像に置き換え、変調パラメータを空間的に不変にし、N = 1に設定することでAdaIN [17]に到達することができます。

- As the modulation parameters are adaptive to the input segmentation mask, the proposed SPADE is better suited for semantic image synthesis.
    - 変調パラメータは入力セグメンテーションマスクに適応するので、提案されたＳＰＡＤＥは意味的画像合成により適している。

### SPADE generator.

- With SPADE, there is no need to feed the segmentation map to the first layer of the gener- ator, since the learned modulation parameters have encoded enough information about the label layout.
    - **SPADEでは、学習された変調パラメータがラベルレイアウトに関する十分な情報をエンコードしているので、セグメンテーションマップをジェネレータの最初のレイヤに供給する必要はありません。**

- Therefore, we discard encoder part of the generator, which is commonly used in recent architectures [20, 40].
    - **したがって、最近のアーキテクチャで一般的に使用されているジェネレータのエンコーダ部分を破棄します[20、40]。**

- This simplification results in a more lightweight network.
    - この単純化により、ネットワークはより軽量になります。

- Furthermore, similarly to existing class-conditional generators [29,31,45], the new generator can take a random vector as input, enabling a sim- ple and natural way for multi-modal synthesis [18, 50].
    - さらに、既存のクラス条件付きジェネレータ[29、31、45]と同様に、新しいジェネレータはランダムベクトルを入力として使用できるため、マルチモーダル合成のための単純で自然な方法が可能になります[18、50]。

---

![image](https://user-images.githubusercontent.com/25688193/60590813-f0809980-9dd7-11e9-90e4-3af1cf2d365c.png)

- > Figure 4: In the SPADE generator, each normalization layer uses the segmentation mask to modulate the layer activations.
    - > 図4：SPADE生成器では、各正規化レイヤはセグメンテーションマスクを使用して層の活性化値を変調します。

- > (left) Structure of one residual block with SPADE.

- > (right) The generator contains a series of SPADE residual blocks with upsampling layers.

- > Our architecture achieves better performance with a smaller number of parameters by removing the downsampling layers of leading image-to-image translation networks (pix2pixHD [40]).
    - > 私たちのアーキテクチャは、主要な画像から画像への変換ネットワークのダウンサンプリングレイヤを削除することによって、より少ない数のパラメータでより良いパフォーマンスを達成します（pix2pixHD [40]）。

---

- Figure 4 illustrates our generator architecture, which employs several ResNet blocks [14] with upsampling layers.
    - 図4は、アップサンプリングレイヤを持ついくつかのResNetブロック[14]を採用している私たちのジェネレータアーキテクチャを示しています。

- The modulation parameters of all the normalization layers are learned using SPADE.
    - すべての正規化レイヤの変調パラメータは、SPADEを使用して学習されます。

- Since each residual block operates at a different scale, SPADE downsamples the semantic mask to match the spatial resolution.
    - **各残差ブロックは異なるスケールで動作するため、SPADEは空間解像度に一致するようにセマンティックマスクをダウンサンプリングします。**

---

- We train the generator with the same multi-scale discriminator and loss function used in pix2pixHD except that we replace the least squared loss term [28] with the hinge loss term [25, 30, 45].
    - 最小二乗損失項[28]をヒンジ損失項[25、30、45]で置き換えることを除いて、pix2pixHDで使用されているものと同じマルチスケール識別器および損失関数を使用してジェネレータをトレーニングします。

- We test several ResNet-based discrim- inators used in recent unconditional GANs [1, 29, 31] but observe similar results at the cost of a higher GPU mem- ory requirement.
    - 我々は最近の無条件GAN [1、29、31]で使用されているいくつかのResNetベースの識別器をテストしますが、より高いGPUメモリ要件を犠牲にして同様の結果を観察します。

- Adding the SPADE to the discriminator also yields a similar performance.
    - 弁別子にSPADEを追加しても同様のパフォーマンスが得られます。

- For the loss function, we observe that removing any loss term in the pix2pixHD loss function lead to degraded generation results.
    - 損失関数については、pix2pixHD損失関数の損失項を削除すると生成結果が悪化することがわかりました。

### Why does SPADE work better?

- A short answer is that it can better preserve semantic information against common normalization layers. Specifically, while normalization layers such as the InstanceNorm [38] are essential pieces in almost all the state-of-the-art conditional image synthesis models [40], they tend to wash away semantic information when applied to uniform or flat segmentation masks
    - 簡単な答えは、それが一般的な正規化層に対してセマンティック情報をよりよく保存することができるということです。 
    - 具体的には、InstanceNorm [38]のような正規化層は、ほとんどすべての　SOTA の条件付き画像合成モデル[40]において不可欠な要素ですが、均一またはフラットセグメンテーションマスクに適用するとセマンティック情報を洗い流す傾向があります。

- Let us consider a simple module that first applies convolution to a segmentation mask and then normalization. Furthermore, let us assume that a segmentation mask with a single label is given as input to the module (e.g., all the pixels have the same label such as sky or grass). Under this setting, the convolution outputs are again uniform with different labels having different uniform values. Now after we apply InstanceNorm to the output, the normalized activation will become all zeros no matter what the input semantic label is given. Therefore, semantic information is totally lost. This limitation applies to a wide range of generator architectures, including pix2pixHD and its variant that concatenates the semantic mask at all intermediate layers, as long as a network applies convolution and then normalization to the semantic mask. In Figure 3, we empirically show this is precisely the case for pix2pixHD. Because a segmentation mask consists of a few uniform regions in general, the issue of information loss emerges when applying normalization.
    - 最初に畳み込みをセグメンテーションマスクに適用し、次に正規化を適用する単純なモジュールを考えてみましょう。さらに、単一のラベルを有するセグメンテーションマスクがモジュールへの入力として与えられると仮定しよう（例えば、すべてのピクセルは空または芝生のような同じラベルを有する）。この設定の下では、畳み込み出力はやはり均一であり、異なるラベルは異なる均一値を有する。 InstanceNormを出力に適用した後、入力意味ラベルが何であっても正規化された活性化はすべてゼロになります。したがって、意味情報は完全に失われます。この制限は、ネットワークが畳み込みを適用してからセマンティックマスクに正規化を適用する限り、すべての中間層でセマンティックマスクを連結するpix2pixHDおよびそのバリアントを含む、幅広いジェネレーターアーキテクチャに適用されます。図3では、これがpix2pixHDの場合とまったく同じであることを経験的に示しています。セグメンテーションマスクは一般にいくつかの均一な領域からなるため、正規化を適用すると情報損失の問題が発生します。

- In contrast, the segmentation mask in the SPADE Gen- erator is fed through spatially adaptive modulation without normalization. Only activations from the previous layer are normalized. Hence, the SPADE generator can better pre- serve semantic information. It enjoys the benefit of normal- ization without losing the semantic input information.
    - これとは対照的に、SPADEジェネレータのセグメンテーションマスクは、正規化なしで空間適応変調を介して供給されます。 前のレイヤーからのアクティベーションのみが正規化されます。 したがって、SPADEジェネレータはセマンティック情報をより適切に保存できます。 意味入力情報を失うことなく正規化の恩恵を受けます。

### Multi-modal synthesis.

- By using a random vector as the input of the generator, our architecture provides a simple way for multi-modal synthesis. Namely, one can attach an encoder that processes a real image into a random vector, which will be then fed to the generator. The encoder and generator form a variational autoencoder [22], in which the encoder tries to capture the style of the image, while the generator combines the encoded style and the segmentation mask information via SPADE to reconstruct the original im- age. The encoder also serves as a style guidance network at test time to capture the style of target images, as used in Fig- ure 1. For training, we add a KL-Divergence loss term [22].
    - 生成器の入力としてランダムベクトルを使用することによって、私たちのアーキテクチャはマルチモーダル合成のための簡単な方法を提供します。
    - すなわち、実画像を処理する符号器をランダムベクトルに取り付けることができ、それは次に発生器に供給される。
    - エンコーダーと生成器は変分オートエンコーダー（VAE）[22]を形成し、エンコーダーはエンコードされたスタイルとSPADEを介してセグメンテーションマスク情報を組み合わせて元の画像を再構築します。
    - 図1で使用されているように、エンコーダはテスト時にターゲット画像のスタイルをキャプチャするスタイルガイダンスネットワークとしても機能します。
    - **トレーニングのために、KL-Divergence loss termを追加します[22]。**


---

![image](https://user-images.githubusercontent.com/25688193/60729693-4bec8c00-9f7e-11e9-8be1-29cd76bb8242.png)

- > Figure 3: Comparing results given uniform segmentation maps: while SPADE generator produces plausible textures, pix2pixHD [40] produces identical outputs due to the loss of the semantic information after the normalization layer.
    - > 図3：一様なセグメンテーションマップを与えられた結果を比較する：
    - > ＳＰＡＤＥ生成器はもっともらしいテクスチャを生成するが、ｐｉｘ２ｐｉｘＨＤ ［４０］は正規化層の後の意味情報の損失のために同一の出力を生成する。

# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## 4. Experiments

### Implementation details. 

- We apply the Spectral Norm [30] to all the layers in both the generator and discriminator.
    - **生成器と識別器の全ての層に、Spectral Norm を適用している。**

- The learning rates for the generator and discriminator are set to 0.0001 and 0.0004, respectively [15]. We use the ADAM [21] and set β1 = 0, β2 = 0.999. All the exper- iments are conducted on an NVIDIA DGX1 with 8 V100 GPUs.

- We use synchronized mean and variance computation, i.e., these statistics are collected from all the GPUs.
    - 我々は、同期された平均および分散計算を使用する、すなわち、これらの統計はすべてのＧＰＵから収集される。

### Datasets.

- We conduct experiments on several datasets.

---

- COCO-Stuff [5] is derived from the COCO dataset [26].
    - COCO-Stuff [5]は、COCOデータセット[26]から派生した [derived from] ものです。

- It has 118,000 training images and 5,000 validation images captured from diverse scenes. 
    - 多様なシーンから抽出した 118,000 枚の学習用画像と 5,000 の検証用画像があります。

- It has 182 semantic classes.

- Due to its large diversity, existing image synthesis models perform poorly on this dataset.
    - その多様性が大きいため、既存の画像合成モデルはこのデータセットではうまく機能しません。

---

- ADE20K [48] consists of 20,210 training and 2,000 validation images. 

- Similarly to COCO, the dataset contains challenging scenes with 150 semantic classes.
    - COCOと同様に、データセットには150の意味クラスを持つ難しいシーンが含まれています。

---

- ADE20K-outdoor is a subset of the ADE20K dataset that only contains outdoor scenes, used in Qi et al. [35].
    - ADE20K-outdoorは、屋外 [outdoor] シーンのみを含むADE20Kデータセットのサブセットです。 [35]。

---

- Cityscapes dataset [8] contains street scene images in German cities. 
    - Cityscapesデータセット[8]には、ドイツの都市における街路 [street] シーンの画像が含まれています。

- The training and validation set sizes are 3,000 and 500, respectively. 

- Recent work has achieved photorealistic semantic image synthesis results [35, 39] on the Cityscapes dataset.

---

- Flickr Landscapes. We collect 41,000 photos from Flickr and use 1,000 samples for the validation set.
    - Flickrの風景 Flickrから41,000枚の写真を収集し、検証セットに1,000個のサンプルを使用します。

- Instead of manual annotation, we use a pre-trained DeepLabV2 model [6] to compute the input segmentation masks.

---

- We train the competing semantic image synthesis methods on the same training set and report their results on the same validation set for each dataset.
    - 我々は同じトレーニングセット上で競合する意味的画像合成方法を訓練し、それらの結果を各データセットについて同じ検証セット上で報告する。

### Performance metrics.

- We adopt the evaluation protocol from previous work [7, 40]. 

- Specifically, we run a semantic segmentation model on the synthesized images and compare how well the predicted segmentation mask matches the ground truth input.
    - 具体的には、合成された画像に対してセマンティックセグメンテーションモデルを実行し、予測されたセグメンテーションマスクがグランドトゥルース入力とどの程度うまく一致するかを比較します。

- This is based on the intuition that if the output images are realistic then a well-trained semantic segmentation model should be able to predict the ground truth label.
    - これは、出力画像がリアルなものであれば、十分に学習されたセマンティックセグメンテーションモデルがグランドトゥルースラベルを予測できるはずであるという直感に基づいています。

- For measuring the segmentation accuracy, we use the mean Intersection-over-Union (mIoU) and pixel accuracy (accu) metrics.
    - **セグメンテーション精度を測定するために、mIoU (平均 IoU) 指標、およびピクセル単位での正解率指標を使用する。**

- We use state-of-the-art segmentation networks for each dataset: DeepLabV2 [6, 32] for COCO-Stuff, UperNet101 [42] for ADE20K, and DRN-D-105 [44] for Cityscapes.
    - 我々は、各データセットに対して、SOTA のセグメンテーションネットワークを使用する。
    - 即ち、COCO-Stuff データセットに対しては、DeepLabV2、ADE20K データセットに対しては、UperNet101、Cityscapes データセットに関しては、DRN-D-105

- In addition to segmentation accuracy, we use the Fre ́chet Inception Distance (FID) [15] to measure the distance between the distributions of synthesized results and the distribution of real images.
    - **セグメンテーションの精度に加えて、我々は合成結果の分布と本物画像の分布との間の距離を測定するためにフレシェント開始距離（FID）[15]を使用する。**

### Baselines.

- We compare our method with three leading semantic image synthesis models: the pix2pixHD model [40], the cascaded refinement network model (CRN) [7], and the semi-parametric image synthesis model (SIMS) [35]. 

- pix2pixHD is the current state-of-the-art GAN-based conditional image synthesis framework.

- CRN uses a deep network that repeatedly refines the output from low to high resolution, while the SIMS takes a semi-parametric approach that composites real segments from a training set and refines the boundaries. 
    - CRNは低解像度から高解像度まで繰り返し出力を改善する [refines] ような、深層ネットワークを使用する。
    - 一方で、SIMSは学習用データセットからリアルなセグメントを合成して境界を改善する半パラメトリックアプローチを採用しています。

- Both the CRN and SIMS are mainly trained using image reconstruction loss.
    - CRN と SIMS は。、主に画像 reconstruction loss を使用して学習される。

- For a fair comparison, we train the CRN and pix2pixHD models using the implementations provided by the authors.
    - **公正な比較のために、著者らが提供した実装を使用してCRNとpix2pixHDモデルを学習します。**

- As synthesizing an image using SIMS requires many queries to the training dataset, it is computationally prohibitive for a large dataset such as COCO-stuff and the full ADE20K.
    - SIMS を使用して画像を合成することは、学習データセットに多くのクエリ（＝問い合わせ） [queries] を要求するので、[as]
    - COCO-stuff や full ADE20K のような多くなデータセットに対しては、計算上では禁止 [prohibitive] である。

- Therefore, we use the result images provided by the authors whenever possible.
    - したがって、可能な限り著者らによって提供された結果画像を使用します。

### Quantitative comparisons.

![image](https://user-images.githubusercontent.com/25688193/60750651-8a1d9600-9fe6-11e9-8429-e18c59082f7e.png)

- As shown in Table 1, our method outperforms the current state-of-the-art methods by a large margin in all the datasets.

- For COCO-Stuff, our method achieves a mIoU score of 35.2, which is about 1.5 times better than the previous leading method.

- Our FID is also 2.2 times better than the previous leading method.

- We note that the SIMS model produces a lower FID score but has poor segmentation performances on the Cityscapes dataset.
    - SIMSモデルではFIDスコアが低くなりますが、Cityscapeデータセットではセグメンテーションパフォーマンスが低下します。

- This is because the SIMS synthesizes an image by first stitching image patches from the training dataset.
    - これは、ＳＩＭＳが最初に学習用データセットから画像パッチをつなぎ合わせることによって画像を合成するためである。

- As using the real image patches, the resulting image distribution can better match the distribution of real images.
    - 本物画像パッチを使用しているので、結果として得られる画像分布は本物画像の分布によりよく一致することができる。

- However, because there is no guarantee that a perfect query (e.g., a person in a particular pose) exists in the dataset, it tends to copy objects with mismatched segments.
    - しかしながら、完全な問い合わせ（例えば特定のポーズをとっている人）がデータセット内に存在するという保証はないので、それは不一致セグメントを有するオブジェクトをコピーする傾向がある。

#### Qualitative results.

![image](https://user-images.githubusercontent.com/25688193/60751071-45e0c480-9feb-11e9-9796-1cb9784f29c9.png)

---

- In Figures 5 and 6, we provide a qualitative comparison of the competing methods.

- We find that our method produces results with much better visual quality and fewer artifacts, especially for diverse scenes in the COCO-Stuff and ADE20K dataset.
    - 私たちの方法は、特にCOCO-StuffとADE20Kデータセットの多様なシーンに対して、はるかに良い視覚品質と少ないアーティファクトで結果を生み出すことがわかりました。

- When the training dataset size is small, the SIMS model also renders images with good visual quality.
    - 学習用データセットのサイズが小さい場合、SIMSモデルは画像を優れた視覚品質でレンダリングします。

- However, the depicted content often deviates from the input segmentation mask (e.g., the shape of the swimming pool in the second row of Figure 6).
    - しかしながら、描かれた内容は、入力セグメンテーションマスク（例えば、図６の第２行のプールの形状）から逸脱することが多い。

---

![image](https://user-images.githubusercontent.com/25688193/60751170-4e85ca80-9fec-11e9-9d0d-e627763e8e17.png)

![image](https://user-images.githubusercontent.com/25688193/60751238-54c87680-9fed-11e9-910a-5617e20d694c.png)

---

- In Figures 7 and 8, we show more example results from the Flickr Landscape and COCO-Stuff datasets.

- The proposed method can generate diverse scenes with high image fidelity.
    - 提案した方法は、高い画像忠実度で多様なシーンを生成することができる。

- More results are included in the appendix.

### Human evaluation.

- We use Amazon Mechanical Turk (AMT) to compare the perceived visual fidelity of our method against existing approaches.
    - 私たちはAmazon Mechanical Turk（AMT）を使用して、既存のアプローチに対して、私たちの手法の近く的な視覚的忠実度を比較する。

- Specifically, we give the AMT workers an input segmentation mask and two synthesis outputs from different methods and ask them to choose the output image that looks more like a corresponding image of the segmentation mask.
    - 具体的には、我々は、AMT ワーカーに、１つの入力セマンティックセグメンテーションマスクと、異なる手法からの２つの合成画像を与え、
    - セグメンテーションマスクの画像と対応するような出力画像を選ぶように依頼する。

- The workers are given unlimited time to make the selection.

- For each comparison, we randomly generate 500 questions for each dataset, and each question is answered by 5 different workers.

- For quality control, only workers with a lifetime task approval rate greater than 98% can participate in our evaluation.
    - 品質管理については、生涯のタスク承認率が98％を超える作業員のみが評価に参加できます。

---

![image](https://user-images.githubusercontent.com/25688193/60753505-666e4600-a00e-11e9-94e3-ad94ea3d741b.png)

- Table 2 shows the evaluation results.

- We find that users strongly favor our results on all the datasets, especially on the challenging COCO-Stuff and ADE20K datasets.

- For the Cityscapes, even when all the competing methods achieve high image fidelity, users still prefer our results.

### The effectiveness of SPADE.

- To study the importance of SPADE, we introduce a strong baseline called pix2pixHD++, which combines all the techniques we find useful for enhancing the performance of pix2pixHD except SPADE.
    - SPADE の重要性を研究するために、pix2pixHD++ と呼ばれる強力なベースラインを紹介する。
    - これは、SPADE を除いて、pix2pixHD のパフォーマンスを高めるために有益である我々が見つけた全てのテクニックを組み合わている。

- We also train models that receive segmentation mask input at all the intermediate layers via concatenation (pix2pixHD++ w/ Concat) in the channel direction.
    - 我々はまた、チャンネル方向で連結（pix2pixHD++ w/ concat）経由で、
    - 全ての中間層で、セグメンテーションマスクを入力として受け取るモデルを学習する。

- Finally, the model that combines the strong baseline with SPADE is denoted as pix2pixHD++ w/ SPADE.
    - 最後に、強力なベースラインと SPADE を組み合わせたモデルを、pix2pixHD++ w/ SPADE として示す。

- Additionally, we compare models with different capacity by using a different number of convolutional filters in the generator.
    - 加えて、我々は、生成器において、異なる数の畳み込みフィルターを使用することによって、容量が異なるモデルを比較する。

---

![image](https://user-images.githubusercontent.com/25688193/60753643-f4e3c700-a010-11e9-8c84-ac3e8a78ee08.png)

- Table 3: mIoU scores are boosted when SPADE layers are used, for both the decoder architecture (Figure 4) and encoder-decoder architecture of pix2pixHD++ (our improved baseline over pix2pixHD [40]).
    - 表３：mIoU スコアは、生成器のデコーダーのアーキテクチャにと pix2pixHD++ の encoder-decoder アーキテクチャの両方対して、SPADE の正規化層が使用されるときに、ブーストされる。

- On the other hand, simply concatenating semantic input at every layer fails to do so.
    - 一方で、各層で、セマンティック入力を単純に連結することは、そうしない。

- Moreover, our compact model with smaller depth at all layers outperforms all baselines.
    - さらに、すべての層で深さがより浅い、我々のコンパクトなモデルは、すべてのベースラインよりも優れています。

---

- As shown in Table 3 the architectures with the proposed SPADE consistently outperforms its counterparts, in both the decoder-style architecture described in Figure 4 and more traditional encoder-decoder architecture used in pix2pixHD.
    - 表3に示すように、提案されたSPADEのアーキテクチャは、図4で説明した decoder スタイルのアーキテクチャと、pix2pixHDで使用されているより伝統的な encoder-decoder スタイルのアーキテクチャの両方において、常に該当するものを上回っています。

- We also find that concatenating segmentation masks at all intermediate layers, an intuitive alternative to SPADE to provide semantic signal, does not achieve the same performance as SPADE.
    - 我々はまた、セマンティック信号を提供するためのSPADEの直感的な代替手段である、すべての中間層でのセグメンテーションマスクの結合は、SPADEと同じパフォーマンスを達成しないことを見出した。

- Furthermore, the decoder-style SPADE generator achieves better performance than the strong baselines even when using a smaller number of parameters.
    - 更に言えば、decoder スタイルの SPADE 生成器は、パラメーターの数が少ないにも関わらず、強力なベースラインよりもよいパラメーターを達成する。

### Variations of SPADE generator.

![image](https://user-images.githubusercontent.com/25688193/60753952-09c25980-a015-11e9-8329-6f61de93a7c6.png)

- Table 4: The SPADE generator works with different configurations.
    - 表４：異なる設定での SPADE 生成器の動作。

- We change the input of the generator, the convolutional kernel size acting on the segmentation map, the capacity of the network, and the parameter-free normalization method.
    - 生成器の入力、セグメンテーションマスク画像で動作する畳み込みカーネルサイズ、ネットワークの容量、パラメーターなしの正規化手法を変える。

- The settings used in the paper are boldfaced.
    - この論文で使用されている設定は太字 [太字活字] で示しています。

---

- Table 4 reports the performance of variations of our generator.
    - 表４は、我々の生成器の変種のパフォーマンスを報告している。

- First, we compare two types of the input to the generator: random noise or downsampled segmentation maps.
    - 最初に、生成器の入力の２つのタイプを比較する。
    - 即ち、入力ランダムノイズか、ダウンサンプリングされたセグメンテーションマップ

- We find that both render similar performance, and conclude that the modulation by SPADE alone provides sufficient signal about the input mask.
    - 我々は両者が同様の性能を与えることを見出し、
    - そして SPADE 単独による変調は入力マスクについて十分な信号を提供すると結論づける。

- Second, we vary the type of parameter-free normalization layers before applying the modulation parameters.
    - 次に、変調パラメーター γ, β を適用する前のパラメーターフリーな正規化層をのタイプを変更する。 [vary]

- We observe that SPADE works reliably across different normalization methods.
    - SPADEはさまざまな正規化方法に渡って、確実に機能することを観測する。

- Next, we vary the convolutional kernel size acting on the label map, and find that kernel size of 1x1 hurts performance, likely because it prohibits utilizing the context of the label.
    - 次に、ラベルマップに作用する畳み込みカーネルサイズを変えて、1x1 のカーネルサイズがパフォーマンスを損なうことを見いし、これはおそらくラベルの内容を利用することを禁止するためである。

- Lastly, we modify the capacity of the generator network by changing the number of convolutional filters.
    - 最後に、畳み込みフィルターの数を変えることによって、生成器のネットワークの容量を修正する。

- We present more variations and ablations in the appendix for more detailed investigation.
    - より詳細な調査のために、付録にはより多くの変種と切断があります。


### Multi-modal synthesis.

![image](https://user-images.githubusercontent.com/25688193/60755853-afcf8d00-a030-11e9-945d-d857050cbf68.png)

- > Figure 9: Our model attains multimodal synthesis capability when trained with the image encoder.
    - > 図9：我々のモデルは、画像エンコーダで訓練されたときにマルチモーダル合成能力を達成します [attains]。

- > During deployment, by using different random noise, our model synthesizes outputs with diverse appearances but all having the same semantic layouts depicted in the input mask.

- > For reference, the ground truth image is shown inside the input segmentation mask.

---

- In Figure 9, we show the multimodal image synthesis results on the Flickr Landscape dataset.

- For the same input segmentation mask, we sample different noise inputs to achieve different outputs.

- More results are included in the appendix.

### Semantic manipulation and guided image synthesis

- In Figure 1, we show an application where a user draws different segmentation masks, and our model renders the corresponding landscape images.

- Moreover, our model allows users to choose an external style image to control the global appearances of the output image.

- We achieve it by replacing the input noise with the embedding vector of the style image computed by the image encoder.


# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


## A. Additional Implementation Details

### Learning objective. 

- We use the learning objective function in the pix2pixHD work [40] except that we replace its LS- GAN loss [28] term with the Hinge loss term [25, 30, 45]. We use the same weighting among the loss terms in the ob- jective function as that in the pix2pixHD work.
    - pix2pixHDの研究[40]では、学習目的関数を使用していますが、そのLS-GAN損失[28]の項をヒンジ損失の項[25、30、45]に置き換えています。 目的関数の損失項の間では、pix2pixHDの研究と同じ重み付けを使用します。

- When training the proposed framework with the image encoder for multi-modal synthesis and style-guided image synthesis, we include a KL Divergence loss:
    - マルチモーダル合成およびスタイルガイド画像合成のための画像エンコーダを用いて提案されたフレームワークをトレーニングするとき、我々はKL発散損失を含める：

- where the prior distribution p(z) is a standard Gaussian dis- tribution and the variational distribution q is fully deter- mined by a mean vector and a variance vector [22]. We use the reparamterization trick [22] for back-propagating the gradient from the generator to the image encoder. The weight for the KL Divergence loss is 0.05.
    - ここで、事前分布p（z）は標準ガウス分布であり、変分分布qは平均ベクトルと分散ベクトルによって完全に決定されます[22]。 ジェネレータから画像エンコーダへの勾配の逆伝播には、再パラメータ化の手法[22]を使用します。 KL発散損失の重みは0.05です。


## B. Additional Ablation Study

- Table 5: Additional ablation study results regarding mIoU scores: the table shows that both the perceptual loss and GAN feature matching loss terms are important. Making the discriminator deeper does not lead to a performance boost.
    - 表５：mIoU スコアに関する追加のアブレーション研究の結果：
    - この表は、perceptual loss と feature matching loss の両方の項が重要であることを示している。 識別器を深くしても、パフォーマンスは向上しません。

- The table also shows that the components (Synchro- nized Batch Normalization, Spectral Normalization, TTUR, Hinge loss, and SPADE) used in the proposed method also helps our strong baseline, pix2pixHD++.
    - この表はまた、提案された方法で使用される成分（同期化されたバッチ正規化、スペクトル正規化、TTUR、ヒンジ損失、およびSPADE）も我々の強力なベースラインpix2pixHD ++を助けることを示しています。

---

- Table 5 provides additional ablation study results analyzing the contribution of individual components in the proposed method. We first find that both of the perceptual loss and GAN feature matching loss inherited from the learning objective function of the pix2pixHD [40] are impor- tant. Removing any of them leads to a performance drop. We also find that increasing the depth of the discrimina- tor by inserting one more convolutional layer to the top of the pix2pixHD discriminator does not lead to a performance boost.
    - 表５は、提案された方法における個々の構成要素の寄与を分析する追加のアブレーション研究結果を提供する。 まず、pix2pixHDの学習目的関数[40]から受け継いだ知覚的損失とGAN特徴マッチング損失の両方が重要であることがわかった。 いずれかを削除すると、パフォーマンスが低下します。 また、pix2pixHDディスクリミネーターの上部にもう1つの畳み込みレイヤーを挿入してディスクリミネーターの深さを増やしても、パフォーマンスが向上するわけではないこともわかりました。

- In Table 5, we also analyze the effectiveness of each component used in our strong baseline, the pix2pixHD++ method, derived from the pix2pixHD method. We found that the spectral norm, synchronized batch norm, TTUR [15], and hinge loss all contribute to the perfor- mance boost. However, with adding the SPADE to the strong baseline, the performance further improves. Note that pix2pixHD++ w/o Sync Batch Norm and w/o Spec- tral Norm still differs from pix2pixHD in that it uses the hinge loss, TTUR, a large batch size, and Glorot initializa- tion [11].
    - 表5では、pix2pixHDメソッドから派生した、強力なベースラインであるpix2pixHD ++メソッドで使用される各コンポーネントの有効性も分析しています。 スペクトルノルム、シンクロナイズドバッチノルム、TTUR [15]、ヒンジ損失がすべて性能向上に寄与することがわかりました。 ただし、強力なベースラインにSPADEを追加すると、パフォーマンスはさらに向上します。 同期バッチノルムとスペクトルノルムなしのpix2pixHD ++は、ヒンジ損失、TTUR、大規模バッチサイズ、およびGlorot初期化[11]を使用するという点でpix2pixHDとは異なります。

