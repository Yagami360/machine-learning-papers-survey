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
    - 最小二乗損失項[28]をヒンジ損失項[25、30、45]で置き換えることを除いて、pix2pixHDで使用されているものと同じマルチスケール弁別器および損失関数を使用してジェネレータをトレーニングします。

- We test several ResNet-based discrim- inators used in recent unconditional GANs [1, 29, 31] but observe similar results at the cost of a higher GPU mem- ory requirement.
    - 我々は最近の無条件GAN [1、29、31]で使用されているいくつかのResNetベースの識別器をテストしますが、より高いGPUメモリ要件を犠牲にして同様の結果を観察します。

- Adding the SPADE to the discriminator also yields a similar performance.
    - 弁別子にSPADEを追加しても同様のパフォーマンスが得られます。

- For the loss function, we observe that removing any loss term in the pix2pixHD loss function lead to degraded generation results.
    - 損失関数については、pix2pixHD損失関数の損失項を削除すると生成結果が悪化することがわかりました。

### Why does SPADE work better?



# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


