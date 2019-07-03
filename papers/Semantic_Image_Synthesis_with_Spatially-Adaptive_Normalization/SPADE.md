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
    - 我々は、spatially- adaptive（空間的に適合して）学習された変換を介して、正規化レイヤにおいて、活性化関数を調整する [modulating] ために入力レイアウトを使用することを提案する。

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
    - **畳み込み層・正規化層。非線形層を積み重ねた伝統的な [conventional] ネットワークが、ベストな準最適解であることを示す。**
    - **それらの正規化層が、入力セマンティックマスクにおいて、情報を洗い流す傾向があるため、**

- To address the issue, we propose spatially-adaptive normalization, a conditional normalization layer that modulates the activations using input semantic layouts through a spatially-adaptive, learned transformation and can effectively propagate the semantic information throughout the network.
    - **この問題に対処するために、我々は、spatially-adaptive normalization を提案する。**
    - **これは、空間的に適応して学習された変換を介して、入力セマンティックレイアウトを使用して活性化関数を調整し [modulates]、ネットワーク全体にセマンティック情報を効果的に伝播 [propagate] できる条件付き正規化レイヤを提案する。**

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

## x. 論文の項目名 (Conclusion)


# ■ 何をしたか？詳細

## 3. Semantic Image Synthesis

- Let m ∈ L^H×W be a semantic segmentation mask where L is a set of integers denoting the semantic labels, and H and W are the image height and width.
    - ｍ∈Ｌ^Ｈ×Ｗ をセマンティックセグメンテーションマスクとする。
    - ここで、Ｌはセマンティックラベルを表す整数の集合であり、ＨおよびＷは画像の高さおよび幅である。

- Each entry in m denotes the semantic label of a pixel.
    - m の各エントリは、ピクセルの意味ラベルを表します。

- We aim to learn a mapping function that can convert an input segmentation mask m to a photorealistic image.
    - **入力セグメンテーションマスクｍを写実的な画像に変換することができる写像関数を学ぶことを目的とする。**

### Spatially-adaptive denormalization

- > Figure 2: In SPADE, the mask is first projected onto an embedding space, and then convolved to produce the modulation parameters γ and β. 
    - > **SPADEでは、マスクはまず埋め込み空間に投影され、次に畳み込まれて調整パラメータγとβが生成されます。**

- > Unlike prior conditional normalization methods, γ and β are not vectors, but tensors with spatial dimensions.
    - > 従来の条件付き正規化法とは異なり、γとβはベクトルではなく空間次元を持つテンソルです。

- > The produced γ and β are multiplied and added to the normalized activation element-wise.
    - > 生成されたγおよびβは乗算され、正規化された活性化要素に要素ごとに加算される。
    
---

- Let h^i denote the activations of the i-th layer of a deep convolutional network given a batch of N samples.

- Let Ci be the number of chan- nels in the layer.

- Let Hi and Wi be the height and width of the activation map in the layer.

- We propose a new conditional normalization method called SPatially-Adaptive (DE)normalization1 (SPADE).
    - SPatially-Adaptive（DE）正規化1（SPADE）と呼ばれる新しい条件付き正規化方法を提案します。

- Similar to Batch Normaliza- tion [19], the activation is normalized in the channel-wise manner, and then modulated with learned scale and bias.
    - バッチ正規化[19]と同様に、活性化はチャネルごとに正規化され、その後学習されたスケールとバイアスで変調されます。 

- Figure 2 illustrates the SPADE design. The activation value at site (n ∈ N,c ∈ Ci,y ∈ Hi,x ∈ Wi) is given by
    - 図2は、SPADEデザインを示しています。 サイトでの活性化値（n∈N、c∈Ci、y∈Hi、x∈Wi）は次式で与えられます。

- where h_{n,c,y,x}^i is the activation at the site before normalization, μ_c^i and σ_c^i are the mean and standard deviation of the activation in channel c:

---

- The variables γci,y,x(m) and βci,y,x(m) in (1) are the learned modulation parameters of the normalization layer.

- In contrast to BatchNorm [19], they depend on the input segmentation mask and vary with respect to the location (y, x).
    - BatchNorm [19]とは対照的に、それらは入力セグメンテーションマスクに依存し、位置（y、x）に関して変化します。

- We use the symbol γci,y,x and βci,y,x to denote the functions that convert the input segmentation mask m to the scaling and bias values at the site (c, y, x) in the i-th activation map.

- We implement the functions γci,y,x and βci,y,x using a simple two-layer convolutional network, whose detail design can be found in the appendix.

---

- In fact, SPADE is related to, and is a generalization of several existing normalization layers.
    - 実際、SPADEはいくつかの既存の正規化層に関連しており、それを一般化したものです。

- First, replacing the segmentation mask m with the image class label and making the modulation parameters spatially-invariant (i.e.,γci,y1,x1 ≡ γci,y2,x2 and βci,y1,x1 ≡ βci,y2,x2 for any y1, y2 ∈ {1,2,...,Hi} and x1,x2 ∈ {1,2,...,Wi}), we arrive at the form of Conditional Batch Normalization layer [10].

- Indeed, for any spatially-invariant conditional data, our method reduces to Conditional BN. Similarly, we can arrive at AdaIN [17] by replacing the segmentation mask with an- other image, making the modulation parameters spatially- invariant and setting N = 1. As the modulation parameters are adaptive to the input segmentation mask, the proposed SPADE is better suited for semantic image synthesis.

# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


