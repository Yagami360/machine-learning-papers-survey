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

- Each entry in m denotes the semantic label of a pixel.

- We aim to learn a mapping function that can convert an input segmentation mask m to a photorealistic image.


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


