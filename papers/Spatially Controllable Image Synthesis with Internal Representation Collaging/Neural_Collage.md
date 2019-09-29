# ■ 論文
- 論文タイトル："Spatially Controllable Image Synthesis with Internal Representation Collaging"
- 論文リンク： https://arxiv.org/abs/1811.10153
- 論文投稿日付：2018/11/26
- 被引用数（記事作成時点）：xxx 件
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- We present a novel CNN-based image editing strategy that allows the user to change the semantic information of an image over an arbitrary region by manipulating the feature-space representation of the image in a trained GAN model. We will present two variants of our strategy: (1) spatial conditional batch normalization (sCBN), a type of conditional batch normalization with user-specifiable spa- tial weight maps, and (2) feature-blending, a method of directly modifying the intermediate features. Our methods can be used to edit both artificial image and real image, and they both can be used together with any GAN with conditional normalization layers. We will demonstrate the power of our method through experiments on various types of GANs trained on different datasets. Code will be available at https://github.com/pfnet-research/ neural-collage.
    - ユーザーが訓練されたGANモデルで画像の特徴空間表現を操作することにより、任意の領域で画像の意味情報を変更できる新しいCNNベースの画像編集戦略を提示します。 戦略の2つのバリエーションを示します：（1）Spatial conditional batch normalization（sCBN）、ユーザー指定可能な空間ウェイトマップを使用した条件付きバッチ正規化のタイプ、および（2）Feature blending 、中間特徴を直接変更する方法 。 私たちの方法は、人工画像と実画像の両方を編集するために使用でき、両方とも条件付き正規化レイヤーを持つ任意のGANと一緒に使用できます。 異なるデータセットでトレーニングされたさまざまなタイプのGANの実験を通じて、この方法の威力を実証します。 コードはhttps://github.com/pfnet-research/ neuro-collageで入手できます。


# ■ イントロダクション（何をしたいか？）

## 1. Introduction

- xxx

---

- Meanwhile, in creative tasks, for example, it often becomes necessary to transform just small regions of interest in an image. Many methods in practice today use annotation datasets of semantic segmentation. These methods include [16, 39, 31], which can be used to construct a photo-realistic image from a doodle (label map). To author’s best knowledge, however, not much has been done for the unsupervised transformation of images with spatial freedom.
    - 一方、たとえば、創造的なタスクでは、画像内の小さな関心領域のみを変換することが必要になることがよくあります。 今日の実際の多くのメソッドは、セマンティックセグメンテーションの注釈データセットを使用しています。 これらの方法には[16、39、31]が含まれ、Doodle（ラベルマップ）から写実的な画像を作成するために使用できます。 しかし、著者の最善の知識を得るために、空間の自由を備えた画像の教師なし変換については、あまり多くのことは行われていません。

- Recently, GAN dissection [1] explored the semantic relation between the output and the intermediate features and succeeded in using the inferred relation for photo-realistic transformation. 
     - 最近、GAN dissection [1]は、出力と中間の特徴間の意味的関係を調査し、推測された関係を写真のようにリアルな変換に使用することに成功しました。

- In this paper, we present a strategy of image transformation that is strongly inspired by the findings in [1]. Our strategy is to manipulate the intermediate features of the target image in a trained generator network. We present a pair of novel methods based on this strategy—Spatial conditional batch normalization and Feature blending—that apply affine transformations on the intermediate features of the target image in a trained generator model. Our methods allow the user to edit the semantic information of the image in a copy and paste fashion.
    -  この論文では、GAN dissection [1] の発見に強く触発された画像変換の戦略を提示します。
    - 私たちの戦略は、訓練されたジェネレーターネットワークでターゲットイメージの中間機能を操作することです。
    - トレーニングされたジェネレーターモデルのターゲットイメージの中間特徴量にアフィン変換を適用するという、この戦略に基づいた2つの新しい方法（Spatial conditional batch normalization と Feature blending ）を紹介します。
    - このメソッドを使用すると、ユーザーはコピーアンドペースト方式で画像のセマンティック情報を編集できます。

---

- Our Spatial conditional batch normalization is a spatial extension of conditional normalization [5, 33, 12], and it allows the user to blend the semantic information of multiple labels based on the user-specified spatial map of mixing- coefficients (label collaging). With sCBN, we can not only generate an image from a label map but also make local semantic changes to the image like changing the eyes of a husky to eyes of a Pomeranian (Fig 1a). On the other hand, our Feature blending is a method that directly mixes multiple images in the intermediate feature space, and it enables local blending of more intricate features (feature collaging). With this technique, we can make modifications to the image like changing the posture of an animal without providing the model with the explicit definition of the posture (Fig 1b).
    - Spatial条件付きバッチ正規化は、条件付き正規化[5、33、12]の空間拡張であり、ユーザーは混合係数のユーザー指定の空間マップに基づいて複数のラベルの意味情報をブレンドできます（ラベルコラージュ）。 sCBNを使用すると、ラベルマップから画像を生成できるだけでなく、ハスキーの目をポメラニアンの目に変更するなど、画像に局所的な意味の変更を加えることもできます（図1a）。 一方、フィーチャーブレンドは、中間のフィーチャースペースで複数の画像を直接ミックスする方法であり、より複雑なフィーチャーのローカルブレンドを可能にします（フィーチャーコラージュ）。 この手法を使用すると、モデルに姿勢の明示的な定義を提供せずに、動物の姿勢を変更するなど、画像を変更できます（図1b）。

---

> 図

- > Figure 1: Examples of label collaging by our sCBN (a) and feature collaging by our feature blending (b). With our sCBN, the user can change the label of the user-specified parts of an image to the user-specified target labels by specifying an appropriate label map. In each row of (a), the label information of the base image over the red colored region and the green colored region are altered by sCBN. The image at the right bottom corner is the result of changing the label of the red region to yawl and changing the label of the green region to lakeside. The images in the right-most column of the panel (b) are the results of our feature blending method. In each row, the red-framed regions in the reference images are blended into the base image. In the second row, eye features of the left reference and mouth features of the right reference are blended into the base male image. Our methods can be applied to a wide variety of images.
    - > 図1：sCBNによるラベルのコラージュの例（a）および機能のブレンドによる機能のコラージュ（b）。 sCBNを使用すると、ユーザーは適切なラベルマップを指定することで、画像のユーザー指定部分のラベルをユーザー指定のターゲットラベルに変更できます。 （a）の各行では、赤色の領域と緑色の領域の上のベースイメージのラベル情報がsCBNによって変更されます。 右下隅の画像は、赤い領域のラベルをヨールに変更し、緑の領域のラベルを湖sideに変更した結果です。 パネルの右端の列（b）の画像は、機能のブレンド方法の結果です。 各行では、参照画像の赤枠の領域がベース画像にブレンドされます。 2行目では、左の参照の目の特徴と右の参照の口の特徴が基本の男性画像にブレンドされています。 当社の方法は、さまざまな画像に適用できます。

---

- One significant strength common to both our methods is that they only require a trained GAN that is equipped with AdaIN/CBN structure; there is no need to train an additional model. Our methods can be applied to practically any types of images for which there is a well-trained GAN. Both methods can be used together as well to make an even wider variety of semantic manipulation on images. Also, by combining our methods with manifold projection [44], we can manipulate the local semantic information of a real image (Fig 2). Our experiments with the most advanced species of GANs [27, 2, 20] shows that our strategy of “standing on the shoulder of giants” is a sound strategy for the task of unsupervised local semantic transformation.
    - 両方の方法に共通する大きな強みの1つは、AdaIN / CBN構造を備えたトレーニング済みGANのみが必要なことです。 追加のモデルをトレーニングする必要はありません。 私たちの方法は、よく訓練されたGANが存在する実質的にあらゆるタイプの画像に適用できます。 両方の方法を一緒に使用して、画像に対してさらに多様なセマンティック操作を行うこともできます。 また、メソッドを多様体投影[44]と組み合わせることにより、実画像の局所的な意味情報を操作できます（図2）。 GAN [27、2、20]の最新の種を用いた実験は、「巨人の肩の上に立つ」という戦略が、教師なしのローカルセマンティック変換のタスクに適した戦略であることを示しています。

# ■ 結論

## x. Conclusion


# ■ 何をしたか？詳細

## 3. Two Methods of Collaging the Internal Representations

- The central idea common to both our sCBN and Feature blending is to modify the intermediate features of the target image in a trained generator using a user-specifiable masking function. In this section, we will describe the two methods in more formality.

### 3.1. Spatial Conditional Batch Normalization

- As can be inferred from our naming, spatial conditional batch normalization (sCBN) is closely related to conditional batch normalization (CBN) [7, 5], a variant of Batch Normalization that encodes the class-specific semantic information into the parameters of BN. For locally changing the class label of an image, we will apply spatially varying transformations to the parameters of condit ional batch normalization (sCBN) (Fig 3).
    - 命名から推測できるように、空間条件付きバッチ正規化（sCBN）は、クラス固有のセマンティック情報をBNのパラメーターにエンコードするバッチ正規化のバリアントである条件付きバッチ正規化（CBN）[7、5]と密接に関連しています。 画像のクラスラベルをローカルで変更するには、空間的に変化する変換を条件付きバッチ正規化（sCBN）のパラメーターに適用します（図3）。

> 図

- > Figure 3: Schematic comparison of CBN and sCBN layers; CBN layers gradually introduce the class-specific features into the generated image with spatially uniform strength (left). sCBN layers do practically the same thing, except that they apply the class- specific features with user-specified mixing intensities that vary spatially across the image.
    - > 図3：CBN層とsCBN層の概略比較。 CBNレイヤーは、空間的に均一な強度で生成された画像にクラス固有の機能を徐々に導入します（左）。 sCBNレイヤーは、画像全体で空間的に変化するユーザー指定の混合強度を持つクラス固有の機能を適用することを除いて、実質的に同じことを行います。

---

- Given a batch of images sampled from a single class, the conditional batch normalization [7, 5] normalizes the set of intermediate features produced from the batch by a pair of class-specific scale and bias parameters.
    - 単一のクラスからサンプリングされた画像のバッチが与えられると、条件付きバッチ正規化[7、5]は、クラス固有のスケールとバイアスパラメーターのペアによってバッチから生成された一連の中間特徴を正規化します。

- Let Fk,h,w represent the feature of l-th layer at the channel k, the height location h, and the width location w. Given a batch {Fi,k,h,w} of Fk,h,ws generated from a class c, the CBN at layer l normalizes Fi,k,h,w by:

- xxx


- If the user wants to modify an artificial image generated by the generator function G, the user may replace the CBN of G at (a) user-chosen layer(s) with sCBN with (a) user- chosen weight map(s).
    - <font color="Pink">ユーザーがジェネレーター関数Gによって生成された人工画像を変更したい場合、ユーザーは、「（a）ユーザーが選択したレイヤー (s) 」でのGのCBNを、「（a）ユーザーが選択したウェイトマップ(s)」でsCBNに置き換えることができます。</font>

- The region in the feature space that will alter the user-specified region of interest can be inferred with relative ease by observing the downsampling relation in G.
    - Gのダウンサンプリング関係を観察することにより、ユーザー指定の関心領域を変更する機能空間の領域を比較的簡単に推測できます。

- The user can also control the intensity of the feature of the class c at an arbitrary location (h, c) by choosing the value of W_{h,w(c)} (larger the stronger).
    - ユーザーは、W_{h,w(c)} の値を選択することで、任意の位置 (h,c) でクラスcの特徴の強度を制御することもできます（強度が大きいほど）。

- By choosing W to have strong intensities for different classes in different regions, the user can transform multiple disjoint parts of the images into different classes (see figure 1a). 
    - 異なる地域の異なるクラスに対して強い強度を持つようにWを選択することにより、ユーザーは画像の複数のばらばらの部分を異なるクラスに変換できます（図1aを参照）。
    
- As we will show in section 5-3, the choice of the layer(s) in G to apply sCBN have interesting effects on the intensity of the transformation. The figure 3 shows the schematic overview of the mechanism of sCBN. By using manifold projection, sCBN can be applied to real images as well. We will elaborate more on the application of our method to real images in section 4.
    - セクション5-3で示すように、sCBNを適用するGのレイヤーの選択は、変換の強度に興味深い影響を与えます。 図3は、sCBNのメカニズムの概要を示しています。 多様体投影を使用することにより、sCBNを実際の画像にも適用できます。 セクション4では、実際の画像へのメソッドの適用について詳しく説明します。

### 3.2. Spatial Feature Blending

- Our spatial feature blending is a method that can extract the features of a particular part of one image and blend it into another. 
    - 空間的特徴のブレンドは、ある画像の特定の部分の特徴を抽出し、それを別の画像にブレンドできる方法です。

- Suppose that images xi are generated from latent variables zi by a trained generator G, and that F (l) are the feature map of the image xi that can be obtained by applying l layers of G to zi.
    - 画像xiが訓練されたジェネレーターGによって潜在変数ziから生成され、F（l）がGのl層をziに適用することによって得られる画像xiの特徴マップであると仮定します。

- xxx

- U(l) is an optional shift operator that uniformly translate the feature map F (l) to a specified direction, which can be used to move a specific local feature to an arbitrary position.
    - U（l）は、特定のローカルフィーチャを任意の位置に移動するために使用できる、フィーチャマップF（l）を指定された方向に均一に変換するオプションのシフト演算子です。

- As a map that is akin to the class map W (l) in sCBN, the user may choose M(l) in a similar way as in the previous section to spatially control the effect of the blending. Spatial feature blending can also be applied to real images by using the method of manifold projection. The figure 4 is an overview of the feature-blending process in which the goal is to transplant a feature (front facing open mouth) of an image G(z2) to the target image G(z1)(a dog with a closed mouth). All the user has to do in this scenario is to provide a mixing map M that has high intensity on the region that corresponds to the region of the mouth. As we will show in the experiment section, our method is quite robust to the alignment, and the region of mouth in G(z2) and G(z1) needs to be only roughly aligned.
    - sCBNのクラスマップW（l）に似たマップとして、ユーザーは前のセクションと同様の方法でM（l）を選択して、ブレンドの効果を空間的に制御できます。
    - 空間的特徴の混合は、多様体投影の方法を使用することにより、実際の画像に適用することもできます。 図4は、画像G（z2）の特徴（口を開けて正面）をターゲット画像G（z1）（口を閉じた犬）に移植することを目標とする特徴ブレンドプロセスの概要です。 。 このシナリオでユーザーがしなければならないことは、口の領域に対応する領域に高い強度を持つ混合マップMを提供することです。 実験セクションで示すように、この方法はアライメントに対して非常に堅牢であり、G（z2）とG（z1）の口の領域は大まかにアライメントするだけで済みます。


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. Related Work

- SPADE [31] is also a method based on conditional normalization that was developed almost simultaneously with our method, and it can learn a function that maps an arbitrary segmentation map to an appropriate parameter map of the normalization layer that can convert the segmentation map to a photo-realistic image. Naturally, the training of the SPADE-model requires a dataset contains annotated segmentation map. However, it can be a nontrivial task to obtain a generator model that is well trained on a dataset of annotated segmentation maps for the specific image-type of interest, let alone the dataset of annotated segmentation map itself.
    - SPADE [31]も条件付き正規化に基づく方法であり、この方法とほぼ同時に開発されました。また、任意のセグメンテーションマップを正規化レイヤーの適切なパラメーターマップにマッピングする機能を学習できます。写実的な画像。当然、SPADEモデルのトレーニングには、注釈付きのセグメンテーションマップを含むデータセットが必要です。ただし、注釈付きセグメンテーションマップ自体のデータセットはもちろん、関心のある特定の画像タイプの注釈付きセグメンテーションマップのデータセットで十分にトレーニングされたジェネレーターモデルを取得するのは、簡単な作業ではありません。

- Our method makes a remedy by taking the approach of modifying the conditional normalization layers of a trained GAN. Our spatial conditional batch normalization (sCBN) takes a simple strategy of applying position- dependent affine-transformations to the normalization parameters of a trained network, and it can spatially modify the semantic information of the image without the need of training a new network. Unlike the manipulation done in style transfer [12], we can also edit the conditional information at multiple levels in the network and control the effect of the modification. As we will investigate further in the later section, modification to a layer that is closer to the input tends to transform more global features.
    - 私たちの方法は、訓練されたGANの条件付き正規化レイヤーを変更するアプローチをとることにより改善します。空間条件付きバッチ正規化（sCBN）は、位置に依存するアフィン変換を訓練されたネットワークの正規化パラメーターに適用する単純な戦略を取り、新しいネットワークを訓練することなく画像の意味情報を空間的に変更できます。スタイル転送[12]で行われる操作とは異なり、ネットワーク内の複数のレベルで条件付き情報を編集し、変更の効果を制御することもできます。後のセクションでさらに調査するように、入力に近いレイヤーへの変更は、よりグローバルなフィーチャを変換する傾向があります。
