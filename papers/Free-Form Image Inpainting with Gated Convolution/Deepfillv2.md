# ■ 論文
- 論文タイトル："Free-Form Image Inpainting with Gated Convolution"
- 論文リンク：https://arxiv.org/abs/1806.03589
- 論文投稿日付：
- 被引用数（記事作成時点）：xxx 件
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- We present a novel deep learning based image inpainting system to complete images with free-form masks and inputs. The system is based on gated convolutions learned from millions of images without additional labelling efforts. 
    - 自由形式のマスクと入力を使用して画像を完成させる、新しい深層学習ベースの画像修復システムを紹介します。このシステムは、追加のラベリング作業なしで、数百万の画像から学習したゲート畳み込みに基づいています。

- The proposed gated convolution solves the issue of vanilla convolution that treats all input pixels as valid ones, generalizes partial convolution by providing a learnable dynamic feature selection mechanism for each channel at each spatial location across all layers. 
    - 提案されたゲート畳み込みは、すべての入力ピクセルを有効なものとして扱うバニラ畳み込みの問題を解決し、すべてのレイヤーの各空間位置で各チャネルに学習可能な動的特徴選択メカニズムを提供することにより部分畳み込みを一般化します。

- Moreover, as free-form masks may appear anywhere in images with any shapes, global and local GANs designed for a single rectangular mask are not suitable. To this end, we also present a novel GAN loss, named SN-PatchGAN, by applying spectral-normalized discriminators on dense image patches. It is simple in formulation, fast and stable in training.
    - さらに、自由形式のマスクは任意の形状の画像のどこにでも表示される可能性があるため、単一の長方形マスク用に設計されたグローバルおよびローカルGANは適切ではありません。この目的のために、高密度画像パッチにスペクトル正規化弁別器を適用することにより、SN-PatchGANという新しいGAN損失も提示します。処方が簡単で、トレーニングが迅速で安定しています。

- Results on automatic image inpainting and user-guided extension demonstrate that our system generates higher-quality and more flexible results than previous methods. We show that our system helps users quickly remove distracting objects, modify image layouts, clear watermarks, edit faces and interactively create novel objects in images. Furthermore, visualization of learned feature representations reveals the effectiveness of gated convolution and provides an interpretation of how the proposed neural network fills in missing regions. More high-resolution results and video materials are available at http://jiahuiyu.com/deepfill2.
    - 自動画像修復とユーザーガイド拡張の結果は、当社のシステムが以前の方法よりも高品質で柔軟な結果を生成することを示しています。このシステムは、ユーザーが気を散らすオブジェクトをすばやく削除し、画像レイアウトを変更し、透かしをクリアし、顔を編集し、画像内に新しいオブジェクトをインタラクティブに作成するのに役立つことを示します。さらに、学習された特徴表現の視覚化により、ゲート畳み込みの有効性が明らかになり、提案されたニューラルネットワークが欠落領域を埋める方法の解釈が提供されます。より高解像度の結果とビデオ資料は、http：//jiahuiyu.com/deepfill2で入手できます。


# ■ イントロダクション（何をしたいか？）

## 1. Introduction

- xx

---

- However, deep generative models based on vanilla convolutional networks are naturally ill-fitted for image hole-filling because convolutional filters treat all input pixels as same valid ones. For hole- filling, the input images/features are composed of both regions with valid pixels outside holes and invalid or synthesized pixels in masked regions. Vanilla convolutions apply same filters on all valid, invalid and mixed (on hole boundaries) pixels, leading to visual artifacts such as color discrepancy, blurriness and obvious edge responses surrounding holes when tested on free-form masks.
    - ただし、畳み込みフィルタはすべての入力ピクセルを同じ有効なピクセルとして処理するため、バニラ畳み込みネットワークに基づく深い生成モデルは、当然ながら画像の穴埋めには適していません。 穴埋めの場合、入力画像/機能は、穴の外側の有効なピクセルとマスクされた領域の無効なピクセルまたは合成されたピクセルの両方の領域で構成されます。 バニラコンボリューションは、すべての有効、無効、および混合（ホール境界上）ピクセルに同じフィルターを適用し、フリーフォームマスクでテストすると、色の不一致、ぼやけ、穴を囲むエッジ応答などの視覚的なアーティファクトにつながります。

---

- To address this limitation, partial convolution [Liu et al. 2018] is recently proposed where the convolution is masked and re-normalized to be conditioned only on valid pixels.
    - この制限に対処するため、部分畳み込み[Liu et al。 2018]は、有効なピクセルのみで条件付けされるように畳み込みがマスクおよび再正規化される最近提案されました。

- It is then followed by a mask-update step to re-compute new mask layer by layer.
    - 次に、マスク更新ステップが続き、レイヤーごとに新しいマスクを再計算します。

- Partial convolution is essentially a hard-gating single-channel un-learnable layer multiplied to input feature maps. It heuristically categorizes all pixel locations to be either valid or invalid, and multiplies hard-gating values (e.g. ones or zeros) to input images/features. However this assumption has several problems. 
    - 部分畳み込みは、基本的に、入力フィーチャマップに乗算されたハードゲーティングの単一チャネルの学習不可能なレイヤーです。すべてのピクセル位置をヒューリスティックに有効または無効に分類し、ハードゲーティング値（1または0など）を入力画像/機能に乗算します。ただし、この仮定にはいくつかの問題があります。

- First, if we want to extend it to user-guided image inpainting with conditional channels where users provide sparse sketches inside the mask, should these pixel locations be considered as valid or invalid? How to properly update the mask for next layer?
    - まず、ユーザーがマスク内でスパーススケッチを提供する条件付きチャネルを使用したユーザーガイド付き画像修復に拡張する場合、これらのピクセル位置は有効または無効と見なされるべきですか？

- Secondly, for partial convolution the invalid pixels will progressively disappear in deep layers, leaving all gating values to be ones (Figure 3). However, we will show that if we allow the network to learn the optimal gating values by itself, the network assigns different gating values to different locations in different channels based on input masks and sketches, even in deep layers, as shown in visualization results in Figure 3.
    - 次のレイヤーのマスクを適切に更新する方法は？第二に、部分畳み込みの場合、無効なピクセルは深層で徐々に消え、すべてのゲーティング値が1のままになります（図3）。ただし、ネットワークが最適なゲーティング値を単独で学習できるようにする場合、ネットワークは、入力マスクとスケッチに基づいて、異なるレイヤーの異なる位置に異なるゲーティング値を割り当てます。図3

---

- We propose gated convolution that learns a dynamic feature selection mechanism for each channel and each spatial location (e.g. inside or outside masks, RGB or user-input channels) for the task of free-form image inpainting.
    - 自由形式の画像修復のタスクのために、各チャネルおよび各空間位置（たとえば、マスクの内側または外側、RGBまたはユーザー入力チャネル）の動的な特徴選択メカニズムを学習するゲート畳み込みを提案します。

- Specifically we consider the formulation where the input feature is firstly used to compute gating values д = σ (w_g x ) (σ is sigmoid function, w_g is learnable parameter). The final output is a multiplication of learned feature and gating values y = φ(wx) ⊙ g in which φ is any activation function. Gated convolution is easy to implement and performs significantly better when (1) the masks have arbitrary shapes and (2) the inputs are no longer simply RGB channels with a mask but also has conditional inputs like sketches.
    - 具体的には、ゲーティング値g =σ（w_g x）（σはシグモイド関数、w_gは学習可能なパラメーター）を計算するために入力特徴が最初に使用される定式化を検討します。最終的な出力は、学習した特徴とゲーティング値y =φ（wx）⊙gの乗算です。ここで、φは任意の活性化関数です。ゲート畳み込みは実装が簡単で、（1）マスクが任意の形状を持ち、（2）入力がマスク付きの単純なRGBチャンネルではなく、スケッチのような条件付き入力を持つ場合に大幅に改善されます。

- For network architectures, we stack gated convolution to form a simple encoder-decoder network [Yu et al. 2018]. Skip connections with a U-Net [Ronneberger et al. 2015], as adopted in some image inpainting networks [Liu et al. 2018], are not effective for non-narrow masks, mainly because inputs of these skip connections are almost zeros thus cannot propagate detailed color or texture information to decoder. This can be explained by our visualization of learned feature representation of encoder. Our inpainting network also integrates contextual attention module [Yu et al. 2018] within same refinement network to better capture long- range dependencies.
    - ネットワークアーキテクチャの場合、単純なエンコーダ/デコーダネットワークを形成するためにゲート畳み込みをスタックします[Yu et al。 2018]。 U-Netとの接続をスキップ[Ronneberger et al。 2015]、一部の画像修復ネットワークで採用されている[Liu et al。 2018]、主にこれらのスキップ接続の入力はほとんどゼロであるため、詳細な色またはテクスチャ情報をデコーダに伝播できないため、非狭マスクには効果がありません。これは、エンコーダの学習された特徴表現の視覚化によって説明できます。私たちの修復ネットワークは、コンテキスト注意モジュール[Yu et al。 2018]同じ改良ネットワーク内で、長距離の依存関係をより適切にキャプチャします。

---

- Without degradation of performance, we also significantly simplify training objectives into two terms: a pixel-wise reconstruction loss and an adversarial loss. The modification is mainly designed for free-form image inpainting. As the holes may appear anywhere in images with any shapes, global and local GANs [Iizuka et al. 2017] designed for a single rectangular mask are not suitable. Instead, we propose a variant of generative adversarial networks, named SN- PatchGAN, motivated by global and local GANs [Iizuka et al. 2017], MarkovianGANs [Li and Wand 2016], perceptual loss [Johnson et al. 2016] and recent work on spectral-normalized GANs [Miyato et al. 2018]. The discriminator of SN-PatchGAN directly computes hinge loss on each point of the output map with format Rh×w×c, formulating h × w × c number of GANs focusing on different locations and different semantics (represented in different channels) of input image. SN-PatchGAN is simple in formulation, fast and stable for training and produces high-quality inpainting results.
    - また、パフォーマンスを低下させることなく、トレーニング目標をピクセル単位の再構築損失と敵対的損失という2つの用語に大幅に簡素化します。この変更は、主に自由形式の画像修復用に設計されています。穴は、任意の形状の画像内のどこにでも表示される可能性があるため、グローバルおよびローカルGAN [飯塚ら。 2017]単一の長方形マスク用に設計されたものは適切ではありません。代わりに、グローバルおよびローカルGANを動機とするSN-PatchGANという名前の生成的敵対ネットワークのバリアントを提案します[飯塚ら。 2017]、MarkovianGAN [Li and Wand 2016]、知覚損失[Johnson et al。 2016]およびスペクトル正規化GANに関する最近の研究[Miyato et al。 2018]。 SN-PatchGANの弁別器は、Rh×w×c形式で出力マップの各ポイントのヒンジ損失を直接計算し、入力画像の異なる場所と異なるセマンティクス（異なるチャネルで表される）に焦点を当てたh×w×c GANの数を定式化します。 SN-PatchGANは、処方が簡単で、トレーニング用に高速で安定しており、高品質の修復結果を生成します。

---


# ■ 結論

## x. Conclusion


# ■ 何をしたか？詳細

## x. 論文の項目名


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. Related Work


