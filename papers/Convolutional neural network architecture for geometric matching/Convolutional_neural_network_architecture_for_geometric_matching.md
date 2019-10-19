# ■ 論文
- 論文タイトル："Convolutional neural network architecture for geometric matching"
- 論文リンク：https://arxiv.org/abs/1703.05593
- 論文投稿日付：
- 被引用数（記事作成時点）：xxx 件
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- We address the problem of determining correspondences between two images in agreement with a geometric model such as an affine or thin-plate spline transformation, and estimating its parameters. The contributions of this work are three-fold.
    - アフィンまたは薄板スプライン変換などの幾何モデルと一致する2つの画像間の対応を決定し、そのパラメーターを推定する問題に対処します。 この作業の貢献は3つあります。

- First, we propose a convolutional neural network architecture for geometric matching. The architecture is based on three main components that mimic the standard steps of feature extraction, matching and simultaneous in-lier detection and model parameter estimation, while being trainable end-to-end.
    - まず、幾何学的マッチングのための畳み込みニューラルネットワークアーキテクチャを提案します。 このアーキテクチャは、「特徴抽出」、「マッチング」、「同時インライア検出」、「モデルパラメーター推定」の標準ステップを模倣する3つの主要コンポーネントに基づいており、トレーニング可能なエンドツーエンドです。

- Second, we demonstrate that the net- work parameters can be trained from synthetically generated imagery without the need for manual annotation and that our matching layer significantly increases generalization capabilities to never seen before images.
    - 第二に、手動で注釈を付ける必要なく、合成で生成された画像からネットワークパラメータをトレーニングできること、およびマッチングレイヤーが画像の前に見られない一般化機能を大幅に向上させることを実証します。

- Finally, we show that the same model can perform both instance-level and category-level matching giving state-of-the-art results on the challenging Proposal Flow dataset.
    - 最後に、同じモデルがインスタンスレベルとカテゴリレベルの両方のマッチングを実行して、課題のあるプロポーザルフローデータセットで最先端の結果を提供できることを示します。

# ■ イントロダクション（何をしたいか？）

## x. Introduction

- 第１パラグラフ

---

- Traditionally, correspondences consistent with a geometric model such as epipolar geometry or planar affine transformation, are computed by detecting and matching local features (such as SIFT [40] or HOG [12, 23]), followed by pruning incorrect matches using local geometric constraints [45, 49] and robust estimation of a global geometric transformation using algorithms such as RANSAC [19] or Hough transform [34, 36, 40].
    - 従来、エピポーラ幾何や平面アフィン変換などの幾何モデルと一致する対応は、ローカルフィーチャを検出および照合し、ローカル幾何制約[45、49]を使用して誤った照合を削除し、アルゴリズムを使用してグローバル幾何変換のロバスト推定によって計算されます RANSAC [19]やHough変換[34、36、40]など。

- This approach works well in many cases but fails in situations that exhibit (i) large changes of depicted appearance due to e.g. intra-class variation [23], or (ii) large changes of scene layout or non-rigid deformations that require complex geometric models with many parameters which are hard to estimate in a manner robust to outliers.
    - このアプローチは多くの場合にうまく機能しますが、（i）描かれた外観の大きな変化を示す状況では失敗します。 クラス内変動[23]、または（ii）外れ値に対してロバストな方法で推定するのが困難な多くのパラメータを持つ複雑な幾何モデルを必要とするシーンレイアウトまたは非剛体変形の大きな変化。

---

- In this work we build on the traditional approach and develop a convolutional neural network (CNN) architecture that mimics the standard matching process. First, we replace the standard local features with powerful trainable convolutional neural network features [33, 48], which allows us to handle large changes of appearance between the matched images. Second, we develop trainable matching and transformation estimation layers that can cope with noisy and incorrect matches in a robust way, mimicking the good practices in feature matching such as the second nearest neighbor test [40], neighborhood consensus [45, 49] and Hough transform-like estimation [34, 36, 40].
    - この作業では、従来のアプローチに基づいて、標準のマッチングプロセスを模倣するたたみ込みニューラルネットワーク（CNN）アーキテクチャを開発します。 まず、標準のローカル機能を強力なトレーニング可能な畳み込みニューラルネットワーク機能[33、48]に置き換えます。これにより、一致した画像間の外観の大きな変化を処理できます。 次に、ロバストな方法でノイズの多い不正確な一致に対処できるトレーニング可能なマッチングおよび変換推定レイヤーを開発し、2番目の最近傍テスト[40]、近隣コンセンサス[45、49]、ハフなどの特徴マッチングの優れた実践を模倣します 変換のような推定[34、36、40]。


# ■ 結論

## x. Conclusion


# ■ 何をしたか？詳細

## 3. Architecture for geometric matching

- xxx

- The classical approach consists of the following stages: (i) local descriptors (e.g. SIFT) are extracted from both input images, (ii) the descriptors are matched across images to form a set of tentative correspondences, which are then used to (iii) robustly estimate the parameters of the geometric model using RANSAC or Hough voting.
    - 古典的なアプローチは、次の段階で構成されます：（i）ローカル記述子（例えば、SIFT）は両方の入力画像から抽出され、（ii）記述子は画像全体で一致して暫定的な対応のセットを形成し、次に（iii）に使用されます RANSACまたはハフ投票を使用して、幾何モデルのパラメーターをロバストに推定します。

---

- Our architecture, illustrated in Fig. 2, mimics this process by: (i) passing input images IA and IB through a siamese architecture consisting of convolutional layers, thus extracting feature maps fA and fB which are analogous to dense local descriptors, (ii) matching the feature maps (“descriptors”) across images into a tentative correspondence map fAB , followed by a (iii) regression network which directly outputs the parameters of the geometric model, θˆ, in a robust manner. The inputs to the network are the two images, and the outputs are the parameters of the chosen geometric model, e.g. a 6-D vector for an affine transformation.
    - 図2に示すアーキテクチャは、このプロセスを模倣します：（i）入力画像IAおよびIBを畳み込み層で構成される「シャムアーキテクチャ」に通過させ、密なローカル記述子に類似する特徴マップfAおよびfBを抽出します（ ii）画像全体の特徴マップ（「記述子」）を暫定的な対応マップfABに一致させ、続いて（iii）幾何学モデルのパラメータθˆをロバストに直接出力する回帰ネットワーク。 ネットワークへの入力は2つの画像であり、出力は選択された幾何モデルのパラメーターです。 アフィン変換用の6次元ベクトル。


### 3.1. Feature extraction

- xxx

# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. Related Work


