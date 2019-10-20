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

- The first stage of the pipeline is feature extraction, for which we use a standard CNN architecture. A CNN without fully connected layers takes an input image and produces a feature map f ∈ Rh×w×d, which can be interpreted as a h × w dense spatial grid of d-dimensional local descriptors. A similar interpretation has been used previously in instance retrieval [4, 6, 7, 21] demonstrating high discriminative power of CNN-based descriptors. Thus, for feature extraction we use the VGG-16 network [48], cropped at the pool4 layer (before the ReLU unit), followed by perfeature L2-normalization. We use a pre-trained model, originally trained on ImageNet [13] for the task of image classification. As shown in Fig. 2, the feature extraction network is duplicated and arranged in a siamese configuration such that the two input images are passed through two identical networks which share parameters.
    - パイプラインの最初の段階は、標準CNNアーキテクチャを使用する機能抽出です。 完全に接続されたレイヤーのないCNNは、入力画像を取得し、特徴マップf∈Rh×w×dを生成します。これは、d次元ローカル記述子のh×w密空間グリッドとして解釈できます。 同様の解釈が以前にインスタンス検索で使用されており[4、6、7、21]、CNNベースの記述子の高い識別力を示しています。 したがって、特徴の抽出には、pool4層（ReLUユニットの前）でトリミングされたVGG-16ネットワーク[48]を使用し、その後に特徴L2正規化が続きます。 もともと画像分類のタスクのためにImageNet [13]でトレーニングされた事前トレーニングモデルを使用します。 図2に示すように、特徴抽出ネットワークは、2つの入力画像がパラメーターを共有する2つの同一のネットワークを通過するように、シャム構成で複製および配置されます。

### 3.2. Matching network

- The image features produced by the feature extraction networks should be combined into a single tensor as input to the regressor network to estimate the geometric transformation. We first describe the classical approach for generating tentative correspondences, and then present our matching layer which mimics this process.
    - 特徴抽出ネットワークによって生成された画像特徴は、リグレッサーネットワークへの入力として単一のテンソルに結合して、幾何学的変換を推定する必要があります。 最初に、暫定的な対応を生成するための古典的なアプローチを説明し、次にこのプロセスを模倣するマッチングレイヤーを示します。

---

Tentative matches in classical geometry estimation.

- Classical methods start by computing similarities between all pairs of descriptors across the two images. From this point on, the original descriptors are discarded as all the necessary information for geometry estimation is contained in the pairwise descriptor similarities and their spatial locations. Secondly, the pairs are pruned by either thresholding the similarity values, or, more commonly, only keeping the matches which involve the nearest (most similar) neighbors. Furthermore, the second nearest neighbor test [40] prunes the matches further by requiring that the match strength is significantly stronger than the second best match involving the same descriptor, which is very effective at discarding ambiguous matches.
    - 古典的な方法は、2つの画像にわたる記述子のすべてのペア間の類似性を計算することから始まります。 この時点から、ジオメトリ推定に必要なすべての情報がペアワイズ記述子の類似性とそれらの空間位置に含まれているため、元の記述子は破棄されます。 次に、類似値をしきい値処理するか、より一般的には、最も近い（最も類似した）近隣を含む一致のみを保持することによって、ペアを枝刈りします。 さらに、2番目の最近傍テスト[40]は、一致強度が同じ記述子を含む2番目に良い一致よりもかなり強いことを要求することにより、一致をさらに切り取ります。

---

Matching layer. 

- Our matching layer applies a similar procedure. Analogously to the classical approach, only descriptor similarities and their spatial locations should be considered for geometry estimation, and not the original descriptors themselves.
    - マッチングレイヤーは同様の手順を適用します。 従来のアプローチと同様に、ジオメトリの推定では記述子の類似性とその空間位置のみを考慮し、元の記述子自体は考慮しないでください。

---

- To achieve this, we propose to use a correlation layer followed by normalization. Firstly, all pairs of similarities between descriptors are computed in the correlation layer. Secondly, similarity scores are processed and normalized such that ambiguous matches are strongly down-weighted.
    - これを実現するために、正規化が後に続く「correlation layer」の使用を提案します。 まず、記述子間の類似性のすべてのペアが correlation layer で計算されます。 第二に、あいまいな一致が強くダウンウェイトされるように、類似性スコアが処理および正規化されます。

---

- As is done in the classical methods for tentative correspondence estimation, it is important to postprocess the pairwise similarity scores to remove ambiguous matches. To this end, we apply a channel-wise normalization of the correlation map at each spatial location to produce the final tentative correspondence map fAB. The normalization is performed by ReLU, to zero out negative correlations, followed by L2-normalization, which has two desirable effects. First, let us consider the case when descriptor fB correlates well with only a single feature in fA. In this case, the normalization will amplify the score of the match, akin to the nearest neighbor matching in classical geometry estimation. Second, in the case of the descriptor fB matching multiple features in fA due to the existence of clutter or repetitive patterns, matching scores will be down-weighted similarly to the second nearest neighbor test [40]. However, note that both the correlation and the normalization operations are differentiable with respect to the input descriptors, which facilitates backpropagation thus enabling end-to-end learning.
    - 暫定的な対応推定の古典的な方法で行われているように、あいまいな一致を除去するために、ペアワイズ類似性スコアを後処理することが重要です。このため、各空間位置で相関マップのチャネルごとの正規化を適用して、最終的な暫定的な対応マップfABを作成します。正規化は、負の相関をゼロにするためにReLUによって実行され、L2正規化が続きます。これには2つの望ましい効果があります。まず、記述子fBがfAの単一の特徴とのみ相関する場合を考えてみましょう。この場合、正規化により、一致のスコアが増幅されます。これは、従来のジオメトリ推定での最近傍一致に似ています。第二に、乱雑または反復パターンの存在によりfAの複数の特徴に一致する記述子fBの場合、一致スコアは2番目の最近傍検定と同様に重みが低くなります[40]。ただし、相関操作と正規化操作の両方が入力記述子に関して微分可能であることに注意してください。これにより、逆伝播が容易になり、エンドツーエンドの学習が可能になります。

    

# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. Related Work


