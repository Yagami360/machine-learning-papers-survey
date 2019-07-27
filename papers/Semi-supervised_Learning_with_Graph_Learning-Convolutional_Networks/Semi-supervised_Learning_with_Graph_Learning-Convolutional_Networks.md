# ■ 論文
- 論文タイトル："Semi-supervised Learning with Graph Learning-Convolutional Networks"
- 論文リンク：http://openaccess.thecvf.com/content_CVPR_2019/papers/Jiang_Semi-Supervised_Learning_With_Graph_Learning-Convolutional_Networks_CVPR_2019_paper.pdf
- 論文投稿日付：
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- Graph Convolutional Neural Networks (graph CNNs) have been widely used for graph data representation and semi-supervised learning tasks. However, existing graph CNNs generally use a fixed graph which may not be optimal for semi-supervised learning tasks.
    - グラフ畳込みニューラルネットワーク（グラフＣＮＮ）は、グラフデータ表現および半教師つき学習タスクに広く使用されてきた。 しかしながら、既存のグラフＣＮＮは一般に、半教師付き学習タスクには最適ではないかもしれない固定グラフを使用する。

- In this paper, we propose a novel Graph Learning-Convolutional Network (GLCN) for graph data representation and semi-supervised learning. The aim of GLCN is to learn an optimal graph structure that best serves graph CNNs for semi-supervised learning by integrating both graph learning and graph convolution in a unified network architecture. The main advantage is that in GLCN both given labels and the estimated labels are incorporated and thus can provide useful ‘weakly’ supervised information to refine (or learn) the graph construction and also to facilitate the graph convolution operation for unknown label estimation.
    - **本論文では、グラフデータ表現と半教師つき学習のための新しいグラフ学習 - 畳み込みネットワーク（GLCN）を提案した。 GLCNの目的は、統一ネットワークアーキテクチャにおいてグラフ学習とグラフ畳み込みの両方を統合することによって、半教師つき学習のためにグラフCNNに最も役立つ最適グラフ構造を学習することです。 主な利点は、GLCNでは、与えられたラベルと推定されたラベルの両方が組み込まれているため、グラフ構造を改良（または学習）し、未知ラベル推定のグラフ畳み込み演算を容易にするための有用な「弱い」監視情報を提供できることです。**

- Experimental results on seven benchmarks demonstrate that GLCN significantly outperforms the state-of-the-art traditional fixed structure based graph CNNs.
    - 7つのベンチマークに関する実験結果は、GLCNが最先端の伝統的な固定構造ベースのグラフCNNを大幅に上回ることを実証しています。


# ■ イントロダクション（何をしたいか？）

## 1. Introduction

- Deep neural networks have been widely used in many computer vision and pattern recognition tasks. Recently, many methods have been proposed to generalize the convolution operation on arbitrary graphs to address graph structure data [5, 1, 15, 11, 19, 21]. Overall, these methods can be categorized into spatial convolution and spectral convolution methods [22]. For spatial methods, they generally define graph convolution operation directly by defining an operation on node groups of neighbors. For example, Duvenaud et al [5] propose a convolutional neural network that operates directly on graphs and provide an end-to-end feature learning for graph data. Atwood and Towsley [1] propose Diffusion-Convolutional Neural Networks (DCNNs) by employing a graph diffusion process to incorporate the contextual information of node in graph node classification.
    - ディープニューラルネットワークは、多くのコンピュータビジョンおよびパターン認識タスクにおいて広く使用されてきた。 最近、グラフ構造データを扱うために任意のグラフ上の畳み込み演算を一般化するための多くの方法が提案されている［５、１、１５、１１、１９、２１］。 全体として、これらの方法は空間畳み込み法とスペクトル畳み込み法に分類することができます[22]。 空間的方法の場合、それらは一般に、近隣のノードグループに対する操作を定義することによって直接グラフ畳み込み操作を定義する。 例えば、Duvenaud et al [5]は、グラフ上で直接動作し、グラフデータのためのエンドツーエンドの特徴学習を提供する畳み込みニューラルネットワークを提案しています。 Atwood and Towsley [1]は、グラフ拡散プロセスを使用してノードのコンテキスト情報をグラフノード分類に組み込むことによって、拡散畳み込みニューラルネットワーク（DCNN）を提案します。
    
- Monti et al [15] present mixture model CNNs (MoNet) and provide a unified generalization of CNN architectures on graphs. By designing an attention layer, Velicˇkovic ́ et al [21] present Graph Attention Networks (GAT) for semi- supervised learning. For spectral methods, they generally define graph convolution operation based on spectral representation of graphs. For example, Bruna et al [3] propose to define graph convolution in the Fourier domain based on eigen-decomposition of graph Laplacian matrix. Defferrard et al [4] propose to approximate the spectral filters based on Chebyshev expansion of graph Laplacian to avoid the high computational complexity of eigen-decomposition. Kipf et al [11] propose a more simple Graph Convolutional Network (GCN) for semi-supervised learning.
    - Monti et al [15]は混合モデルCNN（MoNet）を提示し、グラフ上でのCNNアーキテクチャの統一一般化を提供します。 注意層を設計することによって、Velickovicら[21]は、半教師つき学習のためのGraph Attention Networks（GAT）を提示します。 スペクトル法では、一般的にグラフのスペクトル表現に基づいてグラフの畳み込み演算を定義します。 例えば、Bruna et al [3]は、グラフラプラシアン行列の固有分解に基づいてフーリエ領域でグラフ畳み込みを定義することを提案しています。 Defferrardら[4]は、グラフラプラシアンのチェビシェフ展開に基づいてスペクトルフィルタを近似し、固有分解の高い計算量を回避することを提案しています。 Kipf et al [11]は、半教師つき学習のためのより単純なGraph Convolutional Network（GCN）を提案しています。

---

- The above graph CNNs have been widely used for supervised or semi-supervised learning tasks. In this paper, we focus on semi-supervised learning. One important aspect of graph CNNs is the graph structure representation of data. In general, the data we provide to graph CNNs either has a known intrinsic graph structure, such as social networks, or we construct a human established graph for it, such as k-nearest neighbor graph with Gaussian kernel. However, it is difficult to evaluate whether the graphs obtained from domain knowledge (e.g., social network) or established by human are optimal for semi-supervised learning in graph CNNs.
    - 上記のグラフCNNは、教師付きまたは半教師付き学習タスクに広く使用されています。 本稿では、半教師つき学習に焦点を当てます。 グラフCNNの重要な側面の1つは、データのグラフ構造表現です。 一般に、CNNをグラフ化するために提供するデータは、ソーシャルネットワークなどの既知の固有のグラフ構造を持つか、またはGaussianカーネルを使用したk最近傍グラフなどの人間が確立したグラフを作成します。 しかしながら、ドメイン知識（例えばソーシャルネットワーク）から得られたグラフまたは人間によって確立されたグラフがグラフＣＮＮにおける半教師付き学習に最適であるかどうかを評価することは困難である。

- Henaff et al [7] propose to learn a supervised graph with a fully connected network. However, the learned graph is obtained from a separate network which is also not guaranteed to best serve the graph CNNs. Li et al [19] propose optimal graph CNNs, in which the graph is learned adaptively by using a distance metric learning. However, it use an approximate algorithm to estimate graph Laplacian which may lead to weak local optimal solution.
    - Henaff et al [7]は、完全に接続されたネットワークを使って教師付きグラフを学習することを提案しています。 しかしながら、学習されたグラフは、グラフＣＮＮに最もよく役立つことも保証されていない別個のネットワークから得られる。 Ｌｉら［１９］は、グラフが距離計量学習を使用することによって適応的に学習される最適グラフＣＮＮを提案する。 しかしながら、それは、グラフのラプラシアンを推定するために近似アルゴリズムを使用し、それは弱い局所最適解を導くかもしれない。

---

- In this paper, we propose a novel Graph Learning Convolutional Network (GLCN) for semi-supervised learning problem. The main idea of GLCN is to learn an optimal graph representation that best serves graph CNNs for semi- supervised learning by integrating both graph learning and graph convolution simultaneously in a unified network architecture. The main advantages of the proposed GLCN for semi-supervised learning are summarized as follows.
    - 本論文では、半教師つき学習問題のための新しいグラフ学習畳み込みネットワーク（GLCN）を提案した。 GLCNの主なアイデアは、統合ネットワークアーキテクチャでグラフ学習とグラフ畳み込みの両方を同時に統合することによって、半教師つき学習のためにグラフCNNに最も役立つ最適グラフ表現を学習することです。 半教師付き学習のために提案されたGLCNの主な利点は以下のように要約される。

- In GLCN, both given labels and the estimated labels are incorporated and thus can provide useful ‘weakly’ supervised information to refine (or learn) the graph construction and to facilitate the graph convolution operation in graph CNN for unknown label estimation.
    - ＧＬＣＮでは、所与のラベルと推定ラベルの両方が組み込まれているので、未知のラベル推定に対してグラフ構成を改良（または学習）し、グラフＣＮＮにおけるグラフ畳み込み演算を容易にするために有用な「弱い」教師情報を提供することができる。

- GLCNcanbetrainedviaasingleoptimizationmanner, which can thus be implemented simply.
    - GLCNは単一の最適化方法で制限されているため、簡単に実装できます。

---

- To the best of our knowledge, this is the first attempt to build a unified graph learning-convolutional network architecture for semi-supervised learning. Experimental results demonstrate that GLCN outperforms state-of-the-art graph CNNs on semi-supervised learning tasks.
    - 私たちの知る限りでは、これは半教師つき学習のための統一グラフ学習 - 畳み込みネットワークアーキテクチャを構築する最初の試みです。 実験結果は、GLCNが半教師つき学習課題において最先端のグラフCNNよりも優れていることを示しています。

# ■ 結論

## x. Conclusion


# ■ 何をしたか？詳細

## 3. Graph Learning-Convolutional Network

- One core aspect of GCN is the graph representation G(X,A) of data X. In some applications, the graph structure of data are available from domain knowledge, such as chemical molecules, social networks etc. In this case, one can use the existing graph directly for GCN based semi-supervised learning. In many other applications, the graph data are not available. One popular way is to construct a human established graph (e.g k-nearest neighbor graph) [8] for GCN. However, the graphs obtained from domain knowledge or estimated by human are generally independent of GCN (semi-supervised) learning process and thus are not guaranteed to best serve GCN learning. Also, the human established graphs are usually sensitive to the local noise and outliers.
    - ＧＣＮの１つの中心的側面は、データＸのグラフ表現Ｇ（Ｘ、Ａ）である。いくつかの用途では、データのグラフ構造は、化学分子、ソーシャルネットワークなどのドメイン知識から入手可能である。 GCNベースの半教師つき学習のための既存のグラフ他の多くのアプリケーションでは、グラフデータは利用できません。一般的な方法の1つは、GCN用に人間が確立したグラフ（例：k最近傍グラフ）[8]を作成することです。しかしながら、ドメイン知識から得られた、または人間によって推定されたグラフは、一般に、ＧＣＮ（半教師あり）学習プロセスとは無関係であり、したがって、ＧＣＮ学習に最も役​​立つことが保証されていない。また、人間が確立したグラフは通常、ローカルノイズと外れ値に敏感です。

- To overcome these problems, we propose a novel Graph Learning-Convolution Network (GLCN) which integrates graph learning and graph convolution simultaneously in a unified network architecture and thus can learn an adaptive (or optimal) graph representation for GCN learning. In particular, as shown in Figure 1, GLCN contains one graph learning layer, several convolu tion layers and one final perceptron layer. In the following, we explain them in detail.    
    - これらの問題を克服するために、著者らは統一ネットワークアーキテクチャにおいてグラフ学習とグラフ畳込みを同時に統合し、したがってGCN学習のための適応（または最適）グラフ表現を学ぶことができる新しいグラフ学習 - 畳み込みネットワーク（GLCN）を提案する。特に、図1に示すように、GLCNには1つのグラフ学習層、複数の畳み込み層、および1つの最終パーセプトロン層が含まれています。以下では、それらを詳細に説明します。

![image](https://user-images.githubusercontent.com/25688193/61766313-60a09f00-ae1b-11e9-839f-6ed7c40af236.png)

### 3.1. Graph learning architecture

- Given an input X = (x1,x2 ···xn) ∈ Rn×p, we aim to seek a nonnegative function Sij = g(xi, xj ) that represents the pairwise relationship between data xi and xj. We implement g(xi, xj ) via a single-layer neural network, which is parameterized by a weight vector a = (a1, a2, · · · ap)T ∈ Rp×1. Formally, we learn a graph S as
    - 入力X =（x 1、x 2···x n）∈R n×pを考えると、データx iとx jの間のペアワイズ関係を表す非負関数Si j = g（x i、x j）を探すことを目的とする。 重みベクトルa =（a 1、a 2、···ap）T∈R p×1によってパラメータ化される単層ニューラルネットワークを介してg（x i、x j）を実装する。 形式的には、次のようにグラフSを学習します。

![image](https://user-images.githubusercontent.com/25688193/61766585-5501a800-ae1c-11e9-947f-984a19621b95.png)

- where ReLU(·) = max(0,·) is an activation function, which guarantees the nonnegativity of Sij . The role of the above softmax operation on each row of S is to guarantee that the learned graph S can satisfy the following property,
    - ここで、ReLU（・）= max（0、・）は、Sijの非負性を保証する活性化関数です。 Ｓの各行に対する上記のソフトマックス演算の役割は、学習グラフＳが以下の特性を満たすことができることを保証することである。

![image](https://user-images.githubusercontent.com/25688193/61766682-b75aa880-ae1c-11e9-9813-3b7051dea4b6.png)

- We optimize the optimal weight vector a by minimizing the ollowing loss function,
    - 以下の損失関数を最小化することによって、最適な重みベクトルaを最適化します。

![image](https://user-images.githubusercontent.com/25688193/61766723-dbb68500-ae1c-11e9-9a05-1b6d7c1d2b4a.png)

- That is, larger distance ∥xi − xj ∥2 between data point xi and xj encourages a smaller value Sij . The second term is used to control the sparsity of learned graph S because of simplex property of S (Eq.(5)), as discussed in [17].
    - つまり、データ点xiとxjの間の距離∥xi -  xj || 2が大きいほど、小さい値Sijが得られます。 第２項は、Ｓのシンプレックス特性（式（５））のために、学習済みグラフＳのスパース性を制御するために使用される（１７）。

---

- Remark. Minimizing the above loss LGL independently may lead to trivial solution, i.e, a = (0,0···0). We use it as a regularized term in our final loss function, as shown in Eq (15) in §3-2.
    - リマーク。 上記の損失ＬＧＬを独立して最小化することは、些細な解決策、すなわち、ａ ＝（０，０…０）をもたらす可能性がある。 §3-2の式（15）に示すように、これを最終損失関数の中で正規化された項として使用します。

---

- For some problems, when an initial graph A is available, we can incorporate it in our graph learning as
    - いくつかの問題では、初期グラフAが利用可能なとき、それをグラフ学習に組み込むことができます。

![image](https://user-images.githubusercontent.com/25688193/61766945-9b0b3b80-ae1d-11e9-99c9-a5cf011ee9df.png)

- We can also incorporate the information of A by considering a regularized term in the learning loss function as
    - 正規化された項を学習損失関数に考慮することで、Aの情報を次のように組み込むこともできます。

![image](https://user-images.githubusercontent.com/25688193/61767188-66e44a80-ae1e-11e9-9962-8655d9060bbc.png)

---

- On the other hand, when the dimension p of the input data X is large, the above computation of g(xi, xj ) may be less effective due to the long weight vector a needing to be trained. Also, the computation of Euclidean distances ∥xi − xj ∥2 between data pairs in loss function LGL is complex for large dimension p. To solve this problem, we propose to conduct our graph learning in a low- dimensional subspace. We implement this via a single-layer low-dimensional embedding network, parameterized by a projection matrix P ∈ Rp×d , d < p. In particular, we conduct our final graph learning as follows,
    - 一方、入力データＸの次元ｐが大きい場合、長い重みベクトルａを訓練する必要があるため、上記のｇ（ｘ ｉ、ｘ ｊ）の計算はあまり効果的ではない可能性がある。 また、損失関数LGLのデータペア間のユークリッド距離|| xi  -  xj || 2の計算は、大きい次元pに対して複雑です。 この問題を解決するために、低次元部分空間でグラフ学習を行うことを提案する。 我々はこれを射影行列Ｐ∈Ｒｐ×ｄ、ｄ ＜ｐによってパラメータ化された単層低次元埋め込みネットワークを介して実施する。 具体的には、以下のようにして最終的なグラフ学習を行います。

![image](https://user-images.githubusercontent.com/25688193/61767251-a743c880-ae1e-11e9-9b80-cb681de81511.png)

- where A denotes an initial graph. If it is unavailable, we can set Aij = 1 in the above update rule. The loss function becomes
    - ここで、Aは初期グラフを表します。 利用できない場合は、上記の更新規則でAij = 1に設定できます。 損失関数は

![image](https://user-images.githubusercontent.com/25688193/61767334-ea05a080-ae1e-11e9-8f23-123db12fbd40.png)

- The whole architecture of the proposed graph learning network is shown in Figure 2.
    - 提案するグラフ学習ネットワークのアーキテクチャ全体を図2に示します。

![image](https://user-images.githubusercontent.com/25688193/61767362-fa1d8000-ae1e-11e9-9354-e4c1003ad5d6.png)

---

- Remark. The proposed learned graph S has a desired probability property (Eq (5)), i.e , the optimal Sij can be regarded a probability that data xj is connected to xi as a neighboring node. That is, the proposed graph learning (GL) architecture can establish the neighborhood structure of data automatically either based on data feature X only or by further incorporating the prior initial graph A with X. 
    - リマーク。 提案学習グラフＳは、所望の確率特性（式（５））を有する、すなわち、最適なＳｉｊは、データｘｊが隣接ノードとしてｘｉに接続される確率と見なすことができる。 すなわち、提案されたグラフ学習（ＧＬ）アーキテクチャは、データ特徴Ｘのみに基づいて、または先の初期グラフＡにさらにＸを組み込むことによって、自動的にデータの近隣構造を確立することができる。

- The GL architecture indeed provides a kind of nonlinear function S = GGL(X, A; P, a) to predict/compute the neighborhood probabilities between node pairs.
    - ＧＬアーキテクチャは確かに、ノード対間の近隣確率を予測／計算するために一種の非線形関数Ｓ ＝ ＧＧＬ（Ｘ、Ａ； Ｐ、ａ）を提供する。


### 3.2. GLCN architecture

- The proposed graph learning architecture is general and can be incorporated in any graph CNNs. In this paper, we incorporate it into GCN [11] and propose a unified Graph Learning-Convolutional Network (GLCN) for semi- supervised learning problem. Figure 1 shows the overview of GLCN architecture. The aim of GLCN is to learn an optimal graph representation for GCN network and integrate graph learning and convolution simultaneously to boost their respectively performance.
    - 提案されたグラフ学習アーキテクチャは一般的なものであり、任意のグラフＣＮＮに組み込むことができる。 本論文ではこれをGCN [11]に組み込み、半教師つき学習問題のための統一グラフ学習 - 畳み込みネットワーク（GLCN）を提案する。 図1にGLCNアーキテクチャの概要を示します。 GLCNの目的は、GCNネットワークに最適なグラフ表現を学習し、グラフの学習と畳み込みを同時に統合してそれぞれのパフォーマンスを向上させることです。

---

- As shown in Figure 1, GLCN contains one graph learning layer, several graph convolution layers and one final perceptron layer. The graph learning layer aims to provide an optimal adaptive graph representation S for graph convolutional layers. That is, in the convolutional layers, it conducts the layer-wise propagation rule based on the adaptive neighbor graph S returned by graph learning layer, i.e,
    - 図1に示すように、GLCNには1つのグラフ学習層、複数のグラフ畳み込み層、および1つの最終パーセプトロン層が含まれています。 グラフ学習層は、グラフ畳み込み層に対して最適な適応グラフ表現Ｓを提供することを目的としている。 すなわち、畳み込みレイヤでは、それは、グラフ学習レイヤによって返された適応隣接グラフＳに基づいてレイヤごとの伝播規則を実行する。

![image](https://user-images.githubusercontent.com/25688193/61767608-cb53d980-ae1f-11e9-869d-e14df6b6384b.png)

- where k = 0,1···K − 1. Ds = diag(d1,d2,∑···dn) is a diagonal matrix with diagonal element di = nj=1 Sij. W (k) ∈ Rdk ×dk+1 is a layer-specific trainable weight matrix for each convolution layer.
    - ここで、k = 0,1···K  -  1です。Ds = diag（d 1、d 2、Σ···d n）は、対角要素di = n j = 1 Si jの対角行列です。 Ｗ（ｋ）∈Ｒｄｋ×ｄｋ ＋ １は、各畳み込み層に対する層固有の訓練可能な重み行列である。
    
- σ(·) denotes an activation function, such as ReLU(·) = max(0,·), and X(k+1) ∈ R^{n×d_(k+1)} denotes the output of activations in the k-th layer.
    - σ（・）は、ReLU（・）= max（0、・）のような活性化関数を表し、X（k + 1）∈R ^ {n×d_（k + 1）}は、 第k層 の活性化された出力を示す

- Since the learned graph S satisfies j Sij = 1, Sij ≥ 0, thus Eq.(12) can be simplified as
    - 学習グラフSはj Sij = 1を満たすのでSij≧0であり、式（12）は以下のように単純化できる。

![image](https://user-images.githubusercontent.com/25688193/61767764-461cf480-ae20-11e9-9659-0429251bd2a4.png)

---

- For semi-supervised classification task, we define the final perceptron layer as
    - 半教師付き分類タスクでは、最終パーセプトロン層を次のように定義します。

![image](https://user-images.githubusercontent.com/25688193/61768211-b7a97280-ae21-11e9-97b1-b9d68bbf25dd.png)

- where W (K ) ∈ RdK ×c and c denotes the number of classes. The final output Z ∈ Rn×c denotes the label prediction of GLCN network, in which each row Zi denotes the label prediction for the i-th node.
    - ここで、W（K）∈RdK×cであり、cはクラスの数を表します。 **最終出力Ｚ∈Ｒｎ×ｃは、ＧＬＣＮネットワークのラベル予測を表し、各行Ｚｉはｉ番目のノードのラベル予測を表す。**

- The whole network parameters Θ = {P,a,W(0),···W(K)} are jointly trained by minimizing the following loss function as 
    - ネットワーク全体のパラメータΘ= {P、a、W（0）、···W（K）}は、次のような損失関数を最小化することによって共同で学習されます。

![image](https://user-images.githubusercontent.com/25688193/61768272-e889a780-ae21-11e9-90bb-c75ce1c52776.png)

![image](https://user-images.githubusercontent.com/25688193/61767334-ea05a080-ae1e-11e9-8f23-123db12fbd40.png)

![image](https://user-images.githubusercontent.com/25688193/61768413-5afa8780-ae22-11e9-9f57-b4c84e8a2958.png)

- where LGL and LSemi-GCN are defined in Eq (11) and Eq (3), respectively. Parameter λ ≥ 0 is a tradeoff parameter. It is noted that, when λ = 0, the optimal graph S is learned based on labeled data (i.e, cross-entropy loss) only which is also feasible in our GLCN.
    - ここで、LGLとLSemi-GCNは、それぞれ式（11）と式（3）で定義されています。 パラメータλ≧０はトレードオフパラメータである。 λ＝ ０の場合、最適グラフＳはラベル付きデータ（すなわちクロスエントロピー損失）のみに基づいて学習され、これは我々のＧＬＣＮでも実現可能であることに留意されたい。

- Demonstrationandanalysis. Therearetwomainbenefits of the proposed GLCN network:

- In GLCN, both given labels Y and the estimated labels Z are incorporated and thus can provide useful ‘weakly’ supervised information to refine the graph construction S and thus to facilitate the graph convolution operation in GCN for unknown label estimation. That is, the graph learning and semi-supervised learning are conducted jointly in GLCN and thus can boost their respectively performance.
    - ＧＬＣＮでは、所与のラベルＹと推定ラベルＺの両方が組み込まれているので、グラフ構成Ｓを改良し、ひいては未知ラベル推定のためのＧＣＮにおけるグラフ畳み込み演算を容易にするために有用な「弱く」教師あり情報を提供できる。 つまり、グラフ学習と半教師つき学習はGLCNで共同で実施されているため、それぞれのパフォーマンスを向上させることができます。

- GLCN is a unified network which can be trained via a single optimization manner and thus can be implemented simply.
    - ＧＬＣＮは、単一の最適化方法によって訓練することができ、したがって簡単に実施することができる統合ネットワークである。

---

- Figure 3 shows the cross-entropy loss values over labeled node L across different epochs. One can note that, GLCN obtains obviously lower cross-entropy value than GCN at convergence, which clearly demonstrates the higher predictive accuracy of GLCN model. Also, the convergence speed of GLCN is just slightly slower than GCN, indicating the efficiency of GLCN.
    - 図３は、異なるエポックにわたるラベル付きノードＬにわたるクロスエントロピー損失値を示す。 GLCNは、収束時にGCNより明らかに低いクロスエントロピー値を得ていることに気付くことができます。 また、GLCNの収束速度はGCNよりもわずかに遅く、GLCNの効率を示しています。
    
- Figure 4 demonstrates 2D t-SNE [14] visualizations of the feature map output by the first convolutional layer of GCN [11] and GLCN, respectively. Different classes are marked by different colors. One can note that, the data of different classes are distributed more clearly and compactly in our GLCN representation, which demonstrates the desired discriminative ability of GLCN on conducting graph node representation and thus semi-supervised classification tasks.
    - 図4は、それぞれGCN [11]とGLCNの最初のたたみ込みレイヤによって出力された特徴マップの2D t-SNE [14]の可視化を示しています。 異なるクラスは異なる色でマークされています。 異なるクラスのデータは、我々のGLCN表現においてより明確かつコンパクトに分布しており、これはグラフノード表現、したがって半教師付き分類タスクを実行する際のGLCNの望ましい識別能力を実証している。


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. Related Work

![image](https://user-images.githubusercontent.com/25688193/61768618-f3910780-ae22-11e9-8488-cd43bf20acff.png)

