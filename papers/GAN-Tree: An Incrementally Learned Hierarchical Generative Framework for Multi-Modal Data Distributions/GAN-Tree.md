# ■ 論文
- 論文タイトル："GAN-Tree: An Incrementally Learned Hierarchical Generative Framework for Multi-Modal Data Distributions"
- 論文リンク： https://arxiv.org/abs/1908.03919
- 論文投稿日付：2019/08/11
- 被引用数（記事作成時点）：0 件
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- Despite the remarkable success of generative adversarial networks, their performance seems less impressive for diverse training sets, requiring learning of discontinuous mapping functions. Though multi-mode prior or multi- generator models have been proposed to alleviate this problem, such approaches may fail depending on the empirically chosen initial mode components.
    - 生成的敵対ネットワークの顕著な成功にもかかわらず、それらのパフォーマンスは、多様なトレーニングセットに対してそれほど印象的ではないようであり、不連続な [discontinuous] マッピング機能の学習が必要です。
    - この問題を軽減するために、マルチモード事前モデルまたはマルチジェネレータモデルが提案されていますが、経験的に選択された初期モードコンポーネントによっては、このようなアプローチは失敗する場合があります。

- In contrast to such bottom-up approaches, we present GAN-Tree, which follows a hierarchical divisive strategy to address such discontinuous multi-modal data. Devoid of any assumption on the number of modes, GAN-Tree utilizes a novel mode- splitting algorithm to effectively split the parent mode to semantically cohesive children modes, facilitating unsupervised clustering. Further, it also enables incremental addition of new data modes to an already trained GAN-Tree, by updating only a single branch of the tree structure. As compared to prior approaches, the proposed framework offers a higher degree of flexibility in choosing a large variety of mutually exclusive and exhaustive tree nodes called GAN- Set. Extensive experiments on synthetic and natural image datasets including ImageNet demonstrate the superiority of GAN-Tree against the prior state-of-the-art.
    - このようなボトムアップアプローチとは対照的に、 GAN-Tree は、このような不連続なマルチモーダルデータに対処するための階層的な分裂戦略に従っています。モード数に関する仮定がないため、GAN-Treeは新しいモード分割アルゴリズムを使用して、親モードを意味的に結合した [cohesive] 子モードに効果的に分割し、教師なしクラスタリングを容易にします [facilitating]。さらに、ツリー構造の単一のブランチのみを更新することにより、既にトレーニング済みのGANツリーに新しいデータモードを追加することもできます。従来のアプローチと比較して、提案されたフレームワークは、GAN-Setと呼ばれる相互に排他的で網羅的なツリーノードの多種多様な選択において、より高い柔軟性を提供します。 ImageNetを含む合成および自然画像データセットに関する広範な実験により、GAN-Treeが従来の最新技術よりも優れていることが実証されています。

# ■ イントロダクション（何をしたいか？）

## x. Introduction

- xxx

---

- Despite the success of GAN, the potential of such a framework has certain limitations. GAN is trained to look for the best possible approximate Pg(X) of the target data distribution Pd(X) within the boundaries restricted by the choice of latent variable setting (i.e the dimension of latent embedding and the type of prior distribution) and the computational capacity of the generator network (characterized by its architecture and parameter size). Such a limitation is more prominent in the presence of highly diverse intra- class and inter-class variations, where the given target data spans a highly sparse non-linear manifold. This indicates that the underlying data distribution Pd(X) would constitute multiple, sparsely spread, low-density regions. Considering enough capacity of the generator architecture (Universal Approximation Theorem [19]), GAN guarantees convergence to the true data distribution. However, the validity of the theorem does not hold for mapping functions involving discontinuities (Fig 1), as exhibited by natural image or text datasets. Furthermore, various regularizations [7, 32] imposed in the training objective inevitably restrict the generator to exploit its full computational potential.
    - GANの成功にもかかわらず、そのようなフレームワークの可能性には特定の制限があります。 GANは、潜在変数設定（つまり、潜在埋め込みの次元と事前分布のタイプ）の選択によって制限された境界内で、ターゲットデータ分布Pd（X）の可能な限り最良の近似Pg（X）を探すように訓練されます。ジェネレーターネットワークの計算能力（そのアーキテクチャとパラメーターサイズによって特徴付けられる）。そのような制限は、非常に多様なクラス内およびクラス間変動が存在する場合に顕著になり、特定のターゲットデータが非常にまばらな非線形多様体にまたがる場合があります。これは、基礎となるデータ分布Pd（X）が複数のまばらに広がった低密度領域を構成することを示しています。ジェネレーターアーキテクチャの十分な容量を考慮して（ユニバーサル近似定理[19]）、GANは真のデータ分布への収束を保証します。ただし、自然画像またはテキストデータセットによって示されるように、定理の有効性は不連続性を含むマッピング関数（図1）には当てはまりません。さらに、トレーニングの目的に課されるさまざまな正則化[7、32]は、ジェネレーターがその計算能力を最大限に活用することを必然的に制限します。

---

- A reasonable solution to address the above limitations could be to realize multi-modal prior in place of the single- mode distribution in the general GAN framework. Several recent approaches explored this direction by explicitly enforcing the generator to capture diverse multi-modal target distribution [15, 21]. The prime challenge encountered by such approaches is attributed to the choice of the number of modes to be considered for a given set of fully-unlabelled data samples.
    - 上記の制限に対処するための合理的な解決策は、一般的なGANフレームワークのシングルモード分散の代わりにマルチモーダル事前分布を実現することです。 いくつかの最近のアプローチは、ジェネレータに多様なマルチモーダルターゲット分布をキャプチャするよう明示的に強制することにより、この方向を探りました[15、21]。 このようなアプローチが直面する主な課題は、完全にラベル付けされていないデータサンプルの特定のセットに対して考慮されるモードの数の選択に起因します。

- To better analyze the challenging scenario, let us consider an extreme case, where a very high number of modes is chosen in the beginning without any knowledge of the inherent number of categories present in a dataset.
    - 困難なシナリオをよりよく分析するために、データセットに存在する固有の数のカテゴリを知らずに非常に多くのモードが最初に選択される極端なケースを考えてみましょう。

- In such a case, the corresponding generative model would deliver a higher inception score [4] as a result of dedicated prior modes for individual sub-categories or even sample level hierarchy.
    - このような場合、対応する生成モデルは、個々のサブカテゴリまたはサンプルレベルの階層専用の [dedicated] 事前モードの結果として、より高い ISスコア [4]を提供します。

- This is a clear manifestation of overfitting in generative modeling as such a model would generate reduced or a negligible amount of novel samples as compared to a single-mode GAN.
    - このようなモデルでは、シングルモードGANと比較して、減少または無視できる [negligible] 量の新しいサンプルを生成するため、生成モデリングでの過剰適合の明らかな現れ [manifestation] です。

- Intuitively, the ability to interpolate between two samples in the latent embedding space [38, 30] demonstrates continuity and generalizability of a generative model.
    - 直観的には、潜在的な埋め込み空間の2つのサンプル間を補間する能力[38、30]は、生成モデルの連続性と一般化可能性を示しています。
    
- However, such an interpolation is possible only within a pair of samples belonging to the same mode specifically in the case of multi-modal latent distribution.
    - ただし、このような補間は、特にマルチモードの潜在分布の場合に、同じモードに属するサンプルのペア内でのみ可能です。
    
- It reveals a clear trade-off between the two schools of thoughts, that is, multi-modal latent distribution has the potential to model a better estimate of Pd (X ) as compared to a single-mode counterpart, but at a cost of reduced generalizability depending on the choice of mode selection.
    - これは、2つの考え方の系統間の明確なトレードオフを明らかにします。つまり、マルチモードの潜在的分布は、シングルモードの対応モデルと比較して、Pd（X）のより良い推定値をモデル化する可能性がありますが、削減された汎化能力のコストは、モード選択の選択に依存する。

- This also highlights the inherent trade-off between quality (multi-modal GAN) and diversity (single-mode GAN) of a generative model [29] specifically in the absence of a concrete definition of natural data distribution.
    - これはまた、特に自然データ分布の具体的な定義がない場合の、生成モデルの品質（マルチモーダルGAN）と多様性（シングルモードGAN）[29]の固有のトレードオフを強調しています。

---

- An ideal generative framework addressing the above concerns must have the following important traits:
    - 上記の懸念に対処する理想的な生成フレームワークには、次の重要な特性 [traits] が必要です。

- The framework should allow enough flexibility in the design choice of the number of modes to be considered for the latent variable distribution.
    - フレームワークは、潜在変数の分布について考慮されるモードの数の設計選択において十分な柔軟性を可能にする必要があります。

- Flexibility in generation of novel samples depending on varied preferences of quality versus diversity according to the intended application in focus (such as unsupervised clustering, hierarchical classification, nearest neighbor retrieval, etc).
    - 目的のアプリケーション（教師なしクラスタリング、階層分類、最近傍検索など）に応じて、品質と多様性のさまざまな好みに応じた新規サンプルの生成の柔軟性。

- Flexibility to adapt to a similar but different class of additional data samples introduced later in absence of the initial data samples (incremental learning setting).
    - 初期データサンプルがなくても、後で導入される類似の異なるクラスの追加データサンプルに適応する柔軟性（増分学習設定）。


---

- In this work, we propose a novel generative modeling framework, which is flexible enough to address the quality- diversity trade-off in a given multi-modal data distribution. We introduce GAN-Tree, a hierarchical generative modeling framework consisting of multiple GANs organized in a specific order similar to a binary-tree structure.
    - この作業では、特定のマルチモーダルデータ分布における品質と多様性のトレードオフに対処するのに十分な柔軟性を備えた、新しい生成モデリングフレームワークを提案します。 GANツリーを導入します。これは、バイナリツリー構造に類似した特定の順序で編成された複数のGANで構成される階層型生成モデリングフレームワークです。

- In contrast to the bottom-up approach incorporated by recent multi- modal GAN [35, 21, 15], we follow a top-down hierarchical divisive clustering procedure. First, the root node of the GAN-Tree is trained using a single-mode latent distribution on the full target set aiming maximum level of generalizability. Following this, an unsupervised splitting algorithm is incorporated to cluster the target set samples accessed by the parent node into two different clusters based on the most discriminative semantic feature difference. After obtaining a clear cluster of target samples, a bi-modal generative training procedure is realized to enable the generation of plausible novel samples from the predefined children latent distributions. 
    - 最近のマルチモーダルGAN [35、21、15]に組み込まれているボトムアップアプローチとは対照的に、トップダウンの階層的分割クラスタリング手順に従います。 最初に、GANツリーのルートノードは、最大レベルの一般化可能性を目指して、フルターゲットセットのシングルモード潜在分布を使用してトレーニングされます。 これに続いて、教師なし分割アルゴリズムが組み込まれ、親ノードがアクセスするターゲットセットのサンプルを、最も識別的なセマンティックフィーチャの違いに基づいて2つの異なるクラスターにクラスター化します。 ターゲットサンプルの明確なクラスターを取得した後、事前定義された子の潜在的な分布からもっともらしい新規サンプルの生成を可能にするために、バイモーダル生成トレーニング手順が実現されます。

- To demonstrate the flexibility of GAN-Tree, we define GAN-Set, a set of mutually exclusive and exhaustive tree-nodes which can be utilized together with the corresponding prior distribution to generate samples with the desired level of quality vs diversity. Note that the leaf nodes would realize improved quality with reduced diversity whereas the nodes closer to the root would yield a reciprocal effect.
    - GAN-Treeの柔軟性を実証するために、GAN-Setを定義します。GAN-Setは、相互に排他的かつ網羅的なツリーノードのセットであり、対応する事前分布とともに利用して、望ましい品質と多様性のレベルのサンプルを生成できます。 葉のノードは多様性を減らして品質を向上させるのに対し、ルートに近いノードは相互効果をもたらすことに注意してください。

---

- The hierarchical top-down framework opens up interesting future upgradation possibilities, which is highly challenging to realize in general GAN settings. One of them being incremental GAN-Tree, denoted as iGAN-Tree. It supports incremental generative modeling in a much efficient manner, as only a certain branch of the full GAN-Tree has to be updated to effectively model distribution of a new input set. Additionally, the top-down setup results in an unsupervised clustering of the underlying class-labels as a byproduct, which can be further utilized to develop a classification model with implicit hierarchical categorization.
    - 階層的なトップダウンフレームワークは、将来の興味深いアップグレードの可能性を開きます。これは、一般的なGAN設定で実現するのが非常に困難です。 それらの1つは、増分GANツリーであり、iGANツリーと呼ばれます。 新しい入力セットの分布を効果的にモデル化するために、完全なGANツリーの特定のブランチのみを更新する必要があるため、増分 [incremental] 生成モデリングを非常に効率的な方法でサポートします。 さらに、トップダウン設定により、基礎となるクラスラベルの副産物としての教師なしクラスタリングが行われ、暗黙的な階層分類を使用した分類モデルの開発にさらに活用できます。


# ■ 結論

## x. Conclusion

- GAN-Tree is an effective framework to address natural data distribution without any assumption on the inherent number of modes in the given data. Its hierarchical tree structure gives enough flexibility by providing GAN-Sets of varied quality-vs-diversity trade-off. This also makes GAN- Tree a suitable candidate for incremental generative modeling. Further investigation on the limitations and advantages of such a framework will be explored in the future.
    - GAN-Treeは、特定のデータに固有のモード数を仮定せずに、自然なデータ分散に対処するための効果的なフレームワークです。 その階層ツリー構造は、さまざまな品質と多様性のトレードオフのGANセットを提供することにより、十分な柔軟性を提供します。 これにより、GAN-Treeは増分生成モデリングの適切な候補にもなります。 このようなフレームワークの制限と利点に関するさらなる調査は、今後検討されます。

# ■ 何をしたか？詳細

## 3.1. Formalization of GAN-Tree

- A GAN-Tree is a full binary tree where each node indexed with i, GN(i) (GNode), represents an individual GAN framework. The root node is represented as GN(0) with the corresponding children nodes as GN (1) and GN (2) (see Fig. 2). Here we give a brief overview of a general GAN- Tree framework. Given a set of target samples D = (xi)ni=1 drawn from a true data distribution Pd, the objective is to optimize the parameters of the mapping G : Z → X , such that the distribution of generated samples G(z) ∼ Pg approximates the target distribution Pd upon randomly drawn latent vectors z ∼ Pz. Recent generative approaches [7] propose to simultaneously train an inference mapping, E : X → Z to avoid mode-collapse. In this paper, we have used Adversarially Learned Inference (ALI) [12] framework as the basic GAN formulation for each node of GAN-Tree. However, one can employ any other GAN framework for training the individual GAN-Tree nodes, if it satisfies the specific requirement of having an inference mapping.

#### Root node (GN(0)).

- Assuming D(0) as the set of complete target samples, the root node GN(0) is first trained using a single-mode latent prior distribution z ∼ Pz^(0). As shown in Fig 2; Epre^(0), G^(0) and D^(0) are the encoder, gerator and discriminator network respectively for the root node with index-0; which are trained to generate samples, x ∼ Pg^(0) approximating Pd^(0) . Here, Pd^(0) is the true target distribution whose samples are given as x ∈ D(0). After obtaining the best approximate P_g^(0) , the next objective is to to improve the approximation by considering the multi-modal latent distribution in the succeeding hierarchy of GAN-Tree.
    - D（0）を完全なターゲットサンプルのセットと仮定すると、ルートノードGN（0）は最初に、シングルモードの潜在的事前分布z〜Pz ^（0）を使用してトレーニングされます。 図2に示すように。 Epre ^（0）、G ^（0）、およびD ^（0）は、それぞれインデックス0のルートノードのエンコーダー、生成器、および識別器ネットワークです。 サンプルを生成するようにトレーニングされています。x〜Pg ^（0）はPd ^（0）に近似しています。 ここで、Pd ^（0）は、サンプルがx∈D（0）として与えられる真のターゲット分布です。 最適な近似P_g ^（0）を取得した後、次の目的は、GANツリーの後続の階層におけるマルチモーダル潜在分布を考慮することにより、近似を改善することです。

#### Children nodes (GN(l) and GN(r)).

- Without any choice of the initial number of modes, we plan to split each GN-node into two children nodes (see Fig 2). In a general setting, assuming p as the parent node index with the corresponding two children nodes indexed as l and r, we define l = left(p), r = right(p), p = par(l) and p = par(r) for simplifying further discussions. Considering the example shown in Fig 2, with the parent index p = 0, the indices of left and right child would be l = 1 and r = 2 respectively. 
    - モードの初期数を選択せず​​に、各GNノードを2つの子ノードに分割する予定です（図2を参照）。一般的な設定では、対応する2つの子ノードがlおよびrとしてインデックス付けされた親ノードインデックスとしてpを想定して、l = left（p）、r = right（p）、p = par（l）およびp = parを定義します（r）さらなる議論を簡素化するため。図2に示した例で、親インデックスがp = 0の場合、左と右の子のインデックスはそれぞれl = 1とr = 2になります。

- A novel binary Mode-splitting procedure (Section 3.2) is incorporated, which, without using the label information, effectively exploits the most discriminative semantic difference at the latent Z space to realize a clear binary clustering of the input target samples. We obtain cluster-set D(l) and D(r) by applying Mode-splitting on the parent-set D(p) such that D(p) = D(l) ∪ D(r). Note that, a single encoder E(p) network is shared by both the child nodes GN(l) and GN(r) as it is also utilized as a routing network, which can route a given target sample x from the root-node to one of the leaf-nodes by traversing through different levels of the full GAN-Tree. The bi-modal latent distribution at the output of the common encoder E is defined as z ∼ Pz and z ∼ Pz for the left and right child-node respectively.
    - 新しいバイナリモード分割手順（セクション3.2）が組み込まれています。これは、ラベル情報を使用せずに、潜在的なZ空間で最も識別的な意味の違いを効果的に活用して、入力ターゲットサンプルの明確なバイナリクラスタリングを実現します。 D（p）= D（l）∪D（r）となるように親セットD（p）にモード分割を適用することにより、クラスターセットD（l）およびD（r）を取得します。単一のエンコーダーE（p）ネットワークは、ルートノードから特定のターゲットサンプルxをルーティングできるルーティングネットワークとしても利用されるため、子ノードGN（l）とGN（r）の両方で共有されることに注意してください。完全なGANツリーのさまざまなレベルを走査して、リーフノードの1つに移動します。共通エンコーダーEの出力での双峰性潜在分布は、それぞれ左および右の子ノードに対してz〜Pzおよびz〜Pzとして定義されます。

---

- xxx


#### Node Selection for split and stopping-criteria.

- A natural question then arises of how to decide which node to split first out of all the leaf nodes present at a certain state of GAN-Tree?
- For making this decision, we choose the leaf node which gives minimum mean likelihood over the data samples labeled for it (lines 5-6, Algo 1).
- Also, the stopping criteria on the splitting of GAN-Tree has to be defined carefully to avoid overfitting to the given target data samples. For this, we make use of a robust IRC-based stopping criteria [16] over the embedding space Z, preferred against standard AIC and BIC metrics. However, one may use a fixed number of modes as a stopping criteria and extend the training from that point as and when required.
    - GAN-Treeの特定の状態にあるすべてのリーフノードから最初に分割するノードをどのように決定するかという自然な疑問が生じます。
    - この決定を行うために、ラベル付けされたデータサンプルに対して最小平均尤度を与えるリーフノードを選択します（5〜6行目、アルゴ1）。
    - また、特定のターゲットデータサンプルへの過剰適合を避けるために、GANツリーの分割の停止基準を慎重に定義する必要があります。 このために、標準のAICおよびBICメトリックに対して優先される、埋め込みスペースZに対する堅牢なIRCベースの停止基準[16]を使用します。 ただし、停止基準として固定数のモードを使用し、必要に応じてその時点からトレーニングを延長することができます。

## 3.2. Mode-Split procedure

- The mode-split algorithm is treated as a basis of the top-down divisive clustering idea, which is incorporated to construct the hierarchical GAN-Tree by performing binary split of individual GAN-Tree nodes. The splitting algorithm must be efficient enough to successfully exploit the highly discriminative semantic characteristics in a fully-unsupervised (l) (l) (l) manner. 
    - モード分割アルゴリズムは、個々のGANツリーノードのバイナリ分割を実行することにより、階層型GANツリーを構築するために組み込まれたトップダウン分割クラスタリングアイデアの基礎として扱われます。 分割アルゴリズムは、完全に教師なし（l）（l）（l）の方法で高度に識別的なセマンティック特性をうまく活用するのに十分効率的でなければなりません。

- xxx


## 3.5. Incremental GAN-Tree: iGANTree

- We advance the idea of GAN-Tree to iGAN-Tree, wherein we propose a novel mechanism to extend an already trained GAN-Tree T to also model samples from a set D′ of new data samples. An outline of the entire procedure is provided across Algorithms 3 and 4. To understand the mechanism, we start with the following assumptions from the algorithm. On termination of this procedure over T , we expect to have a single leaf node which solely models the distribution of samples from D′; and other intermediate nodes which are the ancestors of this new node, should model a mixture dis- tribution which also includes samples from D′.
    - GAN-TreeのアイデアをiGAN-Treeに進めます。ここでは、既に学習したGAN-Tree Tを拡張して、新しいデータサンプルのセットD 'からサンプルをモデル化する新しいメカニズムを提案します。 アルゴリズム3と4全体の手順全体の概要を説明します。メカニズムを理解するために、アルゴリズムの次の仮定から始めます。 Tを介したこの手順の終了時に、D 'からのサンプルの分布のみをモデル化する単一のリーフノードがあると予想されます。 また、この新しいノードの祖先である他の中間ノードは、D 'からのサンプルも含む混合分布をモデル化する必要があります。


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. Related Work


