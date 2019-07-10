# ■ 論文
- 論文タイトル："Graphonomy: Universal Human Parsing via Graph Transfer Learning"
- 論文リンク：https://arxiv.org/abs/1904.04536
- 論文投稿日付：2019/04/09
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- Prior highly-tuned human parsing models tend to fit towards each dataset in a specific domain or with discrepant label granularity, and can hardly be adapted to other human parsing tasks without extensive re-training.
   - 以前の高度に調整されたヒューマンパースモデル（human parsing models）は、特定のドメイン内の、または矛盾する [discrepant] ラベルの粒度 [granularity] で各データセットに適合する傾向があり、大規模な再トレーニングなしでは他のヒューマンパースタスクにはほとんど適応できません。 

- In this paper, we aim to learn a single universal human parsing model that can tackle all kinds of human parsing needs by unifying label annotations from different domains or at various levels of granularity.
    - 本論文では、異なるドメインからのラベルアノテーションをさまざまなレベルの粒度 [granularity] で統合する [unifying] ことによって、あらゆる種類の人間の解析ニーズに取り組む [tackle] ことの出来るような、単一の汎用 [universal] ヒーマンパースモデル（universal human parsing model）を学習することを目的としています。

- This poses many fundamental learning challenges, e.g. discovering underlying semantic structures among different label granularity, performing proper transfer learning across different image domains, and identifying and utilizing label redundancies across related tasks.    
    - これは多くの基本的な学習課題を引き起こす [poses]。 
    - <font color="Pink">例えば、異なるラベルの粒度の間で根本的な [underlying] セマンティック構造を発見し、異なる画像ドメインにわたって適切な転移学習を実行し、
    - そして関連するタスクにわたってラベルの冗長性（余剰性） [redundancies] を識別し利用する。</font>

---

- To address these challenges, we propose a new universal human parsing agent, named “Graphonomy”, which incorporates hierarchical graph transfer learning upon the conventional parsing network to encode the underlying label semantic structures and propagate relevant semantic information.
    - これらの課題に取り組むために、我々は、「Graphonomy」と呼ばれる新しい汎用 [universal] ヒューマンパースエージェント（universal human parsing agent）を提案する。
    - これは、基礎となる [underlying] ラベルセマンティック構造をエンコードし、関連セマンティック情報を伝播するために、従来の [conventional] 構文パースネットワーク上で、階層 graph transfer を組み込む [incorporates]。

- In particular, Graphonomy first learns and propagates compact high-level graph representation among the labels within one dataset via Intra-Graph Reasoning, and then transfers semantic information across multiple datasets via Inter-Graph Transfer. 
    - 特に、Graphonomy はまず、Intra-Graph Reasoning を介して、1つのデータセット内のラベル間で、コンパクトな高レベルのグラフ表現を学習して伝播し、
    - 次に Inter-Graph Transfer を介して、複数のデータセットにわたってセマンティック情報を転送します。

- Various graph transfer dependencies (e.g., similarity, linguistic knowledge) between different datasets are analyzed and encoded to enhance graph transfer capability.   
    - 異なるデータセット間の様々な graph transfer 依存性（例えば、類似性、言語学的知識）は、graph transfer 能力を向上させるために分析されエンコードされる。

- By distilling universal semantic graph representation to each specific task, Graphonomy is able to predict all levels of parsing labels in one system without piling up the complexity.
    - 汎用のセマンティックグラフ表現をそれぞれの特定のタスクに引き出す [distilling] ことによって、
    - Graphonomy は複雑さを積み重ねる（蓄積する）[piling up] ことなく、1つのシステム内のすべてのレベルの構文解析ラベルを予測することができます。
    
- Experimental results show Graphonomy effectively achieves the state-of-the-art results on three human parsing benchmarks as well as advantageous universal human parsing performance.
    - 実験結果は、Graphonomyが、3 つのヒーマンパースベンチマークと最先端の汎用ヒューマンパースのパフォーマンスを効果的に達成することを示しています。

# ■ イントロダクション（何をしたいか？）

## 1. Introduction

- Human visual systems are capable of accomplishing holistic human understanding at a single glance on a person image, e.g., separating the person from the background, understanding the pose, and recognizing the clothes the person wears.
    - 人間の視覚システムは、人の画像上でひと目見ただけで、全体的な [holistic] 人間の理解を達成することができる。
    - 例えば背景から人を分離し、ポーズを理解し、そしてその人が着ている服を認識することなど、

- Nevertheless, recent research efforts on human understanding have been devoted to developing numerous highly-specific and distinct models for each individual application, e.g. foreground human segmentation task [8, 15], coarse clothes segmentation task [25, 28] and fine-grained human part/clothes parsing task [14, 39].
    - それにもかかわらず、人間の理解に関する最近の研究努力は、個々の応用ごとのために、多数の非常に特殊ではっきりと異なる [distinct] モデルを開発することに向けられてきた。
    - 例えば、前景の人間セグメンテーションタスク[8、15]、粗い衣服セグメンテーションタスク[25、28]、およびきめの細かい人間の部分/衣服解析タスク[14、39]。

- Despite the common underlying human structure and shared intrinsic semantic information (e.g. upper-clothes can be interpreted as coat or shirt), these highly-tuned networks have sacrificed the generalization capability by only fitting towards each dataset domain and discrepant label granularity.
    - 共通の基本的な人間の構造と、共有された固有の [intrinsic] セマンティック情報（例えば、上着はコートやシャツと解釈できる）にもかかわらず、
    - これらの高度に調整されたネットワークは、各データセットのドメインと矛盾する [discrepant] ラベル粒度に適合することによって、汎化能力を犠牲にしました。

- It is difficult to directly adapt the model trained on one dataset to another related task, and thus requires redundant heavy data annotation and extensive computation to train each specific model.
    - あるデータセットで訓練されたモデルを別の関連タスクに直接適合させることは困難であり、
    - したがって、各特定のモデルを訓練するために、冗長な [redundant] 重いデータアノテーションと広範な計算が必要になります。

- To address these realistic challenges and avoid training redundant models for correlated tasks, we make the first attempt to investigate a single universal human parsing agent that tackles human parsing tasks at different coarse to fine-grained levels, as illustrated in Fig. 1.
    - これらの現実的な課題に対処し、関連するタスクのために、冗長なモデルを学習することを避けるために、
    - 我々は、図1に示すような、粗いレベルから細かいレベルで、人間の解析タスクに取り組むような、単一の汎用ヒューマンパースエージェントを調査する最初の試みを行います。

---

![image](https://user-images.githubusercontent.com/25688193/60794193-3193f780-a1a4-11e9-885c-bb681a9245b2.png)

- > Figure 1. With huge different granularity and quantity of semantic labels, human parsing is isolated into multiple level tasks that hinder the model generation capability and data annotation utilization.
    - > 図1：非常に異なる粒度と大量のセマンティックラベルで、人間の構文解析は、モデルの汎化能力とデータアノテーションの利用を妨げる [hinder] ような、複数のレベルのタスクに分離されます。

- > For example, the head region on a dataset is further annotated into several fine-grained concepts on another dataset, such as hat, hair and face.
    - > たとえば、データセットの頭の領域は、帽子、髪の毛、顔などのように、他のデータセットのいくつかの粒度の細かい概念にさらにアノテーションが付けられます。
    
- > However, different semantic parts still have some intrinsic and hierarchical relations (e.g., Head includes the face. Face is next to hair), which can be encoding as intra-graph and inter-graph connections for better information propagation.    
    - > しかしながら、異なるセマンティック部分は依然としていくつかの固有 [intrinsic] かつ階層的な関係を有し（例えば、頭は顔を含む。顔は髪の毛の隣にある）、それはより良い情報伝播のためのグラフ内（intra-graph）接続およびグラフ間（inter-graph）接続として符号化することができる。

- > To alleviate the label discrepancy issue and take advantage of their semantic correlations, we introduce a universal human parsing agent, named as “Graphonomy”, which models the global semantic coherency in multiple domains via graph transfer learning to achieve multiple levels of human parsing tasks.
    - > ラベルの不一致（食い違い、矛盾） [discrepancy] の問題を軽減 [alleviate] し、それらのセマンティックな（＝意味的な）相関関係を利用するために、我々は「Graphonomy」と名付けられた汎用的なヒューマンパースエージェントを導入する。
    - > 「Graphonomy」は、graph transfer learning を介して、複数のドメインにおける、汎用的なセマンティックの一貫性をモデル化し、複数レベルの人間の解析タスクを達成します。

---

- The most straightforward solution to universal human parsing would be posing it as a multi-task learning problem, and integrating multiple segmentation branches upon one shared backbone network [2, 14, 22, 25, 28].
    - 汎用的なヒューマンパースに対する最も直接的な解決策は、それをマルチタスク学習問題と見なし、複数のセグメンテーションブランチを1つの共有バックボーンネットワークに統合することです[2、14、22、25、28]。

- This line of research only considers the brute-force feature-level information sharing while disregarding the underlying common semantic knowledge, such as label hierarchy, label visual similarity, and linguistic/context correlations.
    - この一連の研究では、ラベル階層、ラベルの視覚的類似性、言語的/文脈的相関などの基本的な意味論的知識を無視しながら、力ずくな[brute-force] 特徴レベルの情報共有のみを考慮しています。

- More recently, some techniques are explored to capture the human structure information by resorting to complex graphical models (e.g., Conditional Random Fields (CRFs)) [2], self-supervised loss [14] or human pose priors [9, 12, 23].
    - ごく最近では、複雑なグラフィカルモデル（例えば条件付き確率場（CRF））[2]、自己管理損失[14]、または人間のポーズ前[9]、[12]、[23]に頼ることによって人間の構造情報を捉えるための技術が探求されます。 ]。

- However, they did not explicitly model the semantic corre- lations of different body parts and clothing accessories, and still show unsatisfactory results for rare fine-grained labels.
    - しかしながら、彼らは異なる身体の部分と衣服の付属品の意味的な相関関係を明確にモデル化しておらず、それでもまれなきめの細かいラベルに対して満足のいく結果を示していません。
    
---

- One key factor of designing a universal human parsing agent is to have proper transfer learning and knowledge integration among different human parsing tasks, as the label discrepancy across different datasets [6, 13, 14, 39] largely hinders direct data and model unification.
    - 異なるデータセット間でのラベルの不一致[6]、[13]、[14]、[39]が直接データとモデルの統一 [unification] を妨げるために、
    - 汎用的ヒューマンパースエージェントを設計する際の1つの重要な要素は、異なるヒューマンパーシングタスク間で適切な transfer learning と知識統合を持つことです。

- In this paper, we achieve this goal by explicitly incorporating human knowledge and label taxonomy into intermediate graph representation learning beyond local convolutions, called “Graphonomy” (graph taxonomy).
    - **本論文では、この目標を達成するために、「Graphonomy」と呼ばれる、局所的な畳み込みを超えた中間グラフ表現学習に、人間の知識とラベル分類法 [taxonomy] を明示的に取り入れます。**

- Our Graphonomy learns the global and common semantic coherency in multiple domains via graph transfer learning to solve multiple levels of human parsing tasks and enforce their mutual benefits upon each other.
    - **我々の Graphonomy は、複数レベルのヒューマンパースタスクを解決し、互いの利益を強化するために、**
    -- **graph transfer learning を介して、複数のドメインにおけるグローバルで共通のセマンティック一貫性を学習する。**

---

- Taking advantage of geometric deep learning [19, 20], our Graphonomy simply integrates two cooperative modules for graph transfer learning.
    - 幾何学的深層学習[19、20]を利用して、我々の Graphonomy は単純に graph transfer learning のための2つの協調モジュールを統合しています。

- First, we introduce Intra-Graph Reasoning to progressively refine graph representations within the same graph structure, in which each graph node is responsible for segmenting out regions of one semantic part in a dataset.
    - **まず、同じグラフ構造内でグラフ表現を段階的に改良するために、Intra-Graph Reasoning（グラフ内推論）を導入する。**
    - **（この）各グラフノードは、データセット内の1つのセマンティック部分の領域をセグメント化する責任がある。**

- Specifically, we first project the extracted image features into a graph, where pixels with similar features are assigned to the same semantic vertex.
    - 具体的には、我々は最初に抽出された画像特徴量をグラフに写像する。
    - ここで類似の特徴を有するピクセルは、同じ意味的頂点に割り当てられる。

- We elaborately design the adjacency matrix to encode the semantic relations, constrained by the connection of human body structure, as shown in Fig. 3.
    - 我々は、図3に示すように、人体構造の接続によって制約される、意味関係を符号化するために、隣接行列を念入りに [elaborately] 設計します。

- After the message propagation via graph convolutions, the updated vertexes are reprojected to make the visual feature maps more discrimina- tive for pixel-level classification.
    - グラフ畳み込みによるメッセージ伝播の後に、
    - ピクセルレベルの分類に対してより区別しやすくなるような視覚的特徴マップを作成するために、更新された頂点が投影される。

---

![image](https://user-images.githubusercontent.com/25688193/60809712-43869200-a1c6-11e9-8890-20e2b43b67f8.png)

- > Figure 3. Examples of the definite connections between each two human body parts, which is the foundation to encode the relations between two semantic nodes in the graph for reasoning.
    - > 図3：各2つの人体部分間の明確な接続の例。これは、推論のためにグラフ内の2つのセマンティックノード間の関係をエンコードするための基礎です。

- > Two nodes are defined related if they are connected by a white line.
    - > 2つのノードが白い線で接続されている場合、それらは関連して定義されます。

---

- Additionally, we build an Inter-Graph Transfer module to attentively distill related semantics from the graph in one domain/task to the one in another domain, which bridges the semantic labels from different datasets, and effectively utilize the annotations at multiple levels.
    - **さらに、あるドメイン/タスク内のグラフから、別のドメイン内のグラフに関連するセマンティクスを注意深く [attentively] 抽出する [distill] ために、Inter-Graph Transfer モジュールを構築する。**
    - **（この Inter-Graph Transfer モジュールは、）異なるデータセットのセマンティックラベルを橋渡し、複数レベルでアノテーションを効果的に活用する。**

- To enhance graph transfer capability, we make the first effort to exploit various graph transfer dependencies among different datasets.
    - graph transfer 能力を強化するために、異なるデータセット間でさまざまなグラフ転送の依存関係を利用する最初の努力をします。

- We encode the relationships between two semantic vertexes from different graphs by computing their feature similarity as well as the semantic similarity encapsulated with linguistic knowledge.
    - 言語的知識でカプセル化されたセマンティック類似性と同様に、
    - それらの特徴類似性を計算することによって、異なるグラフからの2つのセマンティック頂点間の関係を符号化する。

---

- We conduct experiments on three human parsing benchmarks that contain diverse semantic body parts and clothes.
    - 私たちは、多様な意味のある身体部分と服を含む3つの人間の構文解析ベンチマークで実験を行います。

- The experimental results show that by seamlessly propagating information via Intra-Graph Reasoning and Inter-Graph Transfer, our Graphonomy is able to associate and distill high-level semantic graph representation constructed from different datasets, which effectively improves multiple levels of human parsing tasks.
    - 実験結果は、グラフ内推論およびグラフ間転送を介して情報をシームレスに伝播することによって、我々のGraphonomyが異なるデータセットから構築された高水準意味グラフ表現を関連付け、蒸留することができることを示している。

---

- Our contributions are summarized in the following aspects. 
    - 私たちの貢献は以下の側面に要約されています。

- 1) We make the first attempts to tackle all levels of human parsing tasks using a single universal model.    
    - 1）私たちは、単一の汎用的なモデルを使って、あらゆるレベルのヒューマンパースタスクに取り組む最初の試みを行います。

- In particular, we introduce Graphonomy, a new Universal Human Parsing agent that incorporates hierarchical graph transfer learning upon the conventional parsing network to predict all labels in one system without piling up the complexity.
    - 特に、複雑さを積み重ねずに1つのシステム内のすべてのラベルを予測するために、従来の構文解析ネットワークに階層型グラフ転送学習を組み込んだ新しいUniversal Human ParsingエージェントであるGraphonomyを紹介します。

- 2) We explore various graph transfer dependencies to enrich graph transfer capability, which enables our Graphonomy to distill universal semantic graph representation and enhance individualized representation for each label graph.
    - 2）グラフ転送能力を高めるためにさまざまなグラフ転送の依存関係を調査します。これにより、私たちのGraphonomyは普遍的な意味グラフ表現を抽出し、各ラベルグラフの個別表現を強化することができます。

- 3) We demonstrate the effectiveness of Graphonomy on universal human parsing, showing that it achieves the state-of-the-art results on three human parsing datasets.
    - 3）我々は、一般的な人間の構文解析に対するGraphonomyの有効性を実証し、それが3つの人間の構文解析データセットに関して最先端の結果を達成することを示している。


# ■ 結論

## 5. Conclusion

- In this work, we move forward to resolve all levels of human parsing tasks using a universal model to alleviate the label discrepancy and utilize the data annotation.
    - この研究では、ラベルの食い違いを軽減し、データ注釈を利用するための普遍的モデルを使用して、人間のあらゆるレベルの解析タスクを解決するために前進します。

- We proposed a new universal human parsing agent, named as Graphonomy, that incorporates hierarchical graph transfer learning upon the conventional parsing network to predict all labels in one system without piling up the complexity. 
    - 我々は、Graphonomyという名前の新しいユニバーサルヒューマン解析エージェントを提案した。これは従来の解析ネットワーク上で階層的グラフ転送学習を組み込んで、複雑さを積み重ねずに1つのシステム内のすべてのラベルを予測する。

- The solid and consistent human parsing improvements of our Graphonomy on all datasets demonstrates the superiority of our proposed method.
    - すべてのデータセットに対するGraphonomyの堅実で一貫した人間解析の改善は、提案された方法の優位性を示しています。

- The advantageous universal human parsing performance further confirms that our Graphonomy is strong enough to unify all kinds of label annotations from different resources and tackle different levels of human parsing needs.
    - さらに、人間の一般的な解析性能が優れていることから、私たちのGraphonomyは、さまざまなリソースからのあらゆる種類のラベル注釈を統合し、さまざまなレベルの人間の解析ニーズに対応するのに十分強力です。

- In future, we plan to generalize Graphonomy to more general semantic segmentation tasks and investigate how to embed more complex semantic relationships naturally into the network design.
    - 将来的には、Graphonomyをより一般的なセマンティックセグメンテーションタスクに一般化し、より複雑なセマンティック関係をネットワーク設計に自然に組み込む方法を調査する予定です。


# ■ 何をしたか？詳細

## 3. Graphonomy

- In order to unify all kinds of label annotations from different resources and tackle different levels of human parsing needs in one system, we aim at explicitly incorporating hierarchical graph transfer learning upon the conventional parsing network to compose a universal human parsing model, named as Graphonomy.
    - さまざまなリソースからのあらゆる種類のラベル注釈を1つのシステムに統合し、さまざまなレベルの人間の解析ニーズに取り組むために、階層型グラフ転送学習を従来の解析ネットワークに明示的に組み込んでGraphonomyという名前の普遍的な人間解析モデルを構成することを目指します。

- Fig. 2 gives an overview of our proposed framework. Our approach can be embedded in any modern human parsing system by enhancing its original image features via graph transfer learning.
    - 図2に提案したフレームワークの概要を示します。 我々のアプローチは、グラフ転送学習を介してそのオリジナルの画像特徴を向上させることによって現代の人間の解析システムに埋め込むことができる。

- We first learn and propagate compact high-level semantic graph representation within one dataset via Intra-Graph Reasoning, and then transfer and fuse the semantic information across multiple datasets via Inter-Graph Transfer driven by explicit hierarchical semantic label structures.  
    - **まず、グラフ内推論を介して1つのデータセット内でコンパクトな高レベルの意味グラフ表現を学習して伝播し、次に明示的な階層的意味ラベル構造によって駆動されるグラフ間転送を介して複数のデータセットにわたって意味情報を転送および融合します。**

---

![image](https://user-images.githubusercontent.com/25688193/60794261-56886a80-a1a4-11e9-9a1e-154370f2b148.png)

- > Figure 2. Illustration of our Graphonomy that tackles universal human parsing via graph transfer learning to achieve multiple levels of human parsing tasks and better annotation utilization.
    - > 図2.グラフ転送学習による普遍的な人間の構文解析に取り組む私たちのGraphonomyの図。人間の構文解析のタスクの複数のレベルとより良い注釈の利用を達成する。

- > The image features extracted by deep convolutional networks are projected into a high-level graph representation with semantic nodes and edges defined according to the body structure.
    - > ディープコンボリューションネットワークによって抽出された画像特徴は、体構造に従って定義された意味論的ノードおよびエッジを有する高水準グラフ表現に投影される。

- > The global information is propagated via Intra-Graph Reasoning and re-projected to enhance the discriminability of visual features.
    - > 大域的情報は、グラフ内推論によって伝播され、視覚的特徴の識別可能性を高めるために再投影される。

- > Further, we transfer and fuse the semantic graph representations via Inter-Graph Transfer driven by hierarchical label correlation to alleviate the label discrepancy across different datasets.
    - > さらに、階層的ラベル相関によって駆動されるグラフ間転送を介して意味グラフ表現を転送し融合させて、異なるデータセットにわたるラベルの不一致を軽減する。

- > During training, our Graphonomy takes advantage of annotated data with different granularity. For inference, our universal human parsing agent generates different levels of human parsing results taking an arbitrary image as input.
    - > トレーニング中に、私たちのGraphonomyは異なる粒度で注釈付きデータを利用します。 推論のために、私たちの普遍的な人間の解析エージェントは入力として任意の画像を取って異なるレベルの人間の解析結果を生成します。

### 3.1. Intra-Graph Reasoning

- Given local feature tensors from convolution layers, we introduce Intra-Graph Reasoning to enhance local features, by leveraging global graph reasoning with external structured knowledge.
    - 畳み込みレイヤからの局所特徴テンソルを与えられて、我々は外部構造化知識による大域グラフ推論を活用することによって局所特徴を強化するために Intra-Graph Reasoning（内部グラフ推論）を導入する。

- To construct the graph, we first summarize the extracted image features into high-level representations of graph nodes.
    - グラフを構築するために、我々は最初に抽出された画像特徴をグラフノードの高レベル表現に要約する。

- The visual features that are correlated to a specific semantic part (e.g., face) are aggregated to depict the characteristic of its corresponding graph node.
    - 特定の意味部分（例えば、顔）に相関する視覚的特徴は、その対応するグラフノードの特徴を表すために集められる。

---

- Firstly, We define an undirected graph as G = (V,E) where V denotes the vertices, E denotes the edges, and N = |V|. 
    - まず、無向グラフを G =（V、E）と定義します。
    - ここで、Vは頂点、Eは辺（エッジ）、N = |V|を表します。

- Formally, we use the feature maps X ∈ R^{H×W×C} as the module inputs, where H, W and C are height, width and channel number of the feature maps.

- We first produce high-level graph representation Z ∈ R^{N×D} of all N vertices, where D is the desired feature dimension for each v ∈ V , and the number of nodes N typically corresponds to the number of target part labels of a dataset.
    - 最初に、すべてのN個の頂点の高レベルグラフ表現 Z∈R^{N×D} を生成します。
    - ここで、Dは各v∈Vに必要な特徴次元であり、ノード数Nは通常ターゲット１つのデータセットの部分ラベルの数に対応します。

- Thus, the projection can be formulated as the function φ:

![image](https://user-images.githubusercontent.com/25688193/60801266-b1748e80-a1b1-11e9-8208-7a0c2e499048.png)

- where W is the trainable transformation matrix for converting each image feature xi ∈ X into the dimension D.
    - ここで、Ｗは各画像特徴ｘ ｉ∈Ｘを次元Ｄに変換するための訓練可能な変換行列である。

---

- Based on the high-level graph feature Z, we leverage semantic constraints from the human body structured knowledge to evolve global representations by graph reasoning.
    - 高水準グラフ特徴Zに基づいて、我々は人体構造化知識からの意味論的制約を利用してグラフ推論によるグローバル表現を進化させる。

- We introduce the connections between the human body parts to encode the relationship between two nodes, as shown in Fig 3.
    - 図3に示すように、2つのノード間の関係をエンコードするために、人体部分間の接続を紹介します。

- For example, hair usually appears with the face so these two nodes are linked. While the hat node and the leg node are disconnected because they have nothing related.
    - たとえば、通常、髪は顔と一緒に表示されるので、これら2つのノードはリンクされています。 一方、ハットノードとレッグノードは関連していないので切断されています。

---

- Following Graph Convolution [19], we perform graph propagation over representations Z of all part nodes with matrix multiplication, resulting in the evolved features Ze:
    - グラフ畳み込み[19]に従って、行列乗算を使用してすべての部分ノードの表現Z上でグラフ伝播を実行します。その結果、進化した特徴Zeが得られます。

![image](https://user-images.githubusercontent.com/25688193/60801297-c0f3d780-a1b1-11e9-99da-cb7e02032172.png)

- where W e ∈ RD×D is a trainable weight matrix and σ is a nonlinear function.
    - ここで、W e∈RD×Dはトレーニング可能な重み行列、σは非線形関数です。
    
- The node adjacency weight av→v′ ∈ Ae is defined according the edge connections in (v,v′) ∈ E, which is a normalized symmetric adjacency matrix. 
    - ノード隣接ウェイトav→v '∈Aeは、正規化された対称隣接行列である（v、v'）∈Eのエッジ接続に従って定義される。
    
- To sufficiently propagate the global information, we employ such graph convolution multiple times (3 times in practice).
    - グローバル情報を十分に伝播するために、我々はそのようなグラフ畳み込みを複数回（実際には３回）用いる。

---

- Finally, the evolved global context can be used to further boost the capability of image representation.
    - 最後に、進化したグローバルコンテキストを使用して、画像表現の機能をさらに向上させることができます。

- Similar to the projection operation (Eq. 1), we again use another transformation matrix to re-project the graph nodes to images features. 
    - 射影演算（式１）と同様に、我々はまた別の変換行列を使用してグラフノードを画像特徴に再射影する。

- We apply residual connection [16] to further enhance visual representation with the original feature maps X.
    - 元の特徴マップＸを用いた視覚的表現をさらに向上させるために、残差接続［１６］を適用する。

- As a result, The image features are updated by the weighted mappings from each graph node that represents different characteristics of semantic parts.
    - その結果、画像特徴は、意味的部分の異なる特性を表す各グラフノードからの加重マッピングによって更新される。


### 3.2. Inter-Graph Transfer

- To attentively distill relevant semantics from one source graph to another target graph, we introduce Inter-Graph Transfer to bridge all semantic labels from different datasets.
    - あるソースグラフから別のターゲットグラフに関連するセマンティクスを注意深く抽出するために、異なるデータセットからすべてのセマンティックラベルを橋渡しする Inter-Graph Transfer を導入します。

- Although different levels of human parsing tasks have diverse distinct part labels, there are explicit hierarchical correlations among them to be exploited.
    - 異なるレベルの人間の構文解析タスクは多様な異なる部分ラベルを持っていますが、利用されるべきそれらの間には明示的な階層的相関関係があります。

- For example, torso label in a dataset includes upper-clothes and pants in another dataset, and the upper-clothes label can be composed of more fine-grained categories (e.g., coat, T-shirt and sweater) in the third dataset, as shown in Fig. 1.
    - たとえば、図１に示すように、データセット内の胴体ラベルには、上着と別のデータセット内のズボンが含まれ、上着ラベルは、3番目のデータセット内のより細かいカテゴリ（コート、Tシャツ、セーターなど）で構成できます。

- We make efforts to explore various graph transfer dependencies between different label sets, including feature-level similarity, handcraft relationship, and learnable weight matrix.
    - 機能レベルの類似性、手作りの関係、学習可能な重み行列など、さまざまなラベルセット間のさまざまなグラフ転送の依存関係を調査するように努めています。

- Moreover, considering that the complex relationships between different semantic labels are arduous to capture from limited training data, we employ semantic similarity that is encapsulated with linguistic knowledge from word embedding [34] to preserve the semantic consistency in a scene. 
    - さらに、異なる意味ラベル間の複雑な関係は限られた訓練データから捉えるのが困難であることを考慮して、我々はシーン内の意味の一貫性を保つために単語埋め込みからの言語知識でカプセル化された意味の類似性を用いる[34]。

- We encode these different types of relationships into the network to enhance the graph transfer capability.
    - グラフ転送機能を強化するために、これらの異なるタイプの関係をネットワークにエンコードします。

---

- Let Gs = (Vs,Es) denotes a source graph and Gt = (Vt , Et ) denotes a target graph, where Gs and Gt may have different structures and characteristics.
    - Ｇｓ ＝（Ｖｓ、Ｅｓ）がソースグラフを表し、Ｇｔ ＝（Ｖｔ、Ｅｔ）がターゲットグラフを表すとする。ここで、ＧｓおよびＧｔは異なる構造および特性を有し得る。

- We can represent a graph as a matrix Z ∈ R^{N×D}, where N = |V| and D is the dimension of each vertex v ∈ V .
    - グラフを行列 Z∈R^{N×D} として表すことができます。ここで、N = |V| で、D は各頂点の次元v∈Vである。

- The graph transformer can be formulated as:

![image](https://user-images.githubusercontent.com/25688193/60810982-446cf300-a1c9-11e9-8a73-66ce19e0e485.png)

- where Atr ∈ RNt×Ns is a transfer matrix for mapping the graph representation from Zs to Zt . Wtr ∈ RDs ×Dt is a trainable weight matrix.
    - ここで、Atr∈RNt×Nsはグラフ表現をZsからZtに写像するための伝達行列である。 Wtr∈RDs×Dtは学習可能な重み行列です。

- We seek to find a better graph transfer dependency Atr = ai,j, i=[1,Nt ], j =[1,Ns ] , where ai,j means the transfer weight from the jth semantic node of source graph to the ith semantic node of target graph.
    - より良いグラフ転送依存性Atr = ai、j、i = [1、Nt]、j = [1、Ns]を見つけようとします。ここで、ai、jはソースグラフのj番目の意味ノードからi番目への転送重みを意味します ターゲットグラフのセマンティックノード。

- We consider and compare four schemes for the transfer matrix.
    - 伝達行列について4つの方式を検討し、比較します。


#### Handcraft relation.

- Considering the inherent correlation between two semantic parts, we first define the relation matrix as a hard weight, i.e., {0, 1}.
    - ２つの意味部分間の固有の [inherent] 相関を考慮して、我々は最初に関係行列をハードウェイト、すなわち｛０、１｝として定義する。

- When two nodes have a subordinate relationship, the value of edge between them is 1, else is 0.
    - 2つのノードが従属関係にある場合、それらの間のedgeの値は1です。それ以外の場合は0です。

- For example, hair is a part of head, so the edge value between hair node of the target graph and the head node of the source graph is 1.
    - たとえば、hairはheadの一部であるため、ターゲットグラフのhairノードとソースグラフのheadノードの間のエッジ値は1です。

#### Learnable matrix.

- In this way, we randomly initialize the transfer matrix Atr , which can be learned with the whole network during training.
    - このようにして、我々はトレーニング中にネットワーク全体で学習することができる伝達行列Atrをランダムに初期化する。

#### Feature similarity.

- The transfer matrix can also be dynamically established by computing the similarity between the source graph nodes and target graph nodes, which have encoded high-level semantic information. The transfer weight ai,j can be calculated as:
    - 伝達行列はまた、高レベルの意味情報を符号化したソースグラフノードとターゲットグラフノードとの間の類似性を計算することによって動的に確立することができる。 移送重量ａ ｉ、ｊは、次のように計算することができる。

![image](https://user-images.githubusercontent.com/25688193/60958531-799c5100-a341-11e9-86f5-a46ea7edac9a.png)

- where sim(x, y) is the cosine similarity between x and y. vis is the features of the ith target node, and vjt is the features of the jth source node.
    - ここで、sim（x、y）はxとyのコサイン類似度です。 visはi番目のターゲットノードの機能、vjtはj番目のソースノードの機能です。

#### Semantic similarity. 

- Besides the visual information, we further explore the linguistic knowledge to construct the transfer matrix.
    - 視覚情報に加えて、我々はさらに伝達マトリックスを構築するための言語学的知識を探求する。

- We use the word2vec model [34] to map the semantic word of labels to a word embedding vector.
    - ラベルの意味語を単語埋め込みベクトルにマッピングするためにword2vecモデル[34]を使用します。
    
- Then we compute the similarity between the nodes of the source graph Vs and the nodes of the target graph Vt, which can be formulated as:
    - 次に、ソースグラフVsのノードとターゲットグラフVtのノードとの間の類似度を計算します。これは次のように定式化できます。

![image](https://user-images.githubusercontent.com/25688193/60958580-8e78e480-a341-11e9-8f47-c90c8effc5fd.png)

- where sij means the cosine similarity between the word embedding vectors of ith target node and jth source node.
    - ここで、sij は、i番目のターゲットノードとj番目のソースノードのベクトルを埋め込む単語間のコサイン類似度を意味します。

---

- With the well-defined transfer matrix, the target graph features and source graph knowledge can be combined and propagated again by graph reasoning, the same as the Eq. 3.
    - 明確に定義された伝達行列を用いて、式３と同様に、ターゲットグラフ特徴とソースグラフ知識とを graph reasoning によって再び組み合わせて伝播させることができる。

- Furthermore, the direction of the transfer is flexible, that is, two graphs can be jointly transferred from each other.
    - さらに、転送の方向は柔軟であり、すなわち、２つのグラフを互いに一緒に転送することができる。

- Accordingly, the hierarchical information of different label sets can be associated and propagated via the cooperation of Intra-Graph Reasoning and Inter-Graph Transfer, which enables the whole network to generate more discriminative features to perform fine-grained pixel-wise classification.
    - したがって、異なるラベルセットの階層情報は、グラフ内推論およびグラフ間転送の協働を介して関連付けおよび伝播することができ、これにより、ネットワーク全体が、よりきめの細かいピクセル単位の分類を実行するためのより識別可能な特徴を生成することができる。

### 3.3. Universal Human Parsing

- As shown in Fig. 2, apart from improving the performance of one model by utilizing the information transferred from other graphs, our Graphonomy can also be naturally used to train a universal human parsing task for combining diverse parsing datasets.
    - 図2に示すように、他のグラフから転送された情報を利用して1つのモデルのパフォーマンスを向上させることとは別に、私たちのGraphonomyは自然に多様な解析データセットを組み合わせるための普遍的な人間解析タスクを訓練するために使用できます。

- As different datasets have large label discrepancy, previous parsing works must tune highly- specific models for each dataset or perform multi-task learning with several independent branches where each of them handles one level of the tasks.
    - 異なるデータセットはラベルの不一致が大きいため、以前の解析作業では各データセットに固有のモデルを調整するか、それぞれが1つのレベルのタスクを処理する複数の独立したブランチでマルチタスク学習を実行する必要があります。

- By contrast, with the proposed Intra-Graph Reasoning and Inter-Graph Transfer, our Graphonomy is able to alleviate the label discrepancy issues and stabilize the parameter optimization during joint training in an end-to-end way.
    - 対照的に、提案されたグラフ内推論およびグラフ間転送を用いると、我々のGraphonomyは、ラベル間の不一致の問題を軽減し、エンドツーエンドの方法で共同トレーニング中のパラメータ最適化を安定させることができる。

---

- Another merit of our Graphonomy is the ability to extend the model capacity in an online way.
    - 私たちのGraphonomyのもう一つの利点は、オンラインでモデル容量を拡張できることです。

- Benefiting from the usage of graph transfer learning and joint training strategy, we can dynamically add and prune semantic labels for different purposes (e.g., adding more dataset) while keeping the network structure and previously learned parameters.
    - グラフ転送学習および共同トレーニング戦略の使用の恩恵を受けて、
    - ネットワーク構造および以前に学習したパラメータを維持しながら、
    - 異なる目的（たとえば、データセットの追加）のために意味ラベルを動的に追加および取り除く [prune] ことができる。

# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## 4. Experiments

- In this section, we first introduce implementation details and related datasets. Then, we report quantitative comparisons with several state-of-the-art methods.
    - このセクションでは、最初に実装の詳細と関連データセットを紹介します。 次に、いくつかの最先端の方法と定量的比較を報告します。

- Furthermore, we conduct ablation studies to validate the effectiveness of each main component of our Graphonomy and present some qualitative results for the perceptual comparison.
    - さらに、我々は我々のGraphonomyの各主成分の有効性を検証し、知覚的比較のためにいくつかの定性的結果を提示するためにアブレーション研究を行っています。

### 4.1. Experimental Settings

#### Implementation Details

- We use the basic structure and network settings provided by DeepLab v3+ [3]. Following [3], we employ the Xception [7] pre-trained on COCO [31] as our network backbone and output stride = 16.
    - 私たちはDeepLab v3 + [3]が提供する基本構造とネットワーク設定を使います。 [3]に続いて、COCO [31]で事前にトレーニングされたXception [7]をネットワークバックボーンとして使用し、ストライド= 16を出力します。
    
- The number of nodes in the graph is set according to the number of categories of the datasets, i.e., N = 7 for Pascal- Person-Part dataset, N = 18 for ATR dataset, N = 20 for CIHP dataset.

- The feature dimension D of each semantic node is 128.

- The Intra-Graph Reasoning module has three graph convolution layers with ReLU activate function.

- For Inter-Graph Transfer, we use the pre-trained model on source dataset and randomly initialize the weight of the target graph. Then we perform end-to-end joint training for the whole network on the target dataset.

---

- During training, the 512x512 inputs are randomly resized between 0.5 and 2, cropped and flipped from the images.

- The initial learning rate is 0.007. Following [3], we employ a “ploy” learning rate policy. We adopt SGD opti- mizer with momentum = 0.9 and weight decay of 5e − 4.

- To stabilize the predictions, we perform inference by averaging results of left-right flipped images and multi-scale inputs with the scale from 0.50 to 1.75 in increments of 0.25.

---

- Our method is implemented by extending the Pytorch framework [33] and we reproduce DeepLab v3+ [3] following all the settings in its paper.

- All networks are trained on four TITAN XP GPUs. Due to the GPU memory limitation, the batch size is set to be 12. For each dataset, we train all models at the same settings for 100 epochs for the good convergence. To stabilize the inference, the resolution of every input is consistent with the original image. The code and models are available at https://github.com/Gaoyiminggithub/Graphonomy.


#### Dataset and Evaluation Metric

- We evaluate the performance of our Graphonomy on three human parsing datasets with different label definition and annotations, including PASCAL-Person-Part dataset [6], ATR dataset [28], and Crowd Instance-Level Human Parsing (CIHP) dataset [13].

- The part labels among them are hierarchically correlated and the label granularity is from coarse to fine.

- Referring to their dataset papers, we use the evaluation metrics including accuracy, the standard intersection over union (IoU) criterion, and average F-1 score.

### 4.2. Comparison with state-of-the-arts

### 4.3. Universal Human Parsing

- To sufficiently utilize all human parsing resources and unify label annotations from different domains or at various levels of granularity, we train a universal human parsing model to unify all kinds of label annotations from different resources and tackle different levels of human parsing, which is denoted as “Graphonomy (Universal Human Pars- ing)”.

- We combine all training samples from three datasets and select images from the same dataset to construct one batch at each step.

- As reported in Table 1, 2, 3, our method achieves favorable performance on all datasets.

- We also compare our Graphonomy with multi-task learning method by appending three parallel branches upon the backbone with each branch predicting the labels of one dataset respectively.

- Superior to multi-task learning, our Graphonomy is able to distill universal semantic graph representation and enhance individualized representation for each label graph.

---

- We also present the qualitative universal human parsing results in Fig. 4.

- Our Graphonomy is able to generate precise and fine-grained results for different levels of human parsing tasks by distilling universal semantic graph representation to each specific task, which further verifies the rationality of our Graphonomy based on the assumption that incorporating hierarchical graph transfer learning upon the deep convolutional networks can capture the critical information across the datasets to achieve good capability in universal human parsing.

### 4.4. Ablation Studies


# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


