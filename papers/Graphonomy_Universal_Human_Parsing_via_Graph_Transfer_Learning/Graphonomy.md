# ■ 論文
- 論文タイトル："Graphonomy: Universal Human Parsing via Graph Transfer Learning"
- 論文リンク：https://arxiv.org/abs/1904.04536
- 論文投稿日付：2019/04/09
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- Prior highly-tuned human parsing models tend to fit towards each dataset in a specific domain or with discrepant label granularity, and can hardly be adapted to other human parsing tasks without extensive re-training.
   - 以前の高度に調整されたヒューマン解析モデルは、特定のドメイン内の、または矛盾するラベルの粒度で各データセットに適合する傾向があり、大規模な再トレーニングなしでは他のヒューマン解析タスクにはほとんど適応できません。 

- In this paper, we aim to learn a single universal human parsing model that can tackle all kinds of human parsing needs by unifying label annotations from different domains or at various levels of granularity.
    - 本論文では、異なるドメインからのラベル注釈をさまざまなレベルの粒度 [granularity] で統合する [unifying] ことによって、あらゆる種類の人間の解析ニーズに対応できる単一のユニバーサル人間解析モデルを学習することを目的としています。

- This poses many fundamental learning challenges, e.g. discovering underlying semantic structures among different label granularity, performing proper transfer learning across different image domains, and identifying and utilizing label redundancies across related tasks.    
    - これは多くの基本的な学習課題を引き起こします。 異なるラベルの粒度の間で根本的な意味構造を発見し、異なる画像ドメインにわたって適切な転送学習を実行し、そして関連するタスクにわたってラベル冗長性を識別し利用する。

---

- To address these challenges, we propose a new universal human parsing agent, named “Graphonomy”, which incorporates hierarchical graph transfer learning upon the conventional parsing network to encode the underlying label semantic structures and propagate relevant semantic information.
    - これらの課題に取り組むために、我々は、「Graphonomy」と呼ばれる新しい普遍的な人間解析エージェントを提案する。
    - これは、基礎となるラベルセマンティック構造を符号化し、関連セマンティック情報を伝播するために、従来の構文解析ネットワーク上で階層グラフ転送学習を組み込む。

- In particular, Graphonomy first learns and propagates compact high-level graph representation among the labels within one dataset via Intra-Graph Reasoning, and then transfers semantic information across multiple datasets via Inter-Graph Transfer. 
    - 特に、Graphonomy はまず Intra-Graph Reasoning を介して1つのデータセット内のラベル間でコンパクトな高水準グラフ表現を学習して伝播し、次にInter-Graph Transferを介して複数のデータセットにわたってセマンティック情報を転送します。

- Various graph transfer dependencies (e.g., similarity, linguistic knowledge) between different datasets are analyzed and encoded to enhance graph transfer capability.   
    - 異なるデータセット間の様々なグラフ転送依存性（例えば、類似性、言語学的知識）は、グラフ転送能力（＝転移学習能力）を向上させるために分析され符号化される。

- By distilling universal semantic graph representation to each specific task, Graphonomy is able to predict all levels of parsing labels in one system without piling up the complexity.
    汎用のセマンティックグラフ表現をそれぞれの特定のタスクに分類することによって、Graphonomyは複雑さを積み重ねることなく1つのシステム内のすべてのレベルの構文解析ラベルを予測することができます。
    
- Experimental results show Graphonomy effectively achieves the state-of-the-art results on three human parsing benchmarks as well as advantageous universal human parsing performance.
    - 実験結果は、Graphonomyが3つの人間の構文解析ベンチマークと最先端の一般的な人間の構文解析パフォーマンスを効果的に達成することを示しています。

# ■ イントロダクション（何をしたいか？）

## 1. Introduction

- Human visual systems are capable of accomplishing holistic human understanding at a single glance on a person image, e.g., separating the person from the background, understanding the pose, and recognizing the clothes the person wears.
    - 人間の視覚システムは、例えば背景から人を分離し、ポーズを理解し、そしてその人が着ている服を認識することなど、人の画像上でひと目で全体的な人間の理解を達成することができる。

- Nevertheless, recent research efforts on human understanding have been devoted to developing numerous highly-specific and distinct models for each individual application, e.g. foreground human segmentation task [8, 15], coarse clothes segmentation task [25, 28] and fine-grained human part/clothes parsing task [14, 39].
    - それにもかかわらず、人間の理解に関する最近の研究努力は、個々の用途ごとに多数の非常に特異的かつ異なるモデルを開発することに向けられてきた。 前景の人間セグメンテーションタスク[8、15]、粗い衣服セグメンテーションタスク[25、28]、およびきめの細かい人間の部分/衣服解析タスク[14、39]。

- Despite the common underlying human structure and shared intrinsic semantic information (e.g. upper-clothes can be interpreted as coat or shirt), these highly-tuned networks have sacrificed the generalization capability by only fitting towards each dataset domain and discrepant label granularity.
    - 共通の基本的な人間の構造と共有された固有のセマンティック情報（例えば、上着はコートやシャツと解釈できる）にもかかわらず、これらの高度に調整されたネットワークは各データセットドメインと不一致ラベル粒度に合わせることによって一般化機能を犠牲にしました。

- It is difficult to directly adapt the model trained on one dataset to another related task, and thus requires redundant heavy data annotation and extensive computation to train each specific model.
    - あるデータセットで訓練されたモデルを別の関連タスクに直接適合させることは困難であり、したがって、各特定のモデルを訓練するために冗長な重いデータ注釈と広範な計算が必要になります。

- To address these realistic challenges and avoid training redundant models for correlated tasks, we make the first attempt to investigate a single universal human parsing agent that tackles human parsing tasks at different coarse to fine-grained levels, as illustrated in Fig. 1.
    - これらの現実的な課題に対処し、相関タスクのための冗長モデルのトレーニングを避けるために、図1に示すように、粗いレベルから細かいレベルで人間の解析タスクに取り組む単一の汎用人間解析エージェントを調査する最初の試みを行います。

---

![image](https://user-images.githubusercontent.com/25688193/60794193-3193f780-a1a4-11e9-885c-bb681a9245b2.png)

- > Figure 1. With huge different granularity and quantity of semantic labels, human parsing is isolated into multiple level tasks that hinder the model generation capability and data annotation utilization.
    - > 図1。非常に異なる粒度と大量の意味ラベルで、人間の構文解析は、モデル生成機能とデータ注釈の利用を妨げる複数のレベルのタスクに分離されます。

- > For example, the head region on a dataset is further annotated into several fine-grained concepts on another dataset, such as hat, hair and face. However, different semantic parts still have some intrinsic and hierarchical relations (e.g., Head includes the face. Face is next to hair), which can be encoding as intra-graph and inter-graph connections for better information propagation.
    - > たとえば、データセットの頭の領域は、帽子、髪の毛、顔など、他のデータセットのいくつかの粒度の細かい概念にさらに注釈が付けられます。 しかしながら、異なる意味的部分は依然としていくつかの本質的かつ階層的な関係を有し（例えば、頭は顔を含む。顔は髪の毛の隣にある）、それはより良い情報伝播のためのグラフ内およびグラフ間接続として符号化することができる。

- > To alleviate the label discrepancy issue and take advantage of their semantic correlations, we introduce a universal human parsing agent, named as “Graphonomy”, which models the global semantic coherency in multiple domains via graph transfer learning to achieve multiple levels of human parsing tasks.
    - > ラベルの不一致の問題を軽減し、それらの意味的な相関関係を利用するために、我々は「Graphonomy」と名付けられた普遍的な人間構文解析エージェントを導入する。

---

- The most straightforward solution to universal human parsing would be posing it as a multi-task learning problem, and integrating multiple segmentation branches upon one shared backbone network [2, 14, 22, 25, 28].
    - 普遍的な人間の構文解析に対する最も直接的な解決策は、それをマルチタスク学習問題と見なし、複数のセグメンテーションブランチを1つの共有バックボーンネットワークに統合することです[2、14、22、25、28]。

- This line of research only considers the brute-force feature-level information sharing while disregarding the underlying common semantic knowledge, such as label hierarchy, label visual similarity, and linguistic/context correlations.
    - この一連の研究では、ラベル階層、ラベルの視覚的類似性、言語的/文脈的相関などの基本的な意味論的知識を無視しながら、力ずくな[brute-force] 特徴レベルの情報共有のみを考慮しています。

- More recently, some techniques are explored to capture the human structure information by resorting to complex graphical models (e.g., Conditional Random Fields (CRFs)) [2], self-supervised loss [14] or human pose priors [9, 12, 23].
    - ごく最近では、複雑なグラフィカルモデル（例えば条件付き確率場（CRF））[2]、自己管理損失[14]、または人間のポーズ前[9]、[12]、[23]に頼ることによって人間の構造情報を捉えるための技術が探求されます。 ]。

- However, they did not explicitly model the semantic corre- lations of different body parts and clothing accessories, and still show unsatisfactory results for rare fine-grained labels.
    - しかしながら、彼らは異なる身体の部分と衣服の付属品の意味的な相関関係を明確にモデル化しておらず、それでもまれなきめの細かいラベルに対して満足のいく結果を示していません。
    
---

- One key factor of designing a universal human parsing agent is to have proper transfer learning and knowledge integration among different human parsing tasks, as the label discrepancy across different datasets [6, 13, 14, 39] largely hinders direct data and model unification.
    - ユニバーサルデータ解析エージェントを設計する際の1つの重要な要素は、異なるデータセット間でのラベルの不一致[6]、[13]、[14]、[39]が直接データとモデルの統一を妨げるためです。

- In this paper, we achieve this goal by explicitly incorporating human knowledge and label taxonomy into intermediate graph representation learning beyond local convolutions, called “Graphonomy” (graph taxonomy).
    - **本論文では、この目標を達成するために、「Graphonomy」（グラフ分類法）と呼ばれる、局所的な畳み込みを超えた中間グラフ表現学習に人間の知識とラベル分類法 [taxonomy] を明示的に取り入れます。**

- Our Graphonomy learns the global and common semantic coherency in multiple domains via graph transfer learning to solve multiple levels of human parsing tasks and enforce their mutual benefits upon each other.
    - **私たちのGraphonomyは、グラフ転送学習を介して複数のドメインにおけるグローバルで共通のセマンティック一貫性を学習し、複数レベルの人間の解析タスクを解決し、互いの利益を強化します。**

---

- Taking advantage of geometric deep learning [19, 20], our Graphonomy simply integrates two cooperative modules for graph transfer learning.
    - 幾何学的深層学習[19、20]を利用して、私たちのGraphonomyは単純にグラフ転送学習のための2つの協調モジュールを統合しています。

- First, we introduce Intra- Graph Reasoning to progressively refine graph representations within the same graph structure, in which each graph node is responsible for segmenting out regions of one semantic part in a dataset.
    - まず、グラフ内推論を導入して、同じグラフ構造内でグラフ表現を段階的に改良します。各グラフノードは、データセット内の1つの意味部分の領域をセグメント化する責任があります。

- Specifically, we first project the extracted image features into a graph, where pixels with similar features are assigned to the same semantic vertex.
    - 具体的には、我々は最初に抽出された画像特徴をグラフに投影し、ここで類似の特徴を有するピクセルは同じ意味的頂点に割り当てられる。

- We elaborately design the adjacency matrix to encode the semantic relations, constrained by the connection of human body structure, as shown in Fig. 3.
    - 我々は、図3に示すように、人体構造の接続によって制約される意味関係を符号化するために、隣接行列を入念に設計します。

- After the message propagation via graph convolutions, the updated vertexes are reprojected to make the visual feature maps more discrimina- tive for pixel-level classification.
    - グラフ畳み込みによるメッセージ伝播の後に、更新された頂点が投影され、視覚的特徴マップがピクセルレベルの分類に対してより区別しやすくなる。

---

- Additionally, we build an Inter-Graph Transfer module to attentively distill related semantics from the graph in one domain/task to the one in another domain, which bridges the semantic labels from different datasets, and effectively utilize the annotations at multiple levels.
    - さらに、Inter-Graph Transferモジュールを構築して、あるドメイン/タスク内のグラフから別のドメイン内のグラフに関連するセマンティクスを注意深く [attentively] 抽出し [distill]、異なるデータセットのセマンティックラベルを橋渡し、複数レベルでアノテーションを効果的に活用します。

- To enhance graph transfer capability, we make the first effort to exploit various graph transfer dependencies among different datasets.
    - グラフ転送能力を強化するために、異なるデータセット間でさまざまなグラフ転送の依存関係を利用する最初の努力をします。

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
    - 1）私たちは、単一の普遍的モデルを使って、あらゆるレベルの人間の解析作業に取り組む最初の試みを行います。

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

- Fig. 2 gives an overview of our proposed framework. Our approach can be embedded in any modern human parsing system by enhancing its original image features via graph transfer learning. We first learn and propagate compact high-level semantic graph representation within one dataset via Intra-Graph Reasoning, and then transfer and fuse the semantic information across multiple datasets via Inter-Graph Transfer driven by explicit hierarchical semantic label structures.
    - 図2に提案したフレームワークの概要を示します。 我々のアプローチは、グラフ転送学習を介してそのオリジナルの画像特徴を向上させることによって現代の人間の解析システムに埋め込むことができる。 まず、グラフ内推論を介して1つのデータセット内でコンパクトな高レベルの意味グラフ表現を学習して伝播し、次に明示的な階層的意味ラベル構造によって駆動されるグラフ間転送を介して複数のデータセットにわたって意味情報を転送および融合します。

---

![image](https://user-images.githubusercontent.com/25688193/60794261-56886a80-a1a4-11e9-9a1e-154370f2b148.png)

- > Figure 2. Illustration of our Graphonomy that tackles universal human parsing via graph transfer learning to achieve multiple levels of human parsing tasks and better annotation utilization.
    - > 図2.グラフ転送学習による普遍的な人間の構文解析に取り組む私たちのGraphonomyの図。人間の構文解析のタスクの複数のレベルとより良い注釈の利用を達成する。

- > The image features extracted by deep convolutional networks are projected into a high-level graph representation with semantic nodes and edges defined according to the body structure.
    - > 大域的情報は、グラフ内推論によって伝播され、視覚的特徴の識別可能性を高めるために再投影される。

- > The global information is propagated via Intra-Graph Reasoning and re-projected to enhance the discriminability of visual features.
    - > ディープコンボリューションネットワークによって抽出された画像特徴は、体構造に従って定義された意味論的ノードおよびエッジを有する高水準グラフ表現に投影される。

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


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


