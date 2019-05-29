# ■ 論文
- 論文タイトル："Toward Characteristic-Preserving Image-based Virtual Try-On Network"
- 論文リンク：https://arxiv.org/abs/1807.07688
- 論文投稿日付：2018/07/20(v1), 2018/09/12(v2)
- 著者：(Sun Yat-sen University, China)
- categories：

# ■ 概要（何をしたか？）

## x. ABSTRACT

- Image-based virtual try-on systems for fitting a new in-shop clothes into a person image have attracted increasing research attention, yet is still challenging.
    - 新しい店内の服を個人にフィットするための、画像ベースの仮想試着 [try-on] システムは、研究の注目を増加しているが、まだ挑戦的である。

- A desirable pipeline should not only transform the target clothes into the most fitting shape seamlessly but also preserve well the clothes identity in the generated image, that is, the key characteristics (e.g. texture, logo, embroidery) that depict the original clothes.
    - 望ましいパイプラインは、ターゲットの服を最も適合する形状にシームレスに変換するだけでなく、
    - 生成された画像内の服のアイデンティティ、すなわち [that is]、元の服を描写する [depict] 重要な特性（例えば、テクスチャ、ロゴ、刺繍 [embroidery]）もよく保存する [preserve]。

- However, previous image-conditioned generation works fail to meet these critical requirements towards the plausible virtual try-on performance since they fail to handle large spatial misalignment between the input image and target clothes.
    - しかしながら、以前の画像調整された [image-conditioned] 生成研究は、入力画像と目標衣服との間の大きな空間的な不均衡 [misalignment] を処理する [handle] ことができないので、もっともらしい [plausible] 仮想試着性能に対するこれらの重大な要求を満たすことができない。

- Prior work explicitly tackled spatial deformation using shape context matching, but failed to preserve clothing details due to its coarse-to-fine strategy.
    - 以前の研究は、形状コンテキストマッチングを使用して空間的な変形 [deformation] に明示的に取り組んだ [tackled] が、
    - その coarse-to-fine 法？（その粗密な戦略？）のために、衣服の詳細を保存することができなかった。

- In this work, we propose a new fully-learnable Characteristic-Preserving Virtual Try-On Network (CP-VTON) for addressing all real-world challenges in this task.
    - この研究では、このタスクにおけるすべての現実的な課題に対処する [adressing] ために、新しい完全に学習可能な Characteristic-Preserving Virtual Try-On Network (CP-VTON) を提案します。

- First, CP-VTON learns a thin-plate spline transformation for transforming the in-shop clothes into fitting the body shape of the target person via a new Geometric Matching Module (GMM) rather than computing correspondences of interest points as prior works did.
    - まず初めに、CP-VTON は、新しい Geometric Matching Module (GMM) 経由で、店内の服を、目標人物の体の形状にフィットするように変換するために、薄板スプライン変換を学習します。
    - 以前の研究のように関心点の対応を計算するのではなく、
    
- Second, to alleviate boundary artifacts of warped clothes and make the results more realistic, we employ a Try-On Module that learns a composition mask to integrate the warped clothes and the rendered image to ensure smoothness.
    - 次に、ゆがんだ [warped] 服の境界加工物 [artifacts] を軽減 [alleviate] し、結果をより現実的にするために、
    - <font color="Pink">我々は、ゆがんだ服と、滑らかさを確保するためのレンダリングされたイメージを統合するために、構成 [composition] マスクを学習するTry-Onモジュールを使用します。</font>

- Extensive experiments on a fashion dataset demonstrate our CP-VTON achieves the state-ofthe-art virtual try-on performance both qualitatively and quantitatively
    - ファッションデータセットに関する広範な実験は、我々の CP-VTON が、定性的にも定量的にも仮想試着パフォーマンスの SOTA を達成することを実証している。

- Code is available at https://github.com/sergeywong/cp-vton.


# ■ イントロダクション（何をしたいか？）

## 1. Introduction

- Online apparel shopping has huge commercial advantages compared to traditional shopping(e.g. time, choice, price) but lacks physical apprehension.
    - オンラインアパレル（衣服） [apparel] ショッピングは、伝統的なショッピングと比較して大きな商業的利点（例えば、時間、選択、価格）を有するが、物理的な不安 [apprehension] を欠いている。

- To create a shopping environment close to reality, virtual try-on technology has attracted a lot of interests recently by delivering product information similar to that obtained from direct product examination.
    - 現実に近いショッピング環境を構築するために、仮想試着技術は、直接的な製品検査から得られるものと同様の、製品情報を配信することによって、最近多くの関心を集めています。

- It allows users to experience themselves wearing different clothes without efforts of changing them physically.
    - それは、ユーザーがそれらを物理的に変える努力なしで、彼ら自身が異なる服を着ているのを経験することを可能にする。 

- This helps users to quickly judge whether they like a garment or not and make buying decisions, and improves sales efficiency of retailers.
    - これにより、ユーザーは衣服 [garment] が好きかどうかをすばやく判断して購入を決定し、小売業者の販売効率を向上させることができます。

- The traditional pipeline is to use computer graphics to build 3D models and render the output images since graphics methods provide precise control of geometric transformations and physical constraints.
    - グラフィックス手法は幾何学的変換と物理的制約の正確な制御を提供するたので、
    - 従来のパイプラインでは、3Dモデルを構築するために、コンピュータグラフィックスを使用し、出力画像をレンダリングしている。

- But these approaches require plenty of manual labor or expensive devices to collect necessary information for building 3D models and massive computations.
    - しかし、これらのアプローチでは、3Dモデルの構築や大量の計算に対して、必要な情報を収集するための、多くの手作業または高価な機器が必要です。

---

- More recently, the image-based virtual try-on system [10] without resorting to 3D information, provides a more economical solution and shows promising results by reformulating it as a conditional image generation problem.
    - より最近では、3D情報に頼ら [resort] ない画像ベースの仮想試着システム [10] は、より経済的な解決策を提供し、
    - 条件付き画像生成問題としてそれを再公式化することによって有望な結果を示しています。

- Given two images, one of a person and the other of an in-shop clothes, such pipeline aims to synthesize a new image that meets the following requirements:
    - １つは人、もう一方は店内の服というような、２つの画像が与えられると、- そのようなパイプラインは、以下の要件を満たす新しい画像を合成することを狙いとする。

- a) the person is dressed in the new clothes;
    - a）その人は新しい服を着ている。

- b) the original body shape and pose are retained;
    - b）元の体型と姿勢が保持されている。

- c) the clothing product with high-fidelity is warped smoothly and seamlessly connected with other parts;
    - c）忠実度 [fidelity] の高い衣料品が、他の部品と滑らかにそして継ぎ目なく接続されている。

- d) the characteristics of clothing product, such as texture, logo and text, are well preserved, without any noticeable artifacts and distortions.
    - d）風合い、ロゴ、文字などの衣料品の特性は、目立ったアーティファクトや歪み [distortions] がなく、よく保存されている。

- Current research and advances in conditional image generation (e.g. image-to-image translation [12,38,5,34,20,6]) make it seem to be a natural approach of facilitating this problem.
    - 条件付き画像生成における現在の研究および進歩（例えば、image-to-image 変換［１２、３８、５、３４、２０、６］）は、それがこの問題を容易にする [facilitating] 自然なアプローチであるように思わせる。

- Besides the common pixel-to-pixel losses (e.g. L1 or L2 losses) and perceptual loss [14], an adversarial loss [12] is used to alleviate the blurry issue in some degree, but still misses critical details.
    - 一般的なピクセル間損失（例えばＬ１またはＬ２の損失）と、perceptual loss（知覚的な損失）［１４］に加えて、敵対的な損失［１２］はある程度ぼやけた問題を軽減するために使用されるが、依然として重大な詳細を見逃している。

- Furthermore, these methods can only handle the task with roughly aligned input-output pairs and fail to deal with large transformation cases.
    - さらに、これらの方法は、大まかに整列された [aligned] 入出力ペアを用いてタスクを処理することしかできず、大きな変換ケースを扱うことができない。

- Such limitations hinder their application on this challenging virtual try-on task in the wild.
    - そのような制限は、このような挑戦的な仮想試着タスクへの応用を妨げる [hinder]。

- One reason is the poor capability in preserving details when facing large geometric changes, e.g. conditioned on unaligned image [23].
    - 一つの理由は、大きな幾何学的変化に直面したときに細部を保存する能力が乏しいことです。（例えば、位置合わせ（アライメント）されていない画像を条件とする[23]）。

- The best practice in image-conditional virtual try-on is still a two-stage pipeline VITON [10].
    - 画像条件付き仮想試行のベストプラクティスは、まだ2段階のパイプラインVITON [10]です。

- But their performances are far from the plausible and desired generation, as illustrated in Fig. 1.
    - しかし、それらのパフォーマンスは、図1に示すように、もっともらしく望ましい生成からは程遠いものです。

- We argue that the main reason lies in the imperfect shape-context matching for aligning clothes and body shape, and the inferior appearance merging strategy.
    - 主な理由は、服と体型を揃える [aligning] ための不完全な形状コンテキストマッチングと、
    - 劣った外観マージ戦略に横たわると考えます。

---

![image](https://user-images.githubusercontent.com/25688193/58450255-123b9080-8149-11e9-8949-cdb43352ffef.png)<br>

> - Fig. 1. The proposed CP-VTON can generate more realistic image-based virtual try-on results that preserve well key characteristics of the in-shop clothes, compared to the state-of-the-art VITON [10].
    > - 図１：提案された CP-VTON は、SOTA の VITON と比較して、服の重要な特性をうまく保存するような、よりリアルな画像ベースの仮想試着結果を生成することが出来る。

---

- To address the aforementioned challenges, we present a new image-based method that successfully achieves the plausible try-on image syntheses while preserving cloth characteristics, such as texture, logo, text and so on, named as Characteristic-Preserving Image-based Virtual Try-On Network (CP-VTON).
    - 前述の [aforementioned] 課題に取り組むために、
    - 我々は、新しい画像ベースの手法を提示する。
    - （この手法というのは、）
    - テクスチャ、ロゴ、テキストなどの布地特性を保存する一方で、
    - もっともらしい [plausible] 試着画像合成をうまく達成するような、
    - Characteristic-Preserving Image-based Virtual Try-On Network (CP-VTON) と名付けられた（手法である）

- In particular, distinguished from the hand-crafted shape context matching, we propose a new learnable thin-plate spline transformation via a tailored convolutional neural network in order to align well the in-shop clothes with the target person.
    - 特に、手作りの形状コンテキストマッチングとは区別されて、
    - 我々は、店内の服を対象の人とうまく合わせるために、適合された [tailored] 畳み込みニューラルネットワーク経由での、新しい学習可能な thin-plate（薄板）スプライン変換を提案する。

- The network parameters are trained from paired images of in-shop clothes and a wearer, without the need of any explicit correspondences of interest points.
    - ネットワークパラメータは、関心点の明示的な対応を必要とせずに、店内の衣服と着用者のペア画像からトレーニングされる。

- Second, our model takes the aligned clothes and clothing-agnostic yet descriptive person representation proposed in [10] as inputs, and generates a pose-coherent image and a composition mask which indicates the details of aligned clothes kept in the synthesized image.
    - 第二に、我々のモデルは、[10] で提案されたような、整列された服と 衣服にとらわれないが、説明的な人物の表現を入力として受け取り、
    - ポーズコヒーレント画像と、合成画像に保持される整列された衣服の詳細を示す合成マスクを生成する。

- The composition mask tends to utilize the information of aligned clothes and balances the smoothness of the synthesized image. 
    - 構図マスクは、整列した衣服の情報を利用する傾向があり、合成画像の滑らかさのバランスをとる。

- Extensive experiments show that the proposed model handles well the large shape and pose transformations and achieves the state-of-art results on the dataset collected by Han et al. [10] in the image-based virtual try-on task.
    - 広範囲の実験は、提案されたモデルが大きな形状と姿勢の変換をうまく処理し、
    - Hanらによって収集されたデータセットにおいて、画像ベースの仮想試着タスクで、SOTA の結果を達成することを示しています。 

---

- Our contributions can be summarized as follows:
    - 私たちの貢献は次のようにまとめることができます。

- We propose a new Characteristic-Preserving image-based Virtual Try-On Network (CP-VTON) that addresses the characteristic preserving issue when facing large spatial deformation challenge in the realistic virtual try-on task.
    - 我々は、リアルな仮想試着タスクにおいて、大きな空間的な変形課題に直面するときの、特性保存問題を解決するような、新しい Characteristic-Preserving image-based Virtual Try-On Network (CP-VTON) を提案する。

- Diffierent from the hand-crafted shape context matching, our CP-VTON in-corporates a full learnable thin-plate spline transformation via a new Geometric Matching Module to obtain more robust and powerful alignment.
    - 手作業の形状コンテキストマッチングとの違いは、我々の CP-VTON は、より多くの自動化とよるパワフルな整形を実現するのために、新しい GMM 経由での、完全に学習可能な thin-plate スプライン変換を取り込む。 [in-corporates]。

- Given aligned images, a new Try-On Module is performed to dynamically merge rendered results and warped results.
    - 整形された画像を与えれば、新しい Try-On Module は、レンダリング結果と歪んだ結果を動的にマージ（統合）することを実行する。

- Signicant superior performances in image-based virtual try-on task achieved by our CP-VTON have been extensively demonstrated by experiments on the dataset collected by Han et al. [10].
    - 画像ベース仮想試着タスクにおける、我々の CP-VTON によって達成されたより優れた重要なパフォーマンスは、Han によって収集されたデータセット [10] での実験によって、実証されている。


# ■ 結論

## 5 Conclusions

- In this paper, we propose a fully learnable image-based virtual try-on pipeline towards the characteristic-preserving image generation, named as CP-VTON, including a new geometric matching module and a try-on module with the new merging strategy.

- The geometric matching module aims at aligning in-shop clothes and target person body with large spatial displacement.

- Given aligned clothes, the try-on module learns to preserve well the detailed characteristic of clothes.

- Extensive experiments show the overall CP-VTON pipeline produces high-delity virtual try-on results that retain well key characteristics of in-shop clothes.

- Our CP-VTON achieves state-of-the-art performance on the dataset collected by Han et al. [10] both qualitatively and quantitatively.


# ■ 何をしたか？詳細

## 3 Characteristic-Preserving Virtual Try-On Network

- We address the task of image-based virtual try-on as a conditional image generation problem.
    - 我々は、条件付き画像生成問題としての、画像ベースの仮想試着のタスクに取り組む。

- Generally, given a reference image I_i of a person wearing in clothes c_i and a target clothes c, the goal of CP-VTON is to synthesize a new image I_o of the wearer in the new cloth c_o, in which the body shape and pose of I_i are retained, the characteristics of target clothes c are reserved and the effiects of the old clothes c_i are eliminated.
    - 一般的に、服 c_i を着ている人物の参照画像 I_i と目標の服 c を与えれば、
    - CP-VTON のゴールは、新しい服 c_o において、試着者の新しい画像 I_o を合成することである。
    - （この合成画像というのは、） I_i の体型や姿勢が保持され、目標の服 c の特性が保持され、古い服 c_i の効果が排除される。（ようなもの）

---

![image](https://user-images.githubusercontent.com/25688193/58453630-d6a7c300-8156-11e9-9b5c-c4426dd64897.png)

---

![image](https://user-images.githubusercontent.com/25688193/58456118-49686c80-815e-11e9-962f-bf209d17ec26.png)

- > Fig. 2. An overview of our CP-VTON, containing two main modules.
    - > 図２。我々の CP-VTON の概要で、２つのメインネットワークを含んでいる。

- > (a) Geometric Matching Module: the in-shop clothes c and input image representation p are aligned via a learnable matching module. 
    - > (a) GAMM : 店内の服 c と入力画像表現 p は、学習可能なマッチングモジュール経由で整理されている。

- > (b) Try-On Module: it generates a composition mask M and a rendered person I_r.
    - > (b) Try-On Module  : これは、構成マスク M とレンダリングされた人物 I_r を生成する。

- > The final results I_o is composed by warped clothes c^ and the rendered person I_r with the composition mask M.
    - > 最終的な結果 I_o は、歪んだ服 c^ と、構成マスク M で描写された人物 I_r によって構成される。

---


- Training with sample triplets (I_i, c, I_t) where I_t is the ground truth of I_o and c is coupled with I_t wearing in clothes c_t, is straightforward but undesirable in practice.
    - ３つの組のサンプル (I_i, c, I_t) での学習は、簡単である [straightforward] が、実際には、望ましくない。
    - （ここで、I_t は、試着者の新しい画像 I_o の ground truth であり、目標の服 ｃ は I_t と衣服 c_t を着用することとを組み合わせる。）

> I_t は教師信号の役割？

- Because these triplets are difficult to collect.
    - これらの３つの組み合わせは、収集するのが困難である。

- It is easier if I_i is same as I_t, which means that c, I_t pairs are enough.
    - もし I_i が I_t と同じであればより簡単である。
    - これは、c, I_t のペアが十分であることを意味する。
    
- These paris are in abundance from shopping websites.
    - これらのペアは、ショッピング web サイトから豊富にある [in abundance]。

- But directly training on (I_t, c, I_t) harms the model generalization ability at testing phase when only decoupled inputs (I_i, c) are available.
    - しかし、(I_t, c, I_t) で直接的に学習することは、分離された [decoupled] 入力 (I_i, c) のみが利用可能であるとき、テストフェイズにおいて、モデルの生成能力を妨げる。

- Prior work [10] addressed this dilemma by constructing a clothing-agnostic person representation p to eliminate the effiects of source clothing item c_i.
    - 以前の研究では、衣服アイテム c_i のソースとしての影響を除外するために、衣服にとらわれない [clothing-agnostic] 人物表現 p を構築することによって、このジレンマに対処した。

> この衣服にとらわれない [clothing-agnostic] 人物表現 p というのは、具体的には、図２のような、OpenPose による骨格情報（＝姿勢情報）などのこと

- With (I_t, c, I_t) transformed into a new triplet form (p, c, I_t), training and testing phase are unified.
    - <font color="Pink">(I_t, c, I_t) を新しい３つの組の形 (p, c, I_t) に変換すれば、
    - トレーニングフェイズとテストフェイズは、統一される [be unified]。</font>

- We adopted this representation in our method and further enhance it by eliminating less information from reference person image.
    - 我々は、この表現を我々の手法に適用した。
    - そして、参照人物画像からの、より少ない情報を推定することによって、それを更に高めた。

- Details are described in Sec. 3.1.
    - 詳細は、セクション 3.1 で記述される。

- One of the challenges of image-based virtual try-on lies in the large spatial misalignment between in-shop clothing item and wearer's body.
    - 画像ベースの仮想試着の課題の１つは、店内の服と試着者の体との間の、大きな空間的な不整合 [misalignment] にある。

- Existing network architectures for conditional image generation (e.g. FCN [21], UNet [28], ResNet [11]) lack the ability to handle large spatial deformation, leading to blurry try-on results.
    - 条件付き画像生成のための存在するネットワークアーキテクチャ（例えば、FCN, UNet, ResNet）は、大きな空間的な変形を操作する能力に欠けており、ぼやけた試着の結果を導く。

- We proposed a Geometric Matching Module (GMM) to explicitly align the input clothes c with aforementioned person representation p and produce a warped clothes image c^.
    - **我々は、入力服 c を、前述の人物表現 p で、明示的に整形し、歪んだ服の画像 c^ を生成するための Geometric Matching Module (GMM) を提案する。**

- GMM is a end-to-end neural network directly trained using pixel-wise L1 loss.
    - GMM は、ピクセル単位の L1損失関数を用いて、直接的に学習されたend-to-end なニューラルネットワークである。

- Sec. 3.2 gives the details.
    - セクション 3.2 で詳細を述べる。

- Sec. 3.3 completes our virtual try-on pipeline with a characteristic-preserving Try-On Module.
    - セクション 3.3 では、特性を保存する Try-On Module での我々の仮想試着パイプラインを完成する。

- The Try-On module synthesizes final try-on results I_o by fusing the warped clothes c^ and the rendered person image I_r.
    - The Try-On module は、歪んだ服 c^ とレンダリングされた人物画像 I_r を融合する [fusing] することによって、最終的な試着結果 I_o を合成し、

- The overall pipeline is depicted in Fig. 2.
    - パイプラインの全体は、図に２図示されている。

### 3.1 Person Representation

- The original cloth-agnostic person representation [10] aims at leaving out the effiects of old clothes c_i like its color, texture and shape, while preserving information of input person I_i as much as possible, including the person's face, hair, body shape and pose.
    - 元の服にとらわれない人物表現は、古い服 c_i の色やテクスチャー、形状のような効果を除外する [leaving out] ことを狙いとしている。
    - 一方で、人物の顔や髪、体型や姿勢を含んでいる、入力人物 I_i の情報を、出来る限り保存することを狙いとしている。

- It contains three components:

- 1. Pose heatmap: an 18-channel feature map with each channel corresponding to one human pose keypoint, drawn as an 11 × 11 white rectangle.
    - 1. **姿勢ヒートマップ：各チャンネルが人間の姿勢キーポイントに一致している 18 チャンネルの特徴マップで、11 × 11 の白い四角系で描写される。**

- 2. Body shape: a 1-channel feature map of a blurred binary mask that roughly covering diffierent parts of human body.
    - **2. 体型：人間の異なる体の部分をおおまかにカバーするような、ぼやけた２値マスクの１チャンネルの特徴マップ**

> くっきりしたマスク画像だと、その部分のみに服が合成され、多様性のない合成画像となるため、あえてぼやけたマスク画像とすることで、服の合成の多様性を確保する。

- 3. Reserved regions: an RGB image that contains the reserved regions to maintain the identity of a person, including face and hair.
    - **3. 保存された領域：顔や髪を含む人物のアイデンティティを維持するための保存された領域を含むような１つの RGB 画像。**

- These feature maps are all scaled to a fixed resolution 256×192 and concatenated together to form the cloth-agnostic person representation map p of k channels, where k = 18 + 1 + 3 = 22.
    - これらの特徴マップは、全て解像度 256×192 で固定してスケーリングされ、k 個のチャンネルをもつ衣服にとらわれない人物表現写像 p の形にお互いが結合される。ここで、 k= 18 + 1 + 3 = 22

- We also utilize this representation in both our matching module and try-on module.
    - 我々はまた、我々のマッチングモジュールと try-on module との両方で、この表現を利用する。

---

![image](https://user-images.githubusercontent.com/25688193/58460839-5b9bd800-8169-11e9-968a-d7bb4e1fff0c.png)


### 3.2 Geometric Matching Module

- The classical approach for the geometry estimation task of image matching consists of three stages:
    - 画像マッチングの形状推定タスクのための、従来のアプローチは、以下の３つののステージから構成される。

- (1) local descriptors (e.g. shape context [2], SIFT [22] ) are extracted from both input images,
    - (1) 局所的な記述子 [descriptor] が、両方の入力画像から抽出される。

- (2) the descriptors are matched across images form a set of tentative correspondences,
    - (2) その記述子が、複数の画像フォームに渡ってマッチングされ、試験的な [tentative] 対応の組を形成する。

- (3) these correspondences are used to robustly estimate the parameters of geometric model using RANSAC [7] or Hough voting [16,22].
    - これらの対応は、RANSAC [7] や Hough voting [16,22] を使用している形状モデルのパラメーターを頑丈に [robustly] 推定するために、使用される。

---

- Rocco et al. [27] mimics this process using diffierentiable modules so that it can be trainable end-to-end for geometry estimation tasks.
    - Rocco ら [27] は、形状推定タスクに対して、end-to-end で学習可能になるように、微分可能な [diffierentiable] モジュールを使用して、このプロセスを模倣している [mimics]。

- Inspired by this work, we design a new Geometric Matching Module (GMM) to transform the target clothes c into warped clothes c^ which is roughly aligned with input person representation p.
    - この研究に触発され、我々は、目標の服 c を、入力の人物表現 p で大まかに整形された歪んだ服 c^ に変換するための新しい Geometric Matching Module (GMM)を設計する。

- As illustrated in Fig. 2, our GMM consists of four parts:
    - 図２に図示されているように、我々の GMM は４つの部分で構成する。

- (1) two networks for extracting high-level features of p and c respectively.
    - (1) p と c の比較的高レベルな特徴量を抽出するためのネットワーク。

- (2) a correlation layer to combine two features into a single tensor as input to the regressor network.
    - (2) ２つの特徴量を、regressor network の入力としての１つのテンソルに結合する correlation layer（相関層） 

- (3) the regression network for predicting the spatial transformation parameters θ.
    - (3) 空間的な変換パラメーター θ を推定するための regression network

- (4) a Thin-Plate Spline (TPS) transformation module T for warping an image into the output ^c = T_θ(c).
    - (4) 画像を、出力 ^c = T_θ(c) に歪ますための Thin-Plate Spline (TPS) 変換モジュール T 
     
- The pipeline is end-to-end learnable and trained with sample triplets (p, c, c_t), under the pixel-wise L1 loss between the warped result ^c and ground truth c_t, where c_t is the clothes worn on the target person in I_t:
    - このパイプラインが、end-to-end に学習可能であり、
    - 歪んだ結果 c^ と ground truth c_t との間のピクセル単位でのL1損失の元で、３つの組のサンプル (p, c, c_t) で学習される。
    - ここで、c_t は、I_t の中の目標人物が来ている服である。

![image](https://user-images.githubusercontent.com/25688193/58463396-b84dc180-816e-11e9-8f45-3b490c971527.png)

---

- The key diffierences between our approach and Rocco et al. [27] are three-fold.
    - 我々のアプローチと Rocco らによるアプローチの間の、重要な違いは、以下の３つである。

- First, we trained from scratch rather than using a pretrained VGG network.
    - １つ目は、事前学習された VGG network を使用するのではなく、スクラッチで学習した。

- Second, our training ground truths are acquired from wearer's real clothes rather than synthesized from simulated warping.
    - **２つ目は、我々の学習している ground truths は、歪みシミュレートからの合成されたものではなく、着用者のリアルな服から獲得されるものである。**

- Most importantly, our GMM is directly supervised under pixel-wise L1 loss between warping outputs and ground truth.
    - 最も重要なことは、我々の GMM は、歪み出力と ground truth との間の ピクセル単位でのL1損失の元で、直接的に監視されている [supervised] ことである。


### 3.3 Try-on Module

- Now that the warped clothes ^c is roughly aligned with the body shape of the target person, the goal of our Try-On module is to fuse ^c with the target person and for synthesizing the final try-on result.
    - 今や歪んだ服 c^ は、目標の人の体型で、大雑把に整形されているので、
    - 我々の Try-On module のゴールは、c^ を目標の人と混合することであり、最終的な試着結果を合成することである。

---

- One straightforward solution is directly pasting ^c onto target person image I_t.
    - １つの単純な解決法は、c^ を目標の人の画像 I_t に、直接的に貼り付けることである。

- It has the advantage that the characteristics of warped clothes are fully preserved, but leads to an unnatural appearance at the boundary regions of clothes and undesirable occlusion of some body parts (e.g. hair, arms).
    - これは、歪んだ服の特性が、完全に保存されるという利点をもつ。
    - しかし、服の領域の境界で不自然な外見と、いくつかの体の部分（例えば、髪や腕）で、望ましくない閉塞 [occlusion] を導く。

- Another solution widely adopted in conditional image generation is translating inputs to outputs by a single forward pass of some encoder-decoder networks, such as UNet [28], which is desirable for rendering seamless smooth images.
    - 条件付き画像生成において、広く適用される他の解決法は、UNet のようないくつかの encoder-decoder ネットワークの単一の順方向経路によって、入力を出力に変換することである。
    - （この encoder-decoder ネットワークというのは、）シームレスでスムーズな画像のために、望ましい（ようなネットワーク）

- However, It is impossible to perfectly align clothes with target body shape.
    - しかしながら、目標の体型に、服を完全に整形することは不可能である。

- Lacking explicit spatial deformation ability, even minor misalignment could make the UNet-rendered output blurry.
    - 空間的な変形の能力を欠けば、小さな不整合さえ、U-Net でレンダリングされた出力をぼやけさせるだろう。

---

![image](https://user-images.githubusercontent.com/25688193/58522089-c72d8600-81f9-11e9-9b8c-23ee29a43e75.png)

---

- Our Try-On Module aims to combine the advantages of both approaches above.
    - 我々の Try-On-Module は、上の両方のアプローチの利点を組み合わせることを狙いとしている。

- As illustrated in Fig. 2, given a concatenated input of person representation p and the warped clothes ^c, UNet simultaneously renders a person image I_r and predicts a composition mask M.
    - 図２で図示したように、人物表現 p の結合された入力と歪んだ服 c^ を与えれば、
    - UNet は同時に、人物画像 I_r を描写し、構成マスク M を推定する。

- The rendered person I_r and the warped clothes ^c are then fused together using the composition mask M to synthesize the final try-on result I_o:
    - レンダリングされた人物 I_r と歪んだ服 c^ はそのとき、最終的な試着結果 I_o を合成するために、構成マスク M を用いて、お互いを融合する。

![image](https://user-images.githubusercontent.com/25688193/58522324-ba5d6200-81fa-11e9-8128-90548d1b5fd0.png)

- where  represents element-wise matrix multiplication.
    - ここで、![image](https://user-images.githubusercontent.com/25688193/58522350-e5e04c80-81fa-11e9-9bce-da9483752ab0.png) は、要素単位での行列の積を表している。

---

- At training phase, given the sample triples (p, c, I_t), the goal of Try-On Module is to minimize the discrepancy between output I_o and ground truth I_t.
    - 学習フェイスでは、３つのサンプルの組 (p, c, I_t) を与えれば、
    - Try-On-Module のゴールは、出力 I_o と ground truth I_t の間の 不一致 [discrepancy] を最小化することである。

- We adopted the widely used strategy in conditional image generation problem that using a combination of L1 loss and VGG perceptual loss [14], where the VGG perceptual loss is dened as follows:
    - 我々は、条件付き画像生成問題で広く使われている戦略を適用した。
    - （この戦略というのは、）L1損失と VGG の知覚的な損失 [14] の組み合わせを使用している（戦略）
    - ここで、VGG の知覚的な損失は、以下のように定義される。

![image](https://user-images.githubusercontent.com/25688193/58522611-22607800-81fc-11e9-9881-b61da1f9420a.png)

- where φ_i(I) denotes the feature map of image I of the i-th layer in the visual perception network φ, which is a VGG19 [32] pre-trained on ImageNet.
    - ここで、φ_i(I) は、視覚的なパーセプトロン φ の中での、i番目の層の画像 I の特徴マップを示しており、
    - これは、ImageNet で VGG19 で事前学習されたものである。

- The layer i ≧ 1 stands for 'conv1 2', 'conv2 2', 'conv3 2', 'conv4 2', 'conv5 2', respectively.
    - 層 i ≧ 1 は、それぞれ、'conv1 2', 'conv2 2', 'conv3 2', 'conv4 2', 'conv5 2' を表している。

---

- Towards our goal of characteristic-preserving, we bias the composition mask M to select warped clothes as much as possible by applying a L1 regularization ||1 - M||_1 on M.
    - 特性保存の我々のゴールに向けて、我々は、M においてのL1正則化 ||1 - M||_1 を適用することによって、出来るだけ歪んだ服を選択するために、構成マスク M にバイアスをかける。

- The overall loss function for Try-On Module (TOM) is:
    - Try-On Module (TOM) に対しての、損失関数の全体は、以下のように定義される。

![image](https://user-images.githubusercontent.com/25688193/58523256-bcc1bb00-81fe-11e9-9f99-d17de9701ab1.png)


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## 4 Experiments and Analysis

### 4.1 Dataset

- We conduct our all experiments on the datasets collected by Han et al. [10].
    - 我々は、Han らによって収集されたデータセットで、全ての実験を行う。

- It contains around 19,000 front-view woman and top clothing image pairs.
    - このデータセットは、正面を向いている女性と上着の服の画像のペアの、およそ 19,000 個のデータを含んでいる。

- There are 16253 cleaned pairs, which are split into a training set and a validation set with 14221 and 2032 pairs, respectively. 
    - 16253 個のクリーニングされたペアが存在する。
    - これは、それぞれ、12332 個のペアと 2032 個のペアで、学習用データセットと検証用データセットに分割されている。

- We rearrange the images in the validation set into unpaired pairs as the testing set.
    - 我々は、この検証用データセットの画像を、テスト用として、ペアリングされていないペアに再度整形する。

### 4.2 Quantitative Evaluation

- We evaluate the quantitative performance of diffierent virtual try-on methods via a human subjective perceptual study.
    - 我々は、人間の主観的で知覚的な研究経由で、仮想試着手法の違いの定量的なパフォーマンスを評価する。

- Inception Score (IS) [29] is usually used as to quantitatively evaluate the image synthesis quality, but not suitable for evaluating this task for that it cannot reflect whether the details are preserved as described in [10].
    - Inception Score (IS) [29] は、画像合成の質を定量的に評価するためのものとして、一般的に使用されている。
    - しかし、詳細が、データセット [10] で記述されているように保存されているかどうかを反映することが出来ないために、このタスクを評価するためには、ふさわしくない。

- We focus on the clothes with rich details since we are interested in characteristic-preservation, instead of evaluating on the whole testing set.
    - 我々は、テストデータセット全体を評価するのではなく、特性の保存に興味があるので、豊富な詳細をもつ服にフォーカスする。

- For simplicity, we measure the detail richness of a clothing image by its total variation (TV) norm.
    - 簡単のため、total variation (TV) ノルムによって、我々は服の画像の細分の豊富さを計測する。

- It is appropriate for this dataset since the background is in pure color and the TV norm is only contributed by clothes itself, as illustrated in Fig. 3.
    - 図３に図示されているように、
    - このデータセットに対しては、背景が純粋な色になっており、TV ノルムは服自身によってのみ貢献されるため、適切である。

- We extracted 50 testing pairs with largest clothing TV norm named as LARGE to evaluate characteristic-preservation of our methods, and 50 pairs with smallest TV norm named as SMALL to ensure that our methods perform at least as good as previous state-of-the-art methods in simpler cases.
    - 我々は 我々の手法の特性保存性を評価するために、LARGE と名付けられた、最も大きな服の TV ノルムを持つような、50 個のテストペアを抽出した。
    - そして、我々の手法が、よりシンプルなケースにおいて、少なくとも前の SOTA 手法と同様によいパフォーマンスを行うことを保証するために、SMALL と名付けられた、最も小さい TV ノルムをもつような、50 個のペアを抽出した。

---

![image](https://user-images.githubusercontent.com/25688193/58524854-8dae4800-8204-11e9-8ba1-6dc667eaa93a.png)

- Fig. 3. From top to bottom, the TV norm values are increasing. 
    - 上段から下段にかけて、TV ノルムは増加している。

- Each line shows some clothes in the same level.
    - 各行は、同じレベルの同じ服を示している。

---

- We conducted pairwise A/B tests on Amazon Mechanical Turk (AMT) platform.
    - 我々は、Amazon Mechanical Turk (AMT) で、ピクセル単位での A/B テストを行った。

- Specically, given a person image and a target clothing image, the worker is asked to select the image which is more realistic and preserves more details of the target clothes between two virtual try-on results from diffierent methods.
    - 特に、人物画像と目標の服画像を与えれば、作業者は、異なる手法からの２つの仮想試着結果との間で、よりリアルでより目標の服の詳細を保存している画像を選択することを求められる。

- There is no time limited for these jobs, and each job is assigned to 4 diffierent workers.
    - これらの処理の時間的制限は存在せず、各処理は、４つの作業者に割り与えられる。

- Human evaluation metric is computed in the same way as in [10].
    - 人間的な評価指標は、[10] と同じ方法で計算される。

### 4.3 Implementation Details

#### Training Setup 

![image](https://user-images.githubusercontent.com/25688193/58523256-bcc1bb00-81fe-11e9-9f99-d17de9701ab1.png)

- In all experiments, we use λ_L1 = λ_vgg = 1.
    - 全ての実験において、我々は、λ_L1 = λ_vgg = 1 を使用する。

- When composition mask is used, we set λ_mask = 1.
    - 構成マスクが使用されるとき、λ_mask = 1 を設定する。

- We trained both Geometric Matching Module and Try-on Module for 200K steps with batch size 4.
    - GMM と TOM の両方を、バッチサイズ 4 で 200K ステップ学習した。

- We use Adam [15] optimizer with β_1 = 0.5 and β_2 = 0.999.
    - β_1 = 0.5 and β_2 = 0.999 で Adam を使用する。

- Learning rate is first fixed at 0.0001 for 100K steps and then linearly decays to zero for the remaining steps.
    - 学習率は、最初の 100K ステップで 0.0001 で固定され、
    - 次に、残りのステップで、ゼロに向かって線形に減衰する。

- All input images are resized to 256 × 192 and the output images have the same resolution.
    - 全ての入力画像は、256 × 192 にリサイズされ、出力画像は同じ解像度を持つ。

#### Geometric Matching Module

![image](https://user-images.githubusercontent.com/25688193/58526055-c2bc9980-8208-11e9-91de-282663c33179.png)

- Feature extraction networks for person representation and clothes have the similar structure, containing four 2-strided down-sampling convolutional layers, succeeded by two 1-strided ones, their numbers of lters being 64, 128, 256, 512, 512, respectively.

- The only diffierence is the number of input channels.

- Regression network contains two 2-strided convolutional layers, two 1-strided ones and one fully-connected output layer.

- The numbers of lters are 512, 256, 128, 64.

- The fully-connected layer predicts the x- and y-coordinate offsets of TPS anchor points, thus has an output size of 2×5×5 = 50.

#### Try-On Module

- xxx

### 4.4 Comparison of Warping Results


### 4.5 Comparison of Try-on Results


### 4.6 Discussion and Ablation Studies


# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


