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
    - 一般的に、服 c_i と目標の服 c を着ている人物の参照画像 I_i を与えれば、
    - CP-VTON のゴールは、新しい服 c_o において、試着者の新しい画像 I_o を合成することである。
    - （この合成画像というのは、）体型や I_i の姿勢が保持され、目標の服 c の特性が保持され、古い服 c_i の効果が排除される。（ようなもの）

---

![image](https://user-images.githubusercontent.com/25688193/58453630-d6a7c300-8156-11e9-9b5c-c4426dd64897.png)

---

- Training with sample triplets (I_i, c, I_t) where I_t is the ground truth of I_o and c is coupled with I_t wearing in clothes c_t, is straightforward but undesirable in practice.
    - ３つの組のサンプル (I_i, c, I_t) での学習、
    - （ここで、I_t は、試着者の新しい画像 I_o の ground truth であり、目標の服 ｃ は I_t と衣服 c_t を着用することとを組み合わせる。）
    - ことは、簡単である [straightforward] が、実際には、望ましくない。

- Because these triplets are difficult to collect.
    - これらの３つの組み合わせは、収集するのが困難である。

- It is easier if I_i is same as I_t, which means that c, I_t pairs are enough.
    - もし I_i が I_t と同じであればより簡単である。

- These paris are in abundance from shopping websites.

- But directly training on (I_t, c, I_t) harms the model generalization ability at testing phase when only decoupled inputs (I_i, c) are available.

- Prior work [10] addressed this dilemma by constructing a clothing-agnostic person representation p to eliminate the effiects of source clothing item c_i.

- With (I_t, c, I_t) transformed into a new triplet form (p, c, I_t), training and testing phase are unied.

- We adopted this representation in our method and further enhance it by eliminating less information from reference person image.

- Details are described in Sec. 3.1.

- One of the challenges of image-based virtual try-on lies in the large spatial misalignment between in-shop clothing item and wearer's body.

- Existing network architectures for conditional image generation (e.g. FCN [21], UNet [28], ResNet [11]) lack the ability to handle large spatial deformation, leading to blurry try-on results.

- We proposed a Geometric Matching Module (GMM) to explicitly align the input clothes c with aforementioned person representation p and produce a warped clothes image c^.

- GMM is a end-to-end neural network directly trained using pixel-wise L1 loss.

- Sec. 3.2 gives the details.

- Sec. 3.3 completes our virtual try-on pipeline with a characteristic-preserving Try-On Module.

- The Try-On module synthesizes nal try-on results Io by fusing the warped clothes c^ and the rendered person image I_r.

- The overall pipeline is depicted in Fig. 2.


### 3.1 Person Representation

- xxx


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


