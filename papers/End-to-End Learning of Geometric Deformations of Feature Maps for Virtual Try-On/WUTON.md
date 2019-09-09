# ■ 論文
- 論文タイトル："End-to-End Learning of Geometric Deformations of Feature Maps for Virtual Try-On"
- 論文リンク：
- 論文投稿日付：2019/06/04
- 被引用数（記事作成時点）：xxx 件
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- The 2D virtual try-on task has recently attracted a lot of interest from the research community, for its direct potential applications in online shopping as well as for its inherent and non-addressed scientific challenges. This task requires to fit an in-shop cloth image on the image of a person. It is highly challenging because it requires to warp the cloth on the target person while preserving its patterns and characteristics, and to compose the item with the person in a realistic manner. Current state-of- the-art models generate images with visible artifacts, due either to a pixel-level composition step or to the geometric transformation.
    - 2Dの仮想試着タスクは、オンラインショッピングでの直接的な潜在的なアプリケーションや、固有の未解決の科学的課題のために、最近、研究コミュニティから多くの関心を集めています。このタスクでは、店内の布の画像を人物の画像に合わせる必要があります。パターンと特性を保持しながら対象の人物に布を反らせ、現実的な方法で人物とアイテムを構成する必要があるため、非常に困難です。現在の最先端のモデルは、ピクセルレベルの合成ステップまたは幾何学的変換のいずれかにより、目に見えるアーチファクトのある画像を生成します。
    
-  In this paper, we propose WUTON: a Warping U-net for a Virtual Try-On system. It is a siamese U-net generator whose skip connections are geometrically transformed by a convolutional geometric matcher. The whole architecture is trained end-to-end with a multi-task loss including an adversarial one. This enables our network to generate and use realistic spatial transformations of the cloth to synthesize images of high visual quality. The proposed architecture can be trained end-to-end and allows us to advance towards a detail-preserving and photo-realistic 2D virtual try-on system. Our method outperforms the current state-of-the-art with visual results as well as with the Learned Perceptual Image Similarity (LPIPS) metric.
    - この論文では、WUTON：Virtual Try-Onシステム用のワーピングU-netを提案します。これは、スキップ接続が畳み込み幾何学的マッチャーによって幾何学的に変換されるシャムU-netジェネレーターです。アーキテクチャ全体は、敵対的なものを含むマルチタスク損失でエンドツーエンドでトレーニングされます。これにより、ネットワークで布のリアルな空間変換を生成および使用して、高品質の画像を合成できます。提案されたアーキテクチャは、エンドツーエンドでトレーニングすることができ、詳細を維持し、写真のようにリアルな2D仮想試着システムに向かって進むことができます。私たちの方法は、視覚的な結果だけでなく、学習した知覚画像の類似性（LPIPS）メトリックで現在の最先端技術よりも優れています。


# ■ イントロダクション（何をしたいか？）

## 1. Introduction

- A photo-realistic virtual try-on system would be a significant improvement for online shopping. Whether used to create catalogs of new products or to propose an immersive environment for shoppers, it could impact e-shop and open the door for new easy image editing possibilities. The training data we consider is made of paired images that are made of the picture of one cloth and the same cloth worn by a model. Then providing an unpaired tuple of images: one picture of cloth and one picture of a model with a different cloth, we aim to replace the cloth worn by the model.
    - フォトリアリスティックな仮想試着システムは、オンラインショッピングの大幅な改善になります。 新製品のカタログを作成したり、買い物客に没入感のある環境を提案したりすると、eショップに影響を与え、新しい簡単な画像編集の可能性を開く可能性があります。 検討するトレーニングデータは、1枚の布とモデルが着用している同じ布の写真で作成されたペアの画像で作成されています。 次に、1枚の布の画像と1枚のモデルの画像を別の布で、ペアになっていない画像のタプルを提供し、モデルが着用している布の交換を目指します。

---

- An early line of work addressed this challenge using 3D measurements and model-based methods [1, 2, 3]. However, these are by nature computationally intensive and require expensive material, which would not be acceptable at scale for shoppers. Recent works aim to leverage deep generative models to tackle the virtual try-on problem [4, 5, 6, 7]. CAGAN [4] proposes a U-net based Cycle- GAN [8] approach. However, this method fails to generate realistic results since these networks cannot handle large spatial deformations. In VITON [5], the authors use the shape context matching algorithm [9] to warp the cloth on a target person and learn an image composition with a U-net generator. To improve this model, CP-VTON [6] incorporates a convolutional geometric matcher [10] which learns the parameters of geometric deformations (i.e thin-plate spline transform [11]) to align the cloth with the target person. In MG-VTON [7], the task is extended to a multi-pose try-on system, which requires to modify the pose as well as the upper-body cloth of the person.
    - 初期の一連の作業では、3D測定とモデルベースの方法を使用してこの課題に対処しました[1、2、3]。しかし、これらは本質的に計算集約的であり、高価な材料を必要とし、買い物客にとって大規模には受け入れられません。最近の研究は、仮想の試着問題に取り組むために、深い生成モデルを活用することを目指しています[4、5、6、7]。 CAGAN [4]は、U-netベースのCycle-GAN [8]アプローチを提案しています。ただし、これらのネットワークは大きな空間変形を処理できないため、この方法では現実的な結果を生成できません。 VITON [5]では、著者は形状コンテキストマッチングアルゴリズム[9]を使用して、対象者に布をゆがめ、U-netジェネレーターで画像合成を学習します。このモデルを改善するために、CP-VTON [6]は、幾何学的変形のパラメーターを学習する畳み込み幾何学的マッチャー[10]（つまり、薄板スプライン変換[11]）を組み込んで、対象の人物に布を合わせます。 MG-VTON [7]では、タスクはマルチポーズの試着システムに拡張されており、ポーズと人の上半身の布を修正する必要があります。

---

- In this second line of approach, a common practice is to use what we call a human parser which is a pre-trained system able to segment the area to replace on the model pictures: the upper-body cloth as well as neck and arms. In the rest of this work, we also assume this parser to be known.
    - この2番目のアプローチでは、一般的なプラクティスは、モデル画像で置き換える領域をセグメント化できる事前トレーニング済みシステムであるヒューマンパーサーと呼ばれるものを使用することです：上半身の布と首と腕。 この作業の残りの部分では、このパーサーが既知であると想定しています。

---

- The recent methods for a virtual try-on struggle to generate realistic spatial deformations, which is necessary to warp and render clothes with complex patterns. Indeed, with solid color tees, unrealistic deformations are not an issue because they are not visible. However, for a tee-shirt with stripes or patterns, it will produce unappealing images with curves, compressions and decompressions. Figure 4 shows these kinds of unrealistic geometric deformations generated by CP-VTON [6].
    - **仮想の試着のための最近の方法は、複雑なパターンの衣服をゆがめてレンダリングするために必要な現実的な空間変形を生成するのに苦労しています。 実際、ソリッドカラーのティーでは、非現実的な変形は目に見えないため問題になりません。 ただし、ストライプまたはパターンのあるTシャツの場合、曲線、圧縮、および解凍を含む魅力のない画像が生成されます。 図4は、CP-VTON [6]によって生成されたこれらの種類の非現実的な幾何学的変形を示しています。**

---

- To alleviate this issue, we propose an end-to-end model composed of two modules: a convolutional geometric matcher [10] and a siamese U-net generator. We train end-to-end, so the geometric matcher benefits from the losses induced by the final synthetic picture generation. Our architecture removes the need for a final image composition step and generates images of high visual quality with realistic geometric deformations. Main contributions of this work are:
    - **この問題を軽減するために、畳み込み幾何マッチャー[10]とシャムUネットジェネレーターの2つのモジュールで構成されるエンドツーエンドモデルを提案します。 エンドツーエンドでトレーニングを行うため、ジオメトリックマッチャーは、最終的な合成画像の生成によって生じる損失の恩恵を受けます。 私たちのアーキテクチャは、最終的な画像合成ステップの必要性を取り除き、現実的な幾何学的変形を伴う高視覚品質の画像を生成します。 この作業の主な貢献は次のとおりです。**

- We propose a simple end-to-end architecture able to generate realistic deformations to preserve complex patterns on clothes such as stripes. This is made by back-propagating the loss computed on the final synthesized images to a learnable geometric matcher.
    - ストライプなどの衣服の複雑なパターンを維持するために、現実的な変形を生成できるシンプルなエンドツーエンドアーキテクチャを提案します。 これは、最終的な合成画像で計算された損失を学習可能な幾何学的マッチャーに逆伝播することによって行われます。

- We suppress the need for a final composition step present in the best current approaches such as [6] using an adversarially trained generator. This performs better on the borders of the replaced object and provides a more natural look of shadows and contrasts.
    - 敵対的に訓練されたジェネレーターを使用する[6]など、現在の最良のアプローチに存在する最終的な構成ステップの必要性を抑制します。 これにより、置き換えられたオブジェクトの境界線でのパフォーマンスが向上し、より自然な影とコントラストの外観が得られます。

- We show that our approach significantly outperforms the state-of-the-art with visual results and with a quantitative metric measuring image similarity, LPIPS [12].
    - 私たちのアプローチは、視覚的な結果と、画像の類似性を測定する定量的メトリックLPIPS [12]で、最先端技術を大幅に上回ることを示しています。

- We identify the contribution of each part of our net by an ablation study. Moreover, we exhibit a good resistance to low-quality human parser at inference time.
    - アブレーション研究により、ネットの各部分の寄与を特定します。 さらに、推論時に低品質の人間のパーサーに対して良好な耐性を示します。

# ■ 結論

## x. Conclusion


# ■ 何をしたか？詳細

## 3 Our approach

- Our task is to build a virtual try-on system that is able to fit a given in-shop cloth on a reference person. We propose a novel architecture trainable end-to-end and composed of two existing modules, i.e. a convolutional geometric matcher STN [10] and a U-net [27] generator G whose skip connections are deformed by STN. The joint training of G and STN allows us to generate realistic deformations that help to synthesize high-quality images. Also, we use an adversarial loss to make the training procedure closer to the actual use of the system which is to replace clothes in the unpaired situation. In previous works [5, 6, 7], the generator is only trained to reconstruct images with supervised triplets (ap, c, p) extracted from the paired. Thus, when generating images in the test-setting, it can struggle to generalize and to warp clothes different from the one worn by the reference person. The adversarial training allows us to train our network in the test-setting, where one wants to fit a cloth on a reference person wearing another cloth.
    - 私たちのタスクは、特定の店内の布を参照者にフィットさせることができる仮想試着システムを構築することです。エンドツーエンドでトレーニング可能な2つの既存のモジュール、つまり畳み込み幾何学的マッチャーSTN [10]と、STNによってスキップ接続が変形されるU-net [27]ジェネレーターGで構成される、新しいアーキテクチャを提案します。 
    - GとSTNの共同トレーニングにより、高品質の画像を合成するのに役立つ現実的な変形を生成できます。また、敵対的な損失を使用して、ペアリングされていない状況で衣服を交換するというシステムの実際の使用により近いトレーニング手順を作成します。**以前の作品[5、6、7]では、ジェネレーターは、ペアリングから抽出された教師付きトリプレット（ap、c、p）で画像を再構成するようにのみ訓練されています。したがって、テスト設定で画像を生成する場合、参照者が着用する服とは異なる服を一般化して反らせるのに苦労する可能性があります。敵対的なトレーニングにより、テスト設定でネットワークをトレーニングすることができます。テスト設定では、別の布を着ている参照者に布をフィットさせたいと考えています。**

### 3.1 Warping U-net

- Our warping U-net is composed of two connected modules, as shown in Fig.1. The first one is a convolutional geometric matcher, which has a similar architecture as [10, 6]. It outputs the parameters θ of a geometric transformation, a TPS transform in our case. This geometric transformation aligns the in-shop cloth image with the reference person. However, in contrast to previous work [5, 6, 7], we use the geometric transformation on the features maps of the generator rather than at a pixel-level. Thus, we learn to deform the feature maps that pass through the skip connections of the second module, a U-net [27] generator which synthesizes the output image p ̃.
    - ワーピングU-netは、図1に示すように、2つの接続されたモジュールで構成されています。 最初のものは、[10、6]と同様のアーキテクチャを持つ畳み込み幾何学的マッチャーです。 幾何学的変換のパラメーターθ、この場合はTPS変換を出力します。 この幾何学的変換は、店内の布の画像を参照者に合わせます。 **ただし、以前の研究[5、6、7]とは対照的に、ピクセルレベルではなく、ジェネレーターの機能マップで幾何学的変換を使用します。** したがって、出力画像p synthesizeを合成するU-net [27]ジェネレーターである2番目のモジュールのスキップ接続を通過する特徴マップを変形することを学びます。

---

- The architecture of the convolutional geometric matcher is taken from CP-VTON [6], which reuses the generic geometric matcher from [10]. It is composed of two feature extractors F1 and F2, which are standard convolutional neural networks. The local vectors of feature maps F1(c) and F2(ap) are then L2-normalized and a correlation map C is computed as follows :
    - 畳み込み幾何学的マッチャーのアーキテクチャはCP-VTON [6]から取得され、CP-VTONは[10]からの汎用幾何学的マッチャーを再利用します。 これは、標準の畳み込みニューラルネットワークである2つの特徴抽出器F1およびF2で構成されています。 特徴マップF1（c）およびF2（ap）のローカルベクトルはL2正規化され、相関マップCは次のように計算されます。

> 式

- where k is the index for the position (m, n). This correlation map captures dependencies between distant locations of the two feature maps, which is useful to align the two images. C is the input of a regression network, which outputs the parameters θ and allows to perform the geometric transformation Tθ. We use TPS transformations [11], which generate smooth sampling grids given control points. Since we transform deep feature maps of a U-net generator, we generate a sampling grid for each scale of the U-net with the same parameters θ.
    - ここで、kは位置（m、n）のインデックスです。 この相関マップは、2つの機能マップの離れた場所間の依存関係をキャプチャします。これは、2つの画像を整列させるのに役立ちます。 Cは回帰ネットワークの入力であり、パラメータθを出力し、幾何学的変換Tθを実行できます。 TPS変換[11]を使用します。これは、制御点を指定して滑らかなサンプリンググリッドを生成します。 U-netジェネレーターのディープフィーチャマップを変換するため、同じパラメーターθを使用してU-netの各スケールのサンプリンググリッドを生成します。

---

- The input of the U-net generator is also the tuple of pictures (ap, c). Since these two images are not spatially aligned, we cannot simply concatenate them and feed a standard U-net. To alleviate this, we use two different encoders E1 and E2 processing each image independently and with non-shared parameters. Then, the feature maps of the in-shop cloth E1(c) are transformed at each scale i: E1i(c) = Tθ(E1i(c)). Then, the feature maps of the two encoders are concatenated and feed the decoder. With aligned feature maps, the generator is able to compose them and to produce realistic results. Because we simply concatenate the feature maps and let the U-net decoder compose them instead of enforcing a pixel-level composition, experiments will show that it has more flexibility and can produce more natural results. We use instance normalization in the U-net generator, which is more effective than batch normalization [32] for image generation [33].
    - U-netジェネレーターの入力は、画像のタプル（ap、c）でもあります。 これらの2つの画像は空間的に整列していないため、単純にそれらを連結して標準のU-netにフィードすることはできません。 これを軽減するために、2つの異なるエンコーダーE1およびE2を使用して、各画像を独立して、非共有パラメーターで処理します。 次に、店内の布E1（c）の特徴マップが各スケールiで変換されます：E1i（c）=Tθ（E1i（c））。 次に、2つのエンコーダーの機能マップが連結され、デコーダーに供給されます。 整列された機能マップにより、ジェネレーターはそれらを構成し、現実的な結果を生成できます。 機能マップを単純に連結し、ピクセルレベルの構成を強制するのではなくU-netデコーダーに構成させるため、実験により柔軟性が増し、より自然な結果が得られることがわかります。 U-netジェネレーターでインスタンス正規化を使用します。これは、画像生成[33]のバッチ正規化[32]よりも効果的です。

### 3.2 Training procedure

- Along with a new architecture for the virtual try-on task (Fig. 1), we also propose a new training procedure, i.e. a different data representation and an adversarial loss for unpaired images.
    - 仮想試着タスクの新しいアーキテクチャ（図1）とともに、新しいトレーニング手順、つまり、異なるデータ表現と不対画像の敵対的損失も提案します。

---

- > Figure1: WUTON:our proposed end-to-end warping U-net architecture. Dotted arrows correspond to the forward pass only performed during training. Green arrows are the human parser. The geometric transforms share the same parameters but do not operate on the same spaces. The different training procedure for paired and unpaired pictures is explained in section 3.2
    - > 図1：WUTON：提案されたエンドツーエンドのワーピングU-netアーキテクチャ。 点線の矢印は、トレーニング中にのみ実行されるフォワードパスに対応しています。 緑の矢印は人間のパーサーです。 幾何学的変換は同じパラメーターを共有しますが、同じ空間では機能しません。 ペア画像と非ペア画像の異なるトレーニング手順については、セクション3.2で説明しています。

---


- While previous works use a rich person representation with more than 20 channels representing human pose, body shape and the RGB image of the head, we only mask the upper-body of the reference person. Our agnostic person representation ap is thus a 3-channel RGB image with a masked area. We compute the upper-body mask from pose and body parsing information provided by a pre-trained neural network from [34]. Precisely, we mask the areas corresponding to the arms, the upper-body cloth and a fixed bounding box around the neck keypoint. However, we show in an ablation study that our method is not sensitive to non-accurate masks at inference time since it can generate satisfying images with simple bounding box masks.
    - 以前の作品では、人間のポーズ、体の形、頭部のRGB画像を表す20を超えるチャネルを持つ豊かな人物表現を使用していましたが、参照人物の上半身のみをマスクします。 したがって、私たちの不可知論者表現apは、マスクされた領域を持つ3チャンネルRGB画像です。 [34]から事前に訓練されたニューラルネットワークによって提供されるポーズおよび身体解析情報から上半身マスクを計算します。 正確には、腕、上半身の布、首のキーポイントの周りの固定境界ボックスに対応する領域をマスクします。 ただし、アブレーション研究では、単純なバウンディングボックスマスクで満足のいく画像を生成できるため、推論時に不正確なマスクの影響を受けないことを示しています。

---

- Using the dataset from [7], we have pairs of in-shop cloth image ca and a person wearing the same cloth pa. Using a human parser and a human pose estimator, we generate ap. From the parsing information, we can also isolate the cloth on the image pa and get ca,p, the cloth worn by the reference person. Moreover, we get the image of another in-shop cloth, cb. The inputs of our network are the two tuples (ap, c ) and (ap, c ). The outputs are respectively (p ̃ , θ ) and (p ̃ , θ ).
    - [7]のデータセットを使用して、店内の布の画像caと同じ布paを着た人物のペアがあります。 人間のパーサーと人間の姿勢推定器を使用して、apを生成します。 解析情報から、画像pa上の布を分離し、参照者が着用している布ca、pを取得することもできます。 さらに、別の店内の布、cbのイメージを取得します。 ネットワークの入力は、2つのタプル（ap、c）と（ap、c）です。 出力はそれぞれ（p ̃、θ）および（p ̃、θ）です。

---

- The cloth worn by the person ca,p allows us to guide directly the geometric matcher with a L1 loss:
    - 人ca、pが着用する布により、幾何学的マッチャーをL1損失で直接誘導できます。

> 式

- The image p of the reference person provides a supervision for the whole pipeline. Similarly to CP- VTON [6], we use two different losses to guide the generation of the final image p ̃a, the pixel-level L1 loss ∥p ̃a − pa∥1 and the perceptual loss [35]. We focus on L1 losses since they are known to generate less blur than L2 for image generation [36]. The latter consists of using the features extracted with a pre-trained neural network, VGG [37] in our case. Specifically, our perceptual loss is:
    - 参照者の画像pは、パイプライン全体の監視を提供します。 CP-VTON [6]と同様に、2つの異なる損失を使用して、最終画像p ̃a、ピクセルレベルL1損失∥p̃a-pa∥1および知覚損失[35]の生成をガイドします。 L1損失は、画像生成のL2よりもぼやけが少ないことが知られているため、焦点を当てています[36]。 後者は、事前学習済みのニューラルネットワーク（この場合はVGG [37]）で抽出された特徴を使用することで構成されます。 具体的には、知覚損失は次のとおりです。

> 式

- where φi(I) are the feature maps of an image I extracted at the i-th layer of the VGG network.
    - ここで、φi（I）は、VGGネットワークのi番目のレイヤーで抽出した画像の特徴マップです。 

- Furthermore, we exploit adversarial training to train the network to fit cb on the same agnostic person representation ap, which is extracted from a person wearing ca. This is only feasible with an adversarial loss, since there is no available ground-truth for this pair (ap,cb). Thus, we feed the discriminator with the synthesized image p ̃b and real images of persons from the dataset. This adversarial loss is also back-propagated to the convolutional geometric matcher, which allows to generate much more realistic spatial transformations. We use the relativistic adversarial loss [38] with gradient-penalty [39, 40], which trains the discriminator to predict relative realness of real images compared to synthesized ones. Finally, the objective function of our network is :
    - さらに、敵対的学習を活用して、caを着ている人から抽出された同じ不可知論者の表現apにcbが適合するようにネットワークをトレーニングします。 これは、このペア（ap、cb）に利用可能なグラウンドトゥルースがないため、敵対的損失でのみ実行可能です。 したがって、合成画像p ̃bとデータセットからの人物の実際の画像を弁別器に供給します。 この敵対的損失は畳み込み幾何学的マッチャーにも逆伝播され、より現実的な空間変換を生成できます。 相対論的敵対損失[38]と勾配ペナルティ[39、40]を使用します。これは、合成画像と比較した実際の画像の相対的な現実性を予測するために弁別器をトレーニングします。 最後に、ネットワークの目的関数は次のとおりです。


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. Related Work


