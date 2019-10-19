# ■ 論文
- 論文タイトル："Soft-Gated Warping-GAN for Pose-Guided Person Image Synthesis"
- 論文リンク：https://arxiv.org/abs/1810.11610
- 論文投稿日付：2018/10/27
- 被引用数（記事作成時点）：xxx 件
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- Despite remarkable advances in image synthesis research, existing works often fail in manipulating images under the context of large geometric transformations. Synthesizing person images conditioned on arbitrary poses is one of the most representative examples where the generation quality largely relies on the capability of identifying and modeling arbitrary transformations on different body parts.
    - 画像合成の研究は目覚しい進歩を遂げていますが、既存の研究では、大規模な幾何学的変換のコンテキストで画像を操作できないことがよくあります。 任意のポーズを条件とする人物画像の合成は、代表的な例の1つです。この例では、生成品質は、さまざまな身体部分の任意の変換を識別およびモデリングする機能に大きく依存します。

- Current generative models are often built on local convolutions and overlook the key challenges (e.g. heavy occlusions, different views or dramatic appearance changes) when distinct geometric changes happen for each part, caused by arbitrary pose manipulations. This paper aims to resolve these challenges induced by geometric variability and spatial displacements via a new Soft-Gated Warping Generative Adversarial Network (Warping-GAN), which is composed of two stages:
    - 現在の生成モデルは、多くの場合、局所的な畳み込みに基づいて構築され、任意のポーズ操作によって引き起こされる各パーツの異なる幾何学的変化が発生した場合の主要な課題（例えば、重度の閉塞、異なるビュー、劇的な外観の変化）を見落としています この論文は、2つのステージで構成される新しいSoft-Gated Warping Generative Adversarial Network（Warping-GAN）を介して、幾何学的な変動性と空間変位によって引き起こされるこれらの課題を解決することを目的としています。

- 1) it first synthesizes a target part segmentation map given a target pose, which depicts the region-level spatial layouts for guiding image synthesis with higher-level structure constraints;
    - 1）最初に、ターゲットポーズが指定されたターゲットパーツセグメンテーションマップを合成します。これは、より高いレベルの構造制約を持つ画像合成をガイドするための領域レベルの空間レイアウトを示します。

- 2) the Warping-GAN equipped with a soft-gated warping-block learns feature-level mapping to render textures from the original image into the generated segmentation map.
    - 2）soft-gated warping-block を備えた Warping-GAN は、特徴レベルのマッピングを学習して、元の画像から生成されたセグメンテーションマップにテクスチャをレンダリングします。

- Warping-GAN is capable of controlling different transformation degrees given distinct target poses. Moreover, the proposed warping-block is lightweight and flexible enough to be injected into any networks. Human perceptual studies and quantitative evaluations demonstrate the superiority of our WarpingGAN that significantly outperforms all existing methods on two large datasets.
    - Warping-GANは、明確なターゲットポーズが与えられると、異なる変換度を制御できます。 さらに、提案されているワーピングブロックは、任意のネットワークに挿入できるほど軽量で柔軟です。 人間の知覚研究と定量的評価は、2つの大きなデータセットで既存のすべての方法を大幅に上回るWarpingGANの優位性を示しています。


# ■ イントロダクション（何をしたいか？）

## x. Introduction

- Person image synthesis, posed as one of most challenging tasks in image analysis, has huge potential applications for movie making, human-computer interaction, motion prediction, etc. Despite recent advances in image synthesis for low-level texture transformations [13, 35, 14] (e.g. style or colors), the person image synthesis is particularly under-explored and encounters with more challenges that cannot be resolved due to the technical limitations of existing models. The main difficulties that affect the generation quality lie in substantial appearance diversity and spatial layout transformations on clothes and body parts, induced by large geometric changes for arbitrary pose manipulations. Existing models [20, 21, 28, 8, 19] built on the encoder-decoder structure lack in considering the crucial shape and appearance misalignments, often leading to unsatisfying generated person images.
    - 画像解析で最も困難なタスクの1つとして提示される人物画像合成は、映画制作、人間とコンピューターの相互作用、動き予測などに非常に大きな用途があります。低レベルのテクスチャ変換[13、35、 14]（スタイルや色など）、人物画像の合成は特に未開拓であり、既存のモデルの技術的な制限のために解決できないより多くの課題に直面しています。 生成品質に影響を与える主な問題は、任意のポーズ操作の大きな幾何学的変化によって引き起こされる、服装と身体部分の実質的な外観の多様性と空間レイアウトの変換にあります。 エンコーダー-デコーダー構造に基づいて構築された既存のモデル[20、21、28、8、19]は、重大な形状と外観の不整合を考慮することを欠いており、多くの場合、満足できない生成された人物画像につながります。

---

- Among recent attempts of person image synthesis, the best-performing methods (PG2 [20], Body- ROI7 [21], and DSCF [28]) all directly used the conventional convolution-based generative models by taking either the image and target pose pairs or more body parts as inputs.
    - 人物画像合成の最近の試みの中で、最高のパフォーマンスを発揮する方法 (PG2 [20], Body- ROI7 [21], and DSCF [28]) はすべて、画像とターゲットポーズペアまたは複数の身体部分を入力として、従来の畳み込みベースの生成モデルを直接使用しました。

- DSCF [28] employed deformable skip connections to construct the generator and can only transform the images in a coarse rectangle scale using simple affinity property. However, they ignore the most critical issue (i.e. large spatial misalignment) in person image synthesis, which limits their capabilities in dealing with large pose changes. Besides, they fail to capture structure coherence between condition images with target poses due to the lack of modeling higher-level part-level structure layouts. Hence, their results suffer from various artifacts, blurry boundaries, missing clothing appearance when large geometric transformations are requested by the desirable poses, which are far from satisfaction. As the Figure 1 shows, the performance of existing state-of-the-art person image synthesis methods is disappointing due to the severe misalignment problem for reaching target poses.
    - DSCF [28]は、変形可能なスキップ接続を使用してジェネレーターを構築し、単純なアフィニティプロパティを使用して、粗い長方形スケールの画像のみを変換できます。ただし、彼らは、人物の画像合成における最も重大な問題（つまり、大きな空間的不整合）を無視しているため、大きなポーズの変化を処理する能力が制限されています。その上、高レベルのパーツレベルの構造レイアウトのモデリングができないため、ターゲット画像と条件画像間の構造の一貫性をキャプチャできません。したがって、彼らの結果は、さまざまなアーティファクト、ぼやけた境界、大きな幾何学的変換が望ましいポーズによって要求されたときに失われた衣服の外観に苦しみます。図1に示すように、既存の最先端の人物画像合成方法のパフォーマンスは、ターゲットポーズに到達するための深刻な不整合の問題のために、期待はずれです。

---

- In this paper, we propose a novel Soft-Gated Warping-GAN to address the large spatial misalignment issues induced by geometric transformations of desired poses, which includes two stages:
    - この論文では、2つの段階を含む、望ましいポーズの幾何学的変換によって引き起こされる大きな空間的ミスアライメントの問題に対処するための新しいSoft-Gated Warping-GANを提案します。

- 1) a pose- guided parser is employed to synthesize a part segmentation map given a target pose, which depicts part-level spatial layouts to better guide the image generation with high-level structure constraints;
    - 1）「ポーズガイド付きパーサー」を使用して、ターゲットポーズが与えられたパーツセグメンテーションマップを合成します。これは、パーツレベルの空間レイアウトを示し、高レベルの構造制約で画像生成をより適切にガイドします。

- 2) a Warping-GAN renders detailed appearances into each segmentation part by learning geometric mappings from the original image to the target pose, conditioned on the predicted segmentation map. The Warping-GAN first trains a light-weight geometric matcher and then estimates its transformation parameters between the condition and synthesized segmentation maps. Based on the learned transformation parameters, the Warping-GAN incorporates a soft-gated warping-block which warps deep feature maps of the condition image to render the target segmentation map.
    - 2）「Warping-GAN」は、予測されたセグメンテーションマップに基づいて、元の画像からターゲットポーズへの幾何学的マッピングを学習することにより、各セグメンテーションパーツに詳細な外観をレンダリングします。 Warping-GANは最初に軽量の幾何学的マッチャーをトレーニングし、次に条件と合成されたセグメンテーションマップ間の変換パラメーターを推定します。 学習した変換パラメーターに基づいて、Warping-GANは、条件画像の深い特徴マップをワープしてターゲットセグメンテーションマップをレンダリングする「ソフトゲートワーピングブロック」を組み込みます。

---

- Our Warping-GAN has several technical merits. First, the warping-block can control the transfor- mation degree via a soft gating function according to different pose manipulation requests. For example, a large transformation will be activated for significant pose changes while a small degree of transformation will be performed for the case that the original pose and target pose are similar. Second, warping informative feature maps rather than raw pixel values could help synthesize more realistic images, benefiting from the powerful feature extraction. Third, the warping-block can adaptively select effective feature maps by attention layers to perform warping.
    - **Warping-GANにはいくつかの技術的なメリットがあります。 まず、ワーピングブロックは、さまざまなポーズ操作要求に従って、ソフトゲーティング機能を介して変換の度合いを制御できます。 たとえば、大幅なポーズ変更では大きな変換がアクティブになり、元のポーズとターゲットポーズが類似している場合は、わずかな変換が実行されます。 第二に、生のピクセル値ではなく有益な特徴マップをワーピングすることで、より現実的な画像を合成し、強力な特徴抽出の恩恵を受けることができます。 第三に、ワーピングブロックは、ワーピングを実行するためにアテンションレイヤーによって効果的なフィーチャマップを適応的に選択できます。**


# ■ 結論

## x. Conclusion


# ■ 何をしたか？詳細

## 3 Soft-Gated Warping-GAN

### 3.1.1 Stage I: Pose-Guided Parsing

- To learn the mapping from condition image to the target pose on a part-level, a pose-guide parser is introduced to generate the human parsing of target image conditioned on the pose. The synthesized human parsing contains pixel-wise class labels that can guide the image generation on the class-level, as it can help to refine the detailed appearance of parts, such as face, clothes, and hands.
    - 条件画像から部分レベルのターゲットポーズへのマッピングを学習するために、ポーズガイドパーサーが導入され、ポーズに条件付けられたターゲット画像の人間の解析を生成します。 合成された人間の解析には、ピクセル単位のクラスラベルが含まれており、顔、衣服、手などのパーツの詳細な外観を改善するのに役立つため、クラスレベルで画像生成をガイドできます。

- Since the DeepFashion and Market-1501 dataset do not have human parsing labels, we use the LIP [9] dataset to train a human parsing network. The LIP [9] dataset consists of 50,462 images and each person has 20 semantic labels.

- To capture the refined appearance of the person, we transfer the synthesized parsing labels into an one-hot tensor with 20 channels. Each channel is a binary mask vector and denotes the one class of person parts. These vectors are trained jointly with condition image and pose to capture the information from both the image features and the structure of the person, which benefits to synthesize more realistic-looking person images. Adapted from Pix2pix [13], the generator of the pose-guided parser contains 9 residual blocks. In addition, we utilize a pixel-wise softmax loss from LIP [9] to enhance the quality of results. As shown in Figure 2, the pose-guided parser consists of one ResNet-like generator, which takes condition image and target pose as input, and outputs the target parsing which obeys the target pose.
    - 人の洗練された外観をキャプチャするために、合成された解析ラベルを20チャネルのワンホットテンソルに転送します。 各チャネルはバイナリマスクベクトルであり、人体の1つのクラスを示します。 これらのベクトルは、条件画像とポーズを組み合わせて訓練され、画像の特徴と人物の構造の両方から情報をキャプチャします。これにより、よりリアルに見える人物画像を合成できます。 Pix2pix [13]から改作された、ポーズガイドパーサーのジェネレーターには、9つの残差ブロックが含まれています。 さらに、LIP [9]からのピクセル単位のソフトマックス損失を利用して、結果の品質を高めます。 図2に示すように、ポーズガイドパーサーは1つのResNetのようなジェネレーターで構成され、条件画像とターゲットポーズを入力として受け取り、ターゲットポーズに従うターゲット解析を出力します。

### 3.1.2 Stage II: Warping-GAN Rendering

- In this stage, we exploit a novel region-wise learning to render the texture details based on specific regions, guided by the synthesized parsing from the stage I. Formally, Let Ii = P(li) denote the function for the region-wise learning, where Ii and li denote the i-th pixel value and the class label of this pixel respectively. And i (0 ≤ i < n) denotes the index of pixel in image. n is the total number of pixels in one image. Note that in this work, the segmentation map also named parsing or human parsing, since our method towards the person images.
    - この段階では、新しい領域ごとの学習を活用して、特定の領域に基づいてテクスチャの詳細をレンダリングします。これは、段階Iからの合成解析によって導かれます。正式には、領域ごとの学習の関数をIi = P（li） ここで、Iiとliは、それぞれi番目のピクセル値とこのピクセルのクラスラベルを示します。 そして、i（0≤i <n）は画像のピクセルのインデックスを示します nは、1つの画像のピクセルの総数です。 この作業では、セグメンテーションマップが解析または人間の解析とも呼ばれていることに注意してください。

#### Geometric Matcher. 

- We train a geometric matcher to estimate the transformation mapping between the condition and synthesized parsing, as illustrate in Figure 3. Different from GEO [25], we handle this issue as parsing context matching, which can also estimate the transformation effectively.
    - 図3に示すように、ジオメトリマッチャーをトレーニングして、条件と合成された解析間の変換マッピングを推定します。GEO[25]とは異なり、この問題を解析コンテキストマッチングとして処理します。

- Due to the lack of the target image in the test phrase, we use the condition and synthesized parsing to compute the transformation parameters. 
    - テストフレーズにターゲットイメージがないため、条件と合成解析を使用して変換パラメーターを計算します。

- In our method, we combine affine and TPS to obtain the transformation mapping through a siamesed convolutional neural network following GEO [25]. 
    - この方法では、アフィンとTPSを組み合わせて、GEO [25]に続く「siamesedたたみ込みニューラルネットワーク」を介して変換マッピングを取得します。

- To be specific, we first estimate the affine transformation between the condition and synthesized parsing. Based on the results from affine estimation, we then estimate TPS transformation parameters between warping results from the affine transformation and target parsing.
    - 具体的には、まず条件と合成解析の間のアフィン変換を推定します。 アフィン推定の結果に基づいて、アフィン変換のワーピング結果とターゲット解析の間のTPS変換パラメーターを推定します。

- The transformation mappings are adopted to transform the extracted features of the condition image, which helps to alleviate the misalignment problem.
    - 変換マッピングは、条件画像の抽出された特徴を変換するために採用されており、ミスアライメントの問題を軽減するのに役立ちます。

#### Soft-gated Warping-Block. 

- Inspired by [3], having obtained the transformation mapping from the geometric matcher, we subsequently use this mapping to warp deep feature maps, which is able to capture the significant high-level information and thus help to synthesize image in an approximated shape with more realistic-looking details.
    - [3]に触発され、幾何学的マッチャーから変換マッピングを取得した後、このマッピングを使用して深い特徴マップをワープします。これにより、重要な高レベルの情報をキャプチャし、より多くの近似形状の画像を合成できます リアルに見える詳細。

- We combine the affine [7] and TPS [2] (Thin-Plate Spline transformation) as the transformation operation of the warping-block. As shown in Figure 4, we denote those transformations as the transformation grid.
    - アフィン[7]とTPS [2]（Thin-Plate Spline変換）を、ワーピングブロックの変換操作として組み合わせます。 図4に示すように、これらの変換を変換グリッドとして示します。

- Formally, let Φ(I) denotes the deep feature map, R(Φ(I)) denotes the residual feature map from Φ(I), W(I) represents the operation of the transformation grid. Thus, we regard T(I) as the transformation operation, we then formulate the transformation mapping of the warping-block as:
    - 正式には、Φ（I）が深い特徴マップを表し、R（Φ（I））がΦ（I）からの残差特徴マップを表し、W（I）が変換グリッドの操作を表すとします。 したがって、T（I）を変換操作と見なし、ワーピングブロックの変換マッピングを次のように定式化します。

    
# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. Related Work


