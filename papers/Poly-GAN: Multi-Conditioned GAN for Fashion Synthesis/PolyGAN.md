# ■ 論文
- 論文タイトル："Poly-GAN: Multi-Conditioned GAN for Fashion Synthesis"
- 論文リンク：https://arxiv.org/abs/1909.02165
- 論文投稿日付：2019/09/05
- 被引用数（記事作成時点）：xxx 件
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- We present Poly-GAN, a novel conditional GAN architecture that is motivated by Fashion Synthesis, an application where garments are automatically placed on images of human models at an arbitrary pose. Poly-GAN allows conditioning on multiple inputs and is suitable for many tasks, including image alignment, image stitching and inpainting. Existing methods have a similar pipeline where three different networks are used to first align garments with the human pose, then perform stitching of the aligned garment and finally refine the results. Poly-GAN is the first instance where a common architecture is used to perform all three tasks. Our novel architecture enforces the conditions at all layers of the encoder and utilizes skip connections from the coarse layers of the encoder to the respective layers of the decoder. Poly-GAN is able to perform a spatial transformation of the garment based on the RGB skeleton of the model at an arbitrary pose. Additionally, Poly-GAN can perform image stitching, regardless of the garment orientation, and inpainting on the garment mask when it contains irregular holes. Our system achieves state-of-the-art quantitative results on Structural Similarity Index metric and Inception Score metric using the DeepFashion dataset.
    - Poly-GANは、ファッション合成によって動機付けられた新しい条件付きGANアーキテクチャであり、任意のポーズで人間のモデルの画像に衣服が自動的に配置されるアプリケーションです。 Poly-GANは複数の入力の調整を可能にし、画像の位置合わせ、画像のつなぎ合わせ、修復などの多くのタスクに適しています。既存の方法には同様のパイプラインがあり、3つの異なるネットワークを使用して最初に衣服を人間のポーズに合わせ、次に調整された衣服のステッチングを実行し、最終的に結果を調整します。 Poly-GANは、3つのタスクすべてを実行するために共通のアーキテクチャが使用される最初のインスタンスです。当社の新しいアーキテクチャは、エンコーダーのすべてのレイヤーで条件を強制し、エンコーダーの粗いレイヤーからデコーダーのそれぞれのレイヤーへのスキップ接続を利用します。 Poly-GANは、任意のポーズでモデルのRGBスケルトンに基づいて衣服の空間変換を実行できます。さらに、Poly-GANは衣服の向きに関係なく画像のステッチングを実行でき、不規則な穴が含まれている場合は衣服マスクを修復できます。私たちのシステムは、DeepFashionデータセットを使用して、構造的類似性指標とインセプションスコア指標で最先端の定量的結果を達成します。


# ■ イントロダクション（何をしたいか？）

## 1. Introduction

- Fashion Synthesis is a challenging task that requires placing a reference garment on a source model who is at an arbitrary pose and wears a different garment Han et al [9] WangB et al [29] Dong et al [5] Lassner et al [15] Cui et al [4]. The arbitrary human pose requirement creates challenges, such as handling self occlusion or limited availability of training data, as the training dataset may or not have the model’s desired pose. Some of the challenges in Fashion Synthesis are encountered in other applications, such as person re-identification Dong et al [6], person modeling LiuJ et al [18], and Image2Image translation Jetchev et al [11]. Existing methods for Fashion Synthesis follow a pipeline consisting of three stages, each requiring different tasks, that are performed by different networks. These tasks include performing an affine transformation to align the reference garment with the source model Rocco et al [25], stitching the garment on the source model, and refining or post-processing to reduce artifacts after stitching. The problem encountered with this pipeline is that stitching the warped garment often results in artifacts due to self occlusion, spill of color, and blurriness in the generation of missing body regions.
    - ファッション合成は、任意のポーズで異なる衣服を着ているソースモデルに参照衣服を配置する必要がある挑戦的な作業です。任意の人間のポーズ要件は、トレーニングデータセットがモデルの望ましいポーズを持っている場合と持っていない場合があるため、セルフオクルージョンの処理やトレーニングデータの利用可能性の制限などの課題を引き起こします。 Fashion Synthesisの課題の一部は、他のアプリケーションでも発生します。
    - ファッション合成の既存の方法は、それぞれが異なるネットワークで実行される異なるタスクを必要とする3つのステージで構成されるパイプラインに従います。これらのタスクには、アフィン変換を実行して参照衣服をソースモデルRocco et al [25]に合わせる、衣服をソースモデルにステッチする、ステッチ後のアーティファクトを減らすための改良または後処理が含まれます。**このパイプラインで発生する問題は、反った衣服を縫い合わせると、欠落した身体領域の生成における自己閉塞、色のこぼれ、およびぼやけによるアーティファクトがしばしば生じることです。**

---

- In this paper, we take a more universal approach by proposing a single architecture for all three tasks in the Fashion Synthesis pipeline. Instead of using an affine transformation to warp the garments to the body shape, we generate garments with our GAN conditioned on an arbitrary human pose. Generating transformed garments overcomes the problem of self occlusion and generates occluding arms and other body parts very effectively. The same architecture is then trained to perform stitching and inpainting. We demonstrate that our proposed GAN architecture not only achieves state of the art results for Fashion Synthesis, but it is also suitable for many tasks. Thus, we name our architecture Poly-GAN. Figure 1 shows representative examples of the performance achieved with Poly-GAN.
    - **このホワイトペーパーでは、ファッション合成パイプラインの3つのタスクすべてに単一のアーキテクチャを提案することにより、より普遍的なアプローチを採用しています。 アフィン変換を使用して衣服を身体形状にワープする代わりに、任意の人間のポーズを条件にしたGANで衣服を生成します。 変換された衣服を生成すると、自己閉塞の問題が克服され、非常に効果的に閉塞腕と他の身体部分が生成されます。 次に、同じアーキテクチャがスティッチングと修復を実行するようにトレーニングされます。 提案したGANアーキテクチャは、ファッション合成の最先端の結果を達成するだけでなく、多くのタスクにも適していることを示しています。 したがって、アーキテクチャにPoly-GANという名前を付けます。 図1に、Poly-GANで達成されたパフォーマンスの代表例を示します。**

---

- Our Fashion Synthesis approach consists of the following stages, illustrated in Figure 2. Stage 1 performs image generation conditioned on an arbitrary human pose, which changes the shape of the reference garment so it can precisely fit on the human body. Stage 2 performs image stitching of the newly generated garment (from Stage 1) with the model after the original garment is segmented out. Stage 3 performs refinement by inpainting the output of Stage 2 to fill any missing regions or spots. Stage 4 is a post-processing step that combines the results from Stages 2 and 3, and adds the model head for the final result. Our approach achieves state of the art quantitative results compared to the popular Virtual Try On (VTON) method WangB et al. [29].
    - Fashion Synthesisのアプローチは、図2に示す次の段階で構成されます。段階1は、任意の人間のポーズを条件とする画像生成を実行します。 ステージ2は、元の衣服がセグメント化された後、モデルで新しく生成された衣服（ステージ1から）の画像縫い合わせ [stitching] を実行します。 ステージ3は、ステージ2の出力を修復して、欠落している領域またはスポットを埋めることにより改良を実行します。 ステージ4は、ステージ2とステージ3の結果を結合し、最終結果のモデルヘッドを追加する後処理ステップです。 私たちのアプローチは、人気の仮想試行（VTON）メソッドWangB et al。と比較して、最先端の定量結果を達成します。 [29]。

---

- The main contributions of this paper can be summarized as follows:


1. We propose a new conditional GAN architecture, which can operate on multiple conditions that manipulate the generated image.

1. In our Poly-GAN architecture, the conditions are fed to all layers of the encoder to strengthen their effects throughout the encoding process. Additionally, skip connections are introduced from the coarse layers of the encoder to the respective layers of the decoder.
    - 当社のPoly-GANアーキテクチャでは、エンコーダーのすべてのレイヤーに条件が与えられ、エンコードプロセス全体の効果を強化します。 さらに、エンコーダーの粗いレイヤーからデコーダーのそれぞれのレイヤーへのスキップ接続が導入されます。

1. We demonstrate that our architecture can perform many tasks, including shape manipulation conditioned on human pose for affine transformations, image stitching of a garment on the model, and image inpainting.
    - 私たちのアーキテクチャは、アフィン変換のための人間のポーズを条件とする形状操作、モデル上の衣服の画像縫い合わせ、画像の修復など、多くのタスクを実行できることを示しています。

1. Poly-GAN is the first GAN to perform an affine transformation of the reference garment based on the RGB skeleton of the model at an arbitrary pose.
    - Poly-GANは、任意のポーズでのモデルのRGBスケルトンに基づいて参照衣服のアフィン変換を実行する最初のGANです。

1. Our method is able to preserve the desired pose of human arms and hands without color spill, even in cases of self occlusion, while performing Fashion Synthesis.
    - 私たちの方法は、ファッション合成を実行しながら、自己閉塞の場合でも、色がこぼれる [spill] ことなく、人間の腕と手の望ましいポーズを維持することができます。

> オクルージョンとは手前にある物体が背後にある物体を隠して見えないようにする状態のことです。

# ■ 結論

## x. Conclusion


# ■ 何をしたか？詳細

## 3 Poly-GAN

- Poly-GAN, a new conditional GAN for fashion synthesis, is the first instance where a common architecture is used to perform many tasks previously performed by different networks. Poly-GAN is flexible and can accept multiple conditions as inputs for various tasks. We begin with an overview of the pipeline used for Fashion Synthesis, shown in Figure 2, and then we present the details of the Poly-GAN architecture.
    - ファッション合成用の新しい条件付きGANであるPoly-GANは、以前はさまざまなネットワークで実行されていた多くのタスクを実行するために共通のアーキテクチャが使用される最初のインスタンスです。 Poly-GANは柔軟性があり、さまざまなタスクの入力として複数の条件を受け入れることができます。 まず、図2に示すファッション合成に使用されるパイプラインの概要から始め、次にPoly-GANアーキテクチャの詳細を示します。

### 3.1 Pipeline for Fashion Synthesis

- Two images are inputs to the pipeline, namely the reference garment image and the model image, which is the source person on whom we wish to place the reference garment. A pre-trained pose estimator is used to extract the pose skeleton of the model, as shown in Figure 2. The model image is passed to the segmentation network to extract the segmented mask of the garment, which is used to replace the old garment on the model.
    - 2つの画像は、パイプラインへの入力です。つまり、リファレンスガーメント画像と、リファレンスガーメントを配置するソース画像であるモデル画像です。 図2に示すように、事前トレーニング済みのポーズ推定器を使用してモデルのポーズスケルトンを抽出します。モデル画像はセグメンテーションネットワークに渡され、衣類のセグメントマスクを抽出します。 モデル。

---

- The entire flow can be divided into 4 stages illustrated in Figure 2. In Stage 1, the RGB pose skeleton is concatenated with the reference garment and passed to the Poly-GAN. The RGB skeleton acts as a condition for generating a garment that is reshaped according to an arbitrary human pose. The Stage 1 output is a newly generated garment which matches the shape and alignment of the RGB skeleton on which Poly-GAN is conditioned. The transformed garment from Stage 1 along with the segmented human body (without garment and without head) and the RGB skeleton are passed to the Poly-GAN in Stage 2. Stage 2 serves the purpose of stitching the generated garment from Stage 1 to the segmented human body which has no garment. In Stage 2, Poly-GAN assumes that the incoming garment may be positioned at any angle and does not require garment alignment with the human body. This assumption makes Poly-GAN more robust to potential misalignment of the generated garment during the transformation in Stage 1. Due to differences in size between the reference garment and the segmented garment on the body of the model, there may be blank areas due to missing regions. To deal with missing regions at the output of Stage 2, we pass the resulting image to Stage 3 along with the difference mask indicating missing regions. In Stage 3, Poly-GAN learns to perform inpainting on irregular holes and refines the final result. In Stage 4, we perform post processing by combining the results of Stage 2 and Stage 3, and stitching the head back on the body for the final result.



# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. Related Work


