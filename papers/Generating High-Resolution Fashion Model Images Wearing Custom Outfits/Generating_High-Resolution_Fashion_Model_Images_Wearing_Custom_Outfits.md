> 論文まとめ（要約 ver） : https://github.com/Yagami360/MachineLearning-Papers_Survey/issues/16

# ■ 論文
- 論文タイトル："Generating High-Resolution Fashion Model Images Wearing Custom Outfits"
- 論文リンク：https://arxiv.org/abs/1908.08847
- 論文投稿日付：2019/08/23
- 被引用数（記事作成時点）：xxx 件
- 著者（組織）：Gokhan Yildirim Nikolay Jetchev Roland Vollgraf Urs Bergmann ¨
- categories：

# ■ 概要（何をしたか？）

## Abstract

- Visualizing an outfit is an essential part of shopping for clothes. Due to the combinatorial aspect of combining fashion articles, the available images are limited to a predetermined set of outfits. In this paper, we broaden these visualizations by generating high-resolution images of fashion models wearing a custom outfit under an input body pose. We show that our approach can not only transfer the style and the pose of one generated outfit to another, but also create realistic images of human bodies and garments.
    - 服を視覚化することは、衣服の買い物に不可欠な部分です。 ファッション記事を組み合わせることの組み合わせの側面により、利用可能な画像は衣装の所定のセットに予め決められます [predetermined]。 この論文では、入力ボディポーズの下でカスタム服を着たファッションモデルの高解像度画像を生成することにより、これらの視覚化を広げます [broaden]。 私たちのアプローチは、生成された服のスタイルとポーズを別の服に移すだけでなく、人体と衣服のリアルな画像を作成できることを示しています。

# ■ イントロダクション（何をしたいか？）

## 1. Introduction

- Fashion e-commerce platforms simplify apparel shopping through search and personalization. A feature that can further enhance user experience is to visualize an outfit on a human body. Previous studies focus on replacing a garment on an already existing image of a fashion model [5, 2] or on generating low-resolution images from scratch by using pose and garment color as input conditions [8]. In this paper, we concentrate on generating high-resolution images of fashion models wearing desired outfits and given poses.
    - ファッションeコマースプラットフォームは、検索とパーソナライズを通じてアパレルショッピング [apparel shopping] を簡素化します。 ユーザーエクスペリエンスをさらに強化できる機能は、人体の衣装を視覚化することです。 以前の研究では、ファッションモデルの既存の画像[5、2]で衣服 [garment] を交換するか、入力条件としてポーズと衣服の色を使用してゼロから低解像度画像を生成することに焦点を当てていました[8]。 このホワイトペーパーでは、目的の衣装と与えられたポーズを身に着けているファッションモデルの高解像度画像の生成に集中します。

---

- In recent years, advances in Generative Adversarial Networks (GANs) [1] enabled sampling realistic images via implicit generative modeling. One of these improvements is Style GAN [7], which builds on the idea of generating high- resoluton images using Progressive GAN [6] by modifying it with Adaptive Instance Normalization (AdaIN) [4].
    - 近年、Generative Adversarial Networks（GAN）[1]の進歩により、暗黙的な生成モデリングを介した現実的な画像のサンプリングが可能になりました。 これらの改善点の1つはStyle GAN [7]です。これは、Adaptive Instance Normalization（AdaIN）[4]で変更することにより、Progressive GAN [6]を使用して高解像度の画像を生成するというアイデアに基づいています。
    
- In this paper, we employ and modify Style GAN on a dataset of model-outfit-pose images under two settings: We first train the vanilla Style GAN on a set of fashion model images and show that we can transfer the outfit color and body pose of one generated fashion model to another. Second, we modify Style GAN to condition the generation process on an outfit and a human pose. This enables us to rapidly visualize custom outfits under different body poses and types.
    - **この論文では、2つの設定の下でモデル衣装ポーズ画像のデータセットでスタイルGANを採用および変更します。まず、ファッションモデル画像のセットでバニラスタイルGANをトレーニングし、衣装の色と体のポーズを転送できることを示します。 生成されたファッションモデルを別のものに 次に、スタイルGANを変更して、衣装と人間のポーズの生成プロセスを調整します。 これにより、さまざまな体のポーズや種類の下でカスタム衣装をすばやく視覚化できます。**


# ■ 結論

## x. Conclusion


# ■ 何をしたか？詳細

## 2. Outfit Dataset

- We use a proprietary image dataset with around 380K entries. Each entry in our dataset consists of a fashion model wearing an outfit with a certain body pose. An outfit is composed of a set of maximum 6 articles. In order to obtain the body pose, we extract 16 keypoints using a deep pose estimator [10]. In Figure 1, we visualize a few samples from our dataset. The red markers on the fashion models represent the extracted keypoints. Both model and articles images have a resolution of 1024 × 768 pixels.
    - 約380Kのエントリを持つ独自の画像データセットを使用します。 データセットの各エントリは、特定のボディポーズの衣装を着たファッションモデルで構成されています。 衣装は、最大6つの記事のセットで構成されます。 体の姿勢を取得するために、深い姿勢推定器を使用して16のキーポイントを抽出します[10]。 図1では、データセットからいくつかのサンプルを視覚化します。 ファッションモデルの赤いマーカーは、抽出されたキーポイントを表しています。 モデル画像と記事画像の両方の解像度は1024×768ピクセルです。


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## 3. Experiments

- The flowchart for the unconditional version of Style GAN is illustrated in Figure 2(a). We have 18 generator layers that receive an affinely transformed copy of the style vector for adaptive instance normalization. The disciminator is identical to the original Style GAN. We train this network for around four weeks on four NVIDIA V100 GPUs, resulting in 160 epochs.
    - Style GANの無条件バージョンのフローチャートを図2（a）に示します。 アダプティブインスタンスの正規化のために、スタイルベクトルのアフィン変換されたコピーを受け取る18のジェネレータレイヤーがあります。 識別器は元のスタイルGANと同じです。 このネットワークを4つのNVIDIA V100 GPUで約4週間トレーニングし、160エポックを達成しました。

- In the conditional version, we modify Style GAN with an embedding network as shown in Figure 2(b). Inputs to this network are the six article images (in total 18 channels) and a 16-channel heatmap image that is computed from 16 keypoints. The article images are concatenated with fixed ordering for semantic consistency across outfits. We can see this ordering in Figure 1. If an outfit does not have an article on a particular semantic category, it is filled with an empty gray image. The embedding network creates a 512- dimensional vector, which is concatenated with the latent vector in order to produce the style vector. This model is also trained for four weeks (resulting in 115 epochs). The discriminator in the conditional model uses a separate network to compute the embedding for the input articles and
    - 条件付きバージョンでは、図2（b）に示すように、埋め込みネットワークを使用してStyle GANを変更します。 このネットワークへの入力は、6つの記事画像（合計18チャネル）と、16のキーポイントから計算される16チャネルのヒートマップ画像です。 記事の画像は、服装全体の意味の一貫性のために固定された順序で連結されます。 この順序を図1に示します。衣装に特定のセマンティックカテゴリに関する記事がない場合、空の灰色の画像で塗りつぶされます。 埋め込みネットワークは、512次元のベクトルを作成します。これは、スタイルベクトルを生成するために、潜在ベクトルと連結されます。 このモデルも4週間トレーニングされます（結果として115エポック）。 条件付きモデルの識別器は、独立したネットワークを使用して、入力記事の埋め込みを計算し、


### 3.1. Unconditional

- During the training, one can regularize the generator by switching the style vectors for certain layers. This has the effect of transferring information from one generated image to another. In Figure 4, we illustrate two examples of information transfer. First, we broadcast the same source style vector to layers 13 to 18 (before the affine transformations in Figure 2) of the generator, which transfers the color of the source outfit to the target generated image, as shown in Figure 4. If we copy the source style vector to earlier layers, this transfers the source pose. In Table 1, we show which layers we broadcast the source and the target style vectors to achieve the desired transfer effect.
    - トレーニング中に、特定のレイヤーのスタイルベクトルを切り替えることにより、ジェネレーターを正規化できます。 これには、生成されたイメージから別のイメージに情報を転送する効果があります。 図4に、情報転送の2つの例を示します。 まず、ジェネレーターのレイヤー13〜18（図2のアフィン変換の前）に同じソーススタイルベクトルをブロードキャストします。これにより、図4に示すように、ソース衣装の色がターゲット生成画像に転送されます。 ソーススタイルベクトルを以前のレイヤーに転送すると、ソースポーズが転送されます。 表1では、目的の転送効果を実現するために、ソースとターゲットのスタイルベクトルをブロードキャストするレイヤーを示しています。

### 3.2. Conditional

- After training our conditional model, we can input a desired set of articles and a pose to visualize an outfit on a human body as shown in Figure 5. We use two different outfits in Figure 5(a) and (b), and four randomly picked body poses to generate model images in Figure 5(c) and (d), respectively. We can observe that the articles are correctly rendered on the generated bodies and the pose is consistent across different outfits. In Figure 5(e), we visualize the generated images using a custom outfit by adding the jacket from the first outfit to the second one. We can see that the texture and the size of the denim jacket are correctly rendered on the fashion model. Note that, due to the spurious correlations within our dataset, the face of a generated model might vary depending on the outfit and the pose.
    - 条件付きモデルをトレーニングした後、図5に示すように、目的の記事とポーズを入力して、人体の衣装を視覚化できます。図5（a）と（b）の2つの異なる衣装と、ランダムに4つの衣装を使用します 図5（c）と（d）のモデル画像を生成するために、それぞれボディポーズを選択しました。 生成されたボディに記事が正しくレンダリングされ、さまざまな衣装でポーズが一貫していることがわかります。 図5（e）では、最初の衣装から2番目の衣装にジャケットを追加することにより、カスタム衣装を使用して生成された画像を視覚化します。 デニムジャケットのテクスチャとサイズが、ファッションモデルで正しくレンダリングされていることがわかります。 データセット内のスプリアス相関により、生成されたモデルの顔は衣装とポーズによって異なる場合があることに注意してください。

---

- In our dataset, we have fashion models with various body types that depend on their gender, build, and weight. This variation is implicitly represented through the relative distances between extracted keypoints. Our conditional model is able to capture and reproduce fashion models with different body types as shown in the fourth generated images in Figure 5. This result is encouraging, and our method might be extended in the future to a wider range of customers through virtual try-on applications.
    - データセットには、性別、体格、体重に依存するさまざまな体型のファッションモデルがあります。 この変動は、抽出されたキーポイント間の相対距離によって暗黙的に表されます。 条件付きモデルは、図5の4番目に生成された画像に示すように、さまざまなボディタイプのファッションモデルをキャプチャして再現できます。この結果は有望であり、仮想トライ アプリケーションで。



# ■ 関連研究（他の手法との違い）

## x. Related Work


