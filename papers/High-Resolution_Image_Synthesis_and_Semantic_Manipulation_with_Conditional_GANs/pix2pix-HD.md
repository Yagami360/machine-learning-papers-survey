# ■ 論文
- 論文タイトル："High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs"
- 論文リンク：https://arxiv.org/abs/1711.11585
- 論文投稿日付：2017/11/30
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- We present a new method for synthesizing high- resolution photo-realistic images from semantic label maps using conditional generative adversarial networks (conditional GANs).
    - 条件付き生成敵対ネットワーク（条件付きGAN）を用いて、意味ラベルマップから高解像度の写実的な画像を合成する新しい方法を提案する。

- Conditional GANs have enabled a variety of applications, but the results are often limited to lowresolution and still far from realistic. 
    - 条件付きGANによってさまざまなアプリケーションが可能になりましたが、結果は低解像度に限定されていることが多く、まだ現実的ではありません。

- In this work, we generate 2048 × 1024 visually appealing results with a novel adversarial loss, as well as new multi-scale generator and discriminator architectures.
    - この作品では、我々は新しい敵対的な損失と同様に新しいマルチスケールジェネレータと弁別器アーキテクチャで2048×1024の視覚的に魅力的な結果を生成します。

- Furthermore, we extend our framework to interactive visual manipulation with two additional features. 
    - **さらに、2つの追加機能を使用して、フレームワークをインタラクティブなビジュアル操作に拡張します。**

- First, we incorporate object instance segmentation information, which enables object manipulations such as removing/adding objects and changing the object category.
    - **まず、オブジェクトインスタンスのセグメンテーション情報を取り入れます。これにより、オブジェクトの削除や追加、オブジェクトカテゴリの変更などのオブジェクト操作が可能になります。**

- Second, we propose a method to generate di- verse results given the same input, allowing users to edit the object appearance interactively.
    - 次に、ユーザが対話的にオブジェクトの外観を編集できるようにするために、同じ入力が与えられたときに異なる結果を生成する方法を提案する。

- Human opinion studies demonstrate that our method significantly outperforms existing methods, advancing both the quality and the resolution of deep image synthesis and editing.
    - ヒューマンオピニオン研究は、本発明の方法が既存の方法よりも著しく優れていることを実証し、深部画像合成および編集の品質および解像度の両方を向上させる。


# ■ イントロダクション（何をしたいか？）

## 1. Introduction

- Photo-realistic image rendering using standard graphics techniques is involved, since geometry, materials, and light transport must be simulated explicitly. Although existing graphics algorithms excel at the task, building and editing virtual environments is expensive and time-consuming. That is because we have to model every aspect of the world explicitly. If we were able to render photo-realistic images using a model learned from data, we could turn the process of graphics rendering into a model learning and inference problem. Then, we could simplify the process of creating new virtual worlds by training models on new datasets. We could even make it easier to customize environments by allowing users to simply specify overall semantic structure rather than modeling geometry, materials, or lighting.
    - ジオメトリ、マテリアル、ライトトランスポートを明示的にシミュレートする必要があるため、標準のグラフィックス技術を使用したフォトリアリスティックな画像レンダリングが必要です。 既存のグラフィックスアルゴリズムはタスクに優れていますが、仮想環境の構築と編集には費用と時間がかかります。 これは、世界のあらゆる側面を明示的にモデル化する必要があるためです。 データから学習したモデルを使用して写真のようにリアルな画像をレンダリングできた場合、グラフィックレンダリングのプロセスをモデルの学習と推論の問題に変えることができます。 次に、新しいデータセットでモデルをトレーニングすることにより、新しい仮想世界を作成するプロセスを簡素化できます。 ユーザーがジオメトリ、マテリアル、または照明をモデリングするのではなく、全体的なセマンティック構造を指定できるようにすることで、環境をカスタマイズしやすくすることさえできます。

---

- In this paper, we discuss a new approach that produces high-resolution images from semantic label maps. This method has a wide range of applications. For example, we can use it to create synthetic training data for training visual recognition algorithms, since it is much easier to create semantic labels for desired scenarios than to generate training images. Using semantic segmentation methods, we can transform images into a semantic label domain, edit the objects in the label domain, and then transform them back to the image domain. This method also gives us new tools for higher-level image editing, e.g, adding objects to images or changing the appearance of existing objects.
    - この論文では、セマンティックラベルマップから高解像度画像を生成する新しいアプローチについて説明します。 この方法には幅広い用途があります。 たとえば、トレーニング画像を生成するよりも目的のシナリオのセマンティックラベルを作成する方がはるかに簡単なので、これを使用して視覚認識アルゴリズムをトレーニングするための合成トレーニングデータを作成できます。 セマンティックセグメンテーションメソッドを使用して、画像をセマンティックラベルドメインに変換し、ラベルドメイン内のオブジェクトを編集してから、それらを画像ドメインに戻すことができます。 この方法は、画像にオブジェクトを追加したり、既存のオブジェクトの外観を変更したりするなど、より高レベルの画像編集用の新しいツールも提供します。
    
---

- To synthesize images from semantic labels, one can use the pix2pix method, an image-to-image translation frame- work [21] which leverages generative adversarial networks (GANs) [16] in a conditional setting. Recently, Chen and Koltun [5] suggest that adversarial training might be un- stable and prone to failure for high-resolution image gen- eration tasks. Instead, they adopt a modified perceptual loss [11, 13, 22] to synthesize images, which are high- resolution but often lack fine details and realistic textures.

---

- Here we address two main issues of the above state-of-the-art methods: (1) the difficulty of generating high-resolution images with GANs [21] and (2) the lack of details and realistic textures in the previous high-resolution results [5]. We show that through a new, robust adversarial learning objective together with new multi-scale generator and discriminator architectures, we can synthesize photo-realistic images at 2048 × 1024 resolution, which are more visually appealing than those computed by previous methods [5, 21]. We first obtain our results with adversarial training only, without relying on any hand-crafted losses [44] or pre-trained networks (e.g VGGNet [48]) for perceptual losses [11, 22] (Figs 9c, 10b). Then we show that adding perceptual losses from pre-trained networks [48] can slightly improve the results in some circumstances (Figs 9d, 10c), if a pre-trained network is available. Both results outperform previous works substantially in terms of image quality.
    - ここでは、上記の最新の方法の2つの主な問題に対処します。（1）GANを使用した高解像度画像の生成の難しさ[21]および（2）以前の高解像度の詳細と現実的なテクスチャの欠如 解決結果[5]。 新しい堅牢な敵対的学習目標と新しいマルチスケールジェネレーターおよびディスクリミネーターアーキテクチャを通じて、2048×1024の解像度でフォトリアリスティックな画像を合成できることを示します。これは、以前の方法で計算されたものより視覚的に魅力的です[5、21 ]。 最初に、知覚的損失[11、22]（図9c、10b）について、手作業による損失[44]または事前に訓練されたネットワーク（例えばVGGNet [48]）に依存することなく、敵対的訓練のみで結果を取得します。 次に、事前に訓練されたネットワークが利用可能な場合、事前に訓練されたネットワーク[48]からの知覚的損失を追加すると、状況によっては結果がわずかに改善されることを示します（図9d、10c）。 どちらの結果も、画質の点で以前の作品を大幅に上回っています。

---

- Furthermore, to support interactive semantic manipulation, we extend our method in two directions. First, we use instance-level object segmentation information, which can separate different object instances within the same category. This enables flexible object manipulations, such as adding/removing objects and changing object types. Second, we propose a method to generate diverse results given the same input label map, allowing the user to edit the appearance of the same object interactively.
    - さらに、インタラクティブなセマンティック操作をサポートするために、メソッドを2つの方向に拡張します。 まず、インスタンスレベルのオブジェクトセグメンテーション情報を使用します。これにより、同じカテゴリ内の異なるオブジェクトインスタンスを分離できます。 これにより、オブジェクトの追加/削除やオブジェクトタイプの変更など、柔軟なオブジェクト操作が可能になります。 次に、同じ入力ラベルマップが与えられた場合に多様な結果を生成し、ユーザーが同じオブジェクトの外観をインタラクティブに編集できるようにする方法を提案します。

---

- We compare against state-of-the-art visual synthesis systems [5, 21], and show that our method outperforms these approaches regarding both quantitative evaluations and human perception studies. We also perform an ablation study regarding the training objectives and the importance of instance-level segmentation information. In addition to semantic manipulation, we test our method on edge2photo applications (Figs 2,13), which shows the generalizability of our approach. Code and data are available at our website


# ■ 結論

## x. 論文の項目名 (Conclusion)


# ■ 何をしたか？詳細

## 3. Instance-Level Image Synthesis

- We propose a conditional adversarial framework for generating high-resolution photo-realistic images from semantic label maps.
    - 意味ラベルマップから高解像度の写実的な画像を生成するための条件付き敵対的フレームワークを提案する。

- We first review our baseline model pix2pix (Sec. 3.1).

- We then describe how we increase the photorealism and resolution of the results with our improved objective function and network design (Sec. 3.2).
    - 次に、改良された目的関数とネットワーク設計を使用して、結果のフォトリアリズムと解像度をどのように向上させるかを説明します（3.2節）。

- Next, we use additional instance-level object semantic information to further improve the image quality (Sec. 3.3).
    - **次に、画像品質をさらに向上させるために、追加のインスタンスレベルのオブジェクト意味情報を使用します（3.3節）。**

- Finally, we introduce an instance-level feature embedding scheme to better handle the multi-modal nature of image synthesis, which enables interactive object editing (Sec. 3.4).
    - **最後に、我々は対話的なオブジェクト編集を可能にする、画像合成のマルチモーダルな性質をより良く扱うためのインスタンスレベルの特徴埋め込みスキームを紹介する（3.4節）。**


### 3.1. The pix2pix Baseline

- The pix2pix method [21] is a conditional GAN framework for image-to-image translation.

- It consists of a generator G and a discriminator D.

- For our task, the objective of the generator G is to translate semantic label maps to realistic-looking images, while the discriminator D aims to distinguish real images from the translated ones.

- The framework operates in a supervised setting.

- In other words, the training dataset is given as a set of pairs of corresponding images {(si , xi )}, where si is a semantic label map and xi is a corresponding natural photo.
    - 言い換えれば、トレーニングデータセットは、対応する画像のペアの集合{（si、xi）}として与えられます。ここで、siは意味ラベルマップ、xiは対応する自然写真です。

- Conditional GANs aim to model the conditional distribution of real images given the input semantic label maps via the following minimax game:

![image](https://user-images.githubusercontent.com/25688193/60664646-6864c780-9e9d-11e9-86a0-7f612587140d.png)

- The pix2pix method adopts U-Net [43] as the generator and a patch-based fully convolutional network [36] as the discriminator. The input to the discriminator is a channel- wise concatenation of the semantic label map and the corresponding image. However, the resolution of the generated images on Cityscapes [7] is up to 256 × 256. We tested directly applying the pix2pix framework to generate high- resolution images but found the training unstable and the quality of generated images unsatisfactory. Therefore, we describe how we improve the pix2pix framework in the next subsection.


### 3.2. Improving Photorealism and Resolution

- We improve the pix2pix framework by using a coarse-to- fine generator, a multi-scale discriminator architecture, and a robust adversarial learning objective function.

#### Coarse-to-fine generator 

- We decompose the generator into two sub-networks: G1 and G2. We term G1 as the global generator network and G2 as the local enhancer network. The generator is then given by the tuple G = {G1, G2} as visualized in Fig. 3. 

- The global generator network operates at a resolution of 1024 × 512, and the local enhancer network outputs an image with a resolution that is 4× the output size of the previous one (2× along each image dimension). 
    - グローバルジェネレータネットワークは１０２４×５１２の解像度で動作し、ローカルエンハンサーネットワークは、前の画像の出力サイズの４倍（各画像次元に沿って２倍）の解像度で画像を出力する。

- For synthesizing images at an even higher resolution, additional local enhancer networks could be utilized.

- For example, the output image resolution of the generator G = {G1 , G2 } is 2048 × 1024, and the output image resolution of G = {G1, G2, G3} is 4096 × 2048.

---

- Our global generator is built on the architecture proposed by Johnson et al. [22], which has been proven successful for neural style transfer on images up to 512 × 512.
    - 私たちのグローバルなジェネレータはJohnsonらによって提案されたアーキテクチャの上に構築されています。 [22]、これは最大512×512までの画像上のニューラルスタイルの転送に成功したことが証明されています。

- It consists of 3 components: a convolutional front-end G_1^F, a 1 set of residual blocks G(R) [18], and a transposed convolutional back-end G(B).

- A semantic label map of resolution 1024 × 512 is passed through the 3 components sequentially to output an image of resolution 1024 × 512.

---

- The local enhancer network also consists of 3 com- ponents: a convolutional front-end G(F ) , a set of residual blocks G(R), and a transposed convolutional back-end 2G(B). The resolution of the input label map to G is 2048 × 1024. 

- Different from the global generator network, the input to the residual block G_2^R is the element-wise sum of two feature maps: the output feature map of G_2^F, and the last feature map of the back-end of the global generator network G_1^B
    - 大域ジェネレータネットワークとは異なり、残差ブロックG_2 ^ Rへの入力は、2つの特徴マップの要素ごとの合計です。G_2^ Fの出力特徴マップと、大域ジェネレータのバックエンドの最後の特徴マップです。 ネットワークG_1 ^ B

- This helps integrating the global information from G1 to G2.
    - これにより、グローバル情報をG1からG2に統合することができます。

- During training, we first train the global generator and then train the local enhancer in the order of their resolutions. We then jointly fine-tune all the networks to- gether. We use this generator design to effectively aggre- gate global and local information for the image synthesis task. We note that such a multi-resolution pipeline is a well- established practice in computer vision [4] and two-scale is often enough [3]. Similar ideas but different architectures could be found in recent unconditional GANs [9, 19] and conditional image generation [5, 57].
    - トレーニング中に、まずグローバルジェネレータをトレーニングし、次にローカルエンハンサーをそれらの解決の順序でトレーニングします。 その後、すべてのネットワークを共同で微調整します。 このジェネレータ設計を使用して、画像合成タスクのためにグローバルおよびローカル情報を効果的に集約します。 我々は、そのような多重解像度パイプラインがコンピュータビジョンにおいて確立された慣行であり[4]、2スケールで十分であることが多い[3]ことに注意する。 最近の無条件GAN [9、19]および条件付き画像生成[5、57]にも、似たようなアイデアだが異なるアーキテクチャが見つかる可能性があります。
    

#### Multi-scale discriminators 

- High-resolution image synthesis poses a significant challenge to the GAN discriminator design.
    - 高解像度画像合成は、GAN識別器の設計に大きな課題を投げかけています。

- To differentiate high-resolution real and synthesized images, the discriminator needs to have a large receptive field.
    - 高解像度の実画像と合成画像とを区別するために、識別器は大きな受容野を有する必要がある。

- This would require either a deeper network or larger convolutional kernels, both of which would increase the network capacity and potentially cause overfitting. 
    - これには、より深いネットワークまたはより大きな畳み込みカーネルのいずれかが必要になります。どちらもネットワーク容量が増加し、場合によってはオーバーフィットが発生する可能性があります。
    
- Also, both choices demand a larger memory footprint for training, which is already a scarce resource for high- resolution image generation.
    また、どちらの選択も学習用に大きなメモリフットプリントを必要とし、これはすでに高解像度画像生成のための希少な [scarce] リソースです。

> フットプリント【footprint】とは、足跡という意味の英単語で、ITの分野ではソフトウェアやシステムなどが稼働時に占有する資源の多さなどの意味でよく用いられる

---

- To address the issue, we propose using multi-scale discriminators.
    - この問題に対処するために、我々はマルチスケール識別器を使用することを提案する。

- We use 3 discriminators that have an identical network structure but operate at different image scales.
    - 我々は、同一のネットワーク構造を有するが異なる画像スケールで動作する３つの識別器を使用する。
    
- We will refer to the discriminators as D1 , D2 and D3 . 
    - 識別器をD1、D2、D3と呼びます。

- Specifically, we downsample the real and synthesized high-resolution images by a factor of 2 and 4 to create an image pyramid of 3 scales.
    - 具体的には、3スケールの画像ピラミッドを作成するために、本物画像と合成された高解像度画像を、2 と 4 の係数でダウンサンプリングする。

- The discriminators D1, D2 and D3 are then trained to differentiate real and synthesized images at the 3 different scales, respectively.
    - 次に、識別器 D1, D2, D3 は、それぞれ３つの異なるスケールで、本物画像と合成画像とを区別する [differentiate] ように学習される。

- Although the discriminators have an identical architecture, the one that operates at the coarsest scale has the largest receptive field.
    - 識別器は同一のアーキテクチャを持ってるが、最も粗いスケールで動作するものが、最大の受容的分野を持っています。

- It has a more global view of the image and can guide the generator to generate globally consistent images.
    - それは画像のより大域的なな視点を持ち、そして大域的に一貫した画像を生成するように生成器を導くことができます。

- On the other hand, the discriminator at the finest scale encourages the generator to produce finer details.
    - 一方、最も細かいスケールでの識別器は、生成器がより細かい詳細を生成するように促します。

- This also makes training the coarse-to-fine generator easier, since extending a low-resolution model to a higher resolution only requires adding a discriminator at the finest level, rather than retraining from scratch.
    - 低解像度モデルを高解像度に拡張することは、スクラッチでゼロから再学習するのではなく、最も細かいレベルで識別器を追加することのみを要求するので、
    - これによって coarse-to-fine 生成器を学習することをより簡単にする。

- Without the multi-scale discriminators, we observe that many repeated patterns often appear in the generated images.
    - マルチスケール識別器なしでは、生成画像において、我々は多くの繰り返されるパターンが現れることを観察する。

- With the discriminators, the learning problem in Eq. (1) then becomes a multi-task learning problem of
    - 識別器では、式 (1) の学習問題は、以下の式のマルチタスク学習問題になる

![image](https://user-images.githubusercontent.com/25688193/60719913-03c07000-9f64-11e9-8925-265b3c6df445.png)

- Using multiple GAN discriminators at the same image scale has been proposed in unconditional GANs [12].
    - 同じ画像スケールでマルチ GAN 識別器を使用することは、unconditional GANs [12] で提案されている。

- Iizuka et al. [20] add a global image classifier to conditional GANs to synthesize globally coherent content for inpainting.
    - 飯塚ら [20] 手法では、大域的な画像分類器を条件付きGANに追加して、修復のために、グローバルにコヒーレントなコンテンツを合成します。

- Here we extend the design to multiple discriminators at different image scales for modeling high-resolution images.
    - ここでは、高解像度画像をモデル化するために、異なる画像スケールで、複数の識別器に設計を拡張します。


### 3.3. Using Instance Maps

- Existing image synthesis methods only utilize semantic label maps [5, 21, 25], an image where each pixel value represents the object class of the pixel.
    - 既存の画像合成方法は、**各画素値がその画素のオブジェクトクラスを表す画像であるセマンティックラベルマップ**［５、２１、２５］のみを利用する。

- This map does not differentiate objects of the same category.
    - **このマップは、同じカテゴリのオブジェクトを区別しません。**

- On the other hand, an instance-level semantic label map contains a unique object ID for each individual object.
    - 一方、物体レベルのセマンティックラベルマップには、個々のオブジェクトごとに一意のオブジェクトIDが含まれています。

- To incorporate the instance map, one can directly pass it into the network, or encode it into a one-hot vector.
    - 物体マップを組み込む [incorporate] ために、それを直接ネットワークに渡すか、またはそれを one-hot ベクトルに符号化することができる。

- However, both approaches are difficult to implement in practice, since different images may contain different numbers of objects of the same category.
    - しかしながら、異なる画像は同じカテゴリの異なる数のオブジェクトを含む可能性があるため、両方のアプローチを実際に実施するのは困難である。

- Alternatively, one can pre-allocate a fixed number of channels (e.g., 10) for each class, but this method fails when the number is set too small, and wastes memory when the number is too large.
    - あるいは、クラスごとに固定数（例えば１０）のチャネルを事前に割り当てることができるが、この方法は、数が小さすぎると失敗し、数が大きすぎるとメモリを浪費する。

---

![image](https://user-images.githubusercontent.com/25688193/60665305-17ee6980-9e9f-11e9-91c8-8ce7052b2d81.png)

- > Figure 4: Using instance maps: (a) a typical semantic label map. 

- > Note that all connected cars have the same label, which makes it hard to tell them apart.

- > (b) The extracted instance boundary map.

- > With this information, separating different objects becomes much easier.

---

- Instead, we argue that the most critical information the instance map provides, which is not available in the semantic label map, is the object boundary.
    - **その代わりに、インスタンスマップが提供する最も重要な情報は、セマンティックラベルマップでは利用できない、オブジェクトの境界であると主張します。**

- For example, when objects of the same class are next to one another, looking at the semantic label map alone cannot tell them apart.
    - たとえば、同じクラスのオブジェクトが隣接している場合、セマンティックラベルマップを見ただけでは区別できません。

- This is especially true for the street scene since many parked cars or walking pedestrians are often next to one another, as shown in Fig. 4a. 
    - 図4aに示すように、多くの駐車中の車や歩行者が隣同士にいることが多いため、これは特にストリートシーンに当てはまります。

- However, with the instance map, separating these objects becomes an easier task.
    - ただし、インスタンスマップでは、これらのオブジェクトを分離することがより簡単な作業になります。

---

- Therefore, to extract this information, we first compute the instance boundary map (Fig. 4b).
    - したがって、この情報を抽出するために、まずインスタンス境界マップを計算します（図4b）。

- In our implementation, a pixel in the instance boundary map is 1 if its object ID is different from any of its 4-neighbors, and 0 otherwise.
    - 我々の実装では、インスタンス境界マップ内のピクセルは、そのオブジェクトIDが4つの近傍のいずれとも異なる場合は1、それ以外の場合は0です。

- The instance boundary map is then concatenated with the one-hot vector representation of the semantic label map, and fed into the generator network.
    - 次に、インスタンス境界マップは、意味ラベルマップのワンホットベクトル表現と連結され、ジェネレータネットワークに入力されます。

- Similarly, the input to the discriminator is the channel-wise concatenation of instance boundary map, semantic label map, and the real/synthesized image.
    - 同様に、識別器への入力は、インスタンス境界マップ、セマンティックラベルマップ、および実／合成画像のチャネル単位での連結である。

- Figure 5b shows an example demonstrating the improvement by using object boundaries.
    - 図５ｂは、オブジェクト境界を使用することによる改善を実証する例を示す。

- Our user study in Sec. 4 also shows the model trained with instance boundary maps renders more photo-realistic object boundaries.
    - セクション４での私たちのユーザー研究はまた、インスタンス境界マップで訓練されたモデルがより写実的なオブジェクト境界をレンダリングすることを示す。


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


