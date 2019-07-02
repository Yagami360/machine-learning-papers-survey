# ■ 論文
- 論文タイトル："Image2StyleGAN: How to Embed Images Into the StyleGAN Latent Space?"
- 論文リンク：https://arxiv.org/abs/1904.03189
- 論文投稿日付：
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- We propose an efficient algorithm to embed a given image into the latent space of StyleGAN.
    - 与えられた画像をStyleGANの潜在空間に埋め込むための効率的なアルゴリズムを提案する。

- This embedding enables semantic image editing operations that can be applied to existing photographs. 
    - この埋め込みは、既存の写真に適用することができるような、意味的画像編集 [semantic image editing] 操作を可能にする。

- Taking the StyleGAN trained on the FFHD dataset as an example, we show results for image morphing, style transfer, and expression transfer.
    - １つの例として、FFHD データセットで学習した StyleGAN を取りあげて、画像モーフィング・スタイル変換・表情変換の結果を示す。

> モーフィング : ある画像を別の画像に滑らかに変化させるグラフィック処理技術

- Studying the results of the embedding algorithm provides valuable insights into the structure of the StyleGAN latent space.
    - 埋め込みアルゴリズムの結果を勉強することは、StyleGAN の潜在空間の構造への価値のある洞察を提供する。

- We propose a set of experiments to test what class of images can be embedded, how they are embedded, what latent space is suitable for embedding, and if the embedding is semantically meaningful.
    - 我々は、どの画像のクラスを埋め込むことが出来るのか？どのように埋め込むのか？なんの潜在空間が埋め込みに適しているのか？そして埋め込みが意味論的に意味のあるものなのか？をテストするための一連の実験を提案する。


# ■ イントロダクション（何をしたいか？）

## 1. Introduction

- Generative Adverserial Networks (GANs) are very successfully applied in various computer vision applications, e.g. texture synthesis [19, 36, 30], video generation [34, 33], image-to-image translation [10, 39, 1, 26] and object detec- tion [20, 21].


---

- In the few past years, the quality of images synthesized by GANs has increased rapidly.

- Compared to the seminal DCGAN framework [27] in 2015, the current state-of-the- art GANs [13, 3, 14, 38, 40] can synthesize at a much higher resolution and produce significantly more realistic images.

- Among them, StyleGAN [14] makes use of an intermediate W latent space that holds the promise of enabling some controlled image modifications.
    - それらの中でも、StyleGAN は、いくつかの制御された画像修正を可能にするという約束を保持するような、中間潜在空間 W を利用する。

- We believe that image modifications are a lot more exciting when it becomes possible to modify a given image rather than a randomly GAN generated one.
    - 我々は、画像修正は、GANでランダムに生成されたときというよりも、与えられた（＝指定された）画像を修正することを可能にするようになるときにむしろ、よりエキサイティングになると信じている。

- This leads to the natural question if it is possible to embed a given photograph into the GAN latent space.
    - これは、もしGAN潜在空間の中に指定の画像を埋め込むことが可能であれば、もっともな質問 [natural question] を導く。

---

- To tackle this question, we build an embedding algo- rithm that can map a given image I in the latent space of StyleGAN pre-trained on the FFHQ dataset.
    - この問題に取り組む [tackle] ために、FFHQ データセットで事前学習された StyleGAN の潜在空間において、指定の画像 I を写像することの出来るような埋め込みアルゴリズムを構築する。

- One of our important insights is that the generalization ability of the pre-trained StyleGAN is significantly enhanced when using an extended latent space W + (See Sec. 3.3).
    - 我々の重要な洞察の１つは、事前学習された StyleGAN の汎化能力 [generalization ability] が、拡張された潜在空間 W+（セクション　3.3 を参照）を使用するとき、大幅に高まるということである。

- As a consequence, somewhat surprisingly, our embedding algorithm is not only able to embed human face images, but also successfully embeds non-face images from different classes.
    - 結論として、いくらか驚くことに、我々の埋め込みアルゴリズムは、人間の顔画像を埋め込むことだけではなく、異なるクラスからの顔以外の画像を埋め込むことに成功する。

- Therefore, we continue our investigation by analyzing the quality of the embedding to see if the embedding is semantically meaningful.
    - それ故に、埋め込みが意味論的に意味のあるものであるかどうかを確認するために [to see if]、埋め込みの品質を分析することによって、我々の調査を継続する。

- To this end, we propose to use three basic operations on vectors in the latent space: linear interpolation, crossover, and adding a vector and a scaled difference vector.
    - この目的を達成するために、我々は３つの基本的なベクトル演算を使用することを提案する。
    - 線形補間、線形交差、ベクトルの加算、異なるベクトルのスケーリング

- These operations correspond to three semantic image processing applications: morphing, style transfer, and expression transfer.
    - これらの演算は、３つの意味論的な画像処理アプリケーションに一致する。
    - モーフィング、スタイル変換、表情変換

- As a result, we gain more insight into the structure of the latent space and can solve the mystery why even instances of non-face images such as cars can be embedded.
    - その結果、潜在空間の構造についてより多くの洞察が得られ、自動車のような顔以外の画像の物体でさえ埋め込むことができる理由の謎を解決することができます。

---

- Our contributions include:
    - An efficient embedding algorithm which can map a given image into the extended latent space W+ of a pre-trained StyleGAN.

    - We study multiple questions providing insight into the structure of the StyleGAN latent space, e.g.: What type of images can be embedded? What type of faces can be embedded? What latent space can be used for the embedding?

    - We propose to use three basic operations on vectors to study the quality of the embedding. As a result, we can better understand the latent space and how dif- ferent classes of images are embedded. As a byprod- uct, we obtain excellent results on multiple face image editing applications including morphing, style transfer, and expression transfer.


# ■ 結論

## x. 論文の項目名 (Conclusion)


# ■ 何をしたか？詳細

## x. 論文の項目名


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


