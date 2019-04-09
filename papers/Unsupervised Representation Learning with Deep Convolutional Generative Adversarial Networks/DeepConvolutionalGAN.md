## ■ 論文
- 論文リンク：[[1511.06434] Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- 論文投稿日付：2015/11/19(v1), 2016/xx/07(v2)
- 著者：Alec Radford, Luke Metz, Soumith Chintala
- categories / Subjects：Machine Learning (cs.LG); Computer Vision and Pattern Recognition (cs.CV)

## ■ 概要（何をしたか？）

### ABSTRACT

- In recent years, supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision applications. 
    - 最近では、CNN の構造をもつ教師あり学習が、コンピュータービジョンのアプリケーションにおいて、莫大に採用 [adoption] されている。

- Comparatively, unsupervised learning with CNNs has received less attention.
    - 比較的 [Comparatively]、CNN の構造を持つ教師なし学習は、あまり注目 [attention] されていない。

- In this work we hope to help bridge the gap between the success of CNNs for supervised learning and unsupervised learning. 
    - この論文では、我々は、教師あり学習と教師なし学習のための CNN の成功の間にあるギャップをなくす [bridge the gap] ことを望んでいる。

- We introduce a class of CNNs called deep convolutional generative adversarial networks (DCGANs), that have certain architectural constraints, and demonstrate that they are a strong candidate for unsupervised learning.
    - 我々は、DCGAN と呼ばれる CNN のクラスを紹介する。
    - これは、特定の [certain] アーキテクチャ的な制約を持ち、教師なし学習のための強い候補であるということを実証する。

- Training on various image datasets, we show convincing evidence that our deep convolutional adversarial pair learns a hierarchy of representations from object parts to scenes in both the generator and discriminator.
    - <font color="Pink">様々なデータセットの学習で、我々は、CNN 構造をもつ敵対的ネットワークのペアが、生成器と識別器の両方のシーンの対象部分からの表現の階層を学習するという説得力のある [convincing] 証拠を見せる</font>

- Additionally, we use the learned features for novel tasks - demonstrating their applicability as general image representations.
    - 加えて、我々は、（タスクの）適用性を一般的な画像表現として実証するような新型の [novel] タスクのために、学習された特徴量を使用する。

## ■ イントロダクション（何をしたいか？）

### 1. INTRODUCTION

- Learning reusable feature representations from large unlabeled datasets has been an area of active research.
    - ラベリングされていない巨大なデータセットからの再利用可能な特徴量の表現を学習することは、活発な研究分野になっている。

- In the context of computer vision, one can leverage the practically unlimited amount of unlabeled images and videos to learn good intermediate representations, which can then be used on a variety of supervised learning tasks such as image classification.
    - コンピュータービジョンでの文脈 [context] では、良い中間表現を学習するために、実質的に無限の量があるラベリングされていない画像やビデオのデータを利用することが出来る。
    - （それらは、）そのうえ [then]、画像分類といった、様々な [a variety of ] 教師あり学習のタスクとして使われることが出来る。

- We propose that one way to build good image representations is by training Generative Adversarial Networks (GANs) (Goodfellow et al., 2014), and later reusing parts of the generator and discriminator networks as feature extractors for supervised tasks.
    - 我々は、GAN や、教師ありタスクでの特徴抽出としてのその後の生成器と識別器のネットワークの構造を再利用している手法を学習することによっての、よい画像表現の構築するための１つの手法を提案する。

- GANs provide an attractive alternative to maximum likelihood techniques.
    - GAN は、最尤度手法の代わりとなる魅力的なテクニックを提供している。

- One can additionally argue that their learning process and the lack of a heuristic cost function (such as pixel-wise independent mean-square error) are attractive to representation learning.
    - （魅力の）１つは、（GANなどの）学習プロセスや、ヒューリスティックな（＝逐次的な）コスト関数（例えば、ピクセル単位の独立した平均２乗誤差）の欠如であると主張することが出来る。

- GANs have been known to be unstable to train, often resulting in generators that produce nonsensical outputs.
    - GAN は、生成器の結果がたびたび無意味な出力を生成してしまうといったように、学習が不安定であることが知られている。

- There has been very limited published research in trying to understand and visualize what GANs learn, and the intermediate representations of multi-layer GANs.
    - GAN が何を学習しているのかといったことを理解したり、可視化したり、多層の GAN の中間表現の公表された研究 [published research] は、非常に限られていた。

<br>

- In this paper, we make the following contributions
    - We propose and evaluate a set of constraints on the architectural topology of Convolutional GANs that make them stable to train in most settings. We name this class of architectures Deep Convolutional GANs (DCGAN)
    - We use the trained discriminators for image classification tasks, showing competitive performance with other unsupervised algorithms.
    - We visualize the filters learnt by GANs and empirically show that specific filters have learned to draw specific objects.
    - We show that the generators have interesting vector arithmetic properties allowing for easy manipulation of many semantic qualities of generated samples.

- この論文では、我々は、以下のような貢献 [contributions] を作り出す。
    - 我々は、（DCGAN を）多くの設定において学習を安定化させるような DCGAN のトポロジーのアーキテクチャ上での制約 [constraints] のセットを提案したり評価する。
    - 我々は、他の教師なし学習アルゴリズムとのパフォーマンス競争を見せるために、画像分類タスクのために学習された識別器を使用する。
    - 我々は、GAN によって学習されたフィルターを可視化する。そして、特定のフィルターが特定ののオブジェクトを描写することを学習していたということを実験的に [empirically] 見せる。
    - 我々は、生成器が、生成されたサンプルの多くの意味的な [semantic] 品質の簡単な操作 [manipulation] を許可するような、興味深い算術 [arithmetic] 特性のベクトルを持つことを示す。
    

### 3. APPROACH AND MODEL ARCHITECTURE

- Historical attempts to scale up GANs using CNNs to model images have been unsuccessful.
    - CNN を用いて GAN をスケールアップする歴史的な試みは、失敗に終わった。

- This motivated the authors of LAPGAN (Denton et al., 2015) to develop an alternative approach to iteratively upscale low resolution generated images which can be modeled more reliably.
    - このことは、LAPGAN の著者が、より信頼できるモデルで生成された低解像度画像の反復的なスケールアップ手法の代わりとなるアプローチを開発する動機になった。

- We also encountered difficulties attempting to scale GANs using CNN architectures commonly used in the supervised literature.
    - 我々はまた、教師あり文献？ [literature] の中で共通に使われている CNN のアーキテクチャを用いて GAN をスケールアップする試みの困難さに遭遇した。

- However, after extensive model exploration we identified a family of archi-tectures that resulted in stable training across a range of datasets and allowed for training higher resolution and deeper generative models.
    - しかしながら、広範囲 [extensive] のモデルの検索のあと、我々は、データセットの範囲に渡って安定的な学習という結果になり、そして、より高解像度でより深い生成モデルを学習することを許容するような、一連のアーキテクチャを特定した。[identify]

<br>

- Core to our approach is adopting and modifying three recently demonstrated changes to CNN architectures.
    - 我々のアプローチのコアは、３つの最近のアプローチを CNN のアーキテクチャに適用することです。
<br>

- The first is the all convolutional net (Springenberg et al., 2014) which replaces deterministic spatial pooling functions (such as maxpooling) with strided convolutions, allowing the network to learn its own spatial downsampling.
    - １つ目は、The All Convolution Net です。
    - （これは、）maxpooling のような決定論的な [deterministic] 空間的な [spatial] pooling 関数を、ストライドでの畳み込みに置き換える。
    - （このストライド畳み込みというのは、）ネットワークに、自身の空間的なダウンサンプリングを学習することを許容するようなストライド畳み込み。

- We use this approach in our generator, allowing it to learn its own spatial upsampling, and discriminator.
    - 我々は、このアプローチを、生成器と識別器に使用する。
    - （この生成器というのは、）自身の空間的なアンサンプリングを学習することを許容するような生成器。

<br>

- Second is the trend towards eliminating fully connected layers on top of convolutional features.
    - ２つ目は、畳込み特徴量の先頭の層に、全結合層を削除する [eliminating] というトレンド方向にあるものです。

- The strongest example of this is global average pooling which has been utilized in state of the art image classification models (Mordvintsev et al.). 
    - この最も強い例は、画像分類モデルの SOTA で利用された [be utilized] global average pooling です。

- We found global average pooling increased model stability but hurt convergence speed.
    - 我々は、global average pooling がモデルの安定性を増加させるが、収束スピードを低下させることを見つけ出した。

- A middle ground of directly connecting the highest convolutional features to the input and output respectively of the generator and discriminator worked well.
    - <font color="Pink">最も高い畳み込み特徴量を、生成器と識別器の各々 [respectively] を入力と出力に、直接的に接続する中間的な根拠 [ground of] はうまく動作した。</font>

- The first layer of the GAN, which takes a uniform noise distribution Z as input, could be called fully connected as it is just a matrix multiplication, but the result is reshaped into a 4-dimensional tensor and used as the start of the convolution stack.
    - 入力としての一様ノイズ分布 z を受け取る GAN の最初の層は、行列積のように全結合と呼ばれる事ができる。
    - しかし、結果は、４次元テンソルへの変形になる。そして、畳み込みスタックの先頭として使用される。

- For the discriminator, the last convolution layer is flattened and then fed into a single sigmoid output.
    - 識別器にとっては、最後の畳み込み層は、flattened で、１つのシグモイド関数の出力が送り込まれる [fed into]。

- See Fig. 1 for a visualization of an example model architecture.
    - モデルのアーキテクチャの例を可視化した、図１を見てください。

![image](https://user-images.githubusercontent.com/25688193/55780777-b27b1e80-5ae3-11e9-8b30-73fca0097e98.png)<br>
- > Figure 1: DCGAN generator used for LSUN scene modeling.
    - > 図１：LSUNシーンモデリングのツールを使用して書いた DCGAN

- > A 100 dimensional uniform distribution Z is projected to a small spatial extent convolutional representation with many feature maps.
    - > 一様分布 Z の 100 次元は、たくさんの特徴マップを持つ、小さな空間的な畳み込み表現に投影される。

- > A series of four fractionally-strided convolutions (in some recent papers, these are wrongly called deconvolutions) then convert this high level representation into a 64 × 64 pixel image.
    - > ４つの逆畳み込み [fractionally-strided convolutions] （最新の同じ論文では、deconvolutions と誤って呼ばれている。）のシリーズは、64 × 64 pixel の画像でのハイレベルな表現に変換する。

- > Notably, no fully connected or pooling layers are used.
    - > 注目すべきことは、全結合や pooling 層が１つも使われていないことです。

<br>

- Third is Batch Normalization (Ioffe & Szegedy, 2015) which stabilizes learning by normalizing the input to each unit to have zero mean and unit variance.

- This helps deal with training problems that arise due to poor initialization and helps gradient flow in deeper models.

- This proved critical to get deep generators to begin learning, preventing the generator from collapsing all samples to a single point which is a common failure mode observed in GANs.

- Directly applying batchnorm to all layers however, resulted in sample oscillation and model instability.

- This was avoided by not applying batchnorm to the generator output layer and the discriminator input layer.

- The ReLU activation (Nair & Hinton, 2010) is used in the generator with the exception of the output layer which uses the Tanh function.

- We observed that using a bounded activation allowed the model to learn more quickly to saturate and cover the color space of the training distribution.

- Within the discriminator we found the leaky rectified activation (Maas et al., 2013) (Xu et al., 2015) to work well, especially for higher resolution modeling.

- This is in contrast to the original GAN paper, which used the maxout activation (Goodfellow et al., 2013).

<br>

- Architecture guidelines for stable Deep Convolutional GANs
    - Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
    - Use batchnorm in both the generator and the discriminator.
    - Remove fully connected hidden layers for deeper architectures.
    - Use ReLU activation in generator for all layers except for the output, which uses Tanh.
    - Use LeakyReLU activation in the discriminator for all layers.



## ■ 結論（何をしたか？詳細）

### 7. CONCLUSION AND FUTURE WORK
- We propose a more stable set of architectures for training generative adversarial networks and we give evidence that adversarial networks learn good representations of images for supervised learning and generative modeling.
    - 我々は、GAN の学習のための、より安定したアーキテクチャのセットを提案する。そして、GAN が教師あり学習と生成モデルたの画像の良い表現を学習しているという証拠を与える。

- There are still some forms of model instability remaining - we noticed as models are trained longer they sometimes collapse a subset of filters to a single oscillating mode.
    - モデルの不安定性には、まだいくつかの型が残っている。
    - <font color="Pink">（即ち、）より長い時間学習されたモデルは、ときどき、フィルターのサブセットを、単一の振動モード [oscillating mode] に崩壊させることがあるということに、我々は気づいた。</font>

<br>

- Further work is needed to tackle this from of instability.
    - さらなる研究では、この不安定型についての取り組み [tackle] が必要がある。

- We think that extending this framework to other domains such as video (for frame prediction) and audio (pre-trained features for speech synthesis) should be very interesting.
    - 我々は、この枠組みを、動画（フレーム予想）や音声（会話のための事前学習された特徴量）といった分野に拡張することと、とても興味深いものだと考えている。

- Further investigations into the properties of the learnt latent space would be interesting as well.
    - 学習された潜在空間 [latent space] についてのさらなる調査も興味深い。

<br>

- ACKNOWLEDGMENTS : We are fortunate and thankful for all the advice and guidance we have received during this work, especially that of Ian Goodfellow, Tobias Springenberg, Arthur Szlam and Durk Kingma. Additionally we’d like to thank all of the folks at indico for providing support, resources, and conversations, especially the two other members of the indico research team, Dan Kuster and Nathan Lintz. Finally, we’d like to thank Nvidia for donating a Titan-X GPU used in this work.


## ■ 実験結果（主張の証明）・議論（手法の良し悪し）

### 5. EMPIRICAL VALIDATION OF DCGANS CAPABILITIES

### 6. INVESTIGATING AND VISUALIZING THE INTERNALS OF THE NETWORKS


## ■ 関連研究（他の手法との違い）・メソッド（実験方法）

### 4. DETAILS OF ADVERSARIAL TRAINING

### 2. RELATED WORK

