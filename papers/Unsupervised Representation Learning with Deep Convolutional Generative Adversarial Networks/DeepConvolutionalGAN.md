# ■ 論文

- 論文リンク：[[1511.06434] Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- 論文投稿日付：2015/11/19(v1), 2016/01/07(v2)
- 著者：Alec Radford, Luke Metz, Soumith Chintala
- categories / Subjects：Machine Learning (cs.LG); Computer Vision and Pattern Recognition (cs.CV)

# ■ 概要（何をしたか？）

## ABSTRACT

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

# ■ イントロダクション（何をしたいか？）

## 1. INTRODUCTION

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
    

## 3. APPROACH AND MODEL ARCHITECTURE

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
    - ３つ目は、Batch Normalization です。
    - （この Batch Normalization というのは、）<font color="Pink">入力を各々のユニットのゼロ平均、分散値で</font>正規化することによって学習を安定化させるようなものである。

- This helps deal with training problems that arise due to poor initialization and helps gradient flow in deeper models.
    - これは不十分 [poor] な初期化が原因で発生する学習問題に対処するのみ役立ち、より深いモデルでの勾配フローに役立つ。

- This proved critical to get deep generators to begin learning, preventing the generator from collapsing all samples to a single point which is a common failure mode observed in GANs.
    - これは、深い構造を持つ生成器を得るために、学習を始めるのに重要であることが証明されている。
    - （この学習というのは、）GAN で観測される共通の故障モードというような１つの点に、生成器が、全てのサンプルで崩壊するのを防ぐようなものである。

    > モード崩壊 [mode collapse] のことを言っている？

- Directly applying batchnorm to all layers however, resulted in sample oscillation and model instability.
    - すべての層に直接的に batchnorm を適用することはしかしながら、サンプルの発振？ [sample oscillation] やモデルの不安定化という結果になる。

- This was avoided by not applying batchnorm to the generator output layer and the discriminator input layer.
    - これは、batchnorm を、生成器の出力層と識別器の入力層に適用しないことによって、回避された。

- The ReLU activation (Nair & Hinton, 2010) is used in the generator with the exception of the output layer which uses the Tanh function.
    - ReLU 活性化関数は、Tanh 関数が使用されている出力層を除いて、生成器に使用されている。

- We observed that using a bounded activation allowed the model to learn more quickly to saturate and cover the color space of the training distribution.
    - 有界の活性化関数を使用することは、モデルに、<font color="Pink">学習分布の色空間を満たし [saturate] たりカバーしたりするために、より早く学習することを許容したこということを観測した。</font>

- Within the discriminator we found the leaky rectified activation (Maas et al., 2013) (Xu et al., 2015) to work well, especially for higher resolution modeling.
    - 識別器の中では、leaky Relu がよりうまくいくことを見つけ出した。
    - とりわけ、より高い解像度のモデリングでは、（うまくいった）

- This is in contrast to the original GAN paper, which used the maxout activation (Goodfellow et al., 2013).
    - maxout 活性化関数が使われているオリジナル GAN 論文とは対照的である。[in contrast]

<br>

- Architecture guidelines for stable Deep Convolutional GANs
    - Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
    - Use batchnorm in both the generator and the discriminator.
    - Remove fully connected hidden layers for deeper architectures.
    - Use ReLU activation in generator for all layers except for the output, which uses Tanh.
    - Use LeakyReLU activation in the discriminator for all layers.

- DCGAN を安定化させるためのアーキテクチャのガイドライン
    - 各々のプーリング層を、識別器のストライド畳み込みや生成器の fractional-strided convolutions に取り替える。
    - 生成器と識別器で共に、batchnorm を使用する。
    - 深いアーキテクチャ構造を持つ、全結合の隠れ層を除外する。
    - 生成器において、Tanh を使用する出力以外は、全ての層で Relu を使用する。
    - 識別器において、全ての層で　LeaklyRelu を使用する。


# ■ 結論（何をしたか？詳細）

## 7. CONCLUSION AND FUTURE WORK
- We propose a more stable set of architectures for training generative adversarial networks and we give evidence that adversarial networks learn good representations of images for supervised learning and generative modeling.
    - 我々は、GAN の学習のための、より安定したアーキテクチャのセットを提案する。そして、GAN が教師あり学習と生成モデルための、画像の良い表現を学習しているという証拠を与える。

- There are still some forms of model instability remaining - we noticed as models are trained longer they sometimes collapse a subset of filters to a single oscillating mode.
    - モデルの不安定性には、まだいくつかの型が残っている。
    - <font color="Pink">（即ち、）より長い時間学習されたモデルは、ときどき、フィルターのサブセットを、単一の振動モード [oscillating mode] に崩壊させることがあるということに、（⇒モード崩壊のこと？）我々は気づいた。</font>

<br>

- Further work is needed to tackle this from of instability.
    - さらなる研究では、この不安定型についての取り組み [tackle] が必要がある。

- We think that extending this framework to other domains such as video (for frame prediction) and audio (pre-trained features for speech synthesis) should be very interesting.
    - 我々は、この枠組みを、動画（フレーム予想）や音声（会話のための事前学習された特徴量）といった分野に拡張することと、とても興味深いものだと考えている。

- Further investigations into the properties of the learnt latent space would be interesting as well.
    - 学習された潜在空間 [latent space] についてのさらなる調査も興味深い。

<br>

- ACKNOWLEDGMENTS : We are fortunate and thankful for all the advice and guidance we have received during this work, especially that of Ian Goodfellow, Tobias Springenberg, Arthur Szlam and Durk Kingma. Additionally we’d like to thank all of the folks at indico for providing support, resources, and conversations, especially the two other members of the indico research team, Dan Kuster and Nathan Lintz. Finally, we’d like to thank Nvidia for donating a Titan-X GPU used in this work.


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）

## 5. EMPIRICAL VALIDATION OF DCGANS CAPABILITIES

### 5.1 CLASSIFYING CIFAR-10 USING GANS AS A FEATURE EXTRACTOR

- One common technique for evaluating the quality of unsupervised representation learning algorithms is to apply them as a feature extractor on supervised datasets and evaluate the performance of linear models fitted on top of these features.
    - 教師なし学習アルゴリズムを評価するための１つの共通のテクニックは、それらを教師あり学習の元での？、特徴抽出機として適応することです。
    - そして、それらの特徴量のトップ？で適合された線形モデルのパフォーマンスを評価することです。

<br>

- On the CIFAR-10 dataset, a very strong baseline performance has been demonstrated from a well tuned single layer feature extraction pipeline utilizing K-means as a feature learning algorithm.
    - CIFAR-10 データセットでは、とても強いベースライン性能（基準性能）が、十分によくチューニングされた単一の特徴抽出パイプラインから、実演されている。
    - （この特徴抽出パイプラインというのは、）特徴量の学習アルゴリズムとしての K-means を利用したものである。

- When using a very large amount of feature maps (4800) this technique achieves 80.6% accuracy.
    - 特徴マップ(4800毎) というとても大きな量を使用するときには、このテクニックは、80.6% の正解率を達成する。

- An unsupervised multi-layered extension of the base algorithm reaches 82.0% accuracy (Coates &Ng, 2011).
    - ベースアルゴリズムの教師なしでの多層への拡張は、82.0% の正解率に到達する。

- To evaluate the quality of the representations learned by DCGANs for supervised tasks, we train on Imagenet-1k and then use the discriminator’s convolutional features from all layers, maxpooling each layers representation to produce a 4 × 4 spatial grid.
    - 教師ありタスクのための、DCGAN によって学習された表現の質を評価するために、我々は Imagenet-1k を学習する。そしてその後、識別器の全ての層からの畳み込み層を使用する。
    - <font color="Pink">4 × 4 の分割されたグリッドを処理するための、各層の表現を maxpooling する？</font>

- These features are then flattened and concatenated to form a 28672 dimensional vector and a regularized linear L2-SVM classifier is trained on top of them.
    - これらの特徴量は、28672 次元のベクトルに平坦化処理や結合処理される。
    - そして、それら特徴量のトップで学習されたL2-SVM 分類器に正則化 [regularized] される。

- This achieves 82.8% accuracy, out performing all K-means based approaches.
    - この手法は、82.8% の正解率を達成し、
    - 全ての K-means をベースとするアプローチを実行しない。

- Notably, the discriminator has many less feature maps (512 in the highest layer) compared to K-means based techniques, but does result in a larger total feature vector size due to the many layers of 4 × 4 spatial locations.
    - とりわけ、識別器は、K-meansをベースとするテクニックと比較して、はるかに少ない特徴マップ（最も高い層で、512枚）ですむ。
    - しかしながら、<font color="Pink">4 × 4 の分割された位置の多くの層によって、より大きな特徴ベクトルサイズ（を持つ）という結果となる。</font>

- The performance of DCGANs is still less than that of Exemplar CNNs (Dosovitskiy et al., 2015), a technique which trains normal discriminative CNNs in an unsupervised fashion to differentiate between specifically chosen, aggressively augmented, exemplar samples from the source dataset.
    - DCGAN のパフォーマンスは、Exemplar CNNs のそれより、まだ低い。
    - （この Exemplar CNNs というのは、）データセットの資源からの典型的な [exemplar] サンプル、積極的に増加された [augmented] サンプル、特別に抽出されたサンプル、との間を区別する [differentiate] ために、教師なしファッションデータで、通常の識別を行う CNN を学習するテクニックである。

- Further improvements could be made by finetuning the discriminator’s representations, but we leave this for future work.
    - さらなる改善は、識別器の表現のファインチューニング（＝既存のモデルの一部を再利用して、新しいモデルを構築する手法）で作り出すことが出来た。
    - しかし、我々は、これを将来の研究に残しておく。

- Additionally, since our DCGAN was never trained on CIFAR-10 this experiment also demonstrates the domain robustness of the learned features.
    - 加えて、我々の DCGAN が、CIFAR-10 で学習されることはないので [since]、この実験はまた、学習された特徴量の分野のロバスト性 [robustness] を実証している。

- Table 1: CIFAR-10 classification results using our pre-trained model.
    - Our DCGAN is not pretrained on CIFAR-10, but on Imagenet-1k, and the features are used to classify CIFAR-10 images.

- 表１：事前学習されたモデルを用いた CIFAR-10 の分類結果。
    - 我々の DCGAN は、CIFAR-10 で事前学習されていない。
    - しかし、Imagenet-1k や？
    - 特徴量では、CIFAR-10 画像を分類するために使われている。

![image](https://user-images.githubusercontent.com/25688193/55852405-d39d4700-5b97-11e9-9f55-fc738bb57aa1.png)<br>


### 5.2 CLASSIFYING SVHN DIGITS USING GANS AS A FEATURE EXTRACTOR

- On the StreetView House Numbers dataset (SVHN)(Netzer et al., 2011), we use the features of the discriminator of a DCGAN for supervised purposes when labeled data is scarce.
    - StreetView House Numbers dataset (SVHN) では、ラベルデータが不足している [scarce] ときの目的のために、我々は、DCGAN の識別器の教師データとしての特徴量を使用する。

- Following similar dataset preparation rules as in the CIFAR-10 experiments, we split off a validation set of 10,000 examples from the non-extra set and use it for all hyperparameter and model selection.
    - CIFAR-10 の実験とよく似たデータセット準備のルールに従って [following]、我々は、非追加セットから、10,000 サンプルのバリデーションデータセットを分割する。
    - そして、ハイパーパラメータやモデルの選択に使用する。

- 1000 uniformly class distributed training examples are randomly selected and used to train a regularized linear L2-SVM classifier on top of the same feature extraction pipeline used for CIFAR-10.
    - <font color="Pink">1000 個の一様なトレーニングサンプルの分布をもつクラスは、
    - CIFAR-10 のために使用されているものと同じ特徴抽出パイプラインのトップ？で、正規化された線形 L2-SVM 分類器を学習するために、
    - ランダムに選択され、使用されたものである。</font>

- This achieves state of the art (for classification using 1000 labels) at 22.48% test error, improving upon another modifcation of CNNs designed to leverage unlabled data (Zhao et al., 2015).
    - この手法は、22.48% の test error での、（1000個のラベルを用いた分類問題での）SOTA を達成する。
    
- Additionally, we validate that the CNN architecture used in DCGAN is not the key contributing factor of the model’s performance by training a purely supervised CNN with the same architecture on the same data and optimizing this model via random search over 64 hyperparameter trials (Bergstra & Bengio,2012).
    - 加えて、DCGAN で使用されている CNN のアーキテクチャが、モデルのパフォーマンスのキーポイント要因ではないことを、検証した。[validate]
    - <font color="Pink">（これは）純粋に、同じデータの同じアーキテクチャと共に、教師ありで（学習された）CNN で学習すること、そして、64 個のハイパーパラメータの試行を超えたランダムサーチ経由でこのモデルを最適化することによって、（検証した）</font>

- It achieves a signficantly higher 28.87% validation error.
    - これは、28.87% validation error より大幅に [signficantly] 高いスコアを達成した。

![image](https://user-images.githubusercontent.com/25688193/55924793-3816df80-5c46-11e9-818c-fe6e6da6a030.png)<br>


## 6. INVESTIGATING AND VISUALIZING THE INTERNALS OF THE NETWORKS

- We investigate the trained generators and discriminators in a variety of ways.
    - 我々は、様々な方法で、学習された生成器と識別器を調査した。

- We do not do any kind of nearest neighbor search on the training set. 
    - 学習用データセットに対して、どんな種類の最近傍探索 [nearest neighbor search] も行っていない。

- Nearest neighbors in pixel or feature space are trivially fooled (Theis et al., 2015) by small image transforms.
    - ピクセルの中や特徴空間での最近傍探索は、小さな画像の変換によって、些細に [trivially] 騙される。[be fooled]

- We also do not use log-likelihood metrics to quantitatively assess the model, as it is a poor (Theis et al., 2015) metric.
    - 我々はまた、モデルを定量的に [quantitatively ] 評価する [assess] ための対数尤度の測定基準を使用しない。
    - 測定基準であるように、

### 6.1 WALKING IN THE LATENT SPACE

- The first experiment we did was to understand the landscape of the latent space.
    - 我々が行った最初の実験は、潜在空間 [latent spece] の景色 [landscape] を理解することです。

- Walking on the manifold that is learnt can usually tell us about signs of memorization (if there are sharp transitions) and about the way in which the space is hierarchically collapsed.
    - 学習された多様体 [manifold] 上を歩くことは、通常、<font color="Pink">記憶の兆候について、我々に教えてくれる。（もしそれらの多様体が急激な変化があれば）
    - そして、空間が階層的に崩壊する方法について、教えてくれる。</font>

    > 多様体学習の話？

- If walking in this latent space results in semantic changes to the image generations (such as objects being added and removed), we can reason that the model has learned relevant and interesting representations.
    - もし、この潜在空間の中を歩くことが、画像生成を意味論的に [semantic] 変えるという結果（例えば、オブジェクトが追加されたり、削除されたりするような変化）となるならば、我々は、モデルが関連して興味深い表現を学習したと考える [reason] ことができる。

- The results are shown in Fig.4.

![image](https://user-images.githubusercontent.com/25688193/55925916-364f1b00-5c4a-11e9-982d-68100d947e4f.png)<br>

- > Figure 4: Top rows: Interpolation between a series of 9 random points in Z show that the space learned has smooth transitions, with every image in the space plausibly looking like a bedroom.
    - > 図４：
    - > 先頭の行：入力ノイズ Z のランダム点の系列の間での補間は、学習された空間が、スムーズに変化していくことを示している。
    - > （このスムーズな変化は、）空間内の全ての画像がベットルームのようにもっともらしく [plausibly] 見えるといったような、（スムーズな変化で）

- In the 6th row, you see a room without a window slowly transforming into a room with a giant window.

- In the 10th row, you see what appears to be a TV slowly being transformed into a window.

### 6.2 VISUALIZING THE DISCRIMINATOR FEATURES

- Previous work has demonstrated that supervised training of CNNs on large image datasets results in very powerful learned features (Zeiler & Fergus, 2014).
    - 以前の研究では、巨大な画像データセットでの CNN の教師あり学習が、とてもパワフルに学習された特徴量になるという結果を実証した。

- Additionally, supervised CNNs trained on scene classification learn object detectors (Oquab et al., 2014).
    - 加えて、シーンの分類問題において学習された CNN が、物体検出を学習した。

- We demonstrate that an unsupervised DCGAN trained on a large image dataset can also learn a hierarchy of features that are interesting.
    - 我々は、巨大な画像データセットで学習した教師なし学習の DCGAN がまた、興味深い特徴の階層を学習することが出来ることを実証する。

- Using guided backpropagation as proposed by (Springenberg et al., 2014), we show in Fig.5 that the features learnt by the discriminator activate on typical parts of a bedroom, like beds and windows.
    - (Springenberg et al., 2014) によって提案された guided backpropagation を使用して、
    - 我々は、識別機によって学習された特徴量が、ベットルームやベッド、窓の典型的な [typical] 部分で活性化するということを図５に示す。

> guided backpropagation<br>
> ![image](https://user-images.githubusercontent.com/25688193/55927584-8b8e2b00-5c50-11e9-8024-8d5d42eeaaad.png)<br>

- For comparison, in the same figure, we give a baseline for randomly initialized features that are not activated on anything that is semantically relevant or interesting.
    - 比較のために、同じ図に、意味的に関連して興味深いもので活性化していないような、ランダムに初期化された特徴量のベースラインを与える。

![image](https://user-images.githubusercontent.com/25688193/55927173-035b5600-5c4f-11e9-90c0-65cc43dcbf7c.png)<br>

- > Figure 5: On the right, guided backpropagation visualizations of maximal axis-aligned responses for the first 6 learned convolutional features from the last convolution layer in the discriminator.
    - > 図５：右側の図では、座標軸に平行な [axis-aligned] 最大軸の guided backpropagation での可視化は、
    - > 識別器の最後の畳み込み層からの、畳み込み特徴量を学習した最初の６つのを応答している？

- > Notice a significant minority of features respond to beds - the central object in the LSUN bedrooms dataset.
    - > 少数とはいえ無視できない数の [significant minority of] 特徴量が、ベッドに反応していることに注意してください。つまり、LSUN bedrooms dataset の中央のオブジェクト。

- > On the left is a random filter baseline. Comparing to the previous responses there is little to no discrimination and random structure.
    - > 左側は、ランダムフィルタのベースラインです。前の反応と比較すると識別するものやランダム構造はほとんどない。

### 6.3 MANIPULATING THE GENERATOR REPRESENTATION

#### 6.3.1 FORGETTING TO DRAW CERTAIN OBJECTS

- In addition to the representations learnt by a discriminator, there is the question of what representations the generator learns.
    - 識別器によって学習された表現に加えて、生成器が何の表現を学習しているのかという疑問が存在する。

- The quality of samples suggest that the generator learns specific object representations for major scene components such as beds, windows, lamps, doors, and miscellaneous furniture.
    - サンプルの品質は、生成器が、例えば、ベッド・窓・ドアなどの主なシーンのコンポーネントのために、特定のオブジェクトの表現を学習したということを提案する。

- In order to explore the form that these representations take, we conducted an experiment to attempt to remove windows from the generator completely.
    - それらの表現を受け取る型を発見するために、我々は、生成器からウインドウを完全に除外する試みるための実験を行った。

<br>

- On 150 samples, 52 window bounding boxes were drawn manually.
    - 150個のサンプルで、52個のウインドウのバウンディングボックスを、手書きで [manually] 描いた。

- On the second highest convolution layer features, logistic regression was fit to predict whether a feature activation was on a window (or not), by using the criterion that activations inside the drawn bounding boxes are positives and random samples from the same images are negatives. 
    - ２番目に高い畳み込み層では、ロジスティック回帰は、特徴量の活性化がウインドウ上に存在しているのか、或いはそうでないかということを、予想するために適合する。
    - 描写されたバウンディングボックスの中での活性化が正であり、同じ画像からのランダムサンプルが負である、という評価を用いることによって、

- Using this simple model, all feature maps with weights greater than zero ( 200 in total) were dropped from all spatial locations.
    - このシンプルなモデルを使うことで、全ての特徴マップ

- Then, random new samples were generated with and without the feature map removal.

<br>

- The generated images with and without the window dropout are shown in Fig.6, and interestingly, the network mostly forgets to draw windows in the bedrooms, replacing them with other objects.

#### 6.3.2 VECTOR ARITHMETIC ON FACE SAMPLES


# ■ 関連研究（他の手法との違い）・メソッド（実験方法）

## 4. DETAILS OF ADVERSARIAL TRAINING

- We trained DCGANs on three datasets, Large-scale Scene Understanding (LSUN) (Yu et al., 2015), Imagenet-1k and a newly assembled Faces dataset. 
    - 我々は、DCGAN を３つのデータセットで学習した。
    - （この３つのデータセットというのは、）Large-scale Scene Understanding (LSUN)、Imagenet-1k、a newly assembled Faces dataset です。

- Details on the usage of each of these datasets are given below.
    - これらのデータセットの使用法 [usage] の詳細は、以下に示している。

> 記載中...

## 2. RELATED WORK

> 記載中...