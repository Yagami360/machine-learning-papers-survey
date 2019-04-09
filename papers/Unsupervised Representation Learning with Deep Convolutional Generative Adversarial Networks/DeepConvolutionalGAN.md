## 0. 論文
- 論文リンク：[[1511.06434] Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)
- 論文投稿日付：2015/11/19(v1), 2016/xx/07(v2)
- 著者：Alec Radford, Luke Metz, Soumith Chintala
- categories / Subjects：Machine Learning (cs.LG); Computer Vision and Pattern Recognition (cs.CV)

## 概要 [Abstract]

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

## 1. イントロダクション

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
    


## 7. 結論 [CONCLUSION AND FUTURE WORK]
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

<br>

- ACKNOWLEDGMENTS : We are fortunate and thankful for all the advice and guidance we have received during this work, especially that of Ian Goodfellow, Tobias Springenberg, Arthur Szlam and Durk Kingma. Additionally we’d like to thank all of the folks at indico for providing support, resources, and conversations, especially the two other members of the indico research team, Dan Kuster and Nathan Lintz. Finally, we’d like to thank Nvidia for donating a Titan-X GPU used in this work.


## 3. APPROACH AND MODEL ARCHITECTURE

## 4. DETAILS OF ADVERSARIAL TRAINING

## 実験結果

### 5. EMPIRICAL VALIDATION OF DCGANS CAPABILITIES

### 6. INVESTIGATING AND VISUALIZING THE INTERNALS OF THE NETWORKS


## 2. 関連研究 [RELATED WORK]

