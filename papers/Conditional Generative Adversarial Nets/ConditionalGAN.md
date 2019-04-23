# ■ 論文
- 論文タイトル：「Conditional Generative Adversarial Nets」
- 論文リンク：https://arxiv.org/abs/1411.1784
- 論文投稿日付：2014/11/06
- 著者：Mehdi Mirza, Simon Osindero
- categories：

# ■ 概要（何をしたか？）

## ABSTRACT
- Generative Adversarial Nets [8] were recently introduced as a novel way to train generative models.
    - Generative Adversarial Nets [8] は、最近、生成モデルを学習するための新手の [novel] 方法として紹介された。

- In this work we introduce the conditional version of generative adversarial nets, which can be constructed by simply feeding the data, y, we wish to condition on to both the generator and discriminator.
    - この研究では、generative adversarial nets の条件付きバージョンを紹介する。
    - （これは、）単にデータを供給する [feed] ことによって、構築される [constructed]（手法である。）
    - 生成器と識別器の両方で条件づけしたい y。

- We show that this model can generate MNIST digits conditioned on class labels.
    - 我々は、このモデルが、クラスラベルで条件付けされた MNIST の数字を生成出来ることを示す。

- We also illustrate how this model could be used to learn a multi-modal model, and provide preliminary examples of an application to image tagging in which we demonstrate how this approach can generate descriptive tags which are not part of training labels.
    - 我々はまた、このモデルがどうのようにして、マルチモードの [multi-modal] モデルを学習したのかを説明する [illustrate]。
    - そして、画像タグ付けへの応用の準備的な [preliminary] 例を提供する。
    - （この例というのは、）このアプローチが、どのようにして、学習ラベルの一部ではないような、記述的な [descriptive] タグを生成することの出来るのかを説明する。


# ■ イントロダクション（何をしたいか？）

## 1. Introduction

- Generative adversarial nets were recently introduced as an alternative framework for training generative models in order to sidestep the difficulty of approximating many intractable probabilistic computations.
    - 多くの手に負えない [intractable] 確率的な計算 [computations] を回避する [sidestep] ために、[in order to]
    - GAN は最近、生成モデルの学習のための、代わりとなるフレームワークとして紹介された。

- Adversarial nets have the advantages that Markov chains are never needed, only backpropagation is used to obtain gradients, no inference is required during learning, and a wide variety of factors and interactions can easily be incorporated into the model.
    - 敵対的ネットワークは、マルコフ連鎖を決して必要とせず、勾配を得るために誤差逆伝搬だけでよいという利点がある。
    - 学習の間、推論 [inference] を要求しない。
    - そして、多種多様な要因と相互作用 [interactions] が、モデルの中に容易に組み込むこと [incorporate] が出来る（という利点が存在する。）

- Furthermore, as demonstrated in [8], it can produce state of the art log-likelihood estimates and realistic samples.
    - 更には、[8] で実証されているように、対数尤度とリアルなサンプル（生成）の SOTA を生み出すことが出来る。

- In an unconditioned generative model, there is no control on modes of the data being generated.
    - 条件付けされていない生成モデルにおいては、生成されたデータのモードを制御できない。

- However, by conditioning the model on additional information it is possible to direct the data generation process.
    - しかしながら、追加の情報に基づいて、モデルを条件付けすることによって、データの生成過程を指示する [direct] ことが可能となる。

- Such conditioning could be based on class labels, on some part of data for inpainting like [5], or even on data from different modality.
    - このような条件付けは、クラスラベルや、
    - [5] のような画像の修復 [inpainting] ための、データのいくつかの一部、
    - 或いは、異なる様式 [modality] からのデータにさえ、
    - に基づくことが出来る。

<br>

- In this work we show how can we construct the conditional adversarial net.
    - この研究では、どのようにして、conditional adversarial net を構築するのかを示す。

- And for empirical results we demonstrate two set of experiment.
    - そして、経験的な結果として、２つの実験を実演する。

- One on MNIST digit data set conditioned on class labels and one on MIR Flickr 25,000 dataset [10] for multi-modal learning.
    - １つは、クラスラベルをもつ MNIST 数字データセットで、
    - そしてもう１つは、マルチモードの学習のための MIR Flickr 25,000 dataset [10] である。


# ■ 結論

## 5 Future Work

- The results shown in this paper are extremely preliminary, but they demonstrate the potential of conditional adversarial nets and show promise for interesting and useful applications.
    - この論文で示された結論は、極端に準備的な [preliminary] ものである。
    - しかし、それらは、CGAN の可能性を実証し、興味深く便利な応用を約束することを示す。

- In future explorations between now and the workshop we expect to present more sophisticated models, as well as a more detailed and thorough analysis of their performance and characteristics.
    - 今から学会 [workshop] までの間の将来の調査において、我々は、
    - より洗練されたモデルだけでなく、より詳細で、パフォーマンスと特性の徹底的な [thorough] 分析も、[as well as]
    - 与えることを期待する。

- Also, in the current experiments we only use each tag individually.
    - また、現在の実験において、我々は、各タグを個別に使用しているだけである。

- But by using multiple tags at the same time (effectively posing generative problem as one of ‘set generation’) we hope to achieve better results.
    - しかし、複数のタグを同時に使うことによって、よりよい結果を達成することを望んでいる。
    - （"set generation" の１つとして、効率的な生成問題を引き起こす [pose]。）

- Another obvious direction left for future work is to construct a joint training scheme to learn the language model.
    - さらなる研究に向けて残された、他の明確な方向は、言語モデルを学習するための、共同の学習計画 [scheme] を構築することである。

- Works such as [12] has shown that we can learn a language model for suited for the specific task.
    - [12] のような研究は、特定のタスクに適した [suited for] 言語モデルを学習することができるということを示している。


# ■ 何をしたか？詳細

## 3. Conditional Adversarial Nets

### 3.1 Generative Adversarial Nets

- Generative adversarial nets were recently introduced as a novel way to train a generative model.
    - GAN は、最近、生成モデルを学習する新手の方法として、紹介された。

- They consists of two ‘adversarial’ models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G.
    - それらは、２つの敵対的モデルを構成されている。
    - 即ち、データ分布からチャプチャーする生成器 G
    - そして、サンプルがGからではなく、学習データから来たのかという確率を推定するような、推定器 D

- Both G and D could be a non-linear mapping function, such as a multi-layer perceptron.
    - G と D の両方は、多層パーセプトロンのような、非線形の写像を行う関数である。

<br>

- To learn a generator distribution p_g over data data x, the generator builds a mapping function from a prior noise distribution p_z(z) to data space as G(z;θ_g).
    - データ x に渡っての、生成器の確率分布 p_g を学習するために、
    - 生成器は、前側のノイズ分布 p_z(z) からデータ空間 G(z;θ_g) への写像する関数を構築する。

- And the discriminator, D(x;θ_d), outputs a single scalar representing the probability that x came from training data rather than p_g.
    - そして、識別器 D(x;θ_d) は、x が p_g ではなく学習データから来たのかという確率を表現するような単一のスカラー値を出力する。

<br>

- G and D are both trained simultaneously:
    - G と D は、同時に学習される。

- we adjust parameters for G to minimize log(1 - D(G(z)) and adjust parameters for D to minimize logD(X), as if they are following the two-player min-max game with value function V (G;D):
    - log(1 - D(G(z)) を最小化するために G に対してのパラメーターを調整する。
    - そして、logD(X) を最小化するために D に対してのパラメーターを調整する。
    - （GとDが）、価値関数 V(G;D) での２人プレイヤーのミニマックスゲームに沿っているかのように [as if]。

![image](https://user-images.githubusercontent.com/25688193/56546804-5bcb1580-65b6-11e9-9093-fa39a7265bff.png)<br>

### 3.2 Conditional Adversarial Nets

- Generative adversarial nets can be extended to a conditional model if both the generator and discriminator are conditioned on some extra information y.
    - GAN は、条件付けされたモデルに拡張することが出来る。
    - もし、生成器と識別器の両方が、同じ追加の情報 y で条件付け出来るならば、

- y could be any kind of auxiliary information, such as class labels or data from other modalities.
    - y は、あらゆる種類の補助の [auxiliary] 情報である。
    - 例えば、クラスラベルや他の様式 [modality] からのデータなど

- We can perform the conditioning by feeding y into the both the discriminator and generator as additional input layer.
    - 追加の入力層として、識別器と生成器の両方の中に、y を供給することによって、条件付けを実行することが出来る。

- In the generator the prior input noise p_z(z), and y are combined in joint hidden representation, and the adversarial training framework allows for considerable flexibility in how this hidden representation is composed. <1>
    - 生成器においては、前側の入力ノイズ P_z(z) と y は、結合された隠れ表現になる。
    - そして、敵対的学習のフレームワークは、この隠れ表現がどのようにして構成されるのか？（＝隠れ表現の構成方法）に、著しい [considerable] 柔軟性を許容する。<1>

- <1> : For now we simply have the conditioning input and prior noise as inputs to a single hidden layer of a MLP, but one could imagine using higher order interactions allowing for complex generation mechanisms that would be extremely difficult to work with in a traditional generative framework.
    - <1> : 今のところ [for now]、MLP の１つの隠れ層に、入力そして、条件付け入力と前側の入力ノイズを、単純に持っている。
    - しかし、１つには、複雑な生成メカリズムを許容するような、より高い次元での相互作用を使用することを想像することが出来る。
    - （この複雑な生成メカリズムというのは、）伝統的な生成モデルのフレームワークにおいて、動作させることが極端に困難なもの。

- In the discriminator x and y are presented as inputs and to a discriminative function (embodied again by a MLP in this case).
    - 識別器において、 x と y は、入力として、
    - そして、識別器の関数（この場合で、MLP によって再度具体化される）に、
    - 提示される。

- The objective function of a two-player minimax game would be as Eq 2
    - ２人プレイヤーのミニマックスゲームの目的関数は、式２となる。

![image](https://user-images.githubusercontent.com/25688193/56548718-c3379400-65bb-11e9-8348-adee71c07775.png)<br>

> 従来の GAN と比較して、D(x) と G(z) の部分が、クラスラベル y に対しての条件付き確率 D(x|y), G(z|y) に変化しただけ。<br>

> クラスラベル y の入力としての付与が、条件付き確率で表現できる理由は？<br>
> → CGAN では要は、<br>
> discriminatorに「今は、6について本物か偽物かを判定してるんですよー」とか<br>
> generatorに「今は、3を書くという条件のもとに画像を生成してるんですよー」ということを教えてながら処理を行っているが、これは条件付き確率で表現できるため。<br>

- Fig 1 illustrates the structure of a simple conditional adversarial net.
    - > 図１は、単純な CGAN の構成を示している。

![image](https://user-images.githubusercontent.com/25688193/56548763-ebbf8e00-65bb-11e9-8040-5ead4a28f7c5.png)<br>


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）

## 4. Experimental Results

### 4.1 Unimodal

- We trained a conditional adversarial net on MNIST images conditioned on their class labels, encoded　as one-hot vectors.
    - 我々は、CGAN を、one-hot ベクトルでエンコードされた、（自身の）クラスラベルで条件付けされた MNIST 画像で学習した。

- In the generator net, a noise prior z with dimensionality 100 was drawn from a uniform distribution within the unit hypercube.
    - 生成器においては、次元数 100 の入力ノイズ z は、単一の球内の、一様分布から引き出されて [drawn] いる。

- Both z and y are mapped to hidden layers with Rectified Linear Unit (ReLu) activation [4, 11], with layer sizes 200 and 1000 respectively, before both being mapped to second, combined hidden ReLu layer of dimensionality 1200. 
    - z と y の両方は、Relu 活性化関数を用いて、隠れ層に写像されている。
    - レイヤーサイズが、各々 [respectively] で、200 と 1000 で、
    - （z と y の）両方が、２番目の層に写像される前に、
    - 次元数 1200 の結合された隠れ層 Relu

- We then have a final sigmoid unit layer as our output for generating the 784-dimensional MNIST samples.
    - 784次元の MNIST サンプルを生成するための出力として、最終的な sigmoid ユニットをもつ。

- The discriminator maps x to a maxout [6] layer with 240 units and 5 pieces, and y to a maxout layer with 50 units and 5 pieces.
    - 識別器は、

- Both of the hidden layers mapped to a joint maxout layer with 240 units and 4 pieces before being fed to the sigmoid layer.
    - xxx

- (The precise architecture of the discriminator is not critical as long as it has sufficient power; we have found that maxout units are typically well suited to the task.)
    - xxx



# ■ メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


