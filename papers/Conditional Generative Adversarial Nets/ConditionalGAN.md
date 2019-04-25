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
    - 識別器は、入力画像 x を 240 個のユニットと5ピースと共に、maxout 層 [6] へ写像する。
    - そして、クラスラベル y を、50 個のユニットと5ピースと共に、maxout 層へ写像する。

![image](https://user-images.githubusercontent.com/25688193/56703728-299bee00-6745-11e9-84a7-962272c0e3a5.png)<br>

- Both of the hidden layers mapped to a joint maxout layer with 240 units and 4 pieces before being fed to the sigmoid layer.
    - 両方の隠れ層は、シグモイド層へ供給される前に、
    - 240 個のユニットと4ピースと共に、 結合された maxout 層へ写像する。

- (The precise architecture of the discriminator is not critical as long as it has sufficient power; we have found that maxout units are typically well suited to the task.)
    - 識別器の正確なアーキテクチャは、十分な [sufficient] 能力をもつ限り、重要ではない。
    - 即ち、我々は、一般的に [typically]、maxout ユニットが、そのタスクにうまく適しているということを見つけ出した。

<br>

- The model was trained using stochastic gradient decent with mini-batches of size 100 and initial learning rate of 0:1 which was exponentially decreased down to :000001 with decay factor of 1:00004.
    - モデルは、サイズ 100 のミニバッチで、確率的勾配法を用いて学習される。
    - そして、0 ~ 1 の初期の学習率は、1:00004 の減衰項で、指数関数的に :000001 に減少する。

- Also momentum was used with initial value of :5 which was increased up to 0:7.
    - モーメンタムが、:5 の初期値として使用される。

- Dropout[9] with probability of 0.5 was applied to both the generator and discriminator.
    - 0.5 の確率でのドロップアウト [9] は、生成器と識別器の両方に適用される。

- And best estimate of log-likelihood on the validation set was used as stopping point.
    - そして、検証用データセットでの対数尤度のベストな推定は、停止点として使用される。

<br>

- Table 1 shows Gaussian Parzen window log-likelihood estimate for the MNIST dataset test data.
    - 表１は、MNIST のテストデータに対しての、対数尤度のカーネル密度推定 [Gaussian Parzen window] を示している。

> カーネル密度推定：データ標本から、その母集団である確率分布を推定する手法。<br>

> ![image](https://user-images.githubusercontent.com/25688193/56705915-edb95680-674d-11e9-8e91-ada8dca04b01.png)<br>

![image](https://user-images.githubusercontent.com/25688193/56705770-59e78a80-674d-11e9-9946-27b1cdf1ddf6.png)<br>

- > Table 1: Parzen window-based log-likelihood estimates for MNIST. We followed the same procedure as [8] for computing these values.


- 1000 samples were drawn from each 10 class and a Gaussian Parzen window was fitted to these samples.
    - 10 個の各クラスからの、1000 個のサンプルが示されている。
    - そして、カーネル密度推定は、これらのサンプルに適合されている。

- We then estimate the log-likelihood of the test set using the Parzen window distribution.
    - カーネル密度分布を用いたテストデータの対数尤度を推定する。

- (See [8] for more details of how this estimate is constructed.)
    - この推定がどのようにして構築されているかの詳細は、[8] を見てください。

<br>

- The conditional adversarial net results that we present are comparable with some other network based, but are outperformed by several other approaches – including non-conditional adversarial nets.
    - 我々が提供する CGAN の結果は、他のネットワークベースと比較出来る.
    - しかし、いくつかの他のアプローチによって、実行される。
    - （このアプローチというのは、）条件付けでない敵対的ネットワークを含んでいるような（アプローチ）

- We present these results more as a proof-of-concept than as demonstration of efficacy, and believe that with further exploration of hyper-parameter space and architecture that the conditional model should match or exceed the non-conditional results.
    - 我々は、効果 [efficacy] の実証よりも、概念実証 [proof-of-concept] で、これらの結果を提示する。
    - そして、ハイパーパラメータの空間やアーキテクチャのさらなる探索で、
    - 条件付きモデルは、条件付けなしのモデルとマッチする、或いは、上回る [exceed]
    - ということを信じている。

- Fig 2 shows some of the generated samples. Each row is conditioned on one label and each column is a different generated sample.
    - 図２は、いくつかの生成されたサンプルを示している。
    - 各行は、１つのラベルで条件付けされており、各列は異なる生成されたサンプルになっている。

![image](https://user-images.githubusercontent.com/25688193/56709244-31668d00-675b-11e9-96dd-2edaf07d6f27.png)<br>

### 4.2 Multimodal

- Photo sites such as Flickr are a rich source of labeled data in the form of images and their associated user-generated metadata (UGM) — in particular user-tags.
    - Flickr のような写真サイトは、
    - 画像の型や、画像に関連付けられた [associated] ユーザーが生成したメタデータ（UGM）（とりわけ、ユーザータグ）において、
    - 豊富なリソースのラベリングされたデータがある。

- User-generated metadata differ from more ‘canonical’ image labelling schems in that they are typically more descriptive, and are semantically much closer to how humans describe images with natural language rather than just identifying the objects present in an image.
    - ユーザーが生成したメタデータは、より標準的な [canonical] 画像ラベリングのスキーマ（図式、計画）とは異なる。
    - 画像内のオブジェクトを単に識別することよりも、
    - 画像を自然言語で記述している方法が、
    - 一般的に、より説明的に、意味的にはるかに近い
    - という点において、

- Another aspect of UGM is that synoymy is prevalent and different users may use different vocabulary to describe the same concepts — consequently, having an efficient way to normalize these labels becomes important.
    - UGM の他の側面は、同義語 [synoymy] が流行しており [prevalent]、
    - 異なるユーザーが、同じコンセプトを記述するために、異なる用語を用いるかもしれないということです。
    - 結果的に、それらのラベルを標準化する効果的な方法を持つことが、重要となる。

- Conceptual word embeddings [14] can be very useful here since related concepts end up being represented by similar vectors.
    - 単語埋め込みの概念は、関連した概念が、同じようなベクトルによって、表現されるという結果になるので、とても便利である。

<br>

- In this section we demonstrate automated tagging of images, with multi-label predictions, using conditional adversarial nets to generate a (possibly multi-modal) distribution of tag-vectors conditional on image features.
    - このセクションでは、画像の自動タグ付けを実証する。
    - 複数のラベル予想で、
    - 画像特徴量で条件付けされたタグベクトルの確率分布（可能であれば、マルチモードの確率分布）を生成するために、CGAN を用いて（実証する。）

<br>

- For image features we pre-train a convolutional model similar to the one from [13] on the full ImageNet dataset with 21,000 labels [15].
    - 画像特徴量のために、
    - 21000 個のラベルをもつ full ImageNet dataset [15] において、[13] からの１つによく似ているような、
    - 畳み込みモデルを、事前に学習している。

- We use the output of the last fully connected layer with 4096 units as image representations.
    - 画像の表現として、4096 個のユニットをもつ fully connected layer の最後の出力を利用する。

<br>

- For the world representation we first gather a corpus of text from concatenation of user-tags, titles and descriptions from YFCC100M 2 dataset metadata.
    - 世界の表現のために、我々ははじめに、
    - ユーザータグ、タイトル、YFCC100M 2 dataset のメタデータからの記述の連結 [concatenation] からテキストのコーパスを集めた

- After pre-processing and cleaning of the text we trained a skip-gram model [14] with word vector size of 200.
    - 事前学習と、テキストのクリーニングの後で、サイズ 200 のワードベクトルで、skip-gram を学習した。

- And we omitted any word appearing less than 200 times from the vocabulary, thereby ending up with a dictionary of size 247465.
    - ボキャブラリーから 200回以下の回数で現れる用語を省略した [omitted ]。
    - それ故、247465 サイズの辞書になるという結果になった。

<br>

- We keep the convolutional model and the language model fixed during training of the adversarial net.
    - 敵対的ネットワークを学習している間、畳み込みモデルと言語モデルを固定し続けた。

- And leave the experiments when we even backpropagate through these models as future work.
    - そして、将来の研究として、実験を残す。
    - これらのモデルを通じたバックプロパゲーションするときに

<br>

- For our experiments we use MIR Flickr 25,000 dataset [10], and extract the image and tags features using the convolutional model and language model we described above.
    - 我々の実験のために、MIR Flickr 25,000 dataset [10] を使用する。
    - そして、上記で記述した畳み込みモデルと言語モデルを用いて、画像とタグ特徴量を抽出する。

- Images without any tag were omitted from our experiments and annotations were treated as extra tags. 
    - いくつかのタグ無しの画像は、我々の実験から除外されている。
    - そして、アノテーションは、追加のタグとして扱われる。

- The first 150,000 examples were used as training set.
    - 最初の 150,000 個のサンプルは、学習用データセットとして使用される。

- Images with multiple tags were repeated inside the training set once for each associated tag.
    - 複数のタグをもつ画像は、各関連タグに対して、学習用データセットの中で、一度だけ繰り返された。

<br>

- For evaluation, we generate 100 samples for each image and find top 20 closest words using cosine similarity of vector representation of the words in the vocabulary to each sample.
    - 評価のために、各画像の 100 個のサンプルを生成する。
    - そして、語彙ベクトルの cos 類似度を用いて、上位 20 個の近いワードを見つけ出す。

- Then we select the top 10 most common words among all 100 samples.
    - 次に、100 個全てのサンプルの間に、上位 10 個の最も共通したワードを選択する。

- Table 4.2 shows some samples of the user assigned tags and annotations along with the generated tags.
    - 表 4.2 は、ユーザーの関連タグと、生成されたタグの間のアノテーションのいくつかのサンプルを示している。

![image](https://user-images.githubusercontent.com/25688193/56711859-32052080-6767-11e9-851a-cf60be39caf7.png)<br>

- The best working model’s generator receives Gaussian noise of size 100 as noise prior and maps it to 500 dimension ReLu layer. 
    - 最もうまく動作している生成器は、100 個のガウス分布にもとづくノイズを、前のノイズとして受け取っている。
    - そして、それを（＝ノイズを）、500 次元の Relu層へ写像している

- And maps 4096 dimension image feature vector to 2000 dimension ReLu hidden layer.
    - そして、4096 次元の画像特徴量を、200 次元の Relu 隠れ層へ、写像している。

- Both of these layers are mapped to a joint representation of 200 dimension linear layer which would output the generated word vectors.
    - これらの層の両方は、生成された語彙ベクトルを出力するような、200 次元の線形層の結合された表現に、写像されている。

- The discriminator is consisted of 500 and 1200 dimension ReLu hidden layers for word vectors and image features respectively and maxout layer with 1000 units and 3 pieces as the join layer which is finally fed to the one single sigmoid unit.
    - 識別器は、各々の [respectively] 語彙ベクトルと画像特徴量のための、500,1200 次元の Relu 隠れ層で構成されている。
    - xxx


# ■ メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


