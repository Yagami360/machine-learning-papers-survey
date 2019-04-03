## 0. 論文
- 論文リンク：[[1406.2661] Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
- 論文投稿日付：2014/06/10
- 著者：Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio

## 概要 [Abstract]
- We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. 
    - 生成モデル [generative models] を推定する新しい枠組みとして、敵対的なプロセスを経由した手法を提案する。
    - これは、同時に G[generator],D[discriminator] の２つのモデルを学習する。
    - G（生成器）のモデルは、データ分布を獲得する。
    - D（識別器）のモデルは、学習用データからのサンプリングしたものと、Gが生成したものを比較し、確率を推定する。

- The training procedure for G is to maximize the probability of D making a mistake. This framework corresponds to a minimax two-player game. 
    - Gの学習過程は、Dが作り出した誤り確率を最大化することである。
    - この枠組みは、（ゲーム理論における）２人プレイヤーのミニマックスゲームに一致する。

- In the space of arbitrary functions G and D, a unique solution exists, with G recovering the training data distribution and D equal to 1/2 everywhere.
    - 任意 [arbitrary] の関数 G,D の空間において、<font color="Pink">G が学習用データの分布を回復？回収？[recovering]し</font>、D（による判定結果の確率値が）はいたるところで 1/2 の値になるといった、唯一 [unique] の解が存在する。

    > ⇒ 要は、完全な学習後では、DはGが生成した画像が本物なのか偽物なのかを識別出来なくなるので、その識別確率はランダム値の1/2 になるということ。

- In the case where G and D are defined by multilayer perceptrons, the entire system can be trained with backpropagation.
    - GとDが、多層パーセプトロンで定義される場合、全体のシステムは、誤差逆伝播法で学習させることが可能である。

- There is no need for any Markov chains or unrolled approximate inference networks during either training or generation of samples. 
    - サンプルデータの学習や生成過程において、マルコフ連鎖、或いは、<font color="Pink">展開された[unrolled]？近似推論ネットワーク [approximate inference networks] </font>は必要ない。

- Experiments demonstrate the potential of the framework through qualitative and quantitative evaluation of the generated samples.
    - 実験は、生成サンプルの定性的 [qualitative]、定量的[quantitative] な評価を通じて [through]、この枠組のポテンシャルを示して [demonstrate] いる 。

    > 論文での示す。示す程度のとしては、<br>
    > demonstrate = prove > show = exhibit = display > indicate > imply = suggest

## 1. Introduction

- The promise of deep learning is to discover rich, hierarchical models [2] that represent probability distributions over the kinds of data encountered in artificial intelligence applications, such as natural images, audio waveforms containing speech, and symbols in natural language corpora.
    - ディープラーニングの<font color="Pink">約束？[promise]</font>は、人工知能アプリケーション（例えば、画像、音声波形、自然言語コーパスなど）において遭遇する [encountered] <font color="Pink">データの種類に渡る [over] ？関する？</font>確率分布で表現される高級で階層的なモデルを発見することである。

- So far, the most striking successes in deep learning have involved discriminative models, usually those that map a high-dimensional, rich sensory input to a class label [14, 22]. 
    - これまでのところ [So far]、ディープラーニングにおいて最も著しい [striking] 成功は、識別モデル [discriminative models] を関与した [involved] ものだった。
    - （この識別モデルというのは、）通常 [usually]、<font color="Pink">高次元の豊かな感覚 [sensory] の入力</font>をクラスラベルに写像する。
    
    > 識別モデル：識別モデルという名の通り、入力データを単純に識別します。モデル化には条件付き確率が使われます。同時分布が必要ない時に活躍します。例えばSVMがこのモデルに属します。

- These striking successes have primarily been based on the backpropagation and dropout algorithms, using piecewise linear units [19, 9, 10] which have a particularly well-behaved gradient.
    - これらの著しい成功は、主に [primarily]、誤差逆伝播法とドロップアウトのアルゴリズムに基づくものであった。
    - （このアルゴリズムというのは、）特に正常に動作する [well-behaved] 勾配をもつ線形区分ユニット [piecewise linear units] を使っている。

    > 線形区分ユニット → 要は、Relu などの線形な活性化関数のこと。

- Deep generative models have had less of an impact, due to the difficulty of approximating many intractable probabilistic computations that arise in maximum likelihood estimation and related strategies, and due to difficulty of leveraging the benefits of piecewise linear units in the generative context. 
    - 最尤推定 [maximum likelihood estimation] や関連する手法を引き起こすような、たくさんの手に負えない [intractable] 確率的な [probabilistic] 計算 [computations] での近似 [approximating] の困難さに起因して [due to]、
    - また、生成コンテンツでの線形区分ユニット [piecewise linear units] の利点を活用すること [leveraging] の困難さに起因して、
    - 深層生成モデルは、それほど影響を与えなものであった。[had less of]。

- We propose a new generative model estimation procedure that sidesteps these difficulties. 
    - 我々は、それらの困難さを回避する [sidesteps] 新しい生成モデルの推定手法を提案する。

- In the proposed adversarial nets framework, the generative model is pitted against an adversary: a discriminative model that learns to determine whether a sample is from the model distribution or the data distribution.
    - 提案された敵対的ネットワークの枠組みでは、生成モデルは、敵対者 [adversary]（即ち、サンプルデータがモデルの分布から来たものなのか？データの分布から来たものなのか？の判断を学習するような識別モデル）に競わせる [pitted against]。

- The generative model can be thought of as analogous to a team of counterfeiters, trying to produce fake currency and use it without detection, while the discriminative model is analogous to the police, trying to detect the counterfeit currency.

- Competition in this game drives both teams to improve their methods until the counterfeits are indistiguishable from the genuine articles.

- This framework can yield specific training algorithms for many kinds of model and optimization algorithm.

- In this article, we explore the special case when the generative model generates samples by passing random noise through a multilayer perceptron, and the discriminative model is also a multilayer perceptron. 

- We refer to this special case as adversarial nets. In this case, we can train both models using only the highly successful backpropagation and dropout algorithms [17] and sample from the generative model using only forward propagation. No approximate inference or Markov chains are necessary.


## 2. Related work

## 3. Adversarial nets

- The adversarial modeling framework is most straightforward to apply when the models are both multilayer perceptrons.
    - モデルが両方とも多層パーセプトロンであるときに、敵対的モデリングフレームワークは最も簡単に適用できます。

- To learn the generator’s distribution $p_g$ over data $x$, we define a prior on input noise variables $p_z(z)$, then represent a mapping to data space as $G(z; \theta_g)$, where G is a differentiable function represented by a multilayer perceptron with parameters $\theta_g$.
    - データ $x$ に対してのジェネレーター G の確率分布 $p_g$ を学習するために、前もって、入力ノイズ変数 $p_z(z)$ を定義し、パラメータ $\theta_g$ を持つ多層パーセプトロンによって表現された微分可能な関数である $G(z; \theta_g)$ としてデータ空間への写像で表現する。

- 