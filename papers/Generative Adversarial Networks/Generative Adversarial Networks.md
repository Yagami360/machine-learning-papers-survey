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

<br>

- In the proposed adversarial nets framework, the generative model is pitted against an adversary: a discriminative model that learns to determine whether a sample is from the model distribution or the data distribution.
    - 提案された敵対的ネットワークの枠組みでは、生成モデルは、敵対者 [adversary]（即ち、サンプルデータがモデルの分布から来たものなのか？データの分布から来たものなのか？の判断を学習するような識別モデル）に競わせる [pitted against]。

- The generative model can be thought of as analogous to a team of counterfeiters, trying to produce fake currency and use it without detection, while the discriminative model is analogous to the police, trying to detect the counterfeit currency.
    - 生成モデルは、偽通貨を生産し、検知されることなしに使用することを試みる、偽札 [counterfeiters] チームの類似 [analogous] として考えられる [be thought of]。
    - 一方で [while]、識別モデルは、偽札を検知しようと試みる警察に類似していると考えられる.

- Competition in this game drives both teams to improve their methods until the counterfeits are indistiguishable from the genuine articles.
    - このゲームの競争は、両方のチームが、本物の品物と識別できなくなるまで、彼らの手法を改善することを駆動する。

<br>

- This framework can yield specific training algorithms for many kinds of model and optimization algorithm.
    - この枠組みは、たくさんの種類のモデルや最適化アルゴリズムで、特別な学習アルゴリズムを生み出す [yield] ことが出来る。

- In this article, we explore the special case when the generative model generates samples by passing random noise through a multilayer perceptron, and the discriminative model is also a multilayer perceptron. 
    - この論文では、生成モデルがランダムノイズを多層パーセプトロンに通してサンプルを生成し、識別モデルが多層パーセプトロンであるような、特別な場合を扱う。

- We refer to this special case as adversarial nets. In this case, we can train both models using only the highly successful backpropagation and dropout algorithms [17] and sample from the generative model using only forward propagation. No approximate inference or Markov chains are necessary.
    - 我々は、この特別な場合を、敵対的ネットワークと呼ぶ [refer to]。
    - この場合、我々は、高い成功性を持つ誤差逆伝播法とドロップアウトのアルゴリズムを使うことでのみ、両方のモデルを学習させる事ができる。
    - 近似的推定法 [approximate inference] やマルコフ連鎖は必要ない。

## 2. Related work

- An alternative to directed graphical models with latent variables are undirected graphical models with latent variables, such as restricted Boltzmann machines (RBMs) [27, 16], deep Boltzmann machines (DBMs) [26] and their numerous variants.
    - 潜在変数 [latent variables] を持つ有向 [directed] グラフィカルなモデルの代わりとなるもの [alternative to] は、洗剤変数を持つ無向 [undirected] なモデルである。例えば、制限ボルツマンマシンや深層ボルツマンマシンやその変種などである。

- The interactions within such models are represented as the product of unnormalized potential functions, normalized by a global summation/integration over all states of the random variables.
    - そのようなモデルの相互作用は、正規化されていないポテンシャル関数の積や、ランダム変数の全ての状態に渡っての和や積分で正規化したものとして、表現される。

- This quantity (the partition function) and its gradient are intractable for all but the most trivial instances, although they can be estimated by Markov chain Monte Carlo (MCMC) methods.
    - 彼ら（＝モデルの相互採用）は、マルコフ連鎖や MCMC法で推定できるにもかかわらず、
    - この量（分配関数 [the partition function]）やその勾配は、最もありふれた [trivial] 例 [instances] を除いて [all but]、扱いにくい [intractable]。

- Mixing poses a significant problem for learning algorithms that rely on MCMC [3, 5].
    - <font color="Pink">混合？は、MCMC に頼る学習アルゴリズムにとって問題を持ち出す？ [pose]。</font>

<br>

- Deep belief networks (DBNs) [16] are hybrid models containing a single undirected layer and several directed layers. 
    - Deep Belief Network (DBNs) は、単体の非直接的な層といくつかの直接的な層を含んだ、ハイブリッドなモデルである。

- While a fast approximate layer-wise training criterion exists, DBNs incur the computational difficulties associated with both undirected and directed models.
    - 高速近似 [fast approximate] layer-wise 学習評価が存在するまでの間、DBNs は、無向モデル、有向モデルの両方に関連した [associated with]、計算上の困難さこうむる [incur]

<br>

- Alternative criteria that do not approximate or bound the log-likelihood have also been proposed, such as score matching [18] and noise-contrastive estimation (NCE) [13]. 
    - 例えば、スコアマッチや NCE のように、
    - 対数尤度を境界化？ [bound]、或いは、近似しないような、代わりとなる基準 [criteria] も提案されている。

- Both of these require the learned probability density to be analytically specified up to a normalization constant.
    - これらの手法は両方とも、<font color="Pink">学習された確率分布に、正規化定数で分析的に設定されること</font>を要求する。

- Note that in many interesting generative models with several layers of latent variables (such as DBNs and DBMs), it is not even possible to derive a tractable unnormalized probability density. 
    - <font color="Pink">留意すべきことは [Note that]、
    - いくつかの層や潜在変数を持つような、多くの興味深い生成モデルでは、
    - 扱いやすい [tractable] 正規化されていない確率分布を導き出す [derive] ことさえ可能ではない。</font>

- Some models such as denoising auto-encoders [30] and contractive autoencoders have learning rules very similar to score matching applied to RBMs.
    - ノイズ除去 [denoising] の auto-encoders や縮小 [contractive] の auto-encoders といったいくつかのモデルは、RBMs に適用されるスコアマッチ手法によく似た学習規則を持つ。

- In NCE, as in this work, a discriminative training criterion is employed to fit a generative model.
    - <font color="Pink">NCE では、その動作の中にあるように、生成モデルに適合するために、特徴的な [discriminative] 学習規則が採用されている。</font>

- However, rather than fitting a separate discriminative model, the generative model itself is used to discriminate generated data from samples a fixed noise distribution.
    - しかしながら、離散的な [separate] 識別モデルを適合させることよりも、生成モデルはそれ自身、固定されたノイズの分布のサンプルからデータを生成することに慣れている。

- Because NCE uses a fixed noise distribution, learning slows dramatically after the model has learned even an approximately correct distribution over a small subset of the observed variables.
    - NCE は固定されたノイズ分布を使うために、
    - 観測された変数の小さなサブセットに渡っての、正しい分布の近似でさえ、
    - 学習済みモデルのあとは、
    - 学習は、劇的に遅くなる。

- Finally, some techniques do not involve defining a probability distribution explicitly, but rather train a generative machine to draw samples from the desired distribution.
    - 最終的に、いくつかのテクニックは、確率分布を明示的に [explicitly] 定義することを巻き込まない。
    - むしろ [but rather]、生成器に、所望の [desired] 分布からサンプルすることを学習する。

- This approach has the advantage that such machines can be designed to be trained by back-propagation.
    - このアプローチは、そのような生成器が、誤差逆伝播法によって学習されることを設計出来るというメリットが存在する。

- Prominent recent work in this area includes the generative stochastic network (GSN) framework [5], which extends generalized denoising auto-encoders [4]: both can be seen as defining a parameterized Markov chain, i.e., one learns the parameters of a machine that performs one step of a generative Markov chain. 
    - xxx

- Compared to GSNs, the adversarial nets framework does not require a Markov chain for sampling.

- Because adversarial nets do not require feedback loops during generation, they are better able to leverage piecewise linear units [19, 9, 10], which improve the performance of backpropagation but have problems with unbounded activation when used ina feedback loop.

- More recent examples of training a generative machine by back-propagating into it include recent work on auto-encoding variational Bayes [20] and stochastic backpropagation [24].


## 3. Adversarial nets

- The adversarial modeling framework is most straightforward to apply when the models are both multilayer perceptrons.
    - モデルが両方とも多層パーセプトロンであるときに、敵対的モデリングフレームワークは最も簡単に適用できます。

- To learn the generator’s distribution $p_g$ over data $x$, we define a prior on input noise variables $p_z(z)$, then represent a mapping to data space as $G(z; \theta_g)$, where G is a differentiable function represented by a multilayer perceptron with parameters $\theta_g$.
    - データ $x$ に対してのジェネレーター G の確率分布 $p_g$ を学習するために、前もって、入力ノイズ変数 $p_z(z)$ を定義し、パラメータ $\theta_g$ を持つ多層パーセプトロンによって表現された微分可能な関数である $G(z; \theta_g)$ としてデータ空間への写像で表現する。

- 