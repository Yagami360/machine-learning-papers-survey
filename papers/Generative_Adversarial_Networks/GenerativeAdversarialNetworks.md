> 論文まとめノート：https://github.com/Yagami360/My_NoteBook/blob/master/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6_%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_%E7%94%9F%E6%88%90%E3%83%A2%E3%83%87%E3%83%AB.md#GAN

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
    - <font color="Pink">NCE では、その動作の中にあるように、生成モデルに適合するために、特徴的な [discriminative] 損失関数 [training criterion] が採用されている。</font>

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
    - この領域の主要な [Prominent] 最近の手法 [work] は、generative stochastic network (GSN) の枠組みを含んでいる。
    - （この枠組は、）generalized denoising auto-encoders を拡張する。
    - つまり、xxx

- Compared to GSNs, the adversarial nets framework does not require a Markov chain for sampling.
    - GSNs と比較すると、敵対的ネットワークは、サンプリングのために、マルコフ連鎖を必要としない。

- Because adversarial nets do not require feedback loops during generation, they are better able to leverage piecewise linear units [19, 9, 10], which improve the performance of backpropagation but have problems with unbounded activation when used ina feedback loop.
    - 敵対的ネットワークは、生成過程でフィードバックループを必要としないため、誤差逆伝播法のパフォーマンスを改善するような区分線形ユニットを利用することをより可能にする。
    - <font color="Pink">しかし、内部のフィードバックループを使う際に、unbounded activation? に問題をもつ。</font>

- More recent examples of training a generative machine by back-propagating into it include recent work on auto-encoding variational Bayes [20] and stochastic backpropagation [24].
    - より最近の誤差逆伝播法を持つ生成器の学習例は、auto-encoding variational Bayes や確率的誤差逆伝播法といった最近の手法を含む。

## 3. Adversarial nets

- The adversarial modeling framework is most straightforward to apply when the models are both multilayer perceptrons.
    - モデルが両方とも多層パーセプトロンであるときに、敵対的モデリングフレームワークは最も簡単に適用できます。

- To learn the generator’s distribution $p_g$ over data $x$, we define a prior on input noise variables $p_z(\vec{z})$, then represent a mapping to data space as $G(\vec{z}; \theta_g)$, where G is a differentiable function represented by a multilayer perceptron with parameters $\theta_g$.
    - データ $x$ に対してのジェネレーター G の確率分布 $p_g$ を学習するために、前もって、入力ノイズ変数 $p_z(\vec{z})$ を定義し、パラメータ $\theta_g$ を持つ多層パーセプトロンによって表現された微分可能な関数である $G(z; \theta_g)$ としてデータ空間への写像で表現する。

- We also define a second multilayer perceptron $D(\vec{x};\theta_d)$ that outputs a single scalar. 
    - 単一のスカラー値を出力する２つ目の多層パーセプトロン $D(\vec{x};\theta_d)$ を定義する。

- $D(\vec{x})$ represents the probability that $x$ came from the data rather than $p_g$.
    - $D(\vec{x})$ は、（生成器の確率分布）$p_g(z)$ ではなく [rather than]、データ $x$ が来た（入力された）ときの、（識別）確率を表わしている。

- We train D to maximize the probability of assigning the correct label to both training examples and samples from G. 
    - 識別器 D が、正しいラベルを割り当てる [assigning] 確率を最大化するように学習する。

- We simultaneously train G to minimize $\log{ (1-D(G(z)) ) }$ :
    - 同時に、生成器 G を、$\log{ (1-D(G(z)) ) }$ が最小になるように学習する。

- In other words, D and G play the following two-player minimax game with value function $V(G;D)$:
    - 言い換えれば、識別器 D と生成器 G は、価値関数（＝損失関数） $V(G;D)$ に対しての２人プレイヤーのミニマックスゲームにそってゲームプレイする。

$$
min_{G} max_D{V(D,G)} = \mathbb{E}_{\vec{x} \sim p_{data}(x) } [\log{D(\vec{x})}] + \mathbb{E}_{\vec{z} \sim p_{z}(z) } [1-\log{D(G(\vec{z})})]
$$

> $\vec{x} \sim p_{data}(\vec{x})$ ： $\vec{x}$ は、確率分布 $p_{data}(\vec{x})$ に従う。（＝確率分布 $p_{data}(\vec{x})$ からサンプリングされたデータ $\vec{x}$ である。）

<br>

- In the next section, we present a theoretical analysis of adversarial nets, essentially showing that the training criterion allows one to recover the data generating distribution as G and D are given enough capacity, i.e., in the non-parametric limit.
    - 次のセクションでは、我々は、敵対的ネットワークの理論的な分析を提示する。
    - <font color="Pink">本質的に [essentially]、生成器 G と識別器 D に十分なキャパシティが与えれているときに、損失関数 [training criterion] はデータを生成する分布をリカバーすることを許容することを示している。</font>


- See Figure 1 for a less formal, more pedagogical explanation of the approach. 
    - 形式的でない [less formal]、より教育的な [pedagogical] アプローチでの説明については、図１を見てください。

- In practice, we must implement the game using an iterative, numerical approach.
    - 実際には [In practice]、我々は、反復的な数値的手法を使って実装しなくてはならない。

- Optimizing $D$ to completion in the inner loop of training is computationally prohibitive, and on finite datasets would result in overfitting.
    - <font color="Pink">内部の学習ループでの完了 [completion] で識別器 D を最適化することは、計算的に[computationally] 禁止であり、</font>有限のデータセットでは、過学習の結果を招く。

- Instead, we alternate between k steps of optimizing D and one step of optimizing G.
    - 代わりに、kステップでの識別器 D の最適化と、1ステップでの生成器 G の最適化の間を、交互に行う。[alternate]

- This results in D being maintained near its optimal solution, so long as G changes slowly enough. 
    - 生成器 G が十分緩やかに変化する限り [so long as]、識別器 D は最適解付近に維持される
    - この xxx での結果、

- This strategy is analogous to the way that SML/PCD [31, 29] training maintains samples from a Markov chain from one learning step to the next in order to avoid burning in a Markov chain as part of the inner loop of learning.
    - この戦略は、SML/PCD といった手法に類似 [analogous] している。
    - xxx

- The procedure is formally presented in Algorithm 1.
    - その処理は、アルゴリズム１に形式的に表している。

<br>

- In practice, equation 1 may not provide sufficient gradient for G to learn well. Early in learning, when G is poor, D can reject samples with high confidence because they are clearly different from the training data. 
    - 実際には、式１は、生成器 G がうまく学習するための十分な勾配を与えないかもしれない。
    - 学習の初期段階において、生成器 G が弱いとき、識別器 D は高い信用度でサンプルを拒絶することが出来る。
    - なぜならば、それらは、明らかに学習用データとはことなるためである。

- In this case, $\log{ (1 - D(G(z))) }$ saturates. Rather than training G to minimize $\log(1 - D(G(z)))$ we can train G to maximize $\log{D(G(z))}$.
    - このようなケースにおいて、$\log{ (1 - D(G(z))) }$ を満たす。[saturates]
    - $\log{ (1 - D(G(z))) }$ を最小化するために、生成器 G を学習するよりも、$\log{ D(G(z)) }$ を最大化するために、生成器 G を学習する。

- This objective function results in the same fixed point of the dynamics of G and D but provides much stronger gradients early in learning.
    - この目的関数 [objective function] は、結果的に、G と D の動的な同じ固定点をもたらす。[result in]
    - しかし、学習の初期段階において、より強い勾配を提供する。

![image](https://user-images.githubusercontent.com/25688193/55600485-1ecde900-5797-11e9-9476-9f904a4d41ac.png)<br>

- > Figure 1: Generative adversarial nets are trained by simultaneously updating the discriminative distribution ($D$, blue, dashed line) so that it discriminates between samples from the data generating distribution (black,dotted line) $p_x$ from those of the generative distribution $p_g(G)$ (green, solid line).
    - > 図１：敵対的生成ネットワークは、データが生成する分布 $p_x$（黒点先）サンプルと、（生成器が生成する確率分布）<font color="Green">$p_g(G)$（緑線）</font>の間を識別するように [so that]、識別器の分布を更新しながら、同時に学習される。（<font color="Blue">識別器 D は青ダッシュ線</font>）

- > The lower horizontal line is the domain from which z is sampled, in this case uniformly. The horizontal line above is part of the domain of x. The upward arrows show how the mapping $x = G(z)$ imposes the non-uniform distribution $p_g$ on transformed samples. 
    - > 下段の水平線は、一様に？ z からサンプルされた z からの領域 [domein] である。
    - > この水平線の上側は、x の領域の一部である。
    - > 上向き矢印は、どうのようして、写像 $x=G(z)$ が、変換されたサンプルに、一様分布でない $p_g$ を強制する [impose on] のかを示している。

- > G contracts in regions of high density and expands in regions of low density of $p_g$. 
    - 生成器 G は、高密度の領域を縮約し [contracts]、$p_g$ の低密度の領域を拡張する。

- > (a) Consider an adversarial pair near convergence: $p_g$ is similar to $p_{data}$ and D is a partially accurate classifier.
    - > (a) 収束 [convergence]　付近の敵対ネットワークのペアを考慮した図。
    - > 即ち、$p_g$ （の形状）は、$p_{data}$ に似ており、識別器 D は、部分的に正確な分類器となっている。

- > (b) In the inner loop of the algorithm D is trained to discriminate samples from data, converging to $D^*(x) = {p_{data}(x)}/{(p_{data}(x)+p_g(x)})$. 
    - > (b) 識別器 D のアルゴリズムの内部ループでは、$D^*(x) = {p_{data}(x)}/{(p_{data}(x)+p_g(x)})$ に収束するような、データからのサンプルを識別することを学習されている。

- > (c) After an update to G, gradient of D has guided $G(z)$ to flow to regions that are more likely to be classified as data.
    > - (c)１回の生成器の更新の後、識別器 D の勾配は、<font color="Pink">$G(z)$ に、データとして分類される可能性の高い [be likely to] 領域に流れるようにガイドする。</font>

- > (d) After several steps of training, if G and D have enough capacity, they will reach a point at which both cannot improve because $p_g = p_{data}$. The discriminator is unable to differentiate between the two distributions, i.e. $D(x) = 1/2$.
    - > (d) 何回かの学習ステップの後、生成器 G と識別器 D が十分なキャパシティを持っていれば、それらは、$p_g = p_{data}$ となるために、両方とも（これ以上は）改善しないという（最適）点に到達するだろう。
    - > 識別器は、例えば、1/2 の確率といったように、２つの分布の違いを識別することができなくなる。


## 4. Theoretical Results

- The generator G implicitly defines a probability distribution $p_g$ as the distribution of the samples $G(z)$ obtained when $z \sim p_z$.
    - 生成器 G は、$z \sim p_z$ のとき、$G(z)$ が手に入れるサンプルの分布として確率分布 $p_g$ を暗に [implicitly] 定義する。

- Therefore, we would like Algorithm 1 to converge to a good estimator of $p_{data}$, if given enough capacity and training time.
    - それ故、もし、十分な容量と学習時間が与えられていれば、アルゴリズム１は、$p_{data}$ のよい推定器に収束したい [converge]。

- The results of this section are done in a nonparametric setting, e.g. we represent a model with infinite capacity by studying convergence in the space of probability density functions.
    - このセクションでの結果は、ノンパラメトリック設定で行われている。
    - 例えば、確率分布関数の空間への収束を学習することにより、無限の容量でモデルを表現する。

- We will show in section 4.1 that this minimax game has a global optimum for $p_g = p_{data}$.
    - セクション4.1 に、このミニマックスゲームが、$p_g = p_{data}$ に対しての、大域的最適を持つことを示す。

- We will then show in section 4.2 that Algorithm 1 optimizes Eq 1, thus obtaining the desired result.
    - セクション 4.2 では、アルゴリズム１が式１を最適化することを示す。
    - 結果として、望ましい結果が得られる。

![image](https://user-images.githubusercontent.com/25688193/55601015-9735a980-5799-11e9-9d6f-0d78a648a4f8.png)<br>

### 4.1 Global Optimality of $p_g = p_{data}$
- We first consider the optimal discriminator D for any given generator G.
    - 最初に、いくつかの生成器 G によって与えられる、識別器 D の最適値について考える。

- Proposition 1. For G fixed, the optimal discriminator D is
    - 命題１：生成器 G を固定のともで、識別器 D の最適値は、

![image](https://user-images.githubusercontent.com/25688193/55698389-820e8400-5a00-11e9-8780-23d260b98b17.png)<br>

- Proof. The training criterion for the discriminator D, given any generator G, is to maximize the quantity V (G;D)
    - 証明：識別器の損失関数 [training criterion] は、いくつかの生成器 G に対して、価値関数を最大化することである。

![image](https://user-images.githubusercontent.com/25688193/55699247-ed5a5500-5a04-11e9-82e1-8941265b149c.png)<br>

- For any $(a; b) \in R^2 \ {(0; 0)}$, the function $y \rightarrow a \log{y} + b \log{(1-y)}$ achieves its maximum in [0; 1] at $a/a+b$ . The discriminator does not need to be defined outside of $Supp{(p_{data})} \cup Supp(p_g)$, concluding the proof.

- Note that the training objective for D can be interpreted as maximizing the log-likelihood for estimating the conditional probability $P(Y = y|x)$, where $Y$ indicates whether $x$ comes from $p_{data}$ (with $y = 1$) or from $p_g$ (with $y = 0$). 
    -  ここで留意すべきは [note that]、識別器 D の学習目的が、条件確率 $P(Y = y|x)$ （$Y$ は $x$ が $p_{data}$ から来るか、$p_g$ から来るかを示している。）を推定するための対数尤度の最大化として、解釈する [interpreted as] ことができる。

- The minimax game in Eq. 1 can now be reformulated as:

![image](https://user-images.githubusercontent.com/25688193/55699707-082dc900-5a07-11e9-96bb-f18ee084eecc.png)<br>

- Theorem 1. The global minimum of the virtual training criterion $C(G)$ is achieved if and only if $p_g = p_{data}$. At that point, $C(G)$ achieves the value $-\log{(4)}$.
    - 定理１：仮想的な損失関数 $C(G)$ の大域的最適値では、$p_g = p_{data}$ が成り立つ。
    - この点では、$C(G)$ は、$-\log{(4)}$ の値になる。

- Proof. For pg = pdata, DG(x) = 12 , (consider Eq. 2). Hence, by inspecting Eq. 4 at DG(x) = 12 , we find C(G) = log 12 + log 12 = 􀀀log 4.

- To see that this is the best possible value of C(G), reached only for pg = pdata, observe that

![image](https://user-images.githubusercontent.com/25688193/55700222-6491e800-5a09-11e9-83a0-cce4e1ca4b9d.png)<br>

- and that by subtracting this expression from C(G) = V (DG;G), we obtain :

![image](https://user-images.githubusercontent.com/25688193/55700261-8be8b500-5a09-11e9-8d2c-5c9a15a7e00c.png)<br>

- where KL is the Kullback–Leibler divergence. We recognize in the previous expression the Jensen–Shannon divergence between the model’s distribution and the data generating process:

![image](https://user-images.githubusercontent.com/25688193/55700282-9dca5800-5a09-11e9-8cb2-df5d18ca3b63.png)<br>

- Since the Jensen–Shannon divergence between two distributions is always non-negative and zero only when they are equal, we have shown that C = 􀀀log(4) is the global minimum of C(G) and that the only solution is pg = pdata, i.e., the generative model perfectly replicating the data generating process.



## 5. Experiments

- We trained adversarial nets an a range of datasets including MNIST[23], the Toronto Face Database (TFD) [28], and CIFAR-10 [21].
    - 我々は、敵対ネットワークを、MNIST, TFD,CIFAR-10 を含むデータセットで学習した。

- The generator nets used a mixture of rectifier linear activations [19,9] and sigmoid activations, while the discriminator net used maxout [10] activations.
    - 生成器は、ReLu [rectifier linear activations] やシグモイド関数使用した。
    - 一方で、識別器は、maxout 関数を使用した。

- Dropout [17] was applied in training the discriminator net.
    - ドロップアウトは、識別器の学習に適用した。

- While our theoretical framework permits the use of dropout and other noise at intermediate layers of the generator, we used noise as the input to only the bottommost layer of the generator network.
    - <font color="Pink">我々の理論的な枠組みが、ドロップアウトや生成器の仲介 [intermediate] 層での他のノイズの使用を許可までの間、生成器のネットワークの一番下の層だけにノイズを入力として使用した。</font>

<br>

- We estimate probability of the test set data under $p_g$ by fitting a Gaussian Parzen window to the samples generated with G and reporting the log-likelihood under this distribution.
    - 我々は、$p_g$ の元で、テストデータセットの確率を推定する。
    - （これは、）<font color="Pink">カーネル密度推定 [Gaussian Parzen window] を、生成器 G とこの分布のもとで対数尤度を報告することによって生成されたサンプルにフィッティングすることにより、</font>（行われる。）

- The $\sigma$ parameter of the Gaussians was obtained by cross validation on the validation set.
    - ガウス分布 [Gaussians] の $\sigma$ パラメーターは、バリデーションデータセットのクロス・バリデーション（CV）によって、手に入れることができる。

- This procedure was introduced in Breuleux et al. [8] and used for various generative models for which the exact likelihood is not tractable [25, 3, 5].
    - この処理は、Breuleux の論文中で紹介されている。
    - そして、正確な尤度は扱いやすくないため [for whith]、様々な種類の生成モデルが使われている。

- Results are reported in Table 1. 
    - 結果は、表１に報告されている。

- This method of estimating the likelihood has somewhat high variance and does not perform well in high dimensional spaces but it is the best method available to our knowledge. 
    - この尤度推定の方法は、いくらか [somewhat] ハイバリアンスとなっており、高次元空間でうまくいかない。しかし、これは、知る限りベストな方法である。

- Advances in generative models that can sample but not estimate likelihood directly motivate further research into how to evaluate such models.
    - <font color="Pink">生成モデルの進歩は、
    - 尤度推定ではなく [but not]、サンプル出来る、
    - そのようなモデルの評価の仕方を、直接的な動機でさらに調査する。</font>

![image](https://user-images.githubusercontent.com/25688193/55677324-6e8bec00-5920-11e9-89d2-f84c98174c96.png)<br>

> - Table 1: Parzen window-based log-likelihood estimates.
    - > 表１：カーネル密度推定ベースの対数尤度推定。

> - The reported numbers on MNIST are the mean loglikelihood of samples on test set, with the standard error of the mean computed across examples.
    > - 報告された MNIST での数は、標準誤差 [standard error of the mean] と共に、テストデータセットのサンプルの対数尤度の平均値である。

> - On TFD, we computed the standard error across folds of the dataset, with a different $\sigma$ chosen using the validation set of each fold. 
    > - TFD では、データセットの標準誤差を計算した。
    > - 各集団 [fold] のバリデーションデータセットの使用で選択された異なる $\sigma$ 値と共に、

- On TFD, $\sigma$ was cross validated on each fold and mean log-likelihood on each fold were computed.
    - TFD では、$\sigma$ 値は、各集団でクロスバリデーションされた値である。そして、各集団で計算された対数尤度の平均値である。

> - For MNIST we compare against other models of the real-valued (rather than binary) version of dataset.
    - MNIST のために、我々は、データセットの（バイナリデータでのはない） real-valued のバージョンの他のモデルと比較する。

<br>

- In Figures 2 and 3 we show samples drawn from the generator net after training. While we make no claim that these samples are better than samples generated by existing methods, we believe that these samples are at least competitive with the better generative models in the literature and highlight the potential of the adversarial framework.

![image](https://user-images.githubusercontent.com/25688193/55677650-8fefd680-5926-11e9-8ef4-fb30fa8aa921.png)<br>


> - Figure 2: Visualization of samples from the model.

> - Rightmost column shows the nearest training example of the neighboring sample, in order to demonstrate that the model has not memorized the training set.

> - Samples are fair random draws, not cherry-picked.

> - Unlike most other visualizations of deep generative models, these images show actual samples from the model distributions, not conditional means given samples of hidden units.

> - Moreover, these samples are uncorrelated because the sampling process does not depend on Markov chain mixing. 
> - a) MNIST 
> - b) TFD 
> - c) CIFAR-10 (fully connected model) 
> - d) CIFAR-10 (convolutional discriminator and “deconvolutional” generator)

