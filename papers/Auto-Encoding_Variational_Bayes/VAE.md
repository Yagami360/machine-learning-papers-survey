> 論文まとめ記事：https://github.com/Yagami360/My_NoteBook/blob/master/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6/%E6%83%85%E5%A0%B1%E5%B7%A5%E5%AD%A6_%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92_%E7%94%9F%E6%88%90%E3%83%A2%E3%83%87%E3%83%AB.md#VAE

# ■ 論文
- 論文タイトル："Auto-Encoding Variational Bayes"
- 論文リンク：https://arxiv.org/abs/1312.6114
- 論文投稿日付：2013/11/20
- 著者（組織）：Diederik P Kingma, Max Welling
- categories：

# ■ 概要（何をしたか？）

## Abstract

- How can we perform efficient inference and learning in directed probabilistic models, in the presence of continuous latent variables with intractable posterior distributions, and large datasets? We introduce a stochastic variational inference and learning algorithm that scales to large datasets and, under some mild differentiability conditions, even works in the intractable case. Our contributions is two-fold.
    - 難解な [intractable] 事後分布 [posterior distributions] をもつ連続的な潜在変数と大きなデータセットの存在 [presence] 下で、有向 [directed] 確率モデルでどのようにして効率的な推論と学習を行うことができるでしょうか。
    - 我々は、大規模なデータセットに対応し、いくつかの穏やかな微分可能性の条件の下では、難解な場合でもうまくいくようなな、確率的な変分推論 [variational inference] と学習のアルゴリズムを紹介します。
    - 私たちの貢献は2つあります。

- First, we show that a reparameterization of the variational lower bound yields a lower bound estimator that can be straightforwardly optimized using standard stochastic gradient methods.
    - まず、変分下限 [variational lower bound] の再パラメータ化は、標準確率勾配法を使用して素直に [straightforwardly] 最適化できる下限推定量を生み出すことを示します。

- Second, we show that for i.i.d. datasets with continuous latent variables per datapoint, posterior inference can be made especially efficient by fitting an approximate inference model (also called a recognition model) to the intractable posterior using the proposed lower bound estimator. Theoretical advantages are reflected in experimental results.
    - 次に、データポイントごとに連続的な潜在変数を持つi.i.dデータセットに対しては、提案された下限推定器を使用して、近似推論モデル [approximate inference model]（認識モデルとも呼ばれる）を、難解な [intractable] 事後（確率分布）にフィッティングすることによって、事後推論が特に効率的になります。
    - 理論的な利点は実験結果に反映されています。


# ■ イントロダクション（何をしたいか？）

## 1. Introduction

- How can we perform efficient approximate inference and learning with directed probabilistic models whose continuous latent variables and/or parameters have intractable posterior distributions?
    - 連続潜在変数やパラメータに難解な事後分布を持つような有向確率モデルを使用して、効率的な近似推論および学習をどのように実行できますか？

- The variational Bayesian (VB) approach involves the optimization of an approximation to the intractable posterior.
    - 変分ベイズ（ＶＢ）のアプローチは、難解な事後分布への近似の最適化を含む。

- Unfortunately, the common mean-field approach requires analytical solutions of expectations w.r.t the approximate posterior, which are also intractable in the general case.
    - 残念なことに、一般的な平均場 [mean-field] アプローチは、事後分布の近似に関して [w.r.t] 、期待値の解析解 [analytical solutions] を必要とし、これも一般的な場合には扱いにくい。

- We show how a reparameterization of the variational lower bound yields a simple differentiable unbiased estimator of the lower bound; this SGVB (Stochastic Gradient Variational Bayes) estimator can be used for efficient approximate posterior inference in almost any model with continuous latent variables and/or parameters, and is straightforward to optimize using standard stochastic gradient ascent techniques.
    - 我々は、変分下限の再パラメータ化が、どのようにして下限の単純な微分可能な不偏推定量 [unbiased estimator] を生み出すのかを示します。
    - 即ち、この SGVB（Stochastic Gradient Variational Bayes）推定量は、連続潜在変数および/またはパラメータを含むほぼすべてのモデルにおいて、効率的な近似事後推論に使用でき、標準の確率的勾配法を使用して最適化するのは簡単です。


---

- For the case of an i.i.d dataset and continuous latent variables per datapoint, we propose the Auto- Encoding VB (AEVB) algorithm.
    - データセットごとのi.i.dデータセットと連続潜在変数の場合、我々は Auto- Encoding VB（AEVB）アルゴリズムを提案する。

- In the AEVB algorithm we make inference and learning especially efficient by using the SGVB estimator to optimize a recognition model that allows us to perform very efficient approximate posterior inference using simple ancestral sampling, which in turn allows us to efficiently learn the model parameters, without the need of expensive iterative inference schemes (such as MCMC) per datapoint.
    - AEVBアルゴリズムでは、SGVB推定器を使用して、単純な伝承サンプリング [ancestral sampling] を使用して、非常に効率的な近似事後推論を許容するような、認識モデルを最適化することで推論と学習を特に効率的にします。
    - これにより、データポイントごとに、MCMCのように高価な反復推論スキームを必要とすることなく、モデルのパラメータを効率的に学習することを順番に？ [in turn] 可能にする。

> 伝承サンプリング [ancestral sampling : ベイジアンネットワークでは、サンプリングとして伝承サンプリング（Ancestral sampling）が用いられます。このサンプリング方法は、ベイジアンネットワークの一番起点となる変数から1つずつサンプリングしていき、全ての変数の値をサンプリングするという方法です。

- The learned approximate posterior inference model can also be used for a host of tasks such as recognition, denoising, representation and visualization purposes. When a neural network is used for the recognition model, we arrive at the variational auto-encoder.
    - 学習された近似事後推論モデルは、認識、ノイズ除去、表現および視覚化の目的などの多くのタスクにも使用できます。 ニューラルネットワークが認識モデルに使用されるとき、我々は変分自動符号器に到達する。


# ■ 結論

## x. Conclusion


# ■ 何をしたか？詳細

## 2 Method

- The strategy in this section can be used to derive a lower bound estimator (a stochastic objective function) for a variety of directed graphical models with continuous latent variables.
    - この節の戦略は、連続潜在変数を含むさまざまな有向グラフィカルモデルの下限推定量（確率目的関数）を導出する [derive] ために使用できます。

- We will restrict ourselves here to the common case where we have an i.i.d. dataset with latent variables per datapoint, and where we like to perform maximum likelihood (ML) or maximum a posteriori (MAP) inference on the (global) parameters, and variational inference on the latent variables.
    - ここでは、データポイントごとに潜在変数を持つiidデータセットがあり、（グローバル）パラメータに対して最尤度推定（ML）または最大事後（MAP）推論、そして潜在変数において変種推定を実行したいような、一般的な場合に限定します。 

- It is, for example, straightforward to extend this scenario to the case where we also perform variational inference on the global parameters; that algorithm is put in the appendix, but experiments with that case are left to future work.
    - たとえば、このシナリオをグローバルパラメータに対して変分推論も実行する場合に拡張することは簡単です。 このアルゴリズムは付録に記載されていますが、その場合の実験は今後の研究に任されています。

- Note that our method can be applied to online, non-stationary settings, e.g streaming data, but here we assume a fixed dataset for simplicity.
    - 本発明の方法は、ストリーミングデータなどのオンラインの非定常設定に適用することができることに留意されたいが、ここでは簡単のために固定データセットを仮定する。

### 2.1 Problem scenario

- Let us consider some dataset X = {x(i)}Ni=1 consisting of N i.i.d samples of some continuous or discrete variable x. We assume that the data are generated by some random process, involving an unobserved continuous random variable z.
    - いくつかの連続変数または離散 [discrete] 変数xのN i.i.d個のサンプルからなる、いくつかのデータセットX = {x（i）} Ni = 1を考えてみましょう。 データは、観測されていない連続確率変数zを含む、何らかのランダム過程によって生成されると仮定します。

- The process consists of two steps: (1) a value z(i) is generated from some prior distribution pθ∗ (z); (2) a value x(i) is generated from some conditional distribution pθ∗ (x|z). 
    - このプロセスは２つのステップからなる。
    -  (1) : 値 z(i) が何らかの事前分布 pθ*(ｚ) から生成される。
    -  (2) : ある条件付き分布 pθ*(x|ｚ) から値 x(i) が生成される。

- We assume that the prior pθ∗ (z) and likelihood pθ∗ (x|z) come from parametric families of distributions pθ (z) and pθ (x|z), and that their PDFs are differentiable almost everywhere w.r.t both θ and z.
    - 事前分布の pθ*(ｚ) と尤度 pθ*(x|ｚ) は、分布 pθ(z) と pθ（x|z）のパラメトリック族に由来し、それらのPDFはθとzの両方に対してほぼどこでも微分可能であると仮定します。 。

- Unfortunately, a lot of this process is hidden from our view: the true parameters θ∗ as well as the values of the latent variables z(i) are unknown to us.
    - 残念ながら、このプロセスの多くは私たちの見解からは隠されています。潜在変数z（i）の値だけでなく真のパラメータθ∗も私たちには未知です。

---

- Very importantly, we do not make the common simplifying assumptions about the marginal or posterior probabilities. Conversely, we are here interested in a general algorithm that even works efficiently in the case of:
    - 非常に重要なことに、周辺確率 [marginal probabilities] または事後確率について共通の単純化仮定をしていません。 逆に言えば [Conversely]、ここでは以下の場合でも効率的に動作する一般的なアルゴリズムに興味があります。

1. Intractability: the case where the integral of the marginal likelihood pθ(x) = ∫pθ(z)pθ(x|z)dz is intractable (so we cannot evaluate or differentiate the marginal like- lihood), where the true posterior density pθ(z|x) = pθ(x|z)pθ(z)/pθ(x) is intractable (so the EM algorithm cannot be used), and where the required integrals for any reasonable mean-field VB algorithm are also intractable. These intractabilities are quite common and appear in cases of moderately complicated likelihood functions pθ(x|z), e.g a neural network with a nonlinear hidden layer.
    - 扱いにくさ [Intractability]：
    - 周辺 [marginal] 尤度の積分 pθ（x）=∫pθ（z）pθ（x | z）dz が難解である（したがって、周辺類似性を評価または微分できない）場合、
    - 真の事後密度 pθ（z | x）=pθ（x | z）pθ（z）/pθ（x）が扱いにくい場合（したがってEMアルゴリズムは使用できません）、
    - また任意の合理的な平均場VBアルゴリズムに必要な積分も扱いにくい場合
    - これらの難治性は非常に一般的であり、適度に [moderately] 複雑な尤度関数ｐθ（ｘ ｜ ｚ）、例えば非線形隠れ層を有するニューラルネットワークの場合に現れる。

2. A large dataset: we have so much data that batch optimization is too costly; we would like to make parameter updates using small minibatches or even single datapoints. Sampling- based solutions, e.g Monte Carlo EM, would in general be too slow, since it involves a typically expensive sampling loop per datapoint.
    - 大規模なデータセット：大量のデータがあるため、バッチ最適化はコストがかかります。 小さなミニバッチや単一のデータポイントを使ってパラメータを更新したいと思います。 例えばモンテカルロＥＭのようなサンプリングベースの解決策は、データポイント毎に典型的に高価なサンプリングループを含むので、一般に遅すぎるであろう。

---

- We are interested in, and propose a solution to, three related problems in the above scenario:
    - 上記のシナリオにおける3つの関連する問題に関心があり、その解決策を提案します。

1. Efficient approximate ML or MAP estimation for the parameters θ. The parameters can be of interest themselves, e.g if we are analyzing some natural process. They also allow us to mimic the hidden random process and generate artificial data that resembles the real data.
    - パラメータθに対する効率的な近似ＭＬまたはＭＡＰ推定。
    - 例えば我々が何らかの自然な過程を分析しているならば、パラメータはそれ自体興味を引くことができる。
    - それらはまた、隠れたランダム過程を模倣し、実際のデータに似た人工データを生成することを可能にします。

2. Efficient approximate posterior inference of the latent variable z given an observed value x for a choice of parameters θ. This is useful for coding or data representation tasks.
    - パラメータθの選択に対して観測値xが与えられたときの、潜在変数zの効率的な近似事後推定。 これは符号化やデータ表現作業に役立ちます。

3. Efficient approximate marginal inference of the variable x. This allows us to perform all kinds of inference tasks where a prior over x is required. Common applications in computer vision include image denoising, inpainting and super-resolution.
    - 変数xの効率的な近似周辺推論 これにより、xに対する事前の処理が必要とされるあらゆる種類の推論タスクを実行することができます。 コンピュータビジョンにおける一般的な用途には、画像のノイズ除去、修復、および超解像が含まれる。

---

- For the purpose of solving the above problems, let us introduce a recognition model qφ(z|x): an approximation to the intractable true posterior pθ(z|x).
    - 上記の問題を解決するために、認識モデル qφ(z|x) を導入しましょう。これは、難解な真の事後分布 pθ(z|x) の近似です。

- Note that in contrast with the approximate posterior in mean-field variational inference, it is not necessarily factorial and its parameters φ are not computed from some closed-form expectation. Instead, we’ll introduce a method for learning the recognition model parameters φ jointly with the generative model parameters θ.
    - 平均場変分推論における近似事後法とは対照的に、それは必ずしも階乗 [factorial] ではなく、そのパラメータφは何らかの閉形式（＝解析解）の期待値から計算されないことに注意してください。 代わりに、生成モデルパラメータθと一緒に認識モデルパラメータφを学習する方法を紹介します。

---

- From a coding theory perspective, the unobserved variables z have an interpretation as a latent representation or code. In this paper we will therefore also refer to the recognition model qφ(z|x) as a probabilistic encoder, since given a datapoint x it produces a distribution (e.g a Gaussian) over the possible values of the code z from which the datapoint x could have been generated.
    - 符号化理論の観点からは、観測されていない変数zは潜在的な表現または符号化値として解釈されます。 したがって、この論文では、認識モデルqφ（z | x）を確率的符号化器とも呼ぶ。なぜなら、データ点xが与えられると、生成された可能性があるデータ点xから符号化値zの可能性のある値について分布（例えばガウス分布）を生成するからである。 

- In a similar vein we will refer to pθ(x|z) as a probabilistic decoder, since given a code z it produces a distribution over the possible corresponding values of x.
    - 同様に [In a similar vein]、符号化値 ｚが与えられると、それは可能な対応するｘの値にわたる分布を生成するので、ｐθ（ｘ ｜ ｚ）を確率的復号器と呼ぶことにする。


### 2.2 The variational bound

- xxx

### 2.3 The SGVB estimator and AEVB algorithm

- In this section we introduce a practical estimator of the lower bound and its derivatives w.r.t the parameters. We assume an approximate posterior in the form qφ(z|x), but please note that the technique can be applied to the case qφ(z), i.e where we do not condition on x, as well. The fully variational Bayesian method for inferring a posterior over the parameters is given in the appendix.
    - この節では、下限とその導関数をパラメータとする実用的な推定量を紹介します。 我々はｑφ（ｚ ｜ ｘ）の形の近似事後分布を仮定するが、この技法はｑφ（ｚ）の場合、すなわちｘに関して条件付けしない場合にも適用できることに注意してください。 パラメータに対して事後分布を推定するための完全変分ベイズ法を付録に示します。

- xxx

### 2.4 The reparameterization trick

- xxx

## 3 Example: Variational Auto-Encoder

- In this section we’ll give an example where we use a neural network for the probabilistic encoder qφ (z|x) (the approximation to the posterior of the generative model pθ (x, z)) and where the parameters φ and θ are optimized jointly with the AEVB algorithm.
    - この節では、確率的符号化器qφ（z | x）（生成モデルpθ（x、z）の事後への近似）にニューラルネットワークを使用し、パラメータφとθが AEVBアルゴリズムと組み合わせて最適化されています。

---

- Let the prior over the latent variables be the centered isotropic multivariate Gaussian pθ(z) = N(z;0,I).
    - 潜在変数に対する事前分布を、中心等方性多変量ガウス分布 pθ(z) = N(z;0,I)）とする。

- Note that in this case, the prior lacks parameters. We let pθ(x|z) be a multivariate Gaussian (in case of real-valued data) or Bernoulli (in case of binary data) whose distribution parameters are computed from z with a MLP (a fully-connected neural network with a single hidden layer, see appendix C).
    - この場合、前者にはパラメータがありません。 pθ（x | z）を多変量ガウス分布（実数値データの場合）またはベルヌーイ分布（バイナリデータの場合）とし、その分布パラメータはMLP（単一の完全結合ニューラルネットワーク）を使用してzから計算します。 隠れ層、付録Cを参照）。

- Note the true posterior pθ(z|x) is in this case intractable.
    - この場合、真の事後pθ（z | x）は扱いにくいことに注意してください。

- While there is much freedom in the form qφ(z|x), we’ll assume the true (but intractable) posterior takes on a approximate Gaussian form with an approximately diagonal covariance. In this case, we can let the variational approximate posterior be a multivariate Gaussian with a diagonal covariance structure2:
    - qφ（z | x）の形式には大きな自由度がありますが、真の（しかし難解な）事後式は近似対角共分散の近似ガウス型をとると仮定します。 この場合、変分近似事後行列を対角共分散構造2をもつ多変量ガウス分布にすることができます。

![image](https://user-images.githubusercontent.com/25688193/62004364-f19ab180-b15e-11e9-97e7-23cf038a61d1.png)

- where the mean and s.d of the approximate posterior, μ(i) and σ(i), are outputs of the encoding MLP, i.e nonlinear functions of datapoint x(i) and the variational parameters φ (see appendix C).
    - ここで、近似事後μ（ｉ）およびσ（ｉ）の平均および分散値は、符号化ＭＬＰの出力、すなわちデータポイントｘ（ｉ）および変分パラメータφの非線形関数である（付録Ｃ参照）。


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. Related Work


