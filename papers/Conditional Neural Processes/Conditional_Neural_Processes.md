# ■ 論文
- 論文タイトル："Conditional Neural Processes"
- 論文リンク：https://arxiv.org/abs/1807.01613
- 論文投稿日付：
- 被引用数（記事作成時点）：xxx 件
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- Deep neural networks excel at function approximation, yet they are typically trained from scratch for each new function. On the other hand, Bayesian methods, such as Gaussian Processes (GPs), exploit prior knowledge to quickly infer the shape of a new function at test time. Yet GPs are computationally expensive, and it can be hard to design appropriate priors. In this paper we propose a family of neural models, Conditional Neural Processes (CNPs), that combine the benefits of both. CNPs are inspired by the flexibility of stochastic processes such as GPs, but are structured as neural networks and trained via gradient descent. CNPs make accurate predictions after observing only a handful of training data points, yet scale to complex functions and large datasets. We demonstrate the performance and versatility of the approach on a range of canonical machine learning tasks, including regression, classification and image completion.
    - ディープニューラルネットワークは関数近似に優れていますが、通常、新しい関数ごとにゼロからトレーニングされます。 一方、ガウス過程（GP）などのベイジアン手法は、事前知識を活用して、テスト時に新しい関数の形状をすばやく推測します。 しかし、GPは計算コストが高く、適切な優先順位を設計するのが難しい場合があります。 このペーパーでは、両方の利点を組み合わせたニューラルモデルのファミリー、条件付きニューラルプロセス（CNP）を提案します。 CNPは、GPなどの確率的プロセスの柔軟性に触発されていますが、ニューラルネットワークとして構造化され、勾配降下法によって訓練されています。 CNPは、少数のトレーニングデータポイントのみを観察した後、正確な予測を行いますが、複雑な機能と大規模なデータセットに拡張します。 回帰、分類、画像補完などの一連の標準的な機械学習タスクで、アプローチのパフォーマンスと汎用性を実証します。


# ■ イントロダクション（何をしたいか？）

## x. Introduction

- Deep neural networks have enjoyed remarkable success in recent years, but they require large datasets for effective training (Lake et al., 2017; Garnelo et al., 2016). One way to mitigate this data efficiency problem is to approach learning in two phases.
    - ディープニューラルネットワークは近年目覚しい成功を収めていますが、効果的なトレーニングには大きなデータセットが必要です（Lake et al。、2017; Garnelo et al。、2016）。 このデータ効率の問題を軽減する1つの方法は、2段階で学習に取り組むことです。

- The first phase learns the statistics of a generic domain, drawing on a large training set, but without committing to a specific learning task within that domain.
    - 最初のフェーズでは、一般的なドメインの統計を学習し、大規模なトレーニングセットを利用しますが、そのドメイン内の特定の学習タスクをコミットしません。

- The second phase learns a function for a specific task, but does so using only a small number of data points by exploiting the domain-wide statistics already learned. Meta-learning with neural networks is one example of this approach (Wang et al., 2016; Reed et al., 2017).
    - 2番目のフェーズでは、特定のタスクの機能を学習しますが、既に学習したドメイン全体の統計情報を活用して、少数のデータポイントのみを使用して学習します。 ニューラルネットワークを使用したメタ学習は、このアプローチの一例です（Wang et al。、2016; Reed et al。、2017）。

---

- For example, consider supervised learning problems. Many of these can be framed as function approximation given a finite set of observations. Consider a dataset {(x,y)}_{i=0}^{n−1} of n inputs xi ∈ X and outputs yi ∈ Y .
    - たとえば、教師あり学習の問題を考えます。 これらの多くは、観測値の有限セットが与えられると、関数近似として組み立てることができます。 n個の入力xi∈Xと出力yi∈Yのデータセット{（x、y）} _ {i = 0} ^ {n-1}を考えます。

- Assume that these represent evaluations yi = f(xi) of some unknown function f :X →Y which may be fixed or arealization of some random function.
    - これらは、修正される可能性のある未知の関数f：X→Yの評価yi = f（xi）またはランダム関数の面積化を表すと仮定します。

- A supervised learning algorithm returns an approximating function g : X → Y or a distribution over such functions. The aim is to minimize a loss between f and g on the entire space X, but in practice the routine is evaluated on a finite set of observations that are held-out (making them effectively unlabelled).
    - 教師あり学習アルゴリズムは、近似関数g：X→Yまたはそのような関数の分布を返します。 目的は、空間X全体でfとgの間の損失を最小限に抑えることですが、実際には、ルーチンは、保持されている有限の観測セットで評価されます（事実上ラベル付けされません）。

- We call these unlabelled data points targets (see figure 1). Classification, regression, dynamics modeling, and image generation can all be cast in this framework.
    - これらのラベルのないデータポイントをターゲットと呼びます（図1を参照）。 このフレームワークでは、分類、回帰、ダイナミクスモデリング、および画像生成をすべて実行できます。

---

- One approach to supervised problems is to randomly initialize a parametric function g anew for each new task and spend the bulk of computation on a costly fitting phase.
    - 教師あり問題への1つのアプローチは、新しいタスクごとに新しいパラメトリック関数をランダムに初期化し、コストのかかるフィッティングフェーズに大量の計算を費やすことです。

- Prior information that a practitioner may have about f is specified via the architecture of g, the loss function, or the training details. This approach encompasses most of deep supervised learning. Since the extent of prior knowledge that can be expressed in this way is relatively limited, and learning cannot be shared between different tasks, the amount of training required is large, and deep learning methods tend to fail when training data is not plentiful.
    - 施術者が真の分布 f について持っている事前情報は、,モデルの分布 g のアーキテクチャ、損失関数、またはトレーニングの詳細によって指定されます。 このアプローチは、ほとんどの深い教師あり学習を網羅しています。 この方法で表現できる事前知識の範囲は比較的限られており、学習を異なるタスク間で共有できないため、必要なトレーニングの量が多く、トレーニングデータが十分でない場合、ディープラーニングメソッドは失敗する傾向があります。

---

- Another approach is to take a probabilistic stance and specify a distribution over functions, known as stochastic processes; Gaussian Processes (GPs) are an example (Rasmussen & Williams, 2004).
    - 別のアプローチは、確率的スタンスを取り、確率過程として知られる関数の分布を指定することです。 ガウス過程（GP）は一例です（Rasmussen＆Williams、2004）。

- On this view, a practitioner’s prior knowledge about f is captured in the distributional assumptions about the prior process and learning corresponds to Bayesian inference over the functional space conditioned on the observed values.
    - この観点では、真の分布 f についての実務者の事前知識が、事前プロセスに関する分布（＝事前分布？）の仮定に取り込まれ、学習は観測値を条件とする関数空間に関するベイジアン推論に対応します。

- In the GP example, assumptions on the smoothness of f are captured a priori via a parametric kernel function, and g is taken to be a random function distributed according to the predictive posterior distribution. Unfortunately, such Bayesian approaches quickly become computationally intractable as the dataset or dimensionality grows (Snelson & Ghahramani, 2006).
    - GPの例では、真の分布 f の滑らかさに関する仮定はパラメトリックカーネル関数を介して事前にキャプチャされ、モデルの分布 g は予測事後分布に従って分布するランダム関数であると見なされます。 残念ながら、そのようなベイジアン手法は、データセットまたは次元が大きくなるにつれて、すぐに計算が困難になります（Snelson＆Ghahramani、2006）。

---

- In this work we propose a family of models that represent solutions to the supervised problem, and an end-to-end training approach to learning them, that combine neural networks with features reminiscent of Gaussian Processes. We call this family of models Conditional Neural Processes (CNPs), as an allusion to the fact that they define conditional distributions over functions given a set of observations.
    - この作業では、教師あり問題の解決策を表すモデルのファミリと、それらを学習するためのエンドツーエンドのトレーニングアプローチを提案します。 この一連のモデルを条件付きニューラルプロセス（CNP）と呼びます。これは、一連の観測値が与えられた関数の条件付き分布を定義するという事実の暗示としてです。

- The dependence of a CNP on the observations is parametrized by a neural network that is invariant under permutations of its inputs. We focus on architectures that scale as O(n + m) at test time, where n, m are the number of observations and targets, respectively.
    - CNPの観測値への依存は、その入力の順列の下で不変のニューラルネットワークによってパラメーター化されます。 テスト時にO（n + m）としてスケーリングするアーキテクチャに焦点を当てます。ここで、n、mはそれぞれ観測値とターゲットの数です。

- In its most basic form a CNP embeds each observation, aggregates these embeddings into a further embedding of fixed dimension with a symmetric aggregator, and conditions the function g on the aggregate embedding; see Figure 1 for a schematic representation. 
    - 最も基本的な形式では、CNPは各観測を埋め込み、これらの埋め込みを対称アグリゲーターを使用して固定次元のさらなる埋め込みに集約し、集約埋め込みの関数gを条件付けます。 - 概略図については、図1を参照してください。 

- CNPs are trained by sampling a random dataset and following a gradient step to maximize the conditional likelihood of a random subset of targets given a random observation set. This encourages CNPs to perform well across a variety of settings, i.e. n ≪ m or n ≫ m.
    - CNPは、ランダムなデータセットをサンプリングし、ランダムな観測セットが与えられたターゲットのランダムなサブセットの条件付き尤度を最大化する勾配ステップに従って追跡されます。 これにより、CNPがさまざまな設定（n≪ mまたはn≫ m）で良好に機能するようになります。

---

- xxx

- We emphasize that although CNPs share some similarities with Bayesian approaches, they do not implement Bayesian inference directly and it is not necessarily true that the conditional distributions will be consistent with respect to some prior process. However, the ability to extract prior knowledge directly from training data tied with scalability at test time can be of equal or greater importance.
    - CNPはベイジアンアプローチといくつかの類似点を共有しますが、ベイジアン推論を直接実装しておらず、条件付き分布が事前プロセスに関して一貫しているとは限らないことを強調します。 ただし、テスト時の拡張性に関連付けられたトレーニングデータから事前知識を直接抽出する機能は、同等以上の重要性があります。


# ■ 結論

## x. Conclusion


# ■ 何をしたか？詳細

## 2. Model

### 2.1. Stochastic Processes

- xxx

- let P be a probability distribution over functions f : X → Y , formally known as a stochastic process1 , then for f ∼ P , set yi = f(xi). P defines a joint distribution over the random variables {f (xi )}n+m−1 , and therefore a conditional i=0 distribution P (f (T ) | O, T ); our task is to predict the output values f(x) for every x ∈ T given O.

- xxx

- A classic assumption to make on P is that all finite sets of function evaluations of f are jointly Gaussian distributed. This class of random functions are known as Gaussian Processes (GPs). In this case, the predictive distribution P (f (T ) | O, T ) has a simple analytic form defined by prior assumptions on the pairwise correlation structure (specified via a kernel function). The framework of stochastic processes is appealing, because Bayes rule allows one to reason consistently about the predictive distribution over f imposed by observing O under a set of probabilistic assumptions. This allows the model to be data efficient, an uncommon characteristic in most deep learning models. However, in practice, it is difficult to design appropriate priors and most interesting examples of stochastic processes are computationally expensive, scaling poorly with n and m. This includes GPs which scale as O((n + m)3).
    - Pについて行う古典的な仮定は、fの関数評価のすべての有限セットが共同ガウス分布であるということです。 このクラスのランダム関数は、ガウス過程（GP）として知られています。 この場合、予測分布P（f（T）| O、T）は、ペアワイズ相関構造（カーネル関数で指定）の事前仮定によって定義された単純な分析形式を持ちます。 確率過程のフレームワークは魅力的です。ベイズ規則により、確率的仮定のセットの下でOを観察することによって課されるf上の予測分布について一貫して推論することができるからです。 これにより、モデルはデータ効率がよくなり、ほとんどのディープラーニングモデルでは一般的ではありません。 ただし、実際には、適切な事前分布を設計することは難しく、確率論的なプロセスの最も興味深い例は計算コストが高く、nとmのスケーリングが不十分です。 これには、O（（n + m）3）としてスケーリングされるGPが含まれます。

### 2.2. Conditional Neural Processes (CNPs)

- As an alternative we propose Conditional Neural Processes (CNPs), models that directly parametrize conditional stochastic processes without imposing consistency with respect to some prior process. CNPs parametrize distributions over f (T ) given a distributed representation of O of fixed dimensionality. By doing so we give up the mathematical guarantees associated with stochastic processes, trading this off for functional flexibility and scalability.
    - 別の方法として、条件付きニューラルプロセス（CNP）を提案します。これは、いくつかの事前プロセスに関して一貫性を課すことなく、条件付き確率プロセスを直接パラメーター化するモデルです。 CNPは、固定次元のOの分散表現が与えられると、f（T）上の分布をパラメータ化します。 そうすることで、確率的プロセスに関連する数学的保証を放棄し、これを機能の柔軟性と拡張性と引き換えにします。

---

- Specifically, given a set of observations O, a CNP is a conditional stochastic process Qθ that defines distributions over f(x) for inputs x ∈ T. θ is the real vector of all parameters defining Q. Inheriting from the properties of stochastic processes, we assume that Qθ is invariant to permutations of O and T.
    - 具体的には、観測Oのセットが与えられると、CNPは入力x∈Tのf（x）上の分布を定義する条件付き確率過程Qθです。θはQを定義するすべてのパラメーターの実数ベクトルです。確率過程の特性から継承し、 QθはOとTの順列に対して不変であると仮定します。

- If O′,T′ are permutations of O and T, respectively, then Qθ(f(T) | O,T) = Qθ(f(T′) | O,T′) =Qθ(f(T) | O′,T).

- In this work, we generally enforce permutation invariance with respect to T by assuming a factored structure. Specifically, we consider Qθs that factor Qθ(f(T)|O,T)=􏰉 Qθ(f(x)|O,x).
    - この作業では、一般に、因数分解された構造を仮定することにより、Tに関する順列不変性を強制します。 具体的には、Qθ（f（T）| O、T）=􏰉Qθ（f（x）| O、x）を因数とするQθを考慮します。

- In the absence x∈T of assumptions on output space Y , this is the easiest way to ensure a valid stochastic process. Still, this framework can be extended to non-factored distributions, we consider such a model in the experimental section.
    - 出力空間Yに仮定がないx∈Tの場合、これは有効な確率過程を保証する最も簡単な方法です。 それでも、このフレームワークは因子分解されていない分布に拡張できます。実験セクションでそのようなモデルを検討します。

---

- xxx

### 2.3. Training CNPs

- We train Qθ by asking it to predict O conditioned on a randomly chosen subset of O. This gives the model a signal of the uncertainty over the space X inherent in the distribution P given a set of observations.
    - ランダムに選択されたOのサブセットを条件とするOを予測するようにQθを訓練します。これにより、一連の観測値が与えられた分布Pに固有の空間Xの不確実性の信号がモデルに与えられます。

- More precisely, let f ∼ P, O = {(xi,yi)}n−1 be a set of observations, N ∼ uniform[0, . . . , n − 1]. We condition on the subset ON = {(xi, yi)}Ni=0 ⊂ O, the first N elements of O. We minimize the negative conditional log probability

- Thus, the targets it scores Qθ on include both the observed and unobserved values. In practice, we take Monte Carlo estimates of the gradient of this loss by sampling f and N .
    - したがって、Qθをスコアリングするターゲットには、観測値と非観測値の両方が含まれます。 実際には、fとNをサンプリングすることにより、この損失の勾配のモンテカルロ推定値を取ります。

---

- This approach shifts the burden of imposing prior knowledge from an analytic prior to empirical data. This has the advantage of liberating a practitioner from having to specify an analytic form for the prior, which is ultimately intended to summarize their empirical experience. Still, we emphasize that the Qθ are not necessarily a consistent set of conditionals for all observation sets, and the training routine does not guarantee that.
    - <font color="Pink">このアプローチにより、経験的データに先立つ分析から事前知識を課す負担が変わります。 これは、最終的に彼らの経験的経験を要約することを目的とする事前の分析形式を指定する必要から実務家を解放するという利点があります。 それでも、Qθは必ずしもすべての観測セットの一貫した条件セットではないことを強調し、トレーニングルーチンはそれを保証しません。</font>

---

- In summary,

- xxx

---

- Within this specification of the model there are still some aspects that can be modified to suit specific requirements. The exact implementation of h, for example, can be adapted to the data type. For low dimensional data the encoder can be implemented as an MLP, whereas for inputs with larger dimensions and spatial correlations it can also include convolutions. Finally, in the setup described the model is not able to produce any coherent samples, as it learns to model only a factored prediction of the mean and the variances, disregarding the covariance between target points. This is a result of this particular implementation of the model. One way we can obtain coherent samples is by introducing a latent variable that we can sample from. We carry out some proof-of-concept experiments on such a model in section 4.2.3.
    - このモデルの仕様内には、特定の要件に合わせて変更できるいくつかの側面がまだあります。 たとえば、hの正確な実装は、データ型に適合させることができます。 低次元データの場合、エンコーダーはMLPとして実装できますが、より大きい次元と空間相関を持つ入力の場合は畳み込みを含めることもできます。 最後に、説明したセットアップでは、モデルはコヒーレントサンプルを生成できません。ターゲットポイント間の共分散を無視して、平均と分散の因数分解された予測のみをモデル化するためです。 これは、モデルのこの特定の実装の結果です。 コヒーレントなサンプルを取得する1つの方法は、サンプリング可能な潜在変数を導入することです。 セクション4.2.3で、このようなモデルに対していくつかの概念実証実験を実行します。
    
# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. Related Work


