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

- One approach to supervised problems is to randomly initialize a parametric function g anew for each new task and spend the bulk of computation on a costly fitting phase. Prior information that a practitioner may have about f is specified via the architecture of g, the loss function, or the training details. This approach encompasses most of deep supervised learning. Since the extent of prior knowledge that can be expressed in this way is relatively limited, and learning cannot be shared between different tasks, the amount of training required is large, and deep learning methods tend to fail when training data is not plentiful.


# ■ 結論

## x. Conclusion


# ■ 何をしたか？詳細

## x. 論文の項目名


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. Related Work


