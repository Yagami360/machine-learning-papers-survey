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

- How can we perform efficient approximate inference and learning with directed probabilistic models whose continuous latent variables and/or parameters have intractable posterior distributions? The variational Bayesian (VB) approach involves the optimization of an approximation to the intractable posterior. Unfortunately, the common mean-field approach requires analytical solutions of expectations w.r.t. the approximate posterior, which are also intractable in the general case. We show how a reparameterization of the variational lower bound yields a simple differentiable unbiased estimator of the lower bound; this SGVB (Stochastic Gradient Variational Bayes) estimator can be used for efficient approximate posterior inference in almost any model with continuous latent variables and/or parameters, and is straightforward to optimize using standard stochastic gradient ascent techniques.


---

- For the case of an i.i.d. dataset and continuous latent variables per datapoint, we propose the Auto- Encoding VB (AEVB) algorithm. In the AEVB algorithm we make inference and learning especially efficient by using the SGVB estimator to optimize a recognition model that allows us to perform very efficient approximate posterior inference using simple ancestral sampling, which in turn allows us to efficiently learn the model parameters, without the need of expensive iterative inference schemes (such as MCMC) per datapoint. The learned approximate posterior inference model can also be used for a host of tasks such as recognition, denoising, representation and visualization purposes. When a neural network is used for the recognition model, we arrive at the variational auto-encoder.


# ■ 結論

## x. Conclusion


# ■ 何をしたか？詳細

## x. 論文の項目名


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. Related Work


