# ■ 論文
- 論文タイトル："Augmented Neural ODEs"
- 論文リンク：https://arxiv.org/abs/1904.01681
- 論文投稿日付：
- 被引用数（記事作成時点）：xxx 件
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- We show that Neural Ordinary Differential Equations (ODEs) learn representations that preserve the topology of the input space and prove that this implies the existence of functions Neural ODEs cannot represent. To address these limitations, we introduce Augmented Neural ODEs which, in addition to being more expressive models, are empirically more stable, generalize better and have a lower computational cost than Neural ODEs.
    - ニューラル常微分方程式（ODE）は入力空間のトポロジを保存する表現を学習し、これがニューラルODEが表現できない関数の存在を暗示していることを証明します。 これらの制限に対処するために、より表現力のあるモデルであることに加えて、経験的に安定であり、一般化が良く、計算コストがニューラルODEよりも低い拡張ニューラルODEを導入します。


# ■ イントロダクション（何をしたいか？）

## x. Introduction

- xxx

---

- In this work, we explore some of the consequences of taking this continuous limit and which restrictions this might create compared with regular neural nets. In particular, we show that there are simple classes of functions Neural ODEs (NODEs) cannot represent. While it is often possible for NODEs to approximate these functions in practice, the resulting flows are complex and lead to ODE problems that are computationally expensive to solve. To overcome these limitations, we introduce Augmented Neural ODEs (ANODEs) which are a simple extension of NODEs. ANODEs augment the space on which the ODE is solved, allowing the model to use the additional dimensions to learn more complex functions using simpler flows (see Fig. 1). In addition to being more expressive models, ANODEs significantly reduce the computational cost of both forward and backward passes of the model compared with NODEs. Our experiments also show that ANODEs generalize better, achieve lower losses with fewer parameters and are more stable to train.


# ■ 結論

## x. Conclusion


# ■ 何をしたか？詳細

## x. 論文の項目名


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. Related Work


