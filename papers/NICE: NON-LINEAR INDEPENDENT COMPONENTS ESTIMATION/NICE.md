> 論文まとめ（要約ver）: https://github.com/Yagami360/MachineLearning-Papers_Survey/issues/10


# ■ 論文
- 論文タイトル："NICE: NON-LINEAR INDEPENDENT COMPONENTS ESTIMATION"
- 論文リンク：https://arxiv.org/abs/1410.8516
- 論文投稿日付：2014/10/30
- 被引用数（記事作成時点）：xxx 件
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- We propose a deep learning framework for modeling complex high-dimensional densities called Non-linear Independent Component Estimation (NICE). It is based on the idea that a good representation is one in which the data has a distribution that is easy to model. 
    - 非線形独立成分推定（NICE）と呼ばれる複雑な高次元密度をモデル化するための深層学習フレームワークを提案します。優れた表現とは、データの分布がモデル化しやすいものであるという考え方に基づいています。

- For this purpose, a non-linear deterministic transformation of the data is learned that maps it to a latent space so as to make the transformed data conform to a factorized distribution, i.e, resulting in independent latent variables.    
    - この目的のために、データの非線形決定論的​​変換が、変換されたデータが因数分解 [factorized] された分布に従うように [so as to]、潜在空間にそれを写像することを学習する。例えば、独立した潜在変数が得ること。
    
- We parametrize this transformation so that computing the determinant of the Jacobian and inverse Jacobian is trivial, yet we maintain the ability to learn complex non-linear transformations, via a composition of simple building blocks, each based on a deep neural network.
    - この変換をパラメーター化して、ヤコビ行列と逆ヤコビ行列の行列式 [determinant] の計算が簡単になる [trivial] ようにしますが、それぞれがディープニューラルネットワークに基づく単純な構成要素の構成により、複雑な非線形変換を学習する能力を維持します。
    
- The training criterion is simply the exact log-likelihood, which is tractable. Unbiased ancestral sampling is also easy. We show that this approach yields good generative models on four image datasets and can be used for inpainting.
    - トレーニング基準は、正確な対数尤度であり、扱いやすいものです。偏りのない先祖サンプリングも簡単です。このアプローチにより、4つの画像データセットで優れた生成モデルが生成され、修復に使用できることがわかります。


# ■ イントロダクション（何をしたいか？）

## x. Introduction

- xxx

- In this paper, we choose f such that the the determinant of the Jacobian is trivially obtained. Moreover, its inverse f^−1 is also trivially obtained, allowing us to sample from pX (x) easily as follows:
    - この論文では、ヤコビアンの行列式が自明に [trivially] 得られるようにfを選択します。 さらに、その逆f ^ -1も簡単に得られるため、次のようにpX（x）から簡単にサンプリングできます。

- A key novelty of this paper is the design of such a transformation f that yields these two properties of “easy determinant of the Jacobian” and “easy inverse”, while allowing us to have as much capacity as needed in order to learn complex transformations. The core idea behind this is that we can split x into two blocks (x1, x2) and apply as building block a transformation from (x1, x2) to (y1, y2) of the form:
    - この論文の重要な新規性は、「ヤコビの簡単な決定要因」と「簡単な逆行列」というこれら2つの特性をもたらす変換fの設計であり、複雑な変換を学習するために必要なだけの能力を備えています。 この背後にあるコアアイデアは、xを2つのブロック（x1、x2）に分割し、（x1、x2）から（y1、y2）への変換をビルディングブロックとして適用できることです。

---

- The details, surrounding discussion, and experimental results are developed below.



# ■ 結論

## x. Conclusion


# ■ 何をしたか？詳細

## x. 論文の項目名


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. Related Work


