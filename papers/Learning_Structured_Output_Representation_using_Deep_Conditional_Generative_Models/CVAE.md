# ■ 論文
- 論文タイトル："Learning Structured Output Representation using Deep Conditional Generative Models"
- 論文リンク：https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models
- 論文投稿日付：
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- Supervised deep learning has been successfully applied to many recognition probems. Although it can approximate a complex many-to-one function well when a large amount of training data is provided, it is still challenging to model complex structured output representations that effectively perform probabilistic inference and make diverse predictions. In this work, we develop a deep conditional generative model for structured output prediction using Gaussian latent variables. The model is trained efficiently in the framework of stochastic gradient variational Bayes, and allows for fast prediction using stochastic feed-forward inference. In addition, we provide novel strategies to build robust structured prediction algorithms, such as input noise-injection and multi-scale prediction objective at training. In experiments, we demonstrate the effectiveness of our proposed algorithm in comparison to the deterministic deep neural network counterparts in generating diverse but realistic structured output predictions using stochastic inference. Furthermore, the proposed training methods are complimentary, which leads to strong pixel-level object segmentation and semantic labeling performance on Caltech-UCSD Birds 200 and the subset of Labeled Faces in the Wild dataset.
    - 教師付き深層学習は、多くの認識問題にうまく適用されてきました。大量のトレーニングデータが提供される場合、複雑な多対一関数によく近似できますが、確率的推論を効果的に実行し、多様な予測を行う複雑な構造化出力表現をモデル化することは依然として困難です。本研究では、Gaussian潜在変数を用いた構造化出力予測のためのディープ条件付き生成モデルを開発する。モデルは確率的勾配変分ベイズの枠組みで効率的に訓練され、確率的フィードフォワード推論を使用して高速予測を可能にします。さらに、入力ノイズインジェクションやトレーニング時のマルチスケール予測など、堅牢な構造化予測アルゴリズムを構築するための新しい戦略を提供します。実験では、確率論的推論を使用して多様ではあるが現実的な構造化出力予測を生成する際に、決定論的ディープニューラルネットワークの対応物と比較して、提案したアルゴリズムの有効性を実証する。さらに、提案されたトレーニング方法は相補的であり、それはCaltech-UCSD Birds 200およびWildデータセット内のLabeled Facesのサブセットに対する強力なピクセルレベルのオブジェクトセグメンテーションおよび意味的ラベリング性能をもたらす。


# ■ イントロダクション（何をしたいか？）

## x. Introduction

- 第１パラグラフ

---

- 第２パラグラフ

# ■ 結論

## x. Conclusion


# ■ 何をしたか？詳細

### 4.1 Output inference and estimation of the conditional likelihood

- Once the model parameters are learned, we can make a prediction of an output y from an input x by following the generative process of the CGM. To evaluate the model on structured output prediction tasks (i.e, in testing time), we can measure a prediction accuracy by performing a deterministic inferencewithoutsamplingz,i.e, y∗ =argmaxypθ(y|x,z∗),z∗ =E[z|x].
    - モデルパラメータが学習されると、CGMの生成過程に従って入力xから出力yを予測することができます。
    - 構造化出力予測タスクでモデルを評価するには（つまり、テスト時間内に）、サンプリングzを使わずに決定論的推論を行うことによって予測精度を測定できます。すなわち、y * =argmaxypθ（y | x、z *）、z * = E [ ｚ ｜ ｘ］。

- Another way to evaluate the CGMs is to compare the conditional likelihoods of the test data. A straightforward approach is to draw samples z’s using the prior network and take the average of the likelihoods. We call this method the Monte Carlo (MC) sampling:
    - CGMを評価するもう1つの方法は、テストデータの条件付き尤度を比較することです。 直接的なアプローチは、従来のネットワークを使用してサンプルを取り出し、そして可能性の平均をとることである。 この方法をモンテカルロ（MC）サンプリングと呼びます。

![image](https://user-images.githubusercontent.com/25688193/62004727-cc5c7200-b163-11e9-9091-1c3335d64e66.png)

- It usually requires a large number of samples for the Monte Carlo log-likelihood estimation to be accurate. Alternatively, we use the importance sampling to estimate the conditional likelihoods [24]:
    - それは通常、モンテカルロ対数尤度推定が正確であるためには多数のサンプルを必要とする。 
    - 代わりに、重要度サンプリングを使用して条件付き尤度を推定します[24]。

![image](https://user-images.githubusercontent.com/25688193/62004762-31b06300-b164-11e9-958a-bcd257396ec4.png)

# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. Related Work


