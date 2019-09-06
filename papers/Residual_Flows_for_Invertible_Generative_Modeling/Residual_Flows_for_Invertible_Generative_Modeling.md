# ■ 論文
- 論文タイトル："Residual Flows for Invertible Generative Modeling"
- 論文リンク：https://arxiv.org/abs/1906.02735
- 論文投稿日付：2019/06/06
- 被引用数（記事作成時点）：xxx 件
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- Flow-based generative models parameterize probability distributions through an invertible transformation and can be trained by maximum likelihood.
    - フローベースの生成モデルは、可逆 [invertible] 変換により確率分布をパラメーター化し、最尤法でトレーニングできます。

- Invertible residual networks provide a flexible family of transformations where only Lipschitz conditions rather than strict architectural constraints are needed for enforcing invertibility.
    - 可逆的 [Invertible] residual networks は、厳格なアーキテクチャ上の制約ではなく、リプシッツ条件のみが可逆性を強制するために必要な、柔軟な変換集合を提供します。

- However, prior work trained invertible residual networks for density estimation by relying on biased log-density estimates whose bias increased with the network’s expressiveness.
    - しかし、以前の研究では、ネットワークの表現力とともにバイアスが増加したバイアスされたログ密度推定に依存することにより、密度推定のための可逆残差ネットワークを訓練していました。 

- We give a tractable unbiased estimate of the log density using a “Russian roulette” estimator, and reduce the memory required during training by using an alternative infinite series for the gradient. 
    - 「ロシアンルーレット」推定器を使用して、対数密度の扱いやすい不偏推定値を与え、勾配に別の無限級数を使用することにより、トレーニング中に必要なメモリを削減します。

- Furthermore, we improve invertible residual blocks by proposing the use of activation functions that avoid derivative saturation and generalizing the Lipschitz condition to induced mixed norms.
    - さらに、微分飽和を回避するような活性化関数の使用を提案することにより、
    - 又、リプシッツ条件を、誘導された混合ノルム [mixed norms] に一般化することにより、
    - 可逆 [invertible] residual blocks 改善します。

- The resulting approach, called Residual Flows, achieves state-of-the-art performance on density estimation amongst flow-based models, and outperforms networks that use coupling blocks at joint generative and discriminative modeling.
    - 残差フローと呼ばれる結果のアプローチは、フローベースのモデル間で密度推定に関する最先端のパフォーマンスを達成し、結合生成および識別モデリングで結合ブロックを使用するネットワークよりも優れています。

# ■ イントロダクション（何をしたいか？）

## 1. Introduction

- Maximum likelihood is a core machine learning paradigm that poses learning as a distribution alignment problem. However, it is often unclear what family of distributions should be used to fit high-dimensional continuous data.
    - 最尤法は、学習を分布整合問題として提起するコアマシン学習パラダイムです。 ただし、高次元の連続データに適合するためにどの分布集合を使用する必要があるかは不明なことがよくあります。

- In this regard, the change of variables theorem offers an appealing way to construct flexible distributions that allow tractable exact sampling and efficient evaluation of its density.
    - この点において [In this regard]、変数定理の変更は、扱いやすい [tractable] 正確なサンプリングとその密度の効率的な評価を可能にするような、柔軟な分布を構築するための魅力的な [appealing] 方法を提供します。
    
- This class of models is generally referred to as invertible or flow-based generative models (Deco and Brauer, 1995; Rezende and Mohamed, 2015).
    - このクラスのモデルは一般に、可逆またはフローベースの生成モデルと呼ばれます（Deco and Brauer、1995; Rezende and Mohamed、2015）。

---

- With invertibility as its core design principle, flow-based models (also referred to as normalizing flows) have shown to be capable of generating realistic images (Kingma and Dhariwal, 2018) and can achieve density estimation performance on-par with competing state-of-the-art approaches (Ho et al., 2019).
    - 可逆性を中核とする設計原則として、フローベースモデル（フローの正規化とも呼ばれます）は、現実的な画像を生成できることが示され（Kingma and Dhariwal、2018）、競合するSOTAアプローチと同等の密度推定パフォーマンスを達成できます（ Ho et al、2019）。

- In applications, they have been applied to study adversarial robustness (Jacobsen et al., 2019) and are used to train hybrid models with both generative and classification capabilities (Nalisnick et al., 2019) using a weighted maximum likelihood objective.
    - アプリケーションでは、敵対的堅牢性の研究に適用され（Jacobsen et al。、2019）、重み付き最尤目標を使用して生成機能と分類機能（Nalisnick et al。、2019）の両方を備えたハイブリッドモデルのトレーニングに使用されます。

---

- Existing flow-based models (Rezende and Mohamed, 2015; Kingma et al., 2016; Dinh et al., 2014; Chen et al., 2018) make use of restricted transformations with sparse or structured Jacobians (Figure 1).
    - 既存のフローベースのモデル（Rezende and Mohamed、2015; Kingma et al。、2016; Dinh et al。、2014; Chen et al。、2018）は、スパースまたは構造化されたヤコビアンによる制限された変換を利用します（図1）。


- These allow efficient computation of the log probability under the model but at the cost of architectural engineering.
- Transformations that scale to high-dimensional data rely on specialized architectures such as coupling blocks (Dinh et al., 2014, 2017) or solving an ordinary differential equation (Grathwohl et al., 2019).
- Such approaches have a strong inductive bias that can hinder their application in other tasks, such as learning representations that are suitable for both generative and discriminative tasks.
    - これらは、モデルの下での対数確率の効率的な計算を可能にしますが、建築工学の費用がかかります。
    - 高次元データにスケーリングする変換は、カップリングブロック（Dinh et al。、2014、2017）や常微分方程式の解法（Grathwohl et al。、2019）などの特殊なアーキテクチャに依存しています。
    - このようなアプローチには、生成的タスクと識別的タスクの両方に適した学習表現など、他のタスクへの適用を妨げる可能性がある強い誘導バイアスがあります。

---

- Recent work by Behrmann et al (2019) showed that residual networks (He et al, 2016) can be made invertible by simply enforcing a Lipschitz constraint, allowing to use a very successful discriminative deep network architecture for unsupervised flow-based modeling.
- Unfortunately, the density evaluation requires computing an infinite series.
    - Behrmann et al（2019）の最近の研究では、残差ネットワーク（He et al、2016）を単純にリプシッツ制約を適用することで可逆的にできることを示し、教師なしフローベースモデリングに非常に成功した判別的ディープネットワークアーキテクチャを使用できるようになりました。
    - 残念ながら、密度の評価には無限級数の計算が必要です。

- The choice of a fixed truncation estimator used by Behrmann et al (2019) leads to substantial bias that is tightly coupled with the expressiveness of the network, and cannot be said to be performing maximum likelihood as bias is introduced in the objective and gradients.
    - Behrmann et al（2019）が使用する固定切り捨て推定器の選択は、ネットワークの表現力と密接に結びついた実質的なバイアスにつながり、バイアスが客観と勾配に導入されるため、最尤を実行するとは言えません。

---

- In this work, we introduce Residual Flows, a flow-based generative model that produces an unbiased estimate of the log density and has memory-efficient backpropagation through the log density computation. 
- This allows us to use expressive architectures and train via maximum likelihood.
- Furthermore, we propose and experiment with the use of activations functions that avoid derivative saturation and induced mixed norms for Lipschitz-constrained neural networks.
    - **この研究では、ログ密度の不偏推定値を生成し、ログ密度計算によるメモリ効率の良い逆伝播を行うフローベースの生成モデルである残差フローを導入します。**
    - **これにより、表現力豊かなアーキテクチャを使用して、最尤法でトレーニングすることができます。**
    - **さらに、リプシッツ制約付きニューラルネットワークの微分飽和と誘導混合ノルムを回避する活性化関数の使用を提案し、実験します。**

# ■ 結論

## 6. Conclusion

- We have shown that invertible residual networks can be turned into powerful generative models.
- The proposed unbiased flow-based generative model, coined Residual Flow, achieves competitive or better performance compared to alternative flow-based models in density estimation, sample quality, and hybrid modeling.
    - 可逆的残差ネットワークは強力な生成モデルに変換できることを示しました。 提案された不偏フローベースの生成モデルである残差フローは、密度推定、サンプル品質、およびハイブリッドモデリングにおいて、フローベースの代替モデルと比較して競争力のあるまたは優れたパフォーマンスを実現します。
    
- More generally, we gave a recipe for introducing stochasticity in order to construct tractable flow-based models with a different set of constraints on layer architectures than competing approaches, which rely on exact log-determinant computations. This opens up a new design space of expressive but Lipschitz-constrained architectures that has yet to be explored.
    - より一般的には、正確な対数決定計算に依存する競合するアプローチとは異なる、レイヤーアーキテクチャ上の制約のセットを持つ扱いやすいフローベースモデルを構築するために、確率論を導入するためのレシピを提供しました。 これにより、まだ探求されていない、表現力豊かであるがリプシッツに制約されたアーキテクチャの新しい設計空間が開かれます。


# ■ 何をしたか？詳細

## x. 論文の項目名


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. Related Work


