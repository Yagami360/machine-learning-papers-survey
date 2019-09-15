> 論文まとめ（要約ver）: https://github.com/Yagami360/MachineLearning-Papers_Survey/issues/15

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
    - フローベースの生成モデルは、可逆 [invertible] な写像により確率分布をパラメーター化し、最尤法で学習できます。

- Invertible residual networks provide a flexible family of transformations where only Lipschitz conditions rather than strict architectural constraints are needed for enforcing invertibility.
    - 可逆的 [Invertible] residual networks は、厳格なアーキテクチャ上の制約ではなく、リプシッツ条件のみが可逆性を強制するために必要な、柔軟な変換集合を提供します。

- However, prior work trained invertible residual networks for density estimation by relying on biased log-density estimates whose bias increased with the network’s expressiveness.
    - しかし、以前の研究では、ネットワークの表現力とともにバイアスが増加したバイアスされたログ密度推定に依存することにより、密度推定のための可逆残差ネットワークを訓練していました。 

> ログ密度推定 : 目的である観測データ x  を生成する確率分布の対数尤度

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
    - この点において [In this regard]、変数変換の公式 [change of variables theorem] は、扱いやすい [tractable] 正確なサンプリングとその密度の効率的な評価を可能にするような、柔軟な分布を構築するための魅力的な [appealing] 方法を提供します。
    
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

## 2 Background

### Maximum likelihood estimation.

- xxx

### Change of variables theorem.

- With an invertible transformation f , the change of variables


- captures the change in density of the transformed samples. 

- A simple base distribution such as a standard normal is often used for logp(f(x)). Tractable evaluation of (2) allows flow-based models to be trained using the maximum likelihood objective (1). In contrast, variational autoen- coders (Kingma and Welling, 2014) can only optimize a stochastic lower bound, and generative adversial networks (Goodfellow et al., 2014) require an extra discriminator network for training.
    - 標準正規分布などの単純な基本分布は、logp（f（x））によく使用されます。 （2）の実行可能な評価により、最尤目標（1）を使用してフローベースモデルをトレーニングできます。 対照的に、変分オートエンコーダー（Kingma and Welling、2014）は確率的下限のみを最適化でき、生成的敵対ネットワーク（Goodfellow et al。、2014）はトレーニングに追加の判別器ネットワークを必要とします。


### Invertible residual networks (i-ResNets).

- Residual networks are composed of simple transformations y = f(x) = x + g(x). Behrmann et al. (2019) noted that this transformation is invertible by the Banach fixed point theorem if g is contractive, i.e with Lipschitz constant strictly less than unity, which was enforced using spectral normalization (Miyato et al., 2018; Gouk et al., 2018).
    - 残存ネットワークは、単純な変換y = f（x）= x + g（x）で構成されます。 Behrmann et al（2019）は、gが収縮性の場合、つまりスペクトル正規化を使用して強制されたリプシッツ定数が厳密に1未満である場合、この変換はバナッハ不動点定理によって可逆的であることに注目しました（Miyato et al。、2018; Gouk et al。 、2018）。

- xxx

## 3 Residual Flows

### 3.1 Unbiased Log Density Estimation for Maximum Likelihood Estimation

- Evaluation of the exact log density function log pθ (·) in (3) requires infinite time due to the power series. Instead, we rely on randomization to derive an unbiased estimator that can be computed in finite time (with probability one) based on an existing concept (Kahn, 1955).
    - （3）の正確な対数密度関数logpθ（・）の評価には、べき級数のために無限の時間が必要です。 代わりに、既存の概念に基づいて有限確率（確率1）で計算できる不偏推定量を導き出すために、ランダム化に依存しています（Kahn、1955）。

- To illustrate the idea, let ∆k denote the k-th term of an infinite series, and suppose we always evaluate the first term then flip a coin b ∼ Bernoulli(q) to determine whether we stop or continue evaluating the remaining terms. By reweighting the remaining terms by 1 , we obtain an unbiased estimator
    - 考え方を説明するために、Δkが無限級数のk番目の項を表し、最初の項を常に評価し、その後コインb〜Bernoulli（q）を反転させて残りの項の評価を停止するか続行するかを決定するとします。 残りの項を1で再重み付けすることにより、不偏推定量を取得します

> 式

- Interestingly, whereas naïve computation would always use infinite compute, this unbiased estimator has probability q of being evaluated in finite time. We can obtain an estimator that is evaluated in finite time with probability one by applying this process infinitely many times to the remaining terms. Directly sampling the number of evaluated terms, we obtain the appropriately named “Russian roulette” estimator (Kahn, 1955)
    - 興味深いことに、ナイーブ計算では常に無限計算が使用されますが、この不偏推定量には有限時間で評価される確率qがあります。 このプロセスを残りの項に無限に何度も適用することで、確率1で有限時間で評価される推定量を取得できます。 評価された用語の数を直接サンプリングして、適切な名前の「ロシアンルーレット」推定量を取得します（Kahn、1955）

> 式

- We note that the explanation above is only meant to be an intuitive guide and not a formal derivation. The peculiarities of dealing with infinite quantities dictate that we must make assumptions on ∆k, p(N ), or both in order for the equality in (5) to hold. While many existing works have made different assumptions depending on specific applications of (5), we state our result as a theorem where the only condition is that p(N ) must have support over all of the indices.
    - 上記の説明は、直感的なガイドであり、正式な派生物ではないことに注意してください。 無限の量を処理する特性により、（5）の等式が成立するためには、Δk、p（N）、またはその両方を仮定する必要があります。 多くの既存の作品は（5）の特定のアプリケーションに応じて異なる仮定を行っていますが、p（N）がすべてのインデックスをサポートする必要があるという唯一の条件である定理として結果を述べています。

- Note that since Jg is constrained to have a spectral radius less than unity, the power series converges exponentially. The variance of the Russian roulette estimator is small when the infinite series exhibits fast convergence (Rhee and Glynn, 2015; Beatson and Adams, 2019), and in practice, we did not have to tune p(N) for 3.0 variance reduction. Instead, in our experiments, we compute two terms exactly and then use 2.50 5 the unbiased estimator on the remaining terms with a single sample from p(N ) = Geom(0.5). This results in an expected compute cost of 4 terms, which is less than the 5 to 10 terms that Behrmann et al. (2019) used for their biased estimator.
    - Jgは1未満のスペクトル半径を持つように制約されているため、べき級数は指数関数的に収束することに注意してください。 無限級数が高速収束を示す場合、ロシアのルーレット推定量の分散は小さく（RheeとGlynn、2015; BeatsonとAdams、2019）、実際には3.0の分散低減のためにp（N）を調整する必要はありませんでした。 代わりに、我々の実験では、2つの項を正確に計算し、p（N）= Geom（0.5）からの単一のサンプルで残りの項に不偏推定量2.50を使用します。 これにより、予想される計算条件は4項になり、Behrmann et al。の5〜10項よりも小さくなります。 （2019）バイアス推定器に使用されます。

- Theorem 1 forms the core of Residual Flows, as we can now perform maximum likelihood training by backpropagating through (6) to obtain unbiased gradients. This allows us to train more expressive networks where a biased estimator would fail (Figure 2). The price we pay for the unbiased estimator is variable compute and memory, as each sample of the log density uses a random number of terms in the power series.
    - 定理1は、（6）を逆伝播して不偏勾配を取得することにより最尤トレーニングを実行できるため、残差フローのコアを形成します。 これにより、バイアスのかかった推定器が失敗する、より表現力のあるネットワークをトレーニングできます（図2）。 対数密度の各サンプルはべき級数の項の乱数を使用するため、不偏推定量に支払う価格は可変計算とメモリです。

### 3.2 Memory-Efficient Backpropagation

- Memory can be a scarce resource, and running out of memory due to a large sample from the unbiased estimator can halt training unexpectedly. To this end, we propose two methods to reduce the memory consumption during training.
    - メモリは希少なリソースである可能性があり、偏りのない推定器からの大きなサンプルが原因でメモリが不足すると、トレーニングが予期せず停止する可能性があります。 この目的のために、トレーニング中のメモリ消費を削減する2つの方法を提案します。

- To see how naïve backpropagation can be problematic, the gradient w.r.t. parameters θ by directly differentiating through the power series (6) can be expressed as
    - ナイーブバックプロパゲーションがどのように問題になるかを確認するために、べき級数（6）を直接微分することによるパラメーターθに関する勾配は、


> 式

- Unfortunately, this estimator requires each term to be stored in memory because ∂/∂θ needs to be applied to each term. The total memory cost is then O(n · m) where n is the number of computed terms and m is the number of residual blocks in the entire network. This is extremely memory-hungry during training, and a large random sample of n can occasionally result in running out of memory.
    - 残念ながら、this /∂θを各項に適用する必要があるため、この推定器では各項をメモリに保存する必要があります。 合計メモリコストはO（n・m）です。ここで、nは計算された項の数、mはネットワーク全体の残差ブロックの数です。 これは、トレーニング中に非常にメモリを消費し、nの大きなランダムサンプルがメモリ不足になる場合があります。


- As the power series in (8) does not need to be differentiated through, using this reduces the memory requirement by a factor of n. This is especially useful when using the unbiased estimator as the memory will be constant regardless of the number of terms we draw from p(N ).


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. Related Work


