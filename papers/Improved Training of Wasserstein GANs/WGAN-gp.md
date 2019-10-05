# ■ 論文
- 論文タイトル："Improved Training of Wasserstein GANs"
- 論文リンク：https://arxiv.org/abs/1704.00028
- 論文投稿日付：
- 被引用数（記事作成時点）：xxx 件
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract


# ■ イントロダクション（何をしたいか？）

## x. Introduction

- 第１パラグラフ

---

- 第２パラグラフ

# ■ 結論

## x. Conclusion


# ■ 何をしたか？詳細

## 3 Difficulties with weight constraints

- We find that weight clipping in WGAN leads to optimization difficulties, and that even when optimization succeeds the resulting critic can have a pathological value surface. We explain these problems below and demonstrate their effects; however we do not claim that each one always occurs in practice, nor that they are the only such mechanisms.
    - WGANでのウェイトクリッピングは最適化の困難につながり、最適化が成功した場合でも、結果のクリティックは病的な価値面を持つことがあります。 これらの問題を以下で説明し、その効果を示します。 しかし、私たちはそれぞれが実際に常に発生すると主張したり、それらがそのようなメカニズムだけであると主張したりはしません。

---

- Our experiments use the specific form of weight constraint from [2] (hard clipping of the magnitude of each weight), but we also tried other weight constraints (L2 norm clipping, weight normalization), as well as soft constraints (L1 and L2 weight decay) and found that they exhibit similar problems.
    - 私たちの実験では、[2]（各重みの大きさのハードクリッピング）から特定の形式の重み制約を使用しますが、他の重み制約（L2ノルムクリッピング、重み正規化）とソフト制約（L1およびL2重み）も試しました。 崩壊）、それらは同様の問題を示すことがわかった。

---

- To some extent these problems can be mitigated with batch normalization in the critic, which [2] use in all of their experiments. However even with batch normalization, we observe that very deep WGAN critics often fail to converge.
    - これらの問題は、批評家のバッチ正規化である程度軽減できます。バッチ正規化は、すべての実験で使用されます[2]。 ただし、バッチの正規化を行っても、非常に深いWGAN批評家は収束に失敗することがよくあります。

### 3.1 Capacity underuse

- Implementing a k-Lipshitz constraint via weight clipping biases the critic towards much simpler functions. As stated previously in Corollary 1, the optimal WGAN critic has unit gradient norm almost everywhere under Pr and Pg ; under a weight-clipping constraint, we observe that our neural network architectures which try to attain their maximum gradient norm k end up learning extremely simple functions.
    - 重みクリッピングを介してk-Lipshitz制約を実装すると、批評家ははるかに単純な関数に偏ります。 系譜1で前述したように、最適なWGAN評論家は、PrとPgの下のほぼどこでも単位勾配ノルムを持っています。 重みクリッピング制約の下で、最大勾配ノルムkを達成しようとするニューラルネットワークアーキテクチャは、非常に単純な関数を学習することになります。

---

- To demonstrate this, we train WGAN critics with weight clipping to optimality on several toy distributions, holding the generator distribution Pg fixed at the real distribution plus unit-variance Gaussian noise. We plot value surfaces of the critics in Figure 1a. We omit batch normalization in the critic. In each case, the critic trained with weight clipping ignores higher moments of the data distribution and instead models very simple approximations to the optimal functions. In contrast, our approach does not suffer from this behavior.
    - これを実証するために、実際の分布に固定されたジェネレーター分布Pgと単位分散ガウスノイズを保持しながら、WGAN批評家にいくつかのおもちゃ分布の最適性に重みクリッピングをトレーニングします。 図1aに批評家の価値面をプロットします。 批評家ではバッチの正規化を省略します。 いずれの場合も、重みクリッピングで訓練された評論家は、データ分布のより高い瞬間を無視し、代わりに最適な関数への非常に単純な近似をモデル化します。 対照的に、私たちのアプローチはこの振る舞いの影響を受けません。


### 3.2 Exploding and vanishing gradients

- xxx

## 4 Gradient penalty

- We now propose an alternative way to enforce the Lipschitz constraint. A differentiable function is 1-Lipschtiz if and only if it has gradients with norm at most 1 everywhere, so we consider directly constraining the gradient norm of the critic’s output with respect to its input. To circumvent tractability issues, we enforce a soft version of the constraint with a penalty on the gradient norm for random samples xˆ ∼ Pxˆ . Our new objective is
    - 次に、リプシッツ制約を強制するための代替方法を提案します。 微分可能関数は1-Lipschtizであり、すべての場所でノルムが最大1の勾配を持っている場合にのみ、入力に対して批評家の出力の勾配ノルムを直接制約することを検討します。 扱いやすさの問題を回避するために、ランダムサンプルxˆ〜Pxˆの勾配ノルムにペナルティを課したソフトバージョンの制約を適用します。 私たちの新しい目標は

Sampling distribution

- We implicitly define Pxˆ sampling uniformly along straight lines between pairs of points sampled from the data distribution Pr and the generator distribution Pg. This is motivated by the fact that the optimal critic contains straight lines with gradient norm 1 connecting coupled points from Pr and Pg (see Proposition 1). Given that enforcing the unit gradient norm constraint everywhere is intractable, enforcing it only along these straight lines seems sufficient and experimentally results in good performance.
    - データ分布Prとジェネレーター分布Pgからサンプリングされたポイントのペア間の直線に沿って、Pxˆサンプリングを暗黙的に定義します。 これは、最適評論家がPrとPgからの結合点を結ぶ勾配ノルム1の直線を含むという事実に基づいています（命題1を参照）。 あらゆる場所で単位勾配ノルム制約を強制することは扱いにくいことを考えると、これらの直線に沿ってのみ強制することは十分と思われ、実験的には良好なパフォーマンスが得られます。
    

# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. Related Work


