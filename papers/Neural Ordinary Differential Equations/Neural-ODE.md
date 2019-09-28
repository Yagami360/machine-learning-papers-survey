> 論文まとめ（要約ver）: https://github.com/Yagami360/MachineLearning-Papers_Survey/issues/21

# ■ 論文
- 論文タイトル："Neural Ordinary Differential Equations"
- 論文リンク：
- 論文投稿日付：
- 被引用数（記事作成時点）：xxx 件
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- We introduce a new family of deep neural network models. Instead of specifying a discrete sequence of hidden layers, we parameterize the derivative of the hidden state using a neural network. The output of the network is computed using a black- box differential equation solver.
    - ディープニューラルネットワークモデルの新しいファミリを紹介します。 隠れ層の離散的な [discrete] シーケンスを指定する代わりに、ニューラルネットワークを使用して隠れ状態の導関数 [derivative] をパラメーター化します。 ネットワークの出力は、ブラックボックス微分方程式 [differential equation] ソルバーを使用して計算されます。

- These continuous-depth models have constant memory cost, adapt their evaluation strategy to each input, and can explicitly trade numerical precision for speed.
    - これらの連続深さモデルは、一定のメモリコストを持ち、評価戦略を各入力に適合させ、数値精度と速度を明示的に交換できます。
    
- We demonstrate these properties in continuous-depth residual networks and continuous-time latent variable models.
    - これらの特性を連続深度残差ネットワークと連続時間潜在変数モデルで示します。 

- We also construct continuous normalizing flows, a generative model that can train by maximum likelihood, without partitioning or ordering the data dimensions.
    - また、データ次元を分割または順序付けすることなく、最尤法でトレーニングできる生成モデルである連続正規化フローも構築します。
    
- For training, we show how to scalably backpropagate through any ODE solver, without access to its internal operations. This allows end-to-end training of ODEs within larger models.
    - トレーニングでは、内部操作にアクセスせずに、ODEソルバーを介してスケーラブルに逆伝播する方法を示します。 これにより、より大きなモデル内でODEのエンドツーエンドのトレーニングが可能になります。

# ■ イントロダクション（何をしたいか？）

## 1. Introduction

- Models such as residual networks, recurrent neural network decoders, and normalizing flows build complicated transformations by composing a sequence of transformations to a hidden state:
    - 残差ネットワーク、リカレントニューラルネットワークデコーダー、正規化フローなどのモデルは、一連の変換を隠れ層に構成することにより、複雑な変換を構築します。

> 式

---

- What happens as we add more layers and take smaller steps? In the limit, we parameterize the continuous dynamics of hidden units using an ordinary differential equation (ODE) specified by a neural network:
    - レイヤーを追加し、より小さなステップを踏むとどうなりますか？ 限界では、ニューラルネットワークで指定された常微分方程式（ODE）を使用して、隠れユニットの連続ダイナミクスをパラメーター化します。

- Starting from the input layer h(0), we can define the output layer h(T ) to be the solution to this ODE initial value problem at some time T . This value can be computed by a black-box differential equation solver, which evaluates the hidden unit dynamics f wherever necessary to determine the solution with the desired accuracy. Figure 1 contrasts these two approaches.
    - 入力層h（0）から開始して、出力層h（T）を、ある時間TにおけるこのODE初期値問題の解となるように定義できます。 この値は、ブラックボックス微分方程式ソルバーによって計算できます。このソルバーは、目的の精度で解を決定するために必要な場合に隠れユニットダイナミクスfを評価します。 図1は、これら2つのアプローチを対比しています。

---

- Defining and evaluating models using ODE solvers has several benefits:

#### Memory efficiency

- In Section 2, we show how to compute gradients of a scalar-valued loss with respect to all inputs of any ODE solver, without backpropagating through the operations of the solver. Not storing any intermediate quantities of the forward pass allows us to train our models with constant memory cost as a function of depth, a major bottleneck of training deep models.
    - セクション2では、ソルバーの操作を逆伝播することなく、ODEソルバーのすべての入力に関してスカラー値損失の勾配を計算する方法を示します。 フォワードパスの中間量を保存しないことで、深さの関数として一定のメモリコストでモデルをトレーニングできます。これは、深層モデルのトレーニングの大きなボトルネックです。


#### Adaptive computation

- Euler’s method is perhaps the simplest method for solving ODEs. There have since been more than 120 years of development of efficient and accurate ODE solvers (Runge, 1895; Kutta, 1901; Hairer et al., 1987). Modern ODE solvers provide guarantees about the growth of approximation error, monitor the level of error, and adapt their evaluation strategy on the fly to achieve the requested level of accuracy. This allows the cost of evaluating a model to scale with problem complexity. After training, accuracy can be reduced for real-time or low-power applications.
    - オイラーの方法は、おそらくODEを解くための最も簡単な方法です。 それ以来、効率的で正確なODEソルバーの開発は120年以上ありました（Runge、1895; Kutta、1901; Hairer et al。、1987）。 最新のODEソルバーは、近似誤差の増大に関する保証を提供し、誤差のレベルを監視し、要求された精度レベルを達成するために評価戦略をその場で調整します。 これにより、モデルを評価するコストが問題の複雑さに応じて拡大します。 トレーニング後、リアルタイムまたは低電力アプリケーションの精度が低下する可能性があります。

#### Parameter efficiency

- When the hidden unit dynamics are parameterized as a continuous function of time, the parameters of nearby “layers” are automatically tied together. In Section 3, we show that this reduces the number of parameters required on a supervised learning task.
    - 非表示のユニットダイナミクスが時間の連続関数としてパラメーター化されると、近くの「レイヤー」のパラメーターが自動的に結び付けられます。 セクション3では、教師あり学習タスクに必要なパラメーターの数を減らすことを示します。

#### Scalable and invertible normalizing flows

- An unexpected side-benefit of continuous transformations is that the change of variables formula becomes easier to compute. In Section 4, we derive this result and use it to construct a new class of invertible density models that avoids the single-unit bottleneck of normalizing flows, and can be trained directly by maximum likelihood.
    - 連続変換の予期しない副次的な利点は、変数の変更式が計算しやすくなることです。 セクション4では、この結果を導出し、それを使用して、フローを正規化する単一ユニットのボトルネックを回避し、最尤法で直接トレーニングできる可逆密度モデルの新しいクラスを構築します。


### 2 Reverse-mode automatic differentiation of ODE solutions

- The main technical difficulty in training continuous-depth networks is performing reverse-mode differentiation (also known as backpropagation) through the ODE solver. Differentiating through the operations of the forward pass is straightforward, but incurs a high memory cost and introduces additional numerical error.
    - 連続深さのネットワークをトレーニングする際の主な技術的困難は、ODEソルバーを介して逆モード微分（逆伝播とも呼ばれます）を実行することです。 フォワードパスの操作を区別することは簡単ですが、メモリコストが高くなり、追加の数値エラーが発生します。

- We treat the ODE solver as a black box, and compute gradients using the adjoint sensitivity method (Pontryagin et al, 1962). This approach computes gradients by solving a second, augmented ODE backwards in time, and is applicable to all ODE solvers. This approach scales linearly with problem size, has low memory cost, and explicitly controls numerical error.
    - ODEソルバーをブラックボックスとして扱い、adjoint sensitivity method を使用して勾配を計算します（Pontryagin et al、1962）。 このアプローチは、2番目の拡張ODEを時間を遡って解くことにより勾配を計算し、すべてのODEソルバーに適用できます。 このアプローチは問題のサイズに比例してスケーリングし、メモリコストが低く、数値エラーを明示的に制御します。

---

- xxx

> 式 (3)

- To optimize L, we require gradients with respect to θ. The first step is to determining how the gradient of the loss depends on the hidden state z(t) at each instant. This quantity is called the adjoint a(t) = ∂L/∂z(t). Its dynamics are given by another ODE, which can be thought of as the instantaneous analog of the chain rule:
    - Lを最適化するには、θに関する勾配が必要です。 最初のステップは、損失の勾配が各瞬間の隠れ状態z（t）にどのように依存するかを決定することです。 この量は、adjoint a（t）=∂L/∂z（t）と呼ばれます。 そのダイナミクスは、チェーンルールの瞬間的な類似物と考えることができる別のODEによって提供されます。

> 式 (4)

- We can compute ∂L/∂z(t0) by another call to an ODE solver. This solver must run backwards, starting from the initial value of ∂L/∂z(t1). One complication is that solving this ODE requires the knowing value of z(t) along its entire trajectory. However, we can simply recompute z(t) backwards in time together with the adjoint, starting from its final value z(t1 ).
    - <font color="Pink">ODEソルバーの別の呼び出しによって、∂L/∂z（t0）を計算できます。 このソルバーは、初期値 L /∂z（t1）から逆方向に実行する必要があります。 複雑な点の1つは、このODEを解くには、その軌跡 [trajectory] 全体に沿ったz（t）の値を知る必要があるということです。 ただし、最終的な値z（t1）から開始して、adjoint とともにz（t）を逆方向に再計算することができます。</font>


- Computing the gradients with respect to the parameters θ requires evaluating a third integral, which depends on both z(t) and a(t):
    - パラメーターθに関する勾配を計算するには、z（t）とa（t）の両方に依存する3番目の積分を評価する必要があります。

> 式 (5)

- The vector-Jacobian products a(t)^T * ∂f/∂z and a(t)^T * ∂f/∂θ in (4) and (5) can be efficiently evaluated by automatic differentiation, at a time cost similar to that of evaluating f.
    - <font color="Pink">（4）および（5）のベクトルヤコビアン積 a(t)^T * ∂f/∂z and a(t)^T * ∂f/∂θ は、f の評価と同様の時間コストで、自動微分により効率的に評価できます。</font>

- All integrals for solving z, adjoint a(t) and ∂L/∂θ can be computed in a single call to an ODE solver, which concatenates the original state, the adjoint, and the other partial derivatives into a single vector. Algorithm 1 shows how to construct the necessary dynamics, and call an ODE solver to compute all gradients at once.
    - <font color="Pink">z と adjoint a（t）および∂L/∂θを解くためのすべての積分は、ODEソルバーの1回の呼び出しで計算でき、元の状態、adjoint、および他の偏微分を単一のベクトルに連結します。 アルゴリズム1は、必要なダイナミクスを構築し、ODEソルバーを呼び出してすべての勾配を一度に計算する方法を示しています。</font>

> アルゴリズム１

---

- Most ODE solvers have the option to output the state z(t) at multiple times. When the loss depends on these intermediate states, the reverse-mode derivative must be broken into a sequence of separate solves, one between each consecutive pair of output times (Figure 2). At each observation, the adjoint must be adjusted in the direction of the corresponding partial derivative ∂L/∂z(ti).
    - <font color="Pink">ほとんどのODEソルバーには、状態z（t）を複数回出力するオプションがあります。 損失がこれらの中間状態に依存する場合、逆モード微分は、出力時間の連続する各ペアの間に1つずつ、一連の個別の解に分割する必要があります（図2）。 各観測で、対応する偏微分∂L/∂z（ti）の方向に adjoint を調整する必要があります。</font>

> 図２

- > Figure 2: Reverse-mode differentiation of an ODE solution. The adjoint sensitivity method solves an augmented ODE backwards in time. The augmented system contains both the original state and the sensitivity of the loss with respect to the state. If the loss depends directly on the state at multiple observation times, the adjoint state must be updated in the direction of the partial derivative of the loss with respect to each observation.
    - > <font color="Pink">ODEソリューションの逆モード微分。 adjoint sensitivity method は、拡張されたODEを時間的に遡って解決します。 拡張 [augmented] システムには、元の状態とその状態に関する損失の感度の両方が含まれます。 損失が複数の観測時間における状態に直接依存する場合、各観測に関して損失の偏微分の方向に adjoint state を更新する必要があります。</font>

---

- The results above extend those of Stapor et al. (2018, section 2.4.2). An extended version of Algorithm 1 including derivatives w.r.t. t0 and t1 can be found in Appendix C. Detailed derivations are provided in Appendix B. Appendix D provides Python code which computes all derivatives for scipy.integrate.odeint by extending the autograd automatic differentiation package. This code also supports all higher-order derivatives. We have since released a PyTorch (Paszke et al., 2017) implementation, including GPU-based implementations of several standard ODE solvers at github.com/rtqichen/torchdiffeq.
    - 上記の結果はStaporらの結果を拡張したものです。 （2018年、セクション2.4.2）。 派生物w.r.tを含むアルゴリズム1の拡張バージョン。 t0およびt1は付録Cにあります。詳細な派生は付録Bにあります。付録Dはautograd自動差分パッケージを拡張することによりscipy.integrate.odeintのすべての派生を計算するPythonコードを提供します。 このコードは、すべての高階微分もサポートしています。 それ以来、github.com / rtqichen / torchdiffeqでいくつかの標準ODEソルバーのGPUベースの実装を含むPyTorch（Paszke et al。、2017）実装をリリースしました。
    

# ■ 結論

## x. Conclusion


# ■ 何をしたか？詳細

## x. 論文の項目名


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. Related Work


