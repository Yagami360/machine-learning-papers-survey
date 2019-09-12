

# ■ 論文
- 論文タイトル："Invertible Residual Networks"
- 論文リンク：
- 論文投稿日付：2018/11/02
- 被引用数（記事作成時点）：xxx 件
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- We show that standard ResNet architectures can be made invertible, allowing the same model to be used for classification, density estimation, and generation. Typically, enforcing invertibility requires partitioning dimensions or restricting network architectures. In contrast, our approach only requires adding a simple normalization step during training, already available in standard frameworks. Invertible ResNets define a generative model which can be trained by maximum likelihood on unlabeled data. To compute likelihoods, we introduce a tractable approximation to the Jacobian log-determinant of a residual block. Our empirical evaluation shows that invertible ResNets perform competitively with both state- of-the-art image classifiers and flow-based generative models, something that has not been previously achieved with a single architecture.
    - 標準のResNetアーキテクチャを可逆的に作成できることを示し、同じモデルを分類、密度推定、および生成に使用できるようにします。 通常、可逆性を適用するには、ディメンションを分割する [partitioning] か、ネットワークアーキテクチャを制限する必要があります。 対照的に、このアプローチでは、トレーニング中に標準フレームワークですでに利用可能な単純な正規化ステップを追加するだけで済みます。 可逆ResNetは、ラベルなしデータの最尤法によりトレーニングできる生成モデルを定義します。 尤度を計算するために、残差ブロックのヤコビアン対数決定因子に扱いやすい [tractable] 近似を導入します。 私たちの実験的 [empirical] 評価は、可逆ResNetが最新の画像分類器とフローベースの生成モデルの両方で競争力のあるパフォーマンスを発揮することを示しています。


# ■ イントロダクション（何をしたいか？）

## 1. Introduction

- One of the main appeals of neural network-based models is that a single model architecture can often be used to solve a variety of related tasks. However, many recent advances are based on special-purpose solutions tailored to particular domains. State-of-the-art architectures in unsupervised learning, for instance, are becoming increasingly domain- specific (Van Den Oord et al., 2016b; Kingma & Dhariwal, 2018; Parmar et al., 2018; Karras et al., 2018; Van Den Oord et al., 2016a). On the other hand, one of the most successful feed-forward architectures for discriminative learning are deep residual networks (He et al., 2016; Zagoruyko & Komodakis, 2016), which differ considerably from their generative counterparts. This divide makes it complicated to choose or design a suitable architecture for a given task. It also makes it hard for discriminative tasks to benefit from unsupervised learning. We bridge this gap with a new class of architectures that perform well in both domains.
    - ニューラルネットワークベースのモデルの主な魅力の1つは、単一のモデルアーキテクチャを使用して、さまざまな関連タスクを解決できることです。 ただし、最近の多くの進歩は、特定のドメインに合わせた特別な目的のソリューションに基づいています。 たとえば、教師なし学習の最新のアーキテクチャは、ますますドメイン固有になりつつあります（Van Den Oord et al。、2016b; Kingma＆Dhariwal、2018; Parmar et al。、2018; Karras et al。、2018 ; Van Den Oord et al。、2016a）。 一方、差別的学習で最も成功したフィードフォワードアーキテクチャの1つは、深層残差ネットワーク（He et al。、2016; Zagoruyko＆Komodakis、2016）です。 この分割 [divide] により、特定のタスクに適したアーキテクチャの選択または設計が複雑になります。 また、差別的なタスクが教師なし学習から利益を得ることを難しくします。 このギャップを、両方のドメインで優れたパフォーマンスを発揮する新しいクラスのアーキテクチャで埋めます [bridge]。

---

- To achieve this, we focus on reversible networks which have been shown to produce competitive performance on discriminative (Gomez et al., 2017; Jacobsen et al., 2018) and generative (Dinh et al., 2014; 2017; Kingma & Dhariwal, 2018) tasks independently, albeit in the same model paradigm.
    - これを達成するために、同じモデルの枠組み [paradigm] にも関わらず [albeit]、差別的（Gomez et al。、2017; Jacobsen et al。、2018）およびジェネレーティブ（Dinh et al。、2014; 2017; Kingma＆Dhariwal、 2018）タスクを独立して実行します。
    
- They typically rely on fixed dimension splitting heuristics, but common splittings interleaved with non-volume conserving elements are constraining and their choice has a significant impact on performance (Kingma & Dhariwal, 2018; Dinh et al., 2017). This makes building reversible networks a difficult task. In this work we show that these exotic designs, necessary for competitive density estimation performance, can severely hurt discriminative performance.
    - 通常、固定次元分割ヒューリスティック（＝発見的） [heuristics] に依存しますが、非ボリューム保存要素でインターリーブされた [interleaved] 一般的な分割は制約があり、その選択はパフォーマンスに大きな影響を及ぼします（Kingma＆Dhariwal、2018; Dinh et al。、2017）。 これにより、リバーシブルネットワークの構築が困難なタスクになります。 この研究では、競合密度推定のパフォーマンスに必要なこれらのエキゾチックなデザインが、識別パフォーマンスを著しく損なう可能性があることを示します。

> インターリーブまたはインターリービングは計算機科学と電気通信において、データを何らかの領域で不連続な形で配置し、性能を向上させる技法を指す。 

---

- To overcome this problem, we leverage the viewpoint of ResNets as an Euler discretization of ODEs (Haber & Ruthotto, 2018; Ruthotto & Haber, 2018; Lu et al., 2017; Ciccone et al., 2018) and prove that invertible ResNets (i- ResNets) can be constructed by simply changing the normalization scheme of standard ResNets.
    - この問題を克服するために、ODEのオイラー離散化としてResNetの視点を活用し（Haber＆Ruthotto、2018; Ruthotto＆Haber、2018; Lu et al。、2017; Ciccone et al。、2018）、その可逆ResNet（ i- ResNets）は、標準ResNetの正規化スキームを変更するだけで構築できます。 

- As an intuition, Figure 1 visualizes the differences in the dynamics learned by standard and invertible ResNets.
    - 直観として、図1は、標準および可逆ResNetによって学習されたダイナミクスの違いを視覚化します。


- > Figure 1. Dynamics of a standard residual network (left) and invertible residual network (right). Both networks map the interval [−2,2] to: 1) noisy x3-function at half depth and 2) noisy identity function at full depth. Invertible ResNets describe a bijective continuous dynamics while regular ResNets result in crossing and collapsing paths (circled in white) which correspond to non-bijective continuous dynamics. Due to collapsing paths, standard ResNets are not a valid density model.

---

- This approach allows unconstrained architectures for each residual block, while only requiring a Lipschitz constant smaller than one for each block. We demonstrate that this restriction negligibly impacts performance when building image classifiers - they perform on par with their non- invertible counterparts on classifying MNIST, CIFAR10 and CIFAR100 images.
    - このアプローチにより、各残差ブロックに制約のないアーキテクチャが可能になりますが、各ブロックに必要なリプシッツ定数は1より小さいだけです。 この制限は、画像分類子を構築する際のパフォーマンスにほとんど影響を与えないことを示しています。MNIST、CIFAR10、およびCIFAR100画像の分類では、これらの制限は非可逆の同等物と同等に機能します。

---

- We then show how i-ResNets can be trained as maximum likelihood generative models on unlabeled data. To compute likelihoods, we introduce a tractable approximation to the Jacobian determinant of a residual block. Like FFJORD (Grathwohl et al., 2019), i-ResNet flows have unconstrained (free-form) Jacobians, allowing them to learn more expressive transformations than the triangular mappings used in other reversible models. Our empirical evaluation shows that i-ResNets perform competitively with both state-of-the-art image classifiers and flow-based generative models, bringing general-purpose architectures one step closer to reality.
    - 次に、ラベルなしデータの最尤生成モデルとしてi-ResNetをトレーニングする方法を示します。 尤度を計算するために、残差ブロックのヤコビ行列式に扱いやすい近似を導入します。 FFJORD（Grathwohl et al。、2019）と同様に、i-ResNetフローには制約のない（自由形式の）ヤコビアンがあり、他の可逆モデルで使用される三角マッピングよりも表現力豊かな変換を学習できます。 私たちの経験的評価は、i-ResNetsが最先端の画像分類器とフローベースの生成モデルの両方で競争力を発揮し、汎用アーキテクチャを現実に一歩近づけることを示しています。

# ■ 結論

## 7. Conclusion

- We introduced a new architecture, i-ResNets, which allow free-form layer architectures while still providing tractable density estimates. The unrestricted form of the Jacobian allows expansion and contraction via the residual blocks, while partitioning-based models (Dinh et al., 2014; 2017; Kingma & Dhariwal, 2018) must include affine blocks and scaling layers to be non-volume preserving.
    - 新しいアーキテクチャであるi-ResNetsを導入しました。これにより、自由形式のレイヤアーキテクチャを実現しながら、扱いやすい密度の推定値を提供できます。 ヤコビアンの無制限の形式では、残差ブロックを介した伸縮が可能ですが、パーティションベースのモデル（Dinh et al。、2014; 2017; Kingma＆Dhariwal、2018）には、アフィンブロックとスケーリングレイヤーを含める必要があります。

---

- Several challenges remain to be addressed in future work. First, our estimator of the log-determinant is biased. However, there have been recent advances in building unbiased estimators for the log-determinant (Han et al., 2018), which we believe could improve the performance of our generative model. Second, learning and designing networks with a Lipschitz constraint is challenging. For example, we need to constrain each linear layer in the block instead of being able to directly control the Lipschitz constant of a block, see Anil et al. (2018) for a promising approach for addressing this problem.
    - 将来の作業では、いくつかの課題に対処する必要があります。 まず、対数決定要因の推定量に偏りがあります。 ただし、対数決定要因の偏りのない推定量の構築には最近の進歩があり（Han et al。、2018）、生成モデルのパフォーマンスを改善できると考えられます。 第二に、リプシッツ制約を使用したネットワークの学習と設計は困難です。 たとえば、ブロックのリプシッツ定数を直接制御するのではなく、ブロック内の各線形レイヤーを制約する必要があります。Anilet al。 （2018）この問題に対処するための有望なアプローチ。

# ■ 何をしたか？詳細

## 2. Enforcing Invertibility in ResNets

- There is a remarkable similarity between ResNet architectures and Euler’s method for ODE initial value problems:

> 式

- where xt ∈ Rd represent activations or states, t represents layer indices or time, h > 0 is a step size, and gθt is a residual block. This connection has attracted research at the intersection of deep learning and dynamical systems (Lu et al., 2017; Haber & Ruthotto, 2018; Ruthotto & Haber, 2018; Chen et al., 2018). However, little attention has been paid to the dynamics backwards in time
    - ここで、xt∈Rdは活性化または状態を表し、tはレイヤーインデックスまたは時間を表し、h> 0はステップサイズ、gθtは残差ブロックです。 この接続は、深層学習システムと動的システムの交差点で研究を集めています（Lu et al。、2017; Haber＆Ruthotto、2018; Ruthotto＆Haber、2018; Chen et al。、2018）。 ただし、ダイナミクスには時間的に後方への注意がほとんど払われていません。

> 式

- which amounts to the implicit backward Euler discretization. In particular, solving the dynamics backwards in time would implement an inverse of the corresponding ResNet. The following theorem states that a simple condition suffices to make the dynamics solvable and thus renders the ResNet invertible:
    - これは暗黙の後方オイラー離散化に相当します。 特に、ダイナミクスを時間的に後方に解決すると、対応するResNetの逆が実装されます。 次の定理は、単純な条件でダイナミクスを解くことができるため、ResNetを反転可能にすることを示しています。

---

- While enforcing Lip(g) < 1 makes the ResNet invertible, we have no analytic form of this inverse. However, we can obtain it through a simple fixed-point iteration, see Algorithm 1. Note, that the starting value for the fixed-point iteration can be any vector, because the fixed-point is unique. However, using the output y = x + g(x) as the initialization x0 := y is a good starting point since y was obtained from x only via a bounded perturbation of the identity. From the Banach fixed-point theorem we have
    - Lip（g）<1を強制するとResNetは可逆になりますが、この逆の分析形式はありません。 ただし、単純な不動点反復で取得できます。アルゴリズム1を参照してください。不動点は一意であるため、不動点反復の開始値は任意のベクトルにできることに注意してください。 ただし、初期化として出力y = x + g（x）を使用するx0：= yは、yがアイデンティティの有界摂動を介してのみxから取得されるため、適切な開始点です。 バナッハの不動点定理から、

- Thus, the convergence rate is exponential in the number of iterations n and smaller Lipschitz constants will yield faster convergence.
    - したがって、収束率は反復回数nで指数関数的になり、リプシッツ定数が小さいほど収束が速くなります。

---

- xxx

## 3. Generative Modelling with i-ResNets

- We implement residual blocks as a composition of contractive nonlinearities φ (e.g. ReLU, ELU, tanh) and linear mappings.

- For example, in our convolutional networks g = W3 φ(W2 φ(W1 )), where Wi are convolutional layers. Hence,

# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. Related Work


