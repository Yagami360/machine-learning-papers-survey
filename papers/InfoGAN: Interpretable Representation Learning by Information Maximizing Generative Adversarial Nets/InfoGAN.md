# ■ 論文
- 論文タイトル："InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets"
- 論文リンク：https://arxiv.org/abs/1606.03657
- 論文投稿日付：
- 被引用数（記事作成時点）：xxx 件
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- This paper describes InfoGAN, an information-theoretic extension to the Generative Adversarial Network that is able to learn disentangled representations in a completely unsupervised manner. InfoGAN is a generative adversarial network that also maximizes the mutual information between a small subset of the latent variables and the observation. We derive a lower bound of the mutual information objective that can be optimized efficiently. Specifically, InfoGAN successfully disentangles writing styles from digit shapes on the MNIST dataset, pose from lighting of 3D rendered images, and background digits from the central digit on the SVHN dataset. It also discovers visual concepts that include hair styles, presence/absence of eyeglasses, and emotions on the CelebA face dataset. Experiments show that InfoGAN learns interpretable representations that are competitive with representations learned by existing supervised methods.
    - このペーパーでは、完全に教師なしの方法で解きほぐされた表現を学習できる、生成敵対ネットワークの情報理論的拡張機能であるInfoGANについて説明します。 InfoGANは、潜在変数の小さなサブセットと観測値間の相互情報を最大化する生成的敵対ネットワークです。 効率的に最適化できる相互情報目的の下限を導き出します。 具体的には、InfoGANは、MNISTデータセットの数字図形、3Dレンダリング画像の照明からのポーズ、およびSVHNデータセットの中央数字からの背景数字の書き込みスタイルを解きます。 また、ヘアスタイル、眼鏡の有無、CelebA顔データセットの感情などの視覚的な概念も発見します。 実験は、InfoGANが既存の教師あり学習された表現と競合する解釈可能な表現を学習することを示しています。

# ■ イントロダクション（何をしたいか？）

## 1. Introduction

- Unsupervised learning can be described as the general problem of extracting value from unlabelled data which exists in vast quantities. A popular framework for unsupervised learning is that of representation learning [1, 2], whose goal is to use unlabelled data to learn a representation that exposes important semantic features as easily decodable factors. A method that can learn such representations is likely to exist [2], and to be useful for many downstream tasks which include classification, regression, visualization, and policy learning in reinforcement learning.
    - 教師なし学習は、大量に存在するラベルのないデータから値を抽出する一般的な問題として説明できます。 教師なし学習の一般的なフレームワークは表現学習のフレームワークです[1、2]。その目的は、ラベルなしデータを使用して、重要なセマンティックフィーチャを簡単にデコード可能な要素として公開する表現を学習することです。 このような表現を学習できる方法は存在する可能性が高く[2]、強化学習における分類、回帰、視覚化、ポリシー学習を含む多くの下流 [downstream] タスクに役立つ可能性があります。

---

- While unsupervised learning is ill-posed because the relevant downstream tasks are unknown at training time, a disentangled representation, one which explicitly represents the salient attributes of a data instance, should be helpful for the relevant but unknown tasks.
    - 関連する下流タスクがトレーニング時に未知であるため、教師なし学習は不良設定問題 [ill-posed] ですが、データインスタンスの顕著な属性を明示的に表す解きほぐされた表現は、関連するが未知のタスクに役立つはずです。

- For example, for a dataset of faces, a useful disentangled representation may allocate a separate set of dimensions for each of the following attributes: facial expression, eye color, hairstyle, presence or absence of eyeglasses, and the identity of the corresponding person.
    - たとえば、顔のデータセットの場合、有用な解きほぐされた表現は、顔の表情、目の色、髪型、眼鏡の有無、および対応する人物の識別情報の各属性に個別のディメンションセットを割り当てることができます。

- A disentangled representation can be useful for natural tasks that require knowledge of the salient attributes of the data, which include tasks like face recognition and object recognition. It is not the case for unnatural supervised tasks, where the goal could be, for example, to determine whether the number of red pixels in an image is even or odd. Thus, to be useful, an unsupervised learning algorithm must in effect correctly guess the likely set of downstream classification tasks without being directly exposed to them.
    - 解きほぐされた表現は、顔認識やオブジェクト認識などのタスクを含む、データの顕著な属性の知識を必要とする自然なタスクに役立ちます。不自然な教師ありタスクの場合はそうではありません。たとえば、画像内の赤いピクセルの数が偶数か奇数かを判断することが目標です。したがって、有用であるために、教師なし学習アルゴリズムは、実際には、下流分類タスクの可能性のあるセットを、それらに直接さらされることなく正しく推測する必要があります。

---

- A significant fraction of unsupervised learning research is driven by generative modelling. It is motivated by the belief that the ability to synthesize, or “create” the observed data entails some form of understanding, and it is hoped that a good generative model will automatically learn a disentangled representation, even though it is easy to construct perfect generative models with arbitrarily bad representations. The most prominent generative models are the variational autoencoder (VAE) [3] and the generative adversarial network (GAN) [4].
    - 教師なし学習研究のかなりの部分は、生成的モデリングによって推進されています。 観察されたデータを合成または「作成」する能力には何らかの形の理解が必要であるという信念に動機付けられており、任意の悪い表現をもつ完璧な生成モデルを構築するのは簡単ですが、優れた生成モデルが解きほぐされた表現を自動的に学習することが望まれます。最も顕著な生成モデルは、変分オートエンコーダー（VAE）[3]および生成的敵対ネットワーク（GAN）[4]です。

---

- In this paper, we present a simple modification to the generative adversarial network objective that encourages it to learn interpretable and meaningful representations. We do so by maximizing the mutual information between a fixed small subset of the GAN’s noise variables and the observations, which turns out to be relatively straightforward. Despite its simplicity, we found our method to be surprisingly effective: it was able to discover highly semantic and meaningful hidden representations on a number of image datasets: digits (MNIST), faces (CelebA), and house numbers (SVHN). The quality of our unsupervised disentangled representation matches previous works that made use of supervised label information [5–9]. These results suggest that generative modelling augmented with a mutual information cost could be a fruitful approach for learning disentangled representations
    - この論文では、生成的敵対的ネットワーク目標に対する単純な修正を提示し、解釈可能で意味のある表現を学習することを奨励します。 そのためには、GANのノイズ変数の固定された小さなサブセットと観測値の間の相互情報を最大化します。これは、比較的簡単であることが判明しました。 そのシンプルさにもかかわらず、私たちの方法は驚くほど効果的であることがわかりました：数字（MNIST）、顔（CelebA）、および家番号（SVHN）の多くの画像データセットで非常に意味的で意味のある隠された表現を発見することができました。 教師なしの解きほぐされた表現の品質は、教師付きラベル情報を使用した以前の作品と一致します[5–9]。 これらの結果は、相互情報コストで増強された生成モデリングが、解きほぐされた表現を学習するための実り多いアプローチである可能性を示唆しています。

---

- In the remainder of the paper, we begin with a review of the related work, noting the supervision that is required by previous methods that learn disentangled representations. Then we review GANs, which is the basis of InfoGAN. We describe how maximizing mutual information results in interpretable representations and derive a simple and efficient algorithm for doing so. Finally, in the experiments section, we first compare InfoGAN with prior approaches on relatively clean datasets and then show that InfoGAN can learn interpretable representations on complex datasets where no previous unsupervised approach is known to learn representations of comparable quality.
    - ペーパーの残りの部分では、関連する作業のレビューから始め、解きほぐされた表現を学習する以前の方法で必要とされる監督に注目します。 次に、InfoGANの基礎であるGANを確認します。 相互情報を最大化すると解釈可能な表現がどのように得られるかを説明し、そのためのシンプルで効率的なアルゴリズムを導き出します。 最後に、実験セクションでは、まずInfoGANを比較的クリーンなデータセットの以前のアプローチと比較してから、InfoGANが、同等の品質の表現を学習する以前の教師なしアプローチが知られていない複雑なデータセットの解釈可能な表現を学習できることを示します。

# ■ 結論

## x. Conclusion


# ■ 何をしたか？詳細

## 4 Mutual Information for Inducing Latent Codes

- The GAN formulation uses a simple factored continuous input noise vector z, while imposing no restrictions on the manner in which the generator may use this noise. As a result, it is possible that the noise will be used by the generator in a highly entangled way, causing the individual dimensions of z to not correspond to semantic features of the data.
    - GAN定式化では、単純な因数分解された連続入力ノイズベクトルzを使用しますが、ジェネレーターがこのノイズを使用する方法に制限はありません。 その結果、ジェネレーターによってノイズが非常に絡み合って使用され、zの個々の次元がデータのセマンティック機能に対応しなくなる可能性があります。

---

- However, many domains naturally decompose into a set of semantically meaningful factors of variation. For instance, when generating images from the MNIST dataset, it would be ideal if the model automatically chose to allocate a discrete random variable to represent the numerical identity of the digit (0-9), and chose to have two additional continuous variables that represent the digit’s angle and thickness of the digit’s stroke.
    - ただし、多くのドメインは当然、意味的に意味のある変動要因のセットに分解されます。 たとえば、MNISTデータセットから画像を生成する場合、モデルが自動的に数字のID（0-9）を表す離散確率変数を割り当てることを選択し、桁の角度と桁のストロークの太さという、2つの追加の連続変数を選択することが理想的です。 

- It is the case that these attributes are both independent and salient, and it would be useful if we could recover these concepts without any supervision, by simply specifying that an MNIST digit is generated by an independent 1-of-10 variable and two independent continuous variables.
    - MNIST数字が独立した1-of-10変数と2つの独立した連続変数によって生成されることを指定するだけにより、教師なしでこれらの概念を回復できれば便利です。
    - これらの属性は独立しており、顕著なものである場合

---

- In this paper, rather than using a single unstructured noise vector, we propose to decompose the input noise vector into two parts: (i) z, which is treated as source of incompressible noise; (ii) c, which we will call the latent code and will target the salient structured semantic features of the data distribution.
    - この論文では、単一の非構造化ノイズベクトルを使用するのではなく、入力ノイズベクトルを2つの部分に分解することを提案します。（i）z、非圧縮性ノイズのソースとして扱われます。 （ii）c。潜在コードと呼ばれ、データ分布の顕著な構造化セマンティック機能を対象とします。

---

- Mathematically, we denote the set of structured latent variables by c1, c2,􏰗. . . , cL. In its simplest form, we may assume a factored distribution, given by P (c1, c2, . . . , cL) = Li=1 P (ci). For ease of notation, we will use latent codes c to denote the concatenation of all latent variables ci.
    - 数学的には、構造化された潜在変数のセットをc1、c2、􏰗で表します。 。 。 、cL。 最も単純な形式では、P（c1、c2、。。。、cL）= Li = 1 P（ci）で与えられる因数分解分布を仮定できます。 表記を簡単にするために、潜在コードcを使用して、すべての潜在変数ciの連結を示します。

---

- We now propose a method for discovering these latent factors in an unsupervised way: we provide the generator network with both the incompressible noise z and the latent code c, so the form of the generator becomes G(z, c). However, in standard GAN, the generator is free to ignore the additional latent code c by finding a solution satisfying PG(x|c) = PG(x). To cope with the problem of trivial codes, we propose an information-theoretic regularization: there should be high mutual information between latent codes c and generator distribution G(z, c). Thus I(c; G(z, c)) should be high.
    - ここで、これらの潜在因子を教師なしの方法で発見する方法を提案します。非圧縮性ノイズzと潜在コードcの両方をジェネレーターネットワークに提供するため、ジェネレーターの形式はG（z、c）になります。 ただし、標準のGANでは、PG（x | c）= PG（x）を満たす解を見つけることにより、ジェネレーターは追加の潜在コードcを自由に無視できます。 些細なコードの問題に対処するために、情報理論的な正則化を提案します。潜在コードcとジェネレーター分布G（z、c）の間には高い相互情報が必要です。 したがって、I（c; G（z、c））は高くなければなりません。

---

- In information theory, mutual information between X and Y , I(X; Y ), measures the “amount of information” learned from knowledge of random variable Y about the other random variable X. The mutual information can be expressed as the difference of two entropy terms:
    - 情報理論では、XとYの間の相互情報、I（X; Y）は、他のランダム変数Xに関するランダム変数Yの知識から学習した「情報量」を測定します。相互情報は、2つのエントロピー項の差として表現できます。

> 式

- This definition has an intuitive interpretation: I (X ; Y ) is the reduction of uncertainty in X when Y is observed. If X and Y are independent, then I (X ; Y ) = 0, because knowing one variable reveals nothing about the other; by contrast, if X and Y are related by a deterministic, invertible function, then maximal mutual information is attained. This interpretation makes it easy to formulate a cost: given any x ∼ PG(x), we want PG(c|x) to have a small entropy. In other words, the information in the latent code c should not be lost in the generation process. Similar mutual information inspired objectives have been considered before in the context of clustering [26–28]. Therefore, we propose to solve the following information-regularized minimax game:
    - この定義には直感的な解釈があります。I（X; Y）は、Yが観測されたときのXの不確実性の減少です。 XとYが独立している場合、I（X; Y）= 0になります。1つの変数を知っていても、他の変数については何も明らかにされないためです 対照的に、XとYが決定論的な可逆関数によって関連付けられている場合、最大の相互情報が得られます。 この解釈により、コストの定式化が容易になります。任意のx〜PG（x）が与えられると、PG（c | x）に小さなエントロピーを持たせることができます。 つまり、潜在コードcの情報は、生成プロセスで失われるべきではありません。 同様の相互情報にヒントを得た目標は、クラスタリングのコンテキストで以前に検討されてきました[26–28]。 したがって、次の情報規制されたミニマックスゲームを解決することを提案します。

## 5 Variational Mutual Information Maximization

- In practice, the mutual information term I(c;G(z,c)) is hard to maximize directly as it requires access to the posterior P (c|x). Fortunately we can obtain a lower bound of it by defining an auxiliary distribution Q(c|x) to approximate P (c|x):
    - 実際には、相互情報項I（c; G（z、c））は事後P（c | x）へのアクセスを必要とするため、直接最大化することは困難です。 幸いなことに、補助分布Q（c | x）を定義してP（c | x）を近似することにより、その下限を取得できます。

---

- This technique of lower bounding mutual information is known as Variational Information Maximization [29]. We note in addition that the entropy of latent codes H(c) can be optimized over as well since for common distributions it has a simple analytical form. However, in this paper we opt for simplicity by fixing the latent code distribution and we will treat H(c) as a constant. So far we have bypassed the problem of having to compute the posterior P (c|x) explicitly via this lower bound but we still need to be able to sample from the posterior in the inner expectation. Next we state a simple lemma, with its proof deferred to Appendix, that removes the need to sample from the posterior.
    - 相互情報を下限とするこの手法は、変分情報最大化[29]として知られています。 さらに、潜在コードH（c）のエントロピーも最適化できることに注意してください。これは、一般的な分布では単純な分析形式を持っているからです。 ただし、このホワイトペーパーでは、潜在コードの分布を修正することで単純化を選択し、H（c）を定数として扱います。 これまで、この下限を介して明示的に事後P（c | x）を計算する必要があるという問題を回避しましたが、内部期待値で事後からサンプリングできる必要があります。 次に、その証拠を付録に委ねた単純な補題を述べ、事後からサンプリングする必要性を取り除きます。


---

- We note that LI (G, Q) is easy to approximate with Monte Carlo simulation. In particular, LI can be maximized w.r.t. Q directly and w.r.t. G via the reparametrization trick. Hence LI (G, Q) can be added to GAN’s objectives with no change to GAN’s training procedure and we call the resulting algorithm Information Maximizing Generative Adversarial Networks (InfoGAN).
    - LI（G、Q）は、モンテカルロシミュレーションで簡単に近似できることに注意してください。 特に、LIは再パラメーター化トリックを介してQおよびGに対して直接最大化できます。 したがって、LI（G、Q）をGANのトレーニング手順に変更を加えることなくGANの目標に追加でき、生成されたアルゴリズムを情報最大化敵対ネットワーク（InfoGAN）と呼びます。

- Eq (4) shows that the lower bound becomes tight as the auxiliary distribution Q approaches the true posterior distribution: Ex[DKL(P(·|x) ∥ Q(·|x))] → 0. In addition, we know that when the variational lower bound attains its maximum LI (G, Q) = H (c) for discrete latent codes, the bound becomes tight and the maximal mutual information is achieved. In Appendix, we note how InfoGAN can be connected to the Wake-Sleep algorithm [30] to provide an alternative interpretation.
    - 式（4）は、補助分布Qが真の事後分布に近づくにつれて下限が厳しくなることを示しています。Ex[DKL（P（・| x）∥Q（・| x））]→0。 離散的な潜在コードについて、変分下限が最大LI（G、Q）= H（c）に達すると、境界が厳しくなり、最大の相互情報が得られます。 付録では、代替解釈を提供するために、InfoGANをWake-Sleepアルゴリズム[30]に接続する方法に注意します。

# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目


# ■ 関連研究（他の手法との違い）

## x. Related Work


