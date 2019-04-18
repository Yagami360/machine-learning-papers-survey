# ■ 論文
- 論文リンク：https://arxiv.org/abs/1701.07875
- 論文投稿日付：2017/01/26 (v1), 2017/03/09 (v2), 2017/12/06 (v3)
- 著者：Martin Arjovsky, Soumith Chintala, Léon Bottou
- categories：

# ■ 概要（何をしたか？）

- We introduce a new algorithm named WGAN, an alternative to traditional GAN training.
    - 伝統的な GAN の学習の代わりとなるような、WGAN と名付けた新しいアルゴリズムを紹介する。

- In this new model, we show that we can improve the stability of learning, get rid of problems like mode collapse, and provide meaningful learning curves useful for debugging and hyperparameter searches.
    - この新しいモデルでは、モード崩壊のような問題を除外して、学習の安定化を改善することが出来る。
    - そして、デバッグやハイパーパラメータの検索に便利な意味のある学習曲線を提供する。

- Furthermore, we show that the corresponding optimization problem is sound, and provide extensive theoretical work highlighting the deep connections to other distances between distributions.
    - 更には、一致する（＝対応する）最適化問題が妥当である [sound] ことを示す。
    - そして、確率分布間の他の距離を深いつながりを強調する [highlighting] ような、広範囲の [extensive] 理論的な研究を提供する。


# ■ イントロダクション（何をしたいか？）

## 1. Introduction

- The problem this paper is concerned with is that of unsupervised learning.
    - この論文の問題は、教師なし学習に関連したものである。

- Mainly, what does it mean to learn a probability distribution?
    - 主に、確率分布を学習することの意味は何なのか？ということである。

- The classical answer to this is to learn a probability density. 
    - この問題に対する古典的な回答は、確率分布を学習することである。

- This is often done by dening a parametric family of densities ![image](https://user-images.githubusercontent.com/25688193/56338664-e04c1b80-61e5-11e9-81cf-026988aa2472.png) and finding the one that maximized the likelihood on our data: if we have real data examples ![image](https://user-images.githubusercontent.com/25688193/56338680-fce85380-61e5-11e9-9fe9-50d47f491c66.png), we would solve the problem<br>
    - これは、しばしば、確率密度 ![image](https://user-images.githubusercontent.com/25688193/56338664-e04c1b80-61e5-11e9-81cf-026988aa2472.png) のパラメリックな族を定義し[deny]、我々のデータの尤度を最大化するものを見つけることによって行われる。
    - 即ち、本物のデータ例 ![image](https://user-images.githubusercontent.com/25688193/56338680-fce85380-61e5-11e9-9fe9-50d47f491c66.png) が存在するとすると、（以下の）問題を解くだろう。

![image](https://user-images.githubusercontent.com/25688193/56338702-11c4e700-61e6-11e9-8ccd-2d64603a63fc.png)<br>

- If the real data distribution ![image](https://user-images.githubusercontent.com/25688193/56339104-ba277b00-61e7-11e9-9454-a661bfd50b19.png) admits a density and ![image](https://user-images.githubusercontent.com/25688193/56339217-3cb03a80-61e8-11e9-9bff-a7ace9111bd7.png) is the distribution of the parametrized density ![image](https://user-images.githubusercontent.com/25688193/56339255-62d5da80-61e8-11e9-9000-67762c55123d.png), then, asymptotically, this amounts to minimizing the Kullback-Leibler divergence ![image](https://user-images.githubusercontent.com/25688193/56339053-8cdacd00-61e7-11e9-990d-02114b588b03.png).
    - もし真のデータ分布 ![image](https://user-images.githubusercontent.com/25688193/56339104-ba277b00-61e7-11e9-9454-a661bfd50b19.png) が、確率密度であることを許可し、（＝確率密度関数で表現できることを前提とし）
    - そして、![image](https://user-images.githubusercontent.com/25688193/56339217-3cb03a80-61e8-11e9-9bff-a7ace9111bd7.png) は、パラメーター化された ![image](https://user-images.githubusercontent.com/25688193/56339255-62d5da80-61e8-11e9-9000-67762c55123d.png) の分布とするならば、
    - そのとき、漸近的に [asymptotically]、この量は、KLダイバージェンス ![image](https://user-images.githubusercontent.com/25688193/56339053-8cdacd00-61e7-11e9-990d-02114b588b03.png) を最小化する。

- For this to make sense, we need the model density ![image](https://user-images.githubusercontent.com/25688193/56339255-62d5da80-61e8-11e9-9000-67762c55123d.png) to exist. 
    - このことが意味をなすのに、モデルの分布 ![image](https://user-images.githubusercontent.com/25688193/56339255-62d5da80-61e8-11e9-9000-67762c55123d.png) が存在している必要がある。

- This is not the case in the rather common situation where we are dealing with distributions supported by low dimensional manifolds. 
    - このことは、低次元の多様体 [manifolds] によってサポートされている（＝台にされている？）分布を扱うような、かなり [rather] 一般的な [common] シチュエーションでは、当てはまらない。[not the case]

- It is then unlikely that the model manifold and the true distribution's support have a non-negligible intersection (see [1]), and this means that the KL distance is not dened (or simply infinite).
    - そのことは、モデルの多様体と真の分布の台が、（無視できないほど）僅かでない [non-negligible] 交差点 [intersection] を持っているということが、ありそうもない。[be unlikely]
    - そして、このことは、KLダイバージェンスが定義されていない（或いは、単に有限である）ことを意味している。

- The typical remedy is to add a noise term to the model distribution.
    - 典型的な [typical] な改善法 [remedy] は、モデルの分布にノイズ項を加えることである。

- This is why virtually all generative models described in the classical machine learning literature include a noise component.
    - このような理由で [this is why]、古典的な機械学習の文献 [literature] に記述されている事実上 [virtually] 全ての生成モデルが、ノイズコンポーネントを含んでいる。

- In the simplest case, one assumes a Gaussian noise with relatively high bandwidth in order to cover all the examples.
    - 最も簡単なケースでは、全ての例をカバーするために、比較的高い帯域幅 [bandwidth] のガウスノイズを仮定する [assume]

- It is well known, for instance, that in the case of image generation models, this noise degrades the quality of the samples and makes them blurry.
    - これは、例えば、画像生成モデルの場合において、
    - このノイズが、サンプルのクオリティを低下させ [degrades]、それらぼやけさせる [blurry]
    - ということがよく知られている。

- For example, we can see in the recent paper [23] that the optimal standard deviation of the noise added to the model when maximizing likelihood is around 0.1 to each pixel in a generated image, when the pixels were already normalized to be in the range [0; 1].
    - 例えば、最近の論文 [23] で見ることが出来る。
    - （この論文というのは、）
    - 尤度を最大化するときに、
    - モデルに加えたノイズの標準偏差 [standard deviation] の最適値が、
    - ピクセルが、[0:1] の値の範囲で正規化されている場合において、
    - 生成画像における各ピクセルに対して、0.1 付近である（という論文）

- This is a very high amount of noise, so much that when papers report the samples of their models, they don't add the noise term on which they report likelihood numbers.
    - これは、ノイズの量がとても多いので、
    - 論文が彼らのモデルを報告するときに、尤度の数を報告するノイズ項を加えない。

- In other words, the added noise term is clearly incorrect for the problem, but is needed to make the maximum likelihood approach work.
    - 言い換えれば、加えられたノイズ項は、問題に対して明らかなに正確ではない。
    - しかし、最尤度推定のアプローチを動作させるために必要となる。

- Rather than estimating the density of ![image](https://user-images.githubusercontent.com/25688193/56339104-ba277b00-61e7-11e9-9454-a661bfd50b19.png) which may not exist, we can define a random variable Z with a fixed distribution p(z) and pass it through a parametric function ![image](https://user-images.githubusercontent.com/25688193/56343805-b94a1580-61f6-11e9-8a61-4834acf729fb.png) (typically a neural network of some kind) that directly generates samples following a certain distribution ![image](https://user-images.githubusercontent.com/25688193/56339217-3cb03a80-61e8-11e9-9bff-a7ace9111bd7.png).
    - 存在しないかもしれない（真の分布）![image](https://user-images.githubusercontent.com/25688193/56339104-ba277b00-61e7-11e9-9454-a661bfd50b19.png) の確率密度関数を推定することよりも、
    - 固定された分布 p(z) でのランダム変数 Z を定義することができる。
    - そして、その（ランダム変数 Z を、）パラメリック関数（通常は [typically] ある種のニューラルネット） ![image](https://user-images.githubusercontent.com/25688193/56343805-b94a1580-61f6-11e9-8a61-4834acf729fb.png) に通す。
    - （この関数というのは、）特定の [certain] 分布 ![image](https://user-images.githubusercontent.com/25688193/56339217-3cb03a80-61e8-11e9-9bff-a7ace9111bd7.png) に従うサンプルを、直接的に生成するような関数

- By varying θ, we can change this distribution and make it close to the real data distribution ![image](https://user-images.githubusercontent.com/25688193/56339104-ba277b00-61e7-11e9-9454-a661bfd50b19.png).
    - θ をベリファイすることによって、この分布（![image](https://user-images.githubusercontent.com/25688193/56339217-3cb03a80-61e8-11e9-9bff-a7ace9111bd7.png) の形状を）変えることができ、真の分布 ![image](https://user-images.githubusercontent.com/25688193/56339104-ba277b00-61e7-11e9-9454-a661bfd50b19.png) に近づけることが出来る。

- This is useful in two ways.
    - これは、２つの方法で便利である。

- First of all, unlike densities, this approach can represent distributions conned to a low dimensional manifold.
    - まず第１に、確率密度分布とは異なり、このアプローチは、低次元多様体に結合された分布を表現することが出来る。

- Second, the ability to easily generate samples is often more useful than knowing the numerical value of the density (for example in image superresolution or semantic segmentation when considering the conditional distribution of the output image given the input image).
    - 第２に、サンプルを簡単に生成する能力は、たびたび、確率密度の数値を知ることよりも便利である。
    - （例えば、入力画像が与えられた出力画像の条件付き確率分布を考慮するときの、画像の超解像度やセマンティックセグメイションにおいて、）

- In general, it is computationally difficult to generate samples given an arbitrary high dimensional density [16].
    - 一般的に、任意の [arbitrary] 高次元確率密度で与えられたサンプルを生成することは、計算的に [computationally] 困難である。[16]

- Variational Auto-Encoders (VAEs) [9] and Generative Adversarial Networks (GANs) [4] are well known examples of this approach. 
    - Variational Auto-Encoders (VAEs) [9] と Generative Adversarial Networks (GANs) [4] は、このアプローチのよく知られた例である。

- Because VAEs focus on the approximate likelihood of the examples, they share the limitation of the standard models and need to fiddle with additional noise terms.
    - VAE は、サンプル例の尤度の近似 [approximate] にフォーカスしているので、
    - （VAE）は標準モデルの制限 [limitation] を共有しており、加えたノイズ項をいじくり回す [fiddle with] 必要がある。

- GANs offer much more flexibility in the denition of the objective function, including Jensen-Shannon [4], and all f-divergences [17] as well as some exotic combinations [6].
    - GANs は、JSダイバージェンス [4] やf-ダイバージェンス [17] 、（及びそれらの）いくつかのエキゾチックな組み合わせ [6] を含むような、目的関数の定義において、はるかな柔軟性を提供する。

- On the other hand, training GANs is well known for being delicate and unstable, for reasons theoretically investigated in [1].
    - 一方で [On the other hand]、GANの学習は、[1] の理論的な研究での理由で、デリケートで不安定になることが知られている。

- In this paper, we direct our attention on the various ways to measure how close the model distribution and the real distribution are, or equivalently, on the various ways to define a distance or divergence ![image](https://user-images.githubusercontent.com/25688193/56346613-7e97ab80-61fd-11e9-801e-7f55c2091755.png).
    - この論文では、どのようにして、モデルの分布と真の分布を近づけるか？或いは、等価とすると？ということを測定する様々な方法について、注意を向ける。
    - 距離を定義したり、ダイバージェンス ![image](https://user-images.githubusercontent.com/25688193/56346613-7e97ab80-61fd-11e9-801e-7f55c2091755.png) を定義したりするような様々な方法で、

- The most fundamental difference between such distances is their impact on the convergence of sequences of probability distributions.
    - このような距離の最も基本的な違いは、確率分布の系列の収束性に関するインパクトである。

- A sequence of distributions ![image](https://user-images.githubusercontent.com/25688193/56346946-2f9e4600-61fe-11e9-840e-a67c017096ed.png) converges if and only if there is a distribution ![image](https://user-images.githubusercontent.com/25688193/56347051-75f3a500-61fe-11e9-84f2-b8de5d09f4de.png) such that ![image](https://user-images.githubusercontent.com/25688193/56347141-a8050700-61fe-11e9-8678-dfba9afca745.png) tends to zero, something that depends on how exactly the distance ρ is defined. 
    - 確率分布の系列 ![image](https://user-images.githubusercontent.com/25688193/56346946-2f9e4600-61fe-11e9-840e-a67c017096ed.png) は、![image](https://user-images.githubusercontent.com/25688193/56347141-a8050700-61fe-11e9-8678-dfba9afca745.png) が０になる傾向のような分布 ![image](https://user-images.githubusercontent.com/25688193/56347051-75f3a500-61fe-11e9-84f2-b8de5d09f4de.png) が存在するときのみ、収束する。
    - 距離 ρ が、どの程度正確に定義されているかに依存したものである。

    > [A if and only if B : 条件 B のときに限り A が成り立つ。]

- Informally, a distance ρ induces a weaker topology when it makes it easier for a sequence of distribution to converge. 
    - 非公式には、[Informally]、距離 ρ は、弱いトポロジー（位相幾何）を誘導する。[induce]
    - それ（距離ρ）が、一連の分布の収束を簡単にするときに、

- Section 2 clarifies how popular probability distances differ in that respect.
    - セクション２では、その側面で、一般的な確率分布の距離がどのように異なるかを明らかにする [clarify]。

- In order to optimize the parameter θ, it is of course desirable to define our model distribution ![image](https://user-images.githubusercontent.com/25688193/56339217-3cb03a80-61e8-11e9-9bff-a7ace9111bd7.png) in a manner that makes the mapping ![image](https://user-images.githubusercontent.com/25688193/56349416-7a6e8c80-6203-11e9-83de-bb405e48d48a.png) continuous.
    - パラメーター θ を最適化するために、写像 ![image](https://user-images.githubusercontent.com/25688193/56349416-7a6e8c80-6203-11e9-83de-bb405e48d48a.png) を連続にするような方法で [manner]、モデルの分布 ![image](https://user-images.githubusercontent.com/25688193/56339217-3cb03a80-61e8-11e9-9bff-a7ace9111bd7.png) を定義することが、もちろん望ましい。

- Continuity means that when a sequence of parameters θt converges to θ, the distributions ![image](https://user-images.githubusercontent.com/25688193/56349793-48a9f580-6204-11e9-8a91-b417a5e717bb.png) also converge to ![image](https://user-images.githubusercontent.com/25688193/56339217-3cb03a80-61e8-11e9-9bff-a7ace9111bd7.png).
    - （この写像の）連続性とは、パラメーターの系列 θ_t が θ に収束するとき、分布の（系列）![image](https://user-images.githubusercontent.com/25688193/56349793-48a9f580-6204-11e9-8a91-b417a5e717bb.png) も、![image](https://user-images.githubusercontent.com/25688193/56339217-3cb03a80-61e8-11e9-9bff-a7ace9111bd7.png) に収束することを意味している。

    > つまりは、リプシッツ連続のこと

- However, it is essential to remember that the notion of the convergence of the distributions ![image](https://user-images.githubusercontent.com/25688193/56349793-48a9f580-6204-11e9-8a91-b417a5e717bb.png) depends on the way we compute the distance between distributions.
    - しかしながら、分布 ![image](https://user-images.githubusercontent.com/25688193/56349793-48a9f580-6204-11e9-8a91-b417a5e717bb.png) の収束の概念 [notion] は、分布間の距離の計算の仕方によって異なることを、覚えておくことが必要不可欠である。

- The weaker this distance, the easier it is to define a continuous mapping from θ-space to ![image](https://user-images.githubusercontent.com/25688193/56339217-3cb03a80-61e8-11e9-9bff-a7ace9111bd7.png)-space, since it's easier for the distributions to converge.
    - この距離が弱いほど、確率分布が収束しやすくなるので、θの空間から、![image](https://user-images.githubusercontent.com/25688193/56339217-3cb03a80-61e8-11e9-9bff-a7ace9111bd7.png) の空間へのリプシッツ連続な写像が定義しやすくなる。

    > 距離が弱いとは、位相的な意味での弱さ？

- The main reason we care about the mapping ![image](https://user-images.githubusercontent.com/25688193/56349416-7a6e8c80-6203-11e9-83de-bb405e48d48a.png) to be continuous is as follows.
    - 写像 ![image](https://user-images.githubusercontent.com/25688193/56349416-7a6e8c80-6203-11e9-83de-bb405e48d48a.png) が、リプシッツ連続であることを気にする理由は、以下の通りである。

- If ρ is our notion of distance between two distributions, we would like to have a loss function ![image](https://user-images.githubusercontent.com/25688193/56350773-8019a180-6206-11e9-9bf8-110790c9c74c.png) that is continuous, and this is equivalent to having the mapping ![image](https://user-images.githubusercontent.com/25688193/56349416-7a6e8c80-6203-11e9-83de-bb405e48d48a.png) be continuous when using the distance between distributions ρ.
    - もし ρ が、２つの分布の間の距離の概念 [notion] になっていれば、連続な損失関数 ![image](https://user-images.githubusercontent.com/25688193/56350773-8019a180-6206-11e9-9bf8-110790c9c74c.png) を持つことが望ましい。
    - そして、これ（＝この損失関数）は、分布間の距離 ρ を使うときに、リプシッツ連続な写像 ![image](https://user-images.githubusercontent.com/25688193/56349416-7a6e8c80-6203-11e9-83de-bb405e48d48a.png) を持つことと等価である。

- The contributions of this paper are:

- In Section 2, we provide a comprehensive theoretical analysis of how the Earth Mover (EM) distance behaves in comparison to popular probability distances and divergences used in the context of learning distributions.
    - セクション２では、学習の分布の文脈で使われている一般的な距離やダイバージェンスの比較において、the Earth Mover (EM) 距離がどのように振る舞うのかの包括的な [comprehensive] 理論的解析を提供する。

- In Section 3, we define a form of GAN called Wasserstein-GAN that minimizes a reasonable and efficient approximation of the EM distance, and we theoretically show that the corresponding optimization problem is sound.
    - セクション３では、合理的 [reasonable] で効率的なEM距離の近似を最小化する Wasserstein-GAN と呼ばれる GAN の型を定義する。
    - そして、一致する（＝対応する）最適化問題が妥当である [sound] ことを理論的に紹介する。

- In Section 4, we empirically show that WGANs cure the main training problems of GANs.
    - In particular, training WGANs does not require maintaining a careful balance in training of the discriminator and the generator, and does not require a careful design of the network architecture either.
    - The mode dropping phenomenon that is typical in GANs is also drastically reduced.
    - One of the most compelling practical benets of WGANs is the ability to continuously estimate the EM distance by training the discriminator to optimality.
    - Plotting these learning curves is not only useful for debugging and hyperparameter searches, but also correlate remarkably well with the observed sample quality.

- セクション４では、WGANs が GAN の学習の問題の解決策 [cure] になっていることを経験的に [empirically] 示す。
    - 特に、WGAN を学習することは、識別器と生成器の学習において、注意深いバランスのとり方を維持することを要求しない。そして、（識別器と生成器の）両方のネットワークアーキテクチャの注意深い設計を要求しない。
    - 典型的な [typical] GAN で発生するモード崩壊の現象が、劇的に減少する。
    - WGAN の最も説得力のある [compelling] 実用的な利点の１つが、最適化のための識別器の学習によって、EM距離を連続的に推定する機能である。
    - それらの学習曲線のプロットでは、デバッグやハイパーパラメータの検索に便利なだけでなく、観測されたサンプルのクオリティと、著しく [remarkably] よく相関する。


# ■ 結論

## 6. Conclusion

- We introduced an algorithm that we deemed WGAN, an alternative to traditional GAN training.
    - 伝統的な GAN の代わりとなるような、WGAN とみなす [deem] アルゴリズムを紹介した。

- In this new model, we showed that we can improve the stability of learning, get rid of problems like mode collapse, and provide meaningful learning curves useful for debugging and hyperparameter searches.
    - この新しいモデルでは、モード崩壊のような問題を除外し、学習の安定性を改善することが出来る。
    - そして、デバッグやハイパーパラメータの検索に便利な意味のある学習曲線を提供する。

- Furthermore, we show that the corresponding optimization problem is sound, and provide extensive theoretical work highlighting the deep connections to other distances between distributions.
    - 更には、一致する（＝対応する）最適化問題が妥当である [sound] ことを示す。
    - そして、確率分布間の他の距離を深いつながりを強調する [highlighting] ような、広範囲の [extensive] 理論的な研究を提供する。


# ■ 何をしたか？詳細

## 2. Diffierent Distances

- We now introduce our notation.
    - 表記法 [notation] を紹介する。

- Let ![image](https://user-images.githubusercontent.com/25688193/56361196-b1539b00-6221-11e9-8ac0-558b8bfdc3d6.png) be a compact metric set (such as the space of images ![image](https://user-images.githubusercontent.com/25688193/56361272-e65fed80-6221-11e9-897e-3b357b4b1ae4.png)) and let Σ denote the set of all the Borel subsets of ![image](https://user-images.githubusercontent.com/25688193/56361196-b1539b00-6221-11e9-8ac0-558b8bfdc3d6.png).
    - ![image](https://user-images.githubusercontent.com/25688193/56361196-b1539b00-6221-11e9-8ac0-558b8bfdc3d6.png) を、（画像の空間 ![image](https://user-images.githubusercontent.com/25688193/56361272-e65fed80-6221-11e9-897e-3b357b4b1ae4.png) のような）コンパクト測度？の集合とし、[Let]
    - Σ は、![image](https://user-images.githubusercontent.com/25688193/56361196-b1539b00-6221-11e9-8ac0-558b8bfdc3d6.png) の全てのボレル部分集合？と意味する。[denote]

> ![image](https://user-images.githubusercontent.com/25688193/56361196-b1539b00-6221-11e9-8ac0-558b8bfdc3d6.png) は、σ加法族。（特に、画像の空間を要素とするようなσ-加法族は、ボレル加法族になる。）<br>
> Σ は、この σ-加法族（＝ボレル集合族になる）![image](https://user-images.githubusercontent.com/25688193/56361196-b1539b00-6221-11e9-8ac0-558b8bfdc3d6.png) の要素であるボレル集合<br>

- Let ![image](https://user-images.githubusercontent.com/25688193/56361987-b6194e80-6223-11e9-8a29-56d8d77ef627.png) denote the space of probability measures defined on ![image](https://user-images.githubusercontent.com/25688193/56361196-b1539b00-6221-11e9-8ac0-558b8bfdc3d6.png). 
    - ![image](https://user-images.githubusercontent.com/25688193/56361987-b6194e80-6223-11e9-8a29-56d8d77ef627.png) は、測度の集合 ![image](https://user-images.githubusercontent.com/25688193/56361196-b1539b00-6221-11e9-8ac0-558b8bfdc3d6.png) ？上で定義された、確率測度空間を意味している。

> ![image](https://user-images.githubusercontent.com/25688193/56361987-b6194e80-6223-11e9-8a29-56d8d77ef627.png) は、ボレル加法族 ![image](https://user-images.githubusercontent.com/25688193/56361196-b1539b00-6221-11e9-8ac0-558b8bfdc3d6.png) 上で定義される確率密度関数の集合

- We can now define elementary distances and divergences between two distributions ![image](https://user-images.githubusercontent.com/25688193/56362134-1c05d600-6224-11e9-967b-9fd55e218c4c.png):
    - ２つの分布 ![image](https://user-images.githubusercontent.com/25688193/56362134-1c05d600-6224-11e9-967b-9fd55e218c4c.png) の間の初歩的な [elementary] 距離とダイバージェンスを定義できる。即ち、

![image](https://user-images.githubusercontent.com/25688193/56362277-65562580-6224-11e9-86a3-8a0132bde1b6.png)<br>

- where both ![image](https://user-images.githubusercontent.com/25688193/56362373-9fbfc280-6224-11e9-9e77-65b5e06c829b.png) and ![image](https://user-images.githubusercontent.com/25688193/56362414-bcf49100-6224-11e9-8fa9-bdb3b2b4308a.png) are assumed to be absolutely continuous, and therefore admit densities, with respect to a same measure μ defined on ![image](https://user-images.githubusercontent.com/25688193/56361196-b1539b00-6221-11e9-8ac0-558b8bfdc3d6.png).(2)
    - ここで [where]、![image](https://user-images.githubusercontent.com/25688193/56362373-9fbfc280-6224-11e9-9e77-65b5e06c829b.png) と ![image](https://user-images.githubusercontent.com/25688193/56362414-bcf49100-6224-11e9-8fa9-bdb3b2b4308a.png) の両方は、絶対連続であると仮定され、そして、それ故、ボレル集合族 ![image](https://user-images.githubusercontent.com/25688193/56361196-b1539b00-6221-11e9-8ac0-558b8bfdc3d6.png) 上で定義された同じ測度 μ に関しての [respect to]、確率密度が認められる。

> 絶対連続：連続性や一様連続性よりも強い条件を課した連続性の概念。リプシッツ連続な写像は絶対連続

- (2) Recall that a probability distribution ![image](https://user-images.githubusercontent.com/25688193/56366544-596f6100-622e-11e9-9959-e6bf64509045.png) admits a density ![image](https://user-images.githubusercontent.com/25688193/56366612-7c9a1080-622e-11e9-8765-89e02d22160d.png) with respect to μ,that is, ![image](https://user-images.githubusercontent.com/25688193/56366642-8e7bb380-622e-11e9-88b0-b48e3fdfe963.png), if and only it is absolutely continuous with respect to μ, that is, ![image](https://user-images.githubusercontent.com/25688193/56366698-aeab7280-622e-11e9-814f-9c88e3b86b05.png).
    - 測度 μ に関しての確率密度

- The KL divergence is famously assymetric and possibly infinite when there are points such that ![image](https://user-images.githubusercontent.com/25688193/56366217-8d965200-622d-11e9-9672-271ab2a10585.png) and ![image](https://user-images.githubusercontent.com/25688193/56366267-b6b6e280-622d-11e9-8cc1-12498d8994ce.png).
    - ![image](https://user-images.githubusercontent.com/25688193/56366217-8d965200-622d-11e9-9672-271ab2a10585.png) と ![image](https://user-images.githubusercontent.com/25688193/56366267-b6b6e280-622d-11e9-8cc1-12498d8994ce.png) のような点が存在するとき、KLダイバージェンスは、よく知られているように非対称性で、無限大になる可能性がある。


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）

## 4. Empirical Results


# ■ メソッド（実験方法）

# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


