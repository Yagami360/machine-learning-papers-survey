# ■ 論文
- 論文タイトル：「Wasserstein GAN」
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
    - もし真のデータ分布 ![image](https://user-images.githubusercontent.com/25688193/56339104-ba277b00-61e7-11e9-9454-a661bfd50b19.png) が、確率密度であることを許可し、（＝確率密度で表現できることを前提とし）
    - そして、![image](https://user-images.githubusercontent.com/25688193/56339217-3cb03a80-61e8-11e9-9bff-a7ace9111bd7.png) は、パラメーター化された ![image](https://user-images.githubusercontent.com/25688193/56339255-62d5da80-61e8-11e9-9000-67762c55123d.png) の分布とするならば、
    - そのとき、漸近的に [asymptotically]、この量は、KLダイバージェンス ![image](https://user-images.githubusercontent.com/25688193/56339053-8cdacd00-61e7-11e9-990d-02114b588b03.png) を最小化する。

- For this to make sense, we need the model density ![image](https://user-images.githubusercontent.com/25688193/56339255-62d5da80-61e8-11e9-9000-67762c55123d.png) to exist. 
    - このことが意味をなすのに、モデルの分布 ![image](https://user-images.githubusercontent.com/25688193/56339255-62d5da80-61e8-11e9-9000-67762c55123d.png) が存在している必要がある。

- This is not the case in the rather common situation where we are dealing with distributions supported by low dimensional manifolds. 
    - このことは、低次元の多様体 [manifolds] によっての確率分布の台を扱うような、かなり [rather] 一般的な [common] シチュエーションでは、当てはまらない。[not the case]

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
    - 存在しないかもしれない（真の分布）![image](https://user-images.githubusercontent.com/25688193/56339104-ba277b00-61e7-11e9-9454-a661bfd50b19.png) の確率密度を推定することよりも、
    - 固定された分布 p(z) でのランダム変数 Z を定義することができる。
    - そして、その（ランダム変数 Z を、）パラメリック関数（通常は [typically] ある種のニューラルネット） ![image](https://user-images.githubusercontent.com/25688193/56343805-b94a1580-61f6-11e9-8a61-4834acf729fb.png) に通す。
    - （この関数というのは、）特定の [certain] 分布 ![image](https://user-images.githubusercontent.com/25688193/56339217-3cb03a80-61e8-11e9-9bff-a7ace9111bd7.png) に従うサンプルを、直接的に生成するような関数

- By varying θ, we can change this distribution and make it close to the real data distribution ![image](https://user-images.githubusercontent.com/25688193/56339104-ba277b00-61e7-11e9-9454-a661bfd50b19.png).
    - θ をベリファイすることによって、この分布（![image](https://user-images.githubusercontent.com/25688193/56339217-3cb03a80-61e8-11e9-9bff-a7ace9111bd7.png) の形状を）変えることができ、真の分布 ![image](https://user-images.githubusercontent.com/25688193/56339104-ba277b00-61e7-11e9-9454-a661bfd50b19.png) に近づけることが出来る。

- This is useful in two ways.
    - これは、２つの方法で便利である。

- First of all, unlike densities, this approach can represent distributions conned to a low dimensional manifold.
    - まず第１に、確率密度分布とは異なり、このアプローチは、低次元多様体に結合された確率分布を表現することが出来る。

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

> ![image](https://user-images.githubusercontent.com/25688193/56361196-b1539b00-6221-11e9-8ac0-558b8bfdc3d6.png) は、σ加法族。<br>
> Σ は、この σ-加法族 ![image](https://user-images.githubusercontent.com/25688193/56361196-b1539b00-6221-11e9-8ac0-558b8bfdc3d6.png) の部分集合であるボレル集合族。（画像の空間 [0,1]^d を要素とするようなσ-加法族は、ボレル加法族になる。）<br>

- Let ![image](https://user-images.githubusercontent.com/25688193/56361987-b6194e80-6223-11e9-8a29-56d8d77ef627.png) denote the space of probability measures defined on ![image](https://user-images.githubusercontent.com/25688193/56361196-b1539b00-6221-11e9-8ac0-558b8bfdc3d6.png). 
    - ![image](https://user-images.githubusercontent.com/25688193/56361987-b6194e80-6223-11e9-8a29-56d8d77ef627.png) は、σ-加法族 ![image](https://user-images.githubusercontent.com/25688193/56361196-b1539b00-6221-11e9-8ac0-558b8bfdc3d6.png) 上で定義された、確率測度の空間を意味している。

> ![image](https://user-images.githubusercontent.com/25688193/56361987-b6194e80-6223-11e9-8a29-56d8d77ef627.png) は、σ-加法族 ![image](https://user-images.githubusercontent.com/25688193/56361196-b1539b00-6221-11e9-8ac0-558b8bfdc3d6.png) 上で定義される確率測度の集合

- We can now define elementary distances and divergences between two distributions ![image](https://user-images.githubusercontent.com/25688193/56362134-1c05d600-6224-11e9-967b-9fd55e218c4c.png):
    - ２つの分布 ![image](https://user-images.githubusercontent.com/25688193/56362134-1c05d600-6224-11e9-967b-9fd55e218c4c.png) の間の初歩的な [elementary] 距離とダイバージェンスを定義できる。即ち、

> GANの文脈では、<br>
> ![image](https://user-images.githubusercontent.com/25688193/56362373-9fbfc280-6224-11e9-9e77-65b5e06c829b.png) は、真の確率分布。<br>
> ![image](https://user-images.githubusercontent.com/25688193/56362414-bcf49100-6224-11e9-8fa9-bdb3b2b4308a.png) は、（生成器が出力する）モデルの確率分布。<br>
> GANでは、このモデルの確率分布を真の確率分布に近づけることが学習の目的となる。

![image](https://user-images.githubusercontent.com/25688193/56362277-65562580-6224-11e9-86a3-8a0132bde1b6.png)<br>

- where both ![image](https://user-images.githubusercontent.com/25688193/56362373-9fbfc280-6224-11e9-9e77-65b5e06c829b.png) and ![image](https://user-images.githubusercontent.com/25688193/56362414-bcf49100-6224-11e9-8fa9-bdb3b2b4308a.png) are assumed to be absolutely continuous, and therefore admit densities, with respect to a same measure μ defined on ![image](https://user-images.githubusercontent.com/25688193/56361196-b1539b00-6221-11e9-8ac0-558b8bfdc3d6.png).(2)
    - ここで [where]、確率分布 ![image](https://user-images.githubusercontent.com/25688193/56362373-9fbfc280-6224-11e9-9e77-65b5e06c829b.png) と ![image](https://user-images.githubusercontent.com/25688193/56362414-bcf49100-6224-11e9-8fa9-bdb3b2b4308a.png) の両方は、絶対連続であると仮定され、そして、それ故、σー加法族 ![image](https://user-images.githubusercontent.com/25688193/56361196-b1539b00-6221-11e9-8ac0-558b8bfdc3d6.png) 上で定義された同じ測度 μ に関しての [respect to]、確率密度が認められる。

> 絶対連続：連続性や一様連続性よりも強い条件を課した連続性の概念。リプシッツ連続な写像は絶対連続

- (2) Recall that a probability distribution ![image](https://user-images.githubusercontent.com/25688193/56366544-596f6100-622e-11e9-9959-e6bf64509045.png) admits a density ![image](https://user-images.githubusercontent.com/25688193/56366612-7c9a1080-622e-11e9-8765-89e02d22160d.png) with respect to μ,that is, ![image](https://user-images.githubusercontent.com/25688193/56366642-8e7bb380-622e-11e9-88b0-b48e3fdfe963.png), if and only it is absolutely continuous with respect to μ, that is, ![image](https://user-images.githubusercontent.com/25688193/56366698-aeab7280-622e-11e9-814f-9c88e3b86b05.png).
    - 確率分布 ![image](https://user-images.githubusercontent.com/25688193/56366544-596f6100-622e-11e9-9959-e6bf64509045.png) は、測度 μ に関しての確率密度 ![image](https://user-images.githubusercontent.com/25688193/56366612-7c9a1080-622e-11e9-8765-89e02d22160d.png) を許容することを思い出すこと。
    - 即ち [that is]、![image](https://user-images.githubusercontent.com/25688193/56366642-8e7bb380-622e-11e9-88b0-b48e3fdfe963.png)
    - 測度 μ に関して、絶対連続の場合にのみ、![image](https://user-images.githubusercontent.com/25688193/56366698-aeab7280-622e-11e9-814f-9c88e3b86b05.png)

- The KL divergence is famously assymetric and possibly infinite when there are points such that ![image](https://user-images.githubusercontent.com/25688193/56366217-8d965200-622d-11e9-9672-271ab2a10585.png) and ![image](https://user-images.githubusercontent.com/25688193/56366267-b6b6e280-622d-11e9-8cc1-12498d8994ce.png).
    - ![image](https://user-images.githubusercontent.com/25688193/56366217-8d965200-622d-11e9-9672-271ab2a10585.png) と ![image](https://user-images.githubusercontent.com/25688193/56366267-b6b6e280-622d-11e9-8cc1-12498d8994ce.png) のような点が存在するとき、KLダイバージェンスは、よく知られているように非対称性で、無限大になる可能性がある。

<br>

![image](https://user-images.githubusercontent.com/25688193/56399783-24ddc280-628b-11e9-803d-7e00e5c130f0.png)<br>

- where Pm is the mixture ![image](https://user-images.githubusercontent.com/25688193/56399825-4d65bc80-628b-11e9-9b94-ce31dc04b7b2.png). 
    - ここで、Pm は、混合物 ![image](https://user-images.githubusercontent.com/25688193/56399825-4d65bc80-628b-11e9-9b94-ce31dc04b7b2.png) である。

- This divergence is symmetrical and always defined because we can choose μ = Pm.
    - このダイバージェンスは、対称であり、μ = Pm を選択できるので、常に定義できる。

<br>

![image](https://user-images.githubusercontent.com/25688193/56399901-cebd4f00-628b-11e9-877c-deb696203dd6.png)<br>

- where ![image](https://user-images.githubusercontent.com/25688193/56399927-00361a80-628c-11e9-8e99-f7918d88bf5a.png) denotes the set of all joint distributions ![image](https://user-images.githubusercontent.com/25688193/56399965-37a4c700-628c-11e9-8df3-d160b87962f9.png) whose marginals are respectively ![image](https://user-images.githubusercontent.com/25688193/56362373-9fbfc280-6224-11e9-9e77-65b5e06c829b.png) and ![image](https://user-images.githubusercontent.com/25688193/56362414-bcf49100-6224-11e9-8fa9-bdb3b2b4308a.png).
    - ここで、![image](https://user-images.githubusercontent.com/25688193/56399927-00361a80-628c-11e9-8e99-f7918d88bf5a.png) は、周辺 [marginals ] 分布が各々 [respectively ] ![image](https://user-images.githubusercontent.com/25688193/56362373-9fbfc280-6224-11e9-9e77-65b5e06c829b.png) と ![image](https://user-images.githubusercontent.com/25688193/56362414-bcf49100-6224-11e9-8fa9-bdb3b2b4308a.png) であるような、同時分布 [joint distributions] ![image](https://user-images.githubusercontent.com/25688193/56399965-37a4c700-628c-11e9-8df3-d160b87962f9.png) の集合を表す。

> ![image](https://user-images.githubusercontent.com/25688193/56408519-ddb5f880-62af-11e9-999e-bfcf760abc05.png)<br>

- Intuitively, ![image](https://user-images.githubusercontent.com/25688193/56399927-00361a80-628c-11e9-8e99-f7918d88bf5a.png) indicates how much "mass" must be transported from x to y in order to transform the distributions ![image](https://user-images.githubusercontent.com/25688193/56362373-9fbfc280-6224-11e9-9e77-65b5e06c829b.png) into the distribution ![image](https://user-images.githubusercontent.com/25688193/56362414-bcf49100-6224-11e9-8fa9-bdb3b2b4308a.png).
    - 直感的には [Intuitively]、同時確率分布 ![image](https://user-images.githubusercontent.com/25688193/56399927-00361a80-628c-11e9-8e99-f7918d88bf5a.png) は、確率分布 ![image](https://user-images.githubusercontent.com/25688193/56362373-9fbfc280-6224-11e9-9e77-65b5e06c829b.png) を確率分布 ![image](https://user-images.githubusercontent.com/25688193/56362414-bcf49100-6224-11e9-8fa9-bdb3b2b4308a.png) に変換するために、どの程度の質量が、x から y に転送されたのかを示している。

- The EM distance then is the "cost" of the optimal transport plan.
    - EM 距離は、このとき、最適輸送問題 [optimal transport plan] のコストになる。

> ![image](https://user-images.githubusercontent.com/25688193/56409771-71d68e80-62b5-11e9-8e0f-10fe770a55de.png)<br>


<br>

- The following example illustrates how apparently simple sequences of probability distributions converge under the EM distance but do not converge under the other distances and divergences denfined above.
    - 以下の例は、どの程度明らかに、確率分布単純な系列が、EM距離のもとで収束するのか？一方、上で定義した他の距離やダイバージェンス（KLダイバージェンス等）のもとで収束しないのか？をイラスト化している。

![image](https://user-images.githubusercontent.com/25688193/56400730-61f88380-6290-11e9-9104-b1875585380c.png)<br>

> ![image](https://user-images.githubusercontent.com/25688193/56401620-cb7a9100-6294-11e9-93e0-f6c3c96efba4.png)<br>

- When θt → 0, the sequence ![image](https://user-images.githubusercontent.com/25688193/56346946-2f9e4600-61fe-11e9-840e-a67c017096ed.png) converges to P0 under the EM distance, but does not converge at all under either the JS, KL, reverse KL, or TV divergences.
    - θt → 0 とすると、系列 ![image](https://user-images.githubusercontent.com/25688193/56346946-2f9e4600-61fe-11e9-840e-a67c017096ed.png) は、EM距離のもとでは、P0 に収束する。
    - しかし、JSダイバージェンス、KLダイバージェンス、TV距離といったものは収束しない。

- Figure 1 illustrates this for the case of the EM and JS distances.

![image](https://user-images.githubusercontent.com/25688193/56401633-d8978000-6294-11e9-8028-c22aa873caf8.png)<br>

- > Figure 1: These plots show ρ(Pθ;P0) as a function of θ when ρ is the EM distance (left plot) or the JS divergence (right plot). 
    - > 図１：これらのプロットは、θの関数である距離 ρ(Pθ;P0) を示している。
    - > 左図は、距離 ρ が EM距離であるとき。右図は、距離 ρ が JSダイバージェンスであるとき。

- > The EM plot is continuous and provides a usable gradient everywhere.
    - > EM 距離のプロットは、連続で、すべての使用可能な場所で勾配を提供している。

- > The JS plot is not continuous and does not provide a usable gradient.
    - > JSダイバージェンスのプロットは、非連続で、すべての使用可能な場所で勾配を提供していない。

> JSダイバージェンスでは、勾配の傾きが０になっており、勾配損失問題が発生している。一方、EM距離では、勾配損失問題は発生していない。

<br>

- Example 1 gives us a case where we can learn a probability distribution over a low dimensional manifold by doing gradient descent on the EM distance.
    - 例１は、EM距離で最急降下法をすることによって、低次元多様体上の確率分布を学習することが出来るケースを提供する。

- This cannot be done with the other distances and divergences because the resulting loss function is not even continuous.
    - このことは、（EM距離以外の）他の距離やダイバージェンスで行うことができない。
    - なぜならば、損失関数の結果が、連続でさえないためである。

- Although this simple example features distributions with disjoint supports, the same conclusion holds when the supports have a non empty intersection contained in a set of measure zero.
    - この単純な例は、（２つの確率分布）ばらばらになっている [disjoint] 台（Supp）を持つ確率分布を特徴としているが、
    - 台が、測度０の集合を含むような、空でない共通部分を持つときに、同じ結論が、維持される。[hold]

- This happens to be the case when two low dimensional manifolds intersect in general position [1].
    - これは、２つの低次元多様体が、一般的な位置で交差するケースで発生する。

> ここでいう低次元多様体とは、いわゆる多様体仮説の話？（データは高次元だが、意味のあるデータは、実は低次元の空間内に局在している。）

> ![image](https://user-images.githubusercontent.com/25688193/56405137-d25dcf80-62a5-11e9-9cdc-0b5aa149f9d8.png)<br>

> ![image](https://user-images.githubusercontent.com/25688193/56405173-0a651280-62a6-11e9-9ed4-decfdc8a7570.png)<br>

![image](https://user-images.githubusercontent.com/25688193/56425280-7d927700-62ee-11e9-9d80-d7837a9ef9b3.png)<br>

<br>

- Since the Wasserstein distance is much weaker than the JS distance (3), we can now ask whether ![image](https://user-images.githubusercontent.com/25688193/56405230-6334ab00-62a6-11e9-9131-345ca06f9f67.png) is a continuous loss function on θ under mild assumptions.
    - Wassertein 距離は、JSダイバージェンスより、（位相的な意味で？）はるかに弱いので (3)、今度は、![image](https://user-images.githubusercontent.com/25688193/56405230-6334ab00-62a6-11e9-9131-345ca06f9f67.png) が、ゆるい仮定のもとで、θ に対して、連続な損失関数であるかを尋ねることが出来る。

- (3) : The argument for why this happens, and indeed how we arrived to the idea that Wasserstein is what we should really be optimizing is displayed in Appendix A. 
    - We strongly encourage the interested reader who is not afraid of the mathematics to go through it.

- (3) : この主張がなぜ起こるのか？そして、どのようにして、本当に最適化すべきものが、Wassertein 距離であるというアイデアを思いつたのか？ということは付録 A に示されている。
    - 我々は、数学に心配ない興味のある読者に、それを経験することを強く勧める。


- This, and more, is true, as we now state and prove.
    - これは、今から述べ証明するように、本当である。

![image](https://user-images.githubusercontent.com/25688193/56402282-94f24580-6297-11e9-9a45-7e418df1e9c8.png)<br>

- The following corollary tells us that learning by minimizing the EM distance makes sense (at least in theory) with neural networks.
    - 以下の命題 [corollary] は、EM距離を最小化することによっての学習は、ニューラルネットワークでは、（少なくとも理論的には）意味を成すことを教えてくれる。

![image](https://user-images.githubusercontent.com/25688193/56405893-7137fb00-62a9-11e9-9479-ab012b103d16.png)<br>
![image](https://user-images.githubusercontent.com/25688193/56405907-890f7f00-62a9-11e9-97dc-365c5762cd75.png)<br>

- (4) : By a feedforward neural network we mean a function composed by affine transformations and pointwise nonlinearities which are smooth Lipschitz functions (such as the sigmoid, tanh, elu, softplus, etc). Note: the statement is also true for rectier nonlinearities but the proof is more technical (even though very similar) so we omit it.


- > You’ll need to refer to the paper to see what “sufficiently nice” means, but for our purposes it’s enough to know that it’s satisfied for feedfoward networks that use standard nonlinearites. Thus, out of JS, KL, and Wassertstein distance, only the Wasserstein distance has guarantees of continuity and differentiability, which are both things you really want in a loss function.
    - > 「十分に素晴らしい」とはどういう意味なのかを確認するには、この論文を参照する必要がありますが、私たちの目的のためには、標準的な非線形であるフィードフォワードネットワークには十分であることを知っていれば十分です。 したがって、JS、KL、およびWassertstein距離のうち、Wasserstein距離のみが連続性と微分可能性を保証します。これらはどちらも損失関数に本当に必要なものです。 

<br>

- All this shows that EM is a much more sensible cost function for our problem than at least the Jensen-Shannon divergence.
    - 以上のことから [All this]、我々の問題に対して、EM距離が、JSダイバージェンスより、はるかに賢明な [sensible] コスト関数であることが分かる。

- The following theorem describes the relative strength of the topologies induced by these distances and divergences, with KL the strongest, followed by JS and TV, and EM the weakest.
    - 以下の定理は、これら距離やダイバージェンスによって誘発される [induced]、位相の相対的な強さを記述している。
    - KLダイバージェンスが最も強く、次にJSダイバージェンスとTV距離が続き、EM距離が最も弱い。

![image](https://user-images.githubusercontent.com/25688193/56406018-3b474680-62aa-11e9-807e-1917d7920556.png)<br>

> つまりは、KLダイバージェンスやJSダイバージェンスで収束するような、全ての確率分布が、Wesseterin 距離のもとでも収束することを示している？


- This highlights the fact that the KL, JS, and TV distances are not sensible cost functions when learning distributions supported by low dimensional manifolds.
    - この強調する事実は、低次元多様体による台となっている確率分布を学習するときに、KL,JS,TV が賢明なコスト関数でないということである。

- However the EM distance is sensible in that setup.
    - しかしながら、EM距離は、この設定のもとで、賢明なコスト関数になっている。

![image](https://user-images.githubusercontent.com/25688193/56408335-ece87680-62ae-11e9-9af5-0d3d25904665.png)<br>

> 結論としては、Wasseerstein Metric（＝EM距離）を使用することで、真の分布とモデルの分布の２つの確率分布の台（Supp）が違っても、コスト関数を連続関数として定義できる。これにより、低次元多様体同士の距離をうまく測れる。

- This obviously leads us to the next section where we introduce a practical approximation of optimizing the EM distance.
    - このことは明らかに、EM距離を最適化する実用的な近似を紹介する次のセクションへ、我々を導く。

## 3. Wasserstein GAN

- Again, Theorem 2 points to the fact that ![image](https://user-images.githubusercontent.com/25688193/56408157-105ef180-62ae-11e9-8010-94b20e700840.png) might have nicer properties when optimized than ![image](https://user-images.githubusercontent.com/25688193/56408226-62077c00-62ae-11e9-85b0-aceec1b2e70b.png).
    - 再度、定理２は、最適化するときに、Wassertein距離 ![image](https://user-images.githubusercontent.com/25688193/56408157-105ef180-62ae-11e9-8010-94b20e700840.png) は、JSダイバージェンス ![image](https://user-images.githubusercontent.com/25688193/56408226-62077c00-62ae-11e9-85b0-aceec1b2e70b.png) よりも、より良い性質 [property] を持つかもしれないという事実を指摘している。

- However, the infimum in (1) is highly intractable.
    - しかしながら、式１の下限（inf）は、非常に扱いにくい [intractable]

- On the other hand, the Kantorovich-Rubinstein duality [22] tells us that
    - 一方、Kantorovich-Rubinstein 双対性は、以下の式を教える。

![image](https://user-images.githubusercontent.com/25688193/56408238-751a4c00-62ae-11e9-9697-30868ee05ec4.png)<br>

> EM距離を計算することそれ自体が最適化問題最適輸送問題での問題設定となっている。ここで、この最適輸送問題は、線形計画法であるので、線形計画法と同様にして、主問題とその双対問題の形式が得られる。

- where the supremum is over all the 1-Lipschitz functions ![image](https://user-images.githubusercontent.com/25688193/56410036-94b57280-62b6-11e9-8022-7f56683468bd.png). 
    - ここで、上限（sup）は、全ての 1-リプシッツ連続な関数 ![image](https://user-images.githubusercontent.com/25688193/56410036-94b57280-62b6-11e9-8022-7f56683468bd.png) の上にある。

- Note that if we replace ![image](https://user-images.githubusercontent.com/25688193/56410209-42c11c80-62b7-11e9-98ee-b21ae7179cc7.png) for ||f||_L ≦ K (consider K-Lipschitz for some constant K), then we end up with K・W(Pr; Pg). 
    - もし、（いくつかの定数 K の K-リプシッツ連続性を考慮して、） ![image](https://user-images.githubusercontent.com/25688193/56410209-42c11c80-62b7-11e9-98ee-b21ae7179cc7.png) を ![image](https://user-images.githubusercontent.com/25688193/56410276-7dc35000-62b7-11e9-8e81-33100f692a2b.png) と置き換えると、![image](https://user-images.githubusercontent.com/25688193/56410299-9895c480-62b7-11e9-9ba3-7c93986c1498.png) という結果になることに注意。

- Therefore, if we have a parameterized family of functions ![image](https://user-images.githubusercontent.com/25688193/56410552-6df83b80-62b8-11e9-93f3-80f32fbf8649.png) that are all K-Lipschitz for some K, we could consider solving the problem
    - それ故、もし、いくつかの K に対しての K-リプシッツ連続であるような、パラメーター化された関数の族 ![image](https://user-images.githubusercontent.com/25688193/56410552-6df83b80-62b8-11e9-93f3-80f32fbf8649.png) を持つならば、以下の問題を解くことを検討する。

![image](https://user-images.githubusercontent.com/25688193/56410069-bca4d600-62b6-11e9-9dd4-68dca44141f8.png)<br>

> ここでの w は、nニューラルネットワークの重みのこと。つまり、リプシッツ連続な関数を、重みパラメーター w で近似することを考えている。

- and if the supremum in (2) is attained for some w ∈ W (a pretty strong assumption akin to what's assumed when proving consistency of an estimator), this process would yield a calculation of ![image](https://user-images.githubusercontent.com/25688193/56411046-5752e400-62ba-11e9-9245-6d3633ddea57.png) up to a multiplicative constant.
    - そしてもし、式 (2) の sup が、いくつかの w ∈ W に対して、達成する [attained for] ならば、
    - これは、（推定器の一貫性 [consistency] を証明するときに、仮定されたものに類似した [akin]、とても強い仮定）
    - このプロセスは、増加する [multiplicative constant] 定数まで、Wasserstein 距離 ![image](https://user-images.githubusercontent.com/25688193/56411046-5752e400-62ba-11e9-9245-6d3633ddea57.png) の計算を生み出す [yield] だろう。

- Furthermore, we could consider diffierentiating ![image](https://user-images.githubusercontent.com/25688193/56411046-5752e400-62ba-11e9-9245-6d3633ddea57.png) (again, up to a constant) by back-proping through equation (2) via estimating ![image](https://user-images.githubusercontent.com/25688193/56411206-f1b32780-62ba-11e9-88a8-ba6c7ffcf825.png).
    - 更には、![image](https://user-images.githubusercontent.com/25688193/56411206-f1b32780-62ba-11e9-88a8-ba6c7ffcf825.png) の推定を通じて、式２の誤差逆伝搬によって、（再び定数の値まで、）Wasserstein 距離 ![image](https://user-images.githubusercontent.com/25688193/56411046-5752e400-62ba-11e9-9245-6d3633ddea57.png) の微分を考慮することができる。

- While this is all intuition, we now prove that this process is principled under the optimality assumption.
    - これらは全て直感的な [intuition] ことであるが、今度は、最適化の仮定の元で、このプロセスは原則的であることを証明する。

![image](https://user-images.githubusercontent.com/25688193/56411645-62a70f00-62bc-11e9-9688-2cdc610601d7.png)<br>

- Now comes the question of finding the function f that solves the maximization problem in equation (2).
    - 今度は、式２の最大化問題を解くような関数 f を見つける問題が出てきた。

- To roughly approximate this, something that we can do is train a neural network parameterized with weights w lying in a compact space W and then backprop through ![image](https://user-images.githubusercontent.com/25688193/56411206-f1b32780-62ba-11e9-88a8-ba6c7ffcf825.png), as we would do with a typical GAN.
    - このおよその近似のために、出来ることは、典型的な GAN で行っていたように、コンパクト空間 W に横たわる [lying] 重み w でパラメーター化されたニューラルネットワークを学習することである。
    - そして、![image](https://user-images.githubusercontent.com/25688193/56411206-f1b32780-62ba-11e9-88a8-ba6c7ffcf825.png) を通じて、誤差逆伝搬することである。

- Note that the fact that W is compact implies that all the functions fw will be K-Lipschitz for some K that only depends on W and not the individual weights, therefore approximating (2) up to an irrelevant scaling factor and the capacity of the `critic' fw. 
    - W がコンパクト空間であるという事実は、Wのみに依存して個々の w に依存しないような、全ての関数 ![image](https://user-images.githubusercontent.com/25688193/56412264-879c8180-62be-11e9-9889-f8a421f65c2c.png) が、ある K に対しての K-リプシッツ連続な関数になることを意味していることに注意。
    - それゆえに、式２の近似は、スケーリング係数とクリティック（＝識別器） ![image](https://user-images.githubusercontent.com/25688193/56412264-879c8180-62be-11e9-9889-f8a421f65c2c.png)の能力に無関係 [irrelevant] になる。

> D(x)はもはや、識別結果（0 or 1）としての意味をもたないため、出力をsigmoid関数によって[0,1]に変換する必要がなくなる。そのため、WGANでは D(x) を f(x) と表したり、識別器の代わりにクリティックと呼んだりします。

> ![image](https://user-images.githubusercontent.com/25688193/56424557-b846e000-62eb-11e9-8333-55696b073b6a.png)<br>

- In order to have parameters w lie in a compact space, something simple we can do is clamp the weights to a fixed box (say ![image](https://user-images.githubusercontent.com/25688193/56412217-65a2ff00-62be-11e9-8f1b-084f61cc7b32.png)) after each gradient update.
    - パラメーター w をコンパクト空間に横たえるために、我々が出来る簡単なことは、各勾配を更新した後に、固定ボックス（![image](https://user-images.githubusercontent.com/25688193/56412217-65a2ff00-62be-11e9-8f1b-084f61cc7b32.png)）に重みを留める [clamp] ことである。

> 要は、fw をリプシッツな関数にするため、重み w のそれぞれの値の絶対値がある小さな値以上にならないようclipすることをいっている。

- The Wasserstein Generative Adversarial Network (WGAN) procedure is described in Algorithm 1.

![image](https://user-images.githubusercontent.com/25688193/56411984-a3ebee80-62bd-11e9-91e1-288b3eec5855.png)<br>

- Weight clipping is a clearly terrible way to enforce a Lipschitz constraint.
    - 重みの固定は、リプシッツ連続を強いるための、明らかにひどい方法である。

- If the clipping parameter is large, then it can take a long time for any weights to reach their limit, thereby making it harder to train the critic till optimality.
    -　もしクリッピングパラメーターが大きれば、いくつかの重みに対して、極限に到達するために長い時間がかかる。
    - それによって [thereby]、最適値になるまでクリティック（＝識別器）を学習することがよく困難になる。

- If the clipping is small, this can easily lead to vanishing gradients when the number of layers is big, or batch normalization is not used (such as in RNNs).
    - もしクリッピング値が小さければ、層の数が多くなるときに、或いは、BatchNormが使われていないときに、容易に勾配損失問題に導く。

- We experimented with simple variants (such as projecting the weights to a sphere) with little diffierence, and we stuck with weight clipping due to its simplicity and already good performance.
    - 我々は、ほとんど違いがないような、単純な変異 [variant]（例えば、球への重みの射影など）を実験した。
    - そして、それ（＝重みクリッピング）の簡単さとすでに良いパフォーマンスであるために、重みクリッピングを行き詰まって使用した。

- However, we do leave the topic of enforcing Lipschitz constraints in a neural network setting for further investigation, and we actively encourage interested researchers to improve on this method.
    - しかしながら、さらなる研究のために、ニューラルネットワークの設定において、リプシッツ連続を強制するトピックを残している。
    - そして、興味ある研究者に、このメソッドを改善する積極的に推奨する。

<br>

- The fact that the EM distance is continuous and diffierentiable a.e. means that we can (and should) train the critic till optimality.
    - EM距離が、連続で、微分可能であるという事実は、ほとんどいたるところで [a.e.]、最適値までクリティック（＝識別器）を学習することが出来る或いはすべきことを意味する。

- The argument is simple, the more we train the critic, the more reliable gradient of the Wasserstein we get, which is actually useful by the fact that Wasserstein is diffierentiable almost everywhere.
    - この主張は簡単で、クリティックを学習すればするほど、Wasserstein距離のより信頼できる [reliable] 勾配が得られる。
    - このことは、Wasserstein距離がほとんどいたるところで微分可能であるという事実によって、実際に便利である。

- For the JS, as the discriminator gets better the gradients get more reliable but the true gradient is 0 since the JS is locally saturated and we get vanishing gradients, as can be seen in Figure 1 of this paper and Theorem 2.4 of [1].
    - JSダイバージェンスの場合、識別器がよくなるほど、勾配はより信頼できるものになる。
    - しかし、この論文の図１と定理2.4でわかるように、JSダイバージェンスは局所的に飽和し、勾配損失するので、真の値は０である。

- In Figure 2 we show a proof of concept of this, where we train a GAN discriminator and a WGAN critic till optimality.
    - 図２では、このコンセプトの証明を見せる。ここで、GANの識別器とWGANのクリティックを最適状態まで学習している。

- The discriminator learns very quickly to distinguish between fake and real, and as expected provides no reliable gradient information.
    - 識別器は、偽物と本物の間の識別を非常に早く学習する。
    - そして、予想されるものとして、信頼性のない勾配の情報を提供する。

- The critic, however, can't saturate, and converges to a linear function that gives remarkably clean gradients everywhere.
    - クリティックは、しかしながら、（これを）満たさない。
    - そして、ほとんどいたるところで著しく [remarkably] きれいな勾配を与えるような、線形関数に収束する。

- The fact that we constrain the weights limits the possible growth of the function to be at most linear in diffierent parts of the space, forcing the optimal critic to have this behaviour.
    - 重みを制限するという事実は、空間の異なる部分において、関数の可能な成長に対して、せいぜい線形（関数）であるように制限し、
    - 最適なクリティックに対して、この振る舞いを強制する。

![image](https://user-images.githubusercontent.com/25688193/56412100-10ff8400-62be-11e9-9363-ee481c3e59ad.png)<br>

- > Figure 2: Optimal discriminator and critic when learning to diffierentiate two Gaussians.
    - > 図２：異なる２つのガウス分布を学習するときの、最適な識別機とクリティック。

- > As we can see, the discriminator of a minimax GAN saturates and results in vanishing gradients.
    - > 見て取れるように、GAN の minimax の識別器は、勾配消失という結果になる。（赤線）

- > Our WGAN critic provides very clean gradients on all parts of the space.
    - > WGAN のクリティックは、全ての空間部分において、とてもきれいな勾配を提供する。（水色線）

<br>

- Perhaps more importantly, the fact that we can train the critic till optimality makes it impossible to collapse modes when we do. 
    - おそらく [Perhaps] より重要なのは、クリティックを最適状態まで学習出来るという事実は、モード崩壊を不可能にする。

- This is due to the fact that mode collapse comes from the fact that the optimal generator for a fixed discriminator is a sum of deltas on the points the discriminator assigns the highest values, as observed by [4] and highlighted in [11].
    - [4] で観測され、[11] で強調されているように、
    - これは、モード崩壊が、固定された識別器に対しての最適な生成器が、識別器が最も高い値を割り当てる [assigns] 点での、デルタ（Δ）の合計である、
    - という事実から来ている、という事実によるものである。

<br>

- In the following section we display the practical benets of our new algorithm, and we provide an in-depth comparison of its behaviour and that of traditional GANs.
    - 次のセクションでは、我々の新しいアルゴリズムの実用的な利点を示す。
    - そして、伝統的な GAN のそれとの詳細な [in-depth] 比較 [comparison] を提供する。


# ■ 実験結果（主張の証明）・メソッド（実験方法）・議論（手法の良し悪し）

## 4. Empirical Results（経験的な結果）

- We run experiments on image generation using ourWasserstein-GAN algorithm and show that there are signicant practical benets to using it over the formulation used in standard GANs.
    - 我々は、WGAN のアルゴリズムを用いて、画像生成の実験を実施した。
    - そして、それ（＝WGAN）を使用することで、標準の GAN で使用されている定式化を超えて、重要な実用的な利点があることを示す。

- We claim two main benets:
    - a meaningful loss metric that correlates with the generator's convergence and sample quality
    - improved stability of the optimization process

- 我々は、２つの主な利点を主張する。
    - 生成器の収束とサンプルのクオリティに相関するような、意味のある損失関数
    - 最適化処理の安定性の向上

> この損失関数値が小さくなるほど、サンプルのクオリティを上昇するという関係は、実験による経験的な知見である？


### 4.1 Experimental Procedure

- We run experiments on image generation.
    - 画像生成の実験を実施する。

- The target distribution to learn is the LSUN-Bedrooms dataset [24] - a collection of natural images of indoor bedrooms.
    - 学習するための確率分布の教師データは、the LSUN-Bedrooms dataset である。
    - これは、室内の寝室の自然なコレクションである。

- Our baseline comparison is DCGAN [18], a GAN with a convolutional architecture trained with the standard GAN procedure using the -logD trick [4].
    - ベースラインの比較は、DCGAN である。
    - （これは、）-logD トリックを使用して、標準的な GAN の処理で学習された、畳み込み構造をもつ GAN である。

- The generated samples are 3-channel images of 64x64 pixels in size.
    - 生成されたサンプルは、3チャンネルの 64×64 pixel のサイズである。

- We use the hyper-parameters specied in Algorithm 1 for all of our experiments.
    - 全ての実験において、アルゴリズム１で指定されたハイパーパラメータを使用する。

### 4.2 Meaningful loss metric

- Because the WGAN algorithm attempts to train the critic f (lines 2-8 in Algorithm1) relatively well before each generator update (line 10 in Algorithm 1), the loss function at this point is an estimate of the EM distance, up to constant factors related to the way we constrain the Lipschitz constant of f.
    - WGAN アルゴリズムが、各生成器が更新する（アルゴリズム１の１０行目）の前に、クリティック f（アルゴリズム１の2-8行目）比較的よく学習しようと試みるため、
    - この地点での損失関数は、f のリプシッツ定数を強制する方法に関連した定数に一致して [up to]、EM距離の推定値（式３）となる。

<br>

- Our first experiment illustrates how this estimate correlates well with the quality of the generated samples.
    - 最初の実験では、この推定値（式３）が、生成されたサンプルのクオリティと、どのようにうまく相関するのかを示している。

- Besides the convolutional DCGAN architecture, we also ran experiments where we replace the generator or both the generator and the critic by 4-layer ReLU-MLP with 512 hidden units.
    - 畳み込みの DCGAN のアーキテクチャに加えて、
    - 生成器、或いは、生成器とクリティックの両方を、512個のユニットをもつ４層の Relu-MLP に置き換える実験を行った。

<br>

- Figure 3 plots the evolution of the WGAN estimate (3) of the EM distance during WGAN training for all three architectures.
    - 図３は、３つの全てのアーキテクチャに対して、WGAN を学習する間の、EM距離のWGANの推定値（式３）の進化をプロットしている。

- The plots clearly show that these curves correlate well with the visual quality of the generated samples.
    - プロットは、これらの曲線が、生成されたサンプルの見た目のクオリティとよく相関しているということを、明らかに見せている。

![image](https://user-images.githubusercontent.com/25688193/56450006-3c38b080-635c-11e9-909c-db1b0ddff722.png)<br>

- > Figure 3: Training curves and samples at diffierent stages of training.
    - > 図３：学習曲線と、異なる学習段階でのサンプル

- > We can see a clear correlation between lower error and better sample quality.
    - > 損失関数値が小さくなることとサンプルのクオリティがよくなることの間に、明らかな相関が見てとれる。

- > Upper left: the generator is an MLP with 4 hidden layers and 512 units at each layer.
    - > 左上図：生成器が、４つの隠れ層と、各層で 512 個のユニットをもつ MLP での図。

- > The loss decreases constistently as training progresses and sample quality increases.
    - > 損失関数値が、学習が進むにつれて一貫して [constistently] 減少しており、サンプルのクオリティは上昇している。

- > Upper right: the generator is a standard DCGAN.
    - > 右上図：生成器が、標準的な DCGAN での図

- > The loss decreases quickly and sample quality increases as well. 
    - > 損失関数値は、急激に減少し、サンプルのクオリティを同様にして上昇する。

- > In both upper plots the critic is a DCGAN without the sigmoid so losses can be subjected to comparison.
    - 両方の上側のプロットにおいて、損失値を比較の対称に出来るように、クリティックは sigmoid なしの DCGAN である。

- > Lower half: both the generator and the discriminator are MLPs with substantially high learning rates (so training failed).
    - > 下側の図：生成器と識別器の両方は、（学習が失敗するように、）十分に [substantially] 高い学習率での MLP である。

- > Loss is constant and samples are constant as well.
    - > 損失値は一定で、サンプルも同様にして一定である。

- > The training curves were passed through a median filter for visualization purposes.
    - > 学習曲線は、可視化の目的に対して、中間フィルターを通してある。

<br>

- To our knowledge, this is the first time in GAN literature that such a property is shown, where the loss of the GAN shows properties of convergence.
    - 我々が知る限り、GAN の文脈において、このような性質 [property] が見られたのは、初めてのことであり、
    - ここで [where]、GAN の loss値は収束の性質を見せる。

- This property is extremely useful when doing research in adversarial networks as one does not need to stare at the generated samples to figure out failure modes and to gain information on which models are doing better over others.
    - この性質は、敵対的ネットワークを調査するときに、極めて有効である。
    - （なぜならば、）１つには、失敗モードを理解し、モデルが他のモデルよりもよく動作しているかの情報を得るために、生成サンプルを、じっと見つめる必要がない（ので、[as]）

<br>

- However, we do not claim that this is a new method to quantitatively evaluate generative models yet.
    - しかしながら、我々は、生成モデルのクオリティを評価するための新しい方法であるとは、まだ主張しない。

- The constant scaling factor that depends on the critic's architecture means it's hard to compare models with diffierent critics.
    - クリティックのアーキテクチャに依存した、スケーリング要因の定数（＝リプシッツ定数K）は、異なるクリティックをもつモデルを比較することが難しいことを意味している。

- Even more, in practice the fact that the critic doesn't have infinite capacity makes it hard to know just how close to the EM distance our estimate really is.
    - さらには、実際上には、クリティックが無限の能力をもっていない、という事実は、
    - EM距離が、我々の推定値とどの程度近いか？ということを知ることが困難にする

- This being said, we have succesfully used the loss metric to validate our experiments repeatedly and without failure, and we see this as a huge improvement in training GANs which previously had no such facility.
    - そうはいっても [This being said]、我々の実験を検証するために、繰り返し、失敗することなしに、loss 値をうまく [succesfully ] 使用してした。
    - そして、我々は、以前にはこのような設備（＝便利さ） [facility] を持たなかった GAN の学習において、これを大きな改善と見ている。

<br>

- In contrast, Figure 4 plots the evolution of the GAN estimate of the JS distance during GAN training.
    - 対称的 [contrast] に、図４は、GAN の学習中に、JSダイバージェンスの推定値での、GAN の進化をプロットしたものである。

- More precisely, during GAN training, the discriminator is trained to maximize
    - より正確には、GAN を学習する間、識別器は、以下の式を最大化するように学習される。

![image](https://user-images.githubusercontent.com/25688193/56452749-66946900-6370-11e9-8c0c-dd86b24a4085.png)<br>

> 通常の GAN の識別器の損失関数の式になっている。

- which is is a lower bound of ![image](https://user-images.githubusercontent.com/25688193/56452838-043c6800-6372-11e9-8779-536d64396a3b.png).
    - （この式は、）![image](https://user-images.githubusercontent.com/25688193/56452838-043c6800-6372-11e9-8779-536d64396a3b.png) の下限 [lower bound] である。

- In the figure, we plot the quantity ![image](https://user-images.githubusercontent.com/25688193/56452858-6d23e000-6372-11e9-90c0-99412295f1c9.png), which is a lower bound of the JS distance.
    - 図では、JSダイバージェンスの下限である量 ![image](https://user-images.githubusercontent.com/25688193/56452858-6d23e000-6372-11e9-90c0-99412295f1c9.png) をプロットしている。

![image](https://user-images.githubusercontent.com/25688193/56452818-bd4e7280-6371-11e9-959e-c9a1028b9b8a.png)<br>

- > Figure 4: JS estimates for an MLP generator (upper left) and a DCGAN generator (upper right) trained with the standard GAN procedure.
    - > 図４：標準の GAN で学習された、MLP での生成器に対しての、JS推定値（左上図）
    - > 標準の GAN で学習された、DCGAN での生成器に 対しての、JS推定値（右上図）。

- > Both had a DCGAN discriminator.
    - > 両方とも DCGAN での識別器をもつ。

- > Both curves have increasing error.
    - > 両方の曲線とも、誤差値が増加している。

- > Samples get better for the DCGAN but the JS estimate increases or stays constant, pointing towards no signicant correlation between sample quality and loss.
    - > DCGAN に対してのサンプル（のクオリティ）は、よくなっている。
    - > しかし、JS推定値は、増加したり、一定になったりしており、
    - > サンプルのクオリティと loss 値の間に、重要な相関がないことを指し示している。[pointing]

- > Bottom: MLP with both generator and discriminator.
    - > 下段：生成器と識別器ともに、MLP

- > The curve goes up and down regardless of sample quality.
    - > 曲線は、サンプルのクオリティに関わらず、上がったり下がったりしている。

- > All training curves were passed through the same median filter as in Figure 3.
    - > 全ての学習曲線は、図３においてしていたように、同じ中間フィルタを通している。

<br>

- This quantity clearly correlates poorly the sample quality.
    - この量（＝JSダイバージェンス）は、明らかにサンプルのクオリティと、不十分に [poorly] 相関している。

- Note also that the JS estimate usually stays constant or goes up instead of going down.
    - JS推定値が、たいてい一定のままである、或いは、減少するかわりに上昇するということにも注意

- In fact it often remains very close to log 2 ≒ 0:69 which is the highest value taken by the JS distance.
    - 実際、JSダイバージェンスによって取られる最も高い値は、たいてい log 2 ≒ 0:69 に非常に近い値のままになる。

- In other words, the JS distance saturates, the discriminator has zero loss, and the generated samples are in some cases meaningful (DCGAN generator, top right plot) and in other cases collapse to a single nonsensical image [4].
    - 言い換えると、JSダイバージェンスが飽和し、識別器が０の loss 値となり、
    - そして、生成されたサンプルが、いくつかのケースにおいて、意味のあるものとなり（DCGANでの生成器：右上図のプロット）、
    - そして、他の場合において、単一の無意味な [nonsensical] 画像に崩壊する。

- This last phenomenon has been theoretically explained in [1] and highlighted in [11].
    - この最後の現象（＝無意味な画像に崩壊する現象）は、[1] で理論的に説明されている。そして、[11] で強調されている。

<br>

- When using the -logD trick [4], the discriminator loss and the generator loss are diffierent.
    - -logD トリックを使用した場合、識別器の loss 値と、生成器の loss 値は異なる。

- Figure 8 in Appendix E reports the same plots for GAN training, but using the generator loss instead of the discriminator loss.
    - 図８は、GAN の学習に対して、同じプロットを報告している。但し、識別器の loss 値の代わりに、生成器の loss 値を使っている。

- This does not change the conclusions.
    - このことは、結果を変えるものではない。

- E : Generator's cost during normal GAN training
    ![image](https://user-images.githubusercontent.com/25688193/56453094-68155f80-6377-11e9-87cf-ff44619ab72e.png)<br>

- > Figure 8: Cost of the generator during normal GAN training, for an MLP generator (upper left) and a DCGAN generator (upper right). Both had a DCGAN discriminator. 

- > Both curves have increasing error.

- > Samples get better for the DCGAN but the cost of the generator increases, pointing towards no signicant correlation between sample quality and loss. 

- > Bottom: MLP with both generator and discriminator.

- > The curve goes up and downregardless of sample quality.

- > All training curves were passed through the same median lter as in Figure 3.

> 図８は、-logD のトリックを使った場合の、通常の GAN での loss 値の変化であるが、このテクニックを使わない通常の GAN での loss 値の変化である図７での結果と同じ。

<br>

- Finally, as a negative result, we report that WGAN training becomes unstable at times when one uses a momentum based optimizer such as Adam [8] (with β1 > 0) on the critic, or when one uses high learning rates.
    - 最後に、否定的な結果となるが、WGAN の学習が、（β1 > 0での）Adam のようなモーメンタムベースの最適化アルゴリズムを使用するときや、高い学習率を使用するときに、ときどき不安定になるという報告をする。

- Since the loss for the critic is nonstationary, momentum based methods seemed to perform worse.
    - クリティックの loss 値が、非定常的 [nonstationary] であるので、モーメンタムベースの手法は、パフォーマンスが悪化するほうに見えた。

- We identfiied momentum as a potential cause because, as the loss blew up and samples got worse, the cosine between the Adam step and the gradient usually turned negative.
    - 我々は、モーメンタムを、潜在的な [potential] 原因 [cause] と特定した。[identified]
    - （なぜならば、）loss 値が増大して、サンプルが悪化するにつれて、Adam ステップと勾配の間のコサイン値 cos [cosine] が、たいてい負の値に変わるので [as]、

- The only places where this cosine was negative was in these situations of instability.
    - このコサイン値が負の値になる唯一の場所では、不安定さの状況であった。

- We therefore switched to RMSProp [21] which is known to perform well even on very nonstationary problems [13].
    - それ故、非常に非定常的な問題でさえも良いパフォーマンスを行うことでよく知られている RMSProp に切り替えた。

### 4.3 Improved stability

- One of the benets of WGAN is that it allows us to train the critic till optimality.
    - WGAN の利点の１つは、最適状態までクリティックを学習することを許容することである。

> loss 値がサンプルのクオリティに比例して、０に向かって安定的に単調減少するので、最適状態まで学習することが出来る。（通常の GAN では、loss 値が上がったり下がったりして、サンプルのクオリティが突然悪くなったりするので、このようなことが行えるとは限らない。）

- When the critic is trained to completion, it simply provides a loss to the generator that we can train as any other neural network. 
    - クリティックが、完成状態まで学習されると、
    - 他のニューラルネットワークとして学習出来るような生成器に、loss 値を提供する。

- This tells us that we no longer need to balance generator and discriminator's capacity properly.
    - このことは、もはや生成器と識別器の能力のバランスをとる必要がない性質を、我々に教えてくれる。

- The better the critic,the higher quality the gradients we use to train the generator.
    - クリティックがよくなるほど、生成器を学習するために、使用する勾配のクオリティが、より高くなる。

<br>

- We observe that WGANs are much more robust than GANs when one varies the architectural choices for the generator.
    - 生成器に対して、アーキテクチャでの選択を変更する [vary] とき、
    - WGAN が、GAN よりはるかにロバストであるということを観測する。

- We illustrate this by running experiments on three generator architectures:
    - 我々はこれを、３つの生成器のアーキテクチャにおいて、実験を実施することによって示す。即ち、

- (1) a convolutional DCGAN generator, 
    - (1) 畳み込み DCGAN での生成器

- (2) a convolutional DCGAN generator without batch normalization and with a constant number of lters, 
    - (2) batc norm なしで、一定数のイテレーション回数の畳み込み DCGAN の生成器

- and (3) a 4-layer ReLU-MLP with 512 hidden units.
    - (3) 512個の隠れ層のユニットをもつ４層の ReLu-MLP

- The last two are known to perform very poorly with GANs.
    - 最後の２つは、GAN で、とても不十分な [poorly] パフォーマンスとなることが知られている。

- We keep the convolutional DCGAN architecture for the WGAN critic or the GAN discriminator.
    - 我々は、WGAN のクリティックや GAN の識別器に対して、畳み込み DCGAN のアーキテクチャを保つ。

<br>

- Figures 5, 6, and 7 show samples generated for these three architectures using both the WGAN and GAN algorithms.
    - 図５，６、７は、WGAN と GAN のアルゴリズムを使用しているこれらの３つのアーキテクチャに対して生成されたサンプルを示している。

- We refer the reader to Appendix F for full sheets of generated samples.
    - 生成されたサンプルのフルシートは、補足 F を参照してください。

- Samples were not cherry-picked.
    - サンプルは、チェリー・ピッキング [cherry-picked] されていない。

> チェリー・ピッキング：数多くの事例の中から自らの論証に有利な事例のみを並べ立てることで、命題を論証しようとする論理上の誤謬、あるいは詭弁術。

- In no experiment did we see evidence of mode collapse for the WGAN algorithm.
    - WGAN のアルゴリズムに対して、モード崩壊の証拠を見せる実験は存在しない。

![image](https://user-images.githubusercontent.com/25688193/56453320-d2c89a00-637b-11e9-89a7-a67ac8479eba.png)<br>

- > Figure 5: Algorithms trained with a DCGAN generator. 
    - DCGAN での生成器で学習されたアルゴリズム（１）

- > Left: WGAN algorithm. Right: standard GAN formulation.
    - 左図：WGAN アルゴリズム。右図：標準の GAN

- > Both algorithms produce high quality samples.
    - 両方のアルゴリズムは、高いクオリティのサンプルを生成している。

<br>

![image](https://user-images.githubusercontent.com/25688193/56453331-04d9fc00-637c-11e9-8cd8-01ab1f140c4a.png)<br>

- > Figure 6: Algorithms trained with a generator without batch normalization and constant number of lters at every layer (as opposed to duplicating them every time as in [18]).
    - > 図６：全ての層で、batc norm なしで、一定数のイテレーション回数の畳み込み DCGAN の生成器で学習されたアルゴリズム（２）
    - （[18] のように、毎回それを複製するのと対称的に、）

- > Aside from taking out batch normalization, the number of parameters is therefore reduced by a bit more than an order of magnitude.
    - > batc norm を除外するだけではなく [Aside from]、
    - > パラメーターの数は、それ故に、１桁分 [an order of magnitude] 以上のビット値で、減少させられる。

- > Left: WGAN algorithm. Right: standard GAN formulation.
    - 左図：WGAN アルゴリズム。右図：標準の GAN

- > As we can see the standard GAN failed to learn while the WGAN still was able to produce samples.
    - > 見られるように、標準の GAN は学習に失敗する。一方で、WGAN は、まだサンプルを生成することが出来る。

<br>

![image](https://user-images.githubusercontent.com/25688193/56453750-a6b01780-6381-11e9-99db-be7fd502d9bf.png)<br>

- > Figure 7: Algorithms trained with an MLP generator with 4 layers and 512 units with ReLU nonlinearities.
    - > 図７：４つの層と512個のユニットを持ち、非線形性の Relu をもつ MLP での生成器で学習されたアルゴリズム（３）

- > The number of parameters is similar to that of a DCGAN, but it lacks a strong inductive bias for image generation.
    - > パラメーターの数は、DCGAN のそれとよく似ている。しかし、画像生成に対しての強い帰納的な [inductive] バイアスが不足している。

- > Left: WGAN algorithm. Right: standard GAN formulation.
    - 左図：WGAN アルゴリズム。右図：標準の GAN

- > The WGAN method still was able to produce samples, lower quality than the DCGAN, and of higher quality than the MLP of the standard GAN.
    - > WGAN は、まだサンプルを生成することができている。
    - > DCGAN のものよりも低いクオリティで、
    - > そして、標準の GAN の MLP でのものよりも、高いクオリティで、

- > Note the signicant degree of mode collapse in the GAN MLP.
    - > MLP での GAN において、モード崩壊の大幅な程度（＝大幅なモード崩壊）に、注意。


# ■ 関連研究（他の手法との違い）

## 5. Related Work


