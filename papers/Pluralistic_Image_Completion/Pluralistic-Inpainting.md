# ■ 論文
- 論文タイトル："Pluralistic Image Completion"
- 論文リンク：https://arxiv.org/abs/1903.04227
- 論文投稿日付：2019/03/11
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- Most image completion methods produce only one result for each masked input, although there may be many reasonable possibilities.
    - ほとんどの画像補完方法では、妥当な可能性は数多くあるにもかかわらず、マスクされた入力ごとに1つの結果しか生成されません。

- In this paper, we present an approach for pluralistic image completion – the task of generating multiple and diverse plausible solutions for image completion.
    - 本論文では、多元的な [pluralistic] 画像補完 [image completion] のためのアプローチを提示する。
    - 即ち、画像補完のための複数の多様なもっともらしい解決策を生成するタスク

- A major challenge faced by learning-based approaches is that usually only one ground truth training instance per label. As such, sampling from conditional VAEs still leads to minimal diversity.
    - 学習ベースのアプローチが直面する大きな課題は、通常、ラベルごとに1つのグラウンドトゥルーストレーニングインスタンスしかないことです。 そういうものとして、条件付きVAEからのサンプリングは依然として最小の多様性につながります。

- To overcome this, we propose a novel and probabilistically principled framework with two parallel paths.
    - これを克服するために、我々は２つの平行な経路を有する新規かつ確率論的原理に基づくフレームワークを提案する。

- One is a reconstructive path that utilizes the only one given ground truth to get prior distribution of missing parts and rebuild the original image from this distribution.
    - 1つは、欠けている部分の事前分布を取得し、この分布から元の画像を再構築するために唯一の与えられた基本的なグラウンドトゥルースを利用する再構成パスです。

- The other is a generative path for which the conditional prior is coupled to the distribution obtained in the reconstructive path. Both are supported by GANs.    
    - もう1つは、条件付き優先順位が再構成パスで取得された分布に結合されている生成パスです。 どちらもGANによってサポートされています。

- We also introduce a new short+long term attention layer that exploits distant relations among decoder and encoder features, improving appearance consistency.
    - また、デコーダとエンコーダの特徴間の遠い [distant] 関係を利用し[exploits]、外観の一貫性を向上させるような、新しい短期+長期のアテンションレイヤを紹介します。 
    
- When tested on datasets with buildings (Paris), faces (CelebA-HQ), and natural images (ImageNet), our method not only generated higher- quality completion results, but also with multiple and diverse plausible outputs.
    - 建物（パリ）、顔（CelebA-HQ）、および自然画像（ImageNet）を含むデータセットでテストした場合、この方法ではより高品質の完成結果が得られるだけでなく、複数の多様な妥当な出力も得られます。

# ■ イントロダクション（何をしたいか？）

## 1. Introduction

- Image completion is a highly subjective process. Supposing you were shown the various images with missing regions in fig. 1, what would you imagine to be occupying these holes?
    - 画像補完は非常に主観的なプロセスです。 図の中に、欠けている領域を含むさまざまな画像が表示されたとします。 1、これらの穴を埋めるために何を想像しますか？

- Bertalmio et al. [4] related how expert conservators would inpaint damaged art by: 1) imagining the semantic content to be filled based on the overall scene; 2) ensuring structural continuity between the masked and unmasked regions; and 3) filling in visually realistic content for missing regions. 
    - Ｂｅｒｔａｌｍｉｏ ｅｔ ａｌ。 [4]専門家の保護者がどのようにして損傷したアートを修復するかを以下のように関連付けた。- 1）シーン全体に基づいてセマンティックコンテンツを埋めることを想像する。
    - ２）マスク領域と非マスク領域との間の構造的連続性を保証する。
    - ３）欠けている領域について視覚的に現実的な内容を記入する。

- Nonetheless, each expert will independently end up creating substantially different details, even if they may universally agree on high-level semantics, such as general placement of eyes on a damaged portrait.
    - それにもかかわらず、たとえ彼らが損なわれた肖像画の上の目の一般的な配置のような高レベルの意味論に普遍的に同意するとしても、それぞれの専門家は独自に実質的に異なる詳細を作成することになります。

---

- Based on this observation, our main goal is thus to generate multiple and diverse plausible results when presented with a masked image — in this paper we refer to this task as pluralistic image completion (depicted in fig. 1). This is as opposed to approaches that attempt to generate only a single “guess” for missing parts.
    - この観察に基づいて、私たちの主な目的は、マスクされた画像と共に提示されたときに複数の多様なもっともらしい結果を生成することです - 本稿では、このタスクを多元的画像完成と呼びます（図1）。 これは、欠けている部分に対して単一の「推測」のみを生成しようとするアプローチとは対照的です。

---

- Early image completion works [4, 7, 5, 8, 3, 13] focus only on steps 2 and 3 above, by assuming that gaps should be filled with similar content to that of the background.
    - 初期の画像補完作業[4、7、5、8、3、13]は、ギャップが背景の内容と同様の内容で埋められるべきであると仮定して、上記のステップ2と3にのみ焦点を当てています。

- Although these approaches produced high-quality texture-consistent images, they cannot capture global semantics and hallucinate new content for large holes. More recently, some learning-based image completion methods [29, 14, 39, 40, 42, 24, 38] were proposed that infer seman tic content (as in step 1).
    - これらのアプローチは質の高い質感に矛盾しない画像を作り出したが、それらは大域的意味論を捉え、大きな穴のための新しい内容を幻覚にすることはできない。 より最近では、（ステップ１のように）意味論的な内容を推論する、いくつかの学習ベースの画像完成方法［２９、１４、３９、４０、４２、２４、３８］が提案された。

- These works treated completion as a conditional generation problem, where the input-to- output mapping is one-to-many. However, these prior works are limited to generate only one “optimal” result, and do not have the capacity to generate a variety of semantically meaningful results.
    - これらの研究は、入力から出力へのマッピングが一対多である条件付き生成問題として完成を扱いました。 しかしながら、これらの先行研究は１つの「最適な」結果のみを生成するように制限されており、そして意味的に意味のある様々な結果を生成する能力を有していない。

---

- To obtain a diverse set of results, some methods utilize conditional variational auto-encoders (CVAE) [34, 37, 2, 10], a conditional extension of VAE [19], which explicitly code a distribution that can be sampled. However, specifically for an image completion scenario, the standard single- path formulation usually leads to grossly underestimating variances. 
    - 多様な結果セットを得るために、いくつかの方法は、サンプリング可能な分布を明示的に符号化する、条件付き変分オートエンコーダ（ＣＶＡＥ）［３４、３７、２、１０］、ＶＡＥの条件付き拡張［１９］を利用する。 しかしながら、特に画像完成シナリオの場合、標準的なシングルパス定式化は通常、著しく過小評価された分散をもたらす。

- This is because when the condition label is itself a partial image, the number of instances in the training data that match each label is typically only one. Hence the estimated conditional distributions tend to have very limited variation since they were trained to reconstruct the single ground truth. This is further elaborated on in section 3.1.
    - これは、条件ラベル自体が部分画像である場合、各ラベルに一致するトレーニングデータ内のインスタンス数は通常1つだけであるためです。 したがって、推定された条件付き分布は、単一のグランドトゥルースを再構築するように訓練されているため、変動が非常に限られている傾向があります。 これについてはセクション3.1でさらに詳しく説明します。

---

- An important insight we will use is that partial images, as a superset of full images, may also be considered as gen- erated from a latent space with smooth prior distributions.
    - 私たちが使用する重要な洞察は、部分画像は、フル画像のスーパーセットとして、滑らかな事前分布を持つ潜在空間から生成されたものと見なすこともできるということです。

- This provides a mechanism for alleviating the problem of having scarce samples per conditional partial image. To do so, we introduce a new image completion network with two parallel but linked training pipelines.
    - これは、条件付き部分画像当たりのサンプルが少ないという問題を軽減するためのメカニズムを提供する。 そうするために、**我々は２つの並列だがリンクされた学習パイプラインを有する新しい画像完成ネットワークを紹介する。**

- The first pipeline is a VAE-based reconstructive path that not only utilizes the full instance ground truth (i.e. both the visible partial image, as well as its complement — the hidden partial image), but also imposes smooth priors for the latent space of complement regions.
    - **最初のパイプラインは、フルインスタンスのグランドトゥルース（つまり、可視部分画像とその補数 - 隠れ部分画像の両方）を利用するだけでなく、補空間 [complement regions]の潜在空間に滑らかな事前分布？を課すVAEベースの再構成パスです。**

- The second pipeline is a generative path that predicts the latent prior distribution for the missing regions conditioned on the visible pixels, from which can be sampled to generate diverse results.
    - **第２のパイプラインは、可視ピクセル上に条件付けられた欠けている領域に対する潜在的な事前分布を予測する生成経路であり、そこからサンプリングして多様な結果を生成することができる。**

- The training process for the latter path does not attempt to steer the output towards reconstructing the instance-specific hidden pixels at all, instead allowing the reasonableness of results be driven by an auxiliary discriminator network [11].
    - 後者の経路のための学習プロセスは、インスタンス固有の隠れピクセルの再構築に向けて出力を誘導する（＝舵を取る） [steer] ことを全く試みず、
    - 代わりに、結果の妥当性を補助識別器ネットワークによって推進することを可能にする[11]。

- This leads to substantially great variability in content generation. 
    - これはコンテンツ生成における実質的に大きな変動性をもたらす。
    
- We also introduce an enhanced short+long term attention layer that significantly increases the quality of our results.
    - **また、結果の質を大幅に向上させる、強化された短期+長期のアテンション層を紹介します。**

---

- We compared our method with existing state-of-the-art approaches on multiple datasets. Not only can higher- quality completion results be generated using our approach, it also presents multiple diverse solutions.
    - 我々は我々の方法を複数のデータセットに関する既存の最先端のアプローチと比較した。 私たちのアプローチを使用してより高品質の完成結果を生み出すことができるだけでなく、それはまた複数の多様な解決策を提示します。

---

- The main contributions of this work are:

1. A probabilistically principled framework for image completion that is able to maintain much higher sample diversity as compared to existing methods;
    - 既存の方法と比較してはるかに高いサンプル多様性を維持することができる、画像完成のための確率論的原理フレームワーク。

1. A new network structure with two parallel training paths, which trades off between reconstructing the original training data (with loss of diversity) and maintaining the variance of the conditional distribution;
    - **元のトレーニングデータの再構築（ダイバーシティの損失あり）と条件付き分布の分散の維持との間でトレードオフする、2つの並列トレーニングパスを持つ新しいネットワーク構造。**

1. A novel self-attention layer that exploits short+long term context information to ensure appearance consistency in the image domain, in a manner superior to purely using GANs; and
    - **純粋にGANを使用するよりも優れた方法で、短期および長期のコンテキスト情報を活用して画像ドメインの外観の一貫性を確保する、新しい自己注意力層。**

1. We demonstrate that our method is able to complete the same mask with multiple plausible results that have substantial diversity, such as those shown in figure 1.
    - 私達は私達の方法が図1に示すような実質的な多様性を持つ複数のもっともらしい結果で同じマスクを完成することができることを示します。

# ■ 結論

## x. 論文の項目名 (Conclusion)


# ■ 何をしたか？詳細

## 3. Approach

- Suppose we have an image, originally Ig, but degraded by a number of missing pixels to become Im (the masked partial image) comprising the observed / visible pixels.
    - 元々Igであったが、観測された/可視のピクセルを含んでいる [comprising] Im（マスクされた部分的な画像）になるために、欠けているピクセルの数だけ劣化した画像があるとします。

- We also define Ic as its complement partial image comprising the ground truth hidden pixels.
    - また、グラウンドトゥルース隠れピクセルを含む補完部分画像としてIcを定義します。

- Classical image completion methods attempt to reconstruct the ground truth unmasked image Ig in a deterministic fashion from Im (see fig. 2 “Deterministic”).
    - 古典的な画像補完方法は、Imから決定論的な方法でグラウンドトルースマスクされていない画像Igを再構築しようとします（図2「決定論的」を参照）。

- This results in only a single solution. In contrast, our goal is to sample from p(Ic|Im).
    - これは唯一の解決策になります。 対照的に、私たちの目標はpからサンプリングすることです（Ic | Im）。

### 3.1. Probabilistic Framework

- In order to have a distribution to sample from, a current approach is to employ the CVAE [34] which estimates a parametric distribution over a latent space, from which sampling is possible (see fig. 2 “CVAE”). This involves a variational lower bound of the conditional log-likelihood of observing the training instances:
    - **サンプリング元の分布を得るために、現在のアプローチは、サンプリングが可能である潜在空間にわたるパラメトリック分布を推定するＣＶＡＥ ［３４］を使用することである（図２「ＣＶＡＥ」参照）。** これは、トレーニングインスタンスを観察する条件付き対数尤度の変分的下限を含みます。

> Conditional Variational Auto Encoder（CVAE） :VAEに対して正解ラベルを付与して学習を行います。

![image](https://user-images.githubusercontent.com/25688193/61367030-c940c680-a8c5-11e9-9584-92ad2c708931.png)

- where zc is the latent vector, qψ(·|·) the posterior importance sampling function, pφ(·|·) the conditional prior, pθ(·|·) the likelihood, with ψ, φ and θ being the deep network parameters of their corresponding functions. 
    - ここで、zcは潜在ベクトル、qψ（・|・）は事後重要度サンプリング関数、pφ（・|・）は条件付き事前確率、pθ（・|・）は尤度、ψ、φ、θはディープネットワークです。 対応する関数のパラメータ 

- This lower bound is maximized w.r.t. all parameters.
    - この下限は、最大化されている。 すべてのパラメータ

---

- For our purposes, the chief difficulty of using CVAE [34] directly is that the high DoF networks of qψ (·|·) and pφ (·|·) are not easily separable in (1) with the KL distance easily driven towards zero, and is approximately equivalent to maximizing Epφ(zc|Im)[logpθ(Ic|zc,Im)] (the “GSNN” variant in [34]).
    - **我々の目的のために、CVAE [34]を直接使用することの主な困難は、**
    - **qψ（・|・）とpφ（・|・）の高DoFネットワークが、容易にゼロに追いやられるKL距離で、簡単に (1) 式で分離できないことです。** 
    - **これは、Epφ（zc | Im）[logpθ（Ic | zc、Im）]（[34]の "GSNN"の変形）を最大化することとほぼ等価である。**

- This consequently learns a delta-like prior of p_φ(zc|Im) → δ(zc − z_c^*), where z∗c is the maximum latent likelihood point of pθ (Ic |·, Im ).
    - **その結果、これはデルタ状の事前確率 p_φ（zc | Im）→δ（z_c  -  z_c^*）を学習します。ここで、z * cはpθの最大潜在尤度点（Ic |・、Im）です。**

- While this low variance prior may be useful in estimating a single solution, sampling from it will lead to negligible diversity in image completion results (as seen in fig. 9).
    - **この低分散事前分布は単一の解を推定するのに有用であり得るが、それからサンプリングすることは画像完成結果において無視できるほどの多様性をもたらすであろう（図９に見られるように）。**

- When the CVAE variant of [37], which has a fixed latent prior, is used instead, the network learns to ignore the latent sampling and directly estimates Ic from Im, also resulting in a single solution.
    - **代わりに固定された潜在事前確率を持つ[37]のCVAE変種が使われるとき、ネットワークは潜在的なサンプリングを無視することを学び、Imから直接Icを推定し、やはり単一の解をもたらします。**

- This is due to the image completion scenario when there is only one training instance per condition label, which is a partial image Im. Details are in the supplemental section B.1.
    - これは、条件ラベルごとに部分的な画像Ｉｍである訓練インスタンスが１つしかないときの画像完成シナリオによるものである。 詳細は補足のセクションB.1にあります。

---

- A possible way to diversify the output is to simply not incentivize the output to reconstruct the instance-specific Ig during training, only needing it to fit in with the training set distribution as deemed by an learned adversarial discriminator (see fig. 2 “Instance Blind”). 
    - 出力を多様化するための可能な方法は、学習中のインスタンス固有の I_g を再構築するために出力を単純に動機づけ [incentivize] しないことであり、
    - 学習された敵対的識別器によって見なされる [deemed] 場合は？、学習データセット分布に適合することのみを必要とする（図2 の“Instance Blind”参照）。
    
- However, this approach is unstable, especially for large and complex scenes [35].
    - しかしながら、このアプローチは、特に大きくて複雑なシーンでは不安定です[35]。

##### B. Mathematical Derivation and Analysis

###### B.1.3 Unconstrained Learning of the Conditional Prior

- Assuming that there is a unique global maximum for log p_φ(zc | Im), the bound achieves equality when the conditional prior becomes a Dirac delta function centered at the maximum latent likelihood point
    - logp_φ（zc | Im）に一意の大域的最大値があると仮定すると、条件付き事前確率が最大潜在尤度点を中心とするディラックデルタ関数になると、範囲は等式になります。

- Intuitively, subject to the vagaries of stochastic gradient descent, the network for p_φ(zc | Im) without further constraints will learn a narrow delta-like function that sifts out maximum latent likelihood value of log p_φ(Ic | zc; Im).
    - 直感的には、確率的勾配降下の変動に応じて、それ以上制約を受けないp_φ（zc | Im）のネットワークは、logp_φ（Ic | zc; Im）の最大潜在尤度値を除外する狭いデルタ様関数を学習する。
    
- As mentioned in section 3.1, although this narrow conditional prior may be helpful in estimating a single solution for Ic given Im during testing during testing, this is poor for sampling a diversity of solutions.
    - セクション3.1で述べたように、この狭い条件付き事前条件は、テスト中のテスト中にImが与えられた場合のIcの単一解を推定するのに役立ちますが、これは多様な解をサンプリングするには不十分です。

- In our framework, the (unconditional) latent priors are imposed for the partial images themselves, which prevent this delta function degeneracy.
    - 我々のフレームワークでは、（無条件の）潜在的な前置詞が部分画像自体に課され、それがこのデルタ関数の縮退を防ぎます。



#### Latent Priors of Holes

- In our approach, we require that missing partial images, as a superset of full images, to also arise from a latent space distribution, with a smooth prior of p(zc).
    - 我々のアプローチでは、完全画像のスーパーセットとしての欠けている部分画像も、滑らかな事前確率p（zc）をもつ潜在空間分布から生じることを要求する。 

- The variational lower bound is:    
    - 変分下限は次のとおりです。

![image](https://user-images.githubusercontent.com/25688193/61367091-eaa1b280-a8c5-11e9-8ad1-0cd0cf8d0d33.png)

- where in [19] the prior is set as p(z_c) = N (0, I). 
    - ここで、[19]では、事前確率はp（zc）= N（0、I）として設定されています。

- However, we can be more discerning when it comes to partial images since they have different numbers of pixels.
    - ただし、部分画像はピクセル数が異なるため、部分画像の場合はより見やすくなります。

- A missing partial image zc with more pixels (larger holes) should have greater latent prior variance than a missing partial image zc with fewer pixels (smaller holes).
    - より多くの画素（より大きな穴）を有する欠けている部分画像ｚｃは、より少ない画素（より小さな穴）を有する欠けている部分画像ｚｃよりも大きい潜在的事前分散を有するべきである。

- Hence we generalize the prior p(z_c) = N_m(0, σ2(n)I) to adapt to the number of pixels n.
    - それ故、我々は、ピクセル数ｎに適応するために、事前のｐ（ｚｃ）＝ Ｎｍ（０、σ２（ｎ）Ｉ）を一般化する。


#### Prior-Conditional Coupling

- Next, we combine the latent priors into the conditional lower bound of (1).
    - 次に、潜在プライアを（1）の条件付き下限に結合します。 

- This can be done by assuming zc is much more closely related to Ic than to Im, so qψ(zc|Ic, Im)≈qψ(zc|Ic).
    - これは、zcがImよりもIcにはるかに密接に関連していると仮定することによって行うことができます。したがって、qψ（zc | Ic、Im）≒qψ（zc | Ic）です。

![image](https://user-images.githubusercontent.com/25688193/61367149-0442fa00-a8c6-11e9-9f71-ba2976ad7226.png)

- Updating (1):
    - (1) 式を更新

- However, unlike in (1), notice that qψ(zc|Ic) is no longer freely learned during training, but is tied to its presence in (2).
    - ただし、（1）とは異なり、qψ（zc | Ic）はトレーニング中に自由に学習されなくなりましたが、（2）ではその存在に関連付けられています。

- Intuitively, the learning of qψ(zc|Ic) is regularized by the prior p(zc) in (2), while the learning of the conditional prior pφ(zc|Im) is in turn regularized by qψ(zc|Ic) in (3).
    - 直感的には、ｑψ（ｚｃ ｜ Ｉｃ）の学習は（２）における事前ｐ（ｚｃ）によって正則化され、条件付き事前ｐφ（ｚｃ ｜ Ｉｍ）の学習は順にｑψ（ｚｃ ｜ Ｉｃ）によって正則化される。 （3）

#### Reconstruction vs Creative Generation

- One issue with (3) is that the sampling is taken from qψ(zc|Ic) during training, but is not available during testing, whereupon sampling must come from pφ(zc|Im) which may not be adequately learned for this role.
    - （３）に関する１つの問題は、サンプリングがトレーニング中にｑψ（ｚｃ ｜ Ｉｃ）から取られるが、試験中には利用できないことであり、サンプリングはｐφ（ｚｃ ｜ Ｉｍ）から来なければならない。

- In order to mitigate this problem, we modify (3) to have a blend of formulations with and without importance sampling. So, with simplified notation:
    - この問題を軽減するために、我々は（３）を修正して、重要度サンプリングを伴う、また伴わない定式化のブレンドを持つようにする。 そのため、表記を簡略化して、

![image](https://user-images.githubusercontent.com/25688193/61367194-17ee6080-a8c6-11e9-9ec2-43cf6825e2fb.png)

- where 0 ≤ λ ≤ 1 is implicitly set by training loss coefficients in section 3.3. When sampling from the importance function qψ (·|Ic ), the full training instance is available and we formulate the likelihood prθ (Ic |zc , Im ) to be focused on reconstructing Ic.
    - ここで、0≦λ≦1はセクション3.3の学習損失係数によって暗黙的に設定されます。
    - 重要度関数ｑψ（・｜ Ｉｃ）からサンプリングすると、完全な訓練事例が利用可能であり、Ｉｃの再構築に焦点を合わせるために尤度ｐｒθ（Ｉｃ ｜ ｚｃ、Ｉｍ）を定式化する。

- Conversely, when sampling from the learned conditional prior pφ (·|Im ) which does not contain Ic, we facilitate creative generation by having the likelihood model pgθ (Ic |zc , Im ) ∼= lgθ (zc , Im ) be independent of the original instance of Ic.
    - 逆に、Icを含まない学習された条件付き事前確率pφ（・| Im）からサンプリングするとき、我々は尤度モデルpgθ（Ic | zc、Im）〜=lgθ（zc、Im）を独立にすることによって創造的生成を促進する。 Icの元のインスタンス。

- Instead it only encourages generated samples to fit in with the overall training distribution.
    - 代わりに、生成されたサンプルが全体のトレーニング分布に適合するように奨励するだけです。

---

- Our overall training objective may then be expressed as jointly maximizing the lower bounds in (2) and (4), with the likelihood in (2) unified to that in (4) as pθ (Ic |zc ) ∼= prθ(Ic|zc,Im).
    - そして、我々の全体的な訓練目的は、（２）と（４）の下限を（４）のものに統一して、（２）と（４）の下限を共同で最大化するように表現される。 zc、Im）。
    
- See the supplemental section B.2.
    - 補足セクションB.2を参照してください。


### 3.2. Dual Pipeline Network Structure

- > Figure 3. Overview of our architecture with two parallel pipelines.

- > The reconstructive pipeline (yellow line) combines information from Im and Ic, which is used only for training.

- > The generative pipeline (blue line) infers the conditional distribution of hidden regions, that can be sampled during testing. 

- > Both representation and generation networks share identical weights.

---

- This formulation is implemented as our dual pipeline framework, shown in fig. 3. It consists of two paths: the upper reconstructive path uses information from the whole image, i.e. Ig={Ic,Im}, while the lower generative path only uses information from visible regions Im. Both representation and generation networks share identical weights. Specifically:
    - この定式化は、図3に示すように、私たちのデュアルパイプラインフレームワークとして実装されています。
    - それは２つの経路からなる：
    - 即ち、上側の再構成経路は画像全体からの情報、すなわちＩｇ ＝ ｛Ｉｃ、Ｉｍ｝を使用し、一方下側の生成経路は可視領域Ｉｍからの情報のみを使用する。 表現ネットワークと生成ネットワークは同じ重みを共有します。 具体的には：

- For the upper reconstructive path,the complement partial image Ic is used to infer the importance function qψ (·|Ic )=Nψ (·) during training. The sampled latent vector zc thus contains information of the missing regions, while the conditional feature fm encodes the information of the visible regions. Since there is sufficient information, the loss function in this path is geared towards reconstructing the original image Ig .
    - 上側再構成経路については、補完部分画像Ｉｃを使用して、トレーニング中に重要度関数ｑψ（・｜ Ｉｃ）＝ Ｎψ（・）を推論する。 したがって、サンプリングされた潜在ベクトル z_c は、欠けている領域の情報を含み、一方、条件付き特徴ｆ ｍは、可視領域の情報を符号化する。 十分な情報があるので、この経路における損失関数は原画像Ｉｇを再構成することに向けられる。

- For the lower generative path, which is also the test path, the latent distribution of the holes Ic is inferred based only on the visible Im. This would be significantly less accurate than the inference in the upper path. Thus the reconstruction loss is only targeted at the visible regions Im (via fm).
    - 生成経路でもあるより低い生成経路については、ホール画像Ｉｃの潜在的分布は可視Ｉｍのみに基づいて推定される。 これは、上位パスでの推論よりもかなり正確度が低くなります。 したがって、再構成損失は可視領域Ｉｍのみを対象とする（ｆｍを介して）。

- In addition, we also utilize adversarial learning net- works on both paths, which ideally ensure that the full synthesized data fit in with the training set distribution, and empirically leads to higher quality images.
    - **さらに、我々はまた、完全に合成されたデータがトレーニングセットの分布に適合することを理想的に保証し、経験的により高品質の画像をもたらすことを理想的に保証する両方の経路で敵対学習ネットワークを利用する。**

### 3.3. Training Loss

- Various terms in (2) and (4) may be more conventionally expressed as loss functions. Jointly maximizing the lower bounds is then minimizing a total loss L, which consists of three groups of component losses:
    - （２）および（４）における様々な用語は、より慣例的に損失関数として表現され得る。 下限を共同で最大化することは、3つのグループの成分損失からなる総損失Lを最小化することです。

- where the LKL group regularizes consistency between pairs of distributions in terms of KL divergences, the Lapp group encourages appearance matching fidelity, and while the Lad group forces sampled images to fit in with the training set distribution. Each of the groups has a separate term for the reconstructive and generative paths.
    - L_KL 項がKLダイバージェンスに関して分布のペア間の一貫性を正規化する場合、L_app 項は見た目の一致の忠実度を奨励し、Ladグループはサンプリングされた画像をトレーニングセットの分布に合わせるよう強制します。 各項には、再構成パスと生成パスに別々の用語があります。

#### Distributive Regularization

- The typical interpretation of the KL divergence term in a VAE is that it regularizes the learned importance sampling function qψ(·|Ic) to a fixed latent prior p(z_c). Defining as Gaussians, we get:
    - VAEにおけるKL発散項の典型的な解釈は、学習された重要度サンプリング関数qψ（・| Ic）を固定ラント前の p(z_c)に正規化することです。 ガウシアンとして定義すると、次のようになります。

- For the generative path, the appropriate interpretation is reversed: the learned conditional prior pφ(·|Im), also a Gaussian, is regularized to qψ(·|Ic).
    - 生成経路の場合、適切な解釈が逆になります。学習された条件付き事前確率pφ（・| Im）はガウス分布でもあるが、qψ（・| Ic）に正規化されます。

#### Appearance Matching Loss

- The likelihood term prθ (Ic |zc , Im ) may be interpreted as probabilistically encouraging appearance matching to the hidden Ic. However, our framework also auto-encodes the visible Im deterministically, and the loss function needs to cater for this reconstruction. As such, the per-instance loss here is:
    - 尤度項ｐｒθ（Ｉｃ ｜ ｚｃ、Ｉｍ）は、隠れたＩｃに見合った確率的に見栄えのよい外観として解釈することができる。 しかしながら、我々のフレームワークはまた可視的Imを決定論的に自動符号化し、そして損失関数はこの再構成を考慮に入れる必要がある。 そのため、ここでのインスタンスごとの損失は次のとおりです。

- where Irec =G(zc , fm ) and Ig are the reconstructed and original full images respectively. In contrast, for the generative path we ignore instance-specific appearance matching for Ic, and only focus on reconstructing Im (via fm):

### 3.4. Short+Long Term Attention

- Extending beyond the Self-Attention GAN [43], we propose not only to use the self-attention map within a decoder layer to harness distant spatial context, but also to further capture feature-feature context between encoder and decoder layers.
    - Self-Attention GAN [43]を超えて、我々は、遠い空間的なコンテキストを利用するために、デコーダ層内でセルフアテンションマップを使用するだけでなく、エンコーダとデコーダレイヤ間の特徴 - 特徴コンテキストをさらに捉えることを提案する。

- Our key novel insight is: doing so would allow the network a choice of attending to the finer-grained features in the encoder or the more semantically generative features in the decoder, depending on circumstances.
    - 私たちの新しい斬新な洞察は、状況に応じて、エンコーダーのよりきめの細かい機能またはデコーダーのより意味的に生成的な機能にネットワークを選択することを可能にするでしょう。

---

- Our proposed structure is shown in fig. 4. We first calculate the self-attention map from the features f_d of a decoder middle layer, using the attention score of:
    - 提案した構造を図4に示します。まず、次のアテンションスコアを使用して、デコーダの中間層の特徴f_dから自己アテンションマップを計算します。

---

- N is the number of pixels, Q(fd )=Wq fd , and Wq is a 1x1 convolution filter. This leads to the short-term intra-layer attention feature (self-attention in fig. 4) and the output yd:
    - Ｎ は画素数であり、Ｑ（ｆｄ）＝ Ｗｑｆｄ であり、Ｗｑは１×１畳み込みフィルタである。 これにより、短期間のレイヤ内アテンション機能（図4の自己アテンション）と出力ydが得られます。

- where, following [43], we use a scale parameter γd to balance the weights between cd and fd. The initial value of γd is set to zero.
    - ここで、[43]に従い、スケールパラメータγdを使用して、cdとfdの間の重みのバランスをとります。 γdの初期値はゼロに設定されています。

- In addition, for attending to features fe from an encoder layer, we have a long-term inter-layer attention feature (contextual flow in fig. 4) and the output ye:
    - さらに、エンコーダ層からの機能 f_e に注意を向けるために、長期的なレイヤ間アテンション機能（図4のコンテキストフロー）と出力 y_e があります。

- As before, a scale parameter γe is used to combine the encoder feature fe and the attention feature ce. However, unlike the decoder feature fd which has information for generating a full image, the encoder feature fe only represents visible parts Im.
    - 前述のように、スケールパラメータγｅを使用して、エンコーダ特徴ｆｅと注意特徴ｃｅを組み合わせる。    
    - しかしながら、フル画像を生成するための情報を有するデコーダ特徴ｆｄとは異なり、エンコーダ特徴ｆｅは可視部分Ｉｍのみを表す。
    
- Hence, a binary mask M (holes=0) is used. Finally, both the short and long term attention features are aggregated and fed into further decoder layers.
    - それ故、バイナリマスクＭ（ホール＝ ０）が使用される。 最後に、短期と長期の両方の注意機能が集約され、さらに別のデコーダレイヤに入力されます。

# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


