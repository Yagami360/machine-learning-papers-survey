# ■ 論文
- 論文タイトル："β-VAE: LEARNING BASIC VISUAL CONCEPTS WITH A CONSTRAINED VARIATIONAL FRAMEWORK"
- 論文リンク：https://openreview.net/forum?id=Sy2fzU9gl
- 論文投稿日付：
- 被引用数（記事作成時点）：xxx 件
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- Learning an interpretable factorised representation of the independent data generative factors of the world without supervision is an important precursor for the development of artificial intelligence that is able to learn and reason in the same way that humans do. We introduce β-VAE, a new state-of-the-art framework for automated discovery of interpretable factorised latent representations from raw image data in a completely unsupervised manner.
    - 監督なしで世界の独立したデータ生成因子の解釈可能な因数分解表現を学習することは、人間と同じ方法で学習し推論することができる人工知能の開発の重要な先駆 [precursor] です。
    - 完全に教師なしの方法で、生の画像データから解釈可能な因数分解された潜在的表現の自動発見のための新しい最先端のフレームワークであるβ-VAEを導入します。

- Our approach is a modification of the variational autoencoder (VAE) framework. We introduce an adjustable hyperparameter β that balances latent channel capacity and independence constraints with reconstruction accuracy. We demonstrate that β-VAE with appropriately tuned β > 1 qualitatively outperforms VAE (β = 1), as well as state of the art unsupervised (InfoGAN) and semi-supervised (DC-IGN) approaches to disentangled factor learning on a variety of datasets (celebA, faces and chairs). Furthermore, we devise a protocol to quantitatively compare the degree of disentanglement learnt by different models, and show that our approach also significantly outperforms all baselines quantitatively.
    - 私たちのアプローチは、変分オートエンコーダー（VAE）フレームワークの修正です。潜在的なチャネル容量と独立性の制約と再構築の精度のバランスをとる調整可能なハイパーパラメーターβを導入します。適切に調整されたβ> 1のβ-VAEは、VAE（β= 1）を上回り、さまざまな因子のもつれを解く因子学習への最先端の教師なし（InfoGAN）および半教師あり（DC-IGN）アプローチを示します。データセット（celebA、顔、椅子）。さらに、異なるモデルによって学習されたもつれ解除の程度を定量的に比較するプロトコルを考案し、このアプローチがすべてのベースラインを定量的にも大幅に上回ることを示します。
    
- Unlike InfoGAN, β-VAE is stable to train, makes few assumptions about the data and relies on tuning a single hyperparameter β, which can be directly optimised through a hyperparameter search using weakly labelled data or through heuristic visual inspection for purely unsupervised data.
    - InfoGANとは異なり、β-VAEはトレーニングに対して安定しており、データに関する仮定をほとんど行わず、単一のハイパーパラメーターβの調整に依存します。これは、弱標識データを使用したハイパーパラメーター検索または純粋に監視されていないデータのヒューリスティックな視覚検査によって直接最適化できます。

# ■ イントロダクション（何をしたいか？）

## x. Introduction

- The difficulty of learning a task for a given machine learning approach can vary significantly depending on the choice of the data representation. Having a representation that is well suited to the particular task and data domain can significantly improve the learning success and robustness of the chosen model (Bengio et al., 2013). It has been suggested that learning a disentangled representation of the generative factors in the data can be useful for a large variety of tasks and domains (Bengio et al., 2013; Ridgeway, 2016). A disentangled representation can be defined as one where single latent units are sensitive to changes in single generative factors, while being relatively invariant to changes in other factors (Bengio et al., 2013). For example, a model trained on a dataset of 3D objects might learn independent latent units sensitive to single independent data generative factors, such as object identity, position, scale, lighting or colour, thus acting as an inverse graphics model (Kulkarni et al., 2015).
- In a disentangled representation, knowledge about one factor can generalise to novel configurations of other factors. According to Lake et al. (2016), disentangled representations could boost the performance of state-of-the-art AI approaches in situations where they still struggle but where humans excel.
- Such scenarios include those which require knowledge transfer, where faster learning is achieved by reusing learnt representations for numerous tasks; zero-shot inference, where reasoning about new data is enabled by recombining previously learnt factors; or novelty detection.
    - 特定の機械学習アプローチのタスクを学習する難しさは、データ表現の選択によって大きく異なります。特定のタスクおよびデータドメインに適した表現を使用すると、選択したモデルの学習の成功と堅牢性を大幅に向上させることができます（Bengio et al。、2013）。データの生成因子のもつれのない表現を学習することは、さまざまなタスクとドメインに役立つことが示唆されています（Bengio et al。、2013; Ridgeway、2016）。もつれのない表現は、単一の潜在ユニットが単一の生成因子の変化に敏感であり、他の因子の変化には比較的不変であるものとして定義できます（Bengio et al。、2013）。たとえば、3Dオブジェクトのデータセットでトレーニングされたモデルは、オブジェクトのアイデンティティ、位置、スケール、照明、色などの単一の独立したデータ生成要因に敏感な独立した潜在ユニットを学習し、逆グラフィックモデルとして機能します（Kulkarni et al。 、2015）。
    - 解きほぐされた表現では、1つの要因に関する知識を他の要因の新しい構成に一般化できます。レイクらによると（2016）、解きほぐされた表現は、依然として苦労しているが人間は優れている状況で、最先端のAIアプローチのパフォーマンスを向上させる可能性があります。
    - このようなシナリオには、知識の伝達を必要とするシナリオが含まれます。そこでは、学習した表現を多数のタスクに再利用することにより、より高速な学習が実現されます。ゼロショット推論。以前に学習した要素を再結合することにより、新しいデータに関する推論が可能になります。またはノベルティ検出。

---

- Unsupervised learning of a disentangled posterior distribution over the underlying generative factors of sensory data is a major challenge in AI research (Bengio et al., 2013; Lake et al., 2016). Most previous attempts required a priori knowledge of the number and/or nature of the data generative factors (Hinton et al., 2011; Rippel & Adams, 2013; Reed et al., 2014; Zhu et al., 2014; Yang et al., 2015; Goroshin et al., 2015; Kulkarni et al., 2015; Cheung et al., 2015; Whitney et al., 2016; Karaletsos et al., 2016). This is not always feasible in the real world, where the newly initialised learner may be exposed to complex data where no a priori knowledge of the generative factors exists, and little to no supervision for discovering the factors is available. Until recently purely unsupervised approaches to disentangled factor learning have not scaled well (Schmidhuber, 1992; Desjardins et al., 2012; Tang et al., 2013; Cohen & Welling, 2014; 2015).
    - 感覚データの根底にある生成因子に対する解きほぐされた事後分布の教師なし学習は、AI研究における主要な課題です。 これまでのほとんどの試みでは、データ生成因子の数および/または性質の事前知識が必要でした。 これは、現実の世界では常に実現可能ではありません。新しく初期化された学習者は、生成因子の事前知識が存在せず、因子を発見するための教師データがほとんどまたはまったくない複雑なデータにさらされる可能性があります。 解きほぐされた因子学習に対する純粋に教師なしのアプローチは、最近までうまくスケーリングされていません（Schmidhuber、1992; Desjardins et al。、2012; Tang et al。、2013; Cohen＆Welling、2014; 2015）。

---

- Recently a scalable unsupervised approach for disentangled factor learning has been developed, called InfoGAN (Chen et al., 2016). InfoGAN extends the generative adversarial network (GAN) (Goodfellow et al., 2014) framework to additionally maximise the mutual information between a subset of the generating noise variables and the output of a recognition network. It has been reported to be capable of discovering at least a subset of data generative factors and of learning a disentangled representation of these factors. The reliance of InfoGAN on the GAN framework, however, comes at the cost of training instability and reduced sample diversity. Furthermore, InfoGAN requires some a priori knowledge of the data, since its performance is sensitive to the choice of the prior distribution and the number of the regularised noise variables. InfoGAN also lacks a principled inference network (although the recognition network can be used as one). The ability to infer the posterior latent distribution from sensory input is important when using the unsupervised model in transfer learning or zero-shot inference scenarios. Hence, while InfoGAN is an important step in the right direction, we believe that further improvements are necessary to achieve a principled way of using unsupervised learning for developing more human-like learning and reasoning in algorithms as described by Lake et al. (2016).
    - 最近、InfoGANと呼ばれる、解きほぐされた因子学習のためのスケーラブルな教師なしアプローチが開発されました（Chen et al。、2016）。 InfoGANは、生成的敵対ネットワーク（GAN）（Goodfellow et al。、2014）フレームワークを拡張して、生成するノイズ変数のサブセットと認識ネットワークの出力間の相互情報をさらに最大化します。
    - データ生成因子の少なくともサブセットを発見し、これらの因子のもつれのない表現を学習できることが報告されています。ただし、GANフレームワークへのInfoGANの依存は、トレーニングの不安定性とサンプルの多様性の低下という犠牲を伴います。さらに、InfoGANのパフォーマンスは事前分布の選択と正規化されたノイズ変数の数に影響されるため、データの事前知識が必要です。 InfoGANには原則的な推論ネットワークもありません（ただし、認識ネットワークは1つとして使用できます）。伝達学習またはゼロショット推論シナリオで教師なしモデルを使用する場合、感覚入力から事後潜在分布を推測する能力は重要です。したがって、InfoGANは正しい方向への重要なステップですが、Lake et al。に記載されているアルゴリズムでより人間に似た学習と推論を開発するための教師なし学習を使用する原則的な方法を実現するには、さらなる改善が必要であると考えています。 （2016）。

---

- Finally, there is currently no general method for quantifying the degree of learnt disentanglement. Therefore there is no way to quantitatively compare the degree of disentanglement achieved by different models or when optimising the hyperparameters of a single model.
    - 最後に、現在、学習したもつれ解除の程度を定量化する一般的な方法はありません。 したがって、異なるモデルによって達成された、または単一のモデルのハイパーパラメーターを最適化するときに達成されるもつれの程度を定量的に比較する方法はありません。

---

- In this paper we attempt to address these issues. We propose β-VAE, a deep unsupervised generative approach for disentangled factor learning that can automatically discover the independent latent factors of variation in unsupervised data. Our approach is based on the variational autoencoder (VAE) framework (Kingma & Welling, 2014; Rezende et al., 2014), which brings scalability and training stability. While the original VAE work has been shown to achieve limited disentangling performance on simple datasets, such as FreyFaces or MNIST (Kingma & Welling, 2014), disentangling performance does not scale to more complex datasets (e.g. Aubry et al., 2014; Paysan et al., 2009; Liu et al., 2015), prompting the development of more elaborate semi-supervised VAE-based approaches for learning disentangled factors (e.g. Kulkarni et al., 2015; Karaletsos et al., 2016).
    - このペーパーでは、これらの問題に対処しようとします。 教師なしデータの変動の独立した潜在的要因を自動的に発見できる、解きほぐされた因子学習のための深い教師なし生成アプローチであるβ-VAEを提案します。 私たちのアプローチは、変分オートエンコーダ（VAE）フレームワーク（Kingma＆Welling、2014; Rezende et al。、2014）に基づいており、スケーラビリティとトレーニングの安定性をもたらします。 オリジナルのVAEの作業は、FreyFacesやMNISTなどの単純なデータセットでの限られた解きほぐしパフォーマンスを達成することが示されています（Kingma＆Welling、2014）。 al。、2009; Liu et al。、2015）、解きほぐされた因子を学習するためのより精巧な半教師付きVAEベースのアプローチの開発を促しています（例：Kulkarni et al。、2015; Karaletsos et al。、2016）。

---

- We propose augmenting the original VAE framework with a single hyperparameter β that modulates the learning constraints applied to the model. These constraints impose a limit on the capacity of the latent information channel and control the emphasis on learning statistically independent latent factors. β-VAE with β = 1 corresponds to the original VAE framework (Kingma & Welling, 2014; Rezende et al., 2014).
    - モデルに適用された学習制約を調整する単一のハイパーパラメーターβを使用して、元のVAEフレームワークを拡張することを提案します。これらの制約は、潜在情報チャネルの容量に制限を課し、統計的に独立した潜在因子の学習の強調を制御します。 β= 1のβ-VAEは、元のVAEフレームワークに対応しています（Kingma＆Welling、2014; Rezende et al。、2014）。

- With β > 1 the model is pushed to learn a more efficient latent representation of the data, which is disentangled if the data contains at least some underlying factors of variation that are independent.
    - β> 1の場合は、データに少なくともいくつかの独立した変動要因が含まれていれば、disentangled されたデータのより効率的な潜在表現を学習するためにモデルがプッシュされます。

- We show that this simple modification allows β-VAE to significantly improve the degree of disentanglement in learnt latent representations compared to the unmodified VAE framework (Kingma & Welling, 2014; Rezende et al., 2014).
    - この単純な変更により、β-VAEは、変更されていないVAEフレームワークと比較して、学習した潜在表現のもつれの程度を大幅に改善できることを示します（Kingma＆Welling、2014; Rezende et al。、2014）。

- Furthermore, we show that β-VAE achieves state of the art disentangling performance against both the best unsupervised (InfoGAN: Chen et al., 2016) and semi-supervised (DC-IGN: Kulkarni et al., 2015) approaches for disentangled factor learning on a number of benchmark datasets, such as CelebA (Liu et al., 2015), chairs (Aubry et al., 2014) and faces (Paysan et al., 2009) using qualitative evaluation.
    - さらに、β-VAEは、CelebA、椅子、顔などの多くのベンチマークデータセットでの質的評価において、解く因子を学習するための最良の教師なしInfoGANおよび半教師ありDC-IGNアプローチの両方に対して、最先端の解きほぐしパフォーマンスを達成することを示します。

- Finally, to help quantify the differences, we develop a new measure of disentanglement and show that β-VAE significantly outperforms all our baselines on this measure (ICA, PCA, VAE Kingma & Ba (2014), DC-IGN Kulkarni et al. (2015), and InfoGAN Chen et al. (2016)).
    - 最後に、差異の定量化を支援するために、解きほぐしの新しい尺度を開発し、β-VAEがこの尺度のすべてのベースラインを大幅に上回ることを示します（ICA、PCA、VAE Kingma＆Ba（2014）、DC-IGN Kulkarni et al。（ 2015）、およびInfoGAN Chen他（2016））。

---

- Our main contributions are the following: 1) we propose β-VAE, a new unsupervised approach for learning disentangled representations of independent visual data generative factors; 2) we devise a protocol to quantitatively compare the degree of disentanglement learnt by different models; 3) we demonstrate both qualitatively and quantitatively that our β-VAE approach achieves state-of-the-art disentanglement performance compared to various baselines on a variety of complex datasets.
    - 主な貢献は次のとおりです。1）独立した視覚データ生成因子のもつれのない表現を学習するための新しい教師なしアプローチであるβ-VAEを提案します。 2）異なるモデルによって学習されたもつれ解除の程度を定量的に比較するプロトコルを考案します。 3）β-VAEアプローチが、さまざまな複雑なデータセットのさまざまなベースラインと比較して、最先端のディスエンタングルメントパフォーマンスを達成することを定性的および定量的に実証します。

---

- xxx

---

- Varying β changes the degree of applied learning pressure during training, thus encouraging different learnt representations. β-VAE where β = 1 corresponds to the original VAE formulation of (Kingma & Welling, 2014).
    - 変化するβは、トレーニング中に適用される学習圧力の度合いを変更し、異なる学習表現を奨励します。 β-VAE（β= 1）は、（Kingma＆Welling、2014）の元のVAE定式化に対応しています。

- We postulate that in order to learn disentangled representations of the conditionally independent data generative factors v, it is important to set β > 1, thus putting a stronger constraint on the latent bottleneck than in the original VAE formulation of Kingma & Welling (2014). 
    - 条件付き独立データ生成因子vの解きほぐされた表現を学習するには、β> 1を設定することが重要であり、したがって、Kingma＆Welling（2014）の元のVAE定式化よりも潜在的なボトルネックにより強い制約を課すことが重要です。

- These constraints limit the capacity of z, which, combined with the pressure to maximise the log likelihood of the training data x under the model, should encourage the model to learn the most efficient representation of the data.
    - これらの制約により、zの容量が制限されます。これは、モデルの下でトレーニングデータxの対数尤度を最大化する圧力と組み合わせて、モデルがデータの最も効率的な表現を学習するようにします。

- Since the data x is generated using at least some conditionally independent ground truth factors v, and the DKL term of the β-VAE objective function encourages conditional independence in qφ(z|x), we hypothesise that higher values of β should encourage learning a disentangled representation of v.
    - データxは、少なくともいくつかの条件付き独立グラウンドトゥルースファクターvを使用して生成され、β-VAE目的関数のDKL項はqφ（z | x）の条件付き独立性を促進するため、βの値が大きいほど、vのもつれのない表現の学習が促進されると仮定します。

- The extra pressures coming from high β values, however, may create a trade-off between reconstruction fidelity and the quality of disentanglement within the learnt latent representations. Disentangled representations emerge when the right balance is found between information preservation (reconstruction cost as regularisation) and latent channel capacity restriction (β > 1). The latter can lead to poorer reconstructions due to the loss of high frequency details when passing through a constrained latent bottleneck. Hence, the log likelihood of the data under the learnt model is a poor metric for evaluating disentangling in β-VAEs. Instead we propose a quantitative metric that directly measures the degree of learnt disentanglement in the latent representation.
    - しかし、高いβ値から生じる余分な圧力は、再構築の忠実度と学習した潜在表現内のもつれ解除の質とのトレードオフを生み出す可能性があります。情報の保存（正則化としての再構築コスト）と潜在的なチャネル容量制限（β> 1）の間に適切なバランスが見られる場合、解きほぐされた表現が現れます。後者は、制約のある潜在的なボトルネックを通過するときに高周波数の詳細が失われるため、再構築の質が低下する可能性があります。したがって、学習モデルの下でのデータの対数尤度は、β-VAEのもつれを評価するための貧弱なメトリックです。代わりに、潜在表現で学習したもつれ解除の程度を直接測定する定量的メトリックを提案します。

---

- Since our proposed hyperparameter β directly affects the degree of learnt disentanglement, we would like to estimate the optimal β for learning a disentangled latent representation directly. However, it is not possible to do so. This is because the optimal β will depend on the value of ε in Eq 2. Different datasets and different model architectures will require different optimal values of ε. However, when optimising β in Eq 4, we are indirectly also optimising ε for the best disentanglement (see Sec.A.7 for details), and while we can not learn the optimal value of β directly, we can instead estimate it using either our proposed disentanglement metric (see Sec 3) or through visual inspection heuristics.
    - 提案されたハイパーパラメータβは、学習したもつれ解除の程度に直接影響するため、もつれ解除された潜在表現を直接学習するための最適なβを推定したいと思います。 ただし、そうすることはできません。 これは、最適なβが式2のεの値に依存するためです。異なるデータセットと異なるモデルアーキテクチャでは、εの異なる最適値が必要になります。 ただし、式4でβを最適化する場合、間接的にεを最適化して最適化を解除し（詳細はセクションA.7を参照）、βの最適値を直接学習することはできませんが、代わりに次のいずれかを使用して推定できます 提案されたもつれ解除メトリック（セクション3を参照）または視覚的検査ヒューリスティックを使用。

# ■ 結論

## x. Conclusion


# ■ 何をしたか？詳細

## 2 β-VAE FRAMEWORK DERIVATION

- Let D = {X,V,W} be the set that consists of images x ∈ R and two sets of ground truth data generative factors: conditionally independent factors v ∈ RK , where log p(v|x) = k log p(vk |x); and conditionally dependent factors w ∈ RH . We assume that the images x are generated by the true world simulator using the corresponding ground truth data generative factors: p(x|v, w) = Sim(v, w).


- We want to develop an unsupervised deep generative model that, using samples from X only, can learn the joint distribution of the data x and a set of generative latent factors z such that z can generate the observed data x; that is, p(x|z) ≈ p(x|v, w) = Sim(v, w). Thus a suitable objective is to maximise the marginal (log-)likelihood of the observed data x in expectation over the whole distribution of latent factors z:
    - Xからのサンプルのみを使用して、zが観測データxを生成できるように、データxと生成潜在因子zの組み合わせの分布を学習できる、教師なしの深い生成モデルを開発します。
    - つまり、p（x | z）≈p（x | v、w）= Sim（v、w）です。 したがって、適切な目的は、潜在因子zの分布全体にわたって期待される観測データxの周辺（対数）尤度を最大化することです。

---

- For a given observation x, we describe the inferred posterior configurations of the latent factors z by a probability distribution qφ(z|x). Our aim is to ensure that the inferred latent factors qφ(z|x) capture the generative factors v in a disentangled manner.
    - 与えられた観測xに対して、確率分布qφ（z | x）によって潜在因子zの推定された事後分布を記述します。 私たちの目的は、推定された潜在因子qφ（z | x）が生成因子vを解きほぐして捕捉することです。

- The conditionally dependent data generative factors w can remain entangled in a separate subset of z that is not used for representing v. 
    - 条件に依存するデータ生成因子wは、vを表すために使用されないzの個別のサブセットに絡み合ったままにすることができます。

- In order to encourage this disentangling property in the inferred qφ(z|x), we introduce a constraint over it by trying to match it to a prior p(z) that can both control the capacity of the latent information bottleneck, and embodies the desiderata of statistical independence mentioned above.
    - 推定されたqφ（z | x）でこのもつれを解く特性を促進するために、潜在情報のボトルネックの容量を制御できる事前分布 p（z）に一致させようとすることで、制約を導入します。 
    - また、上記の統計的独立性の要求を具体化しています。

- This can be achieved if we set the prior to be an isotropic unit Gaussian (p(z) = N (0, I )), hence arriving at the constrained optimisation problem in Eq 2, where ε specifies the strength of the applied constraint.
    - これは、事前を等方性単位ガウス（p（z）= N（0、I））に設定すると達成できます。したがって、式2の制約付き最適化問題に到達します。εは適用された制約の強度を指定します。


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. Related Work


