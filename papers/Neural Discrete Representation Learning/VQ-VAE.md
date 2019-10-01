# ■ 論文
- 論文タイトル："Neural Discrete Representation Learning"
- 論文リンク：https://arxiv.org/abs/1711.00937
- 論文投稿日付：2017/11/02
- 被引用数（記事作成時点）：xxx 件
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- Learning useful representations without supervision remains a key challenge in machine learning. In this paper, we propose a simple yet powerful generative model that learns such discrete representations. Our model, the Vector Quantised- Variational AutoEncoder (VQ-VAE), differs from VAEs in two key ways: the encoder network outputs discrete, rather than continuous, codes; and the prior is learnt rather than static.
    - 監督なしで有用な表現を学習することは、機械学習の重要な課題です。 この論文では、このような離散表現を学習するシンプルでありながら強力な生成モデルを提案します。 私たちのモデルであるベクトル量子化変分オートエンコーダー（VQ-VAE）は、2つの重要な点でVAEと異なります。エンコーダーネットワークは、連続ではなく離散変数を出力します。 また、事前確率は静的ではなく学習されます。

- In order to learn a discrete latent representation, we incorporate ideas from vector quantisation (VQ). Using the VQ method allows the model to circumvent issues of “posterior collapse” -— where the latents are ignored when they are paired with a powerful autoregressive decoder -— typically observed in the VAE framework.
    - 離散的な潜在表現を学習するために、ベクトル量子化（VQ）からのアイデアを組み込みます。 
    - VQメソッドを使用すると、モデルは、通常VAEフレームワークで見られる「後方崩壊」の問題を回避できます。
    - これは、潜在変数は強力な自己回帰デコーダーとペアになったときに無視されるというもの

- Pairing these representations with an autoregressive prior, the model can generate high quality images, videos, and speech as well as doing high quality speaker conversion and unsupervised learning of phonemes, providing further evidence of the utility of the learnt representations.
    - これらの表現を自己回帰事前分布とペアリングすると、モデルは高品質の画像、ビデオ、音声を生成できるだけでなく、高品質のスピーカー変換と音素の教師なし学習を行い、学習した表現の有用性のさらなる証拠を提供できます。

# ■ イントロダクション（何をしたいか？）

## x. Introduction

- xxx

---

- Maximum likelihood and reconstruction error are two common objectives used to train unsupervised models in the pixel domain, however their usefulness depends on the particular application the features are used in.
    - 最尤法と再構成誤差は、ピクセル領域で教師なしモデルをトレーニングするために使用される2つの一般的な目的ですが、それらの有用性は、機能が使用される特定のアプリケーションによって異なります。

- Our goal is to achieve a model that conserves the important features of the data in its latent space while optimising for maximum likelihood.
    - 私たちの目標は、最尤を最適化する一方で、データの重要な特徴を潜在空間で保存するモデルを実現することです。

- As the work in [7] suggests, the best generative models (as measured by log-likelihood) will be those without latents but a powerful decoder (such as PixelCNN). However, in this paper, we argue for learning discrete and useful latent variables, which we demonstrate on a variety of domains.
    - [7]の研究が示唆するように、最適な生成モデル（対数尤度で測定）は、潜在性のない強力なデコーダー（PixelCNNなど）になります。 ただし、このペーパーでは、さまざまな領域で実証する、離散的で有用な潜在変数を学習することを主張します。

---

- Learning representations with continuous features have been the focus of many previous work [16, 39, 6, 9] however we concentrate on discrete representations [27, 33, 8, 28] which are potentially a more natural fit for many of the modalities we are interested in.
    - 連続的な特徴を持つ学習表現は、以前の多くの研究[16、39、6、9]の焦点でしたが、我々が興味がある多くの様式 [modalities] により自然に適合する可能性のある離散表現[27、33、8、28]に集中します 

- Language is inherently discrete, similarly speech is typically represented as a sequence of symbols. Images can often be described concisely by language [40].
    - 言語は本質的に離散的であり、同様に音声は通常、一連の記号として表されます。 多くの場合、画像は言語ごとに簡潔に説明できます[40]。

- Furthermore, discrete representations are a natural fit for complex reasoning, planning and predictive learning (e.g., if it rains, I will use an umbrella).
    - さらに、離散表現は、複雑な推論、計画、および予測学習に自然に適合します（雨が降った場合、傘を使用します）。

- While using discrete latent variables in deep learning has proven challenging, powerful autoregressive models have been developed for modelling distributions over discrete variables [37].    
    - ディープラーニングで離散潜在変数を使用することは困難であることが実証されていますが、離散変数の分布をモデリングするための強力な自己回帰モデルが開発されています[37]

---

- In our work, we introduce a new family of generative models succesfully combining the variational autoencoder (VAE) framework with discrete latent representations through a novel parameterisation of the posterior distribution of (discrete) latents given an observation. Our model, which relies on vector quantization (VQ), is simple to train, does not suffer from large variance, and avoids the “posterior collapse” issue which has been problematic with many VAE models that have a powerful decoder, often caused by latents being ignored. Additionally, it is the first discrete latent VAE model that get similar performance as its continuous counterparts, while offering the flexibility of discrete distributions. We term our model the VQ-VAE.
    - **私たちの仕事では、観察に基づいて（離散）潜在の事後分布の新しいパラメーター化により、変分オートエンコーダー（VAE）フレームワークを離散潜在表現とうまく組み合わせた生成モデルの新しいファミリーを紹介します。 ベクトル量子化（VQ）に依存する我々のモデルは訓練が簡単で、大きな分散の影響を受けず、多くの場合潜在的要因によって引き起こされる強力なデコーダーを持つ多くのVAEモデルで問題となっている「事後崩壊」問題を回避します 無視されます。 さらに、離散分布の柔軟性を提供しながら、連続した対応するものと同様のパフォーマンスを得る最初の離散潜在VAEモデルです。 モデルをVQ-VAEと呼びます。**

---

- Since VQ-VAE can make effective use of the latent space, it can successfully model important features that usually span many dimensions in data space (for example objects span many pixels in images, phonemes in speech, the message in a text fragment, etc.) as opposed to focusing or spending capacity on noise and imperceptible details which are often local.
    - VQ-VAEは潜在空間を効果的に利用できるため、通常はデータ空間の多くの次元にまたがる重要な機能をモデル化できます（たとえば、オブジェクトは画像の多くのピクセル、音声の音素、テキストの断片のメッセージなどにまたがります）。 ）局所的であることが多いノイズや知覚できない詳細に焦点を合わせたり、容量を消費したりするのではなく。

---

- Lastly, once a good discrete latent structure of a modality is discovered by the VQ-VAE, we train a powerful prior over these discrete random variables, yielding interesting samples and useful applications.
    - 最後に、様式の良好な離散潜在構造がVQ-VAEによって発見されると、これらの離散ランダム変数に対する強力な事前学習を行い、興味深いサンプルと有用なアプリケーションを生成します。

- For instance, when trained on speech we discover the latent structure of language without any supervision or prior knowledge about phonemes or words. Furthermore, we can equip our decoder with the speaker identity, which allows for speaker conversion, i.e., transferring the voice from one speaker to another without changing the contents. We also show promising results on learning long term structure of environments for RL.
    - たとえば、スピーチのトレーニングを受けた場合、音素や単語に関する監督や事前の知識なしに、言語の潜在的な構造を発見します。 さらに、デコーダーにスピーカーIDを装備できます。これにより、スピーカーを変換できます。つまり、コンテンツを変更せずに、あるスピーカーから別のスピーカーに音声を転送できます。 また、RLの環境の長期構造を学習する上で有望な結果を示します。

---

- Our contributions can thus be summarised as:

- Introducing the VQ-VAE model, which is simple, uses discrete latents, does not suffer from “posterior collapse” and has no variance issues.
    - VQ-VAEモデルの導入は単純で、離散潜在を使用し、「後方崩壊」の影響を受けず、分散の問題もありません。
    
- We show that a discrete latent model (VQ-VAE) perform as well as its continuous model counterparts in log-likelihood.
    - 離散潜在モデル（VQ-VAE）は、対数尤度の連続モデルと同様に機能することを示します。

- When paired with a powerful prior, our samples are coherent and high quality on a wide variety of applications such as speech and video generation.
    - パワフルな prior と組み合わせると、音声やビデオの生成など、さまざまなアプリケーションで一貫した高品質のサンプルを使用できます。

- We show evidence of learning language through raw speech, without any supervision, and show applications of unsupervised speaker conversion.
    - 私たちは、監督なしで生のスピーチを通して言語学習の証拠を示し、教師なしの話者変換のアプリケーションを示します。

# ■ 結論

## x. Conclusion


# ■ 何をしたか？詳細

## 3 VQ-VAE

- xxx 

- In this work we introduce the VQ-VAE where we use discrete latent variables with a new way of training, inspired by vector quantisation (VQ). The posterior and prior distributions are categorical, and the samples drawn from these distributions index an embedding table. These embeddings are then used as input into the decoder network.
    - この作業では、ベクトル量子化（VQ）に触発された新しいトレーニング方法で離散潜在変数を使用するVQ-VAEを紹介します。 事後分布と事前分布はカテゴリカルであり、これらの分布から抽出されたサンプルは埋め込みテーブルにインデックスを付けます。 これらの埋め込みは、デコーダネットワークへの入力として使用されます。

### 3.1 Discrete Latent variables

- We define a latent embedding space e ∈ RK×D where K is the size of the discrete latent space (i.e, a K-way categorical), and D is the dimensionality of each latent embedding vector ei. Note that there are K embedding vectors ei ∈ RD , i ∈ 1, 2, ..., K .

- As shown in Figure 1, the model takes an input x, that is passed through an encoder producing output ze(x).

- The discrete latent variables z are then calculated by a nearest neighbour look-up using the shared embedding space e as shown in equation 1. The input to the decoder is the corresponding embedding vector ek as given in equation 2.
    - 離散潜在変数zは、方程式1に示すように、共有埋め込み空間eを使用した最近傍ルックアップによって計算されます。
    - デコーダーへの入力は、方程式2に示す対応する埋め込みベクトルekです。

- One can see this forward computation pipeline as a regular autoencoder with a particular non-linearity that maps the latents to 1-of-K embedding vectors. The complete set of parameters for the model are union of parameters of the encoder, decoder, and the embedding space e. For sake of simplicity we use a single random variable z to represent the discrete latent variables in this Section, however for speech, image and videos we actually extract a 1D, 2D and 3D latent feature spaces respectively.
    - このフォワードコンピューティングパイプラインは、潜在性を1-of-K埋め込みベクトルにマッピングする特定の非線形性を備えた通常のオートエンコーダーとして見ることができます。 モデルのパラメーターの完全なセットは、エンコーダー、デコーダー、および埋め込みスペースeのパラメーターの結合です。 簡単にするために、このセクションでは1つのランダム変数zを使用して離散潜在変数を表しますが、音声、画像、およびビデオについては、それぞれ1D、2D、および3Dの潜在特徴空間を実際に抽出します


### 3.2 Learning

- Note that there is no real gradient defined for equation 2, however we approximate the gradient similar to the straight-through estimator [3] and just copy gradients from decoder input zq(x) to encoder output ze(x). One could also use the subgradient through the quantisation operation, but this simple estimator worked well for the initial experiments in this paper.
    - 式2に定義された実際の勾配はありませんが、ストレート推定器[3]と同様の勾配を近似し、勾配をデコーダー入力zq（x）からエンコーダー出力ze（x）にコピーすることに注意してください。 量子化演算で部分勾配を使用することもできますが、この簡単な推定器はこの論文の最初の実験ではうまく機能しました。

---

- During forward computation the nearest embedding zq (x) (equation 2) is passed to the decoder, and during the backwards pass the gradient ∇zL is passed unaltered to the encoder. Since the output representation of the encoder and the input to the decoder share the same D dimensional space, the gradients contain useful information for how the encoder has to change its output to lower the reconstruction loss.
    - 順方向の計算では、最も近い埋め込みzq（x）（式2）がデコーダーに渡され、逆方向のパスでは勾配∇zLがそのままエンコーダーに渡されます。 エンコーダーの出力表現とデコーダーへの入力は同じD次元空間を共有するため、勾配には、エンコーダーが出力を変更して再構成損失を下げる方法に関する有用な情報が含まれています。

---

- As seen on Figure 1 (right), the gradient can push the encoder’s output to be discretised differently in the next forward pass, because the assignment in equation 1 will be different.
    - 図1（右）に示すように、勾配は、方程式1の割り当てが異なるため、次のフォワードパスでエンコーダーの出力を異なる方法で離散化することができます。

---

- Equation 3 specifies the overall loss function. It is has three components that are used to train different parts of VQ-VAE. The first term is the reconstruction loss (or the data term) which optimizes the decoder and the encoder (through the estimator explained above).
    - 式3は、全体的な損失関数を指定します。 VQ-VAEのさまざまな部分をトレーニングするために使用される3つのコンポーネントがあります。 最初の項は、デコーダーとエンコーダーを最適化する再構成損失（またはデータ項）です（上記で説明した推定器を使用）。 

- Due to the straight-through gradient estimation of mapping from ze(x) to zq(x), the embeddings ei receive no gradients from the reconstruction loss log p(z|zq (x)). Therefore, in order to learn the embedding space, we use one of the simplest dictionary learning algorithms, Vector Quantisation (VQ). The VQ objective uses the l2 error to move the embedding vectors ei towards the encoder outputs ze(x) as shown in the second term of equation 3. Because this loss term is only used for updating the dictionary, one can alternatively also update the dictionary items as function of moving averages of ze(x) (not used for the experiments in this work). For more details see Appendix A.1.
    - ze（x）からzq（x）へのマッピングの直接勾配勾配推定により、埋め込みeiは再構成損失ログp（z | zq（x））から勾配を受け取りません。
    - したがって、埋め込みスペースを学習するために、最も単純な辞書学習アルゴリズムの1つであるベクトル量子化（VQ）を使用します。 VQ目的は、方程式2の2番目の項に示すように、l2エラーを使用して埋め込みベクトルeiをエンコーダー出力ze（x）に移動します。この損失項は辞書の更新にのみ使用されるため、 ze（x）の移動平均の関数としてのアイテム（この作品の実験には使用されません）。 詳細については、付録A.1を参照してください。

---

- Finally, since the volume of the embedding space is dimensionless, it can grow arbitrarily if the embeddings ei do not train as fast as the encoder parameters. To make sure the encoder commits to an embedding and its output does not grow, we add a commitment loss, the third term in equation 3. Thus, the total training objective becomes:


### 3.3 Prior

- xxx

# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. Related Work


