# ■ 論文
- 論文タイトル：「Conditional Generative Adversarial Nets」
- 論文リンク：https://arxiv.org/abs/1411.1784
- 論文投稿日付：2014/11/06
- 著者：Mehdi Mirza, Simon Osindero
- categories：

# ■ 概要（何をしたか？）

## ABSTRACT
- Generative Adversarial Nets [8] were recently introduced as a novel way to train generative models.
    - Generative Adversarial Nets [8] は、最近、生成モデルを学習するための新手の [novel ] 方法として紹介された。

- In this work we introduce the conditional version of generative adversarial nets, which can be constructed by simply feeding the data, y, we wish to condition on to both the generator and discriminator.
    - この研究では、generative adversarial nets の条件付きバージョンを紹介する。
    - （これは、）単にデータを供給する [feed] ことによって、構築される [constructed]（手法である。）
    - 生成器と識別器の両方で条件づけしたい y。

- We show that this model can generate MNIST digits conditioned on class labels.
    - 我々は、このモデルが、クラスラベルで条件付けされた MNIST の数字を生成出来ることを示す。

- We also illustrate how this model could be used to learn a multi-modal model, and provide preliminary examples of an application to image tagging in which we demonstrate how this approach can generate descriptive tags which are not part of training labels.
    - 我々はまた、このモデルがどうのようにして、マルチモードの [multi-modal] モデルを学習したのかを説明する [illustrate]。
    - そして、画像タグ付けへの応用の準備的な [preliminary] 例を提供する。
    - （この例というのは、）このアプローチが、どのようにして、学習ラベルの一部ではないような、記述的な [descriptive] タグを生成することの出来るのかを説明する。


# ■ イントロダクション（何をしたいか？）

## 1. Introduction

- Generative adversarial nets were recently introduced as an alternative framework for training generative models in order to sidestep the difficulty of approximating many intractable probabilistic computations.
    - 多くの手に負えない [intractable] 確率的な計算 [computations] を回避する [sidestep] ために、[in order to]
    - GAN は最近、生成モデルの学習のための、代わりとなるフレームワークとして紹介された。

- Adversarial nets have the advantages that Markov chains are never needed, only backpropagation is used to obtain gradients, no inference is required during learning, and a wide variety of factors and interactions can easily be incorporated into the model.
    - 敵対的ネットワークは、マルコフ連鎖を決して必要とせず、勾配を得るために誤差逆伝搬だけでよいという利点がある。
    - 学習の間、推論 [inference] を要求しない。
    - そして、多種多様な要因と相互作用 [interactions] が、モデルの中に容易に組み込むこと [incorporate] が出来る（という利点が存在する。）

- Furthermore, as demonstrated in [8], it can produce state of the art log-likelihood estimates and realistic samples.
    - 更には、[8] で実証されているように、対数尤度とリアルなサンプル（生成）の SOTA を生み出すことが出来る。

- In an unconditioned generative model, there is no control on modes of the data being generated.
    - 条件付けされていない生成モデルにおいては、生成されたデータのモードを制御できない。

- However, by conditioning the model on additional information it is possible to direct the data generation process.
    - しかしながら、追加の情報に基づいて、モデルを条件付けすることによって、データの生成過程を指示する [direct] ことが可能となる。

- Such conditioning could be based on class labels, on some part of data for inpainting like [5], or even on data from different modality.
    - xxx

<br>

- In this work we show how can we construct the conditional adversarial net.
    - xxx

- And for empirical results we demonstrate two set of experiment.
    - xxx

- One on MNIST digit data set conditioned on class labels andone on MIR Flickr 25,000 dataset [10] for multi-modal learning.
    - xxx


# ■ 結論

## x. 論文の項目名 (Conclusion)


# ■ 何をしたか？詳細

## x. 論文の項目名


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）

## x. 論文の項目名


# ■ メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


