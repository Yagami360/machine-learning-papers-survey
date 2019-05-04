# ■ 論文
- 論文タイトル："A Style-Based Generator Architecture for Generative Adversarial Networks"
- 論文リンク：
- 論文投稿日付：2018/11/12(v1), 2019/03/29(v2)
- 著者：Tero Karras(NVIDIA), Samuli Laine(NVIDIA), Timo Aila(NVIDIA)
- categories：

# ■ 概要（何をしたか？）

## Abstract

- We propose an alternative generator architecture for generative adversarial networks, borrowing from style transfer literature.
    - 我々は、style transfer の文献から拝借して、GAN に対しての代わりとなるアーキテクチャを提供する。

- The new architecture leads to an automatically learned, unsupervised separation of high-level attributes (e.g., pose and identity when trained on human faces) and stochastic variation in the generated images (e.g., freckles, hair), and it enables intuitive, scale-specific control of the synthesis.
    - 新しいアーキテクチャは、自動的に学習され、
    - <font color="Pink">高レベルの属性の教師無しでの分割（例えば、人間の顔を学習するときの姿勢とアイデンティティー）と、
    - そして、生成された画像における、確率的な多様性 [variation]（例えば、そばかす[freckles]、髪）
    - そして、これは、直感的な [intuitive] で scale-specific な合成 [synthesis] のコントロールを可能にする。</font>

- The new generator improves the state-of-the-art in terms of traditional distribution quality metrics, leads to demonstrably better interpolation properties, and also better disentangles the latent factors of variation.
    - 新しい生成器は、伝統的な分布のクオリティ指標に関して [in terms of]、SOTA を改善する。
    - 明らかに [demonstrably] より良い、補間 [interpolation] 性質 [demonstrably] に導き、
    - そしてまた、多様性の潜在的な [demonstrably] 要因を、より良く解きほぐす [disentangles]。

- To quantify interpolation quality and disentanglement, we propose two new, automated methods that are applicable to any generator architecture.
    - 補間クオリティと解きほぐし？[disentanglement] を定量化 [quantify] するために、
    - 我々は、どんな生成器のアーキテクチャにも適用できる２つの新しい自動化手法を提案する。

- Finally, we introduce a new, highly varied and high-quality dataset of human faces.
    - 最後に、我々は、高い多様性と品質の、人間の顔の新しいデータセットを紹介する。

# ■ イントロダクション（何をしたいか？）

## 1. Introduction

- The resolution and quality of images produced by generative methods—especially generative adversarial networks (GAN) [22]—have seen rapid improvement recently [30, 45, 5].
    - 生成モデル、とりわけ GAN によって生成された画像の解像度と品質は、最近急速に改善している。

- Yet the generators continue to operate as black boxes, and despite recent efforts [3], the understanding of various aspects of the image synthesis process, e.g., the origin of stochastic features, is still lacking.
    - まだ、生成器がブラックボックスとして、動作し続け、
    - そして最近の努力にも関わらず、
    - 画像合成 [synthesis] 処理の様々な側面の理解（例えば、確率的な特徴の起源）は、まだ欠けている。

- The properties of the latent space are also poorly understood, and the commonly demonstrated latent space interpolations [13, 52, 37] provide no quantitative way to compare different generators against each other.
    - 潜在空間の性質もまた、よく理解されていない。
    - そして、一般に [commonly] 実証されている潜在空間の補間は、互いに異なる生成器を比較するための定量的な [quantitative] 方法を提供していない。

---

- Motivated by style transfer literature [27], we re-design the generator architecture in a way that exposes novel ways to control the image synthesis process.
    - style transfer の文献 [27] によって動機づけされ、
    - 我々は、画像合成処理をコントロールするための目新しい [novel] 方法を
    公開するような方法で、生成器のアーキテクチャを再設計する。

- Our generator starts from a learned constant input and adjusts the “style” of the image at each convolution layer based on the latent code, therefore directly controlling the strength of image features at different scales.
    - 我々の生成器は、学習された一定の入力からスタートする。
    - そして、潜在コード？において、各畳み込み層ベースでの、画像の ”スタイル” を調整する。
    - それ故、異なるスケールで、画像の特徴の強度を直接的にコントロールする。

- Combined with noise injected directly into the network, this architectural change leads to automatic, unsupervised separation of high-level attributes (e.g., pose, identity) from stochastic variation (e.g., freckles,hair) in the generated images, and enables intuitive scale-specific mixing and interpolation operations.
    - ネットワークの中に、直接的に注入されたノイズを組み合わせることで、
    - <font color="Pink">このアーキテクチャの変更は、生成された画像において、確率的な多様性（例えば、そばかすや髪）から、高レベルの属性（例えば、姿勢やアイデンティティー）の、自動的で、教師なしの分割？に導く。</font>

- We do not modify the discriminator or the loss function in any way, and our work is thus orthogonal to the ongoing discussion about GAN loss functions, regularization, and hyperparameters [24, 45, 5, 40, 44, 36].
    - 我々は、あらゆる方法で、識別器や損失関数を修正しない。
    - そして、我々の研究は、そういうわけで、GAN の損失関数、正則化、ハイパーパラメータについて、進行中の議論と直交している。（＝別方向である）

---

- Our generator embeds the input latent code into an intermediate latent space, which has a profound effect on how the factors of variation are represented in the network.
    - 我々の生成器は、入力された潜在コードを、中間の [intermediate] 潜在空間へ埋め込む。
    - （この潜在空間というのは、）多様性の要因が、どのようにして、ネットワークの中で表されるのかということに、深い [profound] 影響を与えるような（潜在空間）

- The input latent space must follow the probability density of the training data, and we argue that this leads to some degree of unavoidable entanglement.

- Our intermediate latent space is free from that restriction and is therefore allowed to be disentangled.

- As previous methods for estimating the degree of latent space disentanglement are not directly applicable in our case, we propose two new automated metrics—perceptual path length and linear separability—for quantifying these aspects of the generator.

- Using these metrics, we show that compared to a traditional generator architecture, our generator admits a more linear, less entangled representation of different factors of variation.

---

# ■ 結論

## x. 論文の項目名 (Conclusion)


# ■ 何をしたか？詳細

## x. 論文の項目名


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


