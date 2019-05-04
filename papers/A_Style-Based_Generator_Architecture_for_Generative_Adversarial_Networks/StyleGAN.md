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
    - <font color="Pink">高レベルの特徴（例えば、人間の顔を学習するときの顔の向きと特徴）の教師無しでの分割と、
    - そして、生成された画像における、確率的な多様性 [variation]（例えば、そばかす[freckles]、髪）
    - そして、これは、直感的な [intuitive] で scale-specific な合成 [synthesis] のコントロールを可能にする。</font>


- The new generator improves the state-of-the-art in terms of traditional distribution quality metrics, leads to demonstrably better interpolation properties, and also better disentangles the latent factors of variation.
    - 新しい生成器は、伝統的な分布のクオリティ指標に関して [in terms of]、SOTA を改善する。
    - 明らかに [demonstrably] より良い、補間 [interpolation] 性質 [demonstrably] に導き、
    - そしてまた、多様性の潜在的な [demonstrably] 要因を、より良く解きほぐす [disentangles]。

- To quantify interpolation quality and disentanglement, we propose two new, automated methods that are applicable to any generator architecture.
    - 補間クオリティと（絡み合った謎の）解きほぐし [disentanglement] を定量化 [quantify] するために、
    - 我々は、どんな生成器のアーキテクチャにも適用できる２つの新しい自動化手法を提案する。

- Finally, we introduce a new, highly varied and high-quality dataset of human faces.
    - 最後に、我々は、高い多様性と品質の、人間の顔の新しいデータセットを紹介する。

> ・特徴が絡み合っている<br>
> →この軸を調整すると、目が大きくなって髪が長くなって輪郭が丸くなるみたいになると各軸が複雑に影響し合って生成しづらい<br>

> ・特徴がしっかり分解されている<br>
> →この軸は目の大きさ、この軸は髪の長さ、この軸は輪郭みたいに特徴が独立していると生成がしやすい<br>

> StyleGAN では、この絡み合った [entanglement] 特徴を解きほぐし、[disentanglement]、特徴を分割する。[separation] <br>


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
    - <font color="Pink">このアーキテクチャの変更は、生成された画像において、確率的な多様性（例えば、そばかすや髪）から、高レベルの特徴（例えば、顔の向きや特徴）の、自動的で、教師なしの分割？に導く。</font>

- We do not modify the discriminator or the loss function in any way, and our work is thus orthogonal to the ongoing discussion about GAN loss functions, regularization, and hyperparameters [24, 45, 5, 40, 44, 36].
    - 我々は、あらゆる方法で、識別器や損失関数を修正しない。
    - そして、我々の研究は、そういうわけで、GAN の損失関数、正則化、ハイパーパラメータについて、進行中の議論と直交している。（＝別方向である）

---

- Our generator embeds the input latent code into an intermediate latent space, which has a profound effect on how the factors of variation are represented in the network.
    - **我々の生成器は、入力潜在コードを、中間の [intermediate] 潜在空間へ埋め込む。**
    - **（この潜在空間というのは、）多様性の要因が、どのようにして、ネットワークの中で表されるのかということに、深い [profound] 影響を与えるような（潜在空間）**

- The input latent space must follow the probability density of the training data, and we argue that this leads to some degree of unavoidable entanglement.
    - 入力潜在空間は、学習データの確率分布に従わなければならない。
    - そして、我々は、このことが、ある程度 [some degree of] 避けられられない絡み合い [entanglement] に至る [leads to] ということを主張する。

- Our intermediate latent space is free from that restriction and is therefore allowed to be disentangled.
    - 我々の中間の潜在空間は、このような制限から自由であり、
    - それ故に、絡み合った謎を解きほぐす [disentangled] ことを許容する。

- As previous methods for estimating the degree of latent space disentanglement are not directly applicable in our case, we propose two new automated metrics—perceptual path length and linear separability—for quantifying these aspects of the generator.
    - 潜在空間の解きほぐしの程度を推定するための以前の方法は、我々のケースに、直接的に応用出来ないので、
    - **我々は、生成器のこれらの側面を定量化するたの、２つの新しい自動的な指標を提案する。**
    - **（この新しい２つの指標というのは、perceptual path length と linear separability ）**

- Using these metrics, we show that compared to a traditional generator architecture, our generator admits a more linear, less entangled representation of different factors of variation.
    - これらの指標を使用することで、
    - 我々は、伝統的な生成器のアーキテクチャとの比較して、
    - 我々の生成器は、多様性の異なる要因の、より線形で、よりもつれ [entangled] の少ない表現になっていることを見せる。

> ・特徴が絡み合っている<br>
> →この軸を調整すると、目が大きくなって髪が長くなって輪郭が丸くなるみたいになると各軸が複雑に影響し合って生成しづらい<br>

> ・特徴がしっかり分解されている<br>
> →この軸は目の大きさ、この軸は髪の長さ、この軸は輪郭みたいに特徴が独立していると生成がしやすい<br>

> StyleGAN では、この絡み合った [entanglement] 特徴を解きほぐし、[disentanglement]、特徴を分割する。[separation] <br>

---

- Finally, we present a new dataset of human faces (Flickr-Faces-HQ, FFHQ) that offers much higher quality and covers considerably wider variation than existing high-resolution datasets (Appendix A).
    - 最後に、我々は、人間の顔の新しいデータセット（Flickr-Faces-HQ, FFHQ）を提供する。
    - （このデータセットというのは、）今存在している高解像度のデータセットより、より高い品質で、考えられる広い多様性をカバーするような（データセット）

- We have made this dataset publicly available, along with our source code and pre-trained networks.(1) 
    - 我々は、このデータセットを公共に利用可能にした。
    - 我々のソースコードと、事前学習されたネットワーク (1) と共に、

- The accompanying video can be found under the same link.
    
(1) : https://github.com/NVlabs/stylegan


# ■ 結論

## 5. Conclusion

- Based on both our results and parallel work by Chen et al.[6], it is becoming clear that the traditional GAN generator architecture is in every way inferior to a style-based design.
    - **我々の結果と、Chen による [6] との並行した研究の両方に基づいて、**
    - **伝統的な GAN のアーキテクチャが、全ての方法において、style-based の設計より劣っていることが明らかになっている。**

- This is true in terms of established quality metrics, and we further believe that our investigations to the separation of high-level attributes and stochastic effects, as well as the linearity of the intermediate latent space will prove fruitful in improving the understanding and controllability of GAN synthesis.
    - このことは、定量的な指標に関して、真実であり、
    - そして、我々は更に信じている。
    - 高レベルの特徴と確率的な効果の分離への、我々の調査。
    - 中間の潜在空間の線形性が、GAN での画像合成の理解と操作性を改善することにおいて、実りのある証明であるということを。

- We note that our average path length metric could easily be used as a regularizer during training, and perhaps some variant of the linear separability metric could act as one, too.
    - 我々は、平均 path length 指標が、学習中の正則化として、簡単に使用lできることと、
    - そして、もしかすると、線形分離指標のいくつかの変形が、それと同じように動作するということを記す。

- In general, we expect that methods for directly shaping the intermediate latent space during training will provide interesting avenues for future work.
    - 一般的に、我々は、学習中に中間潜在空間を直接的に形づくる [shaping] ための方法が、将来の研究のために、興味深い手段 [avenues] を提供することを期待する。


---


# ■ 何をしたか？詳細

## 2. Style-based generator


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


