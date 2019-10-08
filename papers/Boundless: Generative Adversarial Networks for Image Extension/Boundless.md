# ■ 論文
- 論文タイトル："Boundless: Generative Adversarial Networks for Image Extension"
- 論文リンク：https://arxiv.org/abs/1908.07007
- 論文投稿日付：
- 被引用数（記事作成時点）：xxx 件
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- Image extension models have broad applications in image editing, computational photography and computer graphics. While image inpainting has been extensively studied in the literature, it is challenging to directly apply the state-of-the-art inpainting methods to image extension as they tend to generate blurry or repetitive pixels with inconsistent semantics. We introduce semantic conditioning to the discriminator of a generative adversarial network (GAN), and achieve strong results on image extension with coherent semantics and visually pleasing colors and textures. We also show promising results in extreme extensions, such as panorama generation.
    - 画像拡張モデルは、画像編集、計算写真、およびコンピューターグラフィックスに幅広い用途があります。 画像の修復は文献で広く研究されていますが、最新の修復方法を画像拡張に直接適用することは、一貫性のないセマンティクスでぼやけたピクセルや繰り返しピクセルを生成する傾向があるため、困難です。 生成的敵対ネットワーク（GAN）の弁別器にセマンティック条件付けを導入し、コヒーレントなセマンティクスと視覚的に心地よい色とテクスチャを備えた画像拡張で強力な結果を達成します。 また、パノラマ生成などの極端な拡張で有望な結果を示しています。

# ■ イントロダクション（何をしたいか？）

## 1. Introduction

- Across many disparate disciplines there exists a strong need for high quality image extensions. In virtual reality, for example, it is often necessary to simulate different camera extrinsics than were actually used to capture an image, which generally requires filling in content outside of the original image bounds [19]. Panorama stitching generally requires cropping the jagged edges of stitched projections to achieve a rectangular panorama, but high quality image extension could enable filling in the gaps instead [23]. Similarly, extending videos has been shown to create more immersive experiences for viewers [3]. As televisions transition to the 16:9 HDTV aspect ratio, it is appealing to display videos filmed at a different aspect ratio than the screen. [22, 33].


---

- We desire a seamless blending between the original and extended image regions. Moreover, the extended region should match the original at the textural, structural and semantic levels, while appearing a plausible extension.
    - 元の画像領域と拡張された画像領域をシームレスにブレンドすることが望まれます。 さらに、拡張領域は、テクスチャ、構造、およびセマンティックレベルで元の領域と一致する必要がありますが、妥当な拡張として表示されます。

- Boundary conditions are only available on one side of the extended region. This is in contrast to the image inpainting problem [26, 48], where the region to be filled in is surrounded in all directions by original image data, significantly constraining the problem. Therefore, inpainting algorithms tend to have more predictable and higher quality results than image extension algorithms. In fact, we demonstrate in this paper that using inpainting algorithms with no modifications leads to poor results for image extension.
    - 境界条件は、拡張領域の片側でのみ使用できます。 これは、塗りつぶす領域がすべての方向で元の画像データに囲まれ、問題を大幅に制約する画像修復の問題[26、48]とは対照的です。 したがって、修復アルゴリズムは、画像拡張アルゴリズムよりも予測可能かつ高品質の結果になる傾向があります。 実際、この論文では、修正なしで修復アルゴリズムを使用すると、画像拡張の結果が悪いことを示しています。

---

- In the literature, image extension has been studied using both parametric and non-parametric methods [41, 50, 31, 7, 5]. While these methods generally do a good job of blending the extended and original regions, they have significant drawbacks. They either require the use of a carefully chosen guide image from which patches are borrowed, or they mostly extend texture, without taking into account larger scale structure or the semantics of an image. These models are only applicable in a narrow range of use cases and cannot learn from a diverse data set. In practice, we would like image extension models that work on diverse data and can extend structure.
    - 文献では、画像拡張は、パラメトリック法とノンパラメトリック法の両方を使用して研究されています[41、50、31、7、5]。 これらの方法は一般に、拡張領域と元の領域をブレンドするのに適していますが、重大な欠点があります。 パッチの借用元である慎重に選択されたガイド画像を使用するか、大規模な構造や画像のセマンティクスを考慮せずにテクスチャを拡張する必要があります。 これらのモデルは、狭い範囲のユースケースでのみ適用可能であり、多様なデータセットから学習することはできません。 実際には、多様なデータで機能し、構造を拡張できる画像拡張モデルが必要です。

---

- Fast progress in deep neural networks has brought the advent of powerful new classes of image generation models, the most prominent of which are generative adversarial networks (GANs) [14] and variational autoencoders [21]. GANs in particular have demonstrated the ability to generate high quality samples. In this paper, we use GANs, modified as described below, to learn plausible image extensions from large datasets of natural images using self-supervision, similar in spirit to the use of GANs in applications such as inpainting [16] and image superresolution [24].
    - ディープニューラルネットワークの急速な進歩により、強力な新しいクラスの画像生成モデルが登場しました。その中で最も顕著なものは、生成的敵対ネットワーク（GAN）[14]と変分オートエンコーダ[21]です。 特にGANは、高品質のサンプルを生成する能力を実証しています。 本書では、以下に説明するように修正したGANを使用して、自己監視を使用して自然画像の大規模なデータセットから妥当な画像拡張を学習します。これは、修復[16]や画像超解像[24] ]。

---

- For the image extension problem, while state-of-the-art inpainting models [48, 47] provide us a good starting point, we find that the results quickly degrade as we extend further from the image border. We start by pruning the components that do not apply to our setting and then adopt some techniques from the broader study of GANs. Finally, we introduce a novel method, derived from [29], of providing the model with semantic conditioning, that substantially improves the results. In summary, our contributions are:
    - 画像拡張の問題については、最新の修復モデル[48、47]が適切な出発点を提供しますが、画像の境界からさらに拡張すると結果が急速に低下することがわかります。 設定に適用されないコンポーネントを削除することから始めて、GANのより広範な研究からいくつかのテクニックを採用します。 最後に、[29]から派生した、モデルにセマンティック条件付けを提供し、結果を大幅に改善する新しい方法を紹介します。 要約すると、私たちの貢献は次のとおりです。

1. We are one of the first to use GAN’s effectively to learn image extensions, and do so reliably for large extrapolations (up to 3 times the width of the original).
    - GANを効果的に使用して画像の拡張子を学習し、大規模な外挿（元の幅の3倍まで）で確実に学習する最初の1つです。

1. We introduce a stabilization scheme for our training, based on using semantic information from a pretrained deep network to modulate the behavior of the discriminator in a GAN. This stabilization scheme is useful for any adversarial model which has a ground truth sample for each generator input.
    - GANの弁別器の動作を調整するために、事前トレーニングされたディープネットワークからのセマンティック情報を使用することに基づいて、トレーニングの安定化スキームを導入します。 この安定化スキームは、各ジェネレーター入力にグラウンドトゥルースサンプルがある敵対モデルに役立ちます。

1. We show empirically that several architectural components are important for good image extension. We present ablation studies that show the effect of each of these components.
    - いくつかのアーキテクチャコンポーネントが優れたイメージ拡張に重要であることを経験的に示します。 これらの各コンポーネントの効果を示すアブレーション研究を紹介します。

# ■ 結論

## x. Conclusion


# ■ 何をしたか？詳細

## 3. Model

- Our model uses a Wasserstein GAN framework [28] comprising a generator network that is trained with the assistance of a concurrently-trained discriminator network.
    - 私たちのモデルは、同時に訓練された弁別器ネットワークの助けを借りて訓練されたジェネレーターネットワークを含むWasserstein GANフレームワーク[28]を使用します。

---

- Our generator network, G has an input consisting of the image z with pixel values in the range [−1, 1], which is to be extended, and a binary mask M . These are the same dimensions spatially and are concatenated channel-wise. Both z and M consist of a region of known pixels and a region of unknown pixels. In contrast to inpainting frameworks, the unknown region shares a boundary with the known region on only one side. z is set to 0 in the unknown region, while M is set to 1 in the unknown region and 0 in the known region. At training time,
    - ジェネレータネットワークGには、拡張される[-1、1]の範囲のピクセル値を持つイメージzとバイナリマスクMで構成される入力があります。 これらは空間的に同じ次元であり、チャネルごとに連結されます。 zとMは、既知のピクセルの領域と未知のピクセルの領域で構成されます。 修復フレームワークとは対照的に、未知の領域は片側だけで既知の領域と境界を共有します。 zは未知の領域で0に設定され、Mは未知の領域で1に設定され、既知の領域で0に設定されます。 トレーニング時に、

> 式

---

- The output G(z, M ) of G has the same dimensions as z and a pixel loss during training uses this full output. However, the last stage before feeding into the discriminator D is to replace what G synthesized in the unmasked regions with the known input pixels:
    - Gの出力G（z、M）はzと同じ次元をもち、トレーニング中のピクセル損失はこの完全な出力を使用します。 ただし、弁別器Dに入力する前の最後の段階は、マスクされていない領域でGが合成したものを既知の入力ピクセルに置き換えることです。
    
> 式

### 3.1. Generator

- G generally follows the same fully convolutional encoder-decoder architecture as used by [47] (see Figure 2). Each layer in the generator except the last one uses gated convolutions [47] to enable the model to learn to select the contributing features for each spatial location and channel. Following the inpainting guidance in [48], each layer except the last uses an ELU activation function [10], and the final layer clips its outputs to the range [−1, 1]. As in [16, 47, 48], the innermost layers utilize dilated convolutions to increase their receptive field size.
    - Gは通常、[47]で使用されているのと同じ完全な畳み込みエンコーダーデコーダーアーキテクチャに従います（図2を参照）。 ジェネレーターの各レイヤーは、最後のレイヤーを除き、ゲート畳み込み[47]を使用して、モデルが各空間位置とチャネルに寄与する特徴を選択することを学習できるようにします。 [48]の修復ガイダンスに従って、最後のレイヤーを除く各レイヤーはELUアクティベーション関数[10]を使用し、最後のレイヤーはその出力を[-1、1]の範囲にクリップします。 [16、47、48]のように、最も内側の層は拡張畳み込みを利用して、受容野の大きさを増やします。

---

- To address the image extension problem, we deviated from the generator architecture proposed by [47] in a few crucial ways.
- We eliminated the refinement network, including the contextual attention layer, since this layer is biased towards copying patches from the unmasked portion of the input. While borrowing patches is a useful property for inpainting of images [7], in the case of image extension, it is less likely that repeated patterns will result in convincing extension.
- Figure 3 shows the effect of the contextual attention layer of [48, 47]. We also compare to Adobe Photoshop’s PatchMatch-based [7] Content Aware Fill tool, which generates similar artifacts due to copying patches. These copying artifacts occur on a large fraction of the output images.
    - 画像拡張の問題に対処するために、[47]で提案されているジェネレータアーキテクチャからいくつかの重要な方法で逸脱しました。
    - このレイヤーは入力のマスクされていない部分からパッチをコピーするようにバイアスされているため、コンテキストアテンションレイヤーを含む改良ネットワークを削除しました。 パッチの借用は画像の修復に有用な特性ですが[7]、画像の拡張の場合、繰り返しパターンが説得力のある拡張をもたらす可能性は低くなります。
    - 図3は、[48、47]のコンテキストアテンションレイヤーの効果を示しています。 また、Adobe PhotoshopのPatchMatchベースの[7] Content Aware Fillツールと比較します。このツールは、パッチのコピーにより同様のアーティファクトを生成します。 これらのコピーアーティファクトは、出力画像の大部分で発生します。

### 3.2. Discriminator

- The objective of the discriminator network (see Figure 2) is determining whether an image is generator-produced or real. In our problem setup, the concern is not just whether the output of G appears real, but also that it is a plausible extension of G’s inputs. To this end, we design our discriminator to be conditioned on the specific generator inputs when evaluating whether what is fed into the discriminator is real or fake. We condition the discriminator in two ways.
    - ディスクリミネーターネットワークの目的（図2を参照）は、画像がジェネレーターで作成されたものであるか、実際のものであるかを判断することです。 問題の設定では、Gの出力が実際に表示されるかどうかだけでなく、Gの入力の妥当な拡張であるかどうかも懸念されます。 このため、識別器に供給されるものが本物であるか偽物であるかを評価するときに、特定のジェネレーター入力を条件とするように識別器を設計します。 識別器は2つの方法で調整します。

---

- First, when a generated image is input, we copy the known pixels from z to overwrite the corresponding generated pixels, as described in eq 2, and we additionally input the mask M itself.
    - まず、生成された画像が入力されると、既知のピクセルをzからコピーして対応する生成されたピクセルを上書きします（式2を参照）。さらにマスクM自体を追加入力します。 
    
- This on its own provides a major advantage to the discriminator in the adversarial game, since it can focus in on the area right around the seam at the edge of the real content and easily determine that an image is fake if there is any abrupt change in image statistics along that seam.
    - これは、実際のコンテンツの端の縫い目の周りの領域に焦点を合わせ、急激な変化がある場合に画像が偽物であると簡単に判断できるため、敵対ゲームの弁別者に大きな利点を提供します その継ぎ目に沿った画像統計。 
    
- We see this play out during training, as the generated image content close to the seam is the first to improve and the quality improvement gradually spreads towards the opposite edge of the image as training progresses.
    - シームに近い生成された画像コンテンツが最初に改善され、トレーニングが進むにつれて品質の改善が画像の反対側に徐々に広がるため、これはトレーニング中に再生されます。 
    
- On its own, this form of conditioning produces seamless results, but the quality of generated content still deteriorates as it moves further from the real content.
    - この形式のコンディショニングはシームレスに結果を生成しますが、生成されたコンテンツの品質は、実際のコンテンツから離れるにつれて低下します。

---

- To address this, we add another form of conditioning, which is a modified version of the conditional projection discriminator (cGAN) [29]. In the original cGAN paper, a one-hot class label y is passed into the discriminator in addition to the image x∗ to be classified as real or fake. The discriminator output is then
    - これに対処するために、条件付き射影識別器（cGAN）[29]の修正版である別の形式の条件付けを追加します。 元のcGAN論文では、画像x ∗に加えてワンホットクラスラベルyが弁別器に渡されて、本物または偽物として分類されます。 弁別器の出力は

> 式

- where φ is a learned function mapping an image to a vector, fφ is a learned fully-connected layer that maps that vector to a scalar, fy is a learned fully-connected layer mapping y to a vector of the same size as the output of φ, and ⟨·, ·⟩ denotes an inner product. The cGAN paper shows that this parameterization of the GAN objective enables the model to simultaneously learn the distributions p(x) and p(y|x).
    - ここで、φは画像をベクトルにマッピングする学習済み関数、fφはそのベクトルをスカラーにマッピングする学習済みの完全接続層、fyはyを出力と同じサイズのベクトルにマッピングする学習済みの完全接続層です φ、および、・、・⟩は内積を示します。 cGAN論文は、GAN目標のこのパラメーター化により、モデルが分布p（x）とp（y | x）を同時に学習できることを示しています。

---

- In our setting we don’t necessarily have class labels available, and we also want our conditioning vectors to contain more information than class labels would provide. To this end, we were inspired by previous work on perceptual metrics [18, 49] to replace y with the activations of a pretrained image classification network, C, when applied to x (the ground truth image). We chose to instantiate C as an InceptionV3 [37] network trained on ImageNet [11] with the final softmax removed. We found that it helps to normalize these activations by subtracting the mean activation over the dataset and then dividing the result by its l2 norm. Note that since the discriminator is only used during training, we can condition on the full unmasked image (x), which also means that these activations can be precomputed before training. This conditioning encourages the generated content to semantically match the target image, which especially helps avoid semantic drift in larger extensions. Formally, we replace eq. 3 with



# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. Related Work


