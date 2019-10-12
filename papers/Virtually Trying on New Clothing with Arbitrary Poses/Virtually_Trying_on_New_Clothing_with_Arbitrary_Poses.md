# ■ 論文
- 論文タイトル："xxx"
- 論文リンク：
- 論文投稿日付：
- 被引用数（記事作成時点）：xxx 件
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- xxx

- Therefore, in this work, we introduce a new try-on setting, which enables the changes of both the clothing item and the person’s pose. 

- Towards this end, we propose a pose-guided virtual try-on scheme based on the generative adversarial networks (GANs) with a bi-stage strategy.
    - この目的に向けて、二段階戦略を用いた生成的敵対ネットワーク（GAN）に基づくポーズガイド付き仮想試着計画案を提案します。

- In particular, in the first stage, we propose a shape enhanced clothing deformation model for deforming the clothing item, where the user body shape is incorporated as the intermediate guidance. For the second stage, we present an attentive bidirectional GAN, which jointly models the attentive clothing-person alignment and bidirectional generation consistency.
    - 特に、第一段階では、ユーザーの身体形状が中間ガイダンスとして組み込まれている、衣服アイテムを変形するための形状強化衣服変形モデルを提案します。 2番目の段階では、注意深い双方向GANを提示します。これは、注意深い衣服と人のアライメントと双方向生成の一貫性を共同でモデル化します。

- xxx


# ■ イントロダクション（何をしたいか？）

## x. Introduction

- xxx

---

- Fortunately, for more intuitive exhibition, fashion-oriented e-commerce websites, such as Zalando5, usually display well-posed fashion model images wearing their products as well as the pure product image. In a sense, the tremendous try-on images online have opened the door to the possibility of fulfilling the virtual try-on task with the economic 2D modeling. Although several pioneer researches have achieved promising performance, most of existing efforts can only generate the single-view try-on result, that is, keeping the person’s pose unchanged while simply changing the clothing item. However, in reality, people may want to check different views of themselves in the new clothing item before making the decision on whether to buy it or not. In the light of this, in this work, we define a new virtual try-on task, where given a person image, a desired pose, and a target clothing item, we aim to automatically generate the try-on look of the person with the target clothing item in his/her desired pose, as illustrated in Figure 1.
    - 幸いなことに、より直感的な展示のために、Zalando5などのファッション志向のeコマースWebサイトは、通常、純粋な製品画像だけでなく、製品を着たポーズの良いファッションモデル画像を表示します。ある意味で、オンラインの膨大な試着画像は、経済的な2Dモデリングで仮想試着タスクを実行する可能性への扉を開いた。いくつかの先駆的な研究が有望なパフォーマンスを達成しましたが、既存の努力のほとんどは、シングルビューの試着結果、つまり、衣服を変更するだけで人の姿勢を変えないという結果を生み出すことができるだけです。しかし、実際には、人々はそれを購入するかどうかの決定を下す前に、新しい衣料品で自分の異なる見解をチェックしたいかもしれません。これを踏まえて、この作業では、新しい仮想試着タスクを定義します。ここでは、人物の画像、希望のポーズ、対象の衣服アイテムが与えられ、人の試着外観を自動的に生成することを目指しています。図1に示すように、希望するポーズの対象衣料品。

---

- Indeed, advanced image generation models such as Generative Adversarial Networks (GANs) [8] and Variational Autoencoders (VAEs) [15] have demonstrated remarkable success in various image generation tasks. However, it is non-trivial to directly apply these methods to fulfil our proposed task due to the following challenges.
    - 実際、Generative Adversarial Networks（GAN）[8]やVariation Autoencoders（VAE）[15]などの高度な画像生成モデルは、さまざまな画像生成タスクで顕著な成功を示しています。 ただし、次の課題により、提案されたタスクを実行するためにこれらの方法を直接適用することは簡単ではありません。

1. In the context of virtual try-on, the body shape and the desired pose of the person highly affect the final look of the target clothing item on the person. Accordingly, how to properly deform the new clothing item and seamlessly align with the target person is a major challenge. 
    - 仮想試着のコンテキストでは、人の体の形と望ましいポーズは、その人のターゲット衣料品の最終的な外観に大きな影響を与えます。 したがって、新しい衣料品を適切に変形し、対象者とシームレスに位置合わせする方法は大きな課題です。

2. How to generate the try-on image that maintains not only the detailed visual features of the clothing item, like the texture and color, but also the other body parts of the person, while changing the person pose is another tough challenge.
    - テクスチャや色などの衣料品の詳細な視覚的特徴だけでなく、人の他の身体部分も維持しながら、人のポーズを変更する試着画像を生成する方法は、別の難しい課題です。


3. there is no large-scale benchmark dataset that can support the research of our new virtual try-on task. Therefore, how to create a large-scale dataset constitutes a crucial challenge.
    - 新しい仮想試着タスクの研究をサポートできる大規模なベンチマークデータセットはありません。 したがって、大規模なデータセットを作成する方法は非常に重要な課題です。

---

- To address the aforementioned challenges and guarantee the try-on quality, we present a pose-guided virtual try-on scheme with two stages, similar to several state-of-the-art coarse-to-fine pipelines [21, 41]. In particular, in the first stage, we propose a shape enhanced clothing deformation approach working on deforming the given clothing item naturally to match the target body shape of the person, which can be internally predicted from the person’s desired pose. In the second stage, our scheme focuses on generating the try-on image based on the deformed clothing item above, the conditional person image and the desired pose. In particular, we present an attentive bidirectional generative adversarial network to synthesize the realistic try-on image, named AB-GAN, which jointly models the attentive clothing-person alignment and bidirectional generation consistency. Pertaining to the evaluation, we create a new large-scale FashionTryOn dataset from the fashion-oriented e-commerce website Zalando6 , consisting of 28, 714 triplets.
    - 前述の課題に対処し、試着の品質を保証するために、いくつかの最先端の粗から細パイプラインに類似した2段階のポーズガイド付き仮想試着スキームを提示します[21、41]。特に、最初の段階では、与えられた衣服を自然に変形させて、人の目標の体型に一致させる形状強化衣服変形アプローチを提案します。第2段階では、上記の変形された衣服アイテム、条件付き人物画像、および望ましいポーズに基づいて試着画像を生成することに焦点を当てています。具体的には、AB-GANという名前の現実的な試着画像を合成するために、注意深い双方向の生成的敵対ネットワークを提示します。評価に関連して、28,714のトリプレットで構成される、ファッション指向の電子商取引WebサイトZalando6から新しい大規模なFashionTryOnデータセットを作成します。

- The main contributions are summarized as follows:

- We present a novel pose-guided virtual try-on scheme in a bi-stage manner. To our best knowledge, we are the first to address the new task of generating realistic try-on images with any desired pose, which has both great theoretical and practical significance.
    - 二段階の方法で新しいポーズガイド付き仮想試着スキームを提示します。 私たちの知る限り、私たちは、理論上および実用上の大きな意義を持つ、任意のポーズで現実的な試着画像を生成するという新しいタスクに取り組む最初の企業です。

- We propose a shape enhanced clothing deformation model, which aims to generate the warped clothing item based on both the target body shape and the desired pose. In addition, we present an attentive bidirectional generative adversarial network to synthesize the final try-on images, which simultaneously regularizes the attentive clothing-person alignment and the bidirectional generation consistency.
    - 私たちは、形状を強化した衣服変形モデルを提案します。これは、ターゲットの身体形状と目的のポーズの両方に基づいて、反った衣服を生成することを目指しています。 さらに、注意深い双方向の生成的敵対ネットワークを提示して、最終的な試着画像を合成します。これにより、注意深い衣服と人の位置合わせと双方向生成の一貫性が同時に調整されます。

- We create a large-scale benchmark dataset, FashionTryOn, and extensive experiments conducted on that demonstrate the superiority of our proposed scheme over the state-of-the- art methods. Moreover, we have released the FashionTryOn dataset and codes to benefit other researchers7.
    - 大規模なベンチマークデータセットであるFashionTryOnを作成し、最新の手法に対する提案されたスキームの優位性を示す広範な実験を実施しました。 さらに、他の研究者のために、FashionTryOnデータセットとコードをリリースしました7。

# ■ 結論

## x. Conclusion


# ■ 何をしたか？詳細

## 3 METHODOLOGY

- In this work, we aim to fulfil the virtual try-on task comprehensively by not only naturally transferring the given clothing item to the corresponding part of the person, but also accurately transforming the person pose to get a novel view of the person. Formally, given a person image IA, a desired pose PB, and a new clothing item c, our goal is to learn a generator to synthesize the person image IB in the new clothing item c with the pose PB .
    - この作業では、与えられた衣服を人の対応する部分に自然に転送するだけでなく、人の新しい視点を得るために人のポーズを正確に変換することにより、仮想試着タスクを包括的に実現することを目指しています。正式には、人物画像IA、所望のポーズPB、および新しい衣服アイテムcが与えられると、新しい衣服アイテムcの人物画像IBをポーズPBと合成するジェネレーターを学習することが目標です。

- To guarantee the try-on effect and alleviate the burden of the generator, we propose a bi-stage pose-guided virtual try-on GAN network, as illustrated in Figure 2. 
    - 試着効果を保証し、ジェネレーターの負担を軽減するために、図2に示すように、2段階のポーズガイド付き仮想試着GANネットワークを提案します。

- The first stage focuses on deforming the given clothing item c according to the target pose and the body shape with the Shape Enhanced Clothing Deformation (Sec. 3.1), where the target body shape is predicted as an auxiliary guidance.
    - 最初の段階では、ターゲットの体型が補助的なガイダンスとして予測される「Shape Enhanced Clothing Deformation」を使用して、ターゲットのポーズと体型に応じて特定の衣服cを変形します。

- The second stage works on generating the ultimate try-on image based on the warped clothing item Tθ (c), the given person image IA, the predicted body shape Sˆ, and the target pose PB with the Attentive Bidirectional GAN (Sec. 3.2).
    - 第2段階では、「Attentive Bidirectional GAN」を使用して、ゆがんだ衣服アイテムTθ（c）、指定された人物画像IA、予測体形Sˆ、およびターゲットポーズPBに基づいて、最終的な試着画像を生成します。

### 3.1 Shape Enhanced Clothing Deformation

- In fact, the target clothing item deformation plays a pivotal role in generating the natural try-on image. In a sense, the given clothing item c should be warped according to not only the target pose but also the target human body shape. Due to the fact that the target body shape is not directly available in our context, we first focus on the prediction of the target body shape mask, which acts as an auxiliary guidance on the clothing item deformation.
    - 実際、対象の衣料品の変形は、自然な試着画像の生成に重要な役割を果たします。 ある意味では、与えられた衣服cは、ターゲットのポーズだけでなく、ターゲットの人体の形状に合わせてワープする必要があります。 私たちのコンテキストではターゲットの身体形状が直接利用できないという事実のために、まずターゲットの身体形状マスクの予測に焦点を合わせます。これは、衣料品の変形に関する補助的なガイダンスとして機能します。

#### Body Shape Mask Prediction.

- In fact, owing to the recent advance of deep neural networks [24], several efforts have been dedicated to the pose-guided parsing, which aims to learn the interplay between the human parsing and human pose. Inspired by [5], we propose to predict the target body shape mask based on the given target pose and the conditional body shape of the given person image.
    - 実際、ディープニューラルネットワーク[24]の最近の進歩により、人間の構文解析と人間のポーズとの相互作用を学習することを目的としたポーズガイド型構文解析にいくつかの努力が注がれています。 [5]に着想を得て、与えられたターゲットポーズと与えられた人物画像の条件付きボディ形状に基づいてターゲットボディ形状マスクを予測することを提案します。

---

- xxx

---

- In particular, we devise the body shape mask prediction network P with the encoder-decoder architecture, where the concatenation of the given body shape mask SA and desired pose PB are fed as the input. As the body shape mask prediction can be deemed as a set of binary classification problems, on the top of the decoder, we adopt the cross-entropy loss for each entry as follows:
    - 特に、エンコーダー/デコーダーアーキテクチャを備えたボディ形状マスク予測ネットワークPを考案します。このアーキテクチャでは、所定のボディ形状マスクSAと所望のポーズPBの連結が入力として供給されます。 ボディ形状マスクの予測は一連のバイナリ分類問題と見なすことができるため、デコーダーの上部で、次のように各エントリのクロスエントロピー損失を採用します。

---

- xxx

#### Clothing Item Deformation.


## 3.2 Attentive Bidirectional GAN (AB-GAN)

- Having obtained the warped clothing item Tθ (c), we can proceed to the presentation of our pose-guided try-on network, AB-GAN, which aims to synthesize the ultimate try-on image with the desired pose and the naturally deformed clothing item. Due to the huge success of GANs in various image generation tasks, we adopt the GAN as the backbone of our pose-guided try-on network. A typical GAN consists of a generator G and a discriminator D, between whom a min-max strategy game would be performed. The generator G attempts to fool the discriminator D by generating realistic images, while the discriminator D tries to distinguish the synthesized fake images from the real ones.

---

- xxx

- To accomplish the above generator, we introduce a human feature encoder Fhuman, a clothing feature encoder Fclothing and a unified try-on image decoder Fdec to our generator G. In particular, the human feature encoder Fhuman works on embedding the conditional person image IA and the desired pose PB and the predicted body shape mask SˆB, while the clothing feature encoder Fclothing is designed to extract the key features of the warped clothing item Tθ (c).

- Then the network seamlessly fuses the human features and clothing features by the dilation-based bottleneck, which has been proved to be effective in image inpainting [37]. Thereafter, the fused features would be decoded to the target person image Iˆ by the try-on image decoder B Fdec .

- For each encoder, we adopt the UNet network [11] with skip connections. In a sense, the skip connections between Fhuman and Fdec serve to propagate the human appearance and the desired pose, while those between Fclothinд and Fdec work on transferring the features of desired clothing item.

# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. Related Work


