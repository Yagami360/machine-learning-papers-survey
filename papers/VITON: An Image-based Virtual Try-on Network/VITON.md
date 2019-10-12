# ■ 論文
- 論文タイトル："VITON: An Image-based Virtual Try-on Network"
- 論文リンク：https://arxiv.org/abs/1711.08447
- 論文投稿日付：
- 被引用数（記事作成時点）：xxx 件
- 著者（組織）：
- categories：
sa
# ■ 概要（何をしたか？）

## Abstract

- We present an image-based VIirtual Try-On Network (VITON) without using 3D information in any form, which seamlessly transfers a desired clothing item onto the corresponding region of a person using a coarse-to-fine strategy. Conditioned upon a new clothing-agnostic yet descriptive person representation, our framework first generates a coarse synthesized image with the target clothing item over- laid on that same person in the same pose. We further enhance the initial blurry clothing area with a refinement network. The network is trained to learn how much detail to utilize from the target clothing item, and where to apply to the person in order to synthesize a photo-realistic image in which the target item deforms naturally with clear visual patterns. Experiments on our newly collected dataset demonstrate its promise in the image-based virtual try-on task over state-of-the-art generative models.
    - 任意の形式の3D情報を使用せずに、画像ベースの仮想試着ネットワーク（VITON）を提示します。これは、粗から密への戦略を使用して、目的の衣服を人の対応する領域にシームレスに転送します。 新しい衣類に依存しないが説明的な人物表現に基づいて、私たちのフレームワークはまず、同じポーズで同じ人物にターゲット衣類アイテムがオーバーレイされた粗い合成画像を生成します。 洗練されたネットワークを使用して、最初のぼやけた衣服の領域をさらに強化します。 ネットワークは、対象の衣料品からどのくらいのディテールを利用するか、そして対象品が明確な視覚パターンで自然に変形するフォトリアリスティックな画像を合成するために人に適用する場所を学習するように訓練されます。 新しく収集されたデータセットの実験は、最先端の生成モデルを介した画像ベースの仮想試着タスクでの約束を示しています。


# ■ イントロダクション（何をしたいか？）

## x. Introduction

- xxx

---

- xxx

---

- Conditional Generative Adversarial Networks (GANs), which have demonstrated impressive results on image generation [37, 26], image-to-image translation [20] and editing tasks [49], seem to be a natural approach for addressing this problem. In particular, they minimize an adversarial loss so that samples generated from a generator are indistinguishable from real ones as determined by a discriminator, conditioned on an input signal [37, 33, 20, 32]. However, they can only transform information like object classes and attributes roughly, but are unable to generate graphic details and accommodate geometric changes [50]. This limits their ability in tasks like virtual try-on, where visual details and realistic deformations of the target clothing item are required in generated samples.
    - 画像生成[37、26]、画像から画像への変換[20]、および編集タスク[49]で印象的な結果を示した条件付き生成的敵対ネットワーク（GAN）は、この問題に対処するための自然なアプローチのようです。 特に、これらは敵の損失を最小限に抑えるため、ジェネレータから生成されたサンプルは、入力信号を条件とする弁別器によって決定されるように、実際のサンプルと区別できません[37、33、20、32]。 ただし、オブジェクトクラスや属性などの情報は大まかにしか変換できませんが、グラフィックの詳細を生成したり、幾何学的な変化に対応することはできません[50]。 これにより、仮想試着などのタスクでの能力が制限されます。仮想試着では、生成されたサンプルでターゲットの衣服アイテムの視覚的な詳細と現実的な変形が必要になります。

---

- To address these limitations, we propose a virtual try-on network (VITON), a coarse-to-fine framework that seamlessly transfers a target clothing item in a product image to the corresponding region of a clothed person in a 2D image. Figure 2 gives an overview of VITON.
    - これらの制限に対処するために、仮想試着ネットワーク（VITON）を提案します。これは、製品画像内の対象衣服を2D画像内の衣服を着た人の対応する領域にシームレスに転送する coarse-to-fine framework です。図2は、VITONの概要を示しています。

- In particular, we first introduce a clothing-agnostic representation consisting of a comprehensive set of features to describe different characteristics of a person.
    - 特に、人のさまざまな特性を説明するための包括的な機能セットで構成される clothing-agnostic representation を最初に紹介します。

- Conditioned on this representation, we employ a multi-task encoder-decoder network to generate a coarse synthetic clothed person in the same pose wearing the target clothing item, and a corresponding clothing region mask.
    - この表現を条件に、マルチタスクエンコーダーデコーダーネットワークを使用して、対象の衣服とそれに対応する衣服領域マスクを着用した同じポーズの粗い合成衣服を生成します

- The mask is then used as a guidance to warp the target clothing item to account for deformations.
    - 次に、マスクをガイダンスとして使用して、対象の衣服を変形させて変形を補正します。

- Furthermore, we utilize a refinement network which is trained to learn how to composite the warped clothing item to the coarse image so that the desired item is transfered with natural deformations and detailed visual patterns. To validate our approach, we conduct a user study on our newly collected dataset and the results demonstrate that VITON generates more realistic and appealing virtual try-on results outperforming state-of-the-art methods.
    - さらに、自然な変形と詳細な視覚パターンで目的のアイテムが転送されるように、歪んだ衣服アイテムを粗い画像に合成する方法を学習するように訓練された refinement network を利用します。アプローチを検証するために、新しく収集したデータセットでユーザー調査を実施し、その結果は、VITONが最先端の方法よりも優れた、より現実的で魅力的な仮想試着結果を生成することを示しています。


# ■ 結論

## x. Conclusion


# ■ 何をしたか？詳細

## 3. VITON

- The goal of VITON is, given a reference image I with a clothed person and a target clothing item c, to synthesize a new image Iˆ, where c is transferred naturally onto the corresponding region of the same person whose body parts and pose information are preserved. Key to a high-quality synthesis is to learn a proper transformation from product images to clothes on the body. A straightforward approach is to leverage training data of a person with fixed pose wearing different clothes and the corresponding product images, which, however, is usually difficult to acquire.
    - VITONの目標は、衣服を着た人物と対象の衣服アイテムcを含む参照画像Iが与えられ、新しい画像Iˆを合成することです。ここで、cは身体の部分と姿勢情報が保存されている同じ人物の対応する領域に自然に転送されます 。 高品質の合成の鍵は、製品画像から身体の衣服への適切な変換を学習することです。 簡単なアプローチは、異なる服を着た固定ポーズの人のトレーニングデータと、対応する製品画像を活用することですが、これは通常取得が困難です。

---

- In a practical virtual try-on scenario, only a reference image and a desired product image are available at test time. Therefore, we adopt the same setting for training, where a reference image I with a person wearing c and the product image of c are given as inputs (we will use c to refer to the product image of c in the following paper). Now the problem becomes given the product image c and the person’s information, how to learn a generator that not only produces I during training, but more importantly is able to generalize at test time – synthesizing a perceptually convincing image with an arbitrary desired clothing item.
    - 実用的な仮想試着シナリオでは、テスト時に参照画像と目的の製品画像のみを使用できます。 したがって、トレーニングに同じ設定を採用し、cを着用している人の参照画像Iとcの製品画像を入力として提供します（cを使用して、次の論文でcの製品画像を参照します）。 問題は、製品画像cと人の情報、トレーニング中にIを生成するだけでなく、より重要なことはテスト時に一般化できること、つまり知覚的に説得力のある画像を任意の希望の衣服と合成する方法を与えることです。

---

- xxx


### 3.1. Person Representation

- 

### Multi-task Encoder-Decoder Generator

- Given the clothing-agnostic person representation p and the target clothing image c, we propose to synthesize the reference image I through reconstruction such that a natural transfer from c to the corresponding region of p can be learned. In particular, we utilize a multi-task encoder- decoder framework that generates a clothed person image along with a clothing mask of the person as well. In addition to guiding the network to focus on the clothing region, the predicted clothing mask will be further utilized to refine the generated result, as will be discussed in Sec 3.3. The encoder-decoder is a general type of U-net architecture [38] with skip connections to directly share information between layers through bypassing connections.
    - 衣服にとらわれない人物表現pとターゲットの衣服画像cが与えられると、cからpの対応する領域への自然な移動を学習できるように、再構成を通じて参照画像Iを合成することを提案します。 具体的には、マルチタスクエンコーダーデコーダーフレームワークを利用して、衣服を着た人物の画像とその人物の衣服マスクも生成します。 ネットワークをガイドして衣服の領域に焦点を合わせることに加えて、予測された衣服マスクは、セクション3.3で説明するように、生成された結果を改良するためにさらに利用されます。 エンコーダーデコーダーは、バイパス接続を介してレイヤー間で情報を直接共有するスキップ接続を備えた一般的なタイプのU-netアーキテクチャーです[38]。

---

- xxx

---


- A simple way to achieve this is to train the network with an L1 loss, which generates decent results when the target is a binary mask like M0. - However, when the desired output is a colored image, L1 loss tends to produce blurry images [20].
    - これを実現する簡単な方法は、L1損失でネットワークをトレーニングすることです。これにより、ターゲットがM0のようなバイナリマスクである場合、適切な結果が生成されます。
    - ただし、目的の出力がカラー画像の場合、L1損失はぼやけた画像を生成する傾向があります[20]。

---

- The perceptual loss forces the synthesized image to match RGB values of the ground truth image and their activations at different layers in a visual perception model as well, allowing the synthesis network to learn realistic patterns.
- The second term in Eq 1 is a regression loss that encourages the predicted clothing mask M to be the same as M0.
    - 知覚的損失は、合成画像をグラウンドトゥルース画像のRGB値と視覚認識モデルの異なるレイヤーでのそれらの活性化に一致させ、合成ネットワークが現実的なパターンを学習できるようにします。
    - 式1の2番目の項は、予測される衣服マスクMがM0と同じになるように促す回帰損失です。

---

- By minimizing Eqn. 1, the encoder-decoder learns how to transfer the target clothing conditioned on the person representation. While the synthetic clothed person conforms to the pose, body parts and identity in the original image (as illustrated in the third column of Figure 5), details of the target item such as text, logo, etc. are missing. This might be attributed to the limited ability to control the process of synthesis in current state-of-the-art generators. They are typically optimized to synthesize images that look similar globally to the ground truth images without knowing where and how to generate details. To address this issue, VITON uses a refinement network together with the predicted clothing mask M to improve the coarse result I ′ .
    - Eqnを最小化することにより。 1、エンコーダー/デコーダーは、人物表現を条件とするターゲット衣服を転送する方法を学習します。 合成の衣服を着た人は、元の画像のポーズ、身体の部分、およびアイデンティティに適合しますが（図5の3列目を参照）、テキスト、ロゴなどのターゲットアイテムの詳細は失われます。 これは、現在の最先端のジェネレーターで合成プロセスを制御する能力が制限されているためである可能性があります。 通常、これらは、詳細をどこでどのように生成するかを知らなくても、グラウンドトゥルースイメージにグローバルに似たイメージを合成するように最適化されています。 この問題に対処するために、VITONは予測された衣服マスクMとともに改良ネットワークを使用して、粗い結果I ′を改善します。


### 3.3. Refinement Network

- We borrow information directly from the target clothing image c to fill in the details in the generated region of the coarse sample. However, directly pasting the product image is not suitable as clothes deform conditioned on the person pose and body shape. Therefore, we warp the clothing item by estimating a thin plate spline (TPS) transformation with shape context matching [3], as illustrated in Figure 4.
    - ターゲット衣服画像cから直接情報を借用して、粗いサンプルの生成された領域の詳細を埋めます。ただし、製品の画像を直接貼り付けることは、人物のポーズや体型に応じて衣服が変形するため、適切ではありません。したがって、図4に示すように、形状コンテキストマッチング[3]で薄板スプライン（TPS）変換を推定することにより、衣料品をゆがめます。

- More specifically, we extract the foreground mask of c and compute shape context TPS warps [3] between this mask and the clothing mask M of the person, estimated with Eqn 1.
    - より具体的には、cの前景マスクを抽出し、このマスクと式1で推定された人物の衣服マスクMとの間の形状コンテキストTPSワープ[3]を計算します。

- These computed TPS parameters are further applied to transform the target clothing image c into a warped version c′. As a result, the warped clothing image conforms to pose and body shape information of the person and fully preserves the details of the target item. The idea is similar to recent 2D/3D texture warping methods for face synthesis [52, 17], where 2D facial keypoints and 3D pose estimation are utilized for warping. In contrast, we rely on the shape context-based warping due to the lack of accurate annotations for clothing items. Note that a potential alternative to estimating TPS with shape context matching is to learn TPS parameters through a Siamese network as in [23]. However, this is particularly challenging for non-rigid clothes, and we empirically found that directly using context shape matching offers better warping results for virtual try-on.
    - これらの計算されたＴＰＳパラメータはさらに適用されて、ターゲット衣服画像ｃをワープバージョンｃ 'に変換する。その結果、ゆがんだ衣服の画像は、人物のポーズおよび体型情報に適合し、ターゲットアイテムの詳細を完全に保持します。この考え方は、顔の合成のための最近の2D / 3Dテクスチャワーピング方法[52、17]に似ており、2D顔のキーポイントと3Dポーズの推定がワーピングに利用されます。対照的に、衣料品の正確な注釈がないため、形状コンテキストベースのワーピングに依存しています。形状コンテキストマッチングを使用してTPSを推定する代わりに、[23]のようにシャムネットワークを介してTPSパラメーターを学習することが考えられます。ただし、これは非剛性の衣服では特に困難であり、コンテキスト形状マッチングを直接使用すると、仮想試着でより良いワーピング結果が得られることが経験的にわかっています。

---

- The composition of the warped clothing item c′ onto the coarse synthesized image I′ is expected to combine c′ seamlessly with the clothing region and handle occlusion properly in cases where arms or hair are in front of the body. Therefore, we learn how to composite with a refinement network. As shown at the bottom of Figure 2, we first concatenate c′ and the coarse output I′ as the input of our refinement network GR. The refinement network then generates a 1-channel composition mask α ∈ (0, 1)m×n , indicating how much information is utilized from each of the two sources, i.e., the warped clothing item c′ and the coarse image I′. The final virtual try-on output of VITON Iˆis a composition of c′ and I′:
    - 粗い合成画像I 'へのゆがんだ衣服アイテムc'の構成は、c 'を衣服領域とシームレスに組み合わせ、腕または髪が体の前にある場合に適切に閉塞を処理することが期待される。 したがって、洗練ネットワークと合成する方法を学びます。 図2の下部に示すように、最初にc 'と粗出力I'を洗練ネットワークGRの入力として連結します。 リファインメントネットワークは、1チャンネル構成マスクα∈（0、1）m×nを生成し、2つのソース（ワープされた衣類c 'と粗い画像I'）のそれぞれからどのくらいの情報が利用されるかを示します。 VITON Iの最終的な仮想試着出力は、c 'とI'の合成です。

# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. Related Work


