# ■ 論文
- 論文タイトル："StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation"
- 論文リンク：https://arxiv.org/abs/1711.09020
- 論文投稿日付：2017/11/24
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- Recent studies have shown remarkable success in image- to-image translation for two domains.
    - 最近の研究では、２つのドメイン間の image-to-image 変換において、特筆すべきすべき成功を見せている。

> ドメイン：ここでの意味は、同じ属性（金髪、男、女など）をもつ画像の組のこと。一枚の画像が複数のドメインに所属することもある。

- However, existing approaches have limited scalability and robustness in handling more than two domains, since different models should be built independently for every pair of image domains.
    - しかしながら既存の [existing] アプローチは、
    - 画像ドメインのペア毎に対して、異なるモデルを個別に [independently] 構築するので、
    - ２つ以上のドメインを処理することにおいて、スケーラビリティー [scalability] やロバスト性に限界がある。

> スケーラビリティ【scalability】とは、機器やソフトウェア、システムなどの拡張性、拡張可能性のこと。

- To address this limitation, we propose StarGAN, a novel and scalable approach that can perform image-to-image translations for multiple domains using only a single model.
    - この限界を解決するために、我々は、StarGAN を提案する。
    - （これは、）１つのモデルのみを使用する、複数のドメインに対しての image-to-image 変換を実行することが可能な、新しく、スケーラビリティーなアプローチである。

- Such a unified model architecture of StarGAN allows simultaneous training of multiple datasets with different domains within a single network.
    - このような StarGAN の統一された [unified] モデルは、１つのネットワーク内の異なるドメインを持つ複数のデータセットの学習を、同時に行うことを可能にする。

- This leads to StarGAN’s superior quality of translated images compared to existing models as well as the novel capability of flexibly translating an input image to any desired target domain.
    - このことは、既存のモデルと比較して StarGAN の優れた変換画像の品質と同様にして、任意の望ましい目標ドメインへの入力画像の柔軟な変換の新しい能力 [capability] を導く。

- We empirically demonstrate the effectiveness of our approach on a facial attribute transfer and a facial expression synthesis tasks
    - 我々は、顔の属性の変換タスクと、顔の表情の合成タスクにおいて、我々の手法の効果性を実験的に [empirically] 実証する。


# ■ イントロダクション（何をしたいか？）

## 1. Introduction

- The task of image-to-image translation is to change a particular aspect of a given image to another, e.g., changing the facial expression of a person from smiling to frowning (see Fig. 1).
    - image-to-image 変換タスクは、与えたれた画像の特定の側面を別の画像に変形することである。
    - 例えば、人部s角顔の表情を、笑いからしかめっ面（＝不機嫌な表情）[] に変化させること（図１）

- This task has experienced significant improvements following the introduction of generative adversarial networks (GANs), with results ranging from changing hair color [9], reconstructing photos from edge maps [7], and changing the seasons of scenery images [33].
    - このタスクは、GAN の導入に従って、大幅な改善を経験した。

---

- Given training data from two different domains, these models learn to translate images from one domain to the other.
    - ２つの異なるドメインでからの学習用データを与えれば、これらのモデルは、１つのドメインから別のドメインへの画像の変換を学習する。

- We denote the terms attribute as a meaningful feature inherent in an image such as hair color, gender or age, and attribute value as a particular value of an attribute, e.g., black/blond/brown for hair color or male/female for gender.
    - 我々は、属性 [attribute] という用語を、髪の色や、性別、年齢のような、画像に固有の [inherent] 意味のある特徴として表す。
    - そして、属性の値 [attribute value] を、属性の特定の値（髪の色に対しては、黒・ブロンド・茶色。性別に対しては、男。女）として表す。

- We further denote domain as a set of images sharing the same attribute value.
    - 更に、ドメイン [domain] を、同じ属性の値を共有している画像の集合として、表す。

- For example, images of women can represent one domain while those of men represent another
    - 例えば、女性の画像は、１つのドメインを表現する。
    - 一方で、男の画像は、別のドメインを表現する。

---

- Several image datasets come with a number of labeled attributes.
    - いくつかの画像データセットは、いくつかの [a number of] ラベリングされた属性を搭載している [come with]。

- For instance, the CelebA[19] dataset contains 40 labels related to facial attributes such as hair color, gender, and age, and the RaFD [13] dataset has 8 labels for facial expressions such as ‘happy’, ‘angry’ and ‘sad’.
    - 例えば、CelebA データセットは、髪の色、性別、年齢のような顔の属性に関連した 40 個のラベルを含んでいる。
    - RaFD データセットは、幸福、怒り、悲しみなどの顔の表情に対しての 8 個のラベルを持つ。

- These settings enable us to perform more interesting tasks, namely multi-domain image-to-image translation, where we change images according to attributes from multiple domains.
    - これらの設定は、複数のドメインからの属性に従って画像を変換する、multi-domain image-to-image 変換と呼ばれる、より興味深いタスクの実行を可能にする。

- The first five columns in Fig. 1 show how a CelebA image can be translated according to any of the four domains, ‘blond hair’, ‘gender’, ‘aged’, and ‘pale skin’.
    - 図１の最初の５つの列は、どのように CelebA 画像が、４つのドメイン（茶色の髪、性別、年齢、薄い [pale] 肌）のいずれか [any of] に従って変換されているかを示している。

- We can further extend to training multiple domains from different datasets, such as jointly training CelebA and RaFD images to change a CelebA image’s facial expression using features learned by training on RaFD, as in the rightmost columns of Fig. 1.
    - **我々は更に、異なるデータセットから、複数のドメインの学習までに及ぶ [extend to] ことが出来る。**
    - **図１の最右端の列のように、RaFDの学習によって学習された特徴を用いて、CelebA 画像の顔表情を変形するために、CelebA と RaFD 画像を一緒に [jointly] 学習するといったように、**

---

- > Figure 1. Multi-domain image-to-image translation results on the CelebA dataset via transferring knowledge learned from the RaFD dataset.
    - > 図１：RaFDデータセットから学習された知識 [knowledge] を転送経由での、CelebAデータセットでの Multi-domain image-to-image 変換結果。

- > The first and sixth columns show input images while the remaining columns are images generated by StarGAN. 
    - > 最初と６番目の列は、入力画像を示している。一方で、残りの列は、StarGAN によって生成された画像を示している。

- > Note that the images are generated by a single generator network, and facial expression labels such as angry, happy, and fearful are from RaFD, not CelebA.
    - > 画像は、１つの生成器ネットワークによって生成され、怒り・悲しみといったような表情ラベルは RaFDから生成されたものであり、CelebA からではないことに注意。


---

- However, existing models are both inefficient and ineffective in such multi-domain image translation tasks. 
    - しかしながら、既存のモデルでは、このようなマルチドメイン画像変換タスクにおいて、非効果的で非効率的の両方である。

- Their inefficiency results from the fact that in order to learn all mappings among k domains, k(k−1) generators have to be trained.
    - このような非効果的な結果は、k 個のドメイン間の全ての写像を学習するには、k(k-1) 個の生成器が学習されなくてはならないという事実から生じる。

- Fig. 2 (a) illustrates how twelve distinct generator networks have to be trained to translate images among four different domains.

- Meanwhile, they are ineffective that even though there exist global features that can be learned from images of all domains such as face shapes, each generator cannot fully utilize the entire training data and only can learn from two domains out of k.
    - それと同時に [Meanwhile]、
    - 顔の輪郭などの全てのドメインの画像から学習することが出来る大域的な特徴が存在しているにも関わらず、
    - 各生成器は、学習用データの全体を完全に利用できず、k の内の２つのドメインから学習出来るのみであるというように、
    - それらは非効果的である。

- Failure to fully utilize training data is likely to limit the quality of generated images.
    - 学習データを完全に利用しないことは、生成された画像の品質を制限しそうだ。

- Furthermore, they are incapable of jointly training domains from different datasets because each dataset is partially labeled, which we further discuss in Section 3.2.
    - 更には、それらは、異なるデータセットから一緒にドメインを学習することが出来ない [incapable of]。
    - なぜならば、各データセットは部分的にラベル化されているため。このことは、セクション 3.2 で更に議論する。

---

- As a solution to such problems we propose StarGAN, a novel and scalable approach capable of learning mappings among multiple domains.

- As demonstrated in Fig. 2 (b), our model takes in training data of multiple domains, and learns the mappings between all available domains using only a single generator.

- The idea is simple.

- Instead of learning a fixed translation (e.g., black-to-blond hair), our generator takes in as inputs both image and domain information, and learns to flexibly translate the image into the corresponding domain.

- We use a label (e.g., binary or one-hot vector) to represent domain information. During training, we randomly generate a target domain label and train the model to flexibly translate an input image into the target domain. 

- By doing so, we can control the domain label and translate the image into any desired domain at testing phase.


# ■ 結論

## x. 論文の項目名 (Conclusion)


# ■ 何をしたか？詳細

## x. 論文の項目名


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


