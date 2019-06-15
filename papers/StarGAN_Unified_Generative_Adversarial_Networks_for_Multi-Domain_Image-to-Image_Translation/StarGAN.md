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

![image](https://user-images.githubusercontent.com/25688193/59489324-cfa1e400-8ebc-11e9-9ae4-1d207f6e7fee.png)

- > Figure 2. Comparison between cross-domain models and our proposed model, StarGAN.
    - > 図２：cross-domainモデルと我々の手法 StarGAN との間の比較

- > (a) To handle multiple domains, cross- domain models should be built for every pair of image domains.
    - > (a) 複数のドメインを処理するためには、cross- domain モデルは画像のドメインの全てのペア毎に構築されなくてはならない。

- > (b) StarGAN is capable of learning mappings among multiple domains using a single generator.
    - > StarGAN は、１つの生成器を使用して、複数のドメイン間の学習写像を可能にする。

- > The figure represents a star topology connecting multi-domains.
    - > 図は、複数のドメインを結合する星型のトポロジーを表している。

---

- As a solution to such problems we propose StarGAN, a novel and scalable approach capable of learning mappings among multiple domains.
    - このような問題の解決法として、我々は、StarGAN を提案する。
    - これは、複数のドメイン間の学習写像を可能にする新しいスケーラビリティーな手法である。

- As demonstrated in Fig. 2 (b), our model takes in training data of multiple domains, and learns the mappings between all available domains using only a single generator.
    - 図２の (b) で示されたように、我々のモデルは複数のドメインの学習用データを受け取り、１つの生成器のみを使用して、全ての可能なドメイン間の写像を学習する。

- The idea is simple.

- Instead of learning a fixed translation (e.g., black-to-blond hair), our generator takes in as inputs both image and domain information, and learns to flexibly translate the image into the corresponding domain.
    - 固定された変換（例えば、黒髪から茶色の髪での変換）を学習する代わりに、
    - **我々の生成器は、入力として画像とドメインの情報の両方を受け取り、**
    - 画像を対応するドメインへ柔軟に変換することを学習する。

- We use a label (e.g., binary or one-hot vector) to represent domain information.
    - **我々は、ドメインの情報を表現するために、ラベル（例えば、バイナリ or one-hot ベクトル）を使用する。**

- During training, we randomly generate a target domain label and train the model to flexibly translate an input image into the target domain. 
- 学習中、目標のドメインのラベルをランダムに生成し、
- 入力画像を目標ドメインに、柔軟に変換するために、モデルを学習する。

- By doing so, we can control the domain label and translate the image into any desired domain at testing phase.
    - そうすることによって、ドメインラベルを制御できるようになり、テストフェイズにおいて、画像を任意の望ましいドメインに変換することが出来る。

---

- We also introduce a simple but effective approach that enables joint training between domains of different datasets by adding a mask vector to the domain label.
    - 我々はまた、マスクベクトルをドメインラベルに追加することによって、異なるデータセットのドメイン間の一緒に学習することを可能にするような、シンプルであるが効果的なアプローチを提示する。

- Our proposed method ensures that the model can ignore unknown labels and focus on the label provided by a particular dataset.
    - 我々の提案された手法は、モデルが、未知のラベルを無視し、特定のデータセットによって提供されたラベルに焦点を当てることが出来るということを保証する。

- In this manner, our model can perform well on tasks such as synthesizing facial expressions of CelebA images using features learned from RaFD, as shown in the right- most columns of Fig. 1.
    - この形式では、図１の最右端で示されているように、RaFD から学習された特徴を用いて、CelebA 画像の表情の合成といったようなタスクを、うまく実行することが出来る。

- As far as our knowledge goes, our work is the first to successfully perform multi-domain image translation across different datasets.
    - 我々の知る限りでは、我々の研究は、異なるデータセットに渡っての、複数のドメインでの画像変換の最初の成功である。

---

- Overall, our contributions are as follows:
    - We propose StarGAN, a novel generative adversarial network that learns the mappings among multiple domains using only a single generator and a discriminator, training effectively from images of all domains.
        - 我々は StarGAN を提示する。これは、１つの生成器と識別器のみを使用した複数ドメイン間の写像を学習し、全てのドメインの画像から効果的に学習するような、新しい GAN である。

    - We demonstrate how we can successfully learn multi- domain image translation between multiple datasets by utilizing a mask vector method that enables StarGAN to control all available domain labels.
        - 我々は、StarGAN に全ての利用可能なドメインラベルを制御することを可能にする手法であるマスクベクトルを利用することにより、複数のデータセット感の複数のドメイン変換をうまく学習する方法を実証する。

    - We provide both qualitative and quantitative results on facial attribute transfer and facial expression synthesis tasks using StarGAN, showing its superiority over baseline models.
        - 我々は、StarGAN を使用して、表情の属性の変換と表情の合成タスクでの、定量的な結果と定性的な結果を提供し、
        - それがベースラインモデルより優れていることを示す。

# ■ 結論

## 6. Conclusion

-  In this paper, we proposed StarGAN, a scalable image- to-image translation model among multiple domains using a single generator and a discriminator.
    - この論文では、我々は StarGAN を提案した。これは、１つの生成器と識別器を使用した、複数のドメイン間のスケーラビリティーな image-to-image 変換である。

- Besides the advantages in scalability, StarGAN generated images of higher visual quality compared to existing methods [16, 23, 33], owing to the generalization capability behind the multi-task learning setting.
    - スケーラビリティーにける利点に加えて、
    - 複数のタスクの学習設定の背後の汎化能力 [generalization capability] のために、
    - StarGAN は既存の手法に比べて視覚的な品質の高い画像を生成する。

- In addition, the use of the proposed simple mask vector enables StarGAN to utilize multiple datasets with different sets of domain labels, thus handling all available labels from them.
    - 加えて、提案されたシンプルなマスクベクトルの使用は、StarGAN に、異なるドメインのセットで複数のデータセットを使用することを可能にする。

- We hope our work to enable users to develop interesting image translation applications across multiple domains.
    - 我々は、我々の研究が、ユーザーに、複数のドメインに渡っての興味深い画像変換の応用を開発することを可能にすることを望んでいる。

# ■ 何をしたか？詳細

## 3. Star Generative Adversarial Networks

- We first describe our proposed StarGAN, a framework to address multi-domain image-to-image translation within a single dataset.
    - 我々ははじめに、我々の提案した　StarGAN を説明する。これは、１つのデータセットでの複数のドメインの image-to-image 変換を解決するフレームワークである。

- Then, we discuss how StarGAN incorporates multiple datasets containing different label sets to flexibly perform image translations using any of these labels.
    - 次に、我々は、これらの（＝異なるラベル集合の）いずれかのラベルを使用して、画像変換を柔軟に実行するために、StarGAN がどのようにして異なるラベル集合を含むような複数のデータセットを取り入れる [incorporates] のかを議論する。


### 3.1. Multi-Domain Image-to-Image Translation

- Our goal is to train a single generator G that learns mappings among multiple domains.
    - 我々のゴールは、複数のドメイン感の写像を学習する１つの生成器を学習することである。

- To achieve this, we train G to translate an input image x into an output image y conditioned on the target domain label c, G(x, c) → y.
    - これを達成するために、
    - **我々は、目標ドメインラベル c で条件付けされた、入力画像 x から出力画像 y への変換 G(x, c) → y を G に学習する。**

- We randomly generate the target domain label c so that G learns to flexibly translate the input image.
    - **我々は、G が柔軟に入力画像を変換するように、目標ドメインラベルをランダムに生成する。**

- We also introduce an auxiliary classifier [22] that allows a single discriminator to control multiple domains.
    - **我々はまた、補助的な分類器 [auxiliary classifier] [22] を提案する。**
    - **これは、１つの識別器で、複数のドメインを制御することを可能にする。**

> 補助的な分類器 [auxiliary classifier] [22] : ACGAN の論文で提案されている補助的な分類器

- That is, our discriminator produces probability distributions over both sources and domain labels, D : x → {D_src(x), D_cls(x)}.
    - 即ち [That is]、我々の識別器は、ソースとドメインラベルの両方の上で、確率分布 ![image](https://user-images.githubusercontent.com/25688193/59498231-f4ec1d80-8ecf-11e9-91ba-97910cab93f0.png) を生成する。

- Fig. 3 illustrates the training process of our proposed approach.
    - 図３は、我々のアプローチの学習プロセスを図示している。

> Generatorに今どのドメインへと変化させようとしているか表すベクトルをconcatして加え、Discriminatorにその画像のReal/Fakeを見分けるだけでなく、どのドメインの画像かの分類も同時に行うようにしています。これによって、多ドメインの多ドメインへの変換を一つの学習器によって行うことが可能となります。

---

![image](https://user-images.githubusercontent.com/25688193/59498315-249b2580-8ed0-11e9-86c6-b246a9d1753c.png)

- > Figure 3. Overview of StarGAN, consisting of two modules, a discriminator D and a generator G.
    - > 図３：StarGAN の概要、識別器と生成器の２つのモジュールで構成されている。

- > (a) D learns to distinguish between real and fake images and classify the real images to its corresponding domain.
    - > (a) 識別器 D は本物画像と偽物画像を区別することを学習し、本物画像をそれに対応するドメインに分類する。

- > (b) G takes in as input both the image and target domain label and generates an fake image. The target domain label is spatially replicated and concatenated with the input image.
    - > (b) 生成器 G は、入力として、画像と目標ドメインラベルの両方を受け取し、偽物画像を生成する。目標ドメインラベルは、部分的に複製され [replicated]、入力画像に連結される。

- > (c) G tries to reconstruct the original image from the fake image given the original domain label.
    - > (c) 生成器 G は、元のドメインラベルを与えれば、偽物画像から元の入力画像を再構成しようとする。

> CycleGAN の逆写像に対応した構造

- > (d) G tries to generate images indistinguishable from real images and classifiable as target domain by D.
    - > (d) 生成器 G は、本物画像から区別のつかない画像を生成しようとし、識別器によって目標ドメインとして分類可能な画像を生成しようとする。


#### Adversarial Loss. 

- To make the generated images indistinguishable from real images, we adopt an adversarial loss
    - 本物画像から区別のつかない生成された画像を作るために、我々は敵対的損失関数を適用した。

![image](https://user-images.githubusercontent.com/25688193/59499905-56fa5200-8ed3-11e9-8911-595d0d5558a8.png)

- where G generates an image G(x, c) conditioned on both the input image x and the target domain label c, while D tries to distinguish between real and fake images.
    - ここで、生成器 G は、入力画像 x と目標ドメインラベル c の両方で条件付けされた画像　G(x,c) を生成する。
    - 一方で、識別器 D は本物画像と偽物画像の間を区別しようとする。

- In this paper, we refer to the term Dsrc(x) as a probability distribution over sources given by D.
    - この論文では、D_src(x) の項を、識別器 D によって与えられたソース上の確率分布としてみなす。

- The generator G tries to minimize this objective, while the discriminator D tries to maximize it.
    - 生成器 G は、この目的関数を最小化しようとし、一方で、識別器 D はそれを最大化しようとする。

#### Domain Classification Loss.

- For a given input image x and a target domain label c, our goal is to translate x into an output image y, which is properly classified to the target domain c.
    - 与えられた入力画像 x と目標ドメインラベル c に対して、
    - 我々のゴールは、x を、目標ドメイン c で適切に分類される出力画像 y に変換することである。

- To achieve this condition, we add an auxiliary classifier on top of D and impose the domain classification loss when optimizing both D and G. 
    - この条件を達成するために、
    - 我々は、識別器 D の先頭に、補助的な分類器を追加し、
    - 識別器 D と生成器 G の両方を最適化するとき、"domain classification loss" を課す [impose]。

- That is, we decompose the objective into two terms: a domain classification loss of real images used to optimize D, and a domain classification loss of fake images used to optimize G.
    - 即ち、目的関数を２つの項で分解する [decompose]。
    - 即ち、識別器 D を最適化するために使用される、本物画像の domain classification loss
    - そして、生成器 G を最適化するために使用される、偽物画像の domain classification loss

- In detail, the former is defined as
    - 詳細には、前者は以下のように定義される。

![image](https://user-images.githubusercontent.com/25688193/59514904-d9e1d380-8ef8-11e9-99ca-1d86794821f8.png)

- where the term D_cls(c′|x) represents a probability distribution over domain labels computed by D.
    - ここで、項 D_cls(c′|x) は、識別器 D によって計算されたドメインラベル上の確率分布を表している。

- By minimizing this objective, D learns to classify a real image x to its corresponding original domain c′.
    - この目的関数を最小化することによって、
    - 識別器 D は、本物画像 x を、それと一致する元のドメイン c' に分類することを学習する。

- We assume that the input image and domain label pair (x, c′) is given by the training data.
    - 我々は、入力画像とドメインラベルのペア (x, c′) が、学習用データによって与えられているということを仮定する。

- On the other hand, the loss function for the domain classification of fake images is defined as
    - 他方では、偽物画像のドメイン分類の対しての損失関数は、以下のように定義される。

![image](https://user-images.githubusercontent.com/25688193/59514927-e9f9b300-8ef8-11e9-90cb-8d069c76688e.png)

- In other words, G tries to minimize this objective to generate images that can be classified as the target domain c.

---

![image](https://user-images.githubusercontent.com/25688193/59548491-2ece2a00-8f8b-11e9-94bb-52f039c86629.png)

#### Reconstruction Loss.

- By minimizing the adversarial and classification losses, G is trained to generate images that are realistic and classified to its correct target domain.
    - 敵対的損失関数と classification losses を最小化することによって、
    - 生成器 G は、リアルでそれを正しい目標ドメインに識別するような画像を生成することを学習される。

- However, minimizing the losses (Eqs. (1) and (3)) does not guarantee that translated images preserve the content of its input images while changing only the domain-related part of the inputs.
    - しかしながら、損失関数を最小化すること（式 (1), (3)）は、
    - 変換された画像が、その入力画像の内容を保存しながら、一方で、入力のドメインに関連した部分のみを変化させるということを保証しない。

- To alleviate this problem, we apply a cycle consistency loss [9, 33] to the generator, defined as
    - この問題を軽減する [alleviate] ために、我々は、生成器に以下で定義される  cycle consistency loss を適用する。

![image](https://user-images.githubusercontent.com/25688193/59547892-3be61b80-8f81-11e9-8f88-b84da31a0e17.png)

- where G takes in the translated image G(x, c) and the original domain label c′ as input and tries to reconstruct the original image x.
    - ここで、G は変換された画像 G(x, c) と元の目標ラベル c' を入力としてを受け取り、
    - 元の画像 x を再構成しようとする。

- We adopt the L1 norm as our reconstruction loss.
    - 我々は、我々の再構成損失 [reconstruction loss] として、L1 ノルムを採用した。

- Note that we use a single generator twice, first to translate an original image into an image in the target domain and then to reconstruct the original image from the translated image.
    - 我々は、１つの生成器を２回使用することに注意。
    - １つ目は、元の画像を目標ドメインにおける１つの画像に変換し、
    - 次に、変換された画像から元の画像を再構成する。

#### Full Objective. 

- Finally, the objective functions to optimize G and D are written, respectively, as
    - 最後に、生成器 G と識別器 D を最適化するための目的関数は、それぞれ、以下のように書かれる。

![image](https://user-images.githubusercontent.com/25688193/59548020-b879f980-8f83-11e9-9c54-fefc6738b271.png)

- where λ_cls and λ_rec are hyper-parameters that control the relative importance of domain classification and reconstruction losses, respectively, compared to the adversarial loss.
    - ここで、λ_cls と λ_rec は、それぞれ domain classification と reconstruction losses の敵対的損失関数と比較しての相対的な重要度を制御するパイパーパラメーターである。

- We use λ_cls = 1 and λ_rec = 10 in all of our experiments.
    - 全ての実験で、λ_cls = 1 と λ_rec = 10 を使用する。


### 3.2. Training with Multiple Datasets

- An important advantage of StarGAN is that it simultaneously incorporates multiple datasets containing different types of labels, so that StarGAN can control all the labels at the test phase.
    - StarGAN の重要な利点は、異なるラベルの種類を含んでいる複数のデータセットを同時に取り込むことであり、
    - そういうわけで、StarGAN は、テストフェイズで全てのラベルを制御することが出来る。

- An issue when learning from multiple datasets, however, is that the label information is only partially known to each dataset.
    - しかしながら、複数のデータセットから学習するときの１つの問題は、
    - ラベル情報が各データセットで部分的にしか知られていないということである。

- In the case of CelebA [19] and RaFD [13], while the former contains labels for attributes such as hair color and gender, it does not have any labels for facial expressions such as ‘happy’ and ‘angry’, and vice versa for the latter.
    - CelebA と RaFD のケースでは、
    - 前者はが髪の色や性別といった属性に対してのラベルを含むのに対し、幸福・怒りのような表情のラベルを持たない。
    - 後者はその逆である。

- This is problematic because the complete information on the label vector c′ is required when reconstructing the input image x from the translated image G(x, c) (See Eq. (4)).
    - これは問題である。
    - なぜならば、変換された画像 G(x,c) から入力画像 x を再構成するときに、ラベルベクトル c' での完全な情報が要求されるためである。（式 (4) を参照）

> 式 (4) の損失関数は、L1ノルムの式であり、これはピクセルの単位での情報で、全てのピクセル間情報（＝完全な情報）を要求している。

#### Mask Vector.

- To alleviate this problem, we introduce a mask vector m that allows StarGAN to ignore unspecified labels and focus on the explicitly known label provided by a particular dataset.
    - この問題を軽減する [alleviate] ために、我々は

- In StarGAN, we use an n-dimensional one-hot vector to represent m, with n being the number of datasets.

- In addition, we define a unified version of the label as a vector

![image](https://user-images.githubusercontent.com/25688193/59548131-312d8580-8f85-11e9-90a8-e6cb3c63511f.png)

- where [·] refers to concatenation, and ci represents a vector for the labels of the i-th dataset.

- The vector of the known label ci can be represented as either a binary vector for bi- nary attributes or a one-hot vector for categorical attributes.

- For the remaining n−1 unknown labels we simply assign zero values.

- In our experiments, we utilize the CelebA and RaFD datasets, where n is two.



# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


