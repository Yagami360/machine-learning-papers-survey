# ■ 論文
- 論文タイトル："Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"
- 論文リンク：https://arxiv.org/abs/1703.10593
- 論文投稿日付：2017/3/20(v1)
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- Image-to-image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs.
    - image-to-image 変換は、整形されたペア画像の学習用データセットを用いて、入力画像と出力画像の間の写像を学習すつことがゴールであるような、コンピュータビジョンとコンピュータグラフィックの類 [a class of] である。   

- However, for many tasks, paired training data will not be available.
    - しかしながら、多くのタスクにおいて、ペア付けされた学習データは利用可能ではない。

- We present an approach for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples.
    - 我々は、ペアリングされた例の欠落 [absence] において（＝ペア画像がない場合において、）、ソース領域 X からターゲット領域 Y へ画像を変換するための学習アプローチを提示する。

- Our goal is to learn a mapping G : X → Y such that the distribution of images from G(X) is indistinguishable from the distribution Y using an adversarial loss.
    - 我々のゴールは、G(X) からの画像の分布が、敵対的損失関数を使用しているような、分布 Y と区別がつかないような、写像 G : X → Y を学習することである。

- Because this mapping is highly under-constrained, we couple it with an inverse mapping F : Y → X and introduce a cycle consistency loss to enforce F(G(X)) ≈ X (and vice versa).
    - この写像は、高い制約下となるので、
    - **それ（＝写像 G : X → Y）を逆写像 F : Y → X で結びつけて [couple] 、F(G(X)) ≈ X（及びその逆）を強制するための cycle consistency loss を提案する。**

- Qualitative results are presented on several tasks where paired training data does not exist, including collection style transfer, object transfiguration, season transfer, photo enhancement, etc.
    - 定性的な結果は、style transfer, object transfiguration, season transfer, photo enhancement などのコレクションを含むような、ペアリングされた学習データが存在しないいくつかのタスクで提示されている。

- Quantitative comparisons against several prior methods demonstrate the superiority of our approach.
    - いくつかの以前の手法に対しての定性的な比較で、我々のアプローチの優位性を実証する。


# ■ イントロダクション（何をしたいか？）

## 1. Introduction

- What did Claude Monet see as he placed his easel by the bank of the Seine near Argenteuil on a lovely spring day in 1873 (Figure 1, top-left)?
    - 1873年の素敵な春の日に、クロード・モネがイーゼルをアルジャントゥイユ近くのセーヌ川のほとり [the bank of (river)] に置いたとき、何が見えましたか（図1、左上）。

> easel ：画架 〔写真の引き伸ばしに用いる〕イーゼル

- A color photograph, had it been invented, may have documented a crisp blue sky and a glassy river reflecting it.
    - カラー写真は、それが発明されたならば、はっきりとした [crisp] 青い空とそれを反映しているガラスのような [glassy] 川を記録したかもしれません。

- Monet conveyed his impression of this same scene through wispy brush strokes and a bright palette.
    - モネは、ぼやけた [wispy] ブラシストロークと明るい [bright] パレットを通して、この同じシーンの印象を伝えました [conveyed]。

---

- What if Monet had happened upon the little harbor in Cassis on a cool summer evening (Figure 1, bottom-left)?
    - もし夏の涼しい夜にモネがカシスの小さな港 [harbor] で起こったとしたらどうなるのであろうか（図1、左下）。

- A brief stroll through a gallery of Monet paintings makes it possible to imagine how he would have rendered the scene: perhaps in pastel shades, with abrupt dabs of paint, and a somewhat flattened dynamic range.
    - モネの絵画 [paintings] のギャラリーを少し散歩する [stroll] と、彼がシーンをどのようにレンダリングしたかを想像することができます。
    - おそらくパステル調の色合いで、突然の [abrupt] ペンキを（ぞんざいに）軽く塗り [dabs of]、そしてやや平坦なダイナミックレンジ。

---

- We can imagine all this despite never having seen a side by side example of a Monet painting next to a photo of the scene he painted. 
    - 彼が描いたシーンの写真の横にモネの絵が並んでいるのを見たことがないにもかかわらず、私たちはこれらすべてを想像することができます。

- Instead, we have knowledge of the set of Monet paintings and of the set of landscape photographs.
    - その代わりに、私たちはモネの絵画のセットと風景写真のセットの知識を持っています。

- We can reason about the stylistic differences between these two sets, and thereby imagine what a scene might look like if we were to “translate” it from one set into the other.
    - これらの２つのセットの間のスタイル（画風）の違いについて、推論し [reason about]、
    − それ故に、あるセットを他のセットに”変換する”した場合に、シーンがどのように見えるだろうかを想像することが出来る。

---

- In this paper, we present a method that can learn to do the same: capturing special characteristics of one image collection and figuring out how these characteristics could be translated into the other image collection, all in the absence of any paired training examples.
    - この論文では、同じことををするように学習出来る手法を提案する。
    - 即ち、ペアリングされた学習サンプル例が全くない状態において、画像コレクションの特別の特徴を抽出し、これらの特徴がどのようにして他の画像のコレクションに変換されるのかを見つけ出す。

---


![image](https://user-images.githubusercontent.com/25688193/59155378-24132100-8ac3-11e9-8c51-1fe48c1de569.png)

- > Figure 2: Paired training data (left) consists of training examples {xi,yi}_{i=1}^N, where the correspondence between xi and yi exists [22].

- > We instead consider unpaired training data (right), consisting of a source set {xi }Ni=1 (xi ∈ X ) and a target set {yj }j =1 (yj ∈ Y ), with no information provided as to which xi matches which yj.

---

- This problem can be more broadly described as image-to-image translation [22], converting an image from one representation of a given scene, x, to another, y, e.g., grayscale to color, image to semantic labels, edge-map to photograph.
    - この問題は、より広くは [broadly]、image-to-image 変換として記述され、
    - 与えられたシーン x の１つの表現から、別の y へ、画像を変換する。
    - 例えば、グレースケールからカラー画像、画像からセマンティックラベル、エッジマップから写真など。

- Years of research in computer vision, image processing, computational photography, and graphics have produced powerful translation systems in the supervised setting, where example image pairs {xi,yi}_{i=1}^N are available (Figure 2, left), e.g., [11, 19, 22, 23, 28, 33, 45, 56, 58, 62]. 
    - コンピュータビジョン、画像処理、計算写真学 [computational photography]、コンピュータグラフィックにおける永年の [Years of] 研究では、画像ペアの例 {xi,yi}_{i=1}^N が利用可能であるような教師あり設定において（図２の左）、パワフルな変換システムを生成した。

> computational photography : 計算写真学（けいさんしゃしんがく、英:computational photography）とは二次元的な画像のみならず奥行きや物体の反射特性などの情報をも撮像素子によりデータとして記録して計算によってその情報を復元する写真。

光学によって被写体の像を得るのではなく、デジタル処理によって画像を生成することを前提としたイメージング技術。

- However, obtaining paired training data can be difficult and expensive. 
    - しかしながら、学習データのペアを取得することは、難関で費用がかかる。

- For example, only a couple of datasets exist for tasks like semantic segmentation (e.g., [4]), and they are relatively small.
    - 例えば、セマンティックセグメンテーションのようなタスクに対しては、２，３個のデータセットだけしか存在せず、それらは比較的少ない。

- Obtaining input-output pairs for graphics tasks like artistic stylization can be even more difficult since the desired output is highly complex, typically requiring artistic authoring.
    - 芸術的な様式化 [stylization] のようなグラフィックタスクに対して、入出力ペアを手に入れることは、望ましい出力が非常の複雑で、芸術的なオーサリング（＝マルチメディアの作成）を比較的要求するので、より困難である。

- For many tasks, like object transfiguration (e.g., zebra↔horse, Figure 1 top-middle), the desired output is not even well-defined.
    - 多くのタクスでは、物体変換（図１の上中央のシマウマ↔馬）のように、望ましい出力は、うまく定義されさえしていない。

---

- We therefore seek an algorithm that can learn to translate between domains without paired input-output examples (Figure 2, right).
    - 我々はそれゆえに、ペア付けされた入出力例なしに、ソースターゲット間の変換を学習することの出来るアルゴリズムを探す。（図２の右）

- We assume there is some underlying relationship between the domains – for example, that they are two different renderings of the same underlying scene – and seek to learn that relationship.
    - 我々は、ソースターゲット間にいくつかの根本的な [underlying] 関係が存在することを仮定する。
    - 即ち、例えば、それらは同じ根本的なシーンの２つの異なるレンダリングである。
    - そして、関連性を学習しようとする [seek to]。

- Although we lack supervision in the form of paired examples, we can exploit supervision at the level of sets: we are given one set of images in domain X and a different set in domain Y.
    - 我々はペア付けされた例の形式での教師ありを欠いているけれども、
    - セットのレベルで教師ありを利用する [exploit] ことが出来る。
    - 即ち、ソース X では画像の１つのセットが与えられ、ターゲット Y では画像の異なるセットが与えられている。

- We may train a mapping G : X → Y such that the output yˆ = G(x), x ∈ X, is indistinguishable from images y ∈ Y by an adversary trained to classify yˆ apart from y.
    - 我々は、y と区別して y^ を分類するように学習された敵対者によって、
    - 出力 yˆ = G(x), x ∈ X が、画像 y ∈ Y から区別できないように、写像 G : X → Y を学習するだろう。

- In theory, this objective can induce an output distribution over yˆ that matches the empirical distribution p_{data}(y) (in general, this requires G to be stochastic) [16].
    - 理論的には、この目的 [objective] は経験分布 [empirical distribution] p_ {data}（y）と一致する y^ 上の出力分布を誘導する [induce] ことができます（一般に、これは写像 G が確率的であることを必要とします）[16]。

> GAN で出てくる真のデータ分布（＝経験分布） p_ {data}（y）とモデルの分布（＝y^ 上の出力分布） p_g を一致させる話

- The optimal G thereby translates the domain X to a domain Yˆ distributed identically to Y.
    - G の最適化は、それ故に、ソース X から、Y と恒等的に分布するターゲット Y^ への変換となる。

- However, such a translation does not guarantee that an individual input x and output y are paired up in a meaningful way – there are infinitely many mappings G that will induce the same distribution over yˆ. 
    - しかしながら、このような変換は、個々の入力 x と出力 y が、意味のある方法でペアされているということを保証しない。
    - 即ち、y^ 上の同じ分布を誘発するような無限に多くの写像 G が存在する。

> 真のデータ分布（＝経験分布） p_ {data}（y）とモデルの分布（＝y^ 上の出力分布） p_g の一致のさせ方には、無限通りの方法があり、これは個々の入力 x と出力 y が、意味のある方法でペア化されるような制約をもっていないことを意味している。

- Moreover, in practice, we have found it difficult to optimize the adversarial objective in isolation: standard procedures often lead to the well-known problem of mode collapse, where all input images map to the same output image and the optimization fails to make progress [15].
    - 更には実際には、我々は、敵対的目標？ [adversarial objective] を孤立して（＝単独で） [in isolation] 最適化することが困難であることを見い出した。
    - 即ち、標準的な手続き [procedures] は、度々よく知られたモード崩壊に導く。
    - （このモード崩壊というのは、）全ての入力画像が同じ出力画像に写像し、最適化処理が進行しなくなる。

---

- These issues call for adding more structure to our objective.
    - これらの問題は、我々の目的に、より多くの構造を追加することを要求する [call for]。

- Therefore, we exploit the property that translation should be “cycle consistent”, in the sense that if we translate, e.g., a sentence from English to French, and then translate it back from French to English, we should arrive back at the original sentence [3].
    - それ故に、
    - もしも、我々が翻訳する [if we translate]。例えば、英語からフランス語の文章、そして次に、フランス語を英語の文章に戻すといったように、我々は元の文章に戻すべきであるということ [that] という意味において [in th sence]、
    - 我々は、翻訳が、”周期的に首尾一貫した” [“cycle consistent”] ものであるべきという性質を利用する。

- Mathematically, if we have a translator G : X → Y and another translator F : Y → X, then G and F should be inverses of each other, and both mappings should be bijections.
    - 数学的には、もし我々が変換器 G : X → Y と他の変換器 F : Y → X を持っていれば、
    - 次に、G と F は、お互いに逆になるべきである。
    - そして、両方の写像は、全単射 [bijections] になるべきである。

- We apply this structural assumption by training both the mapping G and F simultaneously, and adding a cycle consistency loss [64] that encourages F (G(x)) ≈ x and G(F (y)) ≈ y.
    - 我々は、写像 G と F の両方を学習することにより、この構造的な仮定を適用し、
    - F (G(x)) ≈ x and G(F (y)) ≈ y を促進する cycle consistency loss を追加する。

- Combining this loss with adversarial losses on domains X and Y yields our full objective for unpaired image-to-image translation.
    - この損失関数と、ソース X とターゲット Y の敵対的損失関数で組み合わせることは、ペア付けされていない image-to-image 変換に対しての我々の目的を生み出す。

---

- We apply our method to a wide range of applications, including collection style transfer, object transfiguration, season transfer and photo enhancement.
    - 我々は、style transfer, object transfiguration, season transfer, photo enhancement を含む、広い範囲での応用に我々の手法を適用する。

- We also compare against previous approaches that rely either on hand-defined factorizations of style and content, or on shared embedding functions, and show that our method outperforms these baselines.
    - 我々はまた、<font color="Pink">手動で定義したスタイル（画風）と内容の因数分解 [factorizations] と、共有された埋め込み関数の両方に頼る以前のアプローチ</font>と比較する。
    - そして、我々の手法をそれらのベースラインより優れている [outperforms] ことを示す。

- We provide both PyTorch and Torch implementations.

- Check out more results at our website.


# ■ 結論

## x. 論文の項目名 (Conclusion)


# ■ 何をしたか？詳細

## 3. Formulation

- Our goal is to learn mapping functions between two domains X and Y given training samples ![image](https://user-images.githubusercontent.com/25688193/59166788-2f6c5800-8b68-11e9-8471-769ad1b1ed95.png) where x_i ∈ X and {y_j}_{j=1}^M where y_j ∈ Y <1>.
    - 我々のゴールは、学習サンプル ![image](https://user-images.githubusercontent.com/25688193/59166788-2f6c5800-8b68-11e9-8471-769ad1b1ed95.png) （ここで、x_i ∈ X）と ![image](https://user-images.githubusercontent.com/25688193/59166813-7a866b00-8b68-11e9-9c71-3a10223d5892.png) （ここで、y_i ∈ Y）が与えられた２つのソース X とターゲット Y との間の写像関数を学習することである。

- We denote the data distribution as x ∼ p_data(x) and y ∼ pdata(y).
    - 我々は、x ~ p_data(x), y ~ p_data(y) として画像データ分布を示す。

- As illustrated in Figure 3 (a), our model includes two mappings G : X → Y and F : Y → X.
    - 図３の (a) に図示したように、我々のモデルは２つの写像 G : X → Y, F : Y → X を含む。

- In addition, we introduce two adversarial discriminators D_X and D_Y , where D_X aims to distinguish between images {x} and translated images {F (y)}; in the same way, D_Y aims to discriminate between {y} and {G(x)}.
    - 加えて、我々は２つの敵対的な識別器 D_X と D_Y を導入する。
    - ここで、D_X は画像 {x} と変換された画像 {F(y)} との間を区別することを目的としている。
    - 同様にして、D_Y は {y} と {F(x)} との間を区別することを目的としている。

- Our objective contains two types of terms: adversarial losses [16] for matching the distribution of generated images to the data distribution in the target domain; and cycle consistency losses to prevent the learned mappings G and F from contradicting each other.
    - **我々の目的は、２つの項の種類を含む。**
    - **即ち、目標のソースにおいて、生成された画像の分布とデータ分布を一致させるための、adversarial losses**
    - **学習された写像 G と F が互いに矛盾する [contradicting] のを防ぐための cycle consistency losses**

---

![image](https://user-images.githubusercontent.com/25688193/59166710-73129200-8b67-11e9-8d17-444aca9dc833.png)

- > Figure 3: (a) Our model contains two mapping functions G : X → Y and F : Y → X, and associated adversarial discriminators DY and DX.

- > DY encourages G to translate X into outputs indistinguishable from domain Y, and vice versa for DX and F .
    - > D_Y は G に、ソース X をターゲット Y と区別のつかない出力に変換することを促進する。

- > To further regularize the mappings, we introduce two cycle consistency losses that capture the intuition that if we translate from one domain to the other and back again we should arrive at where we started:
    - > 更に写像を正則化するため、我々は、２つの cycle consistency losses を導入する。
    - > これは、１つのソースから別のターゲットへ変換した場合に、開始した場所に再度たどり着くべきという、直感 [intuition] を捕まえるような（損失関数である。）

- > (b) forward cycle-consistency loss: x → G(x) → F (G(x)) ≈ x, 

- > and (c) backward cycle-consistency loss: y → F (y) → G(F (y)) ≈ y

---

![image](https://user-images.githubusercontent.com/25688193/59167375-e3231700-8b6b-11e9-82f8-97c9839f955f.png)

![image](https://user-images.githubusercontent.com/25688193/59167607-fbdffc80-8b6c-11e9-8903-b9e40aee502c.png)

### 3.1. Adversarial Loss

![image](https://user-images.githubusercontent.com/25688193/59168639-8e829a80-8b71-11e9-9f89-b664afffec86.png)

---

- We apply adversarial losses [16] to both mapping functions.

- For the mapping function G : X → Y and its discriminator DY , we express the objective as:

![image](https://user-images.githubusercontent.com/25688193/59168385-7fe7b380-8b70-11e9-9e95-01d418458a32.png)

- where G tries to generate images G(x) that look similar to images from domain Y , while DY aims to distinguish between translated samples G(x) and real samples y.
    - ここで、G はターゲット Y からの画像のによく似た画像 G(x) を生成しようとする。
    - 一方で、D_Y は変換されたサンプル G(x) と本物サンプル y との間を区別できなくすることを狙いとしている。

- G aims to minimize this objective against an adversary D that tries to maximize it, i.e., ![image](https://user-images.githubusercontent.com/25688193/59168469-d228d480-8b70-11e9-9b57-ce93a1113555.png).
    - G は、それを最大化しようとする敵対的な D に対して、この目的を最小化することを狙いとしている。
    - 即ち、![image](https://user-images.githubusercontent.com/25688193/59168469-d228d480-8b70-11e9-9b57-ce93a1113555.png)

- We introduce a similar adversarial loss for the mapping function F : Y → X and its discriminator DX as well: i.e.,![image](https://user-images.githubusercontent.com/25688193/59168419-aa397100-8b70-11e9-8bd8-a2094c427df6.png).


### 3.2. Cycle Consistency Loss

![image](https://user-images.githubusercontent.com/25688193/59167607-fbdffc80-8b6c-11e9-8903-b9e40aee502c.png)

---

- Adversarial training can, in theory, learn mappings G and F that produce outputs identically distributed as target domains Y and X respectively (strictly speaking, this requires G and F to be stochastic functions) [15]. 
    - 敵対的な学習は、理論的には、ターゲット Y とソース X のそれぞれとして、同一に分布した出力を生成するような、写像 G と F を学習することが出来る。
    - 厳密に言えば、これは写像 G,F が、確率的な関数であることを要求する。

- However, with large enough capacity, a network can map the same set of input images to any random permutation of images in the target domain, where any of the learned mappings can induce an output distribution that matches the target distribution.
    - しかしながら、十分に莫大な容量のもとでは、ネットワークは同じ入力画像のセットを、ターゲット領域内で任意のランダムな画像の配列 [permutation] に写像することが出来る。
    - ここで、学習された写像のいくつかは、ターゲット分布に一致する出力分布を誘発することが出来る。

- Thus, adversarial losses alone cannot guarantee that the learned function can map an individual input xi to a desired output yi.
    - それ故に、adversarial losses だけでは、学習された関数が、各々の入力 x_i を望ましい出力 y_i に写像することが出来るということを保証できない。

> 写像の自由度が高すぎて（＝制約が弱すぎて）、望ましい出力に写像するとは限らない。

- To further reduce the space of possible mapping functions, we argue that the learned mapping unctions should be cycle-consistent:
    - 写像関数の可能性の空間を更に減らすために、我々は、学習された写像が、cycle-consistent（周期的な首尾一貫性）になるべきということを主張する。

- as shown in Figure 3 (b), for each image x from domain X, the image translation cycle should be able to bring x back to the original image, i.e., x → G(x) → F (G(x)) ≈ x.
    - 即ち、図３の (b) に示されているように、
    - 領域 X からの各画像 x に対して、画像変換サイクルは、x を元の画像に戻すことが出来るべきである。
    - 例えば、x → G(x) → F (G(x)) ≈ x.

- We call this forward cycle consistency.
    - 我々は、これを forward cycle consistency（順伝搬の周期的な首尾一貫性）と呼ぶ。

- Similarly, as illustrated in Figure 3 (c), for each image y from domain Y , G and F should also satisfy backward cycle consistency: y → F (y) → G(F (y)) ≈ y. 
    - 同様にして、図３の (c) に図示されているように、ターゲット Y からの画像 y に対して、G と F　は、逆方向の cycle consistency を満たすべきである。
    - 即ち、y → F (y) → G(F (y)) ≈ y
    
- We incentivize this behavior using a cycle consistency loss:
    - 我々は、cycle consistency loss を用いてこの振る舞いを動機づける。

![image](https://user-images.githubusercontent.com/25688193/59168838-7bbc9580-8b72-11e9-853d-862eae452e1c.png)

- In preliminary experiments, we also tried replacing the L1 norm in this loss with an adversarial loss between F (G(x)) and x, and between G(F(y)) and y, but did not observe improved performance.
    - 予備の [preliminary] 実験では、我々はまた、この損失関数の L1 ノルムを、F (G(x)) と x との間の adversarial loss と、G(F(y)) と y との間の adversarial loss で置き換えようとした。
    - しかし、パフォーマンスの改善は観測されなかった。

---

- The behavior induced by the cycle consistency loss can be observed in Figure 4: the reconstructed images F (G(x)) end up matching closely to the input images x.
    - cycle consistency loss によって誘発される振る舞いは、図４で観測される。
    - 即ち、再構築された画像 F (G(x)) が、入力画像 x と近くで一致するという結果になる。


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


