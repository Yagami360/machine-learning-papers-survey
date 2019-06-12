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
    - 即ち、再構築された画像 F (G(x)) が、入力画像 x と近くで一致するという結果になる

---

![image](https://user-images.githubusercontent.com/25688193/59171930-62bae100-8b80-11e9-8a6a-189062b254f0.png)

- > Figure 4: The input images x, output images G(x) and the reconstructed images F(G(x)) from various experiments.
    - > 図４：様々な実験に対しての、入力画像 x、出力画像 G(x)、と再構築された画像 F(G(x))
    
- > From top to bottom: photo↔Cezanne, horses↔zebras, winter→summer Yosemite, aerial photos↔Google maps.


### 3.3. Full Objective

- Our full objective is:

![image](https://user-images.githubusercontent.com/25688193/59172044-e4ab0a00-8b80-11e9-857e-193b80d623c3.png)

- where λ controls the relative importance of the two objectives.

- We aim to solve:

![image](https://user-images.githubusercontent.com/25688193/59172098-423f5680-8b81-11e9-901b-273fe40022ff.png)

- Notice that our model can be viewed as training two “autoencoders” [20]:
    - 我々のモデルは、２つのオートエンコーダーの学習としてみなすことが出来ることに注意。

- we learn one autoencoder F ◦ G : X → X jointly with another G ◦ F : Y → Y .
    - 即ち、我々は、１つのオートエンコーダー F ◦ G : X → X を、別のオートエンコーダー G ◦ F : Y → Y と共同で [jointly with] 学習する。

- However, these autoencoders each have special internal structures:
    - しかしながら、これらのオートエンコーダーは、お互いに特別な内部構造を持っている。

- they map an image to itself via an intermediate representation that is a translation of the image into another domain.
    - 即ち、それら（＝オートエンコーダー）は、画像の他の領域での変換である中間表現経由で、画像を自身に写像する。

- Such a setup can also be seen as a special case of “adversarial autoencoders” [34], which use an adversarial loss to train the bottleneck layer of an autoencoder to match an arbitrary target distribution. 
    - このようなセットアップはまた、“adversarial autoencoders” の特別なケースとして見られる。
    - <font color="Pink">（この “adversarial autoencoders” というのは、）任意の目標分布に一致するための、オートエンコーダーのボトルネック層を学習するために adversarial loss を使用するようなもの。</font>

- In our case, the target distribution for the X → X autoencoder is that of the domain Y .
    - 我々のケースにおいては、オートエンコーダー X → X に対しての目標分布は、ターゲット Y のそれとなる。

---

- In Section 5.1.4, we compare our method against ablations of the full objective, including the adversarial loss L_GAN alone and the cycle consistency loss L_cyc alone, and empirically show that both objectives play critical roles in arriving at high-quality results.
    - セクション 5.1.4 では、完全な目的関数の切除 [ablations] に対して、我々の手法を比較する。
    - （これは、）adversarial loss L_GAN のみを含み、cycle consistency loss のみを含むようなもの
    - そして、両方の目的関数が、高品質の結果となることにおいて、重要な役割を演じていることを実験的に [empirically] 示す。

> ablations study でパフォーマンス比較

- We also evaluate our method with only cycle loss in one direction and show that a single cycle is not sufficient to regularize the training for this under-constrained problem.
    - 我々はまた、１つの方向の cycle loss のみを用いて、我々の手法を評価し、
    - single loss が、この制約化での問題に対して、学習を正則化するのに十分でないことを示す。


## 4. Implementation

### Network Architecture

![image](https://user-images.githubusercontent.com/25688193/59174338-e9c18680-8b8b-11e9-9c62-3c1c43d703ea.png)

---

- We adopt the architecture for our generative networks from Johnson et al. [23] who have shown impressive results for neural style transfer and superresolution. 
    - 我々は、我々の生成器のネットワークに対して、スタイル変換と超高解像度化に対しての印象深い結果を示した Johnson らの [23] を適用した。

> [23] : . Johnson, A. Alahi, and L. Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. In ECCV, 2016. 2, 3, 5, 7, 18

- This network contains two stride-2 convolutions, several residual blocks [18], and two fractionally-strided convolutions with stride 1/2.
    - このネットワークは、ストライド幅２の２つの畳み込み層、いくつかの ResNet の residual blocks、そして、ストライド幅 1/2 の２つの逆畳み込み層 [fractionally-strided convolutions] を含んでいる。

> fractionally-strided convolutions : 逆畳み込み層のこと 

- We use 6 blocks for 128 × 128 images and 9 blocks for 256 × 256 and higher- resolution training images. 
    - 128 × 128 の画像に対して、6 つのブロックを使用し、
    - 256 × 256 の画像とそれ以上の解像度の学習用画像に対して、9 つのブロックを使用する。

- Similar to Johnson et al. [23], we use instance normalization [53].
    - Johnson らの論文と同じく、instance normalization を使用する。
    
- For the discriminator networks we use 70 × 70 PatchGANs [22, 30, 29], which aim to classify whether 70 × 70 overlapping image patches are real or fake. 
    - 識別器のネットワークに対しては、70 × 70 の PaschGAN を使用する。
    - これは、70 × 70 でオーバラップしている画像パッチが本物か偽物であるかを分類することを狙いとしている。

- Such a patch-level discriminator architecture has fewer parameters than a full-image discriminator and can work on arbitrarily-sized images in a fully convolutional fashion [22].
    - このようなパッチレベルの識別器のアーキテクチャは、フル画像の識別器と比べて、より少ないパラメーターを持ち、
    - 全畳込みの形式において、任意のサイズの画像で動作できる。

### Training details

- We apply two techniques from recent works to stabilize our model training procedure.
    - 我々が、我々のモデルの学習手順を安定化させるための最近の研究から、２つのテクニックを適用する。

- First, for LGAN (Equation 1), we replace the negative log likelihood objective by a least-squares loss [35]. 
    - はじめに、式 (1) の L_GAN に対して、least-squares loss [35] によって、負対数尤度目的関数を置き換える。

- This loss is more stable during training and generates higher quality results.
    - この損失関数は、学習がより安定し、より高い品質の結果を生成する。

- In particular, for a GAN loss ![image](https://user-images.githubusercontent.com/25688193/59173508-1f647080-8b88-11e9-9603-1860a55323c0.png), we train the G to ![image](https://user-images.githubusercontent.com/25688193/59173839-bc73d900-8b89-11e9-88ff-a70e2c56893b.png) and train the D to ![image](https://user-images.githubusercontent.com/25688193/59173922-12488100-8b8a-11e9-9368-17e647b9593b.png).
    - 特に、損失関数 L_GAN 対しては、G を ![image](https://user-images.githubusercontent.com/25688193/59173839-bc73d900-8b89-11e9-88ff-a70e2c56893b.png) になるように学習し、
    - D を ![image](https://user-images.githubusercontent.com/25688193/59173922-12488100-8b8a-11e9-9368-17e647b9593b.png) となるように学習する。

---

- Second, to reduce model oscillation [15], we follow Shrivastava et al.’s strategy [46] and update the discriminators using a history of generated images rather than the ones produced by the latest generators.
    - ２つ目に、モデルの発振を軽減するために、Shrivastava らの戦略 [46] に従い、最新の生成器によって生成されたものよりも、生成された画像の履歴を使用する。

- We keep an image buffer that stores the 50 previously created images.
    - 我々は、50 回以前で生成された画像をストアする画像バッファを保持する。

---

- For all the experiments, we set λ = 10 in Equation 3. 
    - すべての実験に対して、式 (3) で λ = 10 を使用する。

- We use the Adam solver [26] with a batch size of 1.
    - 我々は、バッチサイズ１を持つ Adam を使用する。

- All networks were trained from scratch with a learning rate of 0.0002.
    - すべてのネットワークは、学習率 0.0002 でスクラッチで学習する。

- We keep the same learning rate for the first 100 epochs and linearly decay the rate to zero over the next 100 epochs.
    - 我々は、はじめの 100 エポックで同じ学習率を使用し、次の 100 エポックでゼロに向かって線形に減衰する。

- Please see the appendix (Section 7) for more details about the datasets, architectures, and training procedures
    - データセット、アーキテクチャ、学習手順についてのより詳細は、補足（セクション７）を参照してください。


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## 5. Results

- We first compare our approach against recent methods for unpaired image-to-image translation on paired datasets where ground truth input-output pairs are available for evaluation.
    - 我々は最初に、ground truth の入出力ペアが評価のために利用可能であるようなペア付けされたデータセットにおいての、ペア付けされたいない image-to-image 変換に対しての最近の手法に対して、我々のアプローチを比較する。

- We then study the importance of both the adversarial loss and the cycle consistency loss and compare our full method against several variants.
    - 我々は次に、adversarial loss と cycle consisitency loss の両方の重要性を勉強し、
    - いくつかの変種に対して、我々のフル手法を比較する。

- Finally, we demonstrate the generality of our algorithm on a wide range of applications where paired data does not exist.
    - 最後に、ペア付けされたデータが存在しないような広い範囲の応用で、我々のアルゴリズムの一般性を実証する。

- For brevity, we refer to our method as CycleGAN.
    - 簡単のため、我々の手法を CycleGAN とみなす。

- The PyTorch and Torch code, models, and full results can be found at our website.

### 5.1. Evaluation

- Using the same evaluation datasets and metrics as “pix2pix” [22], we compare our method against several baselines both qualitatively and quantitatively.
    - "pix2pix" として、同じ評価データセットと評価指標を使用し、
    - 定性的にも定量的にも、いくつかのベースラインに対して、我々の手法を比較する。

- The tasks include semantic labels↔photo on the Cityscapes dataset [4], and map↔aerial photo on data scraped from Google Maps.
    - このタスクは、セマンティック labels↔photo や map↔aerial などを含む。

- We also perform ablation study on the full loss function.
    - 完全損失関数で、ablation study を実施する。

####  5.1.1 Evaluation Metrics

##### AMT perceptual studies

- On the map ↔ aerial photo task, we run “real vs fake” perceptual studies on Amazon Mechanical Turk (AMT) to assess the realism of our outputs.
    - 地図 ↔ 航空写真 タスクにおいては、我々の出力のリアル性を評価するために、Amazon Mechanical Turk (AMT) での、”本物” vs "偽物" の知覚的な研究 [perceptual studies] を実施した。

- We follow the same perceptual study protocol from Isola et al. [22], except we only gather data from 25 participants per algorithm we tested.
    - テストしたアルゴリズム毎に、25 人の参加からデータを収集することを除いて、
    - Isola らによる [22] から、同じ perceptual study のプロトコルに従い、

- Participants were shown a sequence of pairs of images, one a real photo or map and one fake (generated by our algorithm or a baseline), and asked to click on the image they thought was real.
    - 参加者は、画像のペアの系列を見せられ、１つは本物の写真、もう１つは偽物（我々のアルゴリズムやベースラインで生成されたもの）
    - そして、彼らが本物だと思う画像をクリックするように尋ねられる。

- The first 10 trials of each session were practice and feedback was given as to whether the participant’s response was correct or incorrect.
    - 各セッションでの最初の１０回の試行は、練習で、
    - 参加者の反応が正解か不正解かどうかとしてのフィードバックが与えられる。

- The remaining 40 trials were used to assess the rate at which each algorithm fooled participants.
    - 残りの 40 試行は、各アルゴリズムが被験者をだました割合を評価するために使用される。

- Each session only tested a single algorithm, and participants were only allowed to complete a single session.
    - 各セッションでは、１つのアルゴリズムのみテストされ、被験者は１つのセッションを完了することのみが許容される。

- The numbers we report here are not directly comparable to those in [22] as our ground truth images were processed slightly differently <2> and the participant pool we tested may be differently distributed from those tested in [22] (due to running the experiment at a different date and time).
    - 我々の ground truth 画像が、わずかに異って処理され、
    - 我々がテストした被疑者の pool は、[22] でテストされたそれとは異なる分布となる。（実験が異なる日時と時間で実際しているため）ので、
    - 我々がここで報告した数は、[22] のそれとは直接的には比較出来ない。

- Therefore, our numbers should only be used to compare our current method against the baselines (which were run under identical conditions), rather than against [22].
    - それ故に、我々の数は、[22] の手法との比較よりもむしろ、我々の手法と（同じ条件下で動作する）ベースラインとの比較のみに使用されるべきである。

##### FCN score

- Although perceptual studies may be the gold standard for assessing graphical realism, we also seek an automatic quantitative measure that does not require human experiments.
    - perceptual studies は、グラフィックなリアリズムの評価のためのゴールドスタンダード（＝代表的な手法）だけれども、
    - 我々はまた、人手での実験を必要としない自動的な定量的な指標を探す。

- For this, we adopt the “FCN score” from [22], and use it to evaluate the Cityscapes labels ↔ photo task.
    - この目的のために、[22] から “FCN score” を採用し、Cityscapes labels ↔ photo (都会の風景 ↔ 写真）のタスクを評価するために使用する。

- The FCN metric evaluates how interpretable the generated photos are according to an off-the-shelf semantic segmentation algorithm (the fully-convolutional network, FCN, from [33]).
    - FCN 指標は、生成した写真がどの程度解釈可能である [interpretable] のかを、セマンティックセグメンテーションアルゴリズムに従って評価する。

- The FCN predicts a label map for a generated photo.
    - FCN は、生成された写真に対してのラベルマップを予想する。

> セマンティックセグメンテーションされたラベルマップ

- This label map can then be compared against the input ground truth labels using standard semantic segmentation metrics described below.
    - このラベルマップは、以下で記述される標準的なセマンティックセグメンテーション指標を用いて、入力 ground truth ラベルに対して比較される。

- The intuition is that if we generate a photo from a label map of “car on the road”, then we have succeeded if the FCN applied to the generated photo detects “car on the road”.
    -　直感的には、もし ”路上の車” のラベルマップから生成された画像を生成した場合、生成された写真を適用した FCN は路上の車を検出することに成功する。

##### Semantic segmentation metrics

- To evaluate the performance of photo ↔ labels, we use the standard metrics from the Cityscapes benchmark [4], including per-pixel accuracy, per-class accuracy, and mean class Intersection-Over-Union (Class IOU) [4].
    - photo ↔ labels のパフォーマンスを評価するために、
    - Cityscapes benchmark [4] から標準的な指標を使用する。
    - （これは、）ピクセル単位での正解率、クラス単位での正解率、IOU の平均値を含んでいる。


#### 5.1.2 Baselines


#### 5.1.3 Comparison against baselines

![image](https://user-images.githubusercontent.com/25688193/59316933-c4ec2100-8cfb-11e9-837d-5fa54ba8e673.png)

- > Figure 5: Different methods for mapping labels↔photos trained on Cityscapes images.
    - > 図５：都市の風景画像で学習された、マッピングラベル ↔ 写真の変換に対しての異なる手法。

- > From left to right: input, Bi- GAN/ALI [7, 9], CoGAN [32], feature loss + GAN, SimGAN [46], CycleGAN (ours), pix2pix [22] trained on paired data, and ground truth.

![image](https://user-images.githubusercontent.com/25688193/59317044-4cd22b00-8cfc-11e9-82c7-2e9e039797e5.png)

- > Figure 6: Different methods for mapping aerial photos↔maps on Google Maps. From left to right: input, BiGAN/ALI [7, 9], CoGAN [32], feature loss + GAN, SimGAN [46], CycleGAN (ours), pix2pix [22] trained on paired data, and ground truth.
    - > 図６：グーグルマップでの、航空写真 ↔ 地図に対しての異なる手法

---

- As can be seen in Figure 5 and Figure 6, we were unable to achieve compelling results with any of the baselines.
    - 図５や図６で見られるように、いくつかのベースラインで、説得力のある [compelling] 結果を達成することはできない。

- Our method, on the other hand, can produce translations that are often of similar quality to the fully supervised pix2pix.
    - 我々の手法は、言い換えれば、完全な教師ありの pix2pix とたびたび同じ品質の変換を生成する。

---

![image](https://user-images.githubusercontent.com/25688193/59317145-a175a600-8cfc-11e9-84a8-70d4da5b35b4.png)

- > Table 1: AMT “real vs fake” test on maps↔aerial photos at 256 × 256 resolution.
    - > 表１：256 × 256 の解像度での、地図↔航空写真での、AMT "本物" vs "偽物"テスト

> パーセンテージ % は、アルゴリズムからの生成画像（＝偽物画像）が、被験者をだました割合。このパーセンテージの値が大きいほど、よい品質であることを示している。

---

- Table 1 reports performance regarding the AMT perceptual realism task.
    - 表１は、AMT perceptual realism task に関してのパフォーマンスを報告している。

- Here, we see that our method can fool participants on around a quarter of trials, in both the maps ↔ aerial photos direction and the aerial photos ↔ maps direction at 256 × 256 resolution <3>.
    - 我々の手法はだいたい施行の 1/4 で、被験者を騙すことが出来いる。
    - ここで、地図から航空写真の方向と、航空写真から地図の方向の両方で、256 × 256 の解像度である。

- All the baselines almost never fooled participants.
    - すべてのベースラインは、ほとんど被験者を騙せていない。

---

![image](https://user-images.githubusercontent.com/25688193/59317552-63798180-8cfe-11e9-8074-905af574ef2e.png)

- > Table 2: FCN-scores for different methods, evaluated on Cityscapes labels→photo.
    - > 表２：異なる手法に対しての 都会風景のラベル ↔ 都会風景写真のタスクで評価した FCN-score。

![image](https://user-images.githubusercontent.com/25688193/59318530-56f72800-8d02-11e9-9dc6-e1f9ad716423.png)

- > Table 3: Classification performance of photo→labels for different methods on cityscapes.

---

- Table 2 assesses the performance of the labels ↔ photo task on the Cityscapes and Table 3 evaluates the opposite　mapping (photos ↔ labels).
    - 表２は、都市風景における ラベル ↔ 写真変換タスクのパフォーマンスを評価し、
    - 表３は、反対の写像（ラベル↔写像）を評価する。

- In both cases, our method again outperforms the baselines.
    - 両方のケースにおいて、我々の手法をベースラインを上回っている。

#### 5.1.4 Analysis of the loss function

![image](https://user-images.githubusercontent.com/25688193/59319213-cbcb6180-8d04-11e9-8510-44c1ffe57e0f.png)

- > Table 4: Ablation study: FCN-scores for different variants of our method, evaluated on Cityscapes labels→photo.

![image](https://user-images.githubusercontent.com/25688193/59319387-6cba1c80-8d05-11e9-9e40-a0d679bc8457.png)

---

- In Table 4 and Table 5, we compare against ablations of our full loss. 
    - 表４と表５では、完全な損失関数での ablations study で比較している。

- Removing the GAN loss substantially degrades results, as does removing the cycle-consistency loss.
    - cycle-consistency loss を除去することと同様にして、GAN の損失関数を除去することは、実質的に [substantially] 結果を低下させる [degrades]。

- We therefore conclude that both terms are critical to our results.
    - 我々はそれ故に、両方の項が、我々の結果に重要であるという結論を下す [conclude]。

- We also evaluate our method with the cycle loss in only one direction: GAN + forward cycle loss ![image](https://user-images.githubusercontent.com/25688193/59319094-65deda00-8d04-11e9-946a-d252f1c843e3.png), or GAN + backward cycle loss ![image](https://user-images.githubusercontent.com/25688193/59319129-7f802180-8d04-11e9-88a2-3fc3fce85a8c.png) (Equation 2) and find that it often incurs training instability and causes mode collapse, especially for the direction of the mapping that was removed.
    - 我々はまた、１つの方向のみの cycle loss で我々の手法を評価する。
    - 即ち、GAN + forward cycle loss ![image](https://user-images.githubusercontent.com/25688193/59319094-65deda00-8d04-11e9-946a-d252f1c843e3.png), or GAN + backward cycle loss ![image](https://user-images.githubusercontent.com/25688193/59319129-7f802180-8d04-11e9-88a2-3fc3fce85a8c.png) (Equation 2)
    - そして、特に、削除された写像の方向に対して、それが学習の不安定性を起こし、モード崩壊の原因となることを見いだす。

- Figure 7 shows several qualitative examples.

---

![image](https://user-images.githubusercontent.com/25688193/59322636-0d163e00-8d12-11e9-9c1c-261cc01cd8aa.png)

- > Figure 7: Different variants of our method for mapping labels↔photos trained on cityscapes.
    - > 図７：都会風景画像で学習されたラベル↔写真間の写像に対しての、我々の変種の違い

- > From left to right: input, cycle- consistency loss alone, adversarial loss alone, GAN + forward cycle-consistency loss (F (G(x)) ≈ x), GAN + backward cycle-consistency loss (G(F(y)) ≈ y), CycleGAN (our full method), and ground truth.

- > Both Cycle alone and GAN + backward fail to produce images similar to the target domain.

- > GAN alone and GAN + forward suffer from mode collapse, producing identical label maps regardless of the input photo.
    - > GAN alone と GAN + forward は、モード崩壊を被り、
    - > 入力画にも関わらず、同じラベルマップを生成する。

# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


