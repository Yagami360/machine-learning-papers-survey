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

- This problem can be more broadly described as image- to-image translation [22], converting an image from one representation of a given scene, x, to another, y, e.g., grayscale to color, image to semantic labels, edge-map to photograph.

- Years of research in computer vision, image processing, computational photography, and graphics have produced powerful translation systems in the supervised setting, where example image pairs {xi,yi}Ni=1 are avail- able (Figure 2, left), e.g., [11, 19, 22, 23, 28, 33, 45, 56, 58, 62]. 

- However, obtaining paired training data can be difficult and expensive. 

- For example, only a couple of datasets exist for tasks like semantic segmentation (e.g., [4]), and they are relatively small.

- Obtaining input-output pairs for graphics tasks like artistic stylization can be even more difficult since the desired output is highly complex, typically requiring artistic authoring.

- For many tasks, like object transfigu- ration (e.g., zebra↔horse, Figure 1 top-middle), the desired output is not even well-defined.

---

- xxx

# ■ 結論

## x. 論文の項目名 (Conclusion)


# ■ 何をしたか？詳細

## x. 論文の項目名


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


