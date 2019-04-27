# ■ 論文
- 論文タイトル："Image-to-Image Translation with Conditional Adversarial Networks"
- 論文リンク：
- 論文投稿日付：
- 著者：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- We investigate conditional adversarial networks as a general-purpose solution to image-to-image translation problems.
    - 我々は、画像から画像への変換問題の、一般的な解決法として、CGAN を調査する。

- These networks not only learn the mapping from input image to output image, but also learn a loss function to train this mapping.
    - これらのネットワークは、入力画像から出力画像への写像を学習するだけでなく、この写像を学習するために損失関数も学習する。

- This makes it possible to apply the same generic approach to problems that traditionally would require very different loss formulations.
    - <font color="Pink">これは、伝統的に非常に異なる損失関数の定式化を要求するような問題へ、同じ 全体的な [generic] アプローチに適用することを可能にする。</font>

- We demonstrate that this approach is effective at synthesizing photos from label maps, reconstructing objects from edge maps, and colorizing images, among other tasks.
    - 我々は、このアプローチが、
    - ラベルマップからの合成 [synthesizing] 写真、エッジマップからのオブジェクトの再構築、そして、画像の色付け、他のタスクの間、
    - において、効果的であることを実証する。

- As a community, we no longer hand-engineer our mapping functions, and this work suggests we can achieve reasonable results without hand-engineering our loss functions either.
    - コミュニティーとして、我々はもはや、写像関数を手作業で設定していない。
    - そして、この研究は、損失関数を手作業で設計しなくとも、合理的な結果が得られることを達成している。

<br>

- Many problems in image processing, computer graphics, and computer vision can be posed as “translating” an input image into a corresponding output image.
    - 画像処理、コンピューターグラフィックス、コンピュータービジョンにおける多くの問題は、入力画像を出力画像に一致させる ”変換” として、提示させる [pose] ことが出来る

- Just as a concept may be expressed in either English or French, a scene may be rendered as an RGB image, a gradient field, an edge map, a semantic label map, etc.
    - ちょうど、概念が、英語やフランス語で説明されるように、
    - シーンは、RGB画像、勾配フィールド、エッジマップか、セマンティックラベルマップとして描写されるかもしれない。

- In analogy to automatic language translation, we define automatic image-to-image translation as the problem of translating one possible representation of a scene into another, given sufficient training data (see Figure 1).
    - 自動言語翻訳の類似 [analogy] して、
    - 我々は、自動の image-to-image 変換を、あるシーンの１つの可能な表現を他のものへ変換の問題として定義する。
    - 十分な学習用データが与えられ場合に、
    - （図１を参照）

![image](https://user-images.githubusercontent.com/25688193/56843938-c051e180-68e2-11e9-91e2-9780a629a4f7.png)<br>

- > Figure 1: Many problems in image processing, graphics, and vision involve translating an input image into a corresponding output image.
    - > 図１：画像処理やコンピューターグラフィックス、コンピュータービジョンにおける多くの問題が、入力画像を出力画像に一致させる変換に取り込む [involve]

- > These problems are often treated with application-specific algorithms, even though the setting is always the same: map pixels to pixels.
    - > これらの問題は、しばしば、アプリケーションに特化したアルゴリズムとして扱われる。
    - > 設定が同じであるにも関わらず、[even though]
    - > （この同じ設定というのは、）即ち、ピクセルからピクセルへの写像

- > Conditional adversarial nets are a general-purpose solution that appears to work well on a wide variety of these problems.
    - > cGAN は、これらの広い種類の問題をうまく動作させるように思えるような、一般的な目的での解決法である。

- > Here we show results of the method on several.
    - ここに、いくつかの方法の結果を示す。

- > In each case we use the same architecture and objective, and simply train on different data.
    - > 各ケースにおいて、同じアーキテクチャと目的、及び、異なるデータでの単純な学習を使用している。

<br>

- One reason language translation is difficult is because the mapping between languages is rarely one-to-one – any given concept is easier to express in one language than another. 
    - 言語の変換が困難な理由の１つは、言語間の写像が、１対１に対応することがめったにないからである。
    - 即ち、いくつかの与えられた概念は、ある１つの言語で、他の（言語）より、説明することがより容易である。

- Similarly, most image-to-image translation problems are either many-to-one (computer vision) – mapping photographs to edges, segments, or semantic labels, or one-to-many (computer graphics) – mapping labels or sparse user inputs to realistic images. 
    - 同様にして [Similarly]、ほとんどの image-to-image 変換問題は、many-to-one である。（コンピュータービジョン）
    - 即ち、写真のエッジへのマッピング、セグメント、セマンティックラベル、- 或いは、one-to-many である（コンピューターグラフィックス）
    - 即ち、ラベルのマッピング、リアルな画像へのユーザ入力をスパーズする

- Traditionally, each of these tasks has been tackled with separate, special-purpose machinery (e.g., [7, 15, 11, 1, 3, 37, 21, 26, 9, 42, 46]), despite the fact that the setting is always the same: predict pixels from pixels.
    - 伝統的に、これらのタスクはのそれぞれ [each of] は、特定の目的の機構に分かれて [separate] 取り込まれてきた。[tackled]
    - 設定がいつも同じにも関わらず、
    - （この同じ設定というのは、）即ち、ピクセルからピクセルへの写像

- Our goal in this paper is to develop a common framework for all these problems.
    - この論文における我々のゴールは、これらの問題の全てに対して、共通のフレームワークを開発することである。

<br>

- The community has already taken significant steps in this direction, with convolutional neural nets (CNNs) becoming the common workhorse behind a wide variety of image prediction problems.
    - コミュニティーは、すでに、この方向性への重要なステップを取っていおり、
    - CNN は、多種多様な画像予想問題の背後にある共通の主役 [workhorse] になりつつある。

- CNNs learn to minimize a loss function – an objective that scores the quality of results – and although the learning process is automatic, a lot of manual effort still goes into designing effective losses.
    - CNN は、損失関数を最小化することを学習する。
    - （この損失関数というのは、）即ち、結果のクオリティをスコア化する目的関数
    - そして、学習処理は自動でるにも関わらず、たくさんの手動での労力が、効果的な損失関数を設定するのに、投入される。[go into]

- In other words, we still have to tell the CNN what we wish it to minimize.
    - 言い換えれば、我々は、CNN に何を最小化したいのかを教えなくてはならない。

- But, just like Midas, we must be careful what we wish for!
    - しかし、Midas と同じように、我々が何を望んでいるかを、注意しなければならない！

- If we take a naive approach, and ask the CNN to minimize Euclidean distance between predicted and ground truth pixels, it will tend to produce blurry results [29, 46].
    - もし、我々が考えが甘い [naive] アプローチをとり、そして、予想値と ground truth ピクセルとの間のユークリッド距離を最小化することを、CNN に依頼するならば、
    - ぼやけた [blurry] 結果の傾向となるだろう。

- This is because Euclidean distance is minimized by averaging all plausible outputs, which causes blurring.
    - これは、ユークリッド距離が、全ての尤度 [plausible] 出力の平均を最小化しているために起こり、この場合にぼやける。

- Coming up with loss functions that force the CNN to do what we really want – e.g., output sharp, realistic images – is an open problem and generally requires expert knowledge.
    - CNN に我々が本当にしてほしいことをさせるような損失関数を思いつくこと、（即ち、例えば、シャープでリアルな画像の出力）
    - は、未解決の問題であり、専門的な知識を必要とする。

<br>

- It would be highly desirable if we could instead specify only a high-level goal, like “make the output indistinguishable from reality”, and then automatically learn a loss function appropriate for satisfying this goal.
    - もし、「本物と区別できない出力を作る」といったような、高いレベルでのゴールだけを代わりに指定すれば、
    - このゴールを満足するのに適した [appropriate for] 損失関数を自動的に学習することが、強く望まれる。[desirable]

- Fortunately, this is exactly what is done by the recently proposed Generative Adversarial Networks (GANs) [14, 5, 30, 36, 47]. 
    - 幸運なことに、これは、最近提案された GAN によって、ちょうど行われていることである。

- GANs learn a loss that tries to classify if the output image is real or fake, while simultaneously training a generative model to minimize this loss.
    - GAN は、出力画像が本物であるのか偽物であるのかを、分類しようと損失値を学習する。
    - この loss 値を最小化するために、生成モデルを同時に学習する間に、

- Blurry images will not be tolerated since they look obviously fake.
    - ぼやけた画像は、明らかに偽物に見えるので、許容され [tolerated] ないだろう。

- Because GANs learn a loss that adapts to the data, they can be applied to a multitude of tasks that traditionally would require very different kinds of loss functions.
    - GAN は、データに適合するように loss 値を学習するので、
    - 伝統的に、かなり異なる種類の損失関数を要求するような、多数の [multitude] タスクに適用できる。

<br>

- In this paper, we explore GANs in the conditional setting.
    - この論文では、条件付き設定で、GAN を探索する。

- Just as GANs learn a generative model of data, conditional GANs (cGANs) learn a conditional generative model [14].
    - GAN が、データの生成モデルを学習ように、
    - cGAN は、条件付き生成モデルを学習する。

- This makes cGANs suitable for image-to-image translation tasks, where we condition on an input image and generate a corresponding output image.
    - これは、cGAN を、image-to-image 変換タスクに適合したものにする。
    - （このタスクというのは、）１つの入力画像を条件付けし、一致する出力画像を生成するような（タスク）

<br>

- GANs have been vigorously studied in the last two years and many of the techniques we explore in this paper have been previously proposed.
    - GAN は、最近の２年間で、精力的に [vigorously] 研究されてきた。
    - そして、我々がこの論文で探求するたくさんのテクニックは、以前に提案されてきた。

- Nonetheless, earlier papers have focused on specific applications, and it has remained unclear how effective image-conditional GANs can be as a general-purpose solution for image-toimage translation. 
    - にも関わらず、以前の論文は、特定のアプリケーションにフォーカスされていた。
    - そして、image-conditional GANs が、image-to-image 変換に対しての、一般的な目的での解決法に、どの程度効果的であるのかということは、不明なままである。

- Our primary contribution is to demonstrate that on a wide variety of problems, conditional GANs produce reasonable results.
    - 我々の主な貢献は、多種多様な問題で、cGAN は合理的な結果をもたらすということを実証することである。

- Our second contribution is to present a simple framework sufficient to achieve good results, and to analyze the effects of several important architectural choices.
    - 我々の２番目の貢献は、よい結果を達成するために、十分なシンプルなフレームワークを提供することである。
    - そして、いくつかの重要なアーキテクチャ上の選択の効果性を分析することである。

- Code is available at https://github.com/phillipi/pix2pix.


# ■ イントロダクション（何をしたいか？）


# ■ 結論

## 4. Conclusion

- The results in this paper suggest that conditional adversarial networks are a promising approach for many imageto-image translation tasks, especially those involving highly structured graphical outputs.
    - この論文での結論は、cGAN は、多くの image-to-image 変換タスクに対して、のアプローチを約束するこということを提案する。
    - （この変換タスクというのは、）とりわけ、これらは、高度に構造化されたグラフィカルな出力を含むような、（変換タスク）

- These networks learn a loss adapted to the task and data at hand, which makes them applicable in a wide variety of settings.
    - これらのネットワークは、手元の [at hand] タスクとデータに適用された損失関数を学習し、
    - それら（ネットワーク）を、多種多様な設定で、適用可能にする。

# ■ 何をしたか？詳細

## 2. Method

- GANs are generative models that learn a mapping from random noise vector z to output image y: G : z → y[14].
    - GAN は、ランダムノイズ入力 z から、出力画像 y:: G:z→yへの写像を学習するような、生成モデルである。

- In contrast, conditional GANs learn a mapping from observed image x and random noise vector z, to y: G : {x,z} → y.
    - 対称的に、cGAN は、観測された画像 x とランダムノイズ z から、y:: G : {x,z} → y への写像を学習する。

- The generator G is trained to produce outputs that cannot be distinguished from “real” images by an adversarially trained discrimintor, D, which is trained to do as well as possible at detecting the generator’s “fakes”.
    - 生成器 G は、敵対的に学習された識別器 D による本物の画像を見分けることのできないような出力を処理するように学習される。
    - （この識別器というのは、）同様にして、生成器の偽物画像を検出することに出来るように学習される。

- This training procedure is diagrammed in Figure 2.
    - この学習プロセスは、図２に、図表で示されている。[diagrammed]

![image](https://user-images.githubusercontent.com/25688193/56846488-d8d5f200-690a-11e9-8fdb-e9446e554b3e.png)<br>

- > Figure 2: Training a conditional GAN to predict aerial photos from maps.
    - > 図２：地図からの航空写真を予想するために、cGAN を学習する。

- > The discriminator, D, learns to classify between real and synthesized pairs.
    - > 識別器 D は、本物画像と合成された [synthesized pairs] ペア画像との間の分類を学習する。

- > The generator learns to fool the discriminator.
    - > 生成器は、識別器をだますように学習する。

- > Unlike an unconditional GAN, both the generator and discriminator observe an input image.
    - > 条件付けのない GAN とは異なって [Unlike]、生成器と識別器の両方は、１つの入力画像を観測する。

## 2.1. Objective

- The objective of a conditional GAN can be expressed as
    - cGAN の目的関数は、以下のように表現される。

![image](https://user-images.githubusercontent.com/25688193/56846579-e9d33300-690b-11e9-9cf3-111e882697b0.png)<br>

- where G tries to minimize this objective against an adversarial D that tries to maximize it, i.e. G^* =arg min_G max_D L_cGAN(G;D).
    - ここで、G は、D に対抗して、この目的関数を最小化しようとする。
    - D は、この目的関数を最大化しようとする。
    - 例えば、G^* =arg min_G max_D L_cGAN(G;D).

- To test the importance of conditioning the discrimintor, we also compare to an unconditional variant in which the discriminator does not observe x:



# ■ 実験結果（主張の証明）・議論（手法の良し悪し）

## x. 論文の項目名


# ■ メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


