# ■ 論文
- 論文タイトル："Image-to-Image Translation with Conditional Adversarial Networks"
- 論文リンク：https://arxiv.org/abs/1611.07004
- 論文投稿日付：2016/11/21(v1), 2018/11/26(v3)
- 著者：Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros
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
    - ぼやけた [blurry] 結果の傾向となるだろう。[29,46]

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
    - 幸運なことに、これは、最近提案された GAN によって、ちょうど行われていることである。[14,5,30,36,47]

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
    - cGAN は、条件付き生成モデルを学習する。[14]

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
    - GAN は、ランダムノイズ入力 z から、出力画像 y:: G:z→yへの写像を学習するような、生成モデルである。[14]

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


> 最も大きな違いは、GANではGeneratorに乱数zしか与えていなかったのが、pix2pix での cGAN では Generator に変換元となる画像も与えるようになっているという点です。そして、Discriminatorについても、GANでは1つの画像が与えられ、学習データに含まれている画像かGeneratorが生成した画像かという判断を下すだけだったのですが、Conditional GANでは2つの画像が与えられ、学習データに含まれている (変換元画像, 変換先画像) というペアなのか、 (変換元画像, 変換元画像からGeneratorが生成した画像) のペアなのかという判断を下すようになっています。

## 2.1. Objective

- The objective of a conditional GAN can be expressed as
    - cGAN の目的関数は、以下のように表現される。

![image](https://user-images.githubusercontent.com/25688193/56846579-e9d33300-690b-11e9-9cf3-111e882697b0.png)<br>

> (z → noise vector, x → input image, y → output image)<br>
> cf : cGAN の損失関数<br>
> ![image](https://user-images.githubusercontent.com/25688193/56847771-94525280-691a-11e9-8515-0b10522aa2f4.png)<br>

- where G tries to minimize this objective against an adversarial D that tries to maximize it, i.e. ![image](https://user-images.githubusercontent.com/25688193/56847145-636e1f80-6912-11e9-8814-f3bdf4df8aa8.png).
    - ここで、G は、D に対抗して、この目的関数を最小化しようとする。
    - D は、この目的関数を最大化しようとする。
    - 例えば、![image](https://user-images.githubusercontent.com/25688193/56847145-636e1f80-6912-11e9-8814-f3bdf4df8aa8.png).

- To test the importance of conditioning the discrimintor, we also compare to an unconditional variant in which the discriminator does not observe x:
    - 識別器を条件付けの重要性をテストするために、
    - 識別器が x を観測しないような、条件付けされていな変種を比較する。
    - 即ち、

![image](https://user-images.githubusercontent.com/25688193/56846822-cbbb0200-690e-11e9-9561-86ad53cc0a1b.png)<br>

- Previous approaches to conditional GANs have found it beneficial to mix the GAN objective with a more traditional loss, such as L2 distance [29].
    - cGAN への以前のアプローチでは、
    - L2 距離のような、より伝統的な損失関数での GAN の目的関数を組み合わせることが、有益である [beneficial] ということがわかっている。[29]

- The discriminator’s job remains unchanged, but the generator is tasked to not only fool the discriminator but also to be near the ground truth output in an L2 sense.
    - 識別器の仕事は、変わらないままである。
    - しかし、生成器は、識別器を騙すだけでなく、L2 の意味 [sense] での、ground truth な出力の近くになるように、仕向けられている。

- We also explore this option, using L1 distance rather than L2 as L1 encourages less blurring:
    - 我々はまた、L1 のほうが、ぼやけが少ないので [as]、L2 よりも L1 を使用し、
    - このオプションを探索する。
    - 即ち、

![image](https://user-images.githubusercontent.com/25688193/56847095-cb703600-6911-11e9-8944-eabea1712c70.png)

- Our final objective is
    - 最終的な目的関数は、

![image](https://user-images.githubusercontent.com/25688193/56847105-f064a900-6911-11e9-91dd-9e8e508cd48b.png)

> L1正則化項は、変換先画像とGeneratorが生成した画像がピクセル単位でどれくらい違っているかということを表しており、これを最小化するということは、ピクセル単位で正解画像に近いような画像を生成するようになるということです。これにより、Discriminatorによって大域的な正しさを判定しつつ、L1正則化でピクセル単位での正しさも勘案する、ということが可能になります。

- Without z, the net could still learn a mapping from x to y, but would produce deterministic outputs, and therefore fail to match any distribution other than a delta function.
    - z なしに、ネットワークは依然として、x から y への写像を学習することはできた。
    - しかし、決定論的な出力を生成するだろう。
    - そして、それ故に、デルタ関数以外の他の（関数）の分布に一致しない。

> 確率分布の形状が、デルタ関数であることは、確率１の決定論的な確率分布になっていることを意味している。

- Past conditional GANs have acknowledged this and provided Gaussian noise z as an input to the generator, in addition to x (e.g., [39]).
    - 過去の CGAN は、これを認めて [acknowledged]、
    - x に加えて、生成器への入力としての、ガウスノイズ z を提供した。（例えば、[39]）

- In initial experiments, we did not find this strategy effective – the generator simply learned to ignore the noise – which is consistent with Mathieu et al. [27].
    - 初期の実験では、我々は、この戦略が効果的であることを見つけ出せなかった。
    - 即ち、生成器は、単純にノイズを無視するように学習した。
    - このことは、Mathieu [27] 等と矛盾なく一致する [consistent]。

- Instead, for our final models, we provide noise only in the form of dropout, applied on several layers of our generator at both training and test time.
    - 代わりに、我々の最終的なモデルために、
    - dropout の形のみで、ノイズを提供する。
    - （このdropuoutでのノイズというのは、）学習フェイズとテストフェイズの両方で、生成器のいくつかの層に適用されるような（dropoutでのノイズ）

- Despite the dropout noise, we observe very minor stochasticity in the output of our nets.
    - dropout ノイズにも関わらず、ネットワークの出力において、とても僅かな確率 [stochasticity] を観測する。

- Designing conditional GANs that produce stochastic output, and thereby capture the full entropy of the conditional distributions they model, is an important question left open by the present work.
    - 確率出力を生成する cGAN を設計すること、
    - そして、それによって [thereby ]、モデルの条件付き確率の全エントロピーを捕まえることは、
    - 現在の研究での、未解決の重要な問題である。

> 生成器 G に入力する入力ノイズ z は、従来の GAN のように、確率分布 U(0,1)  or N(0,1)  から直接サンプリングして実現するのではなく、生成器のネットワークの複数の層に、直接 dropout を施すという意味でのノイズとして実現する。

## 2.2. Network architectures

- We adapt our generator and discriminator architectures from those in [30].
    - [30] のものから、生成器と識別器のアーキテクチャを適用した。

- Both generator and discriminator use modules of the form convolution-BatchNorm-ReLu [18].
    - 生成器と識別器の両方は、convolution-BatchNorm-ReLu の形のモジュール [18] を使用する。

- Details of the architecture are provided in the appendix, with key features discussed below.
    - アーキテクチャの詳細は、以下ので議論される鍵となる特徴とともに、補足に提供している。

![image](https://user-images.githubusercontent.com/25688193/56847826-ce702400-691b-11e9-9e14-b51aef1be691.png)<br>

### 2.2.1 Generator with skips

- A defining feature of image-to-image translation problems is that they map a high resolution input grid to a high resolution output grid.
    - image-to-image 変換問題の決定的な [defining] 特徴は、高解像度の入力グリッドを、高解像度の出力グリッドへ写像するということである。

- In addition, for the problems we consider, the input and output differ in surface appearance, but both are renderings of the same underlying structure.
    - 加えて、我々が考えている問題に対しては、入力と出力は、表面の見え方が異なる。
    - しかし、両方とも、同じ基本的な [underlying] 構造をレンダリングしている。

- Therefore, structure in the input is roughly aligned with structure in the output.
    - これ故、入力の構造は、出力の構造に、おおまかにそろっている。[aligned]

- We design the generator architecture around these considerations.
    - 我々は、これらの考慮して、生成器のアーキテクチャを設計する。

<br>

- Many previous solutions [29, 39, 19, 48, 43] to problems in this area have used an encoder-decoder network [16].
    - この領域において、多くの以前の解決法 [29, 39, 19, 48, 43] が、encoder-decoder network を使用していた。

- In such a network, the input is passed through a series of layers that progressively downsample, until a bottleneck layer, at which point the process is reversed (Figure 3).
    - このようなネットワークにおいては、
    - 入力は、一連の層を通過する。
    - ボトルネックとなっている層まで、徐々に [progressively] ダウンサンプリング
    - （ボトルネックとなっている）その点では、処理は逆方向になる。
    - （図３）

- Such a network requires that all information flow pass through all the layers, including the bottleneck.
    - **このようなネットワークでは、ボトルネックを含めて、全ての情報フローが、全ての層を通過することを要求する。**

- For many image translation problems, there is a great deal of low-level information shared between the input and output, and it would be desirable to shuttle this information directly across the net.
    - **多くの画像変換問題に対して、入力と出力との間で共有されるような、多量の [a great deal of] 低レベルでの情報がある。**
    - **そして、ネットワークを渡って、この情報を直接的にシャッフルすることが望ましい。**

- For example, in the case of image colorizaton, the input and output share the location of prominent edges.
    - 例えば、画像の色付けの場合では、入力と出力は、突き出した [prominent] 辺の位置を共有する。

<br>

- To give the generator a means to circumvent the bottleneck for information like this, we add skip connections, following the general shape of a “U-Net” [34] (Figure 3).
    - **このような情報に対してのボトルネックを回避する [circumvent] ための手段 [mean] を、生成器に与えるために、**
    - **"U-Net" の一般的な構造に従って、skip connections を加える。**

- Specifically, we add skip connections between each layer i and layer n-i, where n is the total number of layers.
    - 特に、層 i と層 n-i の各々の間に、skip connections を加える。
    - ここで、n は、全ての層の数である。

- Each skip connection simply concatenates all channels at layer i with those at layer n - i.
    - 各 skip connection は、層 n-i のものとともに、層 i で、全てのチャンネルで、単純に結合する。

<!--
> pix2pix で UNet の構造が使われている理由は、UNet に入る全ての情報が、全ての層にフローされるために、image-to-image のタスクにおいてこのことが、入力と出力との間の
-->

![image](https://user-images.githubusercontent.com/25688193/56848050-fd3bc980-691e-11e9-9c90-95fc0e206af2.png)<br>

- > Figure 3: Two choices for the architecture of the generator.
    - > 図３：生成器のアーキテクチャのための、２つの選択

- > The “U-Net” [34] is an encoder-decoder with skip connections between mirrored layers in the encoder and decoder stacks.
    - "U-Net" は、encoder の中の mirrored layers と、decoder の stacks の間の skip connections をもつ encoder-decoder である。[34]


> ![image](https://user-images.githubusercontent.com/25688193/56857732-15a5f580-69ac-11e9-8ff9-2ccd55712be4.png)

> UNetでは、CNNでの、Encoder-decoderのように全ての情報をボトルネックまでダウンサンプリングさせるのではなく、共通の特徴量はレイヤー間をスキップさせてボトルネックを回避させる

> pix2pixでは入力と出力の表面上の外見は異なりますが、基本構造は一緒のため、このような方法で共通の特徴量におけるデータ欠損を回避させます。


### 2.2.2 Markovian discriminator (PatchGAN)

> PatchGANは、Discriminatorに画像を与える際に、画像すべてではなく、16x16や70x70といった小領域(=Patch)を切り出してから与えるようにするという仕組みです。これにより、ある程度大域的な判定を残しながらも学習パラメータ数を削減することができ、より効率的に学習することができるそうです。

<!--
> PatachGAN。これはcGANモデルにL1モデルを組み込むことで、大雑把な画像をL1で捉え、cGANがその詳細を捉えるという方法です。L1による画像生成だけでは細部がぼやけ、cGANのみの画像生成だけではDiscriminatorを騙すための違和感が生じてしまうので、これらを組み合わせることで互いの得意な作業を使い分け、精度を向上させます[4]。
-->

- It is well known that the L2 loss – and L1, see Figure 4 – produces blurry results on image generation problems [22].
    - L2損失関数と L1損失関数は、
    - 画像生成問題のぼやけた結果を生成するということが、よく知られている。（図４を参照）

![image](https://user-images.githubusercontent.com/25688193/56848494-cbc5fc80-6924-11e9-9b94-45e4a39005c5.png)<br>

- > Figure 4: Different losses induce different quality of results. 
    - > 図４：異なる損失関数は、異なる結果のクオリティを示している。

- > Each column shows results trained under a different loss. 
    - >各列は、異なる損失関数のもとで、学習された結果である。

- > Please see https://phillipi.github.io/pix2pix/ for additional examples. 

<br>

- Although these losses fail to encourage highfrequency crispness, in many cases they nonetheless accurately capture the low frequencies.
    - これらの損失関数は、高周波成分の鮮明さ [crispness] を推奨 [encourage] しないけれども、
    - 多くのケースにおいて、それにも関わらず [nonetheless]、低周波成分を正確に捉える。

> 画像の高周波成分、低周波成分とは？

> 画像は空間周波数という観点からとらえることができ，低い周波数成分は画像のおおまかな形状を，高い周波数成分は緻密な部分の情報を担っていることがわかります。

- For problems where this is the case, we do not need an entirely new framework to enforce correctness at the low frequencies.
    - これが事実である問題に対して、低周波成分での正しさを強制するために、我々は、新しいフレームワークの全体を必要としない。

- L1 will already do.
    - L1 損失関数は、すでにそうである。

<br>

- This motivates restricting the GAN discriminator to only model high-frequency structure, relying on an L1 term to force low-frequency correctness (Eqn. 4).
    - この動機は、高周波成分の構造のみをもつモデルに、GAN の識別器を、制限する [restricting]。
    - 低周波成分での正確性を強制するような L1項に頼って、

- In order to model high-frequencies, it is sufficient to restrict our attention to the structure in local image patches.
    - **高周波成分のモデルのためには、局所的な画像パッチにおいての構造に、我々の注意を向ければ十分である。**

- Therefore, we design a discriminator architecture – which we term a PatchGAN – that only penalizes structure at the scale of patches. 
    - **これ故、我々は、識別器のアーキテクチャを設計する。**
    - **我々はこれを、PatchGAN と呼ぶ。**
    - **（この PatchGAN というのは、）パッチのスケールで、構造にペナルティーを課すのみであるような（ものである。）**

- This discriminator tries to classify if each N × N patch in an image is real or fake.
    - この識別器は、１つの画像の各 N×N のパッチが、本物か偽物かを分類しようとする。

- We run this discriminator convolutationally across the image, averaging all responses to provide the ultimate output of D.
    - 我々は、この識別器を、画像に渡っての、畳み込みで動作させ、
    - 全ての応答の平均化し、D の最終的な [ultimate] 出力を提供する。

<br>

- In Section 3.4, we demonstrate that N can be much smaller than the full size of the image and still produce high quality results.
    - **セクション 3.4 では、N は、画像のフルサイズよりも、遥かに小さいことと、依然としてハイクオリティな結果を生成することを実証する。**

- This is advantageous because a smaller PatchGAN has fewer parameters, runs faster, and can be applied on arbitrarily large images.
    - **これは、利点である。**
    - **なぜならば、より小さい PatchGAN は、より少ないパラメーターで、よりはやく動作し、そして、任意の [arbitrarily] 大きな画像に適用できるからである。**

<br>

- Such a discriminator effectively models the image as a Markov random field, assuming independence between pixels separated by more than a patch diameter.
    - <font color="Pink">このような識別器は、マルコフ確率場 [Markov random field] として、画像を効果的にモデル化する。
    - パッチの直径 [diameter] よりも離れているピクセルの間の独立性を推定しながら、</font>

- This connection was previously explored in [25], and is also the common assumption in models of texture [8, 12] and style [7, 15, 13, 24].
    - この接続は、以前に [25] で探求された。
    - そして、テスクチャとスタイルのモデルにおいても、共通の仮定である。

- Our PatchGAN can therefore be understood as a form of texture/style loss.
    - 我々の PatchGAN は、それ故、texture/style の損失関数の形として、理解される。


### 2.3. Optimization and inference    

- To optimize our networks, we follow the standard approach from [14]: we alternate between one gradient descent step on D, then one step on G.
    - 我々のネットワークを最適化するために、[14] からの標準的なアプローチに従う。
    - 即ち、D の１つの勾配ステップと、G の１つの勾配ステップの間を交互に入れ替わる [alternate]。

- We use minibatch SGD and apply the Adam solver [20].
    - 我々は、SGD のミニバッチを使用し、Adam soler を適用する。

<br>

- At inference time, we run the generator net in exactly the same manner as during the training phase.
    - 推論フェイズでは、学習フェイズの間とまったく同じ方法 [manner] で、生成器を動作する。

- This differs from the usual protocol in that we apply dropout at test time, and we apply batch normalization [18] using the statistics of the test batch, rather than aggregated statistics of the training batch.
    - これは、テストフェイズでドロップアウトを適用するという点において、一般的なプロトコルとは異なり、
    - 学習バッチの総計の [aggregated] 統計ではなく、テストバッチの統計を使用している batch normalization [18] を適用する

- This approach to batch normalization, when the batch size is set to 1, has been termed “instance normalization” and has been demonstrated to be effective at image generation tasks [38].
    - バッチサイズが１に設定されているとき、
    - この barch normalization に向けてのアプローチは、“instance normalization” と呼ばれてきた。
    - そして、画像生成タスクで、効果的であることが実証されている。[38]

- In our experiments, we use batch size 1 for certain experiments and 4 for others, noting little difference between these two conditions.
    - 我々の実験では、バッチサイズ１をいくらかの [certain] 実験のために使用し、バッチサイズ４を他の実験で使用する。
    - それら２つの条件の間の違いはほとんどない。


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## 3. Experiments

- To explore the generality of conditional GANs, we test the method on a variety of tasks and datasets, including both graphics tasks, like photo generation, and vision tasks, like semantic segmentation:
    - Semantic labels↔photo, trained on the Cityscapes dataset [4].
    - Architectural labels → photo, trained on the CMP Facades dataset [31].
    - Map → aerial photo, trained on data scraped from Google Maps.
    - BW → color photos, trained on [35].
    - Edges → photo, trained on data from [49] and [44]; binary edges generated using the HED edge detector [42] plus postprocessing.
    - Sketch → photo: tests edges!photo models on humandrawn sketches from [10].
    - Day → night, trained on [21].

- cGAN の一般性を探求するために、写真生成のようなグラフィックスタスクと、セマンティックセグメンテーションのようなコンピュータービジョンタスクの両方を、様々なタスクとデータセットでメソッドをテストする。
    - Cityscapes dataset [4] で学習された、セマンティックラベル ↔ 写真。
    - CMP Facades dataset [31] で学習された、アーキテクチャ上のラベル → 写真。
    - xxx

### 3.1. Evaluation metrics

### 3.2. Analysis of the objective function

### 3.3. Analysis of the generator architecture

### 3.4 From PixelGANs to PatchGans to ImageGANs

### 3.5. Perceptual validation


# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


