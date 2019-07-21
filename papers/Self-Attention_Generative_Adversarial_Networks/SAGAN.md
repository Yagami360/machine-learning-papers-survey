# ■ 論文
- 論文タイトル："Self-Attention Generative Adversarial Networks"
- 論文リンク：https://arxiv.org/abs/1805.08318
- 論文投稿日付：2018/03/21
- 著者（組織）：Han Zhang, Ian Goodfellow, Dimitris Metaxas, Augustus Odena
- categories：

# ■ 概要（何をしたか？）

## Abstract

- In this paper, we propose the Self-Attention Generative Adversarial Network (SAGAN) which allows attention-driven, long-range dependency modeling for image generation tasks.
    - 本論文では、画像生成タスクにおいて、注意喚起型 [attention-driven] の長距離依存性モデリングを可能にする自己注意生成型敵対的ネットワーク（SAGAN）を提案する。

- Traditional convolutional GANs generate high-resolution details as a function of only spatially local points in lower-resolution feature maps. In SAGAN, details can be generated using cues from all feature locations.
    - 伝統的な畳み込みGANは、低解像度の特徴マップ内の空間的に局所的な点のみの関数として高解像度の詳細を生成します。 SAGANでは、すべての特徴位置からの手がかり [cues] を使用して詳細を生成できます。

- Moreover, the discriminator can check that highly detailed features in distant portions of the image are consistent with each other.
    - **さらに、識別器は、画像の離れた部分 [portions] において、非常に詳細な特徴が互いに一致していることを確認することができる。**

- Furthermore, recent work has shown that generator conditioning affects GAN performance.
    - さらに、最近の研究は、生成器の条件付がGANの性能に影響を与えることを示しています。

- Leveraging this insight, we apply spectral normalization to the GAN generator and find that this improves training dynamics.
    - この洞察を利用して、我々はスペクトル正規化をGAN生成器に適用し、そしてこれが学習ダイナミクスを改善することを発見する。

- The proposed SAGAN performs better than prior work, boosting the best published Inception score from 36.8 to 52.52 and reducing Fre ́chet Inception distance from 27.62 to 18.65 on the challenging ImageNet dataset.
    - 提案されているSAGANは、以前の研究よりも優れたパフォーマンスを発揮し、挑戦的なImageNetデータセットで、公表されている最良のInceptionスコアを36.8から52.52に向上させ、Frechet Inceptionの距離を27.62から18.65に短縮します。

- Visualization of the attention layers shows that the generator leverages neighborhoods that correspond to object shapes rather than local regions of fixed shape.
    - **注意層の視覚化は、生成器が固定形状の局所領域ではなく、オブジェクト形状に対応する近傍を活用する [leverages] ことを示しています。**


# ■ イントロダクション（何をしたいか？）

## 1. Introduction

- Image synthesis is an important problem in computer vision. There has been remarkable progress in this direction with the emergence of Generative Adversarial Networks (GANs) (Goodfellow et al., 2014), though many open problems remain (Odena, 2019). GANs based on deep convolutional networks (Radford et al., 2016; Karras et al., 2018; Zhang et al.) have been especially successful. 
    - 画像合成は、コンピュータビジョンにおける重要な問題です。 多くの未解決の問題が残っているが（Odena、2019）、生成的敵対的ネットワーク（GAN）の出現により、この方向への著しい進歩があった（Goodfellow et al。、2014）。 ディープコンボリューションネットワークに基づくGAN（Radfordら、2016年; Karrasら、2018年; Zhangら）は特に成功している。

- However, by carefully examining the generated samples from these　models, we can observe that convolutional GANs (Odena et al., 2017; Miyato et al., 2018; Miyato & Koyama, 2018) have much more difficulty in modeling some image classes than others when trained on multi-class datasets (e.g., ImageNet (Russakovsky et al., 2015)). 
    - しかしながら、これらのモデルから生成されたサンプルを注意深く調べることによって、畳み込みGAN（Odena et al。、2017; Miyato et al。、2018; Miyato＆Koyama、2018）は、マルチクラスデータセットについて学習されたとき（例：ImageNet（Russakovsky et al。、2015））に、他のものよりはるかに複数の画像クラスをモデリングすることが困難であることが観測出来る。

- For example, while the state-of-the-art ImageNet GAN model (Miyato & Koyama, 2018) excels at synthesizing image classes with few structural constraints (e.g., ocean, sky and landscape classes, which are distinguished more by texture than by geometry), it fails to capture geometric or structural patterns that occur consistently in some classes (for example, dogs are often drawn with realistic fur texture but without clearly defined separate feet).
    - 例えば、最先端のImageNet GANモデル（Miyato＆Koyama、2018）は、構造的制約がほとんどない画像クラス（例えば、幾何学よりもテクスチャーによってより多く識別出来るような、海洋、空、景観クラスなど）の合成に優れて [excels] いる一方で、 
    - いくつかのクラスで一貫して発生する幾何学的または構造的パターンを捉えることができない（例えば、犬はしばしば現実的な毛皮の質感で描かれているが、明確に定義された別々の足はない）。

- One possible explanation for this is that previous models rely heavily on convolution to model the dependencies across different image regions. Since the convolution operator has a local receptive field, long range dependencies can only be processed after passing through several convolutional layers.
    - これに対する１つの可能な説明は、以前のモデルは異なる画像領域にわたる依存性をモデル化するために畳み込みに大きく依存しているということである。 
    - 畳み込み演算子には局所的な受容野があるので、長距離依存性はいくつかの畳み込み層を通過した後にのみ処理することができます。

- This could prevent learning about long-term dependencies for a variety of reasons: a small model may not be able to represent them, optimization algorithms may have trouble discovering parameter values that carefully coordinate multiple layers to capture these dependencies, and these parameterizations may be statistically brittle and prone to failure when applied to previously unseen inputs.
    - これは、さまざまな理由で長期の依存関係について学ぶのを妨げる可能性があります
    - 小さなモデルではそれらを表現できない、
    - 最適化アルゴリズムでは、これらの依存関係を捉えるために、複数のレイヤーを慎重に調整して [coordinate] パラメーター値を見つけるのは困難です。
    - そして、これらのパラメーター化は、これまで目に見えなかった入力に適用するとき、統計的に [statistically]、脆くなり [brittle]、失敗する傾向があり [prone] ます。

- Increasing the size of the convolution kernels can increase the representational capacity of the network but doing so also loses the computational and statistical efficiency obtained by using local convolutional structure.
    - 畳み込みカーネルのサイズを大きくすると、ネットワークの表現能力を高めることができますが、ローカル畳み込み構造を使用することによって得られる計算上および統計上の効率も低下します。

- Self-attention (Cheng et al., 2016; Parikh et al., 2016; Vaswani et al., 2017), on the other hand, exhibits a better balance between the ability to model long-range dependencies and the computational and statistical efficiency.
    - 一方、自己注意（Cheng et al。、2016; Parikh et al。、2016; Vaswani et al。、2017）は、長期的な依存関係をモデル化する能力と計算上および統計上の効率との間のより良いバランスを示します。 。

- The self-attention module calculates response at a position as a weighted sum of the features at all positions, where the weights – or attention vectors – are calculated with only a small computational cost.
    - 自己注意モジュールは、ある位置での応答をすべての位置での特徴の加重和として計算します。ここで、重み、つまり注意ベクトルは、わずかな計算コストで計算されます。

---

- In this work, we propose Self-Attention Generative Adversarial Networks (SAGANs), which introduce a self-attention mechanism into convolutional GANs. The self-attention module is complementary to convolutions and helps with modeling long range, multi-level dependencies across image regions.
    - 本研究では、自己注意メカニズムを畳み込みGANに導入する自己注意生成型敵対的ネットワーク（SAGAN）を提案する。
    - セルフアテンションモジュールは畳み込みを補完する [complementary] ものであり、画像領域全体にわたる長距離のマルチレベル依存関係のモデリングに役立ちます。

- Armed with self-attention, the generator can draw images in which fine details at every location are carefully coordinated with fine details in distant portions of the image. Moreover, the discriminator can also more accurately enforce complicated geometric constraints on the global image structure.
    - 自己注意で武装して [Armed with] 、生成器は、あらゆる場所の細かい詳細が、画像の遠い部分の細部において細かい詳細で注意深く調整されているような、画像を描くことができます。
    - さらに、識別器はまた、グローバル画像構造において、複雑な幾何学的制約をより正確に強制することもできる。

---

- In addition to self-attention, we also incorporate recent insights relating network conditioning to GAN performance. The work by (Odena et al., 2018) showed that well-conditioned generators tend to perform better. We propose enforcing good conditioning of GAN generators using the spectral normalization technique that has previously been applied only to the discriminator (Miyato et al., 2018).
    - 自己注意に加えて、私達はまたネットワーク調整をGANの性能に関連させる最近の洞察を取り入れています。 （Odena et al。、2018）による研究は、条件の良い発電機がより良く機能する傾向があることを示した。 我々は、これまで弁別器にのみ適用されてきたスペクトル正規化技術を使用して、GAN発生器の良好な条件付けを強化することを提案する（Miyato et al。、2018）。

---

- We have conducted extensive experiments on the ImageNet dataset to validate the effectiveness of the proposed self- attention mechanism and stabilization techniques. SAGAN significantly outperforms prior work in image synthesis by boosting the best reported Inception score from 36.8 to 52.52 and reducing Fre ́chet Inception distance from 27.62 to 18.65. 
    - 提案した自己注意メカニズムと安定化手法の有効性を検証するために、ImageNetデータセットに対して広範な実験を行った。 SAGANは、最もよく報告されているインセプションスコアを36.8から52.52に高め、フレシェインセプション距離を27.62から18.65に減らすことによって、画像合成における以前の研究よりも著しく優れています。

- Visualization of the attention layers shows that the generator leverages neighborhoods that correspond to object shapes rather than local regions of fixed shape.
    - 注意層の視覚化は、ジェネレータが固定形状の局所領域ではなくオブジェクト形状に対応する近傍を利用することを示しています。

- Our code is available at https://github.com/ brain-research/self-attention-gan.

---

![image](https://user-images.githubusercontent.com/25688193/61586929-24bacf00-abba-11e9-95f1-b218ad7a6ddc.png)

- > Figure 1. The proposed SAGAN generates images by leveraging complementary features in distant portions of the image rather than local regions of fixed shape to generate consistent objects/scenarios.
    - > 図１：提案されたＳＡＧＡＮは、一貫したオブジェクト／シナリオを生成するために、固定形状の局所領域ではなく画像の離れた部分における相補的な特徴を利用することによって画像を生成する。

- > In each row, the first image shows five representative query locations with color coded dots. The other five images are attention maps for those query locations, with corresponding color coded arrows summarizing the most-attended regions.
    - > 各行で、最初の画像は色分けされたドットで5つの代表的なクエリの場所を示しています。
    - > 他の5つの画像は、最も注目されている領域をまとめた対応する色分けされた矢印とともに、それらのクエリの場所に対してのアテンションマップです。

# ■ 結論

## x. 論文の項目名 (Conclusion)


# ■ 何をしたか？詳細

## 3. Self-Attention Generative Adversarial Networks

![image](https://user-images.githubusercontent.com/25688193/61586275-a0ad1b00-abaa-11e9-9a38-4c6da8ae7a5f.png)

- > Figure 2. The proposed self-attention module for the SAGAN. The ⊗ denotes matrix multiplication. The softmax operation is performed on each row.

---

- Most GAN-based models (Radford et al., 2016; Salimans et al., 2016; Karras et al., 2018) for image generation are built using convolutional layers. Convolution processes the information in a local neighborhood, thus using convolutional layers alone is computationally inefficient for modeling long-range dependencies in images.
    - 画像生成のためのほとんどのGANベースのモデル（Radfordら、2016年; Salimansら、2016年; Karrasら、2018年）は、たたみ込みレイヤを使用して構築されています。 畳み込みは局所的な近傍の情報を処理するので、畳み込みレイヤのみを使用することは、画像内の長距離依存性をモデル化するために計算上非効率的である。

- In this section, we adapt the non-local model of (Wang et al., 2018) to introduce self-attention to the GAN framework, enabling both the generator and the discriminator to efficiently model relationships between widely separated spatial regions. We call the proposed method Self-Attention Generative Adversarial Networks (SAGAN) because of its self-attention module (see Figure 2).
    - この節では、GANフレームワークに自己注意を導入するために（Wang et al。、2018）の非局所モデルを適応させ、
    - 生成器と識別器の両方が広く離れた空間領域間の関係を効率的にモデル化できるようにします。
    - 我々は、その自己注意モジュールのために提案された方法を自己注意生成型敵対的ネットワーク（ＳＡＧＡＮ）と呼ぶ（図２参照）。

---

- The image features from the previous hidden layer x ∈ RC×N are first transformed into two feature spaces f,g to calculate the attention, where f(x) = W_f * x, g(x) = W_g * x.
    - 前の隠れ層ｘ∈ＲＣ×Ｎからの画像特徴は、最初に２つの特徴空間ｆ、ｇに変換されて注意を計算する。ここで、ｆ（ｘ）＝ Ｗ＿ｆ ＊ ｘ、ｇ（ｘ）＝ Ｗ＿ｇ ＊ ｘである。

![image](https://user-images.githubusercontent.com/25688193/61578203-28a10f80-ab2e-11e9-9f56-1c8fa23ded5b.png)

- and β_{j,i} indicates the extent to which the model attends to the i^{th} location when synthesizing the j^{th} region.
    - <font color="Pink">**β_{j,i} は（中間層からの特徴マップ x の）j 番目の領域を合成するときに、モデルが i 番目の位置に関与する（注意を向ける）度合い [extent to] を示します。**</font>

- Here, C is the number of channels and N is the number of feature locations of features from the previous hidden layer.

- The output of the attention layer is o = (o1,o2,...,oj,...,oN) ∈ RC×N , where,

![image](https://user-images.githubusercontent.com/25688193/61578206-49696500-ab2e-11e9-88d1-3b3c7dc2657c.png)

- In the above formulation, Wg ∈ R Wh ∈ RC ̄×C , and Wv ∈ RC×C ̄ are the learned weight matrices, which are implemented as 1×1 convolutions.

- Since We did not notice any significant performance decrease when reducing the channel number of C ̄ to be C/k, where k = 1, 2, 4, 8 after few training epochs on ImageNet. For memory efficiency, we choose k = 8 (i.e., C ̄ = C/8) in all our experiments.
    - ImageNetでのいくつかの訓練エポックの後、C- のチャネル数をC / kに減らすとき、我々はいかなる大きな性能低下にも気付かなかったので。 メモリ効率のために、我々は全ての実験においてｋ ＝ ８（すなわち、Ｃ- ＝ Ｃ ／ ８）を選択する。

---

- In addition, we further multiply the output of the attention layer by a scale parameter and add back the input feature map. Therefore, the final output is given by,
    - さらに、注意レイヤの出力にスケールパラメータをさらに掛けて、入力フィーチャマップを追加します。 したがって、最終的な出力は以下のようになる。

![image](https://user-images.githubusercontent.com/25688193/61578223-833a6b80-ab2e-11e9-992a-145edc376146.png)

- where γ is a learnable scalar and it is initialized as 0. Introducing the learnable γ allows the network to first rely on the cues in the local neighborhood – since this is easier – and then gradually learn to assign more weight to the non-local evidence.
    - **ここで、γは学習可能なスカラーであり、0として初期化されます。**
    - **学習可能なγを導入すると、まずネットワークの近くにある手がかりに頼ることができます。** 
    - **その後、非局所的証拠により多くの重みを割り当てることを徐々に学びます。**

- The intuition for why we do this is straightforward: we want to learn the easy task first and then progressively increase the complexity of the task.
    - なぜこれをするのかについての直感は簡単です。最初に簡単なタスクを学び、それからタスクの複雑さを徐々に増やしたいです。

- In the SAGAN, the proposed attention module has been applied to both the generator and the discriminator, which are trained in an alternating fashion by minimizing the hinge version of the adversarial loss (Lim & Ye, 2017; Tran et al., 2017; Miyato et al., 2018),
    - SAGANでは、提案されたアテンションモジュールは、生成器と識別器の両方に適用されてきました。 他、２０１８）、

![image](https://user-images.githubusercontent.com/25688193/61578233-a2d19400-ab2e-11e9-9b94-32b9a3191249.png)


## 4. Techniques to Stabilize the Training of GANs

- We also investigate two techniques to stabilize the training of GANs on challenging datasets.

- First, we use spectral normalization (Miyato et al., 2018) in the generator as well as in the discriminator.

- Second, we confirm that the two- timescale update rule (TTUR) (Heusel et al., 2017) is effective, and we advocate using it specifically to address slow learning in regularized discriminators.
    - 次に、2倍スケールの更新規則（TTUR）（Heusel et al。、2017）が有効であることを確認し、それを正則化された弁別器の遅い学習に対処するために特に使用することを主張します。[advocate]


### 4.1. Spectral normalization for both generator and discriminator

- Miyato et al. (Miyato et al., 2018) originally proposed stabilizing the training of GANs by applying spectral normalization to the discriminator network. Doing so constrains the Lipschitz constant of the discriminator by restricting the spectral norm of each layer. Compared to other normalization techniques, spectral normalization does not require extra hyper-parameter tuning (setting the spectral norm of all weight layers to 1 consistently performs well in practice). Moreover, the computational cost is also relatively small.
    - 宮戸ら。 （Miyato et al。、2018）はもともとスペクトル正規化を弁別器ネットワークに適用することによってGANの訓練を安定化することを提案した。 そうすることは、各層のスペクトルノルムを制限することによって弁別子のリプシッツ定数を制約する。 他の正規化手法と比較して、スペクトル正規化は余分なハイパーパラメータ調整を必要としません（すべてのウェイトレイヤのスペクトルノルムを1に設定すると、実際には常にうまく機能します）。 さらに、計算コストも比較的小さい。

---

- We argue that the generator can also benefit from spectral normalization, based on recent evidence that the conditioning of the generator is an important causal factor in GANs’ performance (Odena et al., 2018). Spectral normalization in the generator can prevent the escalation of parameter magnitudes and avoid unusual gradients. 
    - ジェネレーターのコンディショニングがGANのパフォーマンスにおける重要な因果要因であるという最近の証拠に基づいて、ジェネレーターもスペクトル正規化の恩恵を受けることができると主張しています（Odena et al。、2018）。 発生器におけるスペクトル正規化は、パラメータの大きさの増大を防ぎ、異常な勾配を避けることができます。
    
- We find empirically that spectral normalization of both generator and discriminator makes it possible to use fewer discriminator updates per generator update, thus significantly reducing the computational cost of training. The approach also shows more stable training behavior.
    - 我々は経験的に、発生器と弁別器の両方のスペクトル正規化が発生器更新当たりのより少ない弁別器更新を使用することを可能にし、従って訓練の計算コストを著しく減少させることを発見した。 このアプローチはまた、より安定したトレーニング行動を示します。


### 4.2. Imbalanced learning rate for generator and discriminator updates

- In previous work, regularization of the discriminator (Miyato et al., 2018; Gulrajani et al., 2017) often slows down the GANs’ learning process. In practice, methods using regularized discriminators typically require multiple (e.g., 5) discriminator update steps per generator update step during training.
    - 以前の研究では、識別器の正則化（Miyato et al。、2018; Gulrajani et al。、2017）はGANの学習プロセスを遅くすることがよくあります。
    - 実際には、正規化された識別器を使用する方法は通常、学習中に発生器更新ステップごとに複数回の識別器更新ステップ（例えば５回）を必要とする。
    
- Independently, Heusel et al (Heusel et al., 2017) have advocated using separate learning rates (TTUR) for the generator and the discriminator. We propose using TTUR specifically to compensate for the problem of slow learning in a regularized discriminator, making it possible to use fewer discriminator steps per generator step. Using this approach, we are able to produce better results given the same wall-clock time.
    - 独立して、Heusel ら（Heusel et al。、2017）は、発生器と識別器に別々の学習率（TTUR）を使用することを提唱しています。
    - 我々は、正則化された識別器における遅い学習の問題を補償するために特にＴＴＵＲを使用することを提案し、それにより生成器１ステップ当たりに対するより少ない識別器ステップ回数を使用することを可能にする。 このアプローチを使用すると、同じ実時間で同じ結果が得られます。


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## 5. Experiments

- To evaluate the proposed methods, we conducted extensive experiments on the LSVRC2012 (ImageNet) dataset (Russakovsky et al., 2015). First, in Section 5.1, we present experiments designed to evaluate the effectiveness of the two proposed techniques for stabilizing GANs’ training. Next, the proposed self-attention mechanism is investigated in Section 5.2. Finally, our SAGAN is compared with state- of-the-art methods (Odena et al., 2017; Miyato & Koyama, 2018) on the image generation task in Section 5.3. Models were trained for roughly 2 weeks on 4 GPUs each, using sychronous SGD (as there are well known difficulties with asynchronous SGD - see e.g. (Odena, 2016)).
    - 提案された方法を評価するために、我々はLSVRC2012（ImageNet）データセットに対して広範な実験を行った（Russakovsky et al。、2015）。 まず、5.1節で、GANの訓練を安定させるために提案された2つの手法の有効性を評価するために設計された実験を紹介します。 次に提案したセルフアテンションメカニズムについて5.2節で検討する。 最後に、5.3節の画像生成タスクについて、SAGANと最先端の方法（Odena et al。、2017; Miyato＆Koyama、2018）を比較します。 モデルは、同期ＳＧＤを使用して、それぞれ４ＧＰＵについておよそ２週間訓練された（非同期ＳＧＤにはよく知られている困難があるので - 例えば（Ｏｄｅｎａ、２０１６）を参照）。

### Evaluation metrics.

- We choose the Inception score (IS) (Salimans et al., 2016) and the Fre ́chet Inception distance (FID) (Heusel et al., 2017) for quantitative evaluation. Though alternatives exist (Zhou et al., 2019; Khrulkov & Oseledets, 2018; Olsson et al., 2018), they are not widely used. The Inception score (Salimans et al., 2016) computes the KL divergence between the conditional class distribution and the marginal class distribution. Higher Inception score indicates better image quality. We include the Inception score because it is widely used and thus makes it possible to compare our results to previous work. However, it is important to understand that Inception score has serious limitations—it is intended primarily to ensure that the model generates samples that can be confidently recognized as belonging to a specific class, and that the model generates samples from many classes, not necessarily to assess realism of details or intra-class diversity.

- FID is a more principled and comprehensive metric, and has been shown to be more consistent with human evaluation in assessing the realism and variation of the generated samples (Heusel et al., 2017). FID calculates the Wasserstein-2 distance between the gen- erated images and the real images in the feature space of an Inception-v3 network. Besides the FID calculated over the whole data distribution (i.e.., all 1000 classes of images in ImageNet), we also compute FID between the generated images and dataset images within each class (called intra FID (Miyato & Koyama, 2018)). Lower FID and intra FID values mean closer distances between synthetic and real data distributions. In all our experiments, 50k samples are randomly generated for each model to compute the Inception score, FID and intra FID.


# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


