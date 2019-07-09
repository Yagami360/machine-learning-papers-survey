# ■ 論文
- 論文タイトル："xxx"
- 論文リンク：
- 論文投稿日付：
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- Providing vibrotactile feedback that corresponds to the state of the virtual texture surfaces allows users to sense haptic properties of them. However, hand-tuning such vibrotactile stimuli for every state of the texture takes much time.
    - 仮想テクスチャ表面の状態に対応する振動触覚フィードバックを提供することは、ユーザがそれらの触覚特性を感知することを可能にする。しかしながら、テクスチャのあらゆる状態に対してそのような振動触覚刺激を手動で調整することは多くの時間がかかる。

- Therefore, we propose a new approach to create models that realize the automatic vibrotactile generation from texture images or attributes. In this paper, we make the first attempt to generate the vibrotactile stimuli leveraging the power of deep generative adversarial training. Specifically, we use conditional generative adversarial networks (GANs) to achieve generation of vibration during moving a pen on the surface.
    - そこで、テクスチャ画像や属性から自動振動触覚生成を実現するモデルを作成するための新しいアプローチを提案します。本稿では、深い生成的な敵対的訓練の力を利用して振動触覚刺激を生成する最初の試みを行います。具体的には、我々は、表面上でペンを動かしている間に振動の発生を達成するために条件付き生成型敵対的ネットワーク（ＧＡＮ）を使用する。

- The preliminary user study showed that users could not discriminate generated signals and genuine ones and users felt realism for generated signals. Thus our model could provide the appropriate vibration according to the texture images or the attributes of them. Our approach is applicable to any case where the users touch the various surfaces in a predefined way.
    - 予備的なユーザ調査は、ユーザが生成された信号と本物の信号とを区別することができず、ユーザが生成された信号に対してリアリズムを感じることを示した。したがって、我々のモデルは、テクスチャ画像またはそれらの属性に従って適切な振動を提供することができます。私たちのアプローチは、ユーザーが事前定義された方法でさまざまなサーフェスに触れるような場合にも適用できます。


# ■ イントロダクション（何をしたいか？）

## 1. Introduction

- The vibrotactile sense enables humans to perceive texture surface properties through tool-surface interaction. Unfortunately, the richness of the vibrotactile responses from virtual texture surfaces is missing from current tool-surface interactions on a touchscreen. The tool-surface interactions are composed of simple gestures such as tapping or flickering, so vibrotactile designer should find the appropriate vibrotactile signals for each gesture. However, it is difficult to find ones. Though there are vibrotactile datasets which is made public, it is rare to find the appropriate vibrotactile signals from them. It is because such datasets contains at most 100 kinds of textures, as compared to countless kind of texture in the real world.
    - 振動触覚は人間が道具 - 表面相互作用を通してテクスチャー表面特性を知覚することを可能にする。 残念なことに、仮想テクスチャ表面からの振動触覚応答の豊富さは、タッチスクリーン上の現在のツール - 表面相互作用から欠けている。 ツールとサーフェスの相互作用は、タップやちらつきなどの単純なジェスチャで構成されているため、振動触覚設計者は各ジェスチャに適した振動触覚信号を見つける必要があります。 しかし、見つけるのは難しいです。 公開されている振動触覚データセットがありますが、それらから適切な振動触覚信号を見つけることは稀です。 これは、現実の世界では数え切れないほどの種類のテクスチャと比較して、このようなデータセットには最大100種類のテクスチャが含まれているためです。

---

- Instead of looking into datasets, vibrotactile modeling has been studied for a long time to provide such responses. However, there is no model that interactively generates vibrotactile responses based on the state of the tool and the state of the surfaces. Such model should learn the complex mapping between large input and output space; Inputs are state of the tool (ex. tool’s velocity) and the state of the texture surface (ex. texture’s attributes), on the other hand, outputs are vibrotactile signals. Considering a limitation of a representational power that a trained single model can have, it is difficult to train the model that get both states of the tool and state of the texture surface as input. In other words, there is a trade-off between the model’s interactivity for the tool's state and the one for the texture surface’s state.
    - データセットを調べる代わりに、振動触覚モデリングはそのような応答を提供するために長い間研究されてきました。 しかし、ツールの状態とサーフェスの状態に基づいて対話的に振動触覚応答を生成するモデルはありません。 そのようなモデルは大きな入力と出力空間の間の複雑なマッピングを学ぶべきです。 入力はツールの状態（例：ツールの速度）とテクスチャサーフェスの状態（例：テクスチャの属性）です。一方、出力は振動触覚信号です。 訓練された単一モデルが持つことができる表現力の制限を考慮すると、ツールの状態とテクスチャ表面の状態の両方を入力として得るモデルを訓練することは困難です。 つまり、ツールの状態に対するモデルの対話性とテクスチャサーフェスの状態に対するモデルの対話性の間にはトレードオフがあります。

> ここでいうツールとは？：タブレット端末などのこと

---

- Emerging, recent data-driven approach for haptic modeling mainly focus on the interactivity of the tool’s state. Prior studies mapped the normal force and the velocity magnitude of the tool with vibrational patterns [1, 2]. These vibrational patterns were encoded in the autoregressive model. Their model succeeded in mapping the tool’s state and the vibration patterns. They are suitable for interactions where there is much variability with tool’s velocity and applied force. However, the single model generating vibrational signals only supported single kind of texture that is used during training. Thus, when you try to generate vibrations of another kind of texture, you need to replace the model with another one.
    - ハプティックモデリングに対する最近のデータ駆動型アプローチは、主にツールの状態の対話性に焦点を当てています。 以前の研究では、ツールの垂直力と速度の大きさを振動パターンとマッピングしました[1、2]。 これらの振動パターンは自己回帰モデルでエンコードされています。 彼らのモデルは、ツールの状態と振動パターンをマッピングすることに成功しました。 それらは、ツールの速度や加えられた力に大きなばらつきがある相互作用に適しています。 ただし、振動信号を生成する単一モデルは、トレーニング中に使用される単一種類のテクスチャのみをサポートしていました。 したがって、別の種類のテクスチャの振動を生成しようとすると、モデルを別のモデルと置き換える必要があります。

---

- This paper, on the other hand, focuses on the interactivity for the texture’s state instead of the tool’s state. 
    - これに対して、このホワイトペーパーでは、ツールの状態ではなくテクスチャの状態のインタラクティビティに焦点を当てています。

- Current touchscreen interactions are composed of simple gestures such as tapping or flickering, which is completed in a short time. With such gestures, the tool’s velocity or applied force is approximately constant. On the other hand, such gestures are generally used for various texture surfaces. Thus, we pose the modeling task of generating appropriate vibrotactile signals that correspond to the visual information or attributes of texture.
    - **現在のタッチスクリーンの操作は、タップやフリックなどの簡単なジェスチャで構成されており、短時間で完了します。このようなジェスチャでは、ツールの速度または加えられる力はほぼ一定です。一方、そのようなジェスチャは一般に様々なテクスチャ表面に使用される。したがって、我々は視覚的情報またはテクスチャの属性に対応する適切な振動触覚信号を生成するというモデリング作業を提起する。**

- Such capabilities realize generating haptic signals for even unseen textures automatically or manipulating vibrotactile signals by changing attribute values. As an application of this model, we assume a vibrotactile designing toolkit for tool-surface interactions where designers can (1) set attributes of texture or (2) prepare texture images to generate the appropriate signals for gestures. The model that accomplishes it is required to have the capability to capture rich distribution.
    - このような機能により、目に見えないテクスチャに対しても触覚信号を自動的に生成したり、属性値を変更して振動触覚信号を操作したりできます。このモデルの応用として、設計者が（１）テクスチャの属性を設定する、または（２）ジェスチャに適切な信号を生成するためにテクスチャ画像を準備することができる、ツール - サーフェスインタラクション用の振動触覚設計ツールキットを想定する。それを達成するためのモデルは、豊かな流通を捉える能力を持つことが必要です。

---

- Recently, generative methods that produce novel samples from high-dimensional data distributions, such as images, are finding widespread use. Specifically, Generative Adversarial Networks (GANs) [3] have shown promising results in synthesizing real-world images. Prior research demonstrated that GANs could effectively generate images conditioned on labels [4], texts [5], and so on. In spite of these promising results, there are few studies that used GANs to model time-series data distribution. Indeed, the generation of vibration by GANs has not been realized for now. In this study, we make full use of GANs for indirectly generating vibrotactile signals via time-frequency domain representation, which can be calculated as the image. We train the model so that it can generate vibrotactile signals conditioned on texture images or texture attributes.
    - 最近、画像のような高次元データ分布から新規のサンプルを生成する生成法が広く使用されている。 具体的には、Generative Adversarial Networks（GAN）[3]は、実世界の画像の合成において有望な結果を示しています。 以前の研究では、GANがラベル[4]、テキスト[5]などに基づいて調整された画像を効果的に生成できることが実証されました。 これらの有望な結果にもかかわらず、時系列データ分布をモデル化するためにGANを使用した研究はほとんどありません。 確かに、GANによる振動の発生は今のところ実現されていません。 本研究では、画像として計算することができる時間 - 周波数領域表現を介して間接的に振動触覚信号を生成するためにGANを最大限に利用する。 テクスチャ画像またはテクスチャ属性に基づいて調整された振動触覚信号を生成できるようにモデルをトレーニングします。

---

- The contribution of this study is three-fold. First, to our best knowledge, we introduce the problem of vibrotactile generation and are the first to use GANs to solve it. Second, we succeed in indirectly generate time-series data via time-frequency representation using GANs. Third, our trained single model meets the demand for interactiveness for the state of textures by providing the appropriate vibration that corresponds to the texture images or texture attributes.
    - この研究の貢献は3倍です。 まず、私たちの知る限りでは、振動触覚生成の問題を紹介し、それを解決するためにGANを使用する最初のものです。 次に、GANを用いた時間周波数表現により間接的に時系列データを生成することに成功しました。 第三に、私たちの訓練された単一モデルは、テクスチャ画像またはテクスチャ属性に対応する適切な振動を提供することによって、テクスチャの状態のための対話性に対する要求を満たします。


# ■ 結論

## 5. Conclusion

- In this study, we introduced the problem of vibrotactile generation based on various texture images or attributes during predefined tool-surface interaction, and solved it by adversarial training. The user study showed that users could not discriminate generated signals and genuine ones. Our approach is applicable to any case where the users touch the various surfaces in a predefined way. Thus, our study contributes to the broadening the options of vibrotactile signal preparation in such cases.
    - 本研究では、事前定義されたツールとサーフェスの相互作用中のさまざまなテクスチャ画像または属性に基づく振動触覚生成の問題を紹介し、それを敵対者の訓練によって解決しました。 ユーザー調査は、ユーザーが生成された信号と本物の信号を区別できないことを示しました。 私たちのアプローチは、ユーザーが事前定義された方法でさまざまなサーフェスに触れるような場合にも適用できます。 したがって、我々の研究は、そのような場合の振動触覚信号処理の選択肢を広げることに貢献しています。

# ■ 何をしたか？詳細

## 3 Vibrotactile Signal Generation

### 3.1 Concept of Overall Model

- By utilizing GANs’ capability to capture rich data distributions, we would like to make the single generative model that has following features: automatic generation of vibrotactile signals either (1) from given texture images or (2) from given texture attributes.
    - **GANの豊富なデータ分布を捉える機能を利用して、（1）与えられたテクスチャ画像から、または（2）与えられたテクスチャ属性から、振動触覚信号を自動生成するという単一生成モデルを作りたいと思います。**

- Though prior research focuses on the interactive generation based on tool’s state, this paper proves the concept above for predefined tool’s state under constrained touch interactions. Among various touch interactions, we focus on the task of moving a pen on the texture surface.
    - これまでの研究ではツールの状態に基づく対話型生成に焦点が当てられていましたが、このホワイトペーパーでは、拘束されたタッチ操作の下での定義済みのツールの状態に対する上記の概念を証明します。 **さまざまなタッチ操作の中で、テクスチャ表面上でペンを動かす作業に焦点を当てています。**

---

- The overall diagram of our model is shown in Fig. 1. It consists of two parts: an encoder network, and a generator network. They are trained separately. The encoder is trained as an image classifier and it encodes texture images into a label vector c. The generator is trained with discriminator in GANs training framework and generates spectrogram that is a representation of vibration in a time-frequency domain. We describe the training details for each network in the following sections. The overall model enables end-to-end generation from visual images or label attributes of texture to the vibrotactile wave.
    - **このモデルの全体図を図1に示します。これは、エンコーダネットワークとジェネレータネットワークの2つの部分で構成されています。 彼らは別々に訓練されています。**
    - **エンコーダは画像分類子として訓練され、テクスチャ画像をラベルベクトルcにエンコードします。**
    - **ジェネレータは、GANトレーニングフレームワークで弁別器を使用してトレーニングされ、時間 - 周波数領域での振動の表現であるスペクトログラムを生成します。**
    - 次のセクションでは、各ネットワークのトレーニングの詳細について説明します。 全体的なモデルは、視覚的な画像やテクスチャのラベル属性から振動触覚波までのエンドツーエンドの生成を可能にします。

---

- We describe the data flow step by step based on Fig. 1. The input into the model is either a class label that represents the tactile attributes of the texture or a texture image. When the image is input, the label vector c is extracted from the texture image through encoder network. The label vector c is a categorical variable that shows the attributes of the texture. Next, the label vector c is passed into the generator network. The generator concatenates the label c and the random noise z and transforms them into the spectrogram. 
    - 図1に基づいて、データフローを段階的に説明します。モデルへの入力は、テクスチャの触覚属性を表すクラスラベルまたはテクスチャ画像のいずれかです。
    - 画像が入力されると、ラベルベクトルｃがエンコーダネットワークを通してテクスチャ画像から抽出される。 ラベルベクトルcは、テクスチャの属性を示す質的変数です。
    - 次に、ラベルベクトルcがジェネレータネットワークに渡されます。 ジェネレータはラベルcとランダムノイズzを連結し、それらをスペクトログラムに変換します。

- The generated spectrogram is converted into the acceleration wave format by Griffin-Lim algorithm [10]. Then the wave format data is output to the user. With this overall model, users can input either label information or texture images to obtain vibration. That is why we do not adopt the network like pix2pix [6], which only supports input as images and converts images directory into signals.
    - 生成されたスペクトログラムは、Griffin-Limアルゴリズム[10]によって加速度波フォーマットに変換されます。 そして、その波形データをユーザに出力する。 この全体モデルでは、ユーザはラベル情報またはテクスチャ画像を入力して振動を得ることができます。 そのため、画像としての入力のみをサポートし、画像ディレクトリを信号に変換するpix2pix [6]のようなネットワークを採用していません。

---

- Acceleration signals are used as vibrotactile stimulus in our model. In order to train the whole network, we use dataset [11], which contains acceleration signals and captured images during movement task. The pairs of signals and images are annotated with 108 classes.
    - **加速度信号は、私たちのモデルでは振動触覚刺激として使われています。 ネットワーク全体をトレーニングするために、我々はデータセット[11]を使用します。これには、移動タスク中に加速度信号とキャプチャされた画像が含まれます。 信号と画像のペアには108のクラスが付いています。**

### 3.2 Encoder

- We trained the image encoder that encoded texture images into the label vector c. We adopted the deep residual network (ResNet-50) [12] architecture. We fine-tuned all the layers of the ResNet-50 that had been pre-trained with ImageNet [13]. We used Adam optimizer with a mini-batch size of 64. The learning rate started from 1e-3 and was decreased by a factor of 0.1 when the training error plateaued.
    - テクスチャ画像をラベルベクトルcにエンコードする画像エンコーダを訓練しました。 我々はディープ残差ネットワーク（ResNet-50）[12]アーキテクチャを採用した。 ImageNetで事前にトレーニングされたResNet-50のすべてのレイヤーを微調整しました[13]。 Adamオプティマイザをミニバッチサイズ64で使用しました。学習率は1e-3から始まり、トレーニングエラーが横ばいになったときは0.1倍減少しました。

---

- The size of provided images by [11] is 320 x 480. We fed them into the encoder network. For training phase of encoder network, we followed ordinary data augmentation settings. We scaled an image with factors in [1, 1.3], randomly cropped 128 x 128 size of it, flipped it horizontally and vertically, rotated it by a random angle. The recent data augmentation technique of random erasing and mixup were also used.
    - [11]で提供される画像のサイズは320 x 480です。私たちはそれらをエンコーダネットワークに送りました。 エンコーダネットワークのトレーニングフェーズでは、通常のデータ拡張設定に従いました。 画像を[1、1.3]の係数で拡大縮小し、サイズを128 x 128のサイズでランダムにトリミングし、水平方向と垂直方向に反転させ、ランダムな角度で回転させました。 ランダム消去および混合の最近のデータ増強技術も使用された。

---

- As a result of training, the trained encoder achieved a classification accuracy of more than 95 percent on the testing set. After the network was trained, its last layer was removed, and the feature vector of the second to the last layer having dimension of label vector was used as the image encoding in our generator network.
    - トレーニングの結果、トレーニングを受けたエンコーダはテストセットで95％以上の分類精度を達成しました。 ネットワークが訓練された後、その最後の層が除去され、ラベルベクトルの次元を有する最後から二番目の層の特徴ベクトルが、我々のジェネレータネットワークにおける画像符号化として使用された。

### 3.3 Generator

#### Network Architecture and training settings.

- Generator was trained with discriminator in GANs framework. During training, the discriminator learned to discriminate between genuine and generated samples, while the generator learned to fool the discriminator. Generator output samples 𝑥 = 𝐺(𝑧, 𝑐) conditioned on both random noise vector z and a label vector c from dataset. Discriminator had two outputs: D(x) the probability of the sample x being genuine, and 𝑃(𝑥) = 𝒄, the predicted label vector of x. After training, the discriminator was removed and the generator was only used in our model. 
    - ジェネレータはGANの枠組みで弁別器で訓練されました。 訓練中、弁別器は本物のサンプルと生成されたサンプルを区別することを学び、ジェネレータは弁別器をだますことを学びました。 データセットからのランダムノイズベクトルzとラベルベクトルcの両方に基づいて生成された生成器出力サンプル𝑥=𝐺（𝑧、𝑐）。 識別器には2つの出力があります。サンプルxが本物である確率D（x）、およびxの予測ラベルベクトルである𝑃（𝑥）=𝒄。 訓練の後、弁別器は取り除かれ、発生器は我々のモデルでのみ使用されました。

- Inspired by [14], we employed architecture and loss function, which was based on SRResNet[7], DRAGAN[15], and AC-GAN[8]. The architecture of generator and discriminator are shown in Fig. 2.
    - **[14]に触発されて、SRResNet [7]、DRAGAN [15]、AC-GAN [8]に基づいたアーキテクチャと損失関数を採用しました。 生成器と識別器のアーキテクチャを図2に示します。**

---

- Acceleration signals orthogonal to the surface during movement task were used as vibrotactile stimulus and we aim at generating the signals by generator. For now, there are few studies generating time series data using GANs. It is because GANs are poor at generating time-series data though they are good at generating 2D images. Therefore, we chose amplitude spectrogram as a representation of the acceleration signals and trained GANs to generate spectrogram as if that was 2D image. The same dataset used for training encoder contained acceleration signals during movement task. Each signal had 4 seconds long and the sampling rate was 10 kHz. We computed the spectrogram from wave format using 512-point Short-Time Fourier Transform (STFT) with a 512 hamming window and a 128 hop size. Then, the linear amplitude of the spectrogram was converted to the logarithmic scale. We cropped the spectrogram and resized it into 128 x 128 size. As a result, the spectrogram contained the information of time- frequency domain up to 256 Hz for 1.625 seconds long. The values in the spectrogram were normalized into the range from 0 to 1.
    - 運動課題中に表面に直交する加速度信号を振動触覚刺激として使用し、そして我々は信号を発生器によって発生させることを目的とする。今のところ、GANを使用して時系列データを生成する研究はほとんどありません。これは、GANが2D画像の生成には長けているにもかかわらず、時系列データの生成には長けているためです。したがって、加速度信号とトレーニングされたGANの表現として振幅スペクトログラムを選択し、それがあたかも2D画像であるかのようにスペクトログラムを生成しました。エンコーダのトレーニングに使用されたものと同じデータセットには、移動タスク中の加速度信号が含まれていました。各信号の長さは4秒で、サンプリングレートは10 kHzです。我々は、512ハミングウィンドウと128ホップサイズの512ポイント短時間フーリエ変換（STFT）を使用して、ウェーブフォーマットからスペクトログラムを計算しました。次に、スペクトログラムの線形振幅を対数目盛に変換した。スペクトログラムをトリミングして、サイズを128 x 128に変更しました。その結果、スペクトログラムは1.625秒間で最大256Hzまでの時間周波数領域の情報を含んでいた。スペクトログラムの値は、0から1の範囲に正規化されています。

---

- We selected 9 textures out of 108 textures for GANs’ training because it is stable to train conditional GANs with fewer number of conditional label dimensions. Thus, the dimension of categorical label c was 9. On the other hand, the dimension of noise z was 50. The selected 9 textures were representative of 9 groups of LMT haptic texture database [11] (Fig. 3). We used Adam optimizer with a mini-batch size of 64. The learning rate was fixed at 2e-4.
    - GANのトレーニング用に108個のテクスチャから9個のテクスチャを選択しました。条件付きラベル次元の数が少ない条件付きGANをトレーニングするのが安定しているためです。 したがって、カテゴリカルラベルcの次元は9でした。一方、ノイズzの次元は50でした。選択された9つのテクスチャは、LMT触覚テクスチャデータベースの9つのグループを表していました[11]（図3）。 Adamオプティマイザをミニバッチサイズ64で使用しました。学習率は2e-4に固定されました。

#### Training Results of Generator. 

- The spectrogram in test dataset and the one generated by generator are shown in Fig. 4. The comparison between them shows the trained generator could generate the spectrograms that appear indistinguishable from test ones. We describe the qualitative evaluation by user study in section 4. 
    - テストデータセットのスペクトログラムとgeneratorによって生成されたスペクトログラムを図4に示します。それらの比較から、訓練されたジェネレータがテストのものと区別がつかないように見えるスペクトログラムを生成できることがわかります。 我々は4章でユーザースタディによる定性的評価について述べる。
    
- On the other hand, it is generally difficult to quantitatively evaluate the GANs. The ordinary evaluation metric of GANs, namely the “inception score” cannot to be applied to our case because the “inception score” is only applied to standard dataset such as CIFAR-10. Instead, we observe that t-SNE is a good tool to examine the distribution of generated images. A two dimensional t-SNE visualization is shown in Fig. 5. It is shown that the generated and test samples made group for each class label.
    - 一方、一般的にGANを定量的に評価することは難しい。 「インセプションスコア」はCIFAR-10などの標準データセットにのみ適用されるため、GANの通常の評価指標、つまり「インセプションスコア」を私たちのケースに適用することはできません。 代わりに、我々は、t-SNEが生成された画像の分布を調べるための良いツールであることを観察します。 二次元ｔ − ＳＮＥ視覚化を図５に示す。生成されたサンプルおよび試験サンプルが各クラスラベルについてグループ化されたことが示される。


## 3.4 End-to end (E2E) Network

- E2E cross modal generation of signals from texture images are realized by combining encoder and generator. The encoder was trained with 9 classes instead of 108 classes in accordance with input dimension of conditional generator. Fig. 7shows the generated spectrogram from the texture image and genuine one for each test image in the dataset. The comparison between them shows the E2E network could generate the spectrograms that seem indistinguishable from test one. We describe the qualitative evaluation by user study in section 4.
    - テクスチャ画像からの信号のＥ２Ｅクロスモーダル生成は、符号器と生成器を組み合わせることによって実現される。 条件付きジェネレータの入力次元に従って、エンコーダは108クラスではなく9クラスでトレーニングされました。 図７は、テクスチャ画像から生成されたスペクトログラムとデータセット内の各テスト画像についての本物のスペクトログラムを示す。 それらの比較は、E2Eネットワークがテストのものと区別がつかないように見えるスペクトログラムを生成することができることを示します。 第4節では利用者調査による質的評価について述べる。



# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## 4 User study

- In user studies, participants’ task was to move a pen-type device on a surface of a tablet device while receiving vibrotactile feedback. Our experimental system was constituted of the tablet device (Apple Inc., iPad Pro 9.7 inch), an amplifier (Lepai Inc., LP-2020A +), and a pen-type device with a vibrator (Fig. 8). The pen-type device, which we handcrafted, is specifically described in the next paragraph.
    - ユーザーの研究では、参加者の仕事は、振動触覚フィードバックを受けながら、タブレット型デバイスの表面でペン型デバイスを動かすことでした。 我々の実験システムは、タブレットデバイス（Apple Inc.、iPad Pro 9.7インチ）、アンプ（Lepai Inc.、LP-2020A +）、およびバイブレータ付きペン型デバイスから構成されていました（図8）。 我々が手作りしたペン型装置は、次の段落で具体的に説明される。

---

- The pen-type device was about 20 g weight and about 140 mm long. The diameter of the grip part of the pen was about 10 mm. The pen tip wore conductive material that is ordinary used for the stylus. Since the shaft of the pen used in these studies was made of plastic and does not conduct to the grip part, we winded a conductive sheet on the grip to react with a capacitance type touch screen. Inside the pen-type device, the vibrator (ALPS Inc., HAPTICTM Reactor) was embedded at the position of 2cm distance from the tip of the pen where participants gripped. The vibrator was small (35.0 mm × 5.0 mm × 7.5 mm) and light (about 5 g) enough not to prevent participants from moving the pen.
    - ペン型装置は、重量約２０ｇ、長さ約１４０ｍｍであった。 ペンの握り部の直径は約１０ｍｍであった。 ペン先は、通常スタイラスに使用される導電性材料を着用していました。 本研究で使用したペンの軸はプラスチック製でグリップ部とは導通しないため、グリップに導電性シートを巻いて静電容量式タッチスクリーンに反応させた。 ペン型装置の内側に、参加者が握ったペンの先端から２ｃｍの距離の位置にバイブレータ（ＡＬＰＳ Ｉｎｃ．、ＨＡＰＴＩＣ（登録商標）Ｒｅａｃｔｏｒ）を埋め込んだ。 バイブレータは小さく（３５．０ｍｍ×５．０ｍｍ×７．５ｍｍ）、参加者がペンを動かすのを妨げない程度に十分に軽い（約５ｇ）であった。

---

- When participants touched and moved the pen on the surface, the vibration signal was output from earphone jack of the tablet, and amplified by the amplifier, and vibrator embedded on the pen presented the vibration to the participants’ fingers
    - 参加者がペンを表面に触れて動かすと、タブレットのイヤホンジャックから振動信号が出力され、アンプで増幅され、ペンに埋め込まれたバイブレータが参加者の指に振動を与えました
    
### 4.2 Task Design

- These studies used a within-participant design. Participants moved the pen-type device along the two different predefined path on screen in succession, while receiving either test or generated vibrational feedback. After that, participants tried to distinguish which stimulus was generated one. They also evaluated the realism of stimuli. In Generator Ex., generated signals were generated by feeding a label vector that represented each class into the generator. In E2E Ex., generated signals were generated by feeding a test image that represented each class into the encoder network. Corresponding class label texts or texture images were displayed on the touch screen. Participants’ task was the same in Generator Ex. and E2E Ex. except that what they saw on screen was class label texts or texture images.
    - これらの研究は参加者内デザインを使用した。 参加者は、テストまたは生成された振動フィードバックを受信しながら、ペン型デバイスを画面上の2つの異なる定義済みパスに沿って連続して移動させました。 その後、参加者はどの刺激が生成された刺激を区別しようとしました。 彼らはまた刺激の現実性を評価した。 ジェネレータ例では、生成された信号は、各クラスを表すラベルベクトルをジェネレータに供給することによって生成されました。 Ｅ２Ｅ実施例では、生成された信号は、各クラスを表すテスト画像をエンコーダネットワークに供給することによって生成された。 対応するクラスラベルテキストまたはテクスチャ画像がタッチスクリーンに表示されました。 参加者の仕事はGenerator Exでも同じです。 Ｅ２Ｅ Ｅｘ。 ただし、画面上に表示されるのはクラスラベルテキストまたはテクスチャイメージだけでした。

---

- The procedure of one trial in participant’s task is described in this paragraph. Participants moved the pen on a virtual texture surface from left to right for about 100 mm distance at fixed speed with their dominant hands. To control the movement speed and distance, the touch screen visualized a bar that indicated where and how much speed to move. According to the bar elongation, participants moved the pen approximately 100 mm distance in 1.6 seconds. Participants were told to hold the pen at the position where a vibrator was embedded.
    - この段落では、参加者の作業における1つの試行の手順について説明します。参加者は、自分の利き手でペンを仮想テクスチャ表面上で左から右へ約100 mmの距離で一定の速度で動かしました。移動速度と距離を制御するために、タッチスクリーンは移動する場所と速度を示すバーを視覚化しました。バーの伸びによると、参加者はペンを1.6秒で約100 mm移動させました。参加者は、バイブレータが埋め込まれている位置にペンを持つように言われました。 

- After completing movement on two surfaces, they answered which stimulus was felt generated one by tapping one of the two answer buttons.    
    - 2つのサーフェス上で動きを終えた後、2つの答えボタンのうちの1つをタップすることによって、どちらの刺激が生成されたと感じられるか答えました。

- Besides, they answered the degree of realism for each stimulus by visual analogue scale (VAS) ratings [16] (Fig. 9 Right).   
    - そのうえ、彼らは、視覚的アナログ尺度（VAS）の評価[16]によって各刺激のリアリズムの程度に答えました（図9右）。
    
- Participants rated the realism that they felt on an analogue scale in this testing method. 
    - 参加者は、このテスト方法でアナログスケールで感じた現実性を評価しました。

- They answered the question “How much realism did you feel?” by rating realism on a 100 mm line on the touch screen anchored by “definitely not” on the left and “felt realism extremely” on the right.
    - 彼らは質問に答えました「あなたはどのくらいのリアリズムを感じましたか？」と答え、100 mmの直線をタッチスクリーン上で固定し、右側に「絶対にない」と固定しました。

- They used the pen-type device to check on this line. The displayed order of test and generated stimuli in one trial was shuffled.  
    - 彼らはペン型装置を使ってこの線をチェックした。 1回の試行で表示されたテストの順序と生成された刺激がシャッフルされました。

---

- Vibration signals that belonged to nine classes of textures that are modeled in Section 3 (Fig. 4) were prepared for this study. Test signals are randomly extracted from test dataset corresponds to each class, and generated signals are generated for each trial. Participants performed the trial ten times for each class. Therefore, each participant performed 180 trials in total for both Generator Ex. and in E2E Ex. E2E Ex. was conducted after Generator Ex. and these studies were held on separate days in order to prevent any satiation effects. To prevent sequential effects, the presentation order of these factors was randomly assigned and counter-balanced across participants.
    - この研究では、3章（図4）でモデル化された9種類のテクスチャに属する振動信号を作成しました。 テスト信号は各クラスに対応するテストデータセットからランダムに抽出され、生成された信号は各試行に対して生成されます。 参加者は各クラスにつき10回試験を実施した。 したがって、各参加者は両方のジェネレータExについて合計180回の試験を実施した。 およびＥ２Ｅ Ｅｘ。 E2E例 発電機例の後に行われた。 そしてこれらの研究は飽食の影響を防ぐために別々の日に行われた。 連続した影響を防ぐために、これらの要素の提示順序は無作為に割り当てられ、参加者間で相殺された。


### 4.3 Result

- Fig. 10 shows the percentage of correctly identifying which stimulus was generated. We call this value as “Correct answer rate”. If this value is close to 50 %, it means that participants failed to distinguish test data from generated data.
    - **図１０は、どの刺激が生成されたかを正しく識別した割合を示す。 この値を「正解率」と呼びます。 この値が50％に近い場合は、参加者がテストデータと生成データを区別できなかったことを意味します。**

- Thus, it is confirmed that our method could generate the vibrational stimuli that were close to the genuine stimuli.
    - 以上のことから、本手法は本物の刺激に近い振動刺激を生成できることが確認された。
    
- The average and the standard error (SE) of “Correct answer rate” were 47.7 ± 1.49 % for Generator Ex., and 48.2 ± 2.49 % for E2E Ex. To investigate whether these rates were out of 50 %, we applied the Chi-Square goodness of fit test. It revealed that the rate of Carpet and Fine Foam condition in E2E Ex. were significantly lower than 50% (Carpet: p<0.01, Fine Foam: p<0.05).
    - 「正解率」の平均値と標準誤差（SE）は、発電機例で47.7±1.49％、E2E例で48.2±2.49％であった。 これらの率が50％から外れているかどうかを調べるために、カイ二乗適合度検定を適用しました。 それはE2E Exのカーペットおよび良い泡の状態の割合を明らかにしました。 ５０％より有意に低かった（カーペット：ｐ ＜０．０１、細かい泡：ｐ ＜０．０５）。

---

- On the other hand, Fig.11 shows the results of how much realism participants felt for each class.
    - 一方、図11は、クラスごとに参加したリアリズムの参加者の実力の結果を示しています。

- The score of the realism of test data was 72.9 ± 1.49 and that of generated data was 73.1 ± 2.93 in Generator Ex.
    - Generator Exのテストデータの現実性のスコアは72.9±1.49、生成データのスコアは73.1±2.93でした。 

- In E2E Ex., the score of the realism of test data was 71.4 ± 2.04 and that of generated data was 70.3 ± 1.81.
    - E2E Ex では、テストデータの現実性のスコアは71.4±2.04、生成されたデータのスコアは70.3±1.81でした。
    
- We used a Student’s paired t-test to each texture condition, and revealed that there were significant differences between the average score of test and generated data for Bamboo in Generator Ex. (p=0.025), and Squared Aluminum Mesh, Bamboo, Card board in E2E Ex. (p=0.026, 0.025, 0.025). 
    - テクスチャの条件ごとにスチューデントのt検定 [Student’s paired t-test ] を使用したところ、Generator ExでBambooのテストの平均スコアと生成されたデータの間に有意差があることがわかりました。 （ｐ ＝ ０．０２５）、
    - そして Ｅ２Ｅ Ｅｘ においては、Squared Aluminum Mesh, Bamboo, Card boardで有意差あり （ｐ ＝ ０．０２６、０．０２５、０．０２５）。 
    
- There was no significant difference between the average score of test and generated data in total.
    - テストの平均スコアと生成されたデータとの間には、有意な差はありませんでした。
    

# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


# 4.4 Discussion

- “Correct answer rate” of most texture conditions were almost 50%, thus participants could not distinguish test data from generated data. In the post questionnaire, all participants answered that they did not find the difference between the test stimulus and the generated stimulus. Therefore, it can be said that our system has the potential to generate the high realistic vibrotactile signals from given texture attributes or given texture images. “Correct answer rate” of Carpet and Fine Foam class in E2E Ex. were significantly lower than 50% so that participants tended to misunderstand the generated stimulus as an genuine stimulus for these two classes. On the contrary, there were no significant differences in the realism that participants felt between test and generated data in Carpet and Fine Foam classes. There was no correlation between the realism evaluation value and the discrimination rate of generated data.
    - ほとんどのテクスチャ条件の「正解率」はほぼ50％であったため、参加者はテストデータと生成データを区別できませんでした。 ポストアンケートでは、参加者全員がテスト刺激と生成された刺激の違いを見つけられなかったと回答しました。 したがって、我々のシステムは与えられたテクスチャ属性または与えられたテクスチャ画像から現実的な振動触覚信号を生成する可能性があると言える。 E2E Exのカーペットとファインフォームクラスの「正解率」 参加者はこれらの2つのクラスに対する本物の刺激として生成された刺激を誤解する傾向があったので、参加者は50％より有意に低かった。 それどころか、カーペットとファインフォームのクラスのテストデータと生成データの間で、参加者が感じたリアリズムに大きな違いはありませんでした。 リアリズム評価値と生成データの識別率との間に相関はなかった。


---

- Most scores about realism were over 60 and there was no significant difference between the scores of generated and test data in total. These results suggest that the generated vibrotactile stimuli had certain realism equivalent to the genuine stimuli. Focusing on the data for each class, the scores about realism were different between Generator Ex. and E2E Ex. This difference seems to be derived from the impression gap between texture attributes and images. Four out of ten participants answered in the post questionnaire that the impression of test image and label name were different, especially for Bamboo. Also, some participants said that it is difficult to imagine the texture surface from label name displayed in Generator Ex. These answers suggested that we should re-design the attribute axes, which we used class labels as they are in this study. For example, if we use some onomatopoeia as attribute axes, users can intuitively set and manipulate attributes and usability would be improved.
    - リアリズムに関するスコアの大部分は60を超えており、生成されたデータとテストデータのスコアの間に有意差はありませんでした。これらの結果は、生成された振動触覚刺激が本物の刺激と同等のあるリアリズムを有していたことを示唆している。各クラスのデータに焦点を当てると、リアリズムに関するスコアはGenerator Exによって異なりました。 Ｅ２Ｅ Ｅｘ。この違いは、テクスチャ属性とイメージの間の印象のギャップから派生するようです。 10人中4人の参加者がポストアンケートで、テスト画像とラベル名の印象は特にBambooに対して異なると回答しました。また、参加者の中には、Generator Exに表示されるラベル名からテクスチャサーフェスを想像するのは難しいと述べた。これらの回答から、属性ラベルを再設計する必要があることが示唆されました。この研究では、クラスラベルを使用しました。たとえば、属性軸としてオノマトペを使用すると、ユーザーは直感的に属性を設定および操作でき、使い勝手が向上します。

- Users rated generated data significantly higher than test data for Squared Aluminum Mesh in E2E Ex. Five participants answered that they felt periodic vibrotactile stimuli like moving on the mesh as the generated stimuli, so it can be considered that the trained model has enhanced the characteristic attribute like a mesh too much.
    - ユーザーは、E2E ExのSquared Aluminium Meshのテストデータよりもはるかに高い生成データを評価しました。 5人の参加者は、生成された刺激としてメッシュ上を動くような周期的な振動触覚刺激を感じたと回答したので、訓練されたモデルはメッシュのような特徴的な属性を強化しすぎたと考えることができます。
