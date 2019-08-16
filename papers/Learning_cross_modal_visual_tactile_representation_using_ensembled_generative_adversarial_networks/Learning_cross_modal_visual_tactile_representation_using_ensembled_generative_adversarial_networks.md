# ■ 論文
- 論文タイトル："Learning cross-modal visual-tactile representation using ensembled generative adversarial networks"
- 論文リンク：https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8758548
- 論文投稿日付：2019/03/11
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- In this study, the authors study a deep learning model that can convert vision into tactile information, so that different texture images can be fed back to the tactile signal close to the real tactile sensation after training and learning. This study focuses on the classification of different image visual information and its corresponding tactile feedback output mode. A training model of ensembled generative adversarial networks is proposed, which has the characteristics of simple training and stable efficiency of the result. At the same time, compared with the previous methods of judging the tactile output, in addition to subjective human perception, this study also provides an objective and quantitative evaluation system to verify the performance of the model. The experimental results show that the learning model can transform the visual information of the image into the tactile information, which is close to the real tactile sensation, and also verify the scientificity of the tactile evaluation method.
    - この研究では、著者は視覚を触覚情報に変換できるディープラーニングモデルを研究しているため、トレーニングと学習後、実際の触覚に近い触覚信号にさまざまなテクスチャ画像をフィードバックできます。この研究では、さまざまな画像視覚情報の分類とそれに対応する触覚フィードバック出力モードに焦点を当てています。単純なトレーニングと結果の安定した効率の特性を持つ、アンサンブルされた生成的敵対ネットワークのトレーニングモデルを提案します。同時に、主観的な人間の知覚に加えて、触覚出力を判断する以前の方法と比較して、この研究は、モデルの性能を検証するための客観的かつ定量的な評価システムも提供します。実験結果は、学習モデルが画像の視覚情報を実際の触感に近い触覚情報に変換し、触覚評価法の科学性を検証できることを示しています。


# ■ イントロダクション（何をしたいか？）

## 1. Introduction

- With the rapid development of technology and success of artificial intelligence, object material recognition has been extensively used in many industrial fields, such as ecommerce, machinery manufacturing, and intelligent robots [1].
    - 技術の急速な発展と人工知能の成功により、eコマース、機械製造、インテリジェントロボットなど、多くの産業分野で物体材料の認識が広く使用されています[1]。

- At this stage, the distinguishing method of traditional object materials still stays in the visual difference discrimination.
    - この段階では、従来のオブジェクト素材の識別方法は、視覚的な違いの判別にとどまっています。

- However, due to the influence of environmental factors such as illumination and temperature, and the limitations of the imaging system hardware itself the imaging information of different object materials is not much different, which leads to the weakening of the distinguishability of the texture features and reduction of the robustness.
    - ただし、照明や温度などの環境要因の影響、およびイメージングシステムハードウェア自体の制限により、さまざまな物体材料のイメージング情報はそれほど変わらず、テクスチャ機能の識別性の低下と堅牢性の低減につながります。

- In addition, only the texture image of the object cannot accurately reflect the object properties associated with the material.
    - さらに、オブジェクトのテクスチャイメージのみでは、マテリアルに関連付けられたオブジェクトプロパティを正確に反映できません。

- For example, when we buy clothes online, we can only purchase them according to the picture information of the objects, and we cannot accurately feel the difference in the material of the clothes.
    - たとえば、オンラインで衣服を購入する場合、オブジェクトの画像情報に基づいてしか購入できず、衣服の素材の違いを正確に感じることはできません。

- Tactile sense is a way of identifying the material of objects in people's daily life. The information contained in it can make people feel the material properties of different objects more intuitively [2].
    - 触覚は、人々の日常生活における物体の素材を識別する方法です。 そこに含まれる情報により、人々はさまざまなオブジェクトの物質特性をより直感的に感じることができます[2]。

- Introducing touch into the field of object recognition is the future development trend, which can make people using more sensory information to judge and recognise the material properties of objects, which has important research significance.
    - 物体認識の分野へのタッチの導入は将来の開発動向であり、人々はより多くの感覚情報を使用して物体の物質特性を判断および認識することができます。これは重要な研究意義があります。
    
---

- Chang Liu et al proposed to improve the knowledge of human brain is the key to improve artificial intelligence, and multi-modal learning is the key [3]. This paper studies avisual-tactilecrossmodal learning method.
    - Chang Liuらは、人間の脳の知識を改善することが人工知能を改善する鍵であり、マルチモーダル学習が鍵であると提案しました[3]。 この論文は視覚-触覚クロスモーダル学習法を研究しています。

---

- For texture images of different objects, the haptic feedback signals under the corresponding tool interactions can make people better sense their tactile characteristics. Research on haptic modelling has been around for decades, and in haptic modelling it is usually the state of the tool (such as the speed of the tool), the state of the textured surface (such as the properties of the texture), and the output is the vibrational haptic signal. The introduction of tactile signals into object recognition can enrich the properties of the material of the object being sensed, and have a great value in various fields, such as medicine and engineering [3].
    - さまざまなオブジェクトのテクスチャ画像の場合、対応するツールの相互作用の下での触覚フィードバック信号により、人々は触覚特性をよりよく感知できます。 ハプティックモデリングの研究は何十年も前から行われており、ハプティックモデリングでは通常、ツールの状態（ツールの速度など）、テクスチャサーフェスの状態（テクスチャのプロパティなど）、および 出力は振動触覚信号です。 物体認識への触覚信号の導入は、感知されている物体の材料の特性を豊かにし、医学や工学などのさまざまな分野で大きな価値があります[3]。


# ■ 結論

## 5. Conclusion

- This work mainly studies the visual-tactile cross-modal transformation based on deep learning. First, the Resnet is used to obtain the classification information of the image. The next step is to use ensembled GANs to obtain the spectrogram generator G, and then combine the classification information of the resent output with G. It is possible to automatically generate spectrograms of different texture images, and then use Griffin–Lim algorithm to convert the spectrogram into a tactile signal. Finally, haptic signals of different texture images are fed back to the palm through the Bluetooth mouse. The experimental results show that the model can transform the visual information of the texture image into tactile information, which is close to the actual tactile sensation. At the same time, the experimental results also scientifically verify the tactile evaluation method in this study.
    - この研究では、主にディープラーニングに基づく視覚と触覚のクロスモーダル変換を研究しています。 最初に、Resnetを使用して画像の分類情報を取得します。 次のステップは、アンサンブルされたGANを使用してスペクトログラムジェネレーターGを取得し、再送出力の分類情報をGと組み合わせます。異なるテクスチャイメージのスペクトログラムを自動的に生成し、Griffin–Limアルゴリズムを使用して 触覚信号へのスペクトログラム。 最後に、さまざまなテクスチャ画像の触覚信号が、Bluetoothマウスを介して手のひらにフィードバックされます。 実験結果は、モデルがテクスチャ画像の視覚情報を実際の触感に近い触覚情報に変換できることを示しています。 同時に、実験結果は、この研究における触覚の評価方法も科学的に検証しています。

---

- Of course, the research work still has limitations. In the actual tactile test, the tactile feedback of the object material is constrained by the visual classification effect and the visual error will be continued into the tactile feedback. Despite this, this research has great application potential.
    - **もちろん、研究作業にはまだ制限があります。 実際の触覚テストでは、対象物の触覚フィードバックは視覚分類効果によって制約され、視覚エラーは触覚フィードバックに継続されます。 それにもかかわらず、この研究には大きな応用可能性があります。**


# ■ 何をしたか？詳細

## 4 Converting texture image into a haptic signal

### 4.1 Extraction of image features

- The network is mainly composed of two parts, discriminator D and a generator G. The input of the generator is composed of C and Z. C is the image vector of the label information obtained in the previous step (in Section 4.1), and Z is the random noise. In this study, the dimension of the Z is set to 100. The discriminator D first trains the spectrogram data set, establishes a judging index system, then takes the spectrogram obtained by the generator G as an input, and finally completes the discrimination and classification of the generated spectrogram by D. After 20 epochs of confrontation training, the result is compared with the original. Here, three categories are selected for comparison. As shown in Fig. 5, the results generated by the model are very similar to the original data. 
    - ネットワークは、主に2つの部分、弁別器DとジェネレータGで構成されます。ジェネレータの入力はCとZで構成されます。Cは、前のステップ（4.1節）で取得したラベル情報の画像ベクトルです。 はランダムノイズです。 この研究では、Zの次元は100に設定されます。弁別器Dは、最初にスペクトログラムデータセットをトレーニングし、判定インデックスシステムを確立し、次にジェネレータGによって取得されたスペクトログラムを入力として、最後に弁別と分類を完了します Dによって生成されたスペクトログラムの。20エポックの対立トレーニングの後、結果はオリジナルと比較されます。 ここでは、比較のために3つのカテゴリが選択されています。 図5に示すように、モデルによって生成された結果は、元のデータと非常に似ています。


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## 5 Result analysis

### 5.1 User test

- Tactile rendition devices are generally stimulating the tactile
receptors of the skin by various methods, such as air bellows or
nozzles, vibrations generated by electrical excitation, microneedle
arrays, direct current pulses, and functional neuromuscular
stimulators. The tactile output module of this study is an audio
vibrator, which belongs to the vibration device generated by
electric excitation. Its advantage is that the motor is used as an
actuator, which can obtain torque in any direction with relatively
fast response speed, and is very sensitive to different frequencies.
The device can be used to simulate the tool interaction state of the
original haptic signal acquisition, which is the main reason for
selecting this device as the haptic output. Its disadvantage is that
because of the excessive sensitivity to the signal, it may sometime
cause instability of the system, thereby damaging the effect of
force tactile reproduction. Other devices, such as DC pulses and
functional neuromuscular stimulators, are also suitable for this
study, but they have shortcomings such as insufficient stability and
high cost. Based on the above consideratio


### 5.2 Experimental data analysis

- In addition to the artificial subjective evaluation, this study also compares the data obtained from modelling the original tactile signals and the tactile signals generated by the model learning, and finally quantified and analysed the waveforms of the two as shown in Fig 9. 
    - **この研究では、人工的な主観的評価に加えて、元の触覚信号のモデル化から得られたデータとモデル学習によって生成された触覚信号を比較し、最終的に図9に示すように2つの波形を定量化および分析しました。**

---

- Specific calculation methods, such as metal net, are shown in Fig 10.
    - 金属ネットなどの特定の計算方法を図10に示します。

---

- The metal net 1 is a frequency sampling waveform of the original haptic signal, and the coordinates of the centre line are obtained as (x1i, y1i); the metal net 2 is a waveform diagram drawn by signal sampling generated by the model of this study, and the coordinates of its centreline are obtained as (x2i, y2i). Here, x is the time domain information, y is the frequency domain information, and x1i is equivalent to x2i. By comparing the frequency difference of each point at the same time domain, the overall similarity between the two can be obtained.
    - 金属ネット1は元の触覚信号の周波数サンプリング波形であり、中心線の座標は（x1i、y1i）として取得されます。 金属ネット2は、この研究のモデルによって生成された信号サンプリングによって描かれた波形図であり、その中心線の座標は（x2i、y2i）として取得されます。 ここで、xは時間領域情報、yは周波数領域情報、x1iはx2iと同等です。 同じ時間領域で各ポイントの周波数差を比較することにより、2つの間の全体的な類似性を取得できます。

---

- The index of the denominator in the above formula actually finds the Euclidean distance between the generated data waveform and the original data waveform at the same time domain point, so that the waveform difference between the two in the same time domain can be obtained. The denominator has a value range of [1, +∞], so the range of similarity S values is [0,1]. When the frequency difference between the two is larger, the closer S is to 0, the smaller the frequency difference between the two, the closer S is to 1. In this way, the similarity between the two waveforms can be judged, so that the quality of the generated haptic signal can be judged. Finally, the evaluation results of the nine experimental data sets of this study under this calculation method are shown in Fig 11. 
    - 上記の式の分母のインデックスは、同じ時間領域の2つの間の波形の差を取得できるように、同じ時間領域のポイントで生成されたデータ波形と元のデータ波形間のユークリッド距離を実際に見つけます。 分母の値の範囲は[1、+∞]であるため、類似度S値の範囲は[0,1]です。 2つの周波数差が大きい場合、Sが0に近いほど、2つの周波数差が小さくなり、Sは1に近くなります。このように、2つの波形の類似性を判断できます。 生成された触覚信号の品質を判断できます。 最後に、この計算方法によるこの研究の9つの実験データセットの評価結果を図11に示します。
    
# ■ 関連研究（他の手法との違い）

## 2. Related Work

- Currently, the perception of the tactile properties of objects depends mainly on the display of hardware. Komura et al. studied a tactile mouse that converts the design parameters of a virtual reality device into corresponding bump patterns for the palm of the user [4]. The principle is to connect the mouse to the dot matrix convex plate and move the mouse cursor on the image. While moving the mouse cursor towards the virtual object on the screen image, the corresponding stitch will be pushed out to stimulate the palm of the subject. The disadvantage is that the stimulating touch of the dot matrix raised plate feedback is limited, so it is only suitable for the recognition of distinct object image shapes. Strese et al developed a tactile mouse that can display the material properties of an object [5]. It mainly relies on the sensor hardware's judgment of the local data set and the real-time interaction is poor. The latest research in the modelling of haptic data focuses on the real-time interactivity of tool states. Previous research has mainly used the nature of vibration to describe the normal force and velocity of the recording tool [6, 7], which are encoded in autoregressive models. Their model successfully mapped the state and vibration mode of the tool. However, a single model generates a vibration signal that only supports a single texture used during training. Therefore, when attempting to generate vibration tactile of another texture, it is necessary to replace the model with another one. Obviously, the universality of the model is not sufficient.
    - 現在、オブジェクトの触覚特性の認識は、主にハードウェアの表示に依存しています。小村らは、仮想現実デバイスの設計パラメーターをユーザーの手のひらに対応するバンプパターンに変換する触覚マウスを研究しました[4]。原理は、マウスをドットマトリックス凸板に接続し、画像上でマウスカーソルを移動することです。画面イメージ上の仮想オブジェクトにマウスカーソルを移動させながら、対応するステッチを押し出して、対象の手のひらを刺激します。欠点は、ドットマトリックス隆起プレートフィードバックの刺激的なタッチが制限されるため、明確なオブジェクト画像形状の認識にのみ適していることです。 Streseらは、オブジェクトの材料特性を表示できる触覚マウスを開発しました[5]。これは主に、ローカルデータセットのセンサーハードウェアの判断に依存しており、リアルタイムの相互作用は不十分です。触覚データのモデリングに関する最新の研究は、ツールの状態のリアルタイムの対話性に焦点を当てています。以前の研究では、主に振動の性質を使用して、自己回帰モデルでエンコードされた記録ツールの法線力と速度を記述しました[6、7]。彼らのモデルは、ツールの状態と振動モードのマッピングに成功しました。ただし、単一のモデルは、トレーニング中に使用される単一のテクスチャのみをサポートする振動信号を生成します。したがって、別のテクスチャの振動触覚を生成しようとすると、モデルを別のモデルに置き換える必要があります。明らかに、モデルの普遍性は十分ではありません。

---

- Due to the complex mapping between the input and output of the tactile signal and the limitations of the level of the experimental tool, it is sometimes difficult to simultaneously take both the state of the tool and the state of the surface of the texture image as input. At present, there is no experimental model that can feedback its corresponding tactile signal very well. In the past two years, the use of generative adversarial networks (GANs) to generate new samples from high-dimensional data distribution (such as images) has been widely used, and is showing promising results in synthesising real-world images [8–10]. Previous studies have shown that GAN can efficiently generate images on labels [11], text [12], and so on.
    - 触覚信号の入力と出力の間の複雑なマッピングと実験ツールのレベルの制限により、ツールの状態とテクスチャ画像の表面の状態の両方を同時に入力することは困難な場合があります 。 現在、対応する触覚信号を非常によくフィードバックできる実験モデルはありません。 過去2年間、生成的敵対ネットワーク（GAN）を使用して高次元データ分布（画像など）から新しいサンプルを生成することが広く使用されており、実世界の画像の合成で有望な結果を示しています[8-10 ]。 以前の研究では、GANはラベル[11]、テキスト[12]などの画像を効率的に生成できることが示されています。

- [9] used GAN to simulate time series data distribution, which indirectly convert vibrational tactile signals into images, and finally generate vibrotactile signals dependent on texture images or texture properties. 
    - **[9]は、GANを使用して時系列データ分布をシミュレートし、振動触覚信号を間接的に画像に変換し、最終的にテクスチャ画像またはテクスチャプロパティに依存する振動触覚信号を生成します。**

- Although the model realises the tactile transformation of texture images, the evaluation of the tactile generation results of this model still remains on the artificial subjective perception discrimination, and there is no objective quantitative evaluation system, which makes the research results not a good criterion for judging.
    - **このモデルはテクスチャ画像の触覚変換を実現しますが、このモデルの触覚生成結果の評価は依然として人工主観的知覚識別に残っており、客観的な定量的評価システムは存在せず、研究結果を判断するための良い基準とはなりません。**

---

- Based on the deep learning, this paper takes the texture attribute of the image as input, and simulates the tactile signal of the image texture through training and learning, and realizes the cross-modal transformation of the texture image to the tactile signal.
    - ディープラーニングに基づいて、このホワイトペーパーでは、入力として画像のテクスチャ属性を取得し、トレーニングと学習を通じて画像テクスチャの触覚信号をシミュレートし、触覚信号へのテクスチャ画像のクロスモーダル変換を実現します。


- for the tactile signal generated by the model, in addition to the artificial subjective evaluation, this study also samples the vibration frequency of the tactile signal, draws the waveform diagram, and compares the drawn waveform diagram with the frequency sampling diagram of the original real tactile signal. 
    - **モデルによって生成された触覚信号について、人為的な主観的評価に加えて、この研究では触覚信号の振動周波数もサンプリングし、波形図を描画し、描画された波形図を元の実際の触覚信号の周波数サンプリング図と比較します。**

- The similarity of the waveform is invoked as a criterion for the final experimental results, thus providing a quantitative criterion.
    - 波形の類似性は、最終的な実験結果の基準として呼び出され、定量的な基準を提供します。
