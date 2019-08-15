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

## x. 論文の項目名


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## 2. Related Work

- xxx

---

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
