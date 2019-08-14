# ■ 論文
- 論文タイトル："On Face Segmentation, Face Swapping, and Face Perception"
- 論文リンク：
- 論文投稿日付：
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- We show that even when face images are unconstrained and arbitrarily paired, face swapping between them is actually quite simple. 
    - 顔画像に制約がなく、任意にペアリングされている場合でも、それらの間の顔の交換は実際には非常に簡単であることを示します。

- To this end, we make the following contributions.
    - この目的のために、以下の貢献をします。

- (a) Instead of tailoring systems for face segmentation, as others previously proposed, we show that a standard fully convolutional network (FCN) can achieve remarkably fast and accurate segmentations, provided that it is trained on a rich enough example set.
    -（a）顔のセグメンテーション用にシステムを調整する（＝仕立てる）[tailoring] 代わりに、他の人が以前に提案したように、十分に豊富なサンプルセットで学習された、標準的な全畳み込みネットワーク（FCN）が、非常に高速で正確なセグメンテーションを達成できることを示します。

- For this purpose, we describe novel data collection and generation routines which provide challenging segmented face examples.
    - この目的のために、挑戦的なセグメント化された顔の例を提供する新しいデータ収集および生成ルーチンについて説明します。 

- (b) We use our segmentations to enable robust face swapping under unprecedented conditions.
    - （b）セグメンテーションを使用して、前例のない [unprecedented] 条件下で堅牢なフェイススワッピングを可能にします。

- (c) Unlike previous work, our swapping is robust enough to allow for extensive quantitative tests.
    - （c）以前の研究とは異なり、スワッピングは十分に堅牢であり、広範な定量的テストを可能にします。

- To this end, we use the Labeled Faces in the Wild (LFW) benchmark and measure the effect of intra- and inter-subject face swapping on recognition.
    - この目的のために、Labeled Faces in the Wild（LFW）ベンチマークを使用して、認識における [on recognition]、被験者内 [intra-subject] および被験者間 [inter-subject] の顔スワッピングの効果を測定します。

- We show that our intra-subject swapped faces remain as recognizable as their sources, testifying to the effectiveness of our method.
    - 被験者内でスワップされた顔は、そのソースと同じように認識可能なままであり、この方法の有効性を証明しています [testifying]。

- In line with well known perceptual studies, we show that better face swapping produces less recognizable inter-subject results (see, e.g, Fig 1). This is the first time this effect was quantitatively demonstrated for machine vision systems.
    - よく知られている知覚研究に沿って [In line with]、より良い顔すり替えが、認識可能な被験者間結果をより少なくすることを示します（例えば、図1を参照）。
    - コンピュータービジョンシステムでこの効果が定量的に実証されたのはこれが初めてです。

---

![image](https://user-images.githubusercontent.com/25688193/62994183-89a1d600-be95-11e9-9287-dced3c61c4eb.png)

- > Figure 1: Inter-subject swapping. LFW G.W. 

- > Bush photos swapped using our method onto very different subjects and images. Unlike previous work [4, 19], we do not select convenient targets for swapping. Is Bush hard to recognize? We offer quantitative evidence supporting Sinha and Poggio [40] showing that faces and context are both crucial for recognition.
    - > ブッシュの写真は、私たちの方法を使用して非常に異なる被写体と画像に交換しました。 以前の研究[4、19]とは異なり、スワッピングに便利なターゲットを選択しません。 ブッシュは認識しにくいですか？ シンハとポッジオ[40]を裏付ける定量的証拠を提供して、顔と文脈の両方が認識に不可欠であることを示します。

# ■ イントロダクション（何をしたいか？）

## 1. Introduction

- Swapping faces means transferring a face from a source photo onto a face appearing in a target photo, attempting to generate realistic, unedited looking results. Although face swapping today is often associated with viral Internet memes [10, 35], it is actually far more important than this practice may suggest: Face swapping can also be used for preserving privacy [4, 34, 38], digital forensics [35] and as a potential face specific data augmentation method [33] especially in applications where training data is scarce (e.g, facial emotion recognition [26]).
    - 顔の交換とは、ソース写真からターゲット写真に表示される顔に顔を転送し、現実的で未編集の結果を生成しようとすることです。
    - 今日の顔の入れ替えは、しばしばバイラル（＝ウイルスのように拡散する） [viral] インターネットミーム[10、35]に関連付けられていますが、実際には、この実行が示唆するよりもはるかに重要です：
    - 即ち、顔の入れ替えは、プライバシー[4、34、38]、デジタルフォレンジック[35 ]そして、特にトレーニングデータが不足しているアプリケーション（たとえば、顔の感情認識[26]）で、潜在的な顔固有のデータ増大方法[33]として使用されることも可能である。

---

- Going beyond particular applications, face swapping is also an excellent opportunity to develop and test essential face processing capabilities:  
    - 特定のアプリケーションを超えて、顔の交換は、重要な顔処理機能を開発およびテストする絶好の機会です：

- When faces are swapped between arbitrarily selected, unconstrained images, there is no guarantee on the similarity of viewpoints, expressions, 3D face shapes, genders or any other attribute that makes swapping easy [19].
    - 即ち、任意に選択された制約のない画像間で顔が交換された場合、視点、表情、3D顔の形、性別、または 交換を容易にするその他の属性[19]の類似性に関する保証はありません 

- In such cases, swapping requires robust and effective methods for face alignment, segmentation, 3D shape estimation (though we will later challenge this assertion), expression estimation and more.
    - そのような場合、スワッピングには、顔の位置合わせ、セグメンテーション、3D形状推定（後でこの主張に挑戦しますが）、表情推定などのための堅牢で効果的な方法が必要です。

---

- We describe a face swapping method and test it in settings where no control is assumed over the images or their pairings. We evaluate our method using extensive quantitative tests at a scale never before attempted by other face swapping methods. These tests allow us to measure the effect face swapping has on machine face recognition, providing insights from the perspectives of both security applications and face perception.
    - 顔の交換方法を説明し、画像またはそのペアリングを制御しないと仮定される設定でテストします。
    - 他の顔のスワッピング方法でこれまでに試みられたことのない規模で広範な定量的テストを使用して、この方法を評価します。 これらのテストにより、顔のスワッピングがマシンの顔認識に与える影響を測定でき、セキュリティアプリケーションと顔の認識の両方の観点から洞察を提供します。

---

- Technically, we focus on face segmentation and the design of a face swapping pipeline. Our contributions include:
    - 技術的には、顔のセグメンテーションと顔交換パイプラインの設計に焦点を当てています。 私たちの貢献は次のとおりです。

- Semi-supervised labeling of face segmentation. We show how a rich image set with face segmentation labels can be generated with little effort by using motion cues and 3D data augmentation. The data we collect is used to train a standard FCN to segment faces, surpassing previous results in both accuracy and speed.
    - 顔のセグメンテーションの半教師付きラベル付け。 モーションキューと3Dデータ拡張を使用することで、顔のセグメンテーションラベルを含むリッチな画像セットを簡単に生成できることを示します。 収集したデータは、顔をセグメント化するための標準FCNのトレーニングに使用され、精度と速度の両方で以前の結果を上回ります。

- Face swapping pipeline. We describe a pipeline for face swapping and show that our use of improved face segmentation and robust system components leads to high quality results even under challenging unconstrained conditions.
    - フェイススワッピングパイプライン。 フェイススワッピングのパイプラインについて説明し、改善されたフェイスセグメンテーションと堅牢なシステムコンポーネントを使用すると、厳しい制約のない条件下でも高品質の結果が得られることを示します。

- Quantitative tests. Despite over a decade of work and contrary to other face processing tasks (e.g, recognition), face swapping methods were never quantitatively tested. We design two test protocols based on the LFW benchmark [15] to test how intra- and inter- subject face swapping affects face verification.
    - 定量的テスト。 10年以上の作業にもかかわらず、他の顔処理タスク（認識など）に反して、顔の交換方法は定量的にテストされませんでした。 LFWベンチマーク[15]に基づいて2つのテストプロトコルを設計し、被験者内および被験者間の顔のスワッピングが顔の検証に与える影響をテストします。

---

- Our qualitative results show that our swapped faces are as compelling as those produced by others, if not more. Our quantitative tests further show that our intra-subject face swapping has little effect on face verification accuracy: our swapping does not introduce artifacts or otherwise changes these images in ways which affect subject identities.
    - 私たちの定性的な結果は、交換された顔は、少なくとも [if not more]、他の人が作成したものと同じくらい説得力のある [compelling] ものであることを示しています。
    - さらに、定量的テストでは、被験者内の顔の入れ替えが顔検証の精度にほとんど影響を与えないことが示されています。
    - 即ち、私たちのスワッピングは、アーチファクトを導入したり、被写体のアイデンティティに影響するような方法でこれらの画像を変更したりしません。

---

- We report inter-subject results on randomly selected pairs. These tests require facial appearance to change, sometimes substantially, in order to naturally blend source faces into their new surroundings. We show that this changes them, making them less recognizable. Though this perceptual phenomenon was described over two decades ago by Sinha and Poggio [40] in their well-known Clinton- Gore illusion, we are unaware of previous quantitative reports on how this applies to machine face recognition.
    - ランダムに選択されたペアに関する被験者間の結果を報告します。 これらのテストでは、ソースフェースを新しい環境に自然にブレンドするために、場合によっては大幅に顔の外観を変更する必要があります。
    - これによりそれらが変化し、認識しにくくなることがわかります。
    - この知覚現象は、20年以上前にSinhaとPoggio [40]によってよく知られているClinton-Gore錯視で説明されましたが、これがマシンの顔認識にどのように適用されるかについての以前の定量的レポートは認識していません。

- For code, deep models and more information, please see our project webpage.1


# ■ 結論

## 5. Conclusion

- We describe a novel, simple face swapping method which is robust enough to allow for large scale, quantitative testing. From these tests, several important observations emerge. (1) Face segmentation state of the art speed and accuracy, outperforming methods tailored for this task, can be obtained with a standard segmentation network, provided that the network is trained on rich and diverse examples. (2) Collecting such examples is easy. (3) Both faces and their contexts play important roles in recognition. We offer quantitative support for the two decades old claim of Sinha and Poggio [40]. (4) Better swapping, (e.g, to better mask facial spoofing attacks on biometric systems) leads to more facial changes and a drop in recognition. Finally, (5), 3D face shape estimation better blends the two faces together and so produces less recognizable source faces.
    - 大規模な定量的テストを可能にするのに十分な堅牢性を備えた、斬新でシンプルな顔スワッピング方法について説明します。 これらのテストから、いくつかの重要な観察結果が現れます。 （1）ネットワークが豊富で多様な例に基づいて訓練されている場合、標準のセグメンテーションネットワークを使用して、顔のセグメンテーションの最先端の速度と精度、このタスクに合わせたパフォーマンスの高い方法を取得できます。 （2）そのような例の収集は簡単です。 （3）両方の顔とそのコンテキストは、認識において重要な役割を果たします。 SinhaとPoggioの20年の主張に対する定量的なサポートを提供しています[40]。 （4）スワップの改善（たとえば、生体認証システムでの顔のなりすまし攻撃のマスクを改善する）により、顔の変化が増え、認識が低下します。 最後に、（5）、3Dの顔の形状の推定により、2つの顔がより良くブレンドされ、認識可能なソースの顔が少なくなります。


# ■ 何をしたか？詳細

## 3. Swapping faces in unconstrained images

![image](https://user-images.githubusercontent.com/25688193/62996727-9a0b7e00-bea0-11e9-9bea-523d67af930f.png)

- > Figure 2: Overview of our method. 

- > (a) Source (top) and target (bottom) input images. 

- > (b) Detected facial landmarks used to establish 3D pose and facial expression for a 3D face shape (Sec 3.1). We show the 3DMM regressed by [42] but our tests demonstrate that a generic shape often works equally well.
    - > (b) 3D顔の形状の3Dポーズと表情を確立するために使用される検出された顔のランドマーク（セクション3.1）。 [42]によって回帰された3DMMを示しますが、テストでは、一般的な形状がしばしば同等に機能することを示しています。

- > (c) Our face segmentation of Sec 3.2 (red) overlaid on the projected 3D face (gray).
    - > （c）セクション3.2（赤）の顔のセグメンテーションは、投影された3D顔（灰色）にオーバーレイされます。 

- > (d) Source transfered onto target without blending, and the final results.
    - > （d）ブレンドせずにソースをターゲットに転送し、最終結果。

- > (e) after blending (Sec 3.3).

---

- Fig 2 summarizes our face swapping method. When swapping a face from a source image, IS , to a target image, IT , we treat both images the same, apart from the final stage (Fig 2(d)). Our method first localizes 2D facial landmarks in each image (Fig 2(b)). We use an off-the-shelf detector for this purpose [18]. Using these landmarks, we compute 3D pose (viewpoint) and modify the 3D shape to account for expression. These steps are discussed in Sec 3.1.
    - 図2に、顔の交換方法をまとめます。 ソース画像ISからターゲット画像ITに顔を交換する場合、最終段階を除き、両方の画像を同じように扱います（図2（d））。 この方法では、まず各画像の2D顔のランドマークを特定します（図2（b））。 この目的のために、既製の検出器を使用します[18]。 これらのランドマークを使用して、3Dポーズ（視点）を計算し、表現を考慮して3D形状を変更します。 これらの手順については、セクション3.1で説明します。

---

- We then segment the faces from their backgrounds and occlusions (Fig 2(c)) using a FCN trained to predict per-pixel face visibility (Sec 3.2). 
    - 次に、ピクセルごとの顔の可視性を予測するようにトレーニングされたFCNを使用して、背景とオクルージョン（＝閉塞） [occlusions] から顔をセグメント化します（図3.2（c））。 

- We describe how we generate rich labeled data to effectively train our FCN.
    - FCNを効果的にトレーニングするために、豊富なラベル付きデータを生成する方法を説明します。 

- Finally, the source is efficiently warped onto the target using the two, aligned 3D face shapes as proxies, and blended onto the target image (Sec 3.3).
    - 最後に、ソースは、2つの整列した3D顔の形状をプロキシ（＝代理） [proxies] として使用して、ターゲットに効率的に歪まされ [warped]、ターゲットイメージにブレンドされます（セクション3.3）。

> これらの3Dシェイプは、フェイスを交換するときに、あるフェイスから別のフェイスにテクスチャを転送するためのプロキシとしても使用されます（セクション3.3）。 これらの3D形状を取得する2つの代替方法を試しました。


### 3.1. Fitting 3D face shapes

- To enrich our set of examples for training the segmentation network (Sec 3.2) we explicitly model 3D face shapes. These 3D shapes are also used as proxies to transfer textures from one face onto another, when swapping faces (Sec 3.3). We experimented with two alternative methods of obtaining these 3D shapes.
    - セグメンテーションネットワーク（セクション3.2）をトレーニングするためのサンプルセットを充実させるために、3Dの顔の形状を明示的にモデル化します。 これらの3Dシェイプは、フェイスを交換するときに、あるフェイスから別のフェイスにテクスチャを転送するためのプロキシとしても使用されます（セクション3.3）。 これらの3D形状を取得する2つの代替方法を試しました。

---

- The first, inspired by [14] uses a generic 3D face, making no attempt to fit its shape to the face in the image aside from pose (viewpoint) alignment. We, however, also estimate facial expressions and modify the 3D face accordingly.

---

- A second approach uses the recent state of the art, deep method for single image 3D face reconstruction [42]. It was shown to work well on unconstrained photos such as those considered here. To our knowledge, this is the only method quantitatively shown to produce invariant, discriminative and accurate 3D shape estimations. The code they released regresses 3D Morphable face Models (3DMM) in neutral pose and expression. We extend it by aligning 3D shapes with input photos and modifying the 3D faces to account for facial expressions.

#### 3D shape representation and estimation.

- xxx


#### Pose and expression fitting.

- xxx


### 3.2. Deep face segmentation

- xxx


### 3.3. Face swapping and blending

- xxx


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## 4. Experiments

- We performed comprehensive experiments in order to test our method, both qualitatively and quantitatively. Our face swapping method was implemented using MatConvNet [43] for segmentation, DLIB [20] for facial landmark detection and OpenCV [16] for all other image processing tasks. Runtimes were all measured on an Intel Core i7 4820K computer with 32GB DDR4 RAM and an NVIDIA GeForce Titan X. Using the GPU, our system swaps faces at 1.3 fps. On the CPU, this is slightly slower, performing at 0.8 fps. We emphasize again that unlike previous work our implementation will be public.
    - 方法を定性的および定量的にテストするために、包括的な実験を実施しました。 顔スワッピング方法は、セグメンテーションにMatConvNet [43]、顔のランドマーク検出にDLIB [20]、その他すべての画像処理タスクにOpenCV [16]を使用して実装されました。 ランタイムはすべて、32GB DDR4 RAMとNVIDIA GeForce Titan Xを搭載したIntel Core i7 4820Kコンピューターで測定されました。GPUを使用して、システムは1.3 fpsで顔を交換します。 CPUでは、これはわずかに遅く、0.8 fpsで実行されます。 以前の作業とは異なり、実装は公開されることを再度強調します。

### 4.1. Face segmentation results

- Qualitative face segmentation results are provided in Fig. 2 and 4, visualized following [39] to show segmented regions (red) overlaying the aligned 3D face shapes, projected onto the faces (gray).
    - 定性的な顔のセグメンテーションの結果を図2と4に示します。[39]に従って視覚化され、整列した3D顔の形状に重なるセグメント化された領域（赤）が顔に投影されます（灰色）。

---

- We provide also quantitative tests, comparing the accuracy of our segmentations to existing methods. We follow the evaluation procedure described by [11], testing the 507 face photos in the COFW dataset [6]. Previous methods included the regional predictive power (RPP) estimation [45], Structured Forest [17], segmentation-aware part model (SAPM) [11], the deep method of [29], and [39]. Note that Structured Forest [17] and [39] used respectively 300 and 437 images for testing, without reporting which images were used. Result for [29] was computed by us, using their code, out of the box, but optimizing for the segmentation threshold which provided the best accuracy.
    - また、セグメンテーションの精度を既存の方法と比較する定量テストも提供しています。 [11]で説明されている評価手順に従い、COFWデータセットの507枚の顔写真をテストします[6]。 以前の方法には、地域予測力（RPP）推定[45]、構造化フォレスト[17]、セグメンテーション対応パーツモデル（SAPM）[11]、[29]の深い方法、および[39]が含まれていました。 構造化フォレスト[17]と[39]は、使用した画像を報告せずに、それぞれ300と437の画像をテストに使用したことに注意してください。 [29]の結果は、コードを使用してすぐに計算されましたが、最高の精度を提供するセグメンテーションしきい値を最適化しました。

### 4.2. Qualitative face-swapping results

- We provide face swapping examples produced on unconstrained LFW images [15] using randomly selected targets in Fig 1, 2, and 5. We chose these examples to demonstrate a variety of challenging settings. In particular, these results used source and target faces of widely different poses, occlusions and facial expressions. To our knowledge, previous work never showed results for such challenging settings.
    - 図1、2、および5でランダムに選択されたターゲットを使用して、**制約のないLFW画像[15]**で作成された顔交換の例を提供します。 特に、これらの結果は、大きく異なるポーズ、オクルージョン、および表情のソースとターゲットの顔を使用しました。 私たちの知る限り、以前の研究では、このような難しい設定の結果は決して示されませんでした。

---

- In addition, Fig 6 shows a qualitative comparison with the very recent method [19] using the same source-target pairs. We note that [19] used the segmentation of [29] which we show in Sec 4.1 to perform worst than our own. This is qualitatively evident in Fig 6 by the face hairlines. Finally, Fig. 7 describes some typical failure cases and their causes.
    - さらに、図6は、同じソースとターゲットのペアを使用したごく最近の方法[19]との定性的な比較を示しています。 [19]はセクション4.1で示した[29]のセグメンテーションを使用して、私たち自身よりも最悪のパフォーマンスを発揮したことに注意してください。 これは、顔の生え際によって図6で定性的に明らかです。 最後に、図7にいくつかの典型的な故障事例とその原因を示します。

### 4.3. Quantitative tests

#### Inter-subject swapping verification protocols.

- We begin by measuring the effect of inter-subject face swapping on face verification accuracy. To this end, we process all faces in the LFW benchmark, swapping them onto photos of other, randomly selected subjects. We make no effort to verify the quality of the swapped results and if swapping failed (e.g, Fig 7), we treat the result as any other image.
    - 顔検証の精度に対する被験者間顔交換の影響を測定することから始めます。 この目的のために、LFWベンチマークのすべての顔を処理し、ランダムに選択された他の被写体の写真に置き換えます。 交換された結果の品質を確認する努力はせず、交換が失敗した場合（たとえば、図7）、結果を他の画像として扱います。



# ■ 関連研究（他の手法との違い）

## x. Related Work

- xxx

---

- Regardless of the application, previous face swapping systems often share several key aspects. First, some methods restrict the target photos used for transfer. Given an input source face, they search through large face albums to choose ones that are easy targets for face swapping [4, 8, 19]. Such targets are those which share similar appearance properties with the source, including facial tone, pose, expression and more. Though our method can be applied in similar settings, our tests focus on more extreme conditions, where the source and target images are arbitrarily selected and can be (often are) substantially different.
    - アプリケーションに関係なく、以前の顔交換システムは多くの場合、いくつかの重要な側面を共有しています。 まず、転送に使用するターゲット写真を制限する方法があります。 入力ソースの顔が与えられると、彼らは大きな顔のアルバムを検索して、顔の入れ替えの対象になりやすいものを選択します[4、8、19]。 そのようなターゲットは、顔のトーン、ポーズ、表情などを含む、ソースと同様の外観プロパティを共有するターゲットです。 この方法は同様の設定に適用できますが、テストでは、ソース画像とターゲット画像が任意に選択され、かなり異なる場合が多い（多くの場合）極端な条件に焦点を当てています。

