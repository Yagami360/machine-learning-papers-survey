> 論文まとめ：https://github.com/Yagami360/MachineLearning-Papers_Survey/issues/3

# ■ 論文
- 論文タイトル："On Face Segmentation, Face Swapping, and Face Perception"
- 論文リンク：https://arxiv.org/abs/1704.06729
- 論文投稿日付：2017/04/22
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
    - セグメンテーションネットワーク（セクション3.2）をトレーニングするためのサンプルセットを充実させるために、3Dの顔の形状を明示的にモデル化します。
    - これらの3Dシェイプは、フェイスを交換するときに、あるフェイスから別のフェイスにテクスチャを転送するためのプロキシとしても使用されます（セクション3.3）。
    - これらの3D形状を取得する2つの代替方法を試しました。

---

- The first, inspired by [14] uses a generic 3D face, making no attempt to fit its shape to the face in the image aside from pose (viewpoint) alignment.
    - 最初のアプローチは、[14]に触発されて、一般的な3D顔を使用しており、ポーズ（視点）の位置合わせを除き、画像内の顔にその形状を合わせようとしません。

- We, however, also estimate facial expressions and modify the 3D face accordingly.
    - ただし、顔の表情も推定し、状況に応じて [accordingly]、3D顔を修正します。

---

- A second approach uses the recent state of the art, deep method for single image 3D face reconstruction [42].
    - 2番目のアプローチでは、最近の最先端の単一画像の3D顔再構成のためのディープメソッドを使用します[42]。

- It was shown to work well on unconstrained photos such as those considered here.
    - これは、ここで検討されているような、制約のない写真でうまく機能することが示されました。

- To our knowledge, this is the only method quantitatively shown to produce invariant, discriminative and accurate 3D shape estimations.
    - 私たちの知る限り、これは、不変、識別、および正確な3D形状推定を生成するために定量的に示されている唯一の方法です。

- The code they released regresses 3D Morphable face Models (3DMM) in neutral pose and expression.
    - 彼らがリリースしたコードは、3D Morphable Face Models（3DMM）をニュートラルなポーズと表情で退行させます [regresses]。

- We extend it by aligning 3D shapes with input photos and modifying the 3D faces to account for facial expressions.
    - 3Dシェイプを入力写真に整形し、顔の表情を考慮して [account for] 3D顔を修正することによって、それを拡張します。

#### 3D shape representation and estimation.

- Whether generic or regressed, we use the popular Basel Face Model (BFM) [36] to represent faces and the 3DDFA Morphable Model [47] for expressions.
    - 一般的であろうと退行性であろうと、人気のバーゼル顔モデル（BFM）[36]を使用して顔を表現し、3DDFA Morphable Model [47]を表現（＝表情？）に使用します。

- These are both publicly available 3DMM representations.
    - これらは両方とも公的に利用可能な3DMM表現です。

- More specifically, a 3D face shape V ⊂ R3 is modeled by combining the following independent generative models:
    - より具体的には、次の独立した生成モデルを組み合わせて、3Dの顔の形状V⊂R3をモデル化します。

![image](https://user-images.githubusercontent.com/25688193/63164830-a8a29280-c064-11e9-936b-44203ae9670c.png)

- Here, vector v^ is the mean face shape, computed over aligned facial 3D scans in the Basel Faces collection and represented by the concatenated 3D coordinates of their 3D points. When using a generic face shape, we use this average face. Matrices WS (shape) and WE (expression) are principle components obtained from the 3D face scans. Finally, α is a subject-specific 99D parameter vector estimated separately for each image and γ is a 29D parameter vector for expressions. To fit 3D shapes and expressions to an input image, we estimate these parameters along with camera matrices.
    - ここで、ベクトルv ^は、Basel Facesコレクションの位置合わせされた顔の3Dスキャンで計算され、3Dポイントの連結された3D座標で表される平均顔形状です。 一般的な顔の形状を使用する場合、この平均的な顔を使用します。 行列WS（形状）およびWE（式）は、3D顔スキャンから取得される主要なコンポーネントです。 最後に、αは画像ごとに個別に推定された被験者固有の99Dパラメーターベクトルであり、γは式の29Dパラメーターベクトルです。 3D形状と式を入力画像に合わせるために、これらのパラメーターとカメラ行列を推定します。

---

- To estimate per-subject 3D face shapes, we regress α using the deep network of [42]. They jointly estimate 198D parameters for face shape and texture. Dropping the texture components, we obtain α and back-project the regressed face by v^ + WS α, to get the estimated shape in 3D space.
    - 被験者ごとの3D顔の形状を推定するために、[42]の深いネットワークを使用してαを回帰します。 彼らは、顔の形と質感の198Dパラメータを共同で推定します。 テクスチャコンポーネントをドロップして、αを取得し、v ^ + WSαによって回帰面を逆投影して、3D空間で推定された形状を取得します。

#### Pose and expression fitting.

- Given a 3D face shape (generic or regressed) we recover its pose and adjust its expression to match the face in the input image. We use the detected facial landmarks, p = {pi} ⊂ R2, for both purposes. Specifically, we begin by solving for the pose, ignoring expression. We approximate the positions in 3D of the detected 2D facial landmarks V ̃ = {V ̃i} by:
    - 3Dの顔の形状（汎用または回帰）が与えられると、その姿勢を回復し、その表現を調整して、入力画像の顔に一致させます。 両方の目的のために、検出された顔のランドマークp = {pi}⊂R2を使用します。 具体的には、表情を無視してポーズを解くことから始めます。 検出された2D顔のランドマークの3Dでの位置を近似するV V = {V ̃i}：

![image](https://user-images.githubusercontent.com/25688193/63165483-8e69b400-c066-11e9-9b21-100603c98106.png)

- where f(·) is a function selecting the landmark vertices on the 3D model. The vertices of all BFM faces are registered so that the same vertex index corresponds to the same facial feature in all faces. Hence, f need only be manually specified once, at preprocessing. From f we get 2D-3D correspondences, pi ↔ V ̃ i , between detected facial features and their corresponding points on the 3D shape. Similarly to [13], we use these correspondences to estimate 3D pose, computing 3D face rotation, R ∈ R3, and translation vector t ∈ R3 using the EPnP solver [25].
    - ここで、f（・）は3Dモデルのランドマーク頂点を選択する関数です。 すべてのBFM顔の頂点は、同じ頂点インデックスがすべての顔の同じ顔の特徴に対応するように登録されます。 したがって、fは前処理で1回だけ手動で指定する必要があります。 fから、検出された顔の特徴と3D形状上の対応するポイントとの間の2D-3D対応、pi↔V ̃ iを取得します。 [13]と同様に、EPnPソルバー[25]を使用して、これらの対応関係を使用して3Dポーズを推定し、3D顔回転、R∈R3、および並進ベクトルt∈R3を計算します。

---

- xxx

### 3.2. Deep face segmentation

- Our method uses a FCN to segment the visible parts of faces from their context and occlusions. Other methods previously tailored novel network architectures for this task (e.g, [39]). We show that excellent segmentation results can be obtained with a standard FCN, provided that it is trained on plenty of rich and varied example.
    - このメソッドは、FCNを使用して、顔の可視部分をコンテキストとオクルージョンからセグメント化します。 以前は、このタスク用に新規のネットワークアーキテクチャを調整していた他の方法（例[39]） 豊富で多様な例が豊富に訓練されていれば、標準のFCNで優れたセグメンテーション結果が得られることを示しています。

---

- Obtaining enough diverse images with ground truth segmentation labels can be hard: [39], for example, used manually segmented LFW faces and the semi-automatic segmentations of [7]. These labels were costly to produce and limited in their variability and number. We obtain numerous training examples with little manual effort and show that a standard FCN trained on these examples outperforms state of the art face segmentation results.
    - グラウンドトゥルースセグメンテーションラベルを使用して十分に多様な画像を取得するのは困難な場合があります：[39]、たとえば、手動でセグメント化されたLFW顔と[7]の半自動セグメンテーションを使用します。 これらのラベルは生産コストが高く、変動性と数が限られていました。 手作業をほとんど必要とせずに多数のトレーニングサンプルを入手し、これらのサンプルでトレーニングされた標準のFCNが最新の顔セグメンテーション結果よりも優れていることを示します。
    
#### FCN architecture

- We used the FCN-8s-VGG architecture, fine-tuned for segmentation on PASCAL by [30]. Following [30], we fuse information at different locations from layers with different strides. We refer to [30] for more details on this.
    - [30]でPASCALのセグメンテーション用に微調整されたFCN-8s-VGGアーキテクチャを使用しました。 [30]に続いて、歩幅の異なるレイヤーの異なる場所で情報を融合します。 詳細については、[30]を参照してください。

#### Semi-supervised training data collection

- We produce large quantities of segmentation labeled face images by using motion cues in unconstrained face videos. To this end, we process videos from the recent IARPA Janus CS2 dataset [21]. These videos portray faces of different poses, ethnicities and ages, viewed under widely varying conditions. We used 1,275 videos of subjects not included in LFW, of the 2,042 CS2 videos (309 subjects out of 500).
    - **制約のない顔ビデオのモーションキューを使用して、顔画像とラベル付けされた大量のセグメンテーションを生成します。 この目的のために、最近のIARPA Janus CS2データセットからのビデオを処理します[21]。 これらのビデオでは、さまざまなポーズ、民族、年齢の顔を、さまざまな状況で視聴しています。 LFWに含まれていない2,275個のCS2ビデオのうち、1,275個のビデオ（500個中309個のサブジェクト）を使用しました。**

---

- Given a video, we produce a rough, initial segmentation using a method based on [12]. Specifically, we keep a hierarchy of regions with stable region boundaries computed with dense optical flow. Though these regions may be over- or under-segmented, they are computed with temporal coherence and so these segments are consistent across frames.
    - ビデオが与えられると、[12]に基づく方法を使用して、大まかな初期セグメンテーションを作成します。 具体的には、高密度のオプティカルフローで計算された安定した領域境界を持つ領域の階層を保持します。 これらの領域はオーバーまたはアンダーセグメント化されている場合がありますが、時間的一貫性を使用して計算されるため、これらのセグメントはフレーム全体で一貫しています。

---

- We use the method of [18] to detect faces and facial landmarks in each of the frames. Facial landmarks were then used to extract the face contour and extend it to include the forehead. All the segmented regions generated above, that did not overlap with a face contour are then discarded. All intersecting segmented regions are further processed using a simple interface which allows browsing the entire video, selecting the partial segments of [12] and adding or removing them from the face segmentation using simple mouse clicks. Fig. 3(a) shows the interface used in the semi-supervised labeling. A selected frame is typically processed in about five seconds. In total, we used this method to produce 9,818 segmented faces, choosing anywhere between one to five frames from each video, in a little over a day of work.
    - [18]の方法を使用して、各フレームで顔と顔のランドマークを検出します。 次に、顔のランドマークを使用して、顔の輪郭を抽出し、額を含むように拡張しました。 上記で生成され、顔の輪郭と重ならなかったすべてのセグメント化された領域は、その後破棄されます。 交差するすべてのセグメント化された領域は、ビデオ全体を閲覧し、[12]の部分セグメントを選択し、単純なマウスクリックを使用して顔のセグメンテーションに追加または削除できるシンプルなインターフェイスを使用してさらに処理されます。 図3（a）は、半教師付きラベル付けで使用されるインターフェイスを示しています。 通常、選択されたフレームは約5秒で処理されます。 合計で、この方法を使用して9,818個のセグメント化された顔を作成し、1日少しの作業で各ビデオから1〜5フレームを選択しました。

#### Occlusion augmentation

- This collection is further enriched by adding synthetic occlusions. To this end, we explicitly use 3D information estimated for our example faces. Specifically, we estimate 3D face shape for our segmented faces, using the method described in Sec. 3.1. We then use computer graphic (CG) 3D models of various objects (e.g., sunglasses) to modify the faces. We project these CG models onto the image and record their image locations as synthetic occlusions. Each CG object added 9,500 face examples. The detector used in our system [18] failed to accurately localize facial features on the remaining 318 faces, and so this augmentation was not applied to them.
    - このコレクションは、合成オクルージョンを追加することでさらに充実しています。 このため、サンプルの顔に対して推定された3D情報を明示的に使用します。 具体的には、セクション 3.1 で説明した方法を使用して、セグメント化された顔の3D顔の形状を推定します。- 次に、さまざまなオブジェクト（サングラスなど）のコンピューターグラフィック（CG）3Dモデルを使用して、顔を修正します。 これらのCGモデルを画像に投影し、画像の位置を合成オクルージョンとして記録します。 各CGオブジェクトには、9,500の顔の例が追加されました。 私たちのシステム[18]で使用されている検出器は、残りの318の顔の顔の特徴を正確に特定することができなかったため、この増強はそれらに適用されませんでした。

---

- Finally, an additional source of synthetic occlusions was supplied following [39] by overlaying hand images at various positions on our example images. Hand images were taken from the egohands dataset of [3]. Fig 3(b) shows a synthetic hand augmentation and Fig 3(c) a sunglasses augmentation, along with their resulting segmentation labels.
    - 最後に、合成オクルージョンの追加ソースが、サンプル画像のさまざまな位置に手の画像を重ねることによって[39]に続いて提供されました。 手の画像は、[3]のエゴハンドデータセットから取得されました。 図3（b）は、合成の手の増強を示し、図3（c）はサングラスの増強を、その結果のセグメンテーションラベルとともに示します。
    
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

