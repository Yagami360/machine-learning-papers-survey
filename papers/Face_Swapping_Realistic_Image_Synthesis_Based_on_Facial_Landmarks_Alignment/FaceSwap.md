> 論文まとめ：https://github.com/Yagami360/MachineLearning-Papers_Survey/issues/4

# ■ 論文
- 論文タイトル："Face Swapping: Realistic Image Synthesis Based on Facial Landmarks Alignment"
- 論文リンク：http://downloads.hindawi.com/journals/mpe/2019/8902701.pdf
- 論文投稿日付：2019/3/14
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- We propose an image-based face swapping algorithm, which can be used to replace the face in the reference image with the same facial shape and features as the input face. 
    - 画像ベースの顔スワッピングアルゴリズムを提案します。このアルゴリズムを使用して、参照画像の顔を入力顔と同じ顔の形と特徴に置き換えることができます。

- First, a face alignment is made based on a group of detected facial landmarks, so that the aligned input face and the reference face are consistent in size and posture. Secondly, an image warping algorithm based on triangulation is presented to adjust the reference face and its background according to the aligned input faces. 
    - まず、検出された顔のランドマークのグループに基づいて顔の位置合わせが行われ、位置合わせされた入力顔と基準顔のサイズと姿勢が一致します。次に、三角測量に基づく画像ワーピングアルゴリズムを提示して、整列した入力面に応じて参照面とその背景を調整します。

- In order to achieve more accurate face swapping, a face parsing algorithm is introduced to realize the accurate detection of the face-ROIs, and then the face-ROI in the reference image is replaced with the input face-ROI.
    - より正確な顔スワッピングを実現するために、顔解析アルゴリズムを導入して顔ROIの正確な検出を実現し、その後、参照画像内の顔ROIを入力顔ROIに置き換えます。

- Finally, a Poisson image editing algorithm is adopted to realize the boundary processing and color correction between the replacement region and the original background, and then the final face swapping result is obtained.  
    - 最後に、ポアソン画像編集アルゴリズムを採用して、置換領域と元の背景との間の境界処理と色補正を実現し、最終的な顔スワッピング結果を取得します。

- In the experiments, we compare our method with other face swapping algorithms and make a qualitative and quantitative analysis to evaluate the reality and the fidelity of the replaced face. The analysis results show that our method has some advantages in the overall performance of swapping effect.  
    - 実験では、この方法と他の顔交換アルゴリズムを比較し、定性的および定量的な分析を行って、置き換えられた顔の現実と忠実度を評価します。分析結果は、本手法がスワッピング効果の全体的なパフォーマンスにいくつかの利点があることを示しています。

# ■ イントロダクション（何をしたいか？）

## 1. Introduction

- Face synthesis refers to the image processing technology of the automatic fusion of two or more different faces into one face, which is widely used in fields of video synthesis, privacy protection, picture enhancement, and entertainment applications. For example, when we want to share some of the interesting things on social networks, we can use the face synthesis technique which can be regarded as a fusion of facial features and details to change our appearances appropriately without privacy leaks. As another type of face fusion, face swapping combines some parts of one person’s face with other parts of the other’s face to form a new face image. For instance, in the application of virtual hairstyle visualization, the client’s facial area can be fused with the hair areas of the model images to form new photos, so that customers can virtually browse their own figures with different hairstyles.
    - 顔合成とは、2つ以上の異なる顔を1つの顔に自動的に融合する画像処理技術を指します。これは、ビデオ合成、プライバシー保護、画像強調、およびエンターテイメントアプリケーションの分野で広く使用されています。たとえば、ソーシャルネットワーク上で興味深いものを共有したい場合、顔の特徴と詳細の融合と見なすことができる顔合成技術を使用して、プライバシーリークなしに外観を適切に変更できます。別のタイプの顔の融合として、顔の入れ替えは、ある人の顔の一部を他の人の顔の他の部分と組み合わせて、新しい顔画像を形成します。たとえば、仮想ヘアスタイルビジュアライゼーションのアプリケーションでは、クライアントの顔の領域をモデル画像のヘア領域と融合して新しい写真を作成できるため、顧客はさまざまなヘアスタイルで自分のフィギュアを仮想的に閲覧できます。

- This paper focuses on the face swapping problem of virtual browsing applications for hairstyle and dressing. 
    - このホワイトペーパーでは、ヘアスタイルとドレッシングのための仮想ブラウジングアプリケーションの顔交換問題に焦点を当てています。
    
- Our main contributions of the proposed algorithm include the following: (1) construct a pipeline of face swapping which integrates some learning-based modules into the traditional replacement- based approach, (2) improve the sense of reality and reliability of the synthesis face based on the precise detection of the facial landmarks, and (3) the face occlusion problem can be solved by introducing an accurate face parsing algorithm.
    - **提案されたアルゴリズムの主な貢献は次のとおりです。**
    - （1）いくつかの学習ベースのモジュールを従来の置換ベースのアプローチに統合する顔交換のパイプラインを構築します。
    -  **（2）顔のランドマークの正確な検出を元に合成顔の現実感と信頼性を改善します。**
    -  **（3）顔の閉塞問題は、正確な顔解析アルゴリズムを導入することで解決できます。**


# ■ 結論

## 5. Conclusion

- In this paper, a new face swapping algorithm based on facial landmarks detection is proposed, which can achieve fast, stable, and robust face replacement without the three- dimensional model. Our approach introduces the training results of existing learning models directly. The method we proposed does not require any training data, because it uses no learning model for new training.
    - **本論文では、顔のランドマーク検出に基づいた新しい顔交換アルゴリズムを提案します。これは、3次元モデルなしで高速、安定、堅牢な顔の置換を実現できます。** このアプローチでは、既存の学習モデルのトレーニング結果を直接紹介します。**提案した方法は、新しいトレーニングに学習モデルを使用しないため、トレーニングデータを必要としません。**

- The experimental results show that the composite image obtained by our model has a great reality and strong adaptability to the difference of skin color and hair occlusion while retaining most of the facial features of the input image. Compared with other algorithms, our model has some advantages in aspects of visual realism, time complexity, and data requirement.
    - 実験結果は、我々のモデルによって得られた合成画像は、入力画像の顔の特徴のほとんどを保持しながら、肌の色と髪の閉塞の違いに大きな現実性と強い適応性を持っていることを示しています。 他のアルゴリズムと比較して、このモデルには視覚的リアリズム、時間の複雑さ、データ要件の面でいくつかの利点があります。

- However, there is still room for further improvement, which mainly shows that the swapping result is not perfect when the input face and the reference face have a significant difference in perspective and posture. How to predict and generate the face image from a given perspective to another one is essential to solving the problem and also the main direction of our future research.
    - **ただし、入力面と参照面の視点と姿勢に大きな違いがある場合、主にスワッピングの結果が完全ではないことを主に示す、さらなる改善の余地があります。** 問題を解決し、今後の研究の主な方向性を解決するには、特定の視点から別の視点まで顔画像を予測および生成する方法が不可欠です。

# ■ 何をしたか？詳細

## 3. Method

- The algorithm is composed of three steps: face alignment, warping and replacement. The accuracy and robustness of the algorithm are enhanced by introducing some learning-based modules like facial landmark detection and face parsing.
    - このアルゴリズムは、顔の位置合わせ、ワープ、および置換の3つのステップで構成されています。 アルゴリズムの精度と堅牢性は、顔のランドマークの検出や顔の解析などの学習ベースのモジュールを導入することで強化されています。

### 3.1.PipelineofFaceSwapping.

![image](https://user-images.githubusercontent.com/25688193/63206266-64f36b80-c0ec-11e9-8d2a-e18d02e4d86b.png)

- Although face swapping seems uncomplicated and practicable, an elaborately designed algorithm flow still has an impact on the realization of the final result. The pipeline of the proposed algorithm starts with two channels that finally fuse into one, as shown in Figure 1.
- First, the input image is aligned with the reference image based on a facial landmark detection algorithm.
- Second, the reference image is warped to fit the aligned face of the input image. With an advanced face parsing algorithm, in the next step, the face-ROIs are extracted from the aligned input image and the warped reference image, respectively.
- Finally, some common steps of face replacement and color correction are introduced to generate the final composite face image.
- To summarize, the proposed algorithm will be demonstrated into three parts: face alignment, face warping, and face replacement.
    - 顔の交換は複雑で実用的ではないように見えますが、精巧に設計されたアルゴリズムフローは、最終結果の実現に依然として影響を及ぼします。 提案されたアルゴリズムのパイプラインは、図1に示すように、最終的に1つに融合する2つのチャネルから始まります。
    - 最初に、顔のランドマーク検出アルゴリズムに基づいて入力画像を参照画像に合わせます。 
    - 第二に、参照画像は入力画像の整列された顔に合うようにワープされます。 高度な顔解析アルゴリズムを使用すると、次のステップで、顔のROI [region of interest] が位置合わせされた入力画像とワープされた参照画像からそれぞれ抽出されます。
    - 最後に、最終的な合成顔画像を生成するために、顔の置換と色補正のいくつかの一般的な手順が導入されます。
    - 要約すると、提案されたアルゴリズムは、顔の位置合わせ、顔のゆがみ、顔の置換の3つの部分に分かれています。

### 3.2. Face Alignment.

- As the first step of face swapping, alignment refers to aligning the input face image 𝐼in and the reference face image 𝐼ref in size and direction. For the purpose of detecting faces in pictures, we apply the relevant methods in paper [14] which proposes a novel multiple sparse representation framework for visual tracking to detect the faces in pictures. Apart from increasing the speed of the algorithm, the application of sparse coding and dictionary learning also enables these methods to learn more knowledge from relatively fewer sample data. Then we extract several stable key points from the images to mark the faces, referred to as facial landmark detection (FLD for short).
    - 顔スワッピングの最初のステップとして、位置合わせは、入力顔画像𝐼inと参照顔画像𝐼refのサイズと方向の位置合わせを指します。 写真内の顔を検出するために、視覚追跡用の新しい複数のスパース表現フレームワークを提案する写真[14]の関連する方法を適用して、写真内の顔を検出します。 アルゴリズムの速度を上げるだけでなく、スパースコーディングと辞書学習を適用することにより、これらのメソッドは比較的少ないサンプルデータからより多くの知識を学習することもできます。 次に、顔のランドマーク検出（略してFLD）と呼ばれる、画像からいくつかの安定したキーポイントを抽出して顔をマークします。

- In this paper, we employ a popular FLD algorithm [9, 15] based on an ensemble of regression trees to detect facial landmarks Ωs = {𝑠𝑖 | 𝑖 = 1,2,...,68}, as plotted in Figure 2(a).
    - この論文では、回帰ツリーのアンサンブルに基づいた一般的なFLDアルゴリズム[9、15]を使用して、顔のランドマークを検出します。 𝑖= 1,2、...、68}、図2（a）にプロットされています。

- We use this method that relies on our implementation. Each landmark point 𝑠𝑖 is symmetric to another point 𝑠𝑖󸀠 with respect to the central axis of the face such as 𝑠22 and 𝑠23 , 𝑠49 , and 𝑠55 . The points located at the central axis are symmetric to themselves such as 𝑠28 and 𝑠31.

- To evaluate the rotation between the input and the reference face, the central axis of the input face should be extracted previously. According to the basic definition,


### 3.3.FaceWarping.

- Top reserve the shape of the swapped face, we warp the reference image to fit the aligned input face before face replacement. The warping is implemented based on the alignment of facial landmarks. We pick 18 out of the 68 facial landmarks and denote them with Φ𝑟 ={1,2,...,17,34} (see Figure 3(a)), which are considered to have a significant impact on facial shape.
    - 交換された顔の形状を確保し、顔を交換する前に参照画像をワープして整列した入力顔に合わせます。 ワーピングは、顔のランドマークの配置に基づいて実装されます。 68の顔のランドマークから18を選び、それらをΦ𝑟= {1,2、...、17,34}で示します（図3（a）を参照）。これらは顔の形に大きな影響があると考えられます。

- The landmarks of the reference face are denoted with 𝑟𝑖,𝑖 ∈ Φr. The new locations 𝑟w𝑖 of most of the landmarks (except 𝑟1,𝑟17, and 𝑟34) after the image warping should perfectly aligned to the input face, so we have

- Figure 3(b) illustrates the original landmarks (red points) and their new locations (green points). To realize the image warping, the reference image 𝐼ref is firstly decomposed into many triangle pieces based on the landmarks. The triangulation is required to minimize the change of the image background because we generally hope to preserve some parts in the background such as hair, body, and dress (that is why we do not move 𝑟1 , 𝑟17 , and 𝑟34 ). The final layout of triangulation is designed as shown in Figure 3(a). Then the image warping can be realized by applying the specific affine transformation to the corresponding triangle pieces.
    - 図3（b）は、元のランドマーク（赤い点）とその新しい位置（緑の点）を示しています。 画像のゆがみを実現するために、参照画像𝐼refは最初にランドマークに基づいて多くの三角形の断片に分解されます。 一般に、髪の毛、体、ドレスなどの背景の一部を保持したいので、画像の背景の変化を最小限に抑えるために三角測量 [triangulation] が必要です（𝑟1、𝑟17、およびmove34を動かさない理由です）。 三角形分割の最終レイアウトは、図3（a）に示すように設計されています。 次に、対応する三角形の部分に特定のアフィン変換を適用することにより、画像のワーピングを実現できます。

---

- xxx

### 3.4. Face Replacement.

- According to the pipeline of the proposed algorithm, we need to extract the input face-ROI 𝑅ain and the reference face-ROI 𝑅wref from the aligned input image 𝐼a and the warped reference image 𝐼w, respectively. A good in ref face-ROI should include as many facial features as possible while excluding background and distractors. However, most of traditional face swapping algorithms [2, 16] use a convex hull of the facial landmarks as the face-ROI which could cover some distracting areas like hair, hat, forehead, and neck. Therefore, in our model, a higher-precision face parsing algorithm based on deep learning [17] is introduced to extract a more accurate face-ROI. Different from the multiclass parsing network in paper [17], we use only two class labels, face and nonface, to train the parsing neural network based on the Helen dataset [18] which contains 2330 face images with pixel-level ground truth. Because our method requires face parsing rather than skin segmentation, so a contour detector is used. We initialize the encoder with pretrained VGG-16 net and the decoder with random values. During training, we fix the encoder parameters and only optimize the decoder parameters. The architecture of the network is shown in Figure 4. We set the learning rate to 0.0001 and train the network with 30 epochs with all the training images being processed each epoch. Then the face-ROIs can be generated by the retrained face parsing network.
    - 提案されたアルゴリズムのパイプラインによると、位置合わせされた入力画像𝐼aとワープされた参照画像facewからそれぞれ入力面ROI𝑅ainと基準顔ROI𝑅wrefを抽出する必要があります。 ref in ref-ROIには、背景とディストラクタを除外しながら、できるだけ多くの顔の特徴を含める必要があります。ただし、従来の顔交換アルゴリズムのほとんど[2、16]は、顔のランドマークの凸包を顔ROIとして使用し、髪、帽子、額、首などの気を散らす領域をカバーできます。
    - そのため、このモデルでは、より正確な顔ROIを抽出するために、ディープラーニング[17]に基づく高精度の顔解析アルゴリズムが導入されています。論文[17]のマルチクラス解析ネットワークとは異なり、ピクセルレベルのグラウンドトゥルースを持つ2330の顔画像を含むHelenデータセット[18]に基づいて解析ニューラルネットワークをトレーニングするために、顔と非顔の2つのクラスラベルのみを使用します。この方法では、肌のセグメンテーションではなく顔の解析が必要なため、輪郭検出器が使用されます。事前学習済みのVGG-16ネットでエンコーダーを初期化し、ランダム値でデコーダーを初期化します。トレーニング中に、エンコーダパラメータを修正し、デコーダパラメータのみを最適化します。ネットワークのアーキテクチャを図4に示します。学習率を0.0001に設定し、各エポックで処理されるすべてのトレーニングイメージで30エポックでネットワークをトレーニングします。次に、再訓練された顔解析ネットワークによって顔ROIを生成できます。


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## 4. Experiment

- In the experimental part, we will verify the superiority of our model by comparing it with three popular face swapping algorithms [2, 7, 16] and analyze the experimental results from qualitative and quantitative aspects to explain the effectiveness of our method.
    - 実験部分では、3つの一般的な顔交換アルゴリズム[2、7、16]と比較することでモデルの優位性を検証し、定性的および定量的側面から実験結果を分析して手法の有効性を説明します。

---

- In the qualitative experiment, we use the photos of several public figures as input image and reference image, respectively, as shown in Figure 7 [7]. The five reference faces have different postures, genders, skin tones, face shapes, and hairstyles, while the input faces include a male and a female face. The corresponding visual results obtained by the competing algorithms and our model are shown in Figure 8, where the five reference faces are replaced with two input faces, respectively. Figures 8(a) and 8(b), respectively, give the face swapping results for the male input face and female input face, where each column corresponds to a reference face and each row corresponds to a face swapping algorithm. On the whole, the face swapping results of the learning-based algorithm [7] (denoted as L1) is more natural and realistic and has a good adaptability to the different perspectives, hairstyles, and skin colors. However, as mentioned above, this method is only valid for the identities with a large number of training images, and each face identity corresponds to a generative neural network. In other words, this method cannot be applied to untrained new face images, which seriously restricts the application of this method. Our model promises the real sense of the face swapping result as well, and it is only slightly worse than the L-model in dealing with the perspective variance. In addition, this paper introduces the steps of the precise face parsing and the adaptive color correction, so it can effectively solve the problem of skin color difference and hair occlusion. While the other two replacement-based algorithms (denoted as R1 [16] and R2 [2]) have only adopted relatively simple algorithm flow, it is impossible to remove the boundary effects completely.
    - 定性実験では、図7 [7]に示すように、入力画像と参照画像として、それぞれいくつかの公人の写真を使用します。 5つの参照面には、異なる姿勢、性別、肌の色、顔の形、髪型がありますが、入力面には男性と女性の顔が含まれます。競合するアルゴリズムとモデルによって得られた対応する視覚的結果を図8に示します。5つの参照面がそれぞれ2つの入力面に置き換えられています。図8（a）と8（b）は、それぞれ、男性の入力面と女性の入力面の顔交換結果を示しています。各列は参照面に対応し、各行は顔交換アルゴリズムに対応しています。全体として、学習ベースのアルゴリズム[7]（L1と表記）の顔交換結果は、より自然で現実的であり、さまざまな視点、髪型、肌の色によく適応します。ただし、前述のように、この方法は多数のトレーニング画像を持つアイデンティティに対してのみ有効であり、各顔のアイデンティティは生成ニューラルネットワークに対応します。言い換えれば、この方法は、訓練されていない新しい顔の画像には適用できず、この方法の適用が大幅に制限されます。私たちのモデルは、顔交換結果の本当の意味も約束します。また、視点の分散を処理する際に、Lモデルよりわずかに悪いだけです。さらに、このホワイトペーパーでは、正確な顔の解析と適応色補正の手順を紹介しているため、肌の色の違いと髪の閉塞の問題を効果的に解決できます。他の2つの置換ベースのアルゴリズム（R1 [16]およびR2 [2]と表示）は比較的単純なアルゴリズムフローのみを採用していますが、境界効果を完全に除去することは不可能です。

---

- Many specific applications require that the facial features and face shape should remain as they should be in the face swapping, so a quantitative experiment is designed to verify the similarity between the face swapping result and the input face. Table 1 gives the similarity measures between the input face and the swapping results shown in Figure 8, which are obtained by a CNN-based model [21]. The higher scores in Table 1 represent the higher similarity. According to Table 1, our model is obviously better than the L-model and the R2- model while slightly lower than the R1-model in some cases. This is because that the L-model produces the composite face with a slight modification of the facial features to ensure the reality of the swapping results. The R2-model does not involve the color correction step and thus has a lower score. The R1-model has the highest similarity because it retains the complete face region at the expense of partially reducing the reality of the boundaries between forehead and hair. In contrast, our model preserves almost all of the facial features of the original input face while guaranteeing the reality of the swapped face, which leads to a better balance between similarity and realism.

# ■ 関連研究（他の手法との違い）

## x. Related Work


- xxx

---

- In the model-based approach [3], a two-dimensional or three-dimensional parametric feature model is established to represent human face, and the parameters and features are well-adjusted to the input image. Then the face reconstruction is performed on the reference image based on the result of adjusting the model parameters.
    - **モデルベースのアプローチ[3]では、2次元または3次元のパラメトリックフィーチャモデルが確立され、人間の顔が表現され、パラメータとフィーチャが入力画像に合わせて調整されます。 次に、モデルパラメータの調整結果に基づいて、参照画像で顔の再構成が実行されます。**

- An early work presented by Blanz and Volker et al [4] used a 3D model to estimate the face shape and posture, which improved the shortcoming of the unsatisfied performance of the synthesis due to the illumination and the perspective. However, the algorithm requires a 3D input model and a manual initialization to get a better result, which undoubtedly has a stricter requirement for data acquisition. Wang et al [5] proposed an algorithm based on active apparent model (AAM). By using the well trained AAM, the face swapping is realized in two steps: model fitting and component composite. But this method needs to specify the face-ROI manually and a certain number of face images for model training. Lin et al [6] presented a method of constructing a 3D model based on the frontal face image to deal with the different perspectives of reference image and input image. But the reconstructed model does not reflect the characteristics of the original face precisely and takes too much time to compute.
    - BlanzとVolker et al [4]が発表した初期の研究では、3Dモデルを使用して顔の形状と姿勢を推定しました。ただし、アルゴリズムでは、より良い結果を得るために3D入力モデルと手動の初期化が必要であり、間違いなくデータ取得の要件が厳しくなります。 Wang et al [5]は、アクティブ見かけモデル（AAM）に基づくアルゴリズムを提案しました。よく訓練されたAAMを使用することにより、顔の交換は2つのステップで実現されます：モデルの適合とコンポーネントの複合。ただし、この方法では、モデルのトレーニング用に顔のROIと特定の数の顔画像を手動で指定する必要があります。 Linら[6]は、正面画像に基づいて3Dモデルを構築し、参照画像と入力画像のさまざまな視点を処理する方法を提示しました。しかし、再構築されたモデルは元の顔の特徴を正確に反映せず、計算に時間がかかりすぎます。

---

- Above all, the replacement-based approach is simple and fast but sensitive to the variation in posture and perspective. The model-based method can effectively solve the perspective problem; however, it usually needs to collect three-dimensional face data, and robustness is not something to be satisfied. The learning-based approach can produce quite real and natural synthetic face image, while usually requiring a large number of training data and having more restrictions on the input and reference faces. Based on the comprehensive consideration of the characteristics of the above three methods, a face swapping algorithm supported by the facial landmark alignment is proposed under the replacement-based framework.
    - とりわけ、置換ベースのアプローチはシンプルかつ高速ですが、姿勢や視点 [perspective] の変化に敏感です。
    - **モデルベースの方法は、視点の問題を効果的に解決できます。** ただし、通常は3次元の顔データを収集する必要があり、堅牢性は満足すべきものではありません。
    - 学習ベースのアプローチでは、非常にリアルで自然な合成顔画像を生成できますが、通常、大量のトレーニングデータが必要であり、入力顔と参照顔により多くの制限があります。
    - 上記の3つの方法の特性の包括的な考慮に基づいて、顔のランドマーク調整によってサポートされる顔交換アルゴリズムが、置換ベースのフレームワークの下で提案されています。

- In addition, other widely used algorithms have been applied in our methods to achieve better results, such as facial landmark detection [9, 10], facial region segmentation [11, 12], and face warping [13]. And we will detail how these algorithms are applied in the method section.
    - さらに、顔のランドマーク検出[9、10]、顔領域のセグメンテーション[11、12]、顔のワーピング[13]など、他の広く使用されているアルゴリズムがより良い結果を得るために私たちの方法に適用されました。 そして、これらのアルゴリズムがどのように適用されるかについて、メソッドのセクションで詳しく説明します。

