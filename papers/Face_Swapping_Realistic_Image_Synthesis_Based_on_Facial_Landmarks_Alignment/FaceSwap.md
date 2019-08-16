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

---

- 第２パラグラフ

# ■ 結論

## x. Conclusion


# ■ 何をしたか？詳細

## x. 論文の項目名


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. Related Work


- xxx

---

- In the model-based approach [3], a two-dimensional or three-dimensional parametric feature model is established to represent human face, and the parameters and features are well-adjusted to the input image. Then the face reconstruction is performed on the reference image based on the result of adjusting the model parameters.
    - **モデルベースのアプローチ[3]では、2次元または3次元のパラメトリックフィーチャモデルが確立され、人間の顔が表現され、パラメータとフィーチャが入力画像に合わせて調整されます。 次に、モデルパラメータの調整結果に基づいて、参照画像で顔の再構成が実行されます。**

- An early work presented by Blanz and Volker et al. [4] used a 3D model to estimate the face shape and posture, which improved the shortcoming of the unsatisfied performance of the synthesis due to the illumination and the perspective. However, the algorithm requires a 3D input model and a manual initialization to get a better result, which undoubtedly has a stricter requirement for data acquisition. Wang et al. [5] proposed an algorithm based on active apparent model (AAM). By using the well trained AAM, the face swapping is realized in two steps: model fitting and component composite. But this method needs to specify the face-ROI manually and a certain number of face images for model training. Lin et al. [6] presented a method of constructing a 3D model based on the frontal face image to deal with the different perspectives of reference image and input image. But the reconstructed model does not reflect the characteristics of the original face precisely and takes too much time to compute.


---

- Above all, the replacement-based approach is simple and fast but sensitive to the variation in posture and perspective. The model-based method can effectively solve the perspective problem; however, it usually needs to collect three-dimensional face data, and robustness is not something to be satisfied. The learning-based approach can produce quite real and natural synthetic face image, while usually requiring a large number of training data and having more restrictions on the input and reference faces. Based on the comprehensive consideration of the characteristics of the above three methods, a face swapping algorithm supported by the facial landmark alignment is proposed under the replacement-based framework.
    - とりわけ、置換ベースのアプローチはシンプルかつ高速ですが、姿勢や視点の変化に敏感です。 **モデルベースの方法は、視点の問題を効果的に解決できます。** ただし、通常は3次元の顔データを収集する必要があり、堅牢性は満足すべきものではありません。 学習ベースのアプローチでは、非常にリアルで自然な合成顔画像を生成できますが、通常、大量のトレーニングデータが必要であり、入力顔と参照顔により多くの制限があります。 上記の3つの方法の特性の包括的な考慮に基づいて、顔のランドマーク調整によってサポートされる顔交換アルゴリズムが、置換ベースのフレームワークの下で提案されています。