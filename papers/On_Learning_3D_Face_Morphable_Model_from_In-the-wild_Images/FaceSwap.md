# ■ 論文
- 論文タイトル："On Learning 3D Face Morphable Model from In-the-wild Images"
- 論文リンク：https://arxiv.org/abs/1808.09560
- 論文投稿日付：2018/08/28
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- As a classic statistical model of 3D facial shape and albedo, 3D Morphable Model (3DMM) is widely used in facial analysis, e.g, model fitting, image synthesis. Conventional 3DMM is learned from a set of 3D face scans with associated well-controlled 2D face images, and represented by two sets of PCA basis functions. Due to the type and amount of training data, as well as, the linear bases, the representation power of 3DMM can be limited. To address these problems, this paper proposes an innovative framework to learn a nonlinear 3DMM model from a large set of in-the-wild face images, without collecting 3D face scans. Specifically, given a face image as input, a network encoder estimates the projection, lighting, shape and albedo parameters. Two decoders serve as the nonlinear 3DMM to map from the shape and albedo parameters to the 3D shape and albedo, respectively. With the projection parameter, lighting, 3D shape, and albedo, a novel analytically-differentiable rendering layer is designed to reconstruct the original input face. The entire network is end-to-end trainable with only weak supervision. We demonstrate the superior representation power of our nonlinear 3DMM over its linear counterpart, and its contribution to face alignment, 3D reconstruction, and face editing.
    - 3Dの顔の形とアルベドの古典的な統計モデルとして、3D Morphable Model（3DMM）は、顔の分析、たとえばモデルフィッティング、画像合成で広く使用されています。従来の3DMMは、適切に制御された2D顔画像を伴う3D顔スキャンのセットから学習され、PCA基底関数の2つのセットで表されます。トレーニングデータの種類と量、および線形ベースにより、3DMMの表現力は制限される場合があります。これらの問題に対処するために、このペーパーでは、3D顔スキャンを収集することなく、野生の顔画像の大規模なセットから非線形3DMMモデルを学習する革新的なフレームワークを提案します。具体的には、入力として顔画像が与えられると、ネットワークエンコーダーは投影、照明、形状、およびアルベドパラメーターを推定します。 2つのデコーダーが非線形3DMMとして機能し、形状パラメーターとアルベドパラメーターから3D形状とアルベドにそれぞれマッピングされます。投影パラメータ、照明、3D形状、およびアルベドを使用して、元の入力面を再構築するために、分析的に微分可能な新しいレンダリングレイヤーが設計されています。ネットワーク全体はエンドツーエンドのトレーニングが可能であり、弱い監督しかありません。線形3DMMの優れた表現力と、その線形対応、および顔の位置合わせ、3D再構成、および顔編集への貢献を示します。

- Source code and additional results can be found at our project page: http://cvlab.cse.msu.edu/project-nonlinear-3dmm.html

# ■ イントロダクション（何をしたいか？）

## 1 INTRODUCTION

- THE 3D Morphable Model (3DMM) is a statistical model of 3D facial shape and texture in a space where there are explicit correspondences [1]. The morphable model framework provides two key benefits: first, a point-to-point correspondence between the reconstruction and all other models, enabling morphing, and second, modeling underlying transformations between types of faces (male to female, neutral to smile, etc.). 3DMM has been widely applied in numerous areas including, but not limited to, computer vision [1]–[3], computer graphics [4]–[7], human behavioral analysis [8], [9] and craniofacial surgery [10]. 
    - 3D Morphable Model（3DMM）は、明示的な対応がある空間での3D顔の形状とテクスチャの統計モデルです[1]。 モーフィング可能なモデルフレームワークには、2つの重要な利点があります：1つは、再構成とモーフィングを可能にする他のすべてのモデル間のポイントツーポイント対応、2つ目は、顔のタイプ（男性から女性、中立から笑顔など）の基本的な変換のモデリングです ）。 3DMMは、コンピュータービジョン[1] – [3]、コンピューターグラフィックス[4] – [7]、人間の行動分析[8]、[9]、頭蓋顔面外科[10]を含むがこれらに限定されない多くの分野で広く適用されています。 ]。
    
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


