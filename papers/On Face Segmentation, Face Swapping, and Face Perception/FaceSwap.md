# ■ 論文
- 論文タイトル："On Face Segmentation, Face Swapping, and Face Perception"
- 論文リンク：
- 論文投稿日付：
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- We show that even when face images are unconstrained and arbitrarily paired, face swapping between them is actually quite simple. To this end, we make the following contributions. (a) Instead of tailoring systems for face segmentation, as others previously proposed, we show that a standard fully convolutional network (FCN) can achieve remarkably fast and accurate segmentations, provided that it is trained on a rich enough example set. For this purpose, we describe novel data collection and generation routines which provide challenging segmented face examples. (b) We use our segmentations to enable robust face swapping under unprecedented conditions. (c) Unlike previous work, our swapping is robust enough to allow for extensive quantitative tests. To this end, we use the Labeled Faces in the Wild (LFW) benchmark and measure the effect of intra- and inter-subject face swapping on recognition. We show that our intra-subject swapped faces remain as recognizable as their sources, testifying to the effectiveness of our method. In line with well known perceptual studies, we show that better face swapping produces less recognizable inter-subject results (see, e.g, Fig 1). This is the first time this effect was quantitatively demonstrated for machine vision systems.
    - 顔画像に制約がなく、任意にペアリングされている場合でも、それらの間の顔の交換は実際には非常に簡単であることを示します。この目的のために、以下の貢献をします。 （a）顔のセグメンテーション用にシステムを調整する代わりに、他の人が以前に提案したように、十分に豊富なサンプルセットでトレーニングされていれば、標準の完全たたみ込みネットワーク（FCN）が非常に高速で正確なセグメンテーションを達成できることを示します。この目的のために、挑戦的なセグメント化された顔の例を提供する新しいデータ収集および生成ルーチンについて説明します。 （b）セグメンテーションを使用して、前例のない条件下で堅牢なフェイススワッピングを可能にします。 （c）以前の研究とは異なり、スワッピングは十分に堅牢であり、広範な定量的テストを可能にします。この目的のために、Labeled Faces in the Wild（LFW）ベンチマークを使用して、認識における被験者内および被験者間の顔スワッピングの効果を測定します。被験者内でスワップされた顔は、そのソースと同じように認識可能なままであり、この方法の有効性を証明しています。よく知られている知覚研究に沿って、より良い顔のスワッピングが認識可能な被験者間結果をより少なくすることを示します（例えば、図1を参照）。マシンビジョンシステムでこの効果が定量的に実証されたのはこれが初めてです。

# ■ イントロダクション（何をしたいか？）

## x. Introduction

- 第１パラグラフ

---

- 第２パラグラフ

# ■ 結論

## x. Conclusion

- We describe a novel, simple face swapping method which is robust enough to allow for large scale, quantitative testing. From these tests, several important observations emerge. (1) Face segmentation state of the art speed and accuracy, outperforming methods tailored for this task, can be obtained with a standard segmentation network, provided that the network is trained on rich and diverse examples. (2) Collecting such examples is easy. (3) Both faces and their contexts play important roles in recognition. We offer quantitative support for the two decades old claim of Sinha and Poggio [40]. (4) Better swapping, (e.g, to better mask facial spoofing attacks on biometric systems) leads to more facial changes and a drop in recognition. Finally, (5), 3D face shape estimation better blends the two faces together and so produces less recognizable source faces.
    - 大規模な定量的テストを可能にするのに十分な堅牢性を備えた、斬新でシンプルな顔スワッピング方法について説明します。 これらのテストから、いくつかの重要な観察結果が現れます。 （1）ネットワークが豊富で多様な例に基づいて訓練されている場合、標準のセグメンテーションネットワークを使用して、顔のセグメンテーションの最先端の速度と精度、このタスクに合わせたパフォーマンスの高い方法を取得できます。 （2）そのような例の収集は簡単です。 （3）両方の顔とそのコンテキストは、認識において重要な役割を果たします。 SinhaとPoggioの20年の主張に対する定量的なサポートを提供しています[40]。 （4）スワップの改善（たとえば、生体認証システムでの顔のなりすまし攻撃のマスクを改善する）により、顔の変化が増え、認識が低下します。 最後に、（5）、3Dの顔の形状の推定により、2つの顔がより良くブレンドされ、認識可能なソースの顔が少なくなります。


# ■ 何をしたか？詳細

## x. 論文の項目名


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

