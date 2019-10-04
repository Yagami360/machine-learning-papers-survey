# ■ 論文
- 論文タイトル："Pixel Recurrent Neural Networks"
- 論文リンク：https://arxiv.org/abs/1601.06759
- 論文投稿日付：2016/01/25
- 被引用数（記事作成時点）：xxx 件
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- Modeling the distribution of natural images is a landmark problem in unsupervised learning. This task requires an image model that is at once expressive, tractable and scalable. We present a deep neural network that sequentially predicts the pixels in an image along the two spatial dimensions. Our method models the discrete probability of the raw pixel values and encodes the complete set of dependencies in the image. Architectural novelties include fast two- dimensional recurrent layers and an effective use of residual connections in deep recurrent networks. We achieve log-likelihood scores on natural images that are considerably better than the previous state of the art. Our main results also provide benchmarks on the diverse ImageNet dataset. Samples generated from the model appear crisp, varied and globally coherent.
    - 自然画像の分布のモデル化は、教師なし学習の画期的な問題です。 このタスクには、表現力があり、扱いやすく、スケーラブルなイメージモデルが必要です。 2つの空間次元に沿って画像内のピクセルを順次予測するディープニューラルネットワークを提示します。 このメソッドは、生のピクセル値の離散確率をモデル化し、画像内の依存関係の完全なセットをエンコードします。 アーキテクチャの目新しさには、高速の2次元リカレントレイヤーと、ディープリカレントネットワークでの残留接続の効果的な使用が含まれます。 自然画像で対数尤度スコアを達成しますが、これは従来の最新技術よりもかなり優れています。 主な結果は、多様なImageNetデータセットのベンチマークも提供します。 モデルから生成されたサンプルは、鮮明で多様で、全体的に一貫しているように見えます。

# ■ イントロダクション（何をしたいか？）

## x. Introduction

- xxx

---

- One of the most important obstacles in generative modeling is building complex and expressive models that are also tractable and scalable. This trade-off has resulted in a large variety of generative models, each having their advantages. Most work focuses on stochastic latent variable models such as VAE’s (Rezende et al, 2014; Kingma & Welling, 2013) that aim to extract meaningful representations, but often come with an intractable inference step that can hinder their performance.
    - 生成モデリングの最も重要な障害の1つは、扱いやすく拡張可能な複雑で表現力豊かなモデルを構築することです。 このトレードオフにより、さまざまな生成モデルが生成され、それぞれに利点があります。 ほとんどの研究は、意味のある表現を抽出することを目的とするVAEのような確率的潜在変数モデル（Rezende et al、2014; Kingma＆Welling、2013）に焦点を当てていますが、多くの場合、パフォーマンスを妨げる可能性のある難解な推論ステップを伴います。

---

- One effective approach to tractably model a joint distribution of the pixels in the image is to cast it as a product of conditional distributions; this approach has been adopted in autoregressive models such as NADE (Larochelle & Murray, 2011) and fully visible neural networks (Neal, 1992; Bengio & Bengio, 2000). The factorization turns the joint modeling problem into a sequence problem, where one learns to predict the next pixel given all the previously generated pixels. But to model the highly nonlinear and long-range correlations between pixels and the complex conditional distributions that result, a highly expressive sequence model is necessary.
    - 画像内のピクセルの結合分布をうまくモデル化するための効果的なアプローチの1つは、条件付き分布の積としてキャストすることです。 このアプローチは、NADE（Larochelle＆Murray、2011）などの自己回帰モデルおよび完全に可視化されたニューラルネットワーク（Neal、1992、Bengio＆Bengio、2000）で採用されています。 因数分解により、ジョイントモデリング問題がシーケンス問題に変わり、以前に生成されたすべてのピクセルが与えられると、次のピクセルを予測することが学習されます。 しかし、ピクセルとその結果生じる複雑な条件付き分布との間の高度に非線形で長距離の相関をモデル化するには、高度に表現力豊かなシーケンスモデルが必要です。

---

- xxx

---

- In this paper we advance two-dimensional RNNs and apply them to large-scale modeling of natural images. The resulting PixelRNNs are composed of up to twelve, fast two-dimensional Long Short-Term Memory (LSTM) layers. These layers use LSTM units in their state (Hochreiter & Schmidhuber, 1997; Graves & Schmidhuber, 2009) and adopt a convolution to compute at once all the states along one of the spatial dimensions of the data. We design two types of these layers. The first type is the Row LSTM layer where the convolution is applied along each row; a similar technique is described in (Stollenga et al, 2015). The second type is the Diagonal BiLSTM layer where the convolution is applied in a novel fashion along the diagonals of the image. The networks also incorporate residual connections (He et al, 2015) around LSTM layers; we observe that this helps with training of the PixelRNN for up to twelve layers of depth.
    - この論文では、2次元RNNを発展させ、それらを自然画像の大規模モデリングに適用します。 生成されるPixelRNNは、最大12の高速2次元Long Short-Term Memory（LSTM）レイヤーで構成されます。 これらのレイヤーは、状態でLSTMユニットを使用し（Hochreiter＆Schmidhuber、1997; Graves＆Schmidhuber、2009）、畳み込みを採用して、データの空間次元の1つに沿ってすべての状態を一度に計算します。 これらのレイヤーの2つのタイプを設計します。 最初のタイプは、各行に沿って畳み込みが適用される行LSTMレイヤーです。 同様の手法が（Stollenga et al、2015）に記載されています。 2番目のタイプは、画像の対角線に沿って畳み込みが新しい方法で適用される対角BiLSTMレイヤーです。 ネットワークには、LSTMレイヤーの周りの残留接続（He et al、2015）も組み込まれています。 これにより、最大12層の深度でPixelRNNをトレーニングすることができます。

---

- We also consider a second, simplified architecture which shares the same core components as the PixelRNN. We observe that Convolutional Neural Networks (CNN) can also be used as sequence model with a fixed dependency range, by using Masked convolutions. The PixelCNN architecture is a fully convolutional network of fifteen layers that preserves the spatial resolution of its input throughout the layers and outputs a conditional distribution at each location.
    - また、PixelRNNと同じコアコンポーネントを共有する2番目の単純化されたアーキテクチャも検討します。 畳み込みニューラルネットワーク（CNN）は、マスクされた畳み込みを使用することにより、固定された依存範囲を持つシーケンスモデルとしても使用できることがわかります。 PixelCNNアーキテクチャは、15層の完全な畳み込みネットワークであり、層全体で入力の空間解像度を保持し、各場所で条件付き分布を出力します。

---

- Both PixelRNN and PixelCNN capture the full generality of pixel inter-dependencies without introducing independence assumptions as in e.g, latent variable models.
    - PixelRNNとPixelCNNの両方は、例えば潜在変数モデルのような独立性の仮定を導入することなく、ピクセルの相互依存性の完全な一般性をキャプチャします。

- The dependencies are also maintained between the RGB color values within each individual pixel. Furthermore, in contrast to previous approaches that model the pixels as continuous values (e.g., Theis & Bethge (2015); Gregor et al (2014)), we model the pixels as discrete values using a multinomial distribution implemented with a simple soft max layer. We observe that this approach gives both representational and training advantages for our models.
    - 依存関係は、個々のピクセル内のRGBカラー値間でも維持されます。 さらに、ピクセルを連続値としてモデル化する以前のアプローチ（例、Theis＆Bethge（2015）; Gregor et al（2014））とは対照的に、単純なソフト最大層で実装された多項分布を使用して、ピクセルを離散値としてモデル化します 。 このアプローチは、モデルに表現とトレーニングの両方の利点を与えることがわかります。

---

- The contributions of the paper are as follows. In Section 3 we design two types of PixelRNNs corresponding to the two types of LSTM layers; we describe the purely convolutional PixelCNN that is our fastest architecture; and we design a Multi-Scale version of the PixelRNN. In Section 5 we show the relative benefits of using the discrete softmax distribution in our models and of adopting residual connections for the LSTM layers. Next we test the models on MNIST and on CIFAR-10 and show that they obtain log- likelihood scores that are considerably better than previous results. We also provide results for the large-scale ImageNet dataset resized to both 32 × 32 and 64 × 64 pixels; to our knowledge likelihood values from generative models have not previously been reported on this dataset. Finally, we give a qualitative evaluation of the samples generated from the PixelRNNs.
    - 論文の貢献は次のとおりです。 セクション3では、2種類のLSTMレイヤーに対応する2種類のPixelRNNを設計します。 最速のアーキテクチャである純粋に畳み込みのPixelCNNについて説明します。 そして、PixelRNNのマルチスケールバージョンを設計します。 セクション5では、モデルで離散ソフトマックス分布を使用し、LSTMレイヤーに残留接続を採用する相対的な利点を示します。 次に、MNISTとCIFAR-10でモデルをテストし、以前の結果よりもかなり優れた対数尤度スコアを取得することを示します。 32×32ピクセルと64×64ピクセルの両方にサイズ変更された大規模なImageNetデータセットの結果も提供します。 生成モデルの知識尤度値は、このデータセットでは以前に報告されていません。 最後に、PixelRNNから生成されたサンプルの定性評価を行います。

# ■ 結論

## x. Conclusion


# ■ 何をしたか？詳細

## 2. Model

- Our aim is to estimate a distribution over natural images that can be used to tractably compute the likelihood of images and to generate new ones. The network scans the image one row at a time and one pixel at a time within each row. For each pixel it predicts the conditional distribution over the possible pixel values given the scanned context. Figure 2 illustrates this process. The joint distribution over the image pixels is factorized into a product of conditional distributions. The parameters used in the predictions are shared across all pixel positions in the image.
    - 私たちの目的は、画像の尤度を適切に計算し、新しい画像を生成するために使用できる自然画像の分布を推定することです。 ネットワークは、各行内で一度に1行、一度に1ピクセルずつ画像をスキャンします。 各ピクセルについて、スキャンされたコンテキストが与えられると、可能なピクセル値の条件付き分布を予測します。 図2は、このプロセスを示しています。 画像ピクセル全体の結合分布は、条件付き分布の積に因数分解されます。 予測で使用されるパラメーターは、画像内のすべてのピクセル位置で共有されます。

---

> 図

- > Figure 2. Left: To generate pixel xi one conditions on all the previously generated pixels left and above of xi. Center: To generate a pixel in the multi-scale case we can also condition on the subsampled image pixels (in light blue). Right: Diagram of the connectivity inside a masked convolution. In the first layer, each of the RGB channels is connected to previous channels and to the context, but is not connected to itself. In subsequent layers, the channels are also connected to themselves.
    - > 左：ピクセルxiを生成するには、以前に生成されたxiの左と上のすべてのピクセルに1つの条件を設定します。 中央：マルチスケールの場合にピクセルを生成するために、サブサンプリングされた画像ピクセル（水色）を条件にすることもできます。 右：マスクされた畳み込み内部の接続性の図。 最初のレイヤーでは、各RGBチャンネルは前のチャンネルとコンテキストに接続されていますが、それ自体には接続されていません。 後続の層では、チャネルもそれ自体に接続されます。

### 2.2. Pixels as Discrete Variables

- Previous approaches use a continuous distribution for the values of the pixels in the image (e.g. Theis & Bethge (2015); Uria et al. (2014)). By contrast we model p(x) as a discrete distribution, with every conditional distribution in Equation 2 being a multinomial that is modeled with a softmax layer. Each channel variable xi,∗ simply takes one of 256 distinct values. The discrete distribution is representationally simple and has the advantage of being arbitrarily multimodal without prior on the shape (see Fig. 6). Experimentally we also find the discrete distribution to be easy to learn and to produce better performance compared to a continuous distribution (Section 5).
    - 以前のアプローチでは、画像内のピクセル値の連続分布を使用します（例：Theis＆Bethge（2015）; Uria et al。（2014））。 対照的に、p（x）は離散分布としてモデル化されます。式2のすべての条件付き分布は、softmax層でモデル化された多項分布です。 各チャネル変数xi、∗は、単純に256個の異なる値のいずれかを取ります。 離散分布は、表現が単純であり、形状に事前依存することなく任意にマルチモーダルであるという利点があります（図6を参照）。 また、実験的に、離散分布は学習しやすく、連続分布と比較してパフォーマンスが向上することがわかります（セクション5）。

## 3. Pixel Recurrent Neural Networks

- In this section we describe the architectural components that compose the PixelRNN. In Sections 3.1 and 3.2, we describe the two types of LSTM layers that use convolutions to compute at once the states along one of the spatial dimensions. In Section 3.3 we describe how to incorporate residual connections to improve the training of a PixelRNN with many LSTM layers. In Section 3.4 we describe the softmax layer that computes the discrete joint distribution of the colors and the masking technique that ensures the proper conditioning scheme. In Section 3.5 we describe the PixelCNN architecture. Finally in Section 3.6 we describe the multi-scale architecture.

### 3.1. Row LSTM

- The Row LSTM is a unidirectional layer that processes the image row by row from top to bottom computing features for a whole row at once; the computation is performed with a one-dimensional convolution. For a pixel xi the layer captures a roughly triangular context above the pixel as shown in Figure 4 (center). The kernel of the one-dimensional convolution has size k × 1 where k ≥ 3; the larger the value of k the broader the context that is captured. The weight sharing in the convolution ensures translation invariance of the computed features along each row.
    - 行LSTMは、行全体で一度に行ごとに画像機能を処理する単方向レイヤーです。 計算は1次元の畳み込みで実行されます。 ピクセルxiの場合、図4（中央）に示すように、レイヤーはピクセルの上のほぼ三角形のコンテキストをキャプチャします。 1次元畳み込みのカーネルのサイズはk×1で、k≥3です。 kの値が大きいほど、キャプチャされるコンテキストが広くなります。 畳み込みの重み共有により、各行に沿って計算された特徴の変換不変性が保証されます。

---

- The computation proceeds as follows. An LSTM layer has an input-to-state component and a recurrent state-to-state component that together determine the four gates inside the LSTM core. To enhance parallelization in the Row LSTM the input-to-state component is first computed for the entire two-dimensional input map; for this a k × 1 convolution is used to follow the row-wise orientation of the LSTM itself. The convolution is masked to include only the valid context (see Section 3.4) and produces a tensor of size 4h × n × n, representing the four gate vectors for each position in the input map, where h is the number of output feature maps.
    - 計算は次のように進みます。 LSTMレイヤーには、LSTMコア内の4つのゲートを一緒に決定する、状態から入力への入力コンポーネントと状態から状態への繰り返しコンポーネントがあります。 Row LSTMの並列化を強化するために、最初に2次元入力マップ全体に対して入力から状態へのコンポーネントが計算されます。 このため、k×1の畳み込みを使用して、LSTM自体の行方向を追跡します。 畳み込みは有効なコンテキストのみを含むようにマスクされ（セクション3.4を参照）、サイズ4h×n×nのテンソルを生成します。これは、入力マップの各位置の4つのゲートベクトルを表します。hは出力フィーチャマップの数です。

---

- xxx

### 3.2. Diagonal BiLSTM

- The Diagonal BiLSTM is designed to both parallelize the computation and to capture the entire available context for any image size. Each of the two directions of the layer scans the image in a diagonal fashion starting from a corner at the top and reaching the opposite corner at the bottom. Each step in the computation computes at once the LSTM state along a diagonal in the image. Figure 4 (right) illustrates the computation and the resulting receptive field.
    - Diagonal BiLSTMは、計算の並列化と、あらゆる画像サイズの利用可能なコンテキスト全体のキャプチャの両方を行うように設計されています。 レイヤーの2つの方向のそれぞれは、上部のコーナーから始まり、下部の反対側のコーナーに到達する斜めの方法で画像をスキャンします。 計算の各ステップは、画像の対角線に沿ってLSTM状態を一度に計算します。 図4（右）は、計算と結果の受容野を示しています。

---

- The diagonal computation proceeds as follows. We first skew the input map into a space that makes it easy to apply convolutions along diagonals. The skewing operation offsets each row of the input map by one position with respect to the previous row, as illustrated in Figure 3; this results in a map of size n×(2n−1). At this point we can compute the input-to-state and state-to-state components of the Diagonal BiLSTM. For each of the two directions, the input-to-state component is simply a 1 × 1 convolution K is that contributes to the four gates in the LSTM core; the op- eration generates a 4h × n × n tensor. The state-to-state recurrent component is then computed with a column-wise convolution Kss that has a kernel of size 2 × 1. The step takes the previous hidden and cell states, combines the contribution of the input-to-state component and produces the next hidden and cell states, as defined in Equation 3. The output feature map is then skewed back into an n × n map by removing the offset positions. This computation is repeated for each of the two directions. Given the two output maps, to prevent the layer from seeing future pixels, the right output map is then shifted down by one row and added to the left output map.
    - 対角線の計算は次のように進みます。まず、入力マップを空間に傾けて、対角線に沿って畳み込みを簡単に適用できるようにします。スキュー操作は、図3に示すように、入力マップの各行を前の行に対して1つの位置だけオフセットします。これにより、サイズがn×（2n-1）のマップが作成されます。この時点で、対角BiLSTMの入力から状態および状態から状態のコンポーネントを計算できます。 2つの方向のそれぞれについて、状態への入力コンポーネントは、LSTMコアの4つのゲートに寄与する1×1コンボリューションKです。操作は4h×n×nテンソルを生成します。状態から状態への再帰成分は、サイズ2×1のカーネルを持つ列方向の畳み込みKssを使用して計算されます。このステップでは、以前の非表示状態とセル状態を取得し、状態から入力への成分の寄与を結合し、式3で定義されているように、次の非表示の状態とセルの状態を生成します。出力フィーチャマップは、オフセット位置を削除することにより、n×nマップにスキューバックされます。この計算は、2つの方向のそれぞれについて繰り返されます。 2つの出力マップがある場合、レイヤーが将来のピクセルを表示しないように、右の出力マップは1行下にシフトされ、左の出力マップに追加されます。

---

> 図３

- > Figure 3. In the Diagonal BiLSTM, to allow for parallelization along the diagonals, the input map is skewed by offseting each row by one position with respect to the previous row. When the spatial layer is computed left to right and column by column, the output map is shifted back into the original size. The convolution uses a kernel of size 2 × 1.
    - > 図3.対角線BiLSTMでは、対角線に沿った並列化を可能にするために、各行を前の行に対して1つの位置だけオフセットすることにより、入力マップがスキューされます。 空間レイヤーが左から右、列ごとに計算されると、出力マップは元のサイズに戻ります。 畳み込みでは、サイズ2×1のカーネルを使用します。

---

- Besides reaching the full dependency field, the Diagonal BiLSTM has the additional advantage that it uses a convolutional kernel of size 2 × 1 that processes a minimal amount of information at each step yielding a highly non-linear computation. Kernel sizes larger than 2 × 1 are not particularly useful as they do not broaden the already global receptive field of the Diagonal BiLSTM.
    - 完全な依存関係フィールドに到達することに加えて、対角BiLSTMには、各ステップで最小量の情報を処理するサイズ2×1の畳み込みカーネルを使用して、高度な非線形計算を行うという追加の利点があります。 2×1より大きいカーネルサイズは、Diagonal BiLSTMの既にグローバルな受容フィールドを広げないため、特に有用ではありません。

## 3.3. Residual Connections

- We train PixelRNNs of up to twelve layers of depth. As a means to both increase convergence speed and propagate signals more directly through the network, we deploy residual connections (He et al., 2015) from one LSTM layer to the next. Figure 5 shows a diagram of the residual blocks. The input map to the PixelRNN LSTM layer has 2h features. The input-to-state component reduces the number of features by producing h features per gate. After applying the recurrent layer, the output map is upsampled back to 2h features per position via a 1 × 1 convolution and the input map is added to the output map. This method is related to previous approaches that use gating along the depth of the recurrent network (Kalchbrenner et al., 2015; Zhang et al., 2016), but has the advantage of not requiring additional gates. Apart from residual connections, one can also use learnable skip connections from each layer to the output. In the experiments we evaluate the relative effectiveness of residual and layer-to-output skip connections.


## 3.4. Masked Convolution

- The h features for each input position at every layer in the network are split into three parts, each corresponding to one of the RGB channels. When predicting the R channel for the current pixel xi, only the generated pixels left and above of xi can be used as context. When predicting the G channel, the value of the R channel can also be used as context in addition to the previously generated pixels. Likewise, for the B channel, the values of both the R and G channels can be used. To restrict connections in the network to these dependencies, we apply a mask to the input- to-state convolutions and to other purely convolutional layers in a PixelRNN.
    - ネットワーク内のすべてのレイヤーの各入力位置のhフィーチャは3つの部分に分割され、それぞれがRGBチャンネルの1つに対応します。 現在のピクセルxiのRチャネルを予測する場合、xiの左と上に生成されたピクセルのみがコンテキストとして使用できます。 Gチャネルを予測するとき、以前に生成されたピクセルに加えて、Rチャネルの値もコンテキストとして使用できます。 同様に、Bチャネルでは、RチャネルとGチャネルの両方の値を使用できます。 ネットワーク内の接続をこれらの依存関係に制限するために、入力から状態への畳み込みおよびPixelRNNの他の純粋な畳み込み層にマスクを適用します。

---

- We use two types of masks that we indicate with mask A and mask B, as shown in Figure 2 (Right). Mask A is applied only to the first convolutional layer in a PixelRNN and restricts the connections to those neighboring pixels and to those colors in the current pixels that have already been predicted. On the other hand, mask B is applied to all the subsequent input-to-state convolutional transitions and relaxes the restrictions of mask A by also allowing the connection from a color to itself. The masks can be easily implemented by zeroing out the corresponding weights in the input-to-state convolutions after each update. Similar masks have also been used in variational autoencoders (Gregor et al., 2014; Germain et al., 2015).
    - 図2（右）に示すように、マスクAとマスクBで示す2種類のマスクを使用します。 マスクAは、PixelRNNの最初の畳み込み層にのみ適用され、それらの隣接ピクセルと、既に予測されている現在のピクセルの色への接続を制限します。 一方、マスクBは後続のすべての入力から状態への畳み込み遷移に適用され、色からそれ自体への接続も許可することでマスクAの制限を緩和します。 マスクは、各更新後に入力から状態への畳み込みの対応する重みをゼロにすることで簡単に実装できます。 同様のマスクは、変分オートエンコーダーでも使用されています（Gregor et al。、2014; Germain et al。、2015）。

## 3.5. PixelCNN

- The Row and Diagonal LSTM layers have a potentially unbounded dependency range within their receptive field. This comes with a computational cost as each state needs to be computed sequentially. One simple workaround is to make the receptive field large, but not unbounded. We can use standard convolutional layers to capture a bounded receptive field and compute features for all pixel positions at once. The PixelCNN uses multiple convolutional layers that preserve the spatial resolution; pooling layers are not used. Masks are adopted in the convolutions to avoid seeing the future context; masks have previously also been used in non-convolutional models such as MADE (Germain et al., 2015). Note that the advantage of parallelization of the PixelCNN over the PixelRNN is only available during training or during evaluating of test images. The image generation process is sequential for both kinds of networks, as each sampled pixel needs to be given as input back into the network.
    - RowおよびDiagonal LSTMレイヤーには、受容フィールド内に潜在的に無制限の依存範囲があります。 これには、各状態を順番に計算する必要があるため、計算コストが伴います。 1つの簡単な回避策は、受容フィールドを大きくすることですが、無制限ではありません。 標準の畳み込み層を使用して、境界のある受容野をキャプチャし、すべてのピクセル位置の特徴を一度に計算できます。 PixelCNNは、空間解像度を保持する複数の畳み込み層を使用します。 プール層は使用されません。 畳み込みではマスクが採用され、将来のコンテキストが表示されないようにします。 マスクは、MADEなどの非畳み込みモデルでも以前に使用されています（Germain et al。、2015）。 PixelRNNに対するPixelCNNの並列化の利点は、トレーニング中またはテスト画像の評価中にのみ利用可能であることに注意してください。 画像生成プロセスは、サンプリングされた各ピクセルをネットワークへの入力として返す必要があるため、両方の種類のネットワークで連続しています。



## 3.6. Multi-Scale PixelRNN

- The Multi-Scale PixelRNN is composed of an unconditional PixelRNN and one or more conditional PixelRNNs. The unconditional network first generates in the standard way a smaller s×s image that is subsampled from the original image. The conditional network then takes the s × s image as an additional input and generates a larger n × n image, as shown in Figure 2 (Middle).
    - マルチスケールPixelRNNは、無条件のPixelRNNと1つ以上の条件付きPixelRNNで構成されます。 無条件ネットワークは、最初に、元の画像からサブサンプリングされたより小さなs×s画像を標準的な方法で生成します。 図2（中央）に示すように、条件付きネットワークはs×sイメージを追加入力として受け取り、より大きなn×nイメージを生成します。

# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. Related Work


