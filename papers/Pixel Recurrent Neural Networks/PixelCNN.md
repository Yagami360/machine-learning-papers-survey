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


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. Related Work


