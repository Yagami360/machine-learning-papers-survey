# ■ 論文
- 論文タイトル："Disentangled Person Image Generation"
- 論文リンク：https://arxiv.org/abs/1712.02621
- 論文投稿日付：
- 被引用数（記事作成時点）：xxx 件
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- Generating novel, yet realistic, images of persons is a challenging task due to the complex interplay between the different image factors, such as the foreground, background and pose information. In this work, we aim at generating such images based on a novel, two-stage reconstruction pipeline that learns a disentangled representation of the aforementioned image factors and generates novel person images at the same time.
    - 前景、背景、ポーズ情報などのさまざまな画像要素間の複雑な相互作用により、斬新でありながら現実的な人物の画像を生成することは困難な作業です。この作業では、前述のイメージファクターのもつれのない表現を学習し、同時に新しい人物の画像を生成する、新しい2段階の再構築パイプラインに基づいて、このような画像を生成することを目指しています。

- First, a multi-branched reconstruction network is proposed to disentangle and encode the three factors into embedding features, which are then combined to re-compose the input image itself. Second, three corresponding mapping functions are learned in an adversarial manner in order to map Gaussian noise to the learned embedding feature space, for each factor, respectively. Using the proposed framework, we can manipulate the foreground, background and pose of the input image, and also sample new embedding features to generate such targeted manipulations, that provide more control over the generation process. 
    - まず、多分岐再構成ネットワークを提案し、3つの要因を解きほぐして埋め込み特徴にエンコードし、次にそれらを組み合わせて入力画像自体を再構成します。第二に、3つの対応するマッピング関数が、それぞれの要因について、学習した埋め込み特徴空間にガウスノイズをマッピングするために、敵対的に学習されます。提案されたフレームワークを使用して、入力画像の前景、背景、ポーズを操作できます。また、新しい埋め込み機能のサンプルを作成して、ターゲットプロセスを生成し、生成プロセスをより詳細に制御できます。

- Experiments on the Market-1501 and Deep- fashion datasets show that our model does not only generate realistic person images with new foregrounds, backgrounds and poses, but also manipulates the generated factors and interpolates the in-between states. Another set of experiments on Market-1501 shows that our model can also be beneficial for the person re-identification task.
    - Market-1501およびDeep-fashionデータセットでの実験は、モデルが新しい前景、背景、およびポーズを持つ現実的な人物画像を生成するだけでなく、生成された要因を操作し、中間状態を補間することを示します。 Market-1501の別の実験セットは、このモデルが個人の再識別タスクにも有益であることを示しています。

# ■ イントロダクション（何をしたいか？）

## x. Introduction

- To this end, we disentangle the input image into intermediate embedding features, i.e. person images can be reduced to a composition of features of foreground, background, and pose. Compared to existing approaches, we rely on a different technique to generate new samples. In particular, we aim at sampling from a standard distribution, e.g. a Gaussian distribution, to first generate new embedding features and from them generate new images.
    - この目的のために、入力画像を解きほぐして中間の埋め込み機能を作成します。つまり、人物画像を前景、背景、およびポーズの機能の合成に縮小できます。 既存のアプローチと比較して、新しいサンプルを生成するために異なる手法に依存しています。 特に、標準的な分布（例えば、ガウス分布）からのサンプリングを目指しています。。最初に新しい埋め込み機能を生成し、そこから新しい画像を生成します。

- To achieve this, fake embedding features e ̃ are learned in an adversarial manner to match the distribution of the real embedding features e, where the encoded features from the input image are treated as real whilst the ones generated from the Gaussian noise as fake (Fig. 2).
    - これを達成するために、偽の埋め込み特徴e ̃は、実際の埋め込み特徴eの分布に一致する敵対的な方法で学習されます。入力画像からのエンコードされた特徴はリアルとして扱われ、ガウスノイズから生成された特徴は偽として扱われます（図 。2）。

- Consequently, the newly sampled images come from learned fake embedding features e ̃ rather than the original Gaussian noise as in the traditional GAN models. By doing so, the proposed technique enables us not only to sample a controllable input for the generator, but also to preserve the complexity of the composed images (i.e. realistic person images).
    - その結果、新しくサンプリングされた画像は、従来のGANモデルのような元のガウスノイズではなく、学習した偽の埋め込み機能e fromから取得されます。 そうすることで、提案された手法により、ジェネレーターの制御可能な入力をサンプリングできるだけでなく、合成画像（つまり、現実的な人物の画像）の複雑さを保持することもできます。

---

- To sum up, our full pipeline proceeds in two stages as shown in Fig. 2. At stage-I, we use a person’s image as input and disentangle the information into three main factors, namely foreground, background and pose. Each disentangled factor is modeled by embedding features through a reconstruction network. At stage-II, a mapping function is learned to map a Gaussian distribution to a feature embedding distribution.
    - 要約すると、図2に示すように、パイプライン全体が2段階に進みます。ステージIでは、人の画像を入力として使用し、情報を前景、背景、およびポーズの3つの主要な要因に分解します。 もつれ解除された各因子は、再構成ネットワークを介して特徴を埋め込むことによりモデル化されます。 ステージIIでは、マッピング関数を学習して、ガウス分布をフィーチャー埋め込み分布にマッピングします。

---

- 

Our contributions are: 

- 1) A new task of generating natural person images by disentangling the input into weakly correlated factors, namely foreground, background and pose. 
    - 入力を弱相関係数、つまり前景、背景、およびポーズに解きほぐすことにより、自然な人物の画像を生成する新しいタスク。

- 2) A two-stage framework to learn manipulatable embedding features for all three factors. In stage-I, the encoder of the multi-branched reconstruction network serves conditional image generation tasks, whereas in stage-II the mapping functions learned through adversarial training (i.e. mapping noise z to fake embedding features emb) serve sampling tasks (i.e. the input is sampled from a standard Gaussian distribution). 
    - 3つの要素すべての操作可能な埋め込み機能を学習する2段階のフレームワーク。 ステージIでは、多分岐再構成ネットワークのエンコーダーが条件付き画像生成タスクを処理しますが、ステージIIでは、敵対的トレーニングで学習したマッピング関数（つまり、ノイズzを偽埋め込み機能embにマッピングする）がサンプリングタスク（入力は標準のガウス分布からサンプリングされます）を処理します。

- 3) A technique to match the distribution of real and fake embedding features through adversarial training, not bound to the image generation task. 
    - 画像生成タスクに限らず、副次的トレーニングを通じて実物と偽物の埋め込み機能の分布を一致させる手法。

- 4) An approach to generate new image pairs for person re-ID. Sec. 4 constructs a Virtual Market re-ID dataset by fixing foreground features and changing background features and pose keypoints to generate samples of one identity.
    - 個人の再IDの新しい画像ペアを生成するアプローチ。 Sec 4は、前景の特徴を修正し、背景の特徴を変更し、キーポイントをポーズして1つのアイデンティティのサンプルを生成することにより、仮想市場のre-IDデータセットを構築します。


# ■ 結論

## x. Conclusion


# ■ 何をしたか？詳細

## x. 論文の項目名

- In stage-I, we disentangle the foreground, background and pose factors using a reconstruction network in a divide-and-conquer manner. In particular, we reconstruct person images by first disentangling into intermediate embedding features of the three factors, then recover the input image by decoding these features. In stage-II, we treat these features as real to learn mapping functions Φ for mapping a Gaussian distribution to the embedding feature distribution adversarially.

### 3.1. Stage-I: Disentangled image reconstruction

#### Foreground branch. 

- To separate the foreground and background information, we apply the coarse pose mask to the feature maps instead of the input image directly. By doing so, we can alleviate the inaccuracies of the coarse pose mask. Then, in order to further disentangle the foreground from the pose information, we encode pose invariant features with 7 Body Regions-Of-Interest instead of the whole image similar to [40]. Specifically, for each ROI we extract the feature maps resized to 48×48 and pass them into the weight sharing foreground encoder to increase the learning efficiency. Finally, the encoded 7 body ROI embedding features are concatenated into a 224D feature vector. Later, we use BodyROI7 to denote our model which uses 7 body ROIs to extract foreground embedding features, and use Whole-Body to denote our model that extracts foreground embedding features from the whole feature maps directly instead of extracting and resizing the ROI feature map.
    - 前景と背景の情報を分離するために、粗いポーズマスクを入力画像の代わりに特徴マップに直接適用します。 そうすることで、粗いポーズマスクの不正確さを軽減できます。 次に、ポーズ情報から前景をさらに解くために、[40]のような画像全体ではなく、7つの身体の関心領域でポーズ不変の特徴をエンコードします。 具体的には、各ROIについて、48 x 48にサイズ変更された特徴マップを抽出し、それらを重み共有フォアグラウンドエンコーダーに渡して、学習効率を高めます。 最後に、エンコードされた7体のROI埋め込み機能は、224D特徴ベクトルに連結されます。 後で、BodyROI7を使用して、7体のROIを使用して前景の埋め込み機能を抽出するモデルを示し、Whole-Bodyを使用して、ROI特徴マップを抽出およびサイズ変更する代わりに、機能マップ全体から前景の埋め込み機能を直接抽出するモデルを示します。

#### Pose branch.

- For the pose branch, we concatenate the 18- channel heatmaps with the appearance feature maps and pass them into the a “U-Net”-based architecture [29], i.e., convolutional autoencoder with skip connections, to generate the final person image following PG2 (G1+D) [21].
    - ポーズブランチでは、18チャンネルのヒートマップを外観の特徴マップと連結し、それらを「U-Net」ベースのアーキテクチャ[29]、つまりスキップ接続の畳み込みオートエンコーダに渡し、次の最終的な人物画像を生成します PG2（G1 + D）[21]。

- Here, the combination of appearance and pose imposes a strong explicit disentangling constraint that forces the network to learn how to use pose structure information to select the useful appearance information for each pixel. For pose sampling, we use an extra fully-connected network to reconstruct the pose information, so that we can decode the embedded pose features to obtain the heatmaps.
    - ここでは、外観とポーズの組み合わせにより、ネットワークがポーズ構造情報を使用して各ピクセルの有用な外観情報を選択する方法を学習することを強制する強い明示的な解きほぐし制約が課せられます。 ポーズサンプリングでは、追加の完全に接続されたネットワークを使用してポーズ情報を再構築するため、埋め込まれたポーズフィーチャをデコードしてヒートマップを取得できます。 

- Since some body regions may be unseen due to occlusions, we introduce a visibility variable αi ∈ {0, 1}, i = 1, ..., 18 to represent the visibility state of each pose keypoint. Now, the pose information can be represented by a 54-dim vector (36-dim keypoint coordinates γ and 18-dim keypoint visibility α).    
    - オクルージョンのために一部の身体領域が見えない可能性があるため、各ポーズキーポイントの可視性状態を表す可視性変数αi∈{0、1}、i = 1、...、18を導入します。 これで、ポーズ情報は54次元のベクトル（36次元のキーポイント座標γと18次元のキーポイントの可視性α）で表すことができます。


### 3.2. Stage-II: Embedding feature mapping

- Images can be represented by a low-dimensional, continuous feature embedding space. In particular, in [36, 30, 37, 5] it has been shown that they lie on or near a low-dimensional manifold of the original high-dimensional space. Therefore, the distribution of this feature embedding space should be more continuous and easier to learn compared to the real data distribution. Some works [38, 8, 28] have then attempted to use the intermediate feature representations of a pre-trained DNN to guide another DNN.
    - 画像は、空間を埋め込んだ低次元の連続フィーチャで表すことができます。 特に、[36、30、37、5]では、元の高次元空間の低次元多様体の上または近くにあることが示されています。 したがって、この埋めこみ特徴空間の分布は、実際のデータ分布と比較して、より連続的で学習しやすいものでなければなりません。 その後、いくつかの研究[38、8、28]は、事前に訓練されたDNNの中間特徴表現を使用して、別のDNNをガイドしようとしました。

- Inspired by these ideas, we propose a two-step mapping technique as illustrated in Fig. 2. Instead of directly learning to decode Gaussian noise to the image space, we first learn a mapping function Φ that maps a Gaussian space Z into a continuous feature embedding space E, and then use the pre-trained decoder to map the feature embedding space E into the real image space X. The encoder learned in stage-I encodes the FG, BG and Pose factors x into low- dimensional real embedding features e. Then, we treat the features mapped from Gaussian noise z as fake embedding features e ̃ and learn the mapping function Φ adversarially. In this way, we can sample fake embedding features from noise and then map them back to images using the decoder learned in stage-I. The proposed two-step mapping technique is easy to train in a piecewise style and most importantly can be useful for other image generation applications.
    - これらのアイデアに着想を得て、図2に示すような2段階のマッピング手法を提案します。ガウスノイズを画像空間に直接デコードするのではなく、ガウス空間Zを連続的な特徴にマッピングするマッピング関数Φを最初に学習します 段階Eで学習したエンコーダーは、FG、BG、およびポーズファクターxを低次元の実際の埋め込み機能eにエンコードします。 。 次に、ガウスノイズzからマッピングされた特徴を偽の埋め込み特徴e asとして扱い、マッピング関数Φを敵対的に学習します。 このようにして、偽の埋め込み機能をノイズからサンプリングし、ステージIで学習したデコーダーを使用してそれらを画像にマッピングし直すことができます。 提案された2ステップマッピング手法は、区分的なスタイルで簡単にトレーニングでき、最も重要なことは、他の画像生成アプリケーションに役立つ可能性があります。

### 3.3. Person image sampling

- As explained, each image factor can not only be encoded from the input information, but also be sampled from Gaussian noise. As to the latter, to sample a new foreground, background or pose, we combine the decoders learned in stage-I and mapping functions learned in stage-II to construct a z → e ̃ → x ̃ sampling pipeline (Fig. 4). Note that, for foreground and background sampling the decoder is a convolutional “U-net”-based architecture, while for pose sampling the decoder is a fully-connected one. Our experiments show that our framework performs well when used in both a conditional and an unconditional way.
    - 説明したように、各画像要素は入力情報からエンコードできるだけでなく、ガウスノイズからサンプリングすることもできます。 後者については、新しい前景、背景、またはポーズをサンプリングするために、ステージIで学習したデコーダーとステージIIで学習したマッピング関数を組み合わせて、z→e ̃→x ̃サンプリングパイプラインを構築します（図4）。 フォアグラウンドおよびバックグラウンドサンプリングの場合、デコーダーは畳み込みの「U-net」ベースのアーキテクチャであり、ポーズサンプリングの場合、デコーダーは完全に接続されたものであることに注意してください。 私たちの実験は、私たちのフレームワークが条件付きと無条件の両方の方法で使用されたときにうまく機能することを示しています。

### 

# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. Related Work


