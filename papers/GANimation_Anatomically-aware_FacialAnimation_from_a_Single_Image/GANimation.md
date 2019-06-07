# ■ 論文
- 論文タイトル："GANimation: Anatomically-aware Facial Animation from a Single Image"
- 論文リンク：https://arxiv.org/abs/1807.09251
- 論文投稿日付：2018/07/24(v1), 
- 著者：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- Recent advances in Generative Adversarial Networks (GANs) have shown impressive results for task of facial expression synthesis.
    - GAN においての最近の進歩は、顔表現合成のタスクの印象的な結果に示した。

- The most successful architecture is StarGAN [4], that conditions GANs’ generation process with images of a specific domain, namely a set of images of persons sharing the same expression.
    - 最も成功したアーキテクチャはStarGAN [4]で、
    - （これは、）GANの生成プロセスを、特定のドメインの画像、即ち [namely]、同じ表現を共有する人物の画像のセットで条件付ける。 

- While effective, this approach can only generate a discrete number of expressions, determined by the content of the dataset.
    - 効果的ではありますが、このアプローチでは、データセットの内容によって決まる、離散的な [discrete] 表現の数しか生成できません。

- To address this limitation, in this paper, we introduce a novel GAN conditioning scheme based on Action Units (AU) annotations, which describes in a continuous manifold the anatomical facial movements defining a human expression.
    - この制限に対処するために、この論文では、Action Units（AU）アノテーションに基づく、新しいGAN条件付けスキーム（構想）[scheme] を紹介します。
    - （AUアノテーションというのは、）人間の表現を定義するような、解剖学的な顔の動きを連続した多様体で記述する（もの）

- Our approach allows controlling the magnitude of activation of each AU and combine several of them.
    - 我々のアプローチは、各ＡＵの活性化の大きさを制御し、それらのいくつかを組み合わせることを可能にする。

- Additionally, we propose a fully unsupervised strategy to train the model, that only requires images annotated with their activated AUs, and exploit attention mechanisms that make our network robust to changing backgrounds and lighting conditions.
    - さらに、それらの活性化されたAUでアノテーションされた（＝注釈が付けられた）画像のみを必要とするような、モデルを学習するための完全な教師なし戦略を提案し、
    - 変化する背景や照明条件に対して、ネットワークを堅牢にするアテンションメカニズムを利用します [exploit]。

- Extensive evaluation show that our approach goes beyond competing conditional generators both in the capability to synthesize a much wider range of expressions ruled by anatomically feasible muscle movements, as in the capacity of dealing with images in the wild.
    - 広範囲の評価は、
    - 野生の画像を扱う能力のように、
    - 解剖学的に [anatomically] 実現可能な [feasible] 筋肉の動きによってルール付けられる表現の範囲より、はるかに広い範囲の表現を合成する能力において、我々のアプローチが競合する条件付き生成器を超えることを示す。


# ■ イントロダクション（何をしたいか？）

## 1 Introduction

- Being able to automatically animate the facial expression from a single image would open the door to many new exciting applications in different areas, including the movie industry, photography technologies, fashion and e-commerce business, to name but a few. 
    - 1つの画像から自動的に表情をアニメートすることができれば、さまざまな分野における、多くの新しい魅力的な応用への扉が開かれるでしょう。
    - ほんの数例を挙げると [to name but a few]、映画業界、写真技術、ファッション、電子商取引など

- As Generative and Adversarial Networks have become more prevalent, this task has experienced significant advances, with architectures such as StarGAN [4], which is able not only to synthesize novel expressions, but also to change other attributes of the face, such as age, hair color or gender.
    - 生成的および敵対的ネットワークがより流行する [prevalent] につれて、
    - 新しい表現を合成するだけでなく、髪の色や性別、年齢のような顔の他の属性も変えることができる、StarGAN [4]のようなアーキテクチャと共に、
    - このタスクは、重要な進歩を経験した。

- Despite its generality, StarGAN can only change a particular aspect of a face among a discrete number of attributes defined by the annotation granularity of the dataset.
    - その一般性にもかかわらず、
    - StarGANは、データセットのアノテーション粒度 [granularity] によって定義された離散的な数の属性の中で、顔の特定の側面のみを変更できます。

- For instance, for the facial expression synthesis task, [4] is trained on the RaFD [16] dataset which has only 8 binary labels for facial expressions, namely sad, neutral, angry, contemptuous, disgusted, surprised, fearful and happy.
    - 例えば、表情合成タスクでは、
    - [4]は、RaFD [16]データセットについて訓練される。
    - （このデータセットというのは、）表情のための8つのバイナリラベル、すなわち、悲しい、中立、怒っている、軽蔑、うんざりする、驚いている、怖い、そして幸せな

---

- Facial expressions, however, are the result of the combined and coordinated action of facial muscles that cannot be categorized in a discrete and low number of classes.
    - しかしながら、顔の表情は顔の筋肉の複合的かつ協調的な [coordinated] 行動の結果であり、それは離散的で少数のクラスに分類することはできない。

- Ekman and Friesen [6] developed the Facial Action Coding System (FACS) for describing facial expressions in terms of the so-called Action Units (AUs), which are anatomically related to the contractions of specic facial muscles.
    - EkmanとFriesen [6]は、特定の顔面筋の収縮に解剖学的に関連する、いわゆるアクションユニット（AU）の観点から表情を記述するためのフェイシャルアクションコーディングシステム（FACS）を開発しました。

- Although the number of action units is relatively small (30 AUs were found to be anatomically related to the contraction of specic facial muscles), more than 7,000 diffierent AU combinations have been observed [30].
    - AC の数は比較的少ないが（30 AUが特定の顔面筋の収縮に解剖学的に関連していることがわかった）、
    - 7,000以上の異なるAUの組み合わせが観察されている[30]。

- For example, the facial expression for fear is generally produced with activations: Inner Brow Raiser (AU1), Outer Brow Raiser (AU2), Brow Lowerer (AU4), Upper Lid Raiser (AU5), Lid Tightener (AU7), Lip Stretcher (AU20) and Jaw Drop (AU26) [5].
    - 例えば、恐怖のための表情は一般に活性化で作り出されます：眉の内部ライザー（AU1）、外部眉ライザー（AU4）、アッパーリッドライザー（AU5）、リッドタイトナー（AU7）、唇トレッチャー（AU7） AU20）とJaw Drop（AU26）[5]。

- Depending on the magnitude of each AU, the expression will transmit the emotion of fear to a greater or lesser extent.
    - それぞれのAUの大きさに応じて、その表現は恐怖の感情を多かれ少なかれ伝えます。

---

- In this paper we aim at building a model for synthetic facial animation with the level of expressiveness of FACS, and being able to generate anatomically-aware expressions in a continuous domain, without the need of obtaining any facial landmarks [36].
    - 本論文では、FACSの表現力を用いて合成顔面アニメーションのモデルを構築し、顔面の目印 [landmarks] を取得することなく、連続領域で解剖学的に意識した表現を生成できることを目的とする[36]。

- For this purpose we leverage on the recent EmotioNet dataset [3], which consists of one million images of facial expressions (we use 200,000 of them) of emotion in the wild annotated with discrete AUs activations 1. 
    - この目的のために、私たちは最近のEmotioNetデータセット[3]を活用します。
    - これは、離散的なAU活性化で注釈が付けられた野生の感情の表情の100万画像（私たちはそのうち200,000を使用します）から成ります。

- We build a GAN architecture which, instead of being conditioned with images of a specic domain as in [4], it is conditioned on a one-dimensional vector indicating the presence/absence and the magnitude of each action unit.
    - [4]のように特定のドメインの画像で条件付けされるのではなく、
    - 各アクションユニットの有 [presence] / 無 [absence] と大きさを示す1次元ベクトルで条件付けされるGANアーキテクチャを構築します。

- We train this architecture in an unsupervised manner that only requires images with their activated AUs.
    - 我々はこのアーキテクチャを、活性化している AUs を持つ画像のみを要求している教師なし様式 [manner] で学習する。

- To circumvent the need for pairs of training images of the same person under diffierent expressions, we split the problem in two main stages.
    - 異なる表情の下で同一人物のトレーニング画像のペアの必要性を回避する[circumvent] ために、我々は2つの主な段階で問題を分割しました。

- First, we consider an AU-conditioned bidirectional adversarial architecture which, given a single training photo, initially renders a new image under the desired expression.
    - 最初に、我々は、単一のトレーニング写真が与えられると、
    - 最初に望ましい表現の下での新しい画像をレンダリングするような、AU 条件付き双方向敵対的アーキテクチャを検討する。

- This synthesized image is then rendered-back to the original pose, hence being directly comparable to the input image.
    - この合成画像は次に元の姿勢にレンダリングバックされるため、入力画像と直接比較できます。

- We incorporate very recent losses to assess the photorealism of the generated image.
    - 生成された画像のフォトリアリズム（＝写真のように描写する絵画のスタイル）[photorealism] を評価するために、ごく最近の損失を組み込んでいます。

- Additionally, our system also goes beyond state-of-the-art in that it can handle images under changing backgrounds and illumination conditions.
    - **加えて、私たちのシステムは、変化する背景や照明条件の下で画像を扱うことができるという点で、SOTA を超えています。**

- We achieve this by means of an attention layer that focuses the action of the network only in those regions of the image that are relevant to convey the novel expression.
    - 私たちは、新しい表現を伝える [convey] ことに関連している画像のそれらの領域だけに、ネットワークの行動を集中させるような Attention 層によって [by means of]、これを達成します。

> 画像の内、表情に関連する特定の部位のみに反応する注意機構で、変化する背景や照明条件の下で画像を扱うことができることを実現している。

---

![image](https://user-images.githubusercontent.com/25688193/58677674-e2ce9300-8397-11e9-9bd3-2dad17da986e.png)

---

- As a result, we build an anatomically coherent facial expression synthesis method, able to render images in a continuous domain, and which can handle images in the wild with complex backgrounds and illumination conditions.
    - その結果、解剖学的に首尾一貫した [coherent] 表情合成法を構築し、
    - （この表情合成法というのは、）連続領域で画像をレンダリングすることができ、
    - 複雑な背景や照明条件で野生の画像を処理することができます。

- As we will show in the results section, it compares favorably to other conditioned-GANs schemes, both in terms of the visual quality of the results, and the possibilities of generation. 
    - 結果のセクションで示すように、結果の視覚的品質と生成の可能性の両方の点で、他の条件付きGANスキームと比較して優れています。

- Figure 1 shows some example of the results we obtain, in which given one input image, we gradually change the magnitude of activation of the AUs used to produce a smile.


# ■ 結論

## 7. Conclusion

- We have presented a novel GAN model for face animation in the wild that can be trained in a fully unsupervised manner.

- It advances current works which, so far, had only addressed the problem for discrete emotions category editing and portrait images. 
    - それは、これまでのところ、離散感情カテゴリ編集および肖像画 [portrait] に関する問題にのみ対処していた現在の研究を進歩させる。

- Our model encodes anatomically consistent face deformations parameterized by means of AUs.

- Conditioning the GAN model on these AUs allows the generator to render a wide range of expressions by simple interpolation.

- Additionally, we embed an attention model within the network which allows focusing only on those regions of the image relevant for every specic expression.

- By doing this, we can easily process images in the wild, with distracting backgrounds and illumination artifacts.

- We have exhaustively evaluated the model capabilities and limits in the EmotioNet [3] and RaFD [16] datasets as well as in images from movies.

- The results are very promising, and show smooth transitions between diffierent expressions.

- This opens the possibility of applying our approach to video sequences, which we plan to do in the future.


# ■ 何をしたか？詳細

## 3 Problem Formulation

![image](https://user-images.githubusercontent.com/25688193/58678528-fa5b4b00-839a-11e9-82fd-6aa811e8b62f.png)

- > Fig. 2. Overview of our approach to generate photo-realistic conditioned images.

- > The proposed architecture consists of two main blocks: a generator G to regress attention and color masks; and a critic D to evaluate the generated image in its photorealism D_I and expression conditioning fullfilment ![image](https://user-images.githubusercontent.com/25688193/58687076-9006d280-83bb-11e9-934c-746053e30488.png).
    - > 提案されたアーキテクチャは、２つのメインブロックから構成される。
    - > Attention map とカラーマスクを逆行する [regress] ための生成器 G
    - > フォトリアリズム D_I と条件付された表現の実現？ [fullfilment]  ![image](https://user-images.githubusercontent.com/25688193/58687076-9006d280-83bb-11e9-934c-746053e30488.png) において、生成画像を評価するためのクリティック D

- > Note that our systems does not require supervision, i.e., no pairs of images of the same person with diffierent expressions, nor the target image I_{y_g} are assumed to be known.
    - > 我々のシステムは教師データを必要としないこと、すなわち異なる表情を有する同一人物の画像のペアも、目標画像 I_{y_g} も知られていると仮定していないことに留意されたい。

---

- Let us define an input RGB image as I_{y_r} ∈ R^{H×W×3}, captured under an arbitrary facial expression.
    - 任意の表情でキャプチャされた、入力RGB画像を ![image](https://user-images.githubusercontent.com/25688193/58678291-1d392f80-839a-11e9-9d29-6f8abe3a9b87.png) と定義する。

- Every gesture expression is encoded by means of a set of N action units y_r = (y_1, ... ,y_N)^T, where each y_n denotes a normalized value between 0 and 1 to module the magnitude of the n-th action unit.
    - 各ジェスチャ表現は、一組のＮ個の AU ![image](https://user-images.githubusercontent.com/25688193/58678436-b9633680-839a-11e9-9ff8-c39c338b4f3d.png) によって符号化される。
    - ここで各 y_n は、ｎ番目のAUの大きさを調整するための、０から１の間の正規化値を示している。

>  入力画像の他にAction-Unitの状態を記述するためのベクトル y を用意し、これを同時に入力として与える。

- It is worth pointing out that thanks to this continuous representation, a natural interpolation can be done between diffierent expressions, allowing to render a wide range of realistic and smooth facial expressions.
    - この（y による）連続的な表現のおかげで、
    - 異なる表現の間で自然な補間を行うことができ、
    - 広範囲の現実的で滑らかな表情をレンダリングすることが可能になりますということを指摘しておく価値がある。

---

- Our aim is to learn a mapping M to translate I_{y_r} into an output image I_{y_g} conditioned on an action-unit target y_g, i.e., we seek to estimate the mapping M : (I_{y_r} , y_g) → I_{y_g}.
    - 私たちの目的は、I_{y_r} を、AUs の目標値 y_g で条件付された出力画像 I_{y_g} に変換するための写像 ![image](https://user-images.githubusercontent.com/25688193/58681801-fa624780-83a8-11e9-91e2-3f1e3f572797.png) を学ぶことです。
    - すなわち、写像 ![image](https://user-images.githubusercontent.com/25688193/58681734-a6eff980-83a8-11e9-850b-a020148ecac8.png) を推定しようとします [seek to]。

- To this end, we propose to train M in an unsupervised manner, using M training triplets ![image](https://user-images.githubusercontent.com/25688193/58681807-136af880-83a9-11e9-8941-9d2ea3fa42c9.png).
    - この目的のために、M 個の学習する３つの組 ![image](https://user-images.githubusercontent.com/25688193/58681807-136af880-83a9-11e9-8941-9d2ea3fa42c9.png) を使用して、教師なしの形式において、写像 ![image](https://user-images.githubusercontent.com/25688193/58681801-fa624780-83a8-11e9-91e2-3f1e3f572797.png) を学習することを提案する。

- where the target vectors y_g are randomly generated.
    - ここで、目標ベクトル y_g は、ランダムに生成される。

- Importantly, we neither require pairs of images of the same person under diffierent expressions, nor the expected target image I_g.
    - 重要なことに、我々は、異なる表現の下での同一人物の画像のペアも、予想される目標画像 I_g も必要としない。

## 4 Our Approach

![image](https://user-images.githubusercontent.com/25688193/58678528-fa5b4b00-839a-11e9-82fd-6aa811e8b62f.png)

---

- This section describes our novel approach to generate photo-realistic conditioned images, which, as shown in Fig. 2, consists of two main modules.
    - このセクションでは、フォトリアリスティックな画像を生成するための、我々の新しいアプローチを記述する。
    - （これは、）図２に示されているような、２つのメインモジュールから構成される。

- On the one hand, a generator G(I_r | y_g) is trained to realistically transform the facial expression in image I_r to the desired y_g.
    - 一方では、生成器 G(I_r | y_g) は、画像 I_r を、望ましい（AU状態のベクトル） y_g に、リアルに変換することを学習する。

> 望ましい（AU状態のベクトル） y_g の条件下での、画像 I_r への変換を学習するでは？

- Note that G is applied twice, first to map the input image I_r → I_g , and then to render it back I_g → I^_{y_r}.
    - 生成器 G は、２回適用されることに注意。
    - 最初は、入力画像の写像 I_r → I_g。
    - 次に、I_g → I^_{y_r} へ戻すようなレンダリング。

- On the other hand, we use a WGAN-GP [9] based critic D(I_{y_g}) to evaluate the quality of the generated image as well as its expression.
    - 一方で、生成された画像の品質を評価するために、我々は、WGAN-GP ベースのクリティック D(I_{y_g}) を使用する。

### 4.1 Network Architecture

#### Generator.

- Let G be the generator block.

- Since it will be applied bidirectionally (i.e., to map either input image to desired expression and vice-versa) in the following discussion we use subscripts o and f to indicate origin and final.
    - 以下の説明では双方向に（すなわち、入力画像を所望の表現に写像するため、および、その逆も然り [vice-versa] の両方）適用されるので、
    - 原点および最終を示すために、下付き文字 [subscripts] o と f を使用する。

> ここでいう双方向 [bidirectionally] とは、生成器が、入力画像 → アニメーション画像 と アニメーション画像 → 元の画像の２つのプロセスで存在することを示している？

- Given the image ![image](https://user-images.githubusercontent.com/25688193/58684905-d7d62b80-83b4-11e9-8440-4ca261297f48.png) and the N-vector ![image](https://user-images.githubusercontent.com/25688193/58695722-0f52d100-83d1-11e9-8d0e-97fdc2da5fd4.png) encoding the desired expression, we form the input of generator as a concatenation ![image](https://user-images.githubusercontent.com/25688193/58685065-6a76ca80-83b5-11e9-9461-0aad699fd4bf.png), where ![image](https://user-images.githubusercontent.com/25688193/58695859-5b9e1100-83d1-11e9-80ed-90f4fb09b29c.png) has been represented as N arrays of size H × W.
    - 画像 ![image](https://user-images.githubusercontent.com/25688193/58684905-d7d62b80-83b4-11e9-8440-4ca261297f48.png) と望ましい表現でエンコードされた N ベクトル ![image](https://user-images.githubusercontent.com/25688193/58695722-0f52d100-83d1-11e9-8d0e-97fdc2da5fd4.png) を与えれば、
    - 生成器の入力を、連結 ![image](https://user-images.githubusercontent.com/25688193/58685065-6a76ca80-83b5-11e9-9461-0aad699fd4bf.png) として、形成する。
    - ここで、![image](https://user-images.githubusercontent.com/25688193/58695859-5b9e1100-83d1-11e9-80ed-90f4fb09b29c.png) は、サイズ H × W の N 個の配列として表現された。

---

- One key ingredient of our system is to make G focus only on those regions of the image that are responsible of synthesizing the novel expression and keep the rest elements of the image such as hair, glasses, hats or jewelery untouched.
    - 私たちのシステムの重要な成分 [ingredient] の1つは、
    - Gに新しい表現の合成する責任のある画像の領域のみに集中し、
    - 髪の毛、メガネ、帽子、宝石などの画像の残りの部分には、手に触れていないままにしておくことである。

> Attention のこと。

- For this purpose, we have embedded an attention mechanism to the generator.
    - この目的のために、我々は、生成器に attention メカニズムを組み込む。

- Concretely, instead of regressing a full image, our generator outputs two masks, a color mask C and attention mask A.
    - 具体的には、フル画像を逆行させるのではなく、カラーマスクCとアテンションマスクAの2つのマスクを出力します。

- The final image can be obtained as:

![image](https://user-images.githubusercontent.com/25688193/58685509-f0dfdc00-83b6-11e9-8beb-328fc19487df.png)

- where ![image](https://user-images.githubusercontent.com/25688193/58685587-3bf9ef00-83b7-11e9-8470-8e5f9ed7be04.png) and ![image](https://user-images.githubusercontent.com/25688193/58685783-cd696100-83b7-11e9-940b-3df18005fae6.png). 

- The mask A indicates to which extend each pixel of the C contributes to the output image I_f.
    - マスクＡは、Ｃの各ピクセルが出力画像 I_f に貢献している程度 [extend] を示す。

- In this way, the generator does not need to render static elements, and can focus exclusively on the pixels defining the facial movements, leading to sharper and more realistic synthetic images.
    - このようにして、ジェネレータは静的要素をレンダリングする必要がなく、顔の動きを定義するピクセルに、独占的に [exclusively] 焦点を合わせることができ、より鮮明でより現実的な合成画像をもたらす。

- This process is depicted in Fig. 3.

---

![image](https://user-images.githubusercontent.com/25688193/58685985-71eba300-83b8-11e9-926b-79500b977565.png)

- > Fig. 3. Attention-based generator.

- > Given an input image and the target expression, the generator regresses and attention mask A and an RGB color transformation C over the entire image.

- > The attention mask defines a per pixel intensity specifying to which extend each pixel of the original image will contribute in the final rendered image.
    - > Attention マスクは、元のイメージの各ピクセルが最終的にレンダリングされた画像に、どの程度寄与するかを特定する、ピクセル単位での明度（強度）[intensity] を定義します。

#### Conditional Critic.

- This is a network trained to evaluate the generated images in terms of their photo-realism and desired expression fullfilment. 
    - これは生成された画像を、フォトリアリズムと望ましい表現の実現 [fullfilment] という観点から、評価するために学習されたネットワークです。

> フォトリアリズム：アーキテクチャ図におけるクリティックの出力 D_I のこと

> 望ましい表現の実現 [fullfilment]：アーキテクチャ図におけるクリティックの出力 ![image](https://user-images.githubusercontent.com/25688193/58687076-9006d280-83bb-11e9-934c-746053e30488.png) のこと。これは、Action-Unitの状態 y_r を推定するために Discriminator の分枝で追加している y_r を評価するための機構

- The structure of D(I) resembles that of the PatchGan [10] network mapping from the input image I to a matrix ![image](https://user-images.githubusercontent.com/25688193/58686296-66e54280-83b9-11e9-9db8-2bbe14f6695f.png),
    - <font color="Pink">D(I) の構造は、入力画像 I から行列 ![image](https://user-images.githubusercontent.com/25688193/58686296-66e54280-83b9-11e9-9db8-2bbe14f6695f.png) へ写像するネットワークである、PatchGAN に似ている。</font>

- where ![image](https://user-images.githubusercontent.com/25688193/58686425-bc215400-83b9-11e9-8e79-8ca051ef4f2c.png) represents the probability of the overlapping patch i,j to be real.
    - <font color="Pink">ここで、![image](https://user-images.githubusercontent.com/25688193/58686425-bc215400-83b9-11e9-8e79-8ca051ef4f2c.png) は、重なっているパッチ i,j が真になる確率を表現している。</font>

- Also, to evaluate its conditioning, on top of it we add an auxiliary regression head that estimates the AUs activations ![image](https://user-images.githubusercontent.com/25688193/58686538-099dc100-83ba-11e9-9f80-8e862da660a6.png) in the image.
    - また、その条件付けを評価するために、
    - それに加えて [on top of it]、画像内の AUs の活性化 ![image](https://user-images.githubusercontent.com/25688193/58686538-099dc100-83ba-11e9-9f80-8e862da660a6.png) を推定するような、補助的な [auxiliary] 回帰ヘッドを追加します。

> 補助的な [auxiliary] 回帰ヘッド：これは、Action-Unitの状態 y_r を推定するために Discriminator の分枝で追加している y_r を評価するための機構 ![image](https://user-images.githubusercontent.com/25688193/58687076-9006d280-83bb-11e9-934c-746053e30488.png)

### 4.2 Learning the Model

- The loss function we define contains four terms, namely an image adversarial loss [1] with the modification proposed by Gulrajani et al. [9] that pushes the distribution of the generated images to the distribution of the training images;
    - 我々が定義する損失関数は、４つの項を含む。
    - 即ち [namely]、生成された画像の分布を、学習画像の分布にプッシュするようなといった、Gulrajani らによって提案された修正を持つ、image adversarial loss L_I

> image adversarial loss L_I : WGAN-GPで用いられる損失項。生成画像がデータセット内の画像と類似したものになるように働きかける。

- the attention loss to drive the attention masks to be smooth and prevent them from saturating;
    - attention マスクを、スムーズさせて、飽和 [saturating] からそれらを防ぐようにさせる [drive A to B] attention loss L_A。

> attention loss L_A : 隣り合う画素のAttentionが近くなるように、かつAttentionがかかり過ぎないようにする損失項。

- the conditional expression loss that conditions the expression of the generated images to be similar to the desired one;
    - 生成された画像の表現が、望ましいものと同じになるように条件付けされた、conditional expression loss L_y。

> conditional expression loss L_y : Action-Unitの状態yfで条件づけられて生成された画像がちゃんとyfという状態に従うようにはたらきかける損失項。

- and the identity loss that favors to preserve the person texture identity.
    - 人物テクスチャーの恒等性 [identity] を保存するために有利な、identity loss

> identity loss : CycleGANのように戻ってきた結果が同じ画像になるようにする損失項。要素ごとのL1距離で測る。

#### Image Adversarial Loss.

- In order to learn the parameters of the generator G, we use the modication of the standard GAN algorithm [8] proposed by WGAN-GP [9].
    - 生成器 G のパラメーターを学習するために、WGAN-GP によって、提案された標準的な GAN のアルゴリズムを修正したものを使用する。

- Specically, the original GAN formulation is based on the Jensen-Shannon (JS) divergence loss function and aims to maximize the probability of correctly classifying real and rendered images while the generator tries to foul the discriminator.
    - 特に、標準的な GAN の定式化は、JS ダイバージェンスの損失関数を元にしており、
    - 本物にレンダリングされた画像をに正しく分類する確率を最大化しようとする。
    - 一方で、生成器は、識別器をだまそうとする。

- This loss is potentially not continuous with respect to the generators parameters and can locally saturate leading to vanishing gradients in the discriminator.
    - この損失関数は、生成器のパラメーターに関して、潜在的に連続ではなく、
    - 識別器において、局所的に飽和して [saturate]、勾配消失を導くことが出来る。

- This is addressed in WGAN [1] by replacing JS with the continuous Earth Mover Distance.
    - これは、WGAN において、JS を連続な EM距離に置き換えることによって、取り組まれた。

- To maintain a Lipschitz constraint, WGAN-GP [9] proposes to add a gradient penalty for the critic network computed as the norm of the gradients with respect to the critic input.
    - リプシッツ連続性を維持するために、WGAN-GP は、クリティックの入力での、勾配のノルムとして計算されるクリティックネットワークに対しての、勾配ペナルティーを加えることを提案している。

---

- Formally, let I_{y_o} be the input image with the initial condition y_o, y_f the desired final condition, P_o the data distribution of the input image, and P_I~ the　random interpolation distribution.
    - 定式的には、I_{y_o} を初期条件 y_o を持つ入力画像
    - y_f を最終条件
    - P_o を入力画像のデータ分布
    - P_I~ をランダム補間分布

- Then, the critic loss ![image](https://user-images.githubusercontent.com/25688193/58755649-02de8d80-8523-11e9-9a0b-a4235c1b04a3.png) we use is:

![image](https://user-images.githubusercontent.com/25688193/58755652-17228a80-8523-11e9-9e26-08c3db9f1206.png)

- where λ_{gp} is a penalty coefficient.


#### Attention Loss.

- When training the model we do not have ground-truth annotation for the attention masks A.
    - モデルを学習するとき、我々は、attention マスク A に対して、ground-truth アノテーションを持っていない。

> attention マスクに対して、教師データをもっていない。

- Similarly as for the color masks C, they are learned from the resulting gradients of the critic module and the rest of the losses.
    - カラーマスク C はどうかというと [as for] 同様にして、これらは、クリティックモジュールと勾配と損失の残りの結果から学習される。

- However, the attention masks can easily saturate to 1 which makes that ![image](https://user-images.githubusercontent.com/25688193/58755684-9ca63a80-8523-11e9-9cee-67a7cab25479.png), that is, the generator has no effiect.
    - しかしながら、attention マスクは、より簡単に 1 に飽和し、
    - （そのことは、）![image](https://user-images.githubusercontent.com/25688193/58755684-9ca63a80-8523-11e9-9cee-67a7cab25479.png) を作り出す。即ち、生成器が効果を持たない。

> 生成器が、入力画像 I_{y_o} をそのまま出力してしまうようなケース。

- To prevent this situation, we regularize the mask with a l2-weight penalty.
    - この状況を防ぐために、我々は、マスクを、l2重みペナルティーで正規化する。

- Also, to enforce smooth spatial color transformation when combining the pixel from the input image and the color transformation C, we perform a Total Variation Regularization over A.
    - また、入力画像と色変換 C から、ピクセルを混ぜ合わせるとき、スムーズな空間的な色変換を強制するために、
    - A に渡って、全変動正規化 [Total Variation Regularization] を実施する。

- The attention loss ![image](https://user-images.githubusercontent.com/25688193/58755697-cc554280-8523-11e9-8da5-8627e3e7c280.png) can therefore be dened as:
    - attention 損失は、それ所に、以下のように定義される。

![image](https://user-images.githubusercontent.com/25688193/58755690-b6478200-8523-11e9-9a7c-9d0c280605c5.png)

> 隣り合う画素のAttentionが近くなるように、かつAttentionがかかり過ぎないようにする損失項。

> L2重みペナルティーはどの項？

> 全変動正規化処理は、どの項？<br>
> → i,j が画層の幅、高さのインデックスなので、[ Σ(A_{i+1,j}-A_{i,j})^2 + (A_{i,j+1}-A_{i,j})^2 ] の項が該当？

- where ![image](https://user-images.githubusercontent.com/25688193/58755781-ff98d100-8525-11e9-9a06-f14cbee901d8.png)  and A_{i,j} is the i,j entry of A. λ_{TV} is a penalty coefficient.
    - ここで ![image](https://user-images.githubusercontent.com/25688193/58755781-ff98d100-8525-11e9-9a06-f14cbee901d8.png) と A_{i,j} は、 A のエントリー i,j である。
    - λ_{TV} は、ペナルティー係数である。


#### Conditional Expression Loss.

- While reducing the image adversarial loss, the generator must also reduce the error produced by the AUs regression head on top of D.
    - 我々が、画像の adversarial loss を減少させる一方で、
    - 生成器はまた、<font color="Pink">D の先頭の頭の AUs 回帰</font>によって、生成されたエラーを減らさなければならない。

> AUs 回帰：Action-Unitの状態 y_f で条件づけられて生成された画像が、ちゃんと y_f という状態に従うようにする回帰

> D の先頭の頭：アーキテクチャ図における D への入力部分（＝Action-Unitの状態 y_f で条件づけられて生成された生成器の生成画像）

- In this way, G not only learns to render realistic samples but also learns to satisfy the target facial expression encoded by y_f.
    - このようにして、G はリアルなサンプルを描写するだけではなく、y_f によってエンコードされた目標の顔の表現を満足するように学習する。

- This loss is dened with two components: an AUs regression loss with fake images used to optimize G, and an AUs regression loss of real images used to learn the regression head on top of D.
    - この損失は、２つのコンポーネントで定義されている。
    - 即ち、G を最適化するために使用されている偽物画像での AUs 回帰損失 [AUs regression loss]、
    - <font color="Pink">D の先頭の頭の AUs 回帰</font>を学習するために使用されている本物画像の AUs 回帰損失 [AUs regression loss]

- This loss ![image](https://user-images.githubusercontent.com/25688193/58773089-be242680-85f6-11e9-8346-85de65367d83.png)
 is computed as:

![image](https://user-images.githubusercontent.com/25688193/58772967-4c4bdd00-85f6-11e9-8683-4b0e9823f1c2.png)

> Action-Unitの状態 y_f で条件づけられて生成された画像がちゃんと y_f という状態に従うようにはたらきかける損失項。

> 第１項：偽物画像のAUs 回帰損失 [AUs regression loss]<br>
> 第２項：本物画像のAUs 回帰損失 [AUs regression loss]<br>

#### Identity Loss.

- With the previously defined losses the generator is enforced to generate photo-realistic face transformations.
    - 以前に定義された損失関数では、生成器は、フォトリアリスティックな顔の変換を学習することを強いられいる。

- However, without ground-truth supervision, there is no constraint to guarantee that the face in both the input and output images correspond to the same person.
    - しかしながら、教師信号がなしのもとでは、入力画像と出力画像の両方での顔が、同じ人物に一致するというような、保証する [guarantee] ための制約 [constraint] は存在しない。

- Using a cycle consistency loss [38] we force the generator to maintain the identity of each individual by penalizing the difference between the original image I_{yo} and its reconstruction:
    - 周期的に連続な損失関数 [cycle consistency loss] を使用して、
    - 我々は、生成器に各個人のアイデンティティーを維持することを強制する。
    - もとの画像 I_{yo} とその再構成との間の違いにペナルティーを科す[penalizing] ことによって、

![image](https://user-images.githubusercontent.com/25688193/58773194-1ce9a000-85f7-11e9-8744-8fd1edc03b28.png)

> CycleGANのように戻ってきた結果が同じ画像になるようにする損失項。要素ごとのL1距離で測る。

- To produce realistic images it is critical for the generator to model both low and high frequencies.
    - リアルな画像を生成するために、生成器に対して、低周波と高周波の両方でモデル化することが重要である。

> 画像の高周波成分、低周波成分とは？<br>
> 画像は空間周波数という観点からとらえることができ，低い周波数成分は画像のおおまかな形状を，高い周波数成分は緻密な部分の情報を担っていることがわかります。

> 高周波領域：局所的な画像の特徴<br>
> 低周波領域：大域的な画像の特徴<br>

- Our PatchGan based critic D_I already enforces high-frequency correctness by restricting our attention to the structure in local image patches.
    - 我々の PatchGAN ベースのクリティック D_I は、局所的な画像のパッチにおける構造に注意を制限する [restricting] ことにによって、既に高周波での正確さを強化している [enforce]。

- To also capture low-frequencies it is sufficient to use l1-norm.
    - 低周波での（正確さ）も抽出するためには、L1ノルムを使用すれば十分である。

- In preliminary experiments, we also tried replacing l1-norm with a more sophisticated Perceptual [11] loss, although we did not observe improved performance.
    - 予備の [preliminary] な実験では、より洗練されたパーセプトロン損失関数で、L1　ノルムを置き換えようとした。
    - しかしながら、我々は、パフォーマンスの改善は観測されなかった。

#### Full Loss. 

- To generate the target image I_{yg} , we build a loss function L by linearly combining all previous partial losses:
    - 目標画像 I_{yg} を生成するために、前に紹介したすべての部分的な損失関数を線形結合することによって、損失関数 L を構築する。

![image](https://user-images.githubusercontent.com/25688193/58774548-034b5700-85fd-11e9-96ef-4151c11f1a04.png)

- where λ_A, λ_y and λ_{idt} are the hyper-parameters that control the relative importance of every loss term.
    - ここで、λ_A, λ_y, λ_{idt} は、各損失項の重要性を相対的に制御するような、パイパーパラメーターである。

- Finally, we can define the following minimax problem:
    - 最後に、我々は、以下のようなミニマックス問題を定義する。

![image](https://user-images.githubusercontent.com/25688193/58774774-29bdc200-85fe-11e9-8303-d592a961d8d4.png)

- where G^* draws samples from the data distribution.
    - ここで、G^*　は、データ分布からのサンプルを描写する。

- Additionally, we constrain our discriminator D to lie in D, that represents the set of 1-Lipschitz functions.
    - 加えて、我々は、リプシッツ連続な関数の集合を表す D に横たえるために、識別器 D を制約する。

> WGAN を適用するために、識別器の出力がリプシッツ連続な関数であるような、識別器（＝クリティック）とする。

## 5 Implementation Details

- Our generator builds upon the variation of the network from Johnson et al. [11] proposed by [38] as it proved to achieve impressive results for image-to-image mapping.
    - 我々の生成器は、image-to-image 写像に対しての印象深い結果を証明しているような、[38] で提案された Johnson ら（の手法）からのネットワークの変種をもとに構築されている。

- We have slightly modified it by substituting the last convolutional layer with two parallel convolutional layers, one to regress the color mask C and the other to define the attention mask A.
    - 我々は、最後の畳み込み層を、２つの並列的な畳み込み層に置き換えることによって、それを僅かに [slightly] 修正している。
    - １つは、カラーマスク C を回帰するためのもので、
    - その他は、attention マスクを定義するためのもの。

- We also observed that changing batch normalization in the generator by instance normalization improved training stability.
    - 我々はまた、instance normalization による、生成器における batch normalization の変更が、学習の安定性を改善するということを観測した。

- For the critic we have adopted the PatchGan architecture of [10], but removing feature normalization.
    - クリティックの対して、我々は、[10] のアーキテクチャである PatchGAN を適用した。一方で、feature normalization を除外している。

- Otherwise, when computing the gradient penalty, the norm of the critic’s gradient would be computed with respect to the entire batch and not with respect to each input independently.
    - さもなければ、勾配ペナルティを計算するとき、クリティックの勾配のノルムは、各入力に独立してではなく、バッチ全体に関して、計算されるであろう。

> batch norm 


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## 5 Implementation Details

- The model is trained on the EmotioNet dataset [3].
    - モデルは、EmotioNet で学習されている。

- We use a subset of 200,000 samples (over 1 million) to reduce training time.
    - 我々は、学習時間を減らすために、（100万以上のサンプルの内、）200,00サンプルのサブセットを使用する。

- We use Adam [14] with learning rate of 0.0001, beta1 0.5, beta2 0.999 and batch size 25.

- We train for 30 epochs and linearly decay the rate to zero over the last 10 epochs.

- Every 5 optimization steps of the critic network we perform a single optimization step of the generator.

- The weight coefficients for the loss terms in Eq. (5) are set to λgp = 10, λA = 0.1, λTV = 0.0001, λy = 4000, λidt = 10.

- To improve stability we tried updating the critic using a buffer with generated images in different updates of the generator as proposed in [32] but we did not observe performance improvement.
    - 安定性を向上させるために、[32]で提案されているようにジェネレータの異なる更新において生成された画像を持つバッファを使用して、評論家を更新しようとしましたが、パフォーマンスの改善は観察されませんでした。

- The model takes two days to train with a single GeForce GTX 1080 Ti GPU.


## 6 Experimental Evaluation

- This section provides a thorough evaluation of our system.
    - このセクションでは、我々のシステムの徹底的な評価を提供する。

- We first test the main component, namely the single and multiple AUs editing.
    - 我々は最初に、メインコンポーネントをテストする。
    - （これは）所謂、single and multiple AUs editing（単一と複数の AUs 編集)

- We then compare our model against current competing techniques in the task of discrete emotions editing and demonstrate our model's ability to deal with images in the wild and its capability to generate a wide range of anatomically coherent face transformations.
    - 我々は次に、 離散的な感情編集のタスクにおいて、現在競争しているテクニックと我々のモデルを比較する。
    - そして、我々のモデルの野生の画像を扱う能力と、解剖学的に [anatomically] 首尾一貫した [coherent] 顔変換の広い範囲での生成能力を実証する。

- Finally, we discuss the model's limitations and failure cases.
    - 最後に、モデルの限界と失敗ケースについて議論する。

---

- It is worth noting that in some of the experiments the input faces are not cropped.
    - いくつかの実験では、入力画像の顔がトリミングされて [cropped] いないことは注目 [noting] に値する。

- In this cases we first use a detector <2> to localize and crop the face, apply the expression transformation to that area with Eq. (1), and finally place the generated face back to its original position in the image.
    - このケースでは、我々はまず初めに、顔の位置を特定し、トリミングするために、検出器を使用し、<2>
    - 式 (1) を用いて、表情 [expression] 変換を、その領域（＝顔をトリミングした領域）に適用し、
    - 最後に、生成された顔を、画像内の元の位置に戻す。

- > <2> : We use the face detector from https://github.com/ageitgey/face_recognition.


- The attention mechanism guaranties a smooth transition between the morphed cropped face and the original image.
    - attention メカニズムは、モーフィングされ [morphed] クリッピングされた顔と元の画像との間の、スムーズな変換を保証する。

> モーフィング：モーフィングとは、画像を加工する技術のひとつで、2つの画像を合成させて中間状態を作り、一方の姿形から他方の形へと変形していくような様子を生成することである。

- As we shall see later, this three steps process results on higher resolution images compared to previous models.
    - 後で見るように、この３つのステップは、以前のモデルと比べて、より高解像度の画像を処理するという結果となる。

> この３つのステップ：<br>
> 1. まず初めに、顔の位置を特定し、トリミングするために、検出器を使用し<br>
> 2. 式 (1) を用いて、表情変換を、その領域（＝顔をトリミングした領域）に適用し、<br>
> 3. 最後に、生成された顔を、画像内の元の位置に戻す。<br>

- Supplementary material can be found on http://www.albertpumarola.com/research/GANimation/.


### 6.1 Single Action Units Edition

- We first evaluate our model's ability to activate AUs at diffierent intensities while preserving the person's identity. 
    - 我々はまず初めに、人のアイデンティティを保持しながら、異なる強度 [intensities] でAUを作動させるためのモデルの能力を評価します。

- Figure 4 shows a subset of 9 AUs individually transformed with four levels of intensity (0, 0.33, 0.66, 1).
    - 図４は、４つの強度 (0, 0.33, 0.66, 1) での、個々に変換された 9 つの AUs のサブセットを示している。

- For the case of 0 intensity it is desired not to change the corresponding AU.
    - 強度 0 のケースに対しては、一致する AU を変えないことが望ましい。

- The model properly handles this situation and generates an identical copy of the input image for every case.
    - モデルはこの状況を適切に処理し、
    - 全てのケースに対して、入力画像の固有のコピーを生成する。

- The ability to apply an identity transformation is essential to ensure that non-desired facial movement will not be introduced.
    - **恒等変換を適用する能力は、望まれない顔の動きが導入されないことを確実にするために不可欠です。**

---

![image](https://user-images.githubusercontent.com/25688193/58867149-e1350000-86f4-11e9-8eb5-83035affe5be.png)

- > Fig. 4. Single AUs edition.

- > Specic AUs are activated at increasing levels of intensity (from 0.33 to 1).

- > The first row corresponds to a zero intensity application of the AU which correctly produces the original image in all cases.

---

- For the non-zero cases, it can be observed how each AU is progressively accentuated.
    - ゼロ以外の場合、各AUがどのようにして徐々に強調される [accentuated] かを観察できます。

- Note the diffierence between generated images at intensity 0 and 1.

- The model convincingly renders complex facial movements which in most cases are difficult to distinguish from real images.
    - このモデルは説得力を持って複雑な顔の動きをレンダリングしますが、
    - ほとんどの場合、本物画像と区別するのは困難です。

- It is also worth mentioning that the independence of facial muscle cluster is properly learned by the generator.
    - 顔面筋群の独立性が生成器によって正しく学習されていることも言及する価値があります

- AUs relative to the eyes and half-upper part of the face (AUs 1, 2, 4, 5, 45) do not affiect the muscles of the mouth.

- Equivalently, mouth related transformations (AUs 10, 12, 15, 25) do not affect eyes nor eyebrow muscles.

> 目と顔の上部分に関連する AUs（AUs1,AUs2,AUs4,AUs5,AUs45）は、口の筋肉に影響を与えていない。<br>
> 同様にして、口に関連する AUs（AUs10,AUs12,AUs15,AUs25）は、目や眉毛の筋肉に影響を与えていない。<br>
> この関係を、生成器が正しく学習できているかは、１つのポイントとなる。<br>

---

- Fig. 5 displays, for the same experiment, the attention A and color C masks that produced the final result I_{yg}.

- Note how the model has learned to focus its attention (darker area) onto the corresponding AU in an unsupervised manner.
    - 教師なしの方法で、モデルがどのようにして対応するAUに attention を集中させる（暗い領域）かを学習したことに注意してください。

- In this way, it relieves the color mask from having to accurately regress each pixel value.
    - このようにして、それは（＝attentio mask は、）カラーマスクが各ピクセル値を正確に回帰させなければならないことから解放する。

- Only the pixels relevant to the expression change are carefully estimated, the rest are just noise.
    - 表情の変化に関連するピクセルのみが慎重に推定され、残りは単なるノイズです。

- For example, the attention is clearly obviating background pixels allowing to directly copy them from the original image.
    - 例えば、attention は、元の画像から直接それらをコピーすることを可能にする背景ピクセルを、明らかに取り除く [obviating] ことです。

- This is a key ingredient to later being able to handle images in the wild (see Section 6.5).

---

![image](https://user-images.githubusercontent.com/25688193/58870446-26f4c700-86fb-11e9-9e38-7d5a0c301ac6.png)

- > Fig. 5. Attention Model.

- > Details of the intermediate attention mask A (first row) and the color mask C (second row).

- > The bottom row images are the synthesized expressions.

- > Darker regions of the attention mask A show those areas of the image more relevant for each specic AU.

- > Brighter areas are retained from the original image.


### 6.2 Simultaneous Edition of Multiple AUs

![image](https://user-images.githubusercontent.com/25688193/58874847-d8e4c100-8704-11e9-908b-be693b7c6e18.png)

- > Fig. 1. Facial animation from a single image.

- > We propose an anatomically coherent approach that is not constrained to a discrete number of expressions and can animate a given image and render novel expressions in a continuum.
    - > 我々は、離散的な数の表現に制限されない解剖学的に一貫したアプローチを提案し、与えられた画像をアニメートし、連続的に新しい表現をレンダリングすることができる。

- > In these examples, we are given solely the left-most input image I_{yr} (highlighted by a green square), and the parameter α controls the degree of activation of the target action units involved in a smiling-like expression.
    - > これらの例では、一番左の入力画像 I_{yr}（緑色の四角で強調表示されている）だけが表示され、パラメータ α は笑顔のような表現に含まれる目標 AUs の活性化の程度を制御します。

- > Additionally, our system can handle images with unnatural illumination conditions, such as the example in the bottom row.
    - > さらに、下の例のように、不自然な照明条件で画像を処理することもできます。

---


- We next push the limits of our model and evaluate it in editing multiple AUs.
    - 我々は次に、我々のモデルの限界をプッシュし、複数の AUs を編集することで、それを評価する。

- Additionally, we also assess its ability to interpolate between two expressions.
    - 加えて、２つの表情の間を補間する能力も、それで評価する。

- The results of this experiment are shown in Fig. 1, the first column is the original image with expression yr, and the right-most column is a synthetically generated image conditioned on a target expression yg.
    - この実験の結果は図１に示されており、
    - 最初の列は、表情 y_r での元の画像 I_{yr}
    - 一番右の列は、目標表情 y_g で条件付けされて合成的に生成された画像

- The rest of columns result from evaluating the generator conditioned with a linear interpolation of the original and target expressions: α × y_g + (1 - α) × y_r.
    - 残りの列は、元の表情と目標表情の線形補間 α × y_g + (1 - α) × y_r で条件付けされた生成器を評価した結果である。

- The outcomes show a very remarkable smooth an consistent transformation across frames.
    - 結果 [outcomes] は、フレームに渡っての、大変注目に値するスムーズで連続的な変換を示している。

- We have intentionally selected challenging samples to show robustness to light conditions and even, as in the case of the avatar, to non-real world data distributions which were not previously seen by the model.
    - 我々は、照明条件や
    - アバターのケースのように、モデルによって、以前は見られなかった非現実世界のデータ分布
    - に対しての堅牢性を示すために、変換するサンプルを意図的に選択している。

- These results are encouraging to further extend the model to video generation in future works.
    - これらの結果は、将来の研究において、動画生成のために、モデルを更に拡張することを推奨している。


### 6.3 Discrete Emotions Editing

- We next compare our approach, against the baselines DIAT [20], CycleGAN [28], IcGAN [26] and StarGAN [4].
    - 我々は次に、ベースラインの DIAT, CycleGAN, IcGAN, StarGAN に対して、我々の手法と比較する。

- For a fair comparison, we adopt the results of these methods trained by the most recent work, StarGAN, on the task of rendering discrete emotions categories (e.g., happy, sad and fearful) in the RaFD dataset [16].
    - 公正な比較のために、
    - <font color="Pink">RaFD データセットにおける、離散的な感情のカテゴリ （例えば、幸せ、悲しみ、恐怖など）をレンダリングするタスクにおいて、
    - 最も最近の研究である StarGAN によって、学習されたこれらの手法の結果を適用した。</font>

- Since DIAT [20] and CycleGAN [28] do not allow conditioning, they were independently trained for every possible pair of source/target emotions.
    - DIAT と CycleGAN は、条件付けが許可されていないので、
    - それらは、ソース / ターゲット感情の全ての可能なペアに対して、独立的に学習された。

- We next briefly discuss the main aspects of each approach:
    - 我々は次に、各アプローチの主な側面について議論する。


- DIAT [20].
    - Given an input image x ∈ X and a reference image y ∈ Y , DIAT learns a GAN model to render the attributes of domain Y in the image x while conserving the person's identity.
        - 入力画像 x ∈ X と参照画像 y ∈ Y を与えれば、DIAT は、人物の同一性を保存する一方で、画像 x における領域 y の属性をレンダリングするために、GAN のmドエルを学習する。
    - It is trained with the classic adversarial loss and a cycle loss ![image](https://user-images.githubusercontent.com/25688193/58876349-c66c8680-8708-11e9-9df0-f65137c91e8b.png) to preserve the person's identity.
        - これは、人物の同一性を保存するために、古典的な adversarial loss と cycle loss ![image](https://user-images.githubusercontent.com/25688193/58876349-c66c8680-8708-11e9-9df0-f65137c91e8b.png) で学習される。

- CycleGAN [28].
    - Similar to DIAT [20], CycleGAN also learns the mapping between two domains X → Y and Y → X.
        - DIAT とよく似て、CycleGAN も、２つの領域間の写像 X → Y ; Y → X を学習する。
    - To train the domain transfer, it uses a regularization term denoted cycle consistency loss combining two cycles: ![image](https://user-images.githubusercontent.com/25688193/58876440-0e8ba900-8709-11e9-8e50-e762c09c90d1.png).
        - 領域の変換を学習するために、２つのサイクルの組み合したサイクルで一貫性のある [consistency] 損失関数で示される正規化項 ![image](https://user-images.githubusercontent.com/25688193/58876440-0e8ba900-8709-11e9-8e50-e762c09c90d1.png) を使用する。

- IcGAN [26].
    - Given an input image, IcGAN uses a pretrained encoder-decoder to encode the image into a latent representation in concatenation with an expression vector y to then reconstruct the original image.
        - 入力画像を与えれば、
        - 画像を、表情ベクトル y で連結した潜在表現にエンコードし、次に元の画像を再構成するために、
        - IcGAN は事前学習された encoder-decoder を使用する。
    - It can modify the expression by replacing y with the desired expression before going through the decoder.
        - それは、検出器を通す前に、望ましい表情で y を置き換えることによって、表情を修正することが出来る。

- StarGAN [4].
    - An extension of cycle loss for simultaneously training between multiple datasets with diffierent data domains.
        - 異なるデータ領域を持つ複数のデータセットの間の同時に学習するための、サイクル loss の拡張
    - It uses a mask vector to ignore unspecied labels and optimize only on known ground-truth labels.
        - 指定されていない [unspecied] ラベルを無視するためのマスクベクトルを使用し、知られている ground-truth ラベルのみ最適化する。
    - It yields more realistic results when training simultaneously with multiple datasets.
        - これは、複数のデータセットを同時に学習するとき、よりリアルな結果を生み出す。

---

- Our model diffiers from these approaches in two main aspects. 
    - **我々のモデルは、２つの側面において、これらのアプローチとは異なる。**

- First, we do not condition the model on discrete emotions categories, but we learn a basis of anatomically feasible warps that allows generating a continuum of expressions.
    - **まず初めに、我々は、離散的な感情のカテゴリの上で、モデルを条件づけしない。**
    - **しかし、表情の連続体 [continuum] を生成することを可能にする解剖学的に実現可能な [feasible] 歪み（ねじれ） [warps] のベースを学習する。**

- Secondly, the use of the attention mask allows applying the transformation only on the cropped face, and put it back onto the original image without producing any artifact.
    - ２つ目は、attention mask の使用は、クリッピングされた顔のみで変換を適用することを許容し、
    - いかなる人工加工物を生成することなしに、それを（＝クリッピングされた顔のみで変換したもの）元の画像に戻す。

- As shown in Fig. 6, besides estimating more visually compelling images than other approaches, this results on images of higher spatial resolution.
    - 図６に示されているように、
    - 他のアプローチよりも、画像をより視覚的に説得力のある [compelling] 推定を行うことに加えて、
    - この手法は、より高い空間的な解像度となる結果となる。

---

![image](https://user-images.githubusercontent.com/25688193/58924855-66b1c200-8780-11e9-9c0d-8739ed0725c3.png)

- > Fig. 6. Qualitative comparison with state-of-the-art.
    - > SOTA手法でのクオリティー比較

- > Facial Expression Synthesis results for: DIAT [20], CycleGAN [28], IcGAN [26] and StarGAN [4]; and ours.
    - > DIAT [20], CycleGAN [28], IcGAN [26] and StarGAN [4]、我々の手法に対しての顔の表情の合成結果

- > In all cases, we represent the input image and seven diffierent facial expressions.
    - > 全てのケースにおいて、入力画像よ７つの異なる顔の表情を表現している。

- > As it can be seen, our solution produces the best trade-off between visual accuracy and spatial resolution.
    - > （図から）見られるように、我々の解決法は、視覚的な正確性と空間的な解像度の間のトレードオフでベストな（もの）を生成している

- > Some of the results of StarGAN, the best current approach, show certain level of blur.
    - > ベストな現在のアプローチである StarGAN の結果のいくつかは、ある程度 [certain] のレベルのぼやけを示している。

- > Images of previous models were taken from [4].


### 6.4 High Expressions Variability

- Given a single image, we next use our model to produce a wide range of anatomically feasible face expressions while conserving the person's identity.
    - 一つの画像を与えれば、人物のアイデンティティを保存しながら、我々は次に、広い範囲での解剖学的に実現可能な顔の表情を生成するために、我々のモデルを使用する。

- In Fig. 7 all faces are the result of conditioning the input image in the top-left corner with a desired face conguration defined by only 14 AUs.
    - 図７では、全ての顔は、14 個の AUs だけで定義された望ましい顔構成で、最上段左コーナーの画像を条件付けした結果である。

- Note the large variability of anatomically feasible expressions that can be synthesized with only 14 AUs.
    - 大きな種類の解剖学的に実現可能な表情が、14 個の AUs のみでを合成されることが可能ということに注目。

---

- > Fig. 7. Sampling the face expression distribution space.
    - > 図７：顔の表情の分布空間のサンプリング

- > As a result of applying our AU-parametrization through the vector y_g, we can synthesize, from the same source image I_{yr} , a large variety of photo-realistic images.
    - > **AUs 状態ベクトルを通じて、我々の AU-パラメーターを適用した結果として、同じソース画像から大きな種類のフォトリアリスティックな画像合成できる。**

### 6.5 Images in the Wild

- As previously seen in Fig. 5, the attention mechanism not only learns to focus on specic areas of the face but also allows merging the original and generated image background.
    - 以前に図５で見られたように、attention メカニズムは、顔の特定の領域にフォーカスすることを学習するだけでなく、元の画像と生成された画像の背景をマージ（合併）することを許容する。

- This allows our approach to be easily applied to images in the wild while still obtaining high resolution images.
    - このことは、高解像の画像を手に入れながら、我々のアプローチを、wild な画像に用意に適用されることを可能にする。

- For these images we follow the detection and cropping scheme we described before.
    - これらの画像に対して、我々は、検出器と我々が以前記述したクリッピング手法（スキーム）[scheme] に従う。

> 我々が以前記述したクリッピング手法（スキーム）<br>
> →３つのステップ：<br>
> 1. まず初めに、顔の位置を特定し、トリミングするために、検出器を使用し<br>
> 2. 式 (1) を用いて、表情変換を、その領域（＝顔をトリミングした領域）に適用し、<br>
> 3. 最後に、生成された顔を、画像内の元の位置に戻す。<br>


- Fig. 8 shows two examples on these challenging images.
    - 図８は、これらの画像変換の２つの例を示している。

- Note how the attention mask allows for a smooth and unnoticeable merge between the entire frame and the generated faces.
    - attention mask は、どうようにして、フレーム全体と生成された顔画像との間のスムーズで目立たない [unnoticeable] マージンを許容しているのかに注目。

---

![image](https://user-images.githubusercontent.com/25688193/58930732-b6e84e80-8797-11e9-84c1-918a0a961ce5.png)

- > Fig. 8. Qualitative evaluation on images in the wild.

- > Top:We represent an image (left) from the film "Pirates of the Caribbean" and an its generated image obtained by our approach (right).

- > Bottom: In a similar manner, we use an image frame (left) from the series "Game of Thrones" to synthesize five new images with diffierent expressions.


### 6.6 Pushing the Limits of the Model

- We next push the limits of our network and discuss the model limitations.
    - 我々は次に、我々のネットワークの限界をプッシュし、モデルの限界を議論する。

- We have split success cases into six categories which we summarize in Fig. 9-top.
    - 成功事例を６つのカテゴリに分割し、図９の上に示す。

- The first two examples (top-row) correspond to human-like sculptures and non-realistic drawings.
    - 最初の２つのサンプルは、人間のような彫刻とリアルではない絵に対応している。

- In both cases, the generator is able to maintain the artistic effects of the original image.
    - 両方のケースにおいて、生成器は、元の画像の芸術的な [artistic] 効果を維持することが出来る。

- Also, note how the attention mask ignores artifacts such as the pixels occluded by the glasses.
    - また、attention mask がどのようにして、眼鏡で塞がれた [occluded] ピクセルのような、人工物を無視しているのかに注目。

- The third example shows robustness to non-homogeneous textures across the face.
    - ３つ目の例は、顔のいたるところでの [across] 不均一な [non-homogeneous] テスクチャへの堅牢性を示している。

- Observe that the model is not trying to homogenize the texture by adding/removing the beard's hair.
    - モデルは、ひげ [berad] の髪を追加 / 削除することによって、テスクチャを均質化しようとしていないことを、観測する。

- The middle-right category relates to anthropomorphic faces with non-real textures.
    - 中央右のカテゴリは、非現実的なテクスチャーをもつ擬人的な [anthropomorphic] 顔に関連している。

- As for the Avatar image, the network is able to warp the face without affecting its texture.
    - アバターの画像はというと、ネットワークは、そのテクスチャーに影響を与えることなしに、顔を歪ませることが出来る。

- The next category is related to non-standard illuminations/colors for which the model has already been shown robust in Fig. 1.
    - 次のカテゴリは、図１のロバスト性で既に示されたようなモデルに対しての、非標準的な照明 [illuminations] や色に関連したものである。

- The last and most surprising category is face-sketches (bottom-right).
    - 最後のそして最も驚くカテゴリは、顔のイラストである。（右下）

- Although the generated face suffers from some artifacts, it is still impressive how the proposed method is still capable of finding sufficient features on the face to transform its expression from worried to excited.
    - 生成された顔は、いくつかの人工物に悩まされるけれども、
    - 提案された手法がどのようにして、その表情を心配 [worried] から興奮へ変換するために、顔の十分な特徴見つけ出す能力をもつのかということは、依然として印象深い。

- The second case shows failures with non-previously seen occlusions such as an eye patch causing artifacts in the missing face attributes.
    - 次のケースでは、欠けている顔の属性で人工物を引き起こす眼帯ような、以前の閉鎖 [occlusion] では見られなかった失敗を示す。

---

- We have also categorized the failure cases in Fig. 9-bottom, all of them presumably due to insufficient training data.
    - 我々はまた、失敗事例を図9-下に分類したが、それらはすべては、おそらくは [presumably]、十分に訓練データがないためである。

- The first case is related to errors in the attention mechanism when given extreme input expressions.
    - 最初のケースは、極端な入力表情が与えられたときの、attention メカニズムのエラーに関連したものである。

- The attention does not weight sufficiently the color transformation causing transparencies.
    - attention は、透過（透明）を引き起こす色変換に十分に重みを与えない。

---

- The model also fails when dealing with non-human anthropomorphic distributions as in the case of cyclopes.
    - サイクロプスのような、人間ではない擬人化 [anthropomorphic] 分布を扱うときにも、失敗する。

- Lastly, we tested the model behavior when dealing with animals and observed artifacts like human face features.
    - 最後に、動物を扱うときの振る舞いでモデルをテストし、人間の顔のような人工物を観測した。

---

![image](https://user-images.githubusercontent.com/25688193/58932607-5b21c380-879f-11e9-92d6-c30f327e3efa.png)

- > Fig. 9. Success and Failure Cases.

- > In all cases, we represent the source image I_{yr} , the target one I_{yg} , and the color and attention masks C and A, respectively. 

- > Top: Some success cases in extreme situations.

- > Bottom: Several failure cases.


# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


