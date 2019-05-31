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

## x. 論文の項目名 (Conclusion)


# ■ 何をしたか？詳細

## x. 論文の項目名


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

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

- Given the image I_{y_o} ∈ R^{H×W×3} and the N-vector y_f encoding the desired expression, we form the input of generator as a concatenation (I_{y_o} , y_o) ∈ R^{H×W×3}, where y_o has been represented as N arrays of size H × W.
    - 画像 ![image](https://user-images.githubusercontent.com/25688193/58684905-d7d62b80-83b4-11e9-8440-4ca261297f48.png) と望ましい表現でエンコードされたNベクトル y_f を与えれば、
    - 生成器の入力を、連結 ![image](https://user-images.githubusercontent.com/25688193/58685065-6a76ca80-83b5-11e9-9461-0aad699fd4bf.png) として、形成する。

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

> 望ましい表現の実現 [fullfilment]：アーキテクチャ図におけるクリティックの出力 ![image](https://user-images.githubusercontent.com/25688193/58687076-9006d280-83bb-11e9-934c-746053e30488.png) のこと

- The structure of D(I) resembles that of the PatchGan [10] network mapping from the input image I to a matrix ![image](https://user-images.githubusercontent.com/25688193/58686296-66e54280-83b9-11e9-9db8-2bbe14f6695f.png),
    - D(I) の構造は、入力画像 I から行列 ![image](https://user-images.githubusercontent.com/25688193/58686296-66e54280-83b9-11e9-9db8-2bbe14f6695f.png) へ写像するネットワークである、PatchGAN に似ている。

- where ![image](https://user-images.githubusercontent.com/25688193/58686425-bc215400-83b9-11e9-8e79-8ca051ef4f2c.png) represents the probability of the overlapping patch i,j to be real.
    - ここで、![image](https://user-images.githubusercontent.com/25688193/58686425-bc215400-83b9-11e9-8e79-8ca051ef4f2c.png) は、重なっているパッチ i,j が真になる確率を表現している。

- Also, to evaluate its conditioning, on top of it we add an auxiliary regression head that estimates the AUs activations ![image](https://user-images.githubusercontent.com/25688193/58686538-099dc100-83ba-11e9-9f80-8e862da660a6.png) in the image.
    - また、その条件付けを評価するために、
    - それに加えて [on top of it]、画像内の AUs の活性化 ![image](https://user-images.githubusercontent.com/25688193/58686538-099dc100-83ba-11e9-9f80-8e862da660a6.png) を推定するような、補助的な [auxiliary] 回帰ヘッドを追加します。


# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


