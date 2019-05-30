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

- Ekman and Friesen [6] developed the Facial Action Coding System (FACS) for describing facial expressions in terms of the so-called Action Units (AUs), which are anatomically related to the contractions of specic facial muscles.

- Although the number of action units is relatively small (30 AUs were found to be anatomically related to the contraction of specic facial muscles), more than 7,000 diffierent AU combinations have been observed [30].

- For example, the facial expression for fear is generally produced with activations: Inner Brow Raiser (AU1), Outer Brow Raiser (AU2), Brow Lowerer (AU4), Upper Lid Raiser (AU5), Lid Tightener (AU7), Lip Stretcher (AU20) and Jaw Drop (AU26) [5].
    - 例えば、恐怖のための表情は一般に活性化で作り出されます：眉の内部ライザー（AU1）、外部眉ライザー（AU4）、アッパーリッドライザー（AU5）、リッドタイトナー（AU7）、唇トレッチャー（AU7） AU20）とJaw Drop（AU26）[5]。

- Depending on the magnitude of each AU, the expression will transmit the emotion of fear to a greater or lesser extent.

---

- In this paper we aim at building a model for synthetic facial animation with the level of expressiveness of FACS, and being able to generate anatomically-aware expressions in a continuous domain, without the need of obtaining any facial landmarks [36].
    - 本稿では、FACSの表現力を用いて合成顔面アニメーションのモデルを構築し、顔面の目印 [landmarks] を取得することなく、連続領域で解剖学的に意識した表現を生成できることを目的とする[36]。

- For this purpose we leverage on the recent EmotioNet dataset [3], which consists of one million images of facial expressions (we use 200,000 of them) of emotion in the wild annotated with discrete AUs activations 1. 

- We build a GAN architecture which, instead of being conditioned with images of a specic domain as in [4], it is conditioned on a one-dimensional vector indicating the presence/absence and the magnitude of each action unit.
    - [4]のように特定のドメインの画像で条件付けされるのではなく、
    - 各アクションユニットの有 [presence] / 無 [absence] と大きさを示す1次元ベクトルで条件付けされるGANアーキテクチャを構築します。

- We train this architecture in an unsupervised manner that only requires images with their activated AUs.

- To circumvent the need for pairs of training images of the same person under diffierent expressions, we split the problem in two main stages.

- First, we consider an AU-conditioned bidirectional adversarial architecture which, given a single training photo, initially renders a new image under the desired expression.

- This synthesized image is then rendered-back to the original pose, hence being directly comparable to the input image.

- We incorporate very recent losses to assess the photorealism of the generated image. Additionally, our system also goes beyond state-of-the-art in that it can handle images under changing backgrounds and illumination conditions.

- We achieve this by means of an attention layer that focuses the action of the network only in those regions of the image that are relevant to convey the novel expression.

---

- As a result, we build an anatomically coherent facial expression synthesis method, able to render images in a continuous domain, and which can handle images in the wild with complex backgrounds and illumination conditions.

- As we will show in the results section, it compares favorably to other conditioned-GANs schemes, both in terms of the visual quality of the results, and the possibilities of generation. 

- Figure 1 shows some example of the results we obtain, in which given one input image, we gradually change the magnitude of activation of the AUs used to produce a smile.



# ■ 結論
s
## x. 論文の項目名 (Conclusion)


# ■ 何をしたか？詳細

## x. 論文の項目名


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


