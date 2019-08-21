> 論文まとめ（短いver）：https://github.com/Yagami360/MachineLearning-Papers_Survey/issues/5

# ■ 論文
- 論文タイトル："Towards Multi-pose Guided Virtual Try-on Network"
- 論文リンク：https://arxiv.org/abs/1902.11026
- 論文投稿日付：2019/02/28
- 被引用数（記事作成時点）：2 件
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- Virtual try-on system under arbitrary human poses has huge application potential, yet raises quite a lot of challenges, e.g self-occlusions, heavy misalignment among diverse poses, and diverse clothes textures. Existing methods aim at fitting new clothes into a person can only transfer clothes on the fixed human pose, but still show unsatisfactory performances which often fail to preserve the identity, lose the texture details, and decrease the diversity of poses.
    - 任意の人間のポーズの下での仮想試着システムには、大きな応用の可能性がありますが、自己閉塞、多様なポーズの間の激しいずれ、多様な衣服の質感など、非常に多くの課題が生じます。 既存の方法は、新しい衣服を人にフィットさせることを目的としており、固定された人間のポーズでのみ衣服を移すことができますが、依然としてアイデンティティを維持できず、テクスチャの詳細を失い、ポーズの多様性を減らす不満足なパフォーマンスを示します。

- In this paper, we make the first attempt towards multi-pose guided virtual try-on system, which enables transfer clothes on a person image under diverse poses. Given an input person image, a desired clothes image, and a desired pose, the proposed Multi-pose Guided Virtual Try-on Network (MG-VTON) can generate a new person image after fitting the desired clothes into the input image and manipulating human poses.
    - Our MG-VTON is constructed in three stages:
    - 1) a desired human parsing map of the target image is synthesized to match both the desired pose and the desired clothes shape;
    - 2) a deep Warping Generative Adversarial Network (Warp-GAN) warps the desired clothes appearance into the synthesized human parsing map and alleviates the misalignment problem between the input human pose and desired human pose;
    - 3) a refinement render utilizing multi-pose composition masks recovers the texture details of clothes and removes some artifacts. 

    - 本論文では、多様なポーズの人物画像に衣服を移すことを可能にするマルチポーズ誘導仮想試着システムに向けた最初の試みを行う。 入力人物の画像、希望の衣服の画像、および希望のポーズが与えられると、提案されたマルチポーズガイド付き仮想試着ネットワーク（MG-VTON）は、希望の衣服を入力画像に合わせて人間の姿勢を操作した [manipulating] 後、新しい人物の画像を生成できます。
        - MG-VTONは3つの段階で構成されています。
        - 1）目的のポーズと衣服の形状の両方に一致するように、対象画像の目的の人間解析マップを合成します。
        - 2）深いワーピング生成的敵対ネットワーク（Warp-GAN）は、望ましい衣服の外観を合成された人間の解析マップにワープし、入力された人間のポーズと望ましい人間のポーズの間の不整合 [misalignment] の問題を軽減します。
        - 3）マルチポーズコンポジションマスクを利用した洗練された [refinement] レンダリングは、衣服のテクスチャの詳細を回復し、いくつかのアーティファクトを除去します。

- Extensive experiments on well-known datasets and our newly collected largest virtual try-on benchmark demonstrate that our MG-VTON significantly outperforms all state-of-the-art methods both qualitatively and quantitatively with promising multi-pose virtual try-on performances.
    - 有名なデータセットと新しく収集された最大の仮想試着ベンチマークでの広範な実験は、MG-VTONがすべての最先端の方法よりも有望なマルチポーズ仮想試着パフォーマンスで質的および量的に大幅に優れていることを示しています。

# ■ イントロダクション（何をしたいか？）

## 1. Introduction

- Learning to synthesize the image of person conditioned on the image of clothes and manipulate the pose simultaneously is a significant and valuable task in many applications such as virtual try-on, virtual reality, and human-computer interaction. 
- In this work, we propose a multi-stage method to synthesize the image of person conditioned on both clothes and pose.
- Given an image of a person, a desired clothes, and a desired pose, we generate the realistic image that preserves the appearance of both desired clothes and person, meanwhile reconstructing the pose, as illustrated in Figure 1.
- Obviously, delicate and reasonable synthesized outfit with arbitrary pose is helpful for users in selecting clothes while shopping.
    - 衣服の画像で条件付けられた人物の画像を合成し、同時にポーズを操作することを学ぶことは、仮想試着、仮想現実、および人間とコンピューターの相互作用 [interaction] などの多くのアプリケーションで重要かつ貴重なタスクです。
    - 本研究では、衣服とポーズの両方に条件付けられた人物の画像を合成する多段階法を提案する。
    - 図1に示すように、人物、希望の服、および希望のポーズの画像が与えられると、ポーズを再構築している間に [meanwhile]、希望の服と人物の両方の外観を保持するリアルな画像を生成する。
    - 明らかに、買い物中に衣服を選択する際に、ユーザーが任意のポーズで繊細で合理的な合成服を着用すると便利です。

---

- However, recent image synthesis approaches [8, 29] for virtual try-on mainly focus on the fixed pose and fail to preserve the fine details, such as the clothing of lower-body and the hair of the person lose the details and style, as shown in Figure 4. In order to generate the realistic image, those methods apply a coarse-to-fine network to produce the image conditioned on clothes only. They ignore the significant features of the human parsing, which leads to synthesize blurry and unreasonable image, especially in case of conditioned on various poses. For instance, as shown in Figure 4, the clothing of lower-body cannot be preserved while the clothing of upper-body is replaced. The head of the person fail to identify while conditioned different poses.
- Other exiting works [14, 20, 35] usually leverage 3D measurements to solve those issues since the 3D information have abundant details of the shape of the body that can help to generate the realistic results. However, it needs expert knowledge and huge labor cost to build the 3D models, which requires collecting the 3D annotated data and massive computation. These costs and complexity would limit the applications in the practical virtual try-on simulation.
    - **ただし、仮想試着の最近の画像合成アプローチ[8、29]は、主に固定ポーズに焦点を当てており、下半身の衣服や人の髪の毛が細部やスタイルを失うなど、細かい細部を保持できません。図4に示すように、現実的な画像を生成するために、これらの方法は粗から密なネットワークを適用して、衣服のみを条件とした画像を生成します。彼らは、人間の解析の重要な機能を無視します。これは、特にさまざまなポーズを条件とする場合、ぼやけた不合理な画像を合成します。たとえば、図4に示すように、上半身の衣服を交換している間、下半身の衣服を保存することはできません。人の頭は、さまざまなポーズを条件付けているときに識別できません。**
    - 他の既存の作品[14、20、35]は、通常、3D情報を活用してこれらの問題を解決します。3D情報には、現実の結果を生成するのに役立つ身体の形状の詳細が豊富にあるからです。ただし、3Dモデルを構築するには専門知識と膨大な人件費が必要であり、3D注釈付きデータの収集と大規模な計算が必要です。これらのコストと複雑さにより、実用的な仮想試着シミュレーションのアプリケーションが制限されます。

---

- In this paper, we study the problem of virtual try-on conditioned on 2D images and arbitrary poses, which aims to learn a mapping function from an input image of a person to another image of the same person with a new outfit and diverse pose, by manipulating the target clothes and pose. Although the image-based virtual try-on with the fixed pose has been studied widely [8, 29, 37], the task of multi-pose virtual try-on is less explored. In addition, without modeling the mapping of the intricate interplay among of the appearance, the clothes, and the pose, directly using the existing virtual try-on methods to synthesized image based on different poses often result in blurry and artifacts.
    - この論文では、2D画像と任意のポーズを条件とした仮想試着の問題を研究します。これは、新しい服装と多様なポーズを持つ人の入力画像から同じ人の別の画像へのマッピング関数を学習することを目的とし、 ターゲットの服とポーズを操作します。 固定ポーズでの画像ベースの仮想試着は広く研究されていますが[8、29、37]、マルチポーズ仮想試着のタスクはあまり研究されていません。 さらに、外観、衣服、ポーズの複雑な相互作用のマッピングをモデル化せずに、既存の仮想試着方法を直接使用してさまざまなポーズに基づいて合成画像を作成すると、ぼやけやアーティファクトが発生することがよくあります。

---

- Targeting on the problems mentioned above, we propose a novel Multi-pose Guided Virtual Try-on Network (MG-VTON) that can generate a new person image after fitting both desired clothes into the input image and manipulating human poses. Our MG-VTON is a multi-stage framework with generative adversarial learning.
    - 上記の問題をターゲットに、入力画像に希望の衣服を合わせて人間のポーズを操作した後、新しい人物の画像を生成できる新しいマルチポーズガイド付き仮想試着ネットワーク（MG-VTON）を提案します。 MG-VTONは、生成的な敵対的学習を備えたマルチステージフレームワークです。

- Concretely, we design a pose-clothes-guided human parsing network to estimate a plausible human parsing of the target image conditioned on the approximate shape of the body, the face mask, the hair mask, the desired clothes, and the target pose, which could guide the synthesis in an effective way with the precise region of body parts.
    - 具体的には [Concretely]、体、顔のマスク、ヘアマスク、目的の服、およびターゲットポーズのおおよその形状で条件付けられたターゲット画像のもっともらしい [plausible] 人間の解析を推定するために、ポーズ服ガイド付き人間解析ネットワークを設計します。これらは、体の部位の正確な領域で効果的な方法で合成を導くことができます。

- To seamlessly fit the desired clothes on the person, we warp the desired clothes image, by exploiting a geometric matching model to estimate the transformation parameters between the mask of the input clothes image and the mask of the synthesized clothes extracted from the synthesized human parsing.
- In addition, we design a deep Warping Generative Adversarial Network (Warp-GAN) to synthesize the coarse result alleviating the large misalignment caused by the different poses and the diversity of clothes. - Finally, we present a refinement network utilizing multi-pose composition masks to recover the texture details and alleviate the artifact caused by the large misalignment between the reference pose and the target pose.
    - 目的の衣服を人にシームレスにフィットさせるために、入力衣服画像のマスクと、合成された人間の解析から抽出された合成衣服のマスクとの間の変換パラメーターを推定するような幾何学的マッチングモデルを利用する [exploiting] ことにより、目的の衣服画像をワープします。
    - さらに、深いワーピング生成的敵対ネットワーク（Warp-GAN）を設計して、さまざまなポーズや衣服の多様性に起因する大きなミスアライメントを緩和する粗い結果を合成します。 - 最後に、マルチポーズコンポジションマスクを使用してテクスチャの詳細を回復し、基準ポーズとターゲットポーズの間の大きなミスアライメントによって引き起こされるアーティファクトを軽減する改良ネットワークを提示します。    

---

- To demonstrate our model, we collected a new dataset, named MPV, by collecting various clothes image and person images with diverse poses from the same person. In addition, we also conduct experiments on DeepFashion [38] datasets for testing. Following the object evaluation protocol [30], we conduct a human subjective study on the Amazon Mechanical Turk (AMT) platform. Both quantitative and qualitative results indicate that our method achieves effective performance and high-quality images with appealing details. The main contributions are listed as follows:
    - モデルを実証するために、同じ人物からさまざまなポーズのさまざまな衣服画像と人物画像を収集することにより、MPVという名前の新しいデータセットを収集しました。 さらに、テスト用のDeepFashion [38]データセットの実験も行っています。 オブジェクト評価プロトコル[30]に従って、Amazon Mechanical Turk（AMT）プラットフォームに関する人間の主観的な調査を実施します。 定量的および定性的結果の両方は、我々の方法が魅力的な詳細で効果的なパフォーマンスと高品質の画像を達成することを示しています。 主な貢献は次のとおりです。

- A new task of virtual try-on conditioned on multi-pose is proposed, which aims to restructure the person image by manipulating both diverse poses and clothes.
    - マルチポーズを条件とした仮想試着の新しいタスクが提案されています。これは、多様なポーズと服の両方を操作することにより、人物のイメージを再構築することを目的としています。

- We propose a novel Multi-pose Guided Virtual Try-on Network (MG-VTON) that generates a new person image after fitting the desired clothes onto the input person image and manipulating human poses. MG-VTON contains four modules: 1) a pose-clothes-guided human parsing network is designed to guide the image synthesis; 2) a Warp-GAN learns to synthesized realistic image by using a warping features strategy; 3) a refinement network learns to recover the texture details; 4) a mask-based geometric matching network is presented to warp clothes that enhances the visual quality of the generated image.
    - 入力した人物の画像に希望の衣服をフィットさせ、人間のポーズを操作した後に新しい人物の画像を生成する、新しいマルチポーズガイド付き仮想試着ネットワーク（MG-VTON）を提案します。 MG-VTONには4つのモジュールが含まれています。1）ポーズ服ガイドの人間解析ネットワークは、画像合成をガイドするように設計されています。 2）Warp-GANは、ワーピングフィーチャ戦略を使用して、合成された現実的な画像を学習します。 3）洗練ネットワークは、テクスチャの詳細を回復することを学習します。 4）生成された画像の視覚的品質を向上させる服をゆがめるために、マスクベースの幾何学的マッチングネットワークが提示されます。

- A new dataset for the multi-pose guided virtual try- on task is collected, which covers person images with more poses and clothes diversity. The extensive experiments demonstrate that our approach can achieve the competitive quantitative and qualitative results.
    - マルチポーズガイド付き仮想試着タスクの新しいデータセットが収集されます。これは、より多くのポーズと服の多様性を持つ人物の画像を対象としています。 広範な実験は、我々のアプローチが競争力のある定量的および定性的結果を達成できることを示しています。

# ■ 結論

## 5. Conclusion

- In this work, we make the first attempt to investigate the multi-pose guided virtual try-on system, which enables clothes transferred onto a person image under diverse poses.
- We propose a Multi-pose Guided Virtual Try-on Network (MG-VTON) that generates a new person image after fitting the desired clothes into the input image and manipulating human poses.
- Our MG-VTON decomposes the virtual try-on task into three stages, incorporates a human parsing model is to guide the image synthesis, a Warp-GAN learns to synthesize the realistic image by alleviating misalignment caused by diverse pose, and a refinement render recovers the texture details.
- We construct a new dataset for the multi-pose guided virtual try-on task covering person images with more poses and clothes diversity. Extensive experiments demonstrate that our MG-VTON significantly outperforms all state-of-the-art methods both qualitatively and quantitatively with promising performances.
    - この作業では、多様なポーズの下で人物の画像に衣服を移すことを可能にするマルチポーズガイド付き仮想試着システムを調査する最初の試みを行います。
    - 入力画像に希望の衣服をフィットさせ、人間のポーズを操作した後、新しい人物画像を生成するマルチポーズガイド付き仮想試着ネットワーク（MG-VTON）を提案します。 
    - MG-VTONは仮想試着タスクを3段階に分解し、人間の解析モデルを組み込んで画像合成をガイドし、Warp-GANは多様なポーズに起因するミスアライメントを軽減することで現実的な画像を合成することを学習し、洗練されたレンダリングがテクスチャの詳細を回復します。 
    - より多くのポーズと衣服の多様性を持つ人物の画像をカバーする、マルチポーズガイド付き仮想試着タスクの新しいデータセットを構築します。 広範な実験により、当社のMG-VTONは、有望なパフォーマンスで質的にも定量的にもすべての最先端の方法よりも大幅に優れていることが実証されています。

# ■ 何をしたか？詳細

## 3. MG-VTON

- We propose a novel Multi-pose Guided Virtual Try-on Network (MG-VTON) that learns to synthesize the new person image for virtual try-on by manipulating both clothes and pose. Given an input person image, a desired clothes, and a desired pose, the proposed MG-VTON aims to produce a new image of the person wearing the desired clothes and manipulating the pose. Inspired by the coarse-to-fine idea [8, 17], we adopt an outline-coarse-fine strategy that divides this task into three subtasks, including the conditional parsing learning, the Warp-GAN, and the refinement render. The Figure 2 illustrates the overview of MG-VTON.
    - 衣服とポーズの両方を操作することにより、仮想試着用の新しい人物画像を合成することを学習する新しいマルチポーズガイド付き仮想試着ネットワーク（MG-VTON）を提案します。 入力された人物の画像、希望の服、および希望のポーズが与えられると、提案されたMG-VTONは、希望の服を着てポーズを操作する人物の新しい画像を生成することを目的としています。 粗から細へのアイデア[8、17]から着想を得て、このタスクを条件付き解析学習、Warp-GAN、リファインメントレンダリングを含む3つのサブタスクに分割するアウトライン-粗-細戦略を採用します。 図2は、MG-VTONの概要を示しています。

---

![image](https://user-images.githubusercontent.com/25688193/63220630-f505e380-c1c6-11e9-8db5-a178b491a149.png)

- > Figure 2. The overview of the proposed MG-VTON. Stage I: We first decompose the reference image into three binary masks. Then, we concatenate them with the target clothes and target pose as an input of the conditional parsing network to predict human parsing map. Stage II: Next, we warp clothes, remove the clothing from the reference image, and concatenate them with the target pose and synthesized parsing to synthesize the coarse result by using Warp-GAN. Stage III: We finally refine the coarse result with a refinement render, conditioning on the warped clothes, target pose, and the coarse result.
    - 図2.提案されたMG-VTONの概要。 ステージI：最初に参照画像を3つのバイナリマスクに分解します。 次に、それらを条件付き解析ネットワークの入力としてターゲット衣服とターゲットポーズと連結して、人間の解析マップを予測します。 ステージII：次に、衣服をワープし、参照画像から衣服を削除し、それらをターゲットポーズおよび合成解析と連結して、Warp-GANを使用して粗い結果を合成します。 ステージIII：最後に、洗練されたレンダリング、ゆがんだ衣服のコンディショニング、ターゲットポーズ、および粗い結果で、粗い結果を改良します。

---

- We first apply the pose estimator [4] to estimate the pose. Then, we encode the pose as 18 heatmaps, which is filled with ones in a circle with radius 4 pixels and zeros else- where. A human parser [6] is used to predict the human segmentation maps, consisting of 20 labels, from which we extract the binary mask of the face, the hair, and the shape of the body. Following VITON [8], we downsample the shape of the body to a lower resolution (16 × 12) and directly resize it to the original resolution (256×192), which alleviates the artifacts caused by the variety of the body shape.
    - まず、姿勢推定器[4]を適用して姿勢を推定します。 次に、ポーズを18個のヒートマップとしてエンコードします。ヒートマップは、半径4ピクセルの円で囲まれ、それ以外の場合はゼロで埋められます。 人間のパーサー[6]を使用して、20個のラベルで構成される人間のセグメンテーションマップを予測し、そこから顔、髪、および体の形状のバイナリマスクを抽出します。 VITON [8]に続いて、体の形状を低解像度（16×12）にダウンサンプリングし、元の解像度（256×192）に直接サイズ変更します。これにより、体のさまざまな形状に起因するアーティファクトが軽減されます。

---

![image](https://user-images.githubusercontent.com/25688193/63222550-8041a200-c1e4-11e9-9dd4-5ad01abbaa08.png)

- > Figure 3. The network architecture of the proposed MG-VTON. (a)(b): The conditional parsing learning module consists of a pose-clothes-guided network that predicts the human parsing, which helps to generate high-quality person image. (c)(d): The Warp-GAN learns to generate the realistic image by using a warping features strategy due to the misalignment caused by the diversity of pose. (e): The refinement render network learns the pose-guided composition mask that enhances the visual quality of the synthesized image. (f): The geometric matching network learns to estimate the transformation mapping conditioned on the body shape and clothes mask.
    - > 図3.提案されたMG-VTONのネットワークアーキテクチャ。 
    - > （a）（b）：条件付き解析学習モジュールは、高品質の人物画像の生成に役立つ、人間の解析を予測するポーズ服ガイド付きネットワークで構成されています。
    - > （c）（d）：Warp-GANは、ポーズの多様性によって生じる位置ずれのために、ワープフィーチャ戦略を使用して現実的な画像を生成することを学習します。
    - > （e）：リファインメントレンダリングネットワークは、合成画像の視覚的品質を向上させるポーズガイド付き合成マスクを学習します。
    - > （f）：幾何学的マッチングネットワークは、身体の形状と衣服のマスクを条件とする変換マッピングを推定することを学習します。

### 3.1. Conditional Parsing Learning

- To preserve the structural coherence of the person while manipulating both clothes and the pose, we design a pose-clothes-guided human parsing network, conditioned on the image of clothes, the pose heatmap, the approximated shape of the body, the mask of the face, and the mask of hair. As shown in Figure 4, the baseline methods failed to preserve some parts of the person (e.g, the color of the trousers and the style of hair were replaced.), due to feeding the image of the person and clothes into the model directly. In this work, we leverage the human parsing maps to address those problems, which can help generator to synthesize the high- quality image on parts-level.
    - 衣服とポーズの両方を操作しながら人の構造的一貫性を維持するために、衣服の画像・ポーズヒートマップ・身体の近似形状・顔のマスク・髪のマスクで条件づけされた、ポーズ服ガイド付きネットワークを設計した。
    - 図4に示すように、ベースラインメソッドは、人物の画像とモデルを直接モデルにフィードするため、人物の一部を保持できませんでした（たとえば、ズボンの色や髪のスタイルが置き換えられました）。 この作業では、人間の解析マップを活用してこれらの問題に対処します。これにより、ジェネレーターは部品レベルで高品質の画像を合成できます。

---

- Formally, given an input image of person I, an input image of clothes C, and the target pose P, this stage learns to predict the human parsing map St conditioned on clothes C and the pose P. As shown in Figure 3 (a), we first extract the hair mask Mh, the face mask Mf , the body shape Mb, and the target pose P by using a human parser [6] and a pose estimator [4], respectively. We then concatenate them with the image of clothes as input which is fed into the conditional parsing network.
    - 正式には、人物Iの入力画像、衣服Cの入力画像、およびターゲットポーズPが与えられると、この段階では、衣服CおよびポーズPを条件とする人間の解析マップStの予測が学習されます。図3（a）に示すように 、まず、人間のパーサー[6]と姿勢推定器[4]を使用して、それぞれヘアマスクMh、顔マスクMf、体型Mb、およびターゲットポーズPを抽出します。 次に、それらを入力として衣服の画像と連結し、条件付き解析ネットワークに送ります。

- The inference of S'_t can be formulate as maximizing the posterior probability p(S'_t|(Mh,Mf,Mb,C,P)). Furthermore, this stage is based on the conditional generative adversarial network (CGAN) [19] which generates promising results on image manipulating. Thus, the poster probability p(S'_t|(Mh,Mf,Mb,C,P)) is expressed as:
    - S'_tの推論は、事後確率p（S'_t |（Mh、Mf、Mb、C、P））を最大化するように定式化できます。 さらに、この段階は条件付き生成的敵対ネットワーク（CGAN）[19]に基づいており、画像操作に関する有望な結果を生成します。 したがって、ポスター確率p（S'_t |（Mh、Mf、Mb、C、P））は次のように表されます。

![image](https://user-images.githubusercontent.com/25688193/63220705-419dee80-c1c8-11e9-9dd0-0182ca27ef78.png)

- We adopt a ResNet-like network as the generator G to build the conditional parsing model. We adopt the discriminator D directly from the pix2pixHD [30]. We apply the L1 loss for further improving the performance, which is advantageous for generating more smooth results [32]. Inspired by the LIP [6], we apply the pixel-wise softmax loss to encourage the generator to synthesize high-quality human parsing maps. Therefore, the problem of conditional parsing learning can be formulated as:
    - ジェネレーターGとしてResNetのようなネットワークを採用して、条件付き解析モデルを構築します。 pix2pixHD [30]から弁別器Dを直接採用します。 パフォーマンスをさらに改善するためにL1損失を適用します。これは、よりスムーズな結果を生成するのに有利です[32]。 LIP [6]に着想を得て、ピクセル単位のソフトマックス損失を適用して、ジェネレーターが高品質の人間解析マップを合成するようにします。 したがって、条件付き解析学習の問題は次のように定式化できます。

![image](https://user-images.githubusercontent.com/25688193/63222889-4ffc0280-c1e8-11e9-96c9-8e8c8238e284.png)

- where M denotes the concatenation of Mh , Mf , and Mb . The loss Lparsing denotes the pixel-wise softmax loss [6]. The St denotes the ground truth human parsing. The pdata represents the distributions of the real data.
    - ここで、MはMh、Mf、およびMbの連結を示します。 損失Lparsingは、ピクセル単位のソフトマックス損失を示します[6]。 Stは、グラウンドトゥルースの人間による解析を示します。 pdataは、実際のデータの分布を表します。

### 3.2. Warp-GAN

- Since the misalignment of pixels would lead to generate the blurry results [27], we introduce a deep Warping Generative Adversarial Network (Warp-GAN) warps the desired clothes appearance into the synthesized human parsing map, which alleviates the misalignment problem between the input human pose and desired human pose.
- Different from deformableGANs [27] and [1], we warp the feature map from the bottleneck layer by using both the affine and TPS (Thin-Plate Spline) [3] transformation rather than process the pixel directly by using affine only. Thanks to the generalization capacity of [23], we directly use the pre-trained model of [23] to estimate the transformation mapping between the reference parsing and the synthesized parsing. We then warp the w/o clothes reference image by using this transformation mapping.
    - ピクセルのズレはぼやけた結果を生成するため[27]、深いワーピング生成的敵対ネットワーク（Warp-GAN）を導入して、合成された人間の解析マップに目的の衣服の外観をワープし、入力ポーズと望ましい人間のポーズとの間のズレの問題を軽減します。  
    - DeformableGAN [27]および[1]とは異なり、アフィンのみを使用してピクセルを直接処理するのではなく、アフィン変換とTPS（Thin-Plate Spline）[3]変換の両方を使用して、ボトルネックレイヤーからフィーチャマップをワープします。 [23]の一般化容量のおかげで、[23]の事前学習済みモデルを直接使用して、参照解析と合成解析間の変換マッピングを推定します。 次に、この変換マッピングを使用して、衣服なしの参照画像をワープします。

---

- As illustrated in Figure 3 (c) and (d), the proposed deep warping network consists of the Warp-GAN generator Gwarp and the Warp-GAN discriminator Dwarp. We use the geometric matching module to warp clothes image, as described in the section 3.4.
    - 図3（c）および（d）に示すように、提案されているディープワーピングネットワークは、Warp-GANジェネレーターGwarpおよびWarp-GAN弁別器Dwarpで構成されています。 セクション3.4で説明されているように、幾何学的マッチングモジュールを使用して衣服の画像をゆがめます。

- Formally, we take warped clothes image Cw, w/o clothes reference image Iw/o clothes, the target pose P, and the synthesized human parsing St as input of the Warp-GAN generator and synthesize the result Iˆ = Gwarp(Cw, Iw/o clothes, P, St′ ).
    - 形式的には、ワープ衣服画像Cw、衣服なし参照画像Iw / o衣服、ターゲットポーズP、および合成された人間の解析StをWarp-GANジェネレーターの入力として取得し、結果を合成しますIˆ = Gwarp（Cw、Iw / o服、P、St ′）。

- Inspired by [11, 8, 16], we apply a perceptual loss to measure the distances between high-level features in the pre-trained model, which encourages generator to synthesize high-quality and realistic-look images. We formulate the perceptual loss as:
    - [11、8、16]に触発されて、知覚的損失を適用して事前学習済みモデルの高レベルの特徴間の距離を測定します。 知覚的損失は次のように定式化されます。

![image](https://user-images.githubusercontent.com/25688193/63223139-20e79000-c1ec-11e9-9bfb-4d168b74d4e6.png)


- where φi (I) denotes the i-th (i = 0, 1, 2, 3, 4) layer feature map in pre-trained network φ of ground truth image I. We use the pre-trained VGG19 [28] as φ and weightedly sum the L1 norms of last five layer feature maps in φ to represent perceptual losses between images. The αi controls the weight of loss for each layer. In addition, following pixp2pixHD [30], due to the feature map at different scales from different layers of discriminator enhance the performance of image synthesis, we also introduce a feature loss and formulate it as:
    - ここで、φi（I）は、グラウンドトゥルースイメージIの事前学習済みネットワークφのi番目（i = 0、1、2、3、4）のレイヤーフィーチャマップを示します。事前学習済みVGG19 [28]をφとして使用します。 最後の5つのレイヤーフィーチャマップのL1ノルムをφで加重合計して、画像間の知覚損失を表します。 αiは、各層の損失の重みを制御します。 さらに、pixp2pixHD [30]に従って、識別器のさまざまな層からのさまざまな縮尺の特徴マップにより、画像合成のパフォーマンスが向上するため、特徴の損失も導入し、次のように定式化します。

![image](https://user-images.githubusercontent.com/25688193/63223347-dfa4af80-c1ee-11e9-804c-771a381f901d.png)

- where Fi(I) represent the i-th (i = 0, 1, 2) layer feature map of the trained Dwarp. The γi denotes the weight of L1 loss for corresponding layer.
    - ここで、Fi（I）は、訓練されたドワープのi番目（i = 0、1、2）のレイヤーフィーチャマップを表します。 γiは、対応するレイヤーのL1損失の重みを示します。

---

- Furthermore, we also apply the adversarial loss Ladv [7, 19] and L1 loss L1 [32] to improve the performance. We design a weight sum losses as the loss of Gwarp, which encourages the Gwarp to synthesize realistic and natural images in different aspects, written as follows:
    - さらに、パフォーマンスを改善するために、敵対的損失Ladv [7、19]およびL1損失L1 [32]も適用します。 重量合計の損失はGwarpの損失として設計します。これにより、Gwarpは、次のように書かれたさまざまな側面で現実的で自然な画像を合成できます。
    
![image](https://user-images.githubusercontent.com/25688193/63223356-f945f700-c1ee-11e9-9a3a-8edfb79c5689.png)

- where λi (i = 1, 2, 3, 4) denotes the weight of corresponding loss, respectively.


### 3.3. Refinement render

- In the coarse stage, the identification information and the shape of the person can be preserve, but the texture details are lost due to the complexity of the clothes image. Pasting the warped clothes onto the target person directly may lead to generate the artifacts. Learning the composition mask between the warped clothes image and the coarse results also generates the artifacts [8, 29] due to the diversity of pose. To solve the above issues, we present a refinement render utilizing multi-pose composition masks to recover the texture details and remove some artifacts.
    - 粗い段階では、人物の識別情報と形状は保持できますが、衣服の画像が複雑であるためテクスチャの詳細は失われます。 歪んだ衣服を対象者に直接貼り付けると、アーティファクトが生成される場合があります。 ゆがんだ衣服の画像と粗い結果の間の合成マスクを学習すると、ポーズの多様性に起因するアーティファクト[8、29]も生成されます。 上記の問題を解決するために、マルチポーズコンポジションマスクを使用してテクスチャの詳細を回復し、いくつかのアーティファクトを削除する洗練されたレンダリングを提示します。

---

- Formally, we define Cw as an image of warped clothes obtained by geometric matching learning module, Iˆ_c as a coarse result generated by the Warp-GAN, P as the target pose heatmap, and Gp as the generator of the refinement render. As illustrated in Figure 3 (e), taking C , Iˆ, and P as input, the Gp learns to predict a towards multi-pose composition mask and synthesize the rendered result:
    - 正式には、Cwを幾何学的マッチング学習モジュールによって取得されたゆがんだ衣服の画像として定義し、Iˆ_c をWarp-GANによって生成された粗い結果として、Pをターゲットポーズヒートマップとして、Gpを改良レンダリングのジェネレーターとして定義します。 図3（e）に示すように、C、Iˆ、およびPを入力として、Gpはマルチポーズ構成マスクを予測し、レンダリング結果を合成することを学習します。


### 3.4. Geometric matching learning

- Inspired by [23], we adopt the convolutional neural network to learn the transformation parameters, including feature extracting layers, feature matching layers, and the transformation parameters estimating layers. As shown in Figure 3 (f), we take the mask of the clothes image, and the mask of body shape as input which is first passed through the feature extracting layers. Then, we predict the correlation map by using the matching layers. Finally, we apply a regression network to estimate the TPS (Thin-Plate Spline) [3] transformation parameters for the clothes image directly based on the correlation map.
    - [23]に着想を得て、畳み込みニューラルネットワークを採用して、特徴抽出レイヤー、特徴マッチングレイヤー、レイヤーを推定する変換パラメーターなどの変換パラメーターを学習します。 図3（f）に示すように、衣服画像のマスクと、形状抽出レイヤーを最初に通過する入力としての身体形状のマスクを使用します。 次に、一致するレイヤーを使用して相関マップを予測します。 最後に、回帰ネットワークを適用して、相関マップに直接基づいて衣服画像のTPS（Thin-Plate Spline）[3]変換パラメーターを推定します。

---

- Formally, given an input image of clothes C and its mask Cmask, following the stage of conditional parsing learning, we obtain the approximated body shape Mb and the synthesized clothes mask Cˆmask from the synthesized human parsing. This subtask aims to learn the transformation mapping function T with parameter θ for warping the input image of clothes C. Due to the unseen of synthesized clothes but have the synthesized clothes mask, we learn the mapping between the original clothes mask Cmask and the synthesized clothes mask Cˆmask obey body shape Mb. Thus, the objective function of the geometric matching learning can be formulated as:
    - 形式的には、条件付き構文解析学習の段階に続いて、衣服CとそのマスクCmaskの入力画像が与えられると、合成された人間の構文解析から近似体形Mbと合成衣服マスクCˆmaskを取得します。 このサブタスクは、衣服Cの入力画像をワープするためのパラメーターθを持つ変換マッピング関数Tを学習することを目的としています。合成された衣服は見えないが、合成された衣服マスクがあるため、元の衣服マスクCmaskと合成された衣服の間のマッピングを学習します マスクCˆmaskは体型Mbに従います。 したがって、幾何学的マッチング学習の目的関数は次のように定式化できます。


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## 4. Experiments

- In this section, we first make visual comparisons with other methods and then discuss the results quantitatively. We also conduct the human perceptual study and the ablation study, and further train our model on our newly collected dataset MPV test it on the Deepfashion to verify the generation capacity.
    - このセクションでは、最初に他の方法と視覚的に比較してから、結果を定量的に説明します。 また、人間の知覚調査とアブレーション調査を実施し、新しく収集されたデータセットでモデルをさらにトレーニングして、DeepfashionでMPVテストを行い、生成能力を検証します。

### 4.1. Datasets

- Since each person image in the dataset used in VI-TON [8] and CP-VTON [29] only has one fixed pose, we collected the new dataset from the internet, named MPV, which contains 35,687 person images and 13,524 clothes images. Each person image in MPV has different poses. The image is in the resolution of 256 × 192. We extract the 62,780 three-tuples of the same person in the same clothes but with diverse poses. We further divide them into the train set and the test set with 52,236 and 10,544 three-tuples, respectively. Note that we shuffle the test set with different clothes and diverse pose for quality evaluation. DeepFashion [38] only have the pairs of the same person in different poses but do not have the image of clothes. To verify the generalization capacity of the proposed model, we extract 10,000 pairs from DeepFashion, and randomly select clothes image from the test set of the MPV for testing.
    - VI-TON [8]およびCP-VTON [29]で使用されるデータセットの各人物の画像には1つの固定ポーズしかないため、インターネットからMPVという名前の新しいデータセットを収集しました。 MPVの各人物の画像にはさまざまなポーズがあります。 画像の解像度は256×192です。同じ服を着て、さまざまなポーズの同じ人物の62,780個の3タプルを抽出します。 さらに、それぞれ52,236および10,544個の3タプルを含むトレインセットとテストセットに分割します。 品質評価のために、さまざまな衣服とさまざまなポーズでテストセットをシャッフルすることに注意してください。 DeepFashion [38]は、同じ人物の異なるポーズのペアのみを持ち、服のイメージは持ちません。 提案されたモデルの一般化容量を検証するために、DeepFashionから10,000ペアを抽出し、テスト用にMPVのテストセットから服の画像をランダムに選択します。

### 4.2. Evaluation Metrics

- We apply three measures to evaluate the proposed model, including subjective and objective metrics: 1) We perform pairwise A/B tests deployed on the Amazon Mechanical Turk (AMT) platform for human perceptual study. 2) we use Structural SIMilarity (SSIM) [31] to measure the similarity between the synthesized image and ground truth image. In this work, we take the target image (the same person wearing the same clothes) as the ground truth image used to compare with the synthesized image for computing SSIM. 3) We use Inception Score (IS) [25] to measure the quality of the generated images, which is a common method to verify the performances for image generation.
    - 主観的および客観的指標を含む提案モデルを評価するために、3つの指標を適用します。 2）構造的類似性（SSIM）[31]を使用して、合成画像とグラウンドトゥルース画像の類似性を測定します。 この作業では、ターゲット画像（同じ服を着ている同じ人物）を、SSIMを計算するための合成画像と比較するために使用されるグラウンドトゥルース画像として取得します。 3）Inception Score（IS）[25]を使用して、生成された画像の品質を測定します。これは、画像生成のパフォーマンスを検証する一般的な方法です。

### 4.3. Implementation Details

#### Setting. 

- We train the conditional parsing network, Warp-GAN, refinement render, and geometric matching network for 200, 15, 5, 35 epochs, respectively, using ADAM optimizer [13], with the batch size of 40, learning rate of 0.0002, β1 = 0.5, β2 = 0.999. We use two NVIDIA Titan XP GPUs and Pytorch platform on Ubuntu 14.04.

#### Architecture.

- As shown in Figure 3, each generator of MG-VTON is a ResNet-like network, which consists of three downsample layers, three upsample layers, and nine residual blocks, each block has three convolutional layers with 3x3 filter kernels followed by the bath-norm layer and Relu activation function. Their number of filters are 64, 128, 256, 512, 512, 512, 512, 512, 512, 512, 512, 512, 256, 128, 64. For the discriminator, we apply the same architecture as pix2pixHD [30], which can handle the feature map in different scale with different layers. Each discriminator contains four downsample layers which include 4x4 kernels, InstanceNorm, and LeakyReLU activation function.

#### 4.4. Baselines

- VITON [8] and CP-VTON [29] are the state-of-the-art image-based virtual try-on method which assumes the pose of the person is fixed. They all used warped clothes image to improve the visual quality, but lack of the ability to generate image under arbitrary poses. In particular, VTION directly applied shape context matching [2] to compute the transformation mapping. CP-VTON borrowed the idea from [23] to estimate the transformation mapping using a convolutional network. To obtain fairness, we first enriched the input of the VITON and CP-VTON by adding the target pose. Then, we retrained the VITON and CP-VTON on MPV dataset with the same splits (train set and test set) as our model.
    - VITON [8]およびCP-VTON [29]は、最新の画像ベースの仮想試着方法であり、人物の姿勢が固定されていることを前提としています。 彼らはすべて、視覚的な品質を向上させるためにゆがんだ衣服の画像を使用していましたが、任意のポーズの下で画像を生成する能力がありませんでした。 特に、VTIONは形状マッピングマッチング[2]を直接適用して、変換マッピングを計算しました。 CP-VTONは、[23]からのアイデアを借用して、畳み込みネットワークを使用した変換マッピングを推定しました。 公平性を得るために、最初にターゲットポーズを追加して、VITONとCP-VTONの入力を強化しました。 次に、モデルと同じ分割（トレーニングセットとテストセット）でMPVデータセットのVITONとCP-VTONを再トレーニングしました。

#### 4.5. Quantitative Results

- We conduct experiments on two benchmarks and compare against two recent related works using two widely used metrics SSIM and IS to verify the performance of the image synthesis, summarized in Table 1, higher scores are better.
    - 2つのベンチマークで実験を行い、2つの広く使用されているメトリクスSSIMとISを使用して2つの最近の関連作品と比較し、表1に要約されている画像合成のパフォーマンスを検証します。

- The results shows that ours proposed methods significantly achieve higher scores and consistently outperform all baselines on both datasets thanks to the cooperation of our conditional parsing generator, Warp-GAN, and the refinement render.
    - 結果は、条件付き解析ジェネレーター、Warp-GAN、および絞り込みレンダリングの協力のおかげで、提案された方法が両方のデータセットで高いスコアを達成し、すべてのベースラインを一貫して上回ることを示しています。

- Note that the MG-VTON (w/o Render) achieves the best SSIM score and the MG-VTON (w/o Mask) achieve the best IS score, but they obtain worse visual quality results and achieve lower scores in AMT study compare with MG-VTON (ours), as illustrated in the Table 2 and Figure 6.
    - MG-VTON（レンダリングなし）は最高のSSIMスコアを達成し、MG-VTON（マスクなし）は最高のISスコアを達成しますが、AMTスタディではより悪い視覚品質の結果を取得し、より低いスコアを達成することに注意してください。

- As shown in Figure 4, MG-VTON (ours) synthesizes more realistic-looking results than MG-VTON (w/o Render), but the latter achieve higher SSIM score, which also can be observed in [11]. Hence, we believe that the proposed MG-VTON can generate high-quality person image for multipose virtural try-on with convincing results.
    - MG-VTON（私たち）は、表2および図6に示されています。図4に示すように、MG-VTON（私たち）はMG-VTON（レンダリングなし）よりも現実的な結果を合成しますが、後者はより高いSSIMスコア。これも[11]で観察できます。したがって、提案されたMG-VTONは、説得力のある結果を伴う多目的の仮想試着のために高品質の人物画像を生成できると考えています。


# ■ 関連研究（他の手法との違い）

## x. Related Work


