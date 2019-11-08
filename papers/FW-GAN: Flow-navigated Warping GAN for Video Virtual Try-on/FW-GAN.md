# ■ 論文
- 論文タイトル："FW-GAN: Flow-navigated Warping GAN for Video Virtual Try-on"
- 論文リンク：
- 論文投稿日付：
- 被引用数（記事作成時点）：xxx 件
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- Beyond current image-based virtual try-on systems that have attracted increasing attention, we move a step forward to developing a video virtual try-on system that precisely transfers clothes onto the person and generates visually realistic videos conditioned on arbitrary poses.
    - 注目を集めている現在の画像ベースの仮想試着システムを超えて、服を人に正確に転送し、任意のポーズを条件とする視覚的にリアルなビデオを生成するビデオ仮想試着システムの開発に一歩前進します。

- Besides the challenges in image-based virtual try-on (e.g., clothes fidelity, image synthesis), video virtual try-on further requires spatiotemporal consistency. Directly adopting existing image-based approaches often fails to generate coherent video with natural and realistic textures. In this work, we propose Flow-navigated Warping Generative Adversarial Network (FW-GAN), a novel framework that learns to synthesize the video of virtual try-on based on a person im- age, the desired clothes image, and a series of target poses. FW-GAN aims to synthesize the coherent and natural video while manipulating the pose and clothes.
    - 画像ベースの仮想試着（例：服の忠実度、画像合成）の課題に加えて、ビデオ仮想試着にはさらに時空間的な一貫性が必要です。 既存の画像ベースのアプローチを直接採用すると、多くの場合、自然で現実的なテクスチャを備えたコヒーレントビデオを生成できません。 この作業では、フローナビゲーションワーピング生成敵対ネットワーク（FW-GAN）を提案します。これは、人物の画像、目的の服の画像、および一連の ターゲットポーズ。 FW-GANは、ポーズと衣服を操作しながら、一貫した自然なビデオを合成することを目的としています。

- It consists of: (i) a flow-guided fusion module that warps the past frames to assist synthesis, which is also adopted in the discriminator to help enhance the coherence and quality of the synthesized video; (ii) a warping net that is designed to warp clothes image for the refinement of clothes textures; (iii) a parsing constraint loss that alleviates the problem caused by the misalignment of segmentation maps from images with different poses and various clothes. Experiments on our newly collected dataset show that FW-GAN can synthesize high-quality video of virtual try-on and significantly outperforms other methods both qualitatively and quantitatively.
    - （i）合成を支援するために過去のフレームをワープするフローガイド型フュージョンモジュール。これも弁別器で採用され、合成ビデオの一貫性と品質を向上させます。 （ii）衣服の質感を改善するために衣服のイメージを反らせるように設計されたワーピングネット。 （iii）さまざまなポーズとさまざまな衣服の画像からのセグメンテーションマップの不整合によって引き起こされる問題を軽減する解析制約の損失。 新しく収集されたデータセットの実験は、FW-GANが仮想試着の高品質ビデオを合成でき、定性的および定量的に他の方法よりも大幅に優れていることを示しています。

# ■ イントロダクション（何をしたいか？）

## x. Introduction

- To address the above mentioned challenges, we propose an FW-GAN to achieve the controllable video synthesis for virtual try-on, by manipulating both different poses and various clothes. FW-GAN consists of three main components: 

- 1) a "flow-navigated module" that enforces the synthesized video to be spatiotemporal coherent and high-quality visual; 
    - 合成されたビデオを時空間的に一貫性のある高品質のビジュアルにするための "flow-navigated module"。

- 2) a "warping net" adapted to estimate the grid of transformation parameters that warps the desired clothes in order to fit the corresponding region of the person image; 
    - 人物画像の対応する領域に適合するために、所望の衣服をゆがめる変換パラメータのグリッドを推定するように適合された「ゆがみネット」。

- 3) a "human parsing constraint loss" that constrains body layouts to enforce consistency from a global view.
    - 人体レイアウトを制約して大域的観点から一貫性を強制する"human parsing constraint loss"。

- In particular, the optical flow [2] plays a critical role in the proposed FW-GAN for making the generated videos coherent, which warps the pixel of the preceding frames to the new frames, and is also used as the conditioned input of the "flow-embedding discriminator", resulting in more photo-realistic frames and spatiotemporal smoothing videos.
    - 特に、オプティカルフロー[2]は、生成されたビデオをコヒーレントにするために提案されたFW-GANで重要な役割を果たします。これは、前のフレームのピクセルを新しいフレームにワープし、「 フロー埋め込み弁別器」の条件づけされた入力としても使用され、これにより、より写真のようにリアルなフレームと時空間平滑化ビデオが得られます。

- Besides, to preserve the details of the desired clothes, a weight mask is leveraged to adaptively select the pixel values from the warped desired clothes or synthesized clothes.
    - その上、所望の衣服の詳細を保持するために、重みマスクが活用され、歪んだ所望の衣服または合成された衣服からピクセル値を適応的に選択する。

---

- To generate high-quality synthesized video of virtual try-on under a sequence of poses, a person image, and the desired clothes, we propose an FW-GAN to incorporate the optical flow with warping net for warping the frames and clothes images, respectively, which can preserve the details in global and local views.
    - 一連のポーズ、人物画像、および希望の衣服の下での仮想試着の高品質な合成ビデオを生成するために、フレームと衣服の画像をそれぞれワープするためのワーピングネットとオプティカルフローを組み込むFW-GANを提案します 、グローバルビューとローカルビューで詳細を保持できます。

- A flow-embedding discriminator is proposed that incorporate an effective flow input to the discriminator to improve the spatiotemporal smoothing.
    - 時空間平滑化を改善するために、効果的なフロー入力を弁別器に組み込むフロー埋め込み弁別器が提案されています。

- We employ a parsing constraint loss function as one form of structural constraints to explicitly encourage the model to synthesize results under difference poses and various clothes to produce coherent part configurations with the input image.
    - 構造的制約の1つの形式として parsing constraint loss function を使用して、モデルが異なるポーズとさまざまな衣服の下で結果を合成し、入力画像でコヒーレントパーツ構成を生成することを明示的に奨励します。


# ■ 結論

## x. Conclusion


# ■ 何をしたか？詳細

## x. 論文の項目名


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. Related Work


