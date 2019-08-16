# ■ 論文
- 論文タイトル："Authoring New Haptic Textures Based on Interpolation of Real Textures in Affective Space"
- 論文リンク：
- 論文投稿日付：2019/03/08
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- This paper presents a novel haptic texture authoring algorithm. The main goal of this algorithm is to synthesize new virtual textures by manipulating the affective properties of already existing real life textures. 
    - この論文では、新しい触覚テクスチャオーサリング（＝異種データの組み合わせ、編集）アルゴリズムを紹介します。このアルゴリズムの主な目標は、既存の実際のテクスチャの感情的な [affective] 特性を操作することにより、新しい仮想テクスチャを合成することです。

- To this end, two different spaces are established: two-dimensional “affective space” built from a series of psychophysical experiments where real textures are arranged according to affective properties (hard-soft, rough-smooth) and two- dimensional “haptic model space” where real textures are placed based on features from tool-surface contact acceleration patterns (movement-velocity, normal-force). Another space, called “authoring space” is formed to merge the two spaces; correlating changes in affective properties of real life textures to changes in actual haptic signals in haptic space. The authoring space is constructed such that features of the haptic model space that were highly correlated with affective space become axes of the space.
    - このために、2つの異なる空間が確立されます。一連の心理物理実験から構築された2次元の「感情空間」で、実際のテクスチャが感情特性（ハードソフトソフト、ラフスムース）に従って配置され、2次元の「ハプティックモデル空間」 」では、ツールと表面の接触加速パターン（運動速度、垂直力）の特徴に基づいて実際のテクスチャが配置されます。 「オーサリングスペース」と呼ばれる別のスペースは、2つのスペースをマージするために形成されます。実生活のテクスチャの感情特性の変化を、触覚空間の実際の触覚信号の変化に相関させる。オーサリング空間は、感情空間と高度に相関した触覚モデル空間の特徴が空間の軸になるように構築されます。その結果、オーサリング空間の任意のポイントに対応する新しいテクスチャ信号を、知覚的に正しい方法で3つの最も近い実際の表面の加重補間に基づいて合成できます。

- As a result, new texture signals corresponding to any point in authoring space can be synthesized based on weighted interpolation of three nearest real surfaces in perceptually correct manner. The whole procedure including the selection of nearest surfaces, finding weights, and weighted interpolation of multiple texture signals are evaluated through a psychophysical experiment, demonstrating the competence of the approach. The results of evaluation experiment show an average normalized realism score of 94 % for all authored textures.
    - 最も近い表面の選択、重みの検出、および複数のテクスチャ信号の重み付き補間を含む手順全体が、心理物理実験を通じて評価され、アプローチの能力が実証されています。評価実験の結果は、すべてのオーサリングされたテクスチャの平均正規化リアリズムスコアが94％であることを示しています。

# ■ イントロダクション（何をしたいか？）

## I. INTRODUCTION

- xxx

---

- In data-driven modeling, the vibrations originating from interaction with different surfaces are recorded and are sub-sequently used for rendering tactile contents. For instance, the authors in [2] were able to generate virtually perceptible textures based on the scanning velocity and normal force. Similarly, Abdulali et al. extended this idea to recreate more complex textures (anisotropic textures) by incorporating the direction of scan velocity into the equation [14].
    - データ駆動型モデリングでは、異なる表面との相互作用に起因する振動が記録され、触覚コンテンツのレンダリングに引き続き使用されます。たとえば、[2]の著者は、走査速度と法線力に基づいて、実質的に知覚可能なテクスチャを生成できました。同様に、Abdulali等。このアイデアを拡張して、スキャン速度の方向を方程式に組み込むことで、より複雑なテクスチャ（異方性テクスチャ）を再作成しました[14]。

- Recently, a more robust and efficient technique has been employed where Generative Adversarial Networks (GANs) have been trained to create vibrotactile signals based on texture images or attributes albeit using predefined and constrained tool-surface interaction [15]. The upside of data-driven modeling is that the created contents are highly realistic and computationally simpler. However, the recorded model is an arbitrary signal having no physical meaning and is hard to modify meaningfully. This indicates that the number of feedbacks that a designer can generate are limited. In addition, it is impossible to create contents that are not physically available, and model building is a highly time consuming process.
    - **最近、Generative Adversarial Networks（GAN）がトレーニング済みのツールとサーフェスの相互作用を使用しながら、テクスチャ画像または属性に基づいて振動触覚信号を作成するようにトレーニングされた、より堅牢で効率的な手法が採用されました[15]。データ駆動型モデリングの利点は、作成されたコンテンツが非常に現実的で、計算が簡単になることです。ただし、記録されたモデルは物理的な意味を持たない任意の信号であり、意味のある変更は困難です。これは、設計者が生成できるフィードバックの数が制限されていることを示しています。さらに、物理的に利用できないコンテンツを作成することは不可能であり、モデルの構築は非常に時間のかかるプロセスです。**

- In summary, on one hand, the physics based models do not guarantee a high level of realism but can be controlled easily. On the other hand, data-driven models ensure higher realism with limited controllability and authoring power. Increasing realism of physics based approaches generally come at a very high computational cost, which often make a system non-practical. Instead, it seems that more feasible solution would be to keep the data-driven approach and to focus on improving controllability of data-driven models.
    - 要約すると、一方で、物理ベースのモデルは、高レベルのリアリズムを保証するものではなく、簡単に制御できます。一方、データ駆動型モデルは、限られた制御性とオーサリング機能でより高いリアリズムを保証します。物理ベースのアプローチのリアリズムを高めると、一般に非常に高い計算コストがかかり、システムが実用的でなくなることがよくあります。**代わりに、より実行可能な解決策は、データ駆動型のアプローチを維持し、データ駆動型モデルの制御性の改善に集中することです。**

---

- The goal of this paper is to provide an effective method for haptic texture authoring using data-driven haptic texture modeling. We achieve this goal through two contributions. We first established an authoring space where 25 data-driven texture models build from 25 fully featured real surfaces are placed according to their affective properties. The space is made in such a way that it maximizes the correspondence between affective properties of the 25 models and features in the physical signals of the models. Axes of the space are the affective properties, and this space plays a role as a perception- based descriptor of textures. Now, designers can freely select an arbitrary point in the space to author a texture, and then the system automatically synthesizes new texture signal corresponding to the selected affective properties. Our second contribution lies in this part. Our framework interpolates signals from adjacent data-driven models, so that two different haptic models are combined to form the new virtual texture. This step ensures that the new model inherits perceptual characteristics of the parent textures, allowing the aforementioned authoring scenarios. To the best of our knowledge, there is no such work which provides the approximation of physical properties across two different texture models.
    - **このペーパーの目標は、データ駆動型の触覚テクスチャモデリングを使用した触覚テクスチャオーサリングの効果的な方法を提供することです。**この目標は2つの貢献を通じて達成されます。最初に、25の完全な機能を備えた実際のサーフェスから構築された25のデータ駆動型テクスチャモデルが、それらの感情特性に従って配置されるオーサリングスペースを確立しました。スペースは、25個のモデルの感情特性とモデルの物理信号の特徴との間の対応を最大化するような方法で作成されます。空間の軸は感情的なプロパティであり、この空間はテクスチャの知覚ベースの記述子としての役割を果たします。現在、設計者は空間内の任意のポイントを自由に選択してテクスチャを作成でき、システムは選択された効果的なプロパティに対応する新しいテクスチャ信号を自動的に合成します。 2番目の貢献はこの部分にあります。このフレームワークは、隣接するデータ駆動型モデルからの信号を補間するため、2つの異なる触覚モデルが組み合わされて、新しい仮想テクスチャが形成されます。この手順により、新しいモデルが親テクスチャの知覚特性を継承し、前述のオーサリングシナリオが可能になります。私たちの知る限り、2つの異なるテクスチャモデル全体の物理的特性の近似値を提供するような作業はありません。

---

- The significance of this work can be explained through an analogy from the field of vision. It is well known that the RGB space can be used to create most of the colors perceivable to human eye. Image editing tools often provide an RGB color table where a designer can easily select a color to be used. In a similar way, through this work, we want to provide a unified haptic authoring tool comprising of the basic components or dimensions of haptic texture. Such a tool can be utilized by designers and researchers to create haptic models having arbitrary affective properties and would drastically reduce the time and effort required for haptic modeling.
    - この作業の重要性は、視野からの類推によって説明できます。 RGBスペースを使用して、人間の目に知覚できるほとんどの色を作成できることはよく知られています。 画像編集ツールは、多くの場合、デザイナーが使用する色を簡単に選択できるRGBカラーテーブルを提供します。 同様に、この作業を通じて、ハプティックテクスチャの基本的なコンポーネントまたは寸法で構成される統一されたハプティックオーサリングツールを提供したいと考えています。 **このようなツールは、設計者や研究者が任意の感情特性を持つ触覚モデルを作成するために利用でき、触覚モデリングに必要な時間と労力を大幅に削減します。**


# ■ 結論

## VIII. CONCLUSIONS

- In this paper, we provide a novel algorithm for haptic texture authoring. The affective properties of real life textures are manipulated to create virtual textures exhibiting predefined affective properties by using contact acceleration patterns. This algorithm finds great application in the realm of virtual reality, where on demand textures are need of the hour. More specifically, it can provide virtual textures as a combination of various real life textures.
    - この論文では、触覚テクスチャのオーサリングのための新しいアルゴリズムを提供します。 実際のテクスチャの感情特性は、接触加速パターンを使用して、事前定義された感情特性を示す仮想テクスチャを作成するために操作されます。 このアルゴリズムは、オンデマンドテクスチャが1時間必要な仮想現実の領域で優れたアプリケーションを見つけます。 より具体的には、仮想テクスチャをさまざまな実際のテクスチャの組み合わせとして提供できます。


# ■ 何をしたか？詳細

## II. SYSTEM OVERVIEW

- Figure 1 presents a holistic view of the overall system, while Algorithm 1 provides the flow of the system. The methods used to approach our goal are detailed in the following sections.
    - 図1はシステム全体の全体像を示し、アルゴリズム1はシステムの流れを示しています。 目標に近づくために使用される方法については、次のセクションで詳しく説明します。

---

- The current study aims at providing a platform that can manipulate existing data-driven haptic textures in a perceptually meaningful manner. Since a data-driven model is just a recording of haptic-related signals, it is not a trivial task to find a connection between a certain modification in signals and its perceptual result, and vice versa. This relationship is essential for our goal. The paper first tries to establish this relationship. To this end, we first build an affective space where 25 data- driven models are scattered in a two dimensional space defined by two perception-based affective properties (see Sect. III). Another space called haptic model space is built from the multi-dimensional features extracted from the acceleration signals of the same 25 data-driven models (see Sect. IV).
    - 現在の研究の目的は、知覚的に意味のある方法で既存のデータ駆動型触覚テクスチャを操作できるプラットフォームを提供することです。 データ駆動型モデルは触覚関連信号の単なる記録であるため、信号の特定の変更とその知覚結果との間の接続を見つけることは簡単な作業ではありません。 この関係は私たちの目標にとって不可欠です。 論文は最初にこの関係を確立しようとします。 この目的のために、最初に25個のデータ駆動モデルが2つの知覚ベースの感情特性によって定義される2次元空間に散らばる感情空間を構築します（セクションIIIを参照）。 ハプティックモデル空間と呼ばれる別の空間は、同じ25個のデータ駆動モデルの加速度信号から抽出された多次元の特徴から構築されます（セクションIVを参照）。

---

- Now, the two spaces are merged based on the correlation between them, yielding an authoring space (see Sect. V). The main characteristic of the authoring space is that all textures are scattered in this space as a function of their affective properties while also maintaining their connection to the physical acceleration patterns. Each point in affective space is linked with a corresponding acceleration pattern, and a change in affective values is instantly reflected in the acceleration patterns.
    - 現在、2つのスペースは、それらの間の相関に基づいてマージされ、オーサリングスペースが生成されます（セクションVを参照）。 オーサリング空間の主な特徴は、物理的な加速パターンへの接続を維持しながら、すべてのテクスチャが感情特性の関数としてこの空間に散在していることです。 感情空間の各ポイントは対応する加速パターンにリンクされており、感情値の変化は即座に加速パターンに反映されます。

---

- However, we only have acceleration patterns for a few points (the points where 25 real surfaces are located) in the authoring space. Therefore, interpolation is carried out to generate acceleration patterns for any arbitrary point within the convex hull of real surfaces in the authoring space (see Sect. VI). In order to do this in a perceptually correct manner, we did a time-domain acceleration signal interpolation based on distances to the nearest real samples. This results in a new virtual texture having arbitrary affective properties.
    - ただし、オーサリングスペースのいくつかのポイント（25の実際のサーフェスが配置されているポイント）の加速パターンのみがあります。 したがって、補間は、オーサリング空間の実際の表面の凸包内の任意のポイントの加速パターンを生成するために実行されます（セクションVIを参照）。 知覚的に正しい方法でこれを行うために、最も近い実際のサンプルまでの距離に基づいて、時間領域の加速度信号補間を行いました。 これにより、任意の感情的なプロパティを持つ新しい仮想テクスチャが作成されます。

- Finally, the newly authored virtual textures are evaluated using a psychophysical experiment (see Sect. VII).
    - 最後に、新しく作成された仮想テクスチャは、心理物理実験を使用して評価されます（セクションVIIを参照）。


## III. AFFECTIVE SPACE

- Real life textures used in this study are scattered in the affective space as a function of their affective properties. Two psychophysical experiments are carried out to establish the affective space. The first one is a cluster sorting experiment to form a perceptual space and the second one an adjective rating experiment. The first experiment, with the help of multidimen- sional scaling (MDS), resulted in a two dimensional perceptual space where textures are scattered based on differences in textural perception. The second experiment, called as adjective rating, is carried out to find affective properties that best describe the given textures. These affective properties are in the form of adjective pairs. The adjective pairs are regressed into perceptual space to establish an affective space, and the perceptual space is projected onto each adjective pair.
    - この研究で使用される実際のテクスチャは、感情特性の関数として感情空間に散在しています。 感情空間を確立するために、2つの心理物理実験が実行されます。 1つ目は、知覚空間を形成するクラスターソート実験で、2つ目は形容詞評価実験です。 最初の実験では、多次元スケーリング（MDS）を使用して、テクスチャの知覚の違いに基づいてテクスチャが散在する2次元の知覚空間を作成しました。 形容詞評価と呼ばれる2番目の実験は、与えられたテクスチャを最もよく説明する感情的なプロパティを見つけるために実行されます。 これらの感情的な特性は形容詞のペアの形です。 形容詞のペアは、感情空間を確立するために知覚空間に回帰され、知覚空間は各形容詞のペアに投影されます。

---

- Consequently, we are left with two affective axes (one from each adjective pair) where all surfaces are aligned according to one specific property. Furthermore, the two affective axes are combined to form affective space.
    - その結果、2つの感情軸（各形容詞ペアから1つ）が残り、すべてのサーフェスが1つの特定のプロパティに従って整列されます。 さらに、2つの感情軸が組み合わされて感情空間が形成されます。

## IV. HAPTIC MODEL SPACE

- The model space must be based on the characteristics of physical interaction with surfaces because the psychophysical experiments were also based on physical interaction. Since the model space will be mapped with the affective space, these physical characteristics must be the ones that are perceivable by humans. The most common source of haptic texture perception is the high frequency vibrations (acceleration patterns) originated during interaction with a surface. Hence, we decided to use the acceleration patterns for the haptic model space establishment. Various scanning parameters were also taken into consideration while collecting the acceleration patterns, since different scanning parameters affect the spectral characteristics of the resultant vibration signal [20].
    - 心理物理実験も物理的相互作用に基づいているため、モデル空間は表面との物理的相互作用の特性に基づいている必要があります。 モデル空間は感情空間とマッピングされるため、これらの物理的特性は人間が知覚できるものでなければなりません。 触覚テクスチャ知覚の最も一般的な原因は、表面との相互作用中に発生する高周波振動（加速パターン）です。 したがって、触覚モデル空間の確立には加速パターンを使用することにしました。 加速度パターンを収集する際には、さまざまなスキャンパラメーターも考慮されます。これは、異なるスキャンパラメーターが、結果として生じる振動信号のスペクトル特性に影響を与えるためです[20]。

---

- Since the aim of establishing the model space is to find a relationship between acceleration patterns and affective space, it is important to maintain a controlled environment while scanning textures. Same scanning parameters must be used across all texture models. There are two possible ways to collect such data; use a special machine for data collection; or simulate the signal using very sophisticated haptic modeling and rendering framework that accurately reflects real signals, e.g, data-driven haptic texture modeling and rendering. In [14], authors provided a haptic texture modeling algorithm which showed reasonable performance. More importantly, the authors claim that their models are perceptually sound, therefore, it is decided to build haptic texture models based on [14] and use it to simulate the vibration output for a given combination of input parameters, by using the complementary rendering algorithm [21]. The data acquisition setup used for model building was similar to the one provided in [21].
    - モデル空間を確立する目的は、加速パターンと感情空間の関係を見つけることであるため、テクスチャをスキャンしながら制御された環境を維持することが重要です。すべてのテクスチャモデルで同じスキャンパラメータを使用する必要があります。このようなデータを収集する方法は2つあります。データ収集に特別なマシンを使用します。または、実際の信号を正確に反映する非常に洗練されたハプティックモデリングおよびレンダリングフレームワークを使用して信号をシミュレートします（例：データ駆動型のハプティックテクスチャモデリングおよびレンダリング）。 [14]で、著者は妥当なパフォーマンスを示す触覚テクスチャモデリングアルゴリズムを提供しました。さらに重要なことは、著者は自分のモデルが知覚的に健全であると主張しているため、[14]に基づいて触覚テクスチャモデルを構築し、それを使用して相補レンダリングアルゴリズムを使用して入力パラメーターの特定の組み合わせの振動出力をシミュレートすることを決定した[21]。モデル構築に使用されるデータ収集設定は、[21]で提供されたものと同様でした。

---

- The signal recording time for each texture is 40 seconds. Since all the textures in the current study are isotropic in nature, the directionality of the sample texture is deemed irrelevant. Therefore, the input space for the algorithm provided in [14] is reduced to two-dimensions, i.e., velocity magnitude and normal force. Lastly, 25 response signals are approximated using each texture model with a predefined input vector. The responses resulting from combining each value in the velocity vector (50, 100, 150, 200, 250) with each value in force vector (0.1, 0.2, 0.3, 0.4, 0.5) are approximated.
    - 各テクスチャの信号記録時間は40秒です。 現在のスタディのすべてのテクスチャは本質的に等方性であるため、サンプルテクスチャの方向性は無関係であると見なされます。 したがって、[14]で提供されるアルゴリズムの入力スペースは、2次元、つまり速度の大きさと垂直力に削減されます。 最後に、25個の応答信号が、定義済みの入力ベクトルを持つ各テクスチャモデルを使用して近似されます。 速度ベクトルの各値（50、100、150、200、250）と力ベクトルの各値（0.1、0.2、0.3、0.4、0.5）を組み合わせた結果が近似されます。

---

- After calculating the responses, the 25 acceleration patterns are concatenated together to form a single feature vector for each texture. Employing such a strategy ensured that the signal preserves the delicacies induced due to varying scan parameters. This concatenated signal will be used for feature extraction in the next section.
    - 応答を計算した後、25個の加速パターンを連結して、各テクスチャの単一の特徴ベクトルを形成します。 このような戦略を採用することで、スキャンパラメータの変化によって引き起こされる珍味を信号が確実に保持できます。 この連結された信号は、次のセクションで特徴抽出に使用されます。


## V. AUTHORING SPACE

- The main aim of this work is texture authoring, which means that a change in affective space should be replicated accordingly in the haptic model space. For this purpose, the authoring space is established by combining the two spaces. In authoring space, surfaces are scattered based on their affective properties, while at the same time it carries information about physical properties of the surfaces.
    - この作業の主な目的はテクスチャのオーサリングです。つまり、触覚モデル空間で感情空間の変化を複製する必要があります。 この目的のために、オーサリングスペースは2つのスペースを組み合わせて確立されます。 オーサリング空間では、表面は感情的な特性に基づいて散在しますが、同時に表面の物理的特性に関する情報を保持します。

---

- Physical acceleration signals collected from the modeling of various surfaces carry redundant information in addition to useful haptic information. It is required to distill that information and to represent it in a meaningful and reusable way. Therefore, a feature extraction algorithm is used, called as Mel Frequency Cepstral Coefficients (MFCC) [22].
    - さまざまな表面のモデリングから収集された物理的な加速度信号には、有用な触覚情報に加えて冗長な情報が含まれています。 その情報を抽出し、意味のある再利用可能な方法で表現する必要があります。 そのため、メル周波数ケプストラム係数（MFCC）[22]と呼ばれる特徴抽出アルゴリズムが使用されます。

---

- The MFCC features are used to predict the affective properties calculated in Section III. In order to find out which of the MFCC features can be useful for predicting the individual affective axes, further feature reduction and transformation algorithms were used. Sequential Forward Selection (SFS) [23] and Parallel Analysis (PA) [24] are performed to obtain the MFCC features that are highly correlated with the respective affective axes. Afterwards, Principal Component Analysis (PCA) is applied to further reduce the feature dimension. As a result, we are left with a one-to-one correspondence between the features and affective axes, i.e., one feature representing one affective axis. These two features are combined to form a two dimensional authoring space.
    - MFCC機能は、セクションIIIで計算された感情特性を予測するために使用されます。 どのMFCC機能が個々の感情軸を予測するのに役立つかを見つけるために、さらなる機能削減と変換アルゴリズムが使用されました。 順次前方選択（SFS）[23]および並列分析（PA）[24]を実行して、それぞれの感情軸と高度に相関するMFCC機能を取得します。 その後、主成分分析（PCA）を適用して、特徴の次元をさらに縮小します。 その結果、特徴と感情軸の間には1対1の対応、つまり、1つの感情軸を表す1つの特徴が残ります。 これらの2つの機能を組み合わせて、2次元のオーサリングスペースを形成します。


## VI. HAPTIC RENDERING USING WEIGHTED SYNTHESIZATION

- The three haptic models selected as a result of Delaunay triangulation in the authoring space are combined to author the new texture by weighted synthesization. The weights are calculated, using inverse distance method, from the vertices of the Delaunay triangles (the vertices are the three nearest neighbors). The weighted synthesization is carried out in two steps. In the first step, under the given current interaction parameters (stroking velocity and normal force) three vibration wave forms from the three selected models are virtually generated using the rendering algorithm (see first three signals in all graphs in Fig 7). Note that these signals are not physically rendered but only simulated internally. In the second step, these signals are added together in time domain using the weights associated with them (see the last signal in all graphs in Fig 7). Finally, the synthesized signal is sent to a haptic interface to be rendered. It must be noted that in general such signal synthesization takes place as parametric interpolation in frequency domain [2], [14]. Signal addition is usually carried out in frequency domain since it breaks down the signal into individual frequencies and it is easy to keep track of these frequencies. However, weighted addition in time domain has the same effect on the signal according to [26], and the time domain signal can easily be reconstructed from its Fourier transform. Additionally, superposition of time domain signals was also carried out in [27] to study its effect on the neural system.

# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## VII. EVALUATION


# ■ 関連研究（他の手法との違い）

## x. Related Work


