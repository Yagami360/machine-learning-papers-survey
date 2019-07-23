# ■ 論文
- 論文タイトル："Kervolutional Neural Networks"
- 論文リンク：https://arxiv.org/abs/1904.03955
- 論文投稿日付：2019/04/08
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- Convolutional neural networks (CNNs) have enabled the state-of-the-art performance in many computer vision tasks. However, little effort has been devoted to establishing convolution in non-linear space. Existing works mainly leverage on the activation layers, which can only provide point-wise non-linearity.
    - 畳み込みニューラルネットワーク（ＣＮＮ）は、多くのコンピュータビジョンタスクにおいて最先端の性能を可能にした。 しかしながら、非線形空間で畳み込みを確立することにほとんど努力が払われていない。 既存の研究は主に活性化層を利用しており、活性化層は点ごとの非線形性しか提供できません。

- To solve this problem, a new operation, kervolution (kernel convolution), is introduced to approximate complex behaviors of human perception systems leveraging on the kernel trick. It generalizes convolution, enhances the model capacity, and captures higher order interactions of features, via patch-wise kernel functions, but without introducing additional parameters. Extensive experiments show that kervolutional neural networks (KNN) achieve higher accuracy and faster convergence than baseline CNN.
    - この問題を解決するために、カーネルトリックを利用して人間の知覚システムの複雑な挙動を近似するための新しい操作、カーボリューション（カーネルコンボリューション）が導入された。 それは畳み込みを一般化し、モデル容量を強化し、そしてパッチ的なカーネル関数を介して、しかし追加のパラメータを導入することなく、特徴の高次の相互作用を捉える。 広範囲の実験は、ケルボリューションニューラルネットワーク（KNN）がベースラインCNNよりも高い精度と速い収束を達成することを示しています。

# ■ イントロダクション（何をしたいか？）

## 1. Introduction

- Convolutional neural networks (CNNs) have been tremendously successful in computer vision, e.g. image recognition [31, 20] and object detection [16, 42]. The core operator, convolution, was partially inspired by the animal visual cortex where different neurons respond to stimuli in a restricted and partially overlaped region known as the receptive field [27, 28]. Convolution leverages on its equivariance to translation to improve the performance of a machine learning system [18]. Its efficiency lies in that the learnable parameters are sparse and shared across the entire input (receptive field). Nonetheless, convolution still has certain limitations, which will be analyzed below. To address them, this paper introduces kervolution to generalize convolution via the kernel trick. The artificial neural networks containing kervolutional layers are named as kervolutional neural networks (KNN).
    - 畳み込みニューラルネットワーク（ＣＮＮ）は、コンピュータビジョンにおいて非常に成功している。 画像認識[31、20]と物体検出[16、42]。 中核演算子である畳み込みは、受容野として知られる制限され部分的に重なり合った領域で異なるニューロンが刺激に反応する動物の視覚皮質に部分的に触発された[27、28]。 畳み込みは、機械学習システムの性能を向上させるために、翻訳に対するその同等性を利用します[18]。 その効率は、学習可能なパラメータがまばらであり、入力全体（受容野）にわたって共有されることにあります。 それにもかかわらず、たたみ込みにはまだ特定の制限があり、それについては以下で分析します。 それらに対処するために、本稿ではカーネルトリックによる畳み込みを一般化するための kervolution を紹介します。 ケルボリューション層を含む人工ニューラルネットワークは、kervolution ニューラルネットワーク（ＫＮＮ）と呼ばれる。

---

- There is circumstantial evidence that suggests most cells inside striate cortex1 can be categorized as simple, complex, and hypercomplex, with specific response properties [28].
    - 線状皮質1内の大部分の細胞は、単純な、複雑な、そして超複雑なものとして分類でき、特異的な反応特性を持つことを示唆する状況的証拠があります[28]。

- However, the convolutional layers are linear and designed to mimic the behavior of simple cells in human visual cortex [57], hence they are not able to express the non-linear behaviors of the complex and hypercomplex cells inside the striate cortex. It was also demonstrated that higher order non-linear feature maps are able to make subsequent linear classifiers more discriminative [37, 1, 9]. However, the non-linearity that comes from the activation layers, e.g. rectified linear unit (ReLU) can only provide point-wise non-linearity. We argue that CNN may perform better if convolution can be generalized to patch-wise non-linear operations via kernel trick. Because of the increased expressibility and model capacity, better model generalization may be obtained.
    - しかし、たたみ込み層は線形であり、人間の視覚皮質における単純な細胞の振る舞いを模倣するように設計されている[57]。したがって、それらは線状皮質内部の複雑な細胞および超複雑な細胞の非線形の振る舞いを表現できない。 より高次の非線形特徴マップは、その後の線形分類器をより識別可能にすることができることも実証された［３７、１、９］。 しかしながら、活性化層に起因する非線形性、例えば、 整流リニアユニット（ReLU）は、点ごとの非線形性しか提供できません。 我々は、畳み込みがカーネルトリックを介してパッチ単位の非線形演算に一般化されることができれば、ＣＮＮがより良く機能するかもしれないと主張する。 表現性とモデル容量の向上により、より優れたモデル一般化が得られる可能性があります。

---

- Non-linear generalization is simple in mathematics, however, it is generally difficult to retain the advantages of convolution, i.e. (i) sharing weights (weight sparsity) and (ii) low computational complexity. 
    - 非線形一般化は数学的には簡単であるが、畳み込みの利点、すなわち（ｉ）重みを共有すること（重みスパース性）および（ｉｉ）低い計算複雑性を保持することは一般に困難である。

- There exists several works towards non-linear generalization. The non-linear convolutional networks [57] implement a quadratic convolution at the expense of additional n(n + 1)/2 parameters, where n is the size of the receptive field. However, the quadratic form of convolution loses the property of "weight sparsity", since the number of additional parameters of the non-linear terms increases exponentially with the polynomial order, which dramatically increases the training complexity.
    - 非線形一般化に向けていくつかの研究があります。 非線形たたみ込みネットワーク[57]は、追加のn（n + 1）/ 2パラメータを犠牲にして2次たたみ込みを実装します。ここで、nは受容野のサイズです。 しかしながら、非線形項の追加パラメータの数が多項式次数と共に指数関数的に増加するので、畳み込みの二次形式は「重み疎性」の性質を失い、それはトレーニングの複雑さを劇的に増加させる。

- Another strategy to introduce high order features is to explore the pooling layers. The kernel pooling method in [9] directly concatenates the non-linear terms, while it requires the calculation of non-linear terms, resulting in a higher complexity.
    - 高次機能を導入するためのもう1つの戦略は、プール層を探索することです。 ［９］のカーネルプーリング方法は、非線形項を直接連結するが、非線形項の計算を必要とし、その結果、より複雑になる。

---

- To address the above problems, kervolution is introduced in this paper to extend convolution to kernel space while keeping the aforementioned advantages of linear convolutions. Since convolution has been applied to many fields, e.g. image and signal processing, we expect kervolution will also play an important role in those applications.
    - 上記の問題に対処するために、本論文では、前述の線形畳み込みの利点を維持しながら畳み込みをカーネル空間に拡張するために、kervolution を導入する。 畳み込みは多くの分野に適用されてきたので、 画像処理や信号処理では、これらのアプリケーションでも kervolution が重要な役割を果たすことが予想されます。

- However, in this paper, we focus on its applications in artificial neural networks. The contributions of this paper include:
    - しかし、本稿では、人工ニューラルネットワークにおけるその応用に焦点を当てます。 このホワイトペーパーの貢献は次のとおりです。

- (i) via kernel trick, the convolutional operator is generalized to kervolution, which retains the advantages of convolution and brings new features, including the increased model capacity, translational equivariance, stronger expressibility, and better generalization;
    - (i) カーネルトリックを介して、たたみ込み演算子はkervolutionに一般化されます。これはたたみ込みの利点を保持し、モデル容量の増加、並進等価性、より高い表現可能性、より良い一般化などの新しい機能をもたらします。


- (ii) we provide explanations for kervolutional layers from feature view and show that it is a powerful tool for network construction;
    - (ii) 我々は、特徴的な観点からケルボリューション層についての説明を提供し、それがネットワーク構築のための強力なツールであることを示す。

- (iii) it is demonstrated that KNN achieves better accuracy and surpasses the baseline CNN.
    - (iii) KNNがより高い精度を達成し、ベースラインCNNを上回ることが実証されています。

# ■ 結論

## 7. Conclusion

- This paper introduces the kervolution to generalize convolution to non-linear space, and extends convolutional neural networks to kervolutional neural networks. It is shown that kervolution not only retains the advantages of convolution, i.e sharing weights and equivalence to translation, but also enhances the model capacity and captures higher order interactions of features, via patch-wise kernel functions without introducing additional parameters. It has been demonstrated that, with careful kernel chosen, the performance of CNN can be significantly improved on MNIST, CIFAR, and ImageNet dataset via replacing convolutional layers by kervolutional layers. Due to the large number of choices of kervolution, we cannot perform a brute force search for all the possibilities, while this opens a new space for the construction of deep networks. We expect the introduction of kervolutional layers in more architectures and extensive hyperparameter searches can further improve the performance.
    - 本論文では、畳み込みを非線形空間に一般化するためのケルボリューションを紹介し、畳込みニューラルネットワークをケルボリューションニューラルネットワークに拡張した。 kervolutionは畳み込みの利点、すなわち重みと並進との等価性の共有を保持するだけでなく、追加のパラメータを導入することなくパッチごとのカーネル関数を介してモデル容量を高め、特徴の高次相互作用を捉えることが示された。 慎重に選択されたカーネルを選択することで、畳み込み層をkervolutional層に置き換えることによって、CNISTのパフォーマンスがMNIST、CIFAR、およびImageNetデータセットで大幅に改善されることが実証されました。 kervolutionの選択肢が多数あるため、すべての可能性についてブルートフォース検索を実行することはできませんが、これによってディープネットワークを構築するための新しいスペースが開かれます。 より多くのアーキテクチャーでのケルボリューション層の導入と、広範囲のハイパーパラメーター検索がさらにパフォーマンスを向上させることができると期待しています。
    
# ■ 何をしたか？詳細

## 3. Kervolution

- We start from a convolution with output f (x), i.e.


### 3.1. Sharing Weights

- Sharing weights normally mean less trainable parameters and lower computational complexity. It is straightforward that the number of elements in filter w is not increased according to the definition of (5), thus kervolution keeps the sparse connectivity of convolution.
    - 重みを共有すると、通常、学習可能なパラメータが少なくなり、計算量が少なくなります。 フィルタｗの要素数が式（５）の定義に従って増加されないことは簡単であり、それ故、カーボリューションは畳み込みのスパース接続性を維持する。

- As a comparison, we take the Volterra series-based non-linear convolution adopted in [57] as an example, the additional parameters of non-linear terms dramatically increase the training complexity, since the number of learnable parameters increases exponentially with the order of non-linearity. Even an quadratic expression gv(x) in (6) from [57] is of complexity O(n2):
    - 比較として、我々は例として[57]で採用されたボルテラ級数ベースの非線形畳み込みを取ります。学習可能なパラメータの数は次の次数で指数関数的に増加するので、非線形項の追加パラメータはトレーニングの複雑さを劇的に増加させます。 非線形性 [57]の（6）の二次式gv（x）でも、複雑度はO（n2）です。

- xxx

### 3.2. Translational Equivariance

- A crucial aspect of current architectures of deep learning is the encoding of invariances. One of the reasons is that the convolutional layers are naturally equivariant to image translation [18].
    - ディープラーニングの現在のアーキテクチャの重要な側面は、不変性の符号化です。 その理由の1つは、たたみ込みレイヤが自然に画像変換と同等であることです[18]。

- In this section, we show that kervolution preserves this important property. An operator is equivariant to a transform when the effect of the transform is detectable in the operator output [6]. Therefore, we have f(j)(x) = x(j) ⊕ w, which means the input translation results in the output translation [18]. Similarly,
    - この節では、kervolutionがこの重要な性質を保存することを示します。 変換の効果が演算子の出力で検出可能な場合、演算子は変換と同値です[6]。 したがって、f（j）（x）= x（j）≈wとなります。これは、入力変換によって出力変換が行われることを意味します[18]。 同様に

![image](https://user-images.githubusercontent.com/25688193/61710004-79617400-ad8b-11e9-9775-e578ffdee5e3.png)

- Note that the translational invariance of CNN is achieved by concatenating pooling layers to convolutional layers [18], and the translational invariance of KNN can be achieved similarly.
    - CNNの並進不変性は、プール層を畳み込み層に連結することによって達成され[18]、KNNの並進不変性も同様に達成できることに注意されたい。 

- This property is crucial, since when invariances are present in the data, encoding them explicitly in an architecture provides an important source of regularization, which reduces the amount of training data required [23].
    - 不変性がデータ内に存在する場合、それらをアーキテクチャ内で明示的に符号化することが正則化の重要な原因となり、それが必要とされるトレーニングデータの量を減らすため、この特性は非常に重要です。

- As mentioned in Section 2, the same property is also presented in [22], which is achieved by assuming that all the training samples are circular shifts of each other [49], while ours is inherited from convolution. 
    - 2節で述べたように、同じ性質が[22]にも示されており、これはすべてのトレーニングサンプルが互いの循環シフトであると仮定することによって達成され[49]、一方我々のものは畳み込みから継承される。

- Interestingly, the kernel cross-correlator (KCC) defined in [54] is equivariant to any affine transforms (e.g., translation, rotation, and scale), which may be useful for further development of this work.
    - 興味深いことに、[54]で定義されているカーネル相互相関子（KCC）は、この研究をさらに発展させるのに役立つかもしれないあらゆるアフィン変換（例えば、並進、回転、およびスケール）と同等である。

### 3.3. Model Capacity and Features

- It is straightforward that the kernel function (5) takes kervolution to non-linear space, thus the model capacity is increased without introducing extra parameters. Recall that CNN is a powerful approach to extract discriminative local descriptors.
    - カーネル関数（５）が非線形空間へのケルボリューションをとることは簡単であり、それ故、余分なパラメータを導入することなくモデル容量が増大される。 CNNは、識別可能なローカル記述子を抽出するための強力なアプローチであることを思い出してください。

- In particular, the linear kernel (2) of convolution measures the similarity of the input x and filter w, i.e the cosine of the angle θ between the two patches, since ⟨x, w⟩ = cos(θ) · ∥x∥ ∥w∥.
    - 特に、畳み込みの線形カーネル（２）は、入力ｘとフィルタｗの類似性、すなわち、２つのパッチ間の角度θの余弦を測定する。 ⟨x, w⟩ = cos(θ) · ∥x∥ ∥w∥.

- From this point of view, kervolution measures the similarity by match kernels, which are equivalent to extracting specific features [3].
    - この観点から、kervolutionはマッチカーネルによって類似性を測定します。これは特定の特徴を抽出することと同じです[3]。

- We next discuss how to interpret the kernel functions and present a few instances κ( · , · ) of the kervolutional operator. One of the advantages of kervolution is that the non-linear properties can be customized without explicit calculation.
    - 次に、カーネル関数の解釈方法と、kervolutional演算子のいくつかのインスタンスκ（・、・）を提示する方法について説明します。 kervolutionの利点の1つは、非線形特性を明示的な計算なしでカスタマイズできることです。

#### Lp-Norm Kervolution

- The L1-norm in (9a) and L2-norm in (9b) simply measures the Manhattan and Euclidean distances between input x and filter w, respectively.
    - （9a）のL1ノルムと（9b）のL2ノルムは、単純に入力xとフィルタw間のマンハッタン距離とユークリッド距離をそれぞれ測定します。

![image](https://user-images.githubusercontent.com/25688193/61710992-f2fa6180-ad8d-11e9-88e9-809f44303556.png)

- Both "distances" of two points involves aggregating the distances between each element. If vectors are close on most elements, but more discrepant on one of them, Euclidean distance will reduce that discrepancy (elements are mostly smaller than 1 because of the normalization layers), being more influenced by the closeness of the other elements. Therefore, the Euclidean kervolution may be more robust to slight pixel perturbation. This hypothesis is verified by a simple simulation of adversary attack using the fast gradient sign method (FGSM) [19], shown in Table 1, where ‘None’ means the test accuracy on clean data.
    - 2点の「距離」は両方とも、各要素間の距離を集計することを含みます。 ベクトルがほとんどの要素に近いがそれらの1つに矛盾がある場合、ユークリッド距離はその矛盾を減らし（正規化レイヤのために要素はほとんど1未満）、他の要素の近さの影響を受けます。 したがって、ユークリッドケルボリューションは、わずかなピクセル摂動に対してよりロバストになる可能性があります。 この仮説は、表1に示す高速勾配符号法（FGSM）[19]を使用した敵対攻撃の単純なシミュレーションによって検証されます。ここで、「なし」はクリーンなデータに対するテスト精度を意味します。

#### Polynomial Kervolution

- Although the existing literatures have shown that the polynomial kernel (10) works well for the problem of natural language processing (NLP) when dp=2 using SVM [17] , we find its performance is better when dp=3 in KNN for the problem of image recognition.
    - SVMを使用してdp = 2の場合、多項式カーネル（10）が自然言語処理（NLP）の問題に対してうまく機能することが既存の文献では示されていますが、画像認識問題に対してKNNでdp = 3の場合、そのパフォーマンスが優れています。 

![image](https://user-images.githubusercontent.com/25688193/61711312-b8dd8f80-ad8e-11e9-91e9-3bd4be29736d.png)

- where dp (dp ∈ Z+) extends the feature space to dp dimensions; cp (cp ∈ R+) is able to balance the non-linear orders (Intuitively, higher order terms play more important roles when cp < 1).
    - ここで、dp（dp∈Z +）は特徴空間をdp次元に拡張します。 cp（cp∈R +）は非線形次数のバランスをとることができます（直感的には、cp <1の場合、高次の項がより重要な役割を果たします）。
    
- As a comparison, the kernel pooling strategy [9] concatenates the non-linear terms c_j (x^T w)_j directly, while they are finally linearly combined by subsequent fully connected layer, which dramaticaly increases the number of learnable parameters in the linear layer.
    - 比較として、カーネルプーリング戦略[9]は、非線形項 c_j (x^T w)_j を直接連結しますが、それらは最終的に後続の完全結合層によって線形結合され、線形層の学習可能なパラメータの数が劇的に増加します。 

---

![image](https://user-images.githubusercontent.com/25688193/61711718-aa43a800-ad8f-11e9-8cbb-475a3b4d9d69.png)

- > Figure 1. The comparison of learned filters on MNIST from the first layer (six channels and filter size of 5 × 5) of CNN and polynomial KNN.
    - > 図1. CNNと多項式KNNの第1層（6チャネル、フィルター・サイズ5×5）からのMNISTでの学習済みフィルターの比較。

- > It is interesting that some of the learned filters (e.g. channel 4) from KNN are quite similar to CNN. This indicates that part of the kervolutional layer learns linear behavior, which is controlled by the linear part of the polynomial kernel.
    - > 興味深いことに、KNNから学習したフィルタの一部（チャネル4など）がCNNと非常によく似ています。 これは、kervolutionalレイヤの一部が多項式カーネルの線形部分によって制御される線形動作を学習することを示しています。

---


- To show the behavior of polynomial kervolution, the learned filters of LeNet-5 trained for MNIST are visualized in Figure 1, which contains all six channels of the first kervolutional layer using polynomial kernel (dp = 2, cp = 0.5).
    - 多項式ケルボリューションの振る舞いを示すために、MNIST用に学習されたLeNet-5の学習済みフィルターがを、図1に視覚化した。
    - これは、多項式カーネル（dp = 2、cp = 0.5）を使用した最初のケルボリューション層の6チャンネルすべてを含んでいる。

- The optimization process is described in Section 4. For a comparison, the learned filters from CNN are also presented. It is interesting that some of the learned filters of KNN and CNN are quite similar, e.g. channel 4, which means that part of the capacity of KNN learns linear behavior as CNN. This verifies our understanding of polynomial kernel, which is a combination of linear and higher order terms.
    - 最適化プロセスはセクション4で説明されています。比較のために、CNNから学習されたフィルタも提示されています。 興味深いことに、KNNとCNNの学習済みフィルタのいくつかは非常によく似ています。 これは、KNNの容量の一部がCNNとして線形動作を学習することを意味します。 これは、線形項と高次項の組み合わせである多項式カーネルの理解を検証します。

- This phenomenon also indicates that polynomial kervolution introduces higher order feature interaction in a more flexible and direct way than the existing methods.
    - この現象はまた、多項式ケルボリューションが既存の方法よりも柔軟で直接的な方法で高次特徴相互作用を導入することを示しています。

#### Gaussian Kervolution

- The Gaussian RBF kernel (11) extends kervolution to infinite dimensions.

![image](https://user-images.githubusercontent.com/25688193/61712118-8d5ba480-ad90-11e9-8f83-f5aaf909b529.png)

- where γg (γg ∈ R+) is a hyperparameter to control the smoothness of decision boundary. It extends kervolutoin to infinite dimensions because of the i-degree terms in (12).
    - ここで、γg（γg∈R +）は、決定境界の滑らかさを制御するためのハイパーパラメータです。 それは（12）のi次項のためにkervolutoinを無限次元に拡張します。

![image](https://user-images.githubusercontent.com/25688193/61712178-b1b78100-ad90-11e9-9418-4f25ccd27b25.png)

- The expression (12) is helpful for our intuitive understanding, while the recent discovery reveals more information. It is shown in [2] that the Gaussian kernel and its variants are able to measure the similarity of gradient based patch-wise features, e.g. SIFT [39] and HOG [11].
    - 式（12）は私たちの直感的な理解に役立ちますが、最近の発見はより多くの情報を明らかにします。 ［２］には、ガウス核およびその変形は、勾配ベースのパッチ単位の特徴の類似性を測定することができることが示されている。 SIFT [39]とHOG [11]。

- This provides a unified way to generate a rich, diverse visual feature set [15]. However, instead of using the hand-crafted features as kernel SVM, with KNN, we are able to inherit the substantial achievements based on kernel trick while still taking advantage of the great generalization ability of neural networks.
    - これは、豊かで多様な視覚的特徴セットを生成するための統一された方法を提供します[15]。 ただし、KNNでは、カーネルSVMとして手作りの機能を使用する代わりに、ニューラルネットワークの優れた一般化機能を利用しながら、カーネルトリックに基づく実質的な成果を継承できます。

### 3.4. Kervolutional Layers and Learnable Kernel

- Similar to a convolutional layer, the operation of a kervolutional layer is slightly different from the standard definition (3) in which x(i) becomes a 3-D patch in a sliding window on the input.
    - 畳み込みレイヤと同様に、ケルボリューションレイヤの動作は、x（i）が入力上のスライディングウィンドウ内で3Dパッチになるという標準的な定義（3）とは少し異なります。

- To be compatible with existing works, we also implement all popular available structures of convolution in CNN library [41] for kervolution, including the input and output channels, input padding, bias, groups (to control connections between input and output), size, stride, and dilation of the sliding window.
    - 既存の作品との互換性を保つために、入力および出力チャネル、入力パディング、バイアス、グループ（入出力の間の接続を制御するため）、サイズ、ストライド、スライディングウィンドウの拡張など、すべての一般的な利用可能な畳み込み構造をカーボリューション用にCNNライブラリ[41]に実装します。

- Therefore, the convolutional layers of all existing networks can be directly or partially replaced by kervolutional layers, which makes KNN inherit all the the existing achievements of CNN, e.g. network architectures [31, 20] and their numerous applications [42].
    - したがって、既存のすべてのネットワークの畳み込みレイヤを直接または部分的にケルボリューションレイヤで置き換えることができます。これにより、KNNは既存のCNNのすべての成果、たとえばネットワークアーキテクチャ[31]、[20]およびそれらの多数のアプリケーション[42]を継承できます。

---

- With kervolution, we are able to extract specific type of features without paying attention to the weight parameters. However, as aforementioned, we still need to tune the hyperparameters for some specific kernels, e.g. the balancer cp in polynomial kernel, the smoother γg in Gaussian RBF kernel.
    - kervolutionを使うと、重みパラメータに注意を払うことなく特定の種類の特徴を抽出することができます。 しかしながら、前述のように、我々は依然としていくつかの特定のカーネル、例えば多項式カーネルにおけるバランサｃｐ、ガウスＲＢＦカーネルにおけるより滑らかなγｇ、についてハイパーパラメータを調整する必要がある。

- Although we noticed that the model performance is mostly insensitive to the kernel hyperparameters, which is presented in Section 4.2, it is sometimes troublesome when we have no knowledge about the kernel. Therefore, we also implement training the network with learnable kernel hyperparameters based on the back-propagation [43].
    - モデルのパフォーマンスは、セクション4.2で説明しているカーネルのハイパーパラメータにはほとんど影響されないことに気付きましたが、カーネルについての知識がない場合は厄介です。 したがって、我々は逆伝播に基づいて学習可能なカーネルハイパーパラメータを用いてネットワークをトレーニングすることも実装する[43]。

- This slightly increases the training complexity theoretically, but in experiments we found that this brings more flexibility and the additional cost for training several kernel parameters is negligible, compared to learning millions of parameters in the network. Taking the Gaussian kervolution as an example, the gradients are computed as:
    - これは理論的にはトレーニングの複雑さをわずかに増加させますが、実験ではネットワーク内で何百万ものパラメータを学習するのに比べて柔軟性が増し、いくつかのカーネルパラメータをトレーニングするための追加コストは無視できることがわかりました。 例としてガウスのケルボリューションをとると、勾配は次のように計算されます。

![image](https://user-images.githubusercontent.com/25688193/61710722-32747e00-ad8d-11e9-83bc-b596a58ed48c.png)

- Note that the polynomial order dp is not trainable because of the integer limitation, since the real exponent may produce complex numbers, which makes the network complicated.
    - 実数指数は複素数を生成する可能性があり、これがネットワークを複雑にするため、多項式の次数dpは整数制限のために学習できません。


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. Related Work


