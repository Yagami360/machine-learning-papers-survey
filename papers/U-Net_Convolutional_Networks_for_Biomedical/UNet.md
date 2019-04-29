# ■ 論文
- 論文タイトル："U-Net: Convolutional Networks for Biomedical Image Segmentation"
- 論文リンク：https://arxiv.org/abs/1505.04597
- 論文投稿日付：2015/3/18
- 著者：Olaf Ronneberger, Philipp Fischer, Thomas Brox
- categories：

# ■ 概要（何をしたか？）

## Abstract.

- There is large consent that successful training of deep net-works requires many thousand annotated training samples.
    - ディープネットワークの学習の成功には、何千ものアノテーション付き学習サンプルが必要であるという、大きな同意がある。

- In this paper, we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently.
    - この論文では、
    - 利用可能なアノテーション付きサンプルをより効果的に使用するための、データ拡張 [data augmentation] の強力な使用に頼るような、
    - ネットワークと学習戦略を提示する。

- The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization.
    - このアーキテクチャは、コンテンツとを取り込むための contracting path と、正確な位置特定を可能にする 対称的な expanding path で構成される。

- We show that such a network can be trained end-to-end from very few images and outperforms the prior best method (a sliding-window convolutional network) on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks.
    - このようなネットワークは、僅かな画像から、end-to-end に学習される。
    - そして、以前のベストな手法（a sliding-window 畳み込みネットワーク）より性能が優れている。
    - 電子顕微鏡 stacks における、ニューラルネットでのセグメンテーション ための ISBI challenge において、

- Using the same network trained on transmitted light microscopy images (phase contrast and DIC) we won the ISBI cell tracking challenge 2015 in these cate-gories by a large margin. 
    - transmitted light microscopy images （透過型光学顕微鏡画像）で学習された同じネットワークを使用することで、
    - 我々は、これらのカテゴリで、大差をつけて [by a large margin] ISBI cell tracking challenge 2015 に勝利した。

- Moreover, the network is fast. 
    - 更には、このネットワークは早い。

- Segmentation of a 512x512 image takes less than a second on a recent GPU.
    - 512×512 画像のセグメンテーションは、最近の GPU で、1 秒もかからない。

- The full implementation (based on Caffe) and the trained networks are available at http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net.


# ■ イントロダクション（何をしたいか？）

## 1 Introduction

- In the last two years, deep convolutional networks have outperformed the state of the art in many visual recognition tasks, e.g. [7,3].
    - 過去の２年間で、畳み込み深層ネットワークは、多くの視覚認識タスクにおいて、SOTA を上回る性能を発揮していた。

- While convolutional networks have already existed for a long time [8], their success was limited due to the size of the available training sets and the size of the considered networks.
    - 畳み込みネットワークな長い間存在していたが、
    - それらの成功は、利用可能な学習データセットと考慮できるネットワークサイズのために、制限された。

- The breakthrough by Krizhevsky et al. [7] was due to supervised training of a large network with 8 layers and millions of parameters on the ImageNet dataset with 1 million training images.
    - Krizhevsky [7] によるブレークスルーは、１万個の学習画像での ImageNet datset における、８層の大きなネットワークと、何万ものパラメーターでの、教師あり学習に起因するものである。[be due to]

- Since then, even larger and deeper networks have been trained [12].
    - そのとき以来、より大きく深いネットワークが学習されている。

<br>

- The typical use of convolutional networks is on classication tasks, where the output to an image is a single class label.
    - 典型的な畳み込みネットワークの使用法は、画像への出力が１つのクラスラベルであるような、分類タスクである。

- However, in many visual tasks, especially in biomedical image processing, the desired output should include localization, i.e., a class label is supposed to be assigned to each pixel.
    - <font color="Pink">しかしながら、多くの視覚タスクにおいて、（とりわけ生体医療 [biomedical] での画像生成）
    - 望ましい出力は、（物体の位置の）局所化を含むべきである。
    - 例えば、クラスラベルは、各ピクセルに割り当てられると思われる。</font>

> セマンティックセグメンテーション（領域抽出）では、「物体の局所的特徴と全体的位置情報」の両方を元画像上で特定しなければなりません。


- Moreover, thousands of training images are usually beyond reach in biomedical tasks.
    - 更には、何千もの学習用画像は、生体医療タスクの範囲 [reach] を超えている。[beyond]

- Hence, Ciresan et al. [1] trained a network in a sliding-window setup to predict the class label of each pixel by providing a local region (patch) around that pixel as input. 
    - 従って [Hence]、Ciresan [1] は、
    - 入力としてのピクセル周りの（物体の位置の）局所的な領域（パッチ）を提供することによって、
    - 各ピクセルのクラスラベルを予想するための sliding-window 設定で、ネットワークを学習した。

- First, this network can localize. 
    - 初めに、このネットワークは（物体の位置の）局所化が出来る。

- Secondly, the training data in terms of patches is much larger than the number of training images.
    - 次に、パッチの後の学習データは、学習用画像の数より、はるかに多い。

- The resulting network won the EM segmentation challenge at ISBI 2012 by a large margin.
    - 結果のネットワークは、ISBI 2012 での、EM segmentation challenge で、大差をつけて勝利した。

<br>

- Obviously, the strategy in Ciresan et al. [1] has two drawbacks.
    - 明らかに、Ciresan [1] の戦略は、２つの欠点 [drawbacks] を持っている。

- First, it is quite slow because the network must be run separately for each patch, and there is a lot of redundancy due to overlapping patches.
    - 初めに、ネットワークは、各パッチに対して、分かれて動作しなくてはならず、
    - そして、パッチが重なっているために、たくさんの余分性 [redundancy] がある
    - ので、非常に低速である。

- Secondly, there is a trade-off between localization accuracy and the use of context. 
    - ２番目に、（物体の位置の）局所化の正確性と画像内容の利用との間に、トレードオフがある。

- Larger patches require more max-pooling layers that reduce the localization accuracy, while small patches allow the network to see only little context.
    - より大きなパッチは、（物体の位置の）局所化の正確性を減らすような、より多くの max-pooling 層を要求する。
    - 小さなパッチが、ネットワークに、小さな画像内容のみをを見る一方で、

- More recent approaches [11,4] proposed a classier output that takes into account the features from multiple layers.
    - 分類器を目的とした、より最近のアプローチでは、複数の層からの特徴を考慮に入れて、出力する。

- Good localization and the use of context are possible at the same time.
    - よい局所化と、画像内容の利用は、同時に可能である。

> 通常のCNNによって行われる画像のクラス分類（画像認識）では、畳み込み層が物体の局所的な特徴を抽出する役割を担い、プーリング層が物体の全体的な位置情報をぼかす（位置ズレの許容）の役割を担っています。そのため、より深い層ほど、抽出される特徴はより局所的になり、その特徴の全体的な位置情報はより曖昧になります。言い換えれば、プーリング層のおかげで、物体の位置ズレや大きさの違いの影響をあまり受けない、頑強なパターン認識が可能になっているわけです。

> 一方、領域抽出では、「物体の局所的特徴と全体的位置情報」の両方を元画像上で特定しなければなりません。つまり、プーリング層でぼかされた局所的特徴の位置情報を元画像上でpixel単位で正確に復元する必要があります。

> そこで、「物体の局所的特徴と全体的位置情報」の両方を統合して学習させるために開発されたのがU-Netです[1]。図3に示すように、U字型のネットワークになっていることから名付けられました。

<br>

- In this paper, we build upon a more elegant architecture, the so-called "fully convolutional network" [9].
    - この論文では、いわゆる "fully convolutional network" と呼ばれる、エレガントなアーキテクチャを元にしている [build upon]。

- We modify and extend this architecture such that it works with very few training images and yields more precise segmentations; see Figure 1.
    - 我々は、僅かな学習用画像で動作するように、このアーキテクチャを修正、拡張する。
    - そして、より正確なセグメンテーションを生み出す。
    - 図１を参照のこと

- The main idea in [9] is to supplement a usual contracting network by successive layers, where pooling operators are replaced by upsampling operators.
    - **[9] のメインとなるアイデアは、**
    - **pooling オペレーターが、upsampling オペレーターで置き換わっているような、** 
    - **連続した [successive] 層によって、**
    - **通常の縮小している [contracting] ネットワークを補足する [supplement] ことである。**

> 右側の Decoder 部分のこと。

- Hence, these layers increase the resolution of the output.
    - **従って、これらの層は、出力の解像度を増加させる。**

- In order to localize, high resolution features from the contracting path are combined with the upsampled output.
    - （物体の位置の）局所化のためには、contracting path からの高解像度の特徴マップは、アップサンプリングされた出力と組み合わされる。

- A successive convolution layer can then learn to assemble a more precise output based on this information.
    - 連続した畳み込み層は、この情報に基づいた、より正確な出力をアセンブルすることを学習出来る。

![image](https://user-images.githubusercontent.com/25688193/56891559-d6fd6180-6ab7-11e9-8b20-0b46f00b2c38.png)<br>

- > Fig. 1. U-net architecture (example for 32x32 pixels in the lowest resolution).
    - > 図１：U-net のアーキテクチャ（最も低い解像度が、32×32 のピクセル）

- > Each blue box corresponds to a multi-channel feature map.
    - > 各青ボックスは、複数チャンネルの特徴マップに一致する。

- > The number of channels is denoted on top of the box.
    - > チャンネル数は、ボックスの上に示されている。

- > The x-y-size is provided at the lower left edge of the box. 
    - > x-y のサイズは、ボックスの左側の辺の下側に示されている。

- > White boxes represent copied feature maps. 
    - > 白いボックスは、特徴マップのコピーを表している。

- > The arrows denote the diffierent operations.
    - > 矢印は、異なるオペレーションを示している。

<br>


# ■ 結論

## x. 論文の項目名 (Conclusion)


# ■ 何をしたか？詳細

## x. 論文の項目名


# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## x. 論文の項目名（Related Work）


