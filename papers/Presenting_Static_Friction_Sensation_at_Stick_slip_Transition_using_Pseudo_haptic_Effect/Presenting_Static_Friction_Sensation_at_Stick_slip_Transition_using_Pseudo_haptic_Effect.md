# ■ 論文
- 論文タイトル："Presenting Static Friction Sensation at Stick-slip Transition using Pseudo-haptic Effect"
- 論文リンク：https://arxiv.org/pdf/1904.11676.pdf
- 論文投稿日付：2019/04/26
- 著者（組織）：
- categories：

# ■ 概要（何をしたか？）

## Abstract

- Previous studies have aimed at creating a simple hardware implementation of surface friction display. In this study, we propose a new method for presenting static frictional sensation using the pseudo-haptic effect as a first attempt, which is the simplest implementation of presenting static friction sensation. 
    - 以前の研究は、表面摩擦ディスプレイの単純なハードウェア実装を作成することを目的としていました。本研究では、最初の試みとして、擬似触覚効果を用いた静止摩擦感覚を提示する新しい方法を提案します。これは、静止摩擦感覚を提示する最も簡単な実装です。

- We focus on the stick-slip phenomenon while users explore surfaces with an input device, such as a stylus. During the stick phase, we present users with pseudo-haptic feedback that represents static friction on the surface. In our method, users watch a virtual contact point become stuck at the contact point on screen while users freely move the input device. We hypothesize that the perceived probability and intensity of static friction sensation can be controlled by changing the static friction coefficient as a visual parameter.
    - ユーザーがスタイラスなどの入力デバイスを使用して表面を探索している間、スティックスリップ現象に焦点を当てます。スティック段階では、表面の静的摩擦を表す疑似触覚フィードバックをユーザーに提示します。この方法では、ユーザーは入力デバイスを自由に動かしながら、仮想接触点が画面上の接触点で止まるのを見ることができます。静的摩擦感覚の知覚確率と強度は、視覚パラメータとして静的摩擦係数を変更することで制御できると仮定します。

> スタイラス（stylus, 複数形：styli, styluses）は、先の尖った棒状の筆記具で、インクなどを使わずに、押し当てることで筆記する。 現在では、携帯情報端末 (PDA) などのタッチパネル式のポインティングデバイスを操作するものが知られるが、本来は柔らかい素材

> スティックスリップ現象

![image](https://user-images.githubusercontent.com/25688193/63073709-018b0180-bf65-11e9-969b-ab452c7ef533.png)


- User studies were conducted, and results show the threshold value over which users felt the pseudo-haptic static friction sensation at 90% probability. The results also show that the perceived intensity of the sensation changed with respect to the static friction coefficient. The maximum intensity change was 23%. These results confirm the hypothesis and show that our method is a promising option for presenting static friction sensation.    
    - ユーザー調査が実施され、結果は、ユーザーが90％の確率で擬似触覚静的摩擦感覚を感じるしきい値を示しています。結果は、感覚の知覚強度が静止摩擦係数に関して変化したことも示しています。最大強度変化は23％でした。これらの結果は仮説を確認し、我々の方法が静止摩擦感覚を提示するための有望なオプションであることを示しています。


# ■ イントロダクション（何をしたいか？）

## x. Introduction

- Friction displays that present tangential forces over a users’ fingers have been developed in other studies. The studies fell into two categories. First, force displays represent the actual frictional force on the contact surface [1]. Second, the frictional characteristics of the contact surface were changed [2], [3]. Studies in both categories could effectively present a frictional force to users. However, in terms of accuracy, power consumption, and size, mechanical addons for devices (e.g., mobile devices) are often impractical. In such cases, a method for presenting frictional feedback without additional equipment is preferable.
    - ユーザーの指に接線方向の力を与える摩擦ディスプレイは、他の研究で開発されています。 研究は2つのカテゴリーに分類されました。 まず、力の表示は、接触面の実際の摩擦力を表します[1]。 次に、接触面の摩擦特性が変更されました[2]、[3]。 両方のカテゴリの研究は、ユーザーに摩擦力を効果的に提示できます。 ただし、精度、消費電力、サイズの観点から、デバイス（モバイルデバイスなど）の機械的アドオンは実用的でないことがよくあります。 そのような場合、追加の機器なしで摩擦フィードバックを提示する方法が望ましい。

---

- On the other hand, some studies increasingly focus on pseudo-haptics. Pseudo-haptics is a cross-modal effect between visual and haptic senses [4]. The pseudo-haptic effect indicates the haptic perception evoked by vision. A sensation is produced by an appropriate sensory inconsistency between the physical motion of the body and the observed motion of a virtual pointer. For example, when a pointer decelerates in a standard desktop environment with a mouse , users feel a kinetic frictional force without any haptic actuator [5].
    - 一方、いくつかの研究では、擬似触覚にますます焦点を当てています。 疑似触覚は、視覚と触覚のクロスモーダル効果です[4]。 疑似触覚効果は、視覚によって誘発される触覚知覚を示します。 感覚は、身体の物理的な動きと仮想ポインターの観測された動きとの間の適切な感覚の不一致によって生み出されます。 たとえば、マウスを使用して標準デスクトップ環境でポインターが減速すると、ユーザーは触覚アクチュエーターなしで動的な摩擦力を感じます[5]。

---

- While kinetic friction sensation using pseudo-haptics was presented in some papers [5], [6], [7], static friction sensation was not addressed. However, the methods in these studies are ineffective for rendering the fricitonal properties of materials with the same kinetic friction coefficients but different static friction coefficients. Static friction sensation should be presented in order to allow users to recognize and discriminate various material surfaces with a diverse range of static coefficients.
    - いくつかの論文[5]、[6]、[7]で疑似触覚を使用した動摩擦感覚が提示されましたが、静的摩擦感覚は扱われていませんでした。 ただし、これらの研究の方法は、動摩擦係数が同じで静止摩擦係数が異なる材料の摩擦特性をレンダリングするには効果がありません。 静的摩擦感覚は、ユーザーがさまざまな範囲の静的係数でさまざまな材料表面を認識および区別できるようにするために提示する必要があります。

---

![image](https://user-images.githubusercontent.com/25688193/63073895-ca692000-bf65-11e9-9751-20ce01546a83.png)

- > Fig. 1. We propose a method for presenting static friction sensation using pseudo-haptics. During the stick phase in the stick-slip phenomenon, the virtual pointer sticks to a surface and users feel pseudo-haptic static friction.
    - > 疑似触覚を使用して静的摩擦感覚を提示する方法を提案します。 スティックスリップ現象のスティック段階では、仮想ポインターが表面にくっつき、ユーザーは擬似触覚の静的摩擦を感じます。

---

- In this study, we propose a method for presenting static friction sensation using the pseudo-haptic effect. We focused on the stick-slip phenomenon while users explore surfaces with an input device, such as a stylus. During the stick phase, users were presented with pseudo-haptic friction sensation, which represents the frictional properties of a material (see Fig1).
    - 本研究では、疑似触覚効果を用いて静止摩擦感覚を提示する方法を提案する。 ユーザーがスタイラスなどの入力デバイスを使用してサーフェスを探索しているときのスティックスリップ現象に注目しました。 スティック段階では、ユーザーは材料の摩擦特性を表す疑似触覚摩擦感覚を提示されました（図1を参照）。

- However, if we implement the concept in a straightforward way, the visualized contact point becomes stuck and appears to have no relation to a user’s input. As a result, the sense of agency over the point would be lost and it would prevent the induction of pseudo-haptics. Thus, we applied an additional virtual string technique [7] to maintain the sense of agency. Details are described in Section 3. We hypothesize that we can control the perceived probability and intensity of static friction sensation by changing the visual parameters. We conducted user studies to test this hypothesis.
    - ただし、この概念を簡単な方法で実装すると、視覚化された接点が動かなくなり、ユーザーの入力とは関係がないように見えます。 その結果、ポイントに対するエージェンシーの感覚が失われ、疑似触覚の誘導が妨げられます。 したがって、追加の仮想文字列テクニック[7]を適用して、エージェンシーの感覚を維持しました。 詳細についてはセクション3で説明します。視覚パラメーターを変更することで、知覚される確率と静的摩擦感覚の強度を制御できると仮定します。 この仮説を検証するために、ユーザー調査を実施しました。

# ■ 結論

## VI. CONCLUSIONS

- A method for presenting static friction using the pseudohaptic effect is proposed in this paper. The user studies yielded the following findigs:
    - 本論文では、擬似触覚効果を使用して静止摩擦を提示する方法を提案します。 ユーザー調査により、次の発見が得られました。

- When users watch the pointer visually stick to a particular point while freely sliding the input device, they believe that the surface static friction is larger. The effect occurs with greater than 90% probability when the visual parameters exceeded a threshold value. Visualizing a virtual string [7] increased the probability.
    - ユーザーが入力デバイスを自由にスライドさせながらポインターが特定のポイントに視覚的に貼り付いているのを見ると、表面の静止摩擦が大きいと考えています。 視覚パラメータがしきい値を超えると、90％を超える確率で効果が発生します。 仮想文字列の視覚化[7]は、確率を高めました。

- The perceived intensity of the sensation would be larger as the static coefficient setting increases. The maximum intensity change confirmed in the user study was 23%.
    - 知覚された感覚の強さは、静的係数の設定が増加するにつれて大きくなります。 ユーザー調査で確認された最大強度変化は23％でした。

- These results suggests that our method is helpful for presenting static friction sensation with high probability. Our method is simple and can be implemented with current offthe-shelf mobile devices without any haptic actuator.
    - これらの結果は、本手法が静止摩擦感覚を高い確率で提示するのに役立つことを示唆しています。 私たちの方法はシンプルで、触覚アクチュエータなしで現在の市販のモバイルデバイスで実装できます。
    
# ■ 何をしたか？詳細

## III. CONCEPT AND IMPLEMENTATION



# ■ 実験結果（主張の証明）・議論（手法の良し悪し）・メソッド（実験方法）

## x. 論文の項目名


# ■ 関連研究（他の手法との違い）

## 2. Related Work

### A. Real Frictional Feedback

- Presenting frictional feedback ordinarily requires presenting a tangential force, and presenting it with a mechanical interface has been widely researched [1], [2], [3], [8]. Although these approaches can elicit frictional feedback, applying them to handheld devices is often impractical because additional electro-mechanical components are required.
    - **摩擦フィードバックを提示するには、通常、接線方向の力を提示する必要があり、機械的インターフェースで提示することは広く研究されています[1]、[2]、[3]、[8]。 これらのアプローチは摩擦フィードバックを引き出すことができますが、追加の電気機械コンポーネントが必要なため、それらをハンドヘルドデバイスに適用することはしばしば非実用的です。**

