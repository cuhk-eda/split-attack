# SplitAttack
Split manufacturing of integrated circuits means to delegate the front-end-of-line (FEOL) and back-end-of-line (BEOL) parts to different foundries, in order to prevent overproduction, intellectual property (IP) piracy, or targeted insertion of hardware Trojans.
SplitAttack challenges the security promise of split manufacturing by a sophisticated deep neural network that can infer the missing BEOL connections with high accuracy.
In paticular, it features following method for an efficient and effective connection prediction:
* [SplitExtract](https://github.com/cuhk-eda/split-extract) which formulates various layout-level placement and routing hints,
* a neural network makes use of vector-based and image-based layout features simultaneously,
* a loss function that directly and effectively select the most probable BEOL connection among the relevant candidates without suffering from an imbalance between positive and negative samples,
* ...

More details are in the following papers:
* Haocheng Li, Satwik Patnaik, Abhrajit Sengupta, Haoyu Yang, Johann Knechtel, Bei Yu, Evangeline F.Y. Young, Ozgur Sinanoglu, "[Attacking split manufacturing from a deep learning perspective](https://doi.org/10.1145/3316781.3317780)", ACM/IEEE Design Automation Conference (DAC), Las Vegas, NV, USA, June 2-6, 2019.
* Haocheng Li, Satwik Patnaik, Mohammed Ashraf, Haoyu Yang, Johann Knechtel, Bei Yu, Ozgur Sinanoglu, Evangeline F.Y. Young, "[Deep Learning Analysis for Split Manufactured Layouts with Routing Perturbation](https://doi.org/10.1109/TCAD.2020.3037297)", IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (TCAD), 2020.

## 1. How to Download

~~~bash
$ git clone https://github.com/cuhk-eda/split-attack
~~~

### Dependencies

* [Python 3](https://www.python.org)

## 2. How to Run

### Toy Test
