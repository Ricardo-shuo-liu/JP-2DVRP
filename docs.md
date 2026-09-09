核心思想：

针对 TAM 第一阶段 seq2seq 自回归生成子路由带来的中间时序状态累积偏差，借鉴 2D‑Ptr 二维分配矩阵结构，改造为 node 维度独立 softmax 并行完成节点‑车辆分簇；利用 DPN 旋转位置编码对同质车辆做 agent 区分；保留两阶段框架，分簇完成后每个簇执行 TSP 求解；采用联合强化学习损失同时训练分簇模块与路由模块。



两阶段求解器

在 [Generalize Learned Heuristics to Solve Large-scale Vehicle Routing Problems in Real-time | OpenReview](https://openreview.net/forum?id=6ZajpxqTlQ)

中提出了一种两次训练强化学习的方式 

这种训练方式下的推理流程为：

- 先将所有Node拆分成多个簇（数量为veh的数量）
- 然后对于每个簇单独求解

![image-20260817155248479](C:\Users\Ricardo\AppData\Roaming\Typora\typora-user-images\image-20260817155248479.png)

（ 原论文整体流程）

![image-20260817154823601](C:\Users\Ricardo\AppData\Roaming\Typora\typora-user-images\image-20260817154823601.png)

(论文模型的整体流程)



这种训练方式下将问题规模大大减少



论文原始的分簇的模型或者说是生成子路的模型存在显然的问题：

> Modeling the policy in Equation 2 faces two challenges. First, the action space is much larger and we need to sample all nodes inside a sub-route while ignoring the nodes’ order. Second, mask function for constraints is harder to encode without node sequence (Kool et al., 2019; Nazari et al., 2018). To bridge this gap, we propose to transform the policy in Equation 2 into an equivalent sequence-to-sequence policy (Equation 4) with novel reward to eliminating the effect of nodes’ order (Equation 5). We assume the nodes between two depots belong to the same sub-route: 

$$
\begin{aligned} & p_{\theta}(\pi | s)=\prod_{t=1}^{n+l} p_{\theta}\left(\pi_{t} | s_{t}\right) \\ & \pi^{0}, ..., \pi^{i}, ... \pi^{l} \in\left\{I_{0}\right\} \\ & \left\{\pi^{i-1}=\pi_{j}, \pi_{j+1} ..., \pi^{i}\right\} \in r_{i} \forall i=1, ..., l \\ & r_{i} \in \Theta_{i} \forall i=1, ..., l \\ & l \leq l_{m} \end{aligned} (4)
$$

> where \(\pi^{i}\) is the depot visited in \(i^{th }\) time: \(r_{i}\) is \(i^{th }\) sub-route; \(\Theta_{i}\) represents the feasible space of \(r_{i}\) The first line of Equation 4 is a sequence-to-sequence policy. The second line shows the times to visit the depot. The third line transforms the sequence of nodes between two depots into the set of nodes (a sub-route problem). The fourth line shows each sub-route should satisfy some local constraints such as capacity constraint. The fifth line shows the global constraint for maximum vehicle number. With this formulation, we could generate sub-route sequence with sequence-to-sequence model. However, the nodes’ order inside sub-route still affects the performance of the policy.



论文中的输出子路的方式是一种`seq2seq`的方式，虽然在奖励的设计的时候能够忽略node里的顺序关系，但是由于自回归的特性还是存在时序上的先后顺序因此我认为存在限制，所有我希望消除这种时序上的影响。

简而言之整体的流程如下：

```
通过TAM（及分子路的模型）
生成的序列为：
01234056780....

这里0 代表仓库其他的1，2，3等等代表Node的编号
0作为分割
也就是说 1234 为第一簇 5678 为第二簇

整体输出是一个序列 效果上 021340 和 012340 是一样的，而且经过模型推理下一个node的编号的也是具有相同的效果（因为注意力是没有位置信息的）
但是由于在不是完整的簇来看 单独的02 和 01 是完全不同的情况 之后的推理这样的偏差会不断累计因为时序上具有先后顺序
```



我借鉴了2d-ptr的思想 希望来解决这个问题

[2D-Ptr：用于解决异质容量约束车辆路径问题的 2D 数组指针网络方法 | 第 23 届自主代理与多主体系统国际会议论文集 --- 2D-Ptr: 2D Array Pointer Network for Solving the Heterogeneous Capacitated Vehicle Routing Problem | Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems](https://dl.acm.org/doi/10.5555/3635637.3662981)



![image-20260817161101461](C:\Users\Ricardo\AppData\Roaming\Typora\typora-user-images\image-20260817161101461.png)

（原论文的整体流程）



原始论文中对于最后的[veh,node] 这样shape的矩阵采取了全局的softmax 然后在其中选择一个点这样就选择了（veh和node）这样的建模思想是选取action。

我希望使用对于在node这个维度上做单独的softmax 这样就将输出的矩阵转化为该node 在某一个veh的可能值。

对于全部node提取完成分类子路。

也就是说假设存在2个node和2个veh

```
在原始论文中的输出矩阵
[[0.1,0.2],
 [0.3,0.4]]
然后随机采样得到的是选取 某个车辆到第几个node也就是action

而我选择在node上单独的softmax
输出矩阵为
[[0.3,0.7],
 [0.2,0.8]]
 
然后对每个node的维度进行随机采样
这样就可以得到每个node属于第几个veh的簇
然后完成分簇
比如对于上述矩阵就是node1和node2都应该是在veh2的簇对于veh1的簇没有node
```

相较于TAM模型使用2d-ptr的思路好处如下：

- 一次分类出全部簇
- 由于2d-ptr是完全基于注意力的模型 所有没有位置信息 也就是说无论传入的node和veh的序列具体如何 最后的输出都一样
- 由于每个维度单独采样 彼此之间不存在依赖关系



这种先分类再规划的方法在TAM论文中证明了其效果。

| CVRP | METRIC | L2I    | TAM‑LKH3 |
| ---- | ------ | ------ | -------- |
| 2000 | LENGTH | 138.75 | 64.78    |
| 2000 | TIME   | 25.38  | 5.63     |
| 1000 | LENGTH | 93.15  | 46.34    |
| 1000 | TIME   | 6.32   | 1.82     |
| 400  | LENGTH | 30.67  | 25.93    |
| 400  | TIME   | 3.12   | 1.35     |
| 100  | LENGTH | 16.62  | 16.08    |
| 100  | TIME   | 1.01   | 0.86     |

（论文中的部分实验数据）



由于2d-ptr用于HCVRP问题上，每个车是不同的，但是由于CVRP问题中 每车的容量相同所有需要单独标注每个车辆

这里我希望使用 [DPN: Decoupling Partition and Navigation for Neural Solvers of Min-max Vehicle Routing Problems]([[2405.17272v1\] DPN: Decoupling Partition and Navigation for Neural Solvers of Min-max Vehicle Routing Problems](https://arxiv.org/abs/2405.17272v1))的方法，使用类似的旋转位置编码来标注车辆



> In Appendix C.5, we illustrate how a PE conveys the angle‑based information when representing the sub‑graphs in the partition part of the P&N Encoder. The angle‑based interval between sub‑graphs is highly correlated with the depot location (França et al., 1995), so SPE excluding the depot coordinates will harm the partition optimality. To introduce the depot coordinates to PEs, we refer to the concept of Rotary PE (Su et al., 2024) and design the Rotation‑based $PE \in \mathbb R^{M ×d}$ for a depot‑aware agent distinction as 

$$
PE(m, q)=Re\left[\left(x_{d} W_{a}+b_{a}\right) exp \left(i m / 1,000^{\frac{\lfloor q / 2\rfloor}{d}}\right)\right],
$$

> where $i$ is an imaginary unit (i.e., $i^{2}=-1$ ) and $x_{d} W_{a}+b_{a}$ is a trainable linear projection of the depot coordinate $x_{d} \in \mathbb R^{2}$. PE is calculated in the complex vector space, and $\text{Re}$ represents the real part of a complex number. Additionally, Appendix C.6 provides an intuitive way of calculating PE.



在CVRP问题中由于仓库不能多次访问所以在需要在引入一个惩罚项用于防止分簇中的容量大于veh最大容量

但是在HCVRP中因为可以多次访问仓库只需要在2d-ptr输出矩阵上加入掩码即可



基于TAM框架下修改后的训练流程为：

1. 选用一个寻路模型然后在一个车多个节点的情况下训练收敛
2. 使用2d-ptr的分簇模型 将这个寻路模型用于评估分簇效果并加入惩罚项
3. 借助2d-ptr原始论文中的训练方法 将整体流程使用强化学习训练 提高整体效果

$$
\mathcal L_{joint}=-\mathbb{E}\Big[(R-b)\cdot \Big(\log P_{assign}(z)+\alpha\sum_{k}\log P_{route}(\tau_k)\Big)\Big] (联合训练公式)
$$

