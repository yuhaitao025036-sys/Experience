future_tensor 用于支持如下功能:
1）前向循环，反向并发；
2）将循环编码到张量维度里，用 max_length+镂空支持变长循环；
3）前向依托 future tensor 搭建 autograd 的门面；
4）每个 layer 本质上是 actor 而不是 module，是拉模式而不是推模式；
5）实际执行栈和语义网全部包装到 TrajactoryContext 里，在 actor之间传递；
6）actor有两类角色：generator和 validator，前者负责生成内容，后者负责验证内容；
7）模型结构等效于 agent 编排。
8）反思机制不止作用在反向，也会作用在前向。
