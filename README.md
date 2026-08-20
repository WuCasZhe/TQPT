TQPT 是对 [ZGC-LLM-Safety/TrafficLLM](https://github.com/ZGC-LLM-Safety/TrafficLLM) 的有限复现，仅采用 ISCXVPN 2016，DAPT 2020和APP-53 2023数据集。相较于原始 *TrafficLLM*，TQPT 将 *TrafficLLM* 原有的流量领域 *Tokenizer* 构建方式修改为*追加式扩展词表*，在保留基础模型原始词表的基础上加入流量领域相关 Token，两阶段训练分别采用*QLoRA*和*Prefix-Tuning*。

