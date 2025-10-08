# Stanford CS336

## Introduce
> Full understanding of this technology is necessary for fundamental research.

<img src="https://github.com/user-attachments/assets/ac896cc4-3a4f-4e61-8824-8fa906b50fce" alt="drawing" width="600"/>


### Assignment 1
* Implement all of the components (tokenizer, model architecture, optimizer) necessary to train a standard Transformer language model.
* Train a minimal language model.
<!-- * 实现BPE分词器‌ (**实现高度优化的c++ BPE算法，在TinyStories数据上train处理不到2s。**)
* 实现Transformer模型、交叉熵损失函数、AdamW优化器及训练循环‌
* TinyStories和OpenWebText数据集上进行训练‌ 
* 打榜：在H100上给定90分钟内最小化OpenWebText的perplexity -->

### Assignment 2
* Profile and benchmark the model and layers from Assignment 1 using advanced tools, optimize Attention with your own Triton implementation of FlashAttention2.
* Build a memory-efficient, distributed version of the Assignment 1 model training code.
<!-- * 对实现进行基准测试和性能分析‌
* 实现FlashAttention2算法 （**实现casual时负载平衡的forward算法，实现Triton的backward算法**）
* 实现分布式数据并行训练‌
* 实现优化器状态分片‌ -->

### Assignment 3
* Understand the function of each component of the Transformer.
* Query a training API to fit a scaling law to project model scaling.
<!-- * 定义训练API标准化接口
* 提交训练任务（在FLOPs预算内）并收集训练数据
* 对训练数据拟合scaling law
* 提交对scaled up后超参数的预测结果 -->

### Assignment 4
* Convert raw Common Crawl dumps into usable pretraining data.
* Perform filtering and deduplication to improve model performance.
<!-- * Common Crawl HTML转文本‌
* 训练质量与安全内容分类器‌
* 基于MinHash的去重处理‌
* 打榜：在给定token预算下最小化perplexity -->

### Assignment 5
* Apply supervised finetuning and reinforcement learning to train LMs to reason when solving math problems.
* Optional Part 2: implement and apply safety alignment methods such as DPO.
<!-- * 监督微调实现
* 直接偏好优化实现
* 群体相对偏好优化实现 -->

## Resources
- [Course Home](https://stanford-cs336.github.io/spring2025/)