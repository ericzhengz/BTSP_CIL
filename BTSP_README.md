# BTSP Memory Pool - Complete Implementation

## 概述 (Overview)

本实现基于《1.0.txt》方案文档，在PILOT持续学习框架中完整实现了BTSP (Behavioral Timescale Synaptic Plasticity) 内存池系统。

### 核心特性

1. **稀疏随机翻转编码** - 使用4%稀疏度的二进制向量存储类别原型
2. **资格轨迹机制** - 指数衰减的时间窗口，支持延迟学习
3. **分支级门控** - 自适应的随机翻转概率，防止灾难性遗忘  
4. **稳态调节** - 基于占用率的反馈机制，维持系统平衡
5. **信息论容量界限** - 理论指导的容量管理
6. **零样本存储** - 仅存储二进制内存池，无需保存原始样本

## 文件结构

```
models/
├── btsp_mp.py              # 主BTSP模型类 (BTSPMemoryPool)
└── __init__.py             # 模型注册

backbone/
├── vit_btsp.py            # ViT骨干网络封装
└── vit_*.py               # 其他ViT变体

utils/
├── inc_net_btsp.py        # BTSP增量网络 (BTSPIncrementalNet)
└── factory.py             # 模型工厂 (已注册btsp_mp)

exps/
├── btsp_mp_complete.json  # 完整配置文件
├── btsp_mp.json          # 基础配置
└── btsp_capacity_cliff.json # 容量悬崖实验配置

# 测试和分析工具
test_btsp_complete.py      # 完整功能测试
btsp_analysis.py          # 参数分析工具
btsp_params_table.py      # 预计算参数表
run_btsp_demo.bat         # 演示脚本
```

## 快速开始

### 1. 测试BTSP组件
```bash
python test_btsp_complete.py
```

### 2. 运行CIFAR-100增量学习
```bash
python main.py --config exps/btsp_mp_complete.json
```

### 3. 完整演示
```bash
run_btsp_demo.bat
```

## 核心算法

### 1. 内存写入过程

```python
# 特征编码为稀疏二进制码
x_bits = encode_features_to_bits(features, sparsity=0.04)

# 更新资格轨迹 (指数衰减)
eligibility_traces[class_id] *= beta  # beta = exp(-1/tau_e)
eligibility_traces[class_id][x_bits] = 1.0

# 分支级门控和随机翻转
for branch_id in gated_branches:
    eligible_indices = eligibility_traces[class_id] >= theta
    flip_indices = eligible_indices & random_coin_flips()
    memory_weights[class_id][flip_indices] ^= 1  # XOR翻转
```

### 2. 内存检索过程

```python
# 计算Popcount相似度
similarities = []
for class_id in memory_pool:
    match_score = (query_bits & memory_weights[class_id]).sum()
    similarities.append(match_score)

# 温度标定与归一化
rho_est = 0.0  # 遮挡估计
mu = N * p_pre * (1 - rho_est)  # 期望重叠
std = sqrt(N * p_pre * (1 - rho_est))  # 标准差
z_scores = (memory_scores - mu) / std

# 温度缩放
mem_logits = z_scores / T_mem

# 与标准分类器融合
fused_logits = (1-α) * standard_logits + α * mem_logits
```

### 3. 稳态调节机制

```python
# 计算分支占用率（每K步更新一次）
for branch_id in range(num_branches):
    branch_mask = (branch_idx == branch_id)
    current_occupancy = memory_weights[:, branch_mask].float().mean()
    
    # EMA更新
    branch_occupancy[branch_id] = β_ema * branch_occupancy[branch_id] + (1-β_ema) * current_occupancy
    
    # 稳态调节 (目标占用率 α* = 0.02)
    adjustment = exp(η * (α_star - branch_occupancy[branch_id]))
    p_gate_per_branch[branch_id] *= adjustment

# 数值截断
p_gate_per_branch.clamp_(min=p_min, max=p_max)
```

## 参数配置

### 关键BTSP参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `btsp_memory_dim` | 4096 | 内存向量维度 N |
| `btsp_sparsity` | 0.04 | 稀疏度 p_pre (⚠️可能需提升至5-6%) |
| `btsp_branches` | 16 | 分支数量 |
| `btsp_tau_e_steps` | 6 | 资格轨迹时间常数 (步数) |
| `btsp_theta` | 0.3 | 资格轨迹阈值 |
| `btsp_alpha_star` | 0.02 | 目标占用率 |
| `btsp_eta` | 0.05 | 稳态学习率 |
| `btsp_fusion_alpha` | 0.3 | 内存池融合权重 |
| `mem_temperature` | 1.0 | Popcount温度标定 |
| `p_gate_clip` | [1e-4, 0.1] | p_gate截断区间 |

### 生物学时间映射

- **生物时间** → **算法步数**
- 1秒 → 1训练步
- τ_e = 6秒 → 6训练步  
- 有效时间窗口 T_eff = τ_e × ln(1/θ) = 6 × ln(1/0.3) ≈ 7.22步（修正）

## 容量分析

### 理论容量界限

1. **稳定性约束**: ε = ½(1-(1-2p_flip)^M) ≤ ε₀ ⇒ p_flip ≤ (1-(1-2ε₀)^(1/M))/2
   - M=10任务 ⇒ p_flip_max ≈ 5.24×10⁻³
   - M=100任务 ⇒ p_flip_max ≈ 5.27×10⁻⁴

2. **检索可分性**: N×p_pre×(1-ρ_max) ≥ ln(2M/δ)/g(ε₀)，其中g(ε)=(1-2ε)²/8
   - N=4096, ρ_max=0.4, M=100, δ=10⁻³ ⇒ p_pre_min ≈ 0.049
   - **注意**: 当前4%稀疏度可能偏低，建议提升至5-6%或增大N

### 内存效率

- **每类内存**: 
  - Bitset存储: N bits = 4096 bits = 512 bytes/类
  - 稀疏索引: K×log₂(N) = 164×12 ≈ 246 bytes/类
- **100类总内存**: 
  - Bitset: ~50 KB
  - 稀疏索引: ~24 KB
- **容量悬崖**: 理论预测在M=200-500类之间

## 实验结果

### CIFAR-100 (10+10×9设置)
- 初始10类 + 9个任务各10类
- 对比方法: Finetune, iCaRL, L2P, DualPrompt等
- 评估指标: 平均准确率, 遗忘度, 内存效率

### 预期性能
- **准确率**: 保持在合理水平 (具体数值待实验)
- **内存效率**: 比基于样本的方法节省>99%存储
- **容量扩展**: 理论上支持数百个类别

## 与PILOT框架集成

### 继承结构
```python
class BTSPMemoryPool(BaseLearner):
    def incremental_train(self, data_manager):
        # 标准训练 + BTSP内存池构建
    
    def eval_task(self):
        # 融合内存检索的评估
    
    def after_task(self):
        # 任务完成后的清理工作
```

### 兼容性
- ✅ 完全兼容PILOT数据管理器
- ✅ 支持多GPU训练
- ✅ 标准化评估接口
- ✅ 配置文件驱动

## 调试和验证

### 组件测试
```bash
python test_btsp_complete.py
```

验证项目:
- [x] BTSPIncrementalNet前向传播
- [x] 特征编码和二进制化
- [x] 内存池写入/检索
- [x] 分支门控机制
- [x] 稳态调节
- [x] 容量界限分析

### 参数分析
```bash
python btsp_analysis.py
```

生成:
- 参数敏感性分析
- 容量界限曲线
- 时间常数验证
- 门控概率优化

## 后续扩展

### 算法改进
1. **自适应稀疏度**: 根据任务复杂度动态调整p_pre
2. **分层内存**: 不同时间尺度的多层记忆系统
3. **元学习**: 学习最优的门控策略

### 应用场景
1. **大规模类别**: 支持1000+类别的持续学习
2. **在线学习**: 流式数据的实时处理
3. **资源受限**: 移动设备上的轻量级部署

## 引用

如果使用本实现，请引用:
```
BTSP Memory Pool Implementation for Continual Learning
Based on Behavioral Timescale Synaptic Plasticity
Implemented in PILOT Framework, 2024
```

---

**作者注**: 本实现严格遵循《1.0.txt》方案文档的所有技术细节，包括生物学机制映射、数学公式推导和工程实现策略。所有核心算法都已完整实现并通过测试验证。
