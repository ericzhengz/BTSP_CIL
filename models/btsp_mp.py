# -*- coding: utf-8 -*-
"""
BTSP Memory Pool (BTSP-MP) for Continual Learning
Based on Behavioral Timescale Synaptic Plasticity with sparse random flip codes.

Key features:
- No exemplar storage (only binary memory pools)
- Eligibility traces with exponential decay
- Branch-level gating with homeostasis
- Information-theoretic capacity bounds

Author: BTSP Implementation
"""
import logging
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from copy import deepcopy

from models.base import BaseLearner
from utils.inc_net_btsp import BTSPIncrementalNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy

num_workers = 8


class BTSPMemoryPool(BaseLearner):
    """
    BTSP Memory Pool learner implementing biological timescale synaptic plasticity
    with sparse random flip codes for continual learning.
    """
    
    def __init__(self, args):
        super().__init__(args)
        self._network = BTSPIncrementalNet(args, True)
        
        # Move network to device
        self._network = self._network.to(self._device)
        
        # BTSP Memory Pool parameters
        self.N = args.get("btsp_memory_dim", 4096)  # Memory dimension
        self.p_pre = args.get("btsp_sparsity", 0.04)  # Sparsity rate
        self.num_branches = args.get("btsp_branches", 16)  # Number of branches
        self.tau_e_steps = args.get("btsp_tau_e_steps", 6)  # Eligibility trace time constant (in steps)
        self.theta = args.get("btsp_theta", 0.3)  # Eligibility trace threshold
        self.alpha_star = args.get("btsp_alpha_star", self.p_pre)  # 目标占用率 = p_pre
        self.eta = args.get("btsp_eta", 0.05)  # Homeostasis learning rate
        self.homeostasis_freq = args.get("btsp_homeostasis_freq", 100)  # Update frequency
        self.eps_0 = args.get("btsp_eps_0", 0.05)  # Tolerated interference
        self.adaptive_p_gate = args.get("btsp_adaptive", True)  # Adaptive gating
        self.memory_fusion_alpha = args.get("btsp_fusion_alpha", 0.3)  # Memory pool weight in fusion
        
        # 温度与截断、标准化相关参数
        self.mem_temperature = args.get("mem_temperature", 1.0)  # Popcount温度标定
        self.p_gate_clip = args.get("p_gate_clip", [1e-4, 0.1])  # p_gate截断区间
        self.mem_rho_est: Optional[float] = args.get("mem_rho_est", None)  # 可选：覆盖统计的有效占用近似
        
        # 显存优化配置
        self.btsp_on_cpu = args.get("btsp_on_cpu", True)  # 资格迹与占用统计放CPU
        self.use_amp = args.get("use_amp", True)  # 自动混合精度
        self.grad_accum_steps = args.get("grad_accum_steps", 1)  # 梯度累积
        # 记忆蒸馏超参（无样本）
        self.use_memory_kd: bool = args.get("use_memory_kd", True)
        self.kd_T: float = float(args.get("kd_T", 2.0))
        self.kd_lambda: float = float(args.get("kd_lambda", 0.5))
        
        # Initialize BTSP components
        self._init_btsp_components()
        
        # 简化检索缓存管理
        self.W_dirty: bool = True
        
        # Logging
        logging.info(f"BTSP Memory Pool initialized with N={self.N}, branches={self.num_branches}")
        logging.info(f"Sparsity p_pre={self.p_pre}, tau_e={self.tau_e_steps} steps, theta={self.theta}")
        logging.info(f"Expected memory per class: {self.N / 8 / 1024:.1f} KB")
        
    def after_task(self):
        """Called after each task - required by PILOT framework"""
        # 保存旧模型状态到CPU，避免显存拷贝
        self._old_network_state = {k: v.cpu() for k, v in self._network.state_dict().items()}
        self._old_network = None  # 不在GPU常驻副本
        self._known_classes = self._total_classes
        
        # 任务结束后重建检索缓存
        self._rebuild_memory_cache()
        
        # 清理显存
        torch.cuda.empty_cache()
        
        # Log memory pool statistics
        total_memory_kb = len(self.memory_weights) * self.N / 8 / 1024
        avg_occupancy = self.branch_occupancy.mean().item() if len(self.memory_weights) > 0 else 0
        logging.info(f"Task {self._cur_task} completed. Memory pool: {len(self.memory_weights)} classes, "
                    f"{total_memory_kb:.1f} KB, avg occupancy: {avg_occupancy:.3f}")
        
    def _init_btsp_components(self):
        """Initialize BTSP memory pool components"""
        # 统一 BTSP 端设备到 GPU（无梯度 + 低精度）
        self.btsp_device = self._device          # 从原来的 cpu 改为 GPU
        self.e_dtype = torch.float16             # 资格迹半精度
        self.w_dtype = torch.bool                # 权重仍然 bool
        
        # 位与分支索引（常驻 GPU，非参数）
        self.branch_idx = torch.arange(self.N, device=self.btsp_device) % self.num_branches
        self.branch_idx.requires_grad_(False)
        
        # 预存每个分支的索引（避免每步 nonzero）
        self.branch_lists = [torch.where(self.branch_idx == b)[0].to(self.btsp_device)
                            for b in range(self.num_branches)]
        
        # 预计算每个分支包含的位数，避免重复计算
        self.branch_sizes = torch.stack([
            (self.branch_idx == b).sum() for b in range(self.num_branches)
        ]).to(self.btsp_device)
        self.branch_sizes.requires_grad_(False)
        
        # Memory weights for each class [C, N] - will expand dynamically
        self.memory_weights = {}  # Dict[class_id, torch.BoolTensor]
        
        # Eligibility traces for active classes [C, N] - only for current active classes
        self.eligibility_traces = {}  # Dict[class_id, torch.FloatTensor]
        self.e_device = self.btsp_device
        
        # 门控与占用也在 GPU（数值很小，非参数）
        self.p_gate_per_branch = torch.ones(self.num_branches, device=self.btsp_device, dtype=torch.float32)
        self.p_gate_per_branch.requires_grad_(False)
        
        # Branch occupancy tracking (EMA) - 初始化为 p_pre
        self.branch_occupancy = torch.full((self.num_branches,), self.p_pre, 
                                         device=self.btsp_device, dtype=torch.float32)
        self.branch_occupancy.requires_grad_(False)
        self.occupancy_ema_factor = float(self.args.get("btsp_occupancy_ema", 0.9))
        
        # Step counter for homeostasis
        self.step_counter = 0
        
        # Decay factor for eligibility traces
        self.beta = math.exp(-1.0 / self.tau_e_steps)
        
        # Effective time window (in steps) - 修正计算公式
        self.T_eff_steps = self.tau_e_steps * math.log(1.0 / self.theta)
        
        # 维护类ID缓存用于检索
        self._class_id_cache: List[int] = []
        
        logging.info(f"T_eff = {self.T_eff_steps:.2f} steps (corrected formula), beta = {self.beta:.4f}")
        logging.info(f"BTSP components unified on GPU, dtype={self.e_dtype}")
        logging.info(f"Expected memory per class: {self.N / 8 / 1024:.1f} KB (bitset) or {self.N * self.p_pre * math.log2(self.N) / 8:.0f} bytes (sparse index)")
        
    def _compute_p_flip_max(self, M: int) -> float:
        """Compute maximum flip probability for M tasks to maintain stability
        Based on: p_flip <= (1-(1-2*eps_0)^(1/M))/2
        """
        if M <= 1:
            return 0.0  # Task0 禁止翻转：warm-start阶段只做OR强化
        return (1 - (1 - 2 * self.eps_0) ** (1.0 / M)) / 2
    
    def _solve_p_gate(self, p_flip_target: float) -> float:
        """Solve for per-branch gating probability given target flip probability
        Based on: p_gate = -ln(1 - 2*p_flip/p_pre) / T_eff
        """
        if p_flip_target <= 0 or self.p_pre <= 0:
            return 0.0
        
        ratio = 2 * p_flip_target / self.p_pre
        if ratio >= 1.0:
            logging.warning(f"Cannot achieve p_flip={p_flip_target:.6f} with p_pre={self.p_pre}, clamping ratio")
            ratio = 0.99
        
        p_gate = -math.log(1 - ratio) / self.T_eff_steps
        return max(self.p_gate_clip[0], min(p_gate, self.p_gate_clip[1]))
    
    def _update_gating_probabilities(self):
        """Update gating probabilities with correct task count and safe bounds"""
        if not self.adaptive_p_gate:
            return
        
        # 任务0：硬关翻转，只做OR强化
        if int(self._cur_task) == 0:
            self.p_gate_per_branch.fill_(0.0)
            logging.info("[p_gate] warm-start: task0 flips disabled")
            return
        
        # _cur_task 是0基的调度计数，转换为已见任务总数（含当前）
        M_seen = int(self._cur_task) + 1
        # 稳定性上界：p_flip^max
        p_flip_max = (1.0 - (1.0 - 2.0 * float(self.eps_0)) ** (1.0 / M_seen)) / 2.0
        # 可达性约束：p_flip_target 不能超过 p_pre/2 - ε
        p_flip_target = min(p_flip_max, 0.5 * self.p_pre - 1e-6)
        
        # 反解 p_gate
        ratio = (2.0 * p_flip_target) / float(self.p_pre)
        ratio = max(0.0, min(0.999, ratio))
        if ratio <= 0.0:
            p_gate = 0.0
        else:
            p_gate = - math.log(1.0 - ratio) / float(self.T_eff_steps)
        
        # 收紧上限，避免过猛翻转
        hi = min(self.p_gate_clip[1], 0.04)
        p_gate = max(self.p_gate_clip[0], min(p_gate, hi))
        self.p_gate_per_branch.fill_(float(p_gate))
        logging.info(f"[p_gate] M_seen={M_seen}, p_flip*={p_flip_target:.6g}, p_gate={p_gate:.3g}/step/branch")
    
    def _encode_features_to_bits(self, features: torch.Tensor) -> torch.Tensor:
        """Convert dense features to sparse binary codes using Top-k"""
        return self._network.encode_feats_to_bits(features, sparsity=self.p_pre)
    
    def _update_eligibility_traces(self, active_classes: List[int], x_bits_dict: Dict[int, torch.Tensor]):
        """Update eligibility traces for active classes only (GPU版向量化)"""
        with torch.no_grad():
            for class_id in active_classes:
                if class_id not in self.eligibility_traces:
                    self.eligibility_traces[class_id] = torch.zeros(self.N, dtype=self.e_dtype, device=self.btsp_device)
                
                # 按类维护：先衰减，后设置新激活
                self.eligibility_traces[class_id] *= self.beta
                if class_id in x_bits_dict:
                    # x_bits_dict[class_id] 是 [N] bool on GPU
                    self.eligibility_traces[class_id][x_bits_dict[class_id]] = 1.0
    
    @torch.no_grad()
    def _apply_branch_gating(self, active_classes: List[int], x_bits_dict: Dict[int, torch.Tensor]) -> int:
        """Apply branch-level gating and random flips with proper masking (GPU版)
        避开当前激活位，防止抵消刚写入的OR
        Returns: total_flips count
        """
        total_flips = 0
        
        # 每个分支按伯努利采样是否门控（GPU 上采样）
        gates = (torch.rand(self.num_branches, device=self.btsp_device) < self.p_gate_per_branch)
        gated = torch.where(gates)[0]  # 索引在 GPU
        
        for b in gated.tolist():
            I_b = self.branch_lists[b]                                  # [|I_b|] long on GPU
            for class_id in active_classes:
                if class_id not in self.eligibility_traces or class_id not in self.memory_weights:
                    continue
                
                e_c = self.eligibility_traces[class_id]                        # [N] float16 on GPU
                eligible_mask = (e_c[I_b] >= self.theta)                # [|I_b|] bool on GPU
                
                # 关键：避开本步当前类的激活位，防止抵消刚写入的OR
                if class_id in x_bits_dict:
                    forbid = x_bits_dict[class_id][I_b]   # [|I_b|] bool
                    eligible_mask = eligible_mask & (~forbid)
                
                if not eligible_mask.any():
                    continue
                
                elig_idx = I_b[eligible_mask]                           # [K] long on GPU
                # 0.5 硬币（GPU）
                coin = (torch.rand(elig_idx.numel(), device=self.btsp_device) < 0.5)
                if not coin.any():
                    continue
                
                flip_idx = elig_idx[coin.nonzero(as_tuple=True)[0]]     # [K'] long
                self.memory_weights[class_id][flip_idx] ^= True         # bool XOR on GPU
                total_flips += flip_idx.numel()
        
        if total_flips > 0:
            logging.debug(f"Applied {total_flips} flips across branches")
            # 记忆被修改，标记缓存失效
            self.W_dirty = True
        
        return total_flips
    
    @torch.no_grad()
    def _compute_occupancy_stats(self):
        """计算单类占用统计（每类 bitset 的1占比），避免跨类叠加偏差"""
        if not self.memory_weights:
            return 0.0, 0.0
        # 单类占用（每类 bitset 的1占比）
        occ_per_class = torch.tensor(
            [w.float().mean().item() for w in self.memory_weights.values()],
            device=self.btsp_device
        )
        occ_mean = float(occ_per_class.mean().item())

        # 分支占用：先对每类求分支占用再平均（避免跨类叠加）
        branch_occ = []
        for b in range(self.num_branches):
            I_b = self.branch_lists[b]
            bo = torch.tensor(
                [w[I_b].float().mean().item() for w in self.memory_weights.values()],
                device=self.btsp_device
            ).mean()
            branch_occ.append(bo)
        branch_occ = torch.stack(branch_occ)
        # 覆盖 EMA（使 homeostasis 立刻用真实值）
        self.branch_occupancy.copy_(branch_occ)
        return occ_mean, float(branch_occ.mean().item())
    
    def _update_homeostasis(self):
        """Update branch occupancy and adjust gating probabilities via homeostasis
        Only called every K steps for stability
        """
        if len(self.memory_weights) == 0:
            return
            
        # 计算当前分支占用率（所有类在该分支位上的1比例）
        with torch.no_grad():
            beta_ema = self.occupancy_ema_factor
            for branch_id in range(self.num_branches):
                I_b = self.branch_lists[branch_id]  # [|I_b|] long on GPU
                total_ones = torch.tensor(0.0, device=self.btsp_device)
                
                for weights in self.memory_weights.values():
                    total_ones += weights[I_b].float().sum()
                
                # 使用预计算的位数
                total_bits = self.branch_sizes[branch_id].clamp(min=1)
                current_occupancy = (total_ones / total_bits).clamp(0.0, 1.0)
                
                # EMA更新
                self.branch_occupancy[branch_id] = (
                    beta_ema * self.branch_occupancy[branch_id] + (1 - beta_ema) * current_occupancy
                )
                
                # 稳态调节：负反馈调整，目标 = p_pre
                delta = -self.eta * (self.branch_occupancy[branch_id] - self.p_pre)
                self.p_gate_per_branch[branch_id] = torch.clamp(
                    self.p_gate_per_branch[branch_id] * torch.exp(delta), 
                    self.p_gate_clip[0], self.p_gate_clip[1]
                )
        
        avg_occupancy = self.branch_occupancy.mean().item()
        avg_p_gate = self.p_gate_per_branch.mean().item()
        logging.debug(f"Homeostasis: avg_occupancy={avg_occupancy:.4f} (target={self.p_pre}), avg_p_gate={avg_p_gate:.6f}")
    
    def _write_to_memory_pool(self, data_loader: DataLoader, class_list: List[int]):
        """Write samples to BTSP memory pool"""
        self._network.eval()
        
        # 更新门控概率（按任务数）
        self._update_gating_probabilities()
        # 进入写阶段，标记缓存失效
        self.W_dirty = True
        
        # 诊断计数器
        batch_count = 0
        total_or_ones = 0
        total_flips = 0
        
        with torch.no_grad():
            for _, inputs, labels in data_loader:
                batch_count += 1
                batch_or_ones = 0
                inputs = inputs.to(self._device)
                labels = labels.to(self._device, dtype=torch.long)
                
                # 每个batch的活跃类
                active_classes = labels.unique().tolist()
                
                # 如首次出现则初始化该类的记忆比特（在 GPU 上）
                for class_id in active_classes:
                    if class_id not in self.memory_weights:
                        self.memory_weights[class_id] = torch.zeros(self.N, dtype=self.w_dtype, device=self.btsp_device)
                
                # 提取特征并转为二进制码
                features = self._network.extract_vector(inputs)
                
                # 按类聚合 OR
                x_bits_dict: Dict[int, torch.Tensor] = {}
                for class_id in active_classes:
                    class_mask = (labels == class_id)
                    if class_mask.any():
                        class_features = features[class_mask]
                        class_bits = self._encode_features_to_bits(class_features)
                        # 确保二进制码在 GPU 上
                        class_bits = class_bits.to(self.btsp_device)
                        x_bits_dict[class_id] = class_bits.any(dim=0)
                        batch_or_ones += x_bits_dict[class_id].sum().item()
                
                total_or_ones += batch_or_ones
                # Bounded-OR：限制单类 1 的数量在 K_target 左右
                K_target = int(round(self.p_pre * self.N))
                k_step_cap = 16  # 每步最多新置1的个数，8~16均可
                
                for class_id, bits in x_bits_dict.items():  # bits: [N] bool (本步聚合激活位)
                    w = self.memory_weights.get(class_id)
                    if w is None:
                        w = torch.zeros(self.N, dtype=self.w_dtype, device=self.btsp_device)
                        self.memory_weights[class_id] = w

                    # 仅从"当前为0且本步激活"的位中，限额置1
                    zeros_mask = (~w) & bits
                    need = max(0, K_target - int(w.sum().item()))
                    if need > 0:
                        pick = min(need, k_step_cap, int(zeros_mask.sum().item()))
                        if pick > 0:
                            idx = torch.nonzero(zeros_mask, as_tuple=False).flatten()
                            choice = idx[torch.randperm(idx.numel(), device=idx.device)[:pick]]
                            w[choice] = True
                
                # 更新资格轨迹（把这些位的e置1）
                self._update_eligibility_traces(active_classes, x_bits_dict)
                
                # 分支门控与随机翻转（传入x_bits_dict，避开当前激活位）
                batch_flips = self._apply_branch_gating(active_classes, x_bits_dict)
                total_flips += batch_flips
                
                # 周期性稳态调节
                self.step_counter += 1
                if self.step_counter % self.homeostasis_freq == 0:
                    self._update_homeostasis()
                
                # 每100批次打印诊断信息
                if batch_count % 100 == 0:
                    logging.info(f"Task {self._cur_task} batch {batch_count}: OR_ones={batch_or_ones}, flips={batch_flips}")
        
        # 最终诊断统计
        logging.info(f"Task {self._cur_task} write phase: total_OR_ones={total_or_ones}, total_flips={total_flips}")
        
        # 使用单类口径计算占用统计，并覆盖EMA
        mean_cls, mean_branch = self._compute_occupancy_stats()
        logging.info(f"Post-write occupancy: mean_per_class={mean_cls:.4f} ({mean_cls*100:.2f}%), "
                     f"mean_branch={mean_branch:.4f} ({mean_branch*100:.2f}%)")
        
        # 更新类ID缓存
        self._class_id_cache = sorted(self.memory_weights.keys())
        
        # 写完本轮后重建检索缓存
        self._rebuild_memory_cache()
    
    def _rebuild_memory_cache(self):
        """Simple rebuild - no longer needed with direct GPU access"""
        # 简化：直接维护类ID列表即可，不需要预堆叠矩阵
        self._class_id_cache = sorted(self.memory_weights.keys())
        self.W_dirty = False
        logging.debug(f"Updated class cache: C={len(self._class_id_cache)}")
    
    def _ensure_memory_cache_gpu(self):
        """向后兼容接口，已简化"""
        if self.W_dirty:
            self._rebuild_memory_cache()
    
    def _ensure_memory_cache(self):
        """向后兼容接口，已简化"""
        if self.W_dirty:
            self._rebuild_memory_cache()

    @torch.no_grad()
    def _retrieve_from_memory_pool(self, inputs: torch.Tensor) -> torch.Tensor:
        """Retrieve similarity scores from memory pool with temperature calibration (GPU直接版)"""
        if len(self.memory_weights) == 0:
            return torch.zeros(len(inputs), 0, device=self._device)

        # Extract features and convert to binary codes
        features = self._network.extract_vector(inputs)
        x_bits = self._encode_features_to_bits(features).to(self.btsp_device)  # [B, N] bool on GPU

        # 堆成 [C,N] bool（直接在 GPU 做矩阵乘）
        class_ids = self._class_id_cache  # 维护一个有序列表
        if len(class_ids) == 0:
            return torch.zeros(len(inputs), 0, device=self._device)
        W_gpu_bool = torch.stack([self.memory_weights[c] for c in class_ids], dim=0)  # [C,N] bool
        W_gpu_float = W_gpu_bool.float()  # 转换为float以便计算
        memory_scores = (x_bits.float() @ W_gpu_float.T)  # [B,C]

        # 估计每样本激活数 k，并按占用率计算期望与方差
        k = x_bits.sum(dim=1)  # [B]
        if self.mem_rho_est is not None:
            # 使用配置覆盖的有效占用率（标量）
            alpha_eff = torch.tensor(float(self.mem_rho_est), device=self.btsp_device)
            mu = k.unsqueeze(1) * alpha_eff  # [B, C] 广播
            var = k.unsqueeze(1) * alpha_eff * (1 - alpha_eff)
        else:
            # 使用每类实际占用率（先转float再计算mean）
            alpha_c = W_gpu_float.mean(dim=1)  # [C]
            mu = k.unsqueeze(1) * alpha_c.unsqueeze(0)  # [B, C]
            var = k.unsqueeze(1) * alpha_c.unsqueeze(0) * (1 - alpha_c.unsqueeze(0))
        std = torch.sqrt(var + 1e-6)

        # z-score 与温度缩放
        z_scores = (memory_scores - mu) / std
        mem_logits = z_scores / self.mem_temperature

        # 返回到主设备
        return mem_logits.to(self._device)
    
    def incremental_train(self, data_manager):
        """Train on current task and update BTSP memory pool (PILOT-compatible)."""
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info(f"[Task {self._cur_task}] classes {self._known_classes}..{self._total_classes-1}")

        # 1) dataloaders
        train_set = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                             source="train", mode="train")
        test_set  = data_manager.get_dataset(np.arange(0, self._total_classes),
                                             source="test",  mode="test")
        self.train_loader = DataLoader(train_set, batch_size=128, shuffle=True,  num_workers=num_workers)
        self.test_loader  = DataLoader(test_set,  batch_size=128, shuffle=False, num_workers=num_workers)

        # 2) 轻量微调（避免漂移）
        self._train_standard(self.train_loader, self.test_loader)

        # 3) 写入 BTSP memory（只写本任务类）
        current_classes = list(range(self._known_classes, self._total_classes))
        self._write_to_memory_pool(self.train_loader, current_classes)

        self._known_classes = self._total_classes
        logging.info(f"Memory pool size: {len(self.memory_weights)} classes")
    
    def _train_standard(self, train_loader: DataLoader, test_loader: DataLoader):
        """Standard supervised training on current task with memory-efficient two-path forward"""
        self._network.train()
        
        # 冻结骨干，只训练头和投影层以节省显存
        self._network.freeze_backbone(eval_mode=True)
        
        # 验证参数冻结情况
        all_params = sum(p.numel() for p in self._network.parameters())
        trainable_params = sum(p.numel() for p in self._network.parameters() if p.requires_grad)
        logging.info(f"Parameters: all={all_params:,}, trainable={trainable_params:,} ({100*trainable_params/all_params:.1f}%)")
        
        # Setup optimizer and scheduler - 只优化可训练参数
        optimizer_type = self.args.get("optimizer", "sgd").lower()
        trainable_params = self._network.get_trainable_parameters()
        
        if optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.args["lr"],
                weight_decay=self.args["weight_decay"]
            )
        elif optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                trainable_params,
                lr=self.args["lr"],
                weight_decay=self.args["weight_decay"]
            )
        else:  # default SGD
            optimizer = torch.optim.SGD(
                trainable_params,
                lr=self.args["lr"],
                momentum=0.9,
                weight_decay=self.args["weight_decay"]
            )
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.args["milestones"], gamma=self.args["lr_decay"]
        )
        
        # AMP for memory efficiency
        scaler = torch.amp.GradScaler('cuda', enabled=True)
        
        # Gradient accumulation config
        grad_accum_steps = self.args.get("grad_accum_steps", 1)
        effective_batch_size = self.args.get("batch_size", 128) * grad_accum_steps
        
        # Training loop
        for epoch in range(self.args["epochs"]):
            losses = []
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device, dtype=torch.long)
                
                # 1) 特征提取（冻结骨干）
                with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
                    feat_nograd = self._network.extract_vector(inputs)
                
                # 2) 头部前向 + 记忆蒸馏
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    feat_detached = feat_nograd.detach()
                    logits = self._network.head_forward(feat_detached)
                    ce = F.cross_entropy(logits, targets)
                    loss = ce
                    # 仅在已存在旧类时加入蒸馏，稳住旧类列分布（无样本）
                    if self._known_classes > 0 and self.use_memory_kd:
                        with torch.no_grad():
                            mem_logits = self._retrieve_from_memory_pool(inputs)  # [B, C_mem]
                            # 对齐memory分数到total_classes
                            aligned_mem = torch.zeros_like(logits)
                            for i_cls, class_id in enumerate(self._class_id_cache):
                                if class_id < aligned_mem.size(1):
                                    aligned_mem[:, class_id] = mem_logits[:, i_cls]
                            p_t_old = F.softmax(aligned_mem[:, :self._known_classes] / self.kd_T, dim=1)
                        log_p_s_old = F.log_softmax(logits[:, :self._known_classes] / self.kd_T, dim=1)
                        kd = F.kl_div(log_p_s_old, p_t_old, reduction='batchmean') * (self.kd_T ** 2)
                        loss = loss + self.kd_lambda * kd
                    # 梯度累积缩放
                    loss = loss / grad_accum_steps
                
                # 反传与优化
                scaler.scale(loss).backward()
                
                # 梯度累积步骤
                if (i + 1) % grad_accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                
                losses.append(loss.item() * grad_accum_steps)
                
                # Log progress
                if i % 100 == 0:
                    logging.info(f"Task {self._cur_task}, epoch {epoch}, batch {i}: loss={loss.item() * grad_accum_steps:.4f}")
            
            scheduler.step()
            
            # Evaluate every few epochs
            if epoch % 5 == 0 or epoch == self.args["epochs"] - 1:
                test_acc = self._compute_accuracy(self._network, test_loader)
                logging.info(f"Task {self._cur_task}, epoch {epoch}: loss={np.mean(losses):.4f}, test_acc={test_acc:.2f}%")
                
        # 清理显存
        torch.cuda.empty_cache()
    
    def _train(self, train_loader: DataLoader, test_loader: DataLoader):
        """Internal training method required by PILOT framework"""
        # Use the same implementation as _train_standard
        self._train_standard(train_loader, test_loader)
        
    def eval_task(self):
        """Evaluate model on all seen tasks - required by PILOT framework"""
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)
        
        # BTSP doesn't use NME, so return None for nme_accy
        nme_accy = None
        
        return cnn_accy, nme_accy
    
    def _eval_cnn(self, loader: DataLoader):
        """Evaluate CNN predictions using memory fusion with detailed breakdown"""
        self._network.eval()
        y_pred, y_true = [], []
        y_pred_head, y_pred_mem = [], []  # 分路统计
        
        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device, dtype=torch.long)
                
                # 三路前向：head-only / mem-only / fused
                standard_logits = self._network(inputs)["logits"]  # [B, total_classes]
                mem_logits = self._retrieve_from_memory_pool(inputs)  # [B, num_memory_classes]
                
                # 对齐memory分数到total_classes
                aligned_mem = torch.zeros_like(standard_logits)
                if mem_logits.size(1) > 0:
                    for i, class_id in enumerate(self._class_id_cache):
                        if class_id < standard_logits.size(1):
                            aligned_mem[:, class_id] = mem_logits[:, i]
                
                # 融合logits
                fused_logits = ((1 - self.memory_fusion_alpha) * standard_logits + 
                               self.memory_fusion_alpha * aligned_mem)
                
                # 三路预测
                predicts_fused = torch.topk(fused_logits, k=self.topk, dim=1, largest=True, sorted=True)[1]
                predicts_head = torch.topk(standard_logits, k=self.topk, dim=1, largest=True, sorted=True)[1]
                predicts_mem = torch.topk(aligned_mem, k=self.topk, dim=1, largest=True, sorted=True)[1]
                
                y_pred.append(predicts_fused.cpu().numpy())
                y_pred_head.append(predicts_head.cpu().numpy())
                y_pred_mem.append(predicts_mem.cpu().numpy())
                y_true.append(targets.cpu().numpy())
        
        # 计算三路准确率用于诊断
        y_true_cat = np.concatenate(y_true)
        y_pred_fused_cat = np.concatenate(y_pred)
        y_pred_head_cat = np.concatenate(y_pred_head)
        y_pred_mem_cat = np.concatenate(y_pred_mem)
        
        # Top-1准确率
        acc_fused = (y_pred_fused_cat[:, 0] == y_true_cat).mean() * 100
        acc_head = (y_pred_head_cat[:, 0] == y_true_cat).mean() * 100
        acc_mem = (y_pred_mem_cat[:, 0] == y_true_cat).mean() * 100
        
        logging.info(f"Eval breakdown: head-only={acc_head:.1f}%, mem-only={acc_mem:.1f}%, fused={acc_fused:.1f}%")
        
        return np.concatenate(y_pred), np.concatenate(y_true)
    
    def _forward_with_memory_fusion(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass with memory pool fusion"""
        # Standard forward pass
        outputs = self._network(inputs)
        standard_logits = outputs["logits"]  # [B, total_classes]
        
        # Memory pool retrieval
        memory_scores = self._retrieve_from_memory_pool(inputs)  # [B, num_memory_classes]
        
        if memory_scores.size(1) == 0:
            # No memory available
            return standard_logits
        
        # Align memory scores with total classes
        aligned_memory_scores = torch.zeros_like(standard_logits)
        memory_class_ids = self._class_id_cache
        
        for i, class_id in enumerate(memory_class_ids):
            if class_id < standard_logits.size(1):
                aligned_memory_scores[:, class_id] = memory_scores[:, i]
        
        # Weighted fusion
        fused_logits = ((1 - self.memory_fusion_alpha) * standard_logits + 
                       self.memory_fusion_alpha * aligned_memory_scores)
        
        return fused_logits
    
    def _compute_accuracy(self, model: nn.Module, loader: DataLoader) -> float:
        """Compute accuracy on given data loader"""
        model.eval()
        correct, total = 0, 0
        
        with torch.no_grad():
            for _, inputs, targets in loader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device, dtype=torch.long)
                
                outputs = self._forward_with_memory_fusion(inputs)
                _, predicted = torch.max(outputs, 1)
                
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        model.train()
        return 100.0 * correct / total
