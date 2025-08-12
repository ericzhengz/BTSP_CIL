#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTSP Parameter Validation and Capacity Analysis Script

This script provides utilities to:
1. Validate theoretical parameter mappings
2. Run capacity cliff experiments  
3. Analyze homeostasis dynamics
4. Generate parameter recommendations

Usage:
    python btsp_analysis.py --mode validate_params --config exps/btsp_mp.json
    python btsp_analysis.py --mode capacity_cliff --config exps/btsp_capacity_cliff.json
"""

import argparse
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List


def compute_p_flip_max(M: int, eps_0: float = 0.05) -> float:
    """Compute maximum flip probability for M tasks"""
    if M <= 1:
        return 0.01
    return (1 - (1 - 2 * eps_0) ** (1.0 / M)) / 2


def solve_p_gate(p_flip_target: float, p_pre: float, tau_e_steps: float, theta: float) -> float:
    """Solve for per-branch gating probability"""
    if p_flip_target <= 0 or p_pre <= 0:
        return 0.0
    
    T_eff_steps = tau_e_steps * math.log(1.0 / theta)
    ratio = 2 * p_flip_target / p_pre
    
    if ratio >= 1.0:
        print(f"WARNING: Cannot achieve p_flip={p_flip_target:.6f} with p_pre={p_pre}")
        ratio = 0.99
    
    return -math.log(1 - ratio) / T_eff_steps


def binary_entropy(p: float) -> float:
    """Compute binary entropy H(p) in nats"""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log(p) - (1 - p) * math.log(1 - p)


def convolution_error_rate(p_pre: float, p_flip: float) -> float:
    """Compute convolution error rate x = p_pre * p_flip"""
    return p_pre + p_flip - 2 * p_pre * p_flip


def theoretical_capacity(N: int, p_pre: float, p_flip: float, rho_max: float = 0.0) -> float:
    """Compute theoretical task capacity"""
    x = convolution_error_rate(p_pre, p_flip)
    H_x = binary_entropy(x)
    if H_x <= 0:
        return float('inf')
    return N * (1 - rho_max) / H_x


def validate_parameters(config: Dict) -> Dict[str, float]:
    """Validate and compute BTSP parameters from config"""
    N = config.get("btsp_memory_dim", 4096)
    p_pre = config.get("btsp_sparsity", 0.04)
    tau_e_steps = config.get("btsp_tau_e_steps", 6)
    theta = config.get("btsp_theta", 0.3)
    num_branches = config.get("btsp_branches", 16)
    eps_0 = config.get("btsp_eps_0", 0.05)
    
    # Estimate number of tasks from dataset
    total_classes = {"cifar224": 100, "imagenetr": 200}.get(config.get("dataset", "cifar224"), 100)
    increment = config.get("increment", 10)
    M = total_classes // increment
    
    # Compute derived parameters
    T_eff_steps = tau_e_steps * math.log(1.0 / theta)
    p_flip_max = compute_p_flip_max(M, eps_0)
    p_gate = solve_p_gate(p_flip_max, p_pre, tau_e_steps, theta)
    expected_gated_branches = num_branches * p_gate
    
    # Theoretical capacity
    C_tasks = theoretical_capacity(N, p_pre, p_flip_max)
    
    results = {
        "N": N,
        "p_pre": p_pre,
        "M_tasks": M,
        "T_eff_steps": T_eff_steps,
        "p_flip_max": p_flip_max,
        "p_gate_per_branch": p_gate,
        "expected_gated_branches_per_step": expected_gated_branches,
        "theoretical_capacity": C_tasks,
        "capacity_vs_tasks_ratio": C_tasks / M if M > 0 else float('inf')
    }
    
    return results


def plot_parameter_curves(config: Dict, save_path: str = None):
    """Plot parameter relationships and capacity curves"""
    params = validate_parameters(config)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. p_flip_max vs number of tasks
    M_range = np.arange(1, 51)
    p_flip_values = [compute_p_flip_max(M, config.get("btsp_eps_0", 0.05)) for M in M_range]
    
    axes[0, 0].plot(M_range, p_flip_values, 'b-', linewidth=2)
    axes[0, 0].axvline(params["M_tasks"], color='r', linestyle='--', label=f'Current M={params["M_tasks"]}')
    axes[0, 0].set_xlabel('Number of Tasks (M)')
    axes[0, 0].set_ylabel('Max Flip Probability')
    axes[0, 0].set_title('Stability Constraint: p_flip_max vs M')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 2. Theoretical capacity vs p_pre
    p_pre_range = np.linspace(0.01, 0.1, 100)
    capacity_values = [theoretical_capacity(params["N"], p, params["p_flip_max"]) for p in p_pre_range]
    
    axes[0, 1].plot(p_pre_range, capacity_values, 'g-', linewidth=2)
    axes[0, 1].axvline(params["p_pre"], color='r', linestyle='--', label=f'Current p_pre={params["p_pre"]}')
    axes[0, 1].axhline(params["M_tasks"], color='orange', linestyle=':', label='Required capacity')
    axes[0, 1].set_xlabel('Sparsity Rate (p_pre)')
    axes[0, 1].set_ylabel('Theoretical Capacity (tasks)')
    axes[0, 1].set_title('Capacity vs Sparsity Rate')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_ylim(0, min(200, max(capacity_values) * 1.1))
    
    # 3. Memory interference over time
    steps = np.arange(1, params["M_tasks"] + 1)
    interference = [0.5 * (1 - (1 - 2 * params["p_flip_max"]) ** m) for m in steps]
    
    axes[1, 0].plot(steps, interference, 'purple', linewidth=2)
    axes[1, 0].axhline(config.get("btsp_eps_0", 0.05), color='r', linestyle='--', label='Tolerance ε₀')
    axes[1, 0].set_xlabel('Task Number')
    axes[1, 0].set_ylabel('Expected Interference')
    axes[1, 0].set_title('Memory Interference Accumulation')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # 4. Parameter summary table
    axes[1, 1].axis('off')
    param_text = f"""
Parameter Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Memory Dimension (N): {params['N']:,}
Sparsity Rate (p_pre): {params['p_pre']:.3f}
Number of Tasks (M): {params['M_tasks']}
Effective Window: {params['T_eff_steps']:.1f} steps

Derived Parameters:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Max Flip Prob: {params['p_flip_max']:.6f}
Gate Prob/Branch: {params['p_gate_per_branch']:.6f}
Expected Gates/Step: {params['expected_gated_branches_per_step']:.2f}

Capacity Analysis:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Theoretical Capacity: {params['theoretical_capacity']:.1f} tasks
Safety Margin: {params['capacity_vs_tasks_ratio']:.2f}x
"""
    axes[1, 1].text(0.05, 0.95, param_text, transform=axes[1, 1].transAxes, 
                    fontfamily='monospace', fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Parameter analysis plot saved to {save_path}")
    
    plt.show()
    return params


def generate_parameter_grid(base_config: Dict) -> List[Dict]:
    """Generate parameter grid for sensitivity analysis"""
    # Key parameters to vary
    p_pre_values = [0.02, 0.03, 0.04, 0.05, 0.06]
    tau_e_values = [4, 6, 8, 10]
    theta_values = [0.2, 0.3, 0.4, 0.5]
    
    configs = []
    for p_pre in p_pre_values:
        for tau_e in tau_e_values:
            for theta in theta_values:
                config = base_config.copy()
                config.update({
                    "btsp_sparsity": p_pre,
                    "btsp_tau_e_steps": tau_e,
                    "btsp_theta": theta
                })
                configs.append(config)
    
    return configs


def capacity_cliff_analysis(config: Dict):
    """Analyze the theoretical capacity cliff"""
    params = validate_parameters(config)
    
    print("=== Capacity Cliff Analysis ===")
    print(f"Memory dimension N = {params['N']:,}")
    print(f"Sparsity p_pre = {params['p_pre']:.3f}")
    print(f"Max flip prob = {params['p_flip_max']:.6f}")
    print(f"Theoretical capacity = {params['theoretical_capacity']:.1f} tasks")
    print(f"Planned tasks = {params['M_tasks']}")
    
    if params['capacity_vs_tasks_ratio'] < 1.2:
        print("⚠️  WARNING: Very close to theoretical capacity limit!")
        print("   Expect sharp performance degradation.")
    elif params['capacity_vs_tasks_ratio'] < 2.0:
        print("⚠️  CAUTION: Approaching capacity limit.")
        print("   Monitor for gradual degradation.")
    else:
        print("✅ Safe operating region - capacity margin available.")
    
    # Recommend cliff experiment parameters
    cliff_tasks = max(int(params['theoretical_capacity'] * 1.2), params['M_tasks'] + 5)
    print(f"\n📊 Recommended cliff experiment:")
    print(f"   Test up to {cliff_tasks} tasks to observe capacity cliff")
    print(f"   Expected cliff around task {int(params['theoretical_capacity'])}")


def main():
    parser = argparse.ArgumentParser(description="BTSP Parameter Analysis")
    parser.add_argument("--mode", choices=["validate_params", "capacity_cliff", "param_grid"], 
                       default="validate_params", help="Analysis mode")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    parser.add_argument("--output", type=str, help="Output plot path")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    if args.mode == "validate_params":
        params = plot_parameter_curves(config, args.output)
        print("\n=== Parameter Validation Results ===")
        for key, value in params.items():
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")
    
    elif args.mode == "capacity_cliff":
        capacity_cliff_analysis(config)
    
    elif args.mode == "param_grid":
        configs = generate_parameter_grid(config)
        print(f"Generated {len(configs)} parameter combinations")
        
        # Quick analysis of first few
        for i, cfg in enumerate(configs[:5]):
            print(f"\nConfig {i+1}:")
            params = validate_parameters(cfg)
            print(f"  p_pre={cfg['btsp_sparsity']:.3f}, tau_e={cfg['btsp_tau_e_steps']}, theta={cfg['btsp_theta']}")
            print(f"  → p_gate={params['p_gate_per_branch']:.6f}, capacity={params['theoretical_capacity']:.1f}")


if __name__ == "__main__":
    main()
