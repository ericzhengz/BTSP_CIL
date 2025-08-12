#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-computed BTSP Parameter Tables

Based on your parameter calculation formulas, here are ready-to-use parameter sets
for different scenarios. Copy these into your experiment configs.
"""

import math

def compute_parameters(N, p_pre, M, eps_0, tau_e_steps, theta, num_branches):
    """Compute all BTSP parameters for given settings"""
    # Core computations
    T_eff_steps = tau_e_steps * math.log(1.0 / theta)
    p_flip_max = (1 - (1 - 2 * eps_0) ** (1.0 / M)) / 2 if M > 1 else 0.01
    
    # Solve for p_gate
    ratio = 2 * p_flip_max / p_pre
    if ratio >= 1.0:
        p_gate = 0.01  # Fallback
    else:
        p_gate = -math.log(1 - ratio) / T_eff_steps
    
    # Expected gating
    expected_gates_per_step = num_branches * p_gate
    
    # Theoretical capacity (binary entropy)
    x = p_pre + p_flip_max - 2 * p_pre * p_flip_max
    H_x = -x * math.log(x) - (1-x) * math.log(1-x) if 0 < x < 1 else 1e-6
    capacity = N / H_x
    
    return {
        "T_eff_steps": T_eff_steps,
        "p_flip_max": p_flip_max,
        "p_gate": p_gate,
        "expected_gates_per_step": expected_gates_per_step,
        "theoretical_capacity": capacity,
        "safety_margin": capacity / M if M > 0 else float('inf')
    }

# Pre-computed parameter tables
print("=== BTSP Parameter Tables ===\n")

# Table 1: Standard CIFAR-100 (10 tasks)
print("1. CIFAR-100 Standard (10 tasks, 100 classes)")
print("   Recommended: N=4096, p_pre=0.04, τ_e=6 steps, θ=0.3")
params1 = compute_parameters(N=4096, p_pre=0.04, M=10, eps_0=0.05, 
                           tau_e_steps=6, theta=0.3, num_branches=16)
print(f"   → p_flip_max = {params1['p_flip_max']:.6f}")
print(f"   → p_gate = {params1['p_gate']:.6f}")
print(f"   → Expected gates/step = {params1['expected_gates_per_step']:.2f}")
print(f"   → Theoretical capacity = {params1['theoretical_capacity']:.1f} tasks")
print(f"   → Safety margin = {params1['safety_margin']:.2f}x")

config1 = f'''
{{
    "btsp_memory_dim": 4096,
    "btsp_sparsity": 0.04,
    "btsp_branches": 16,
    "btsp_tau_e_steps": 6,
    "btsp_theta": 0.3,
    "btsp_alpha_star": 0.02,
    "btsp_eta": 0.05,
    "btsp_homeostasis_freq": 100,
    "btsp_eps_0": 0.05,
    "btsp_adaptive": true
}}
'''
print(f"   Config: {config1}")

print("\n" + "="*60 + "\n")

# Table 2: ImageNet-R (20 tasks)  
print("2. ImageNet-R (20 tasks, 200 classes)")
print("   Recommended: N=8192, p_pre=0.04, τ_e=6 steps, θ=0.3")
params2 = compute_parameters(N=8192, p_pre=0.04, M=20, eps_0=0.05,
                           tau_e_steps=6, theta=0.3, num_branches=32)
print(f"   → p_flip_max = {params2['p_flip_max']:.6f}")
print(f"   → p_gate = {params2['p_gate']:.6f}")
print(f"   → Expected gates/step = {params2['expected_gates_per_step']:.2f}")
print(f"   → Theoretical capacity = {params2['theoretical_capacity']:.1f} tasks")
print(f"   → Safety margin = {params2['safety_margin']:.2f}x")

config2 = f'''
{{
    "btsp_memory_dim": 8192,
    "btsp_sparsity": 0.04,
    "btsp_branches": 32,
    "btsp_tau_e_steps": 6,
    "btsp_theta": 0.3,
    "btsp_alpha_star": 0.02,
    "btsp_eta": 0.05,
    "btsp_homeostasis_freq": 100,
    "btsp_eps_0": 0.05,
    "btsp_adaptive": true
}}
'''
print(f"   Config: {config2}")

print("\n" + "="*60 + "\n")

# Table 3: Capacity cliff experiment (small N)
print("3. Capacity Cliff Experiment (to observe 断崖)")
print("   Smaller N=2048 to reach capacity faster")
params3 = compute_parameters(N=2048, p_pre=0.04, M=15, eps_0=0.05,
                           tau_e_steps=6, theta=0.3, num_branches=16)
print(f"   → p_flip_max = {params3['p_flip_max']:.6f}")
print(f"   → p_gate = {params3['p_gate']:.6f}")
print(f"   → Expected gates/step = {params3['expected_gates_per_step']:.2f}")
print(f"   → Theoretical capacity = {params3['theoretical_capacity']:.1f} tasks")
print(f"   → Safety margin = {params3['safety_margin']:.2f}x")
print(f"   ⚠️  Expect cliff around task {int(params3['theoretical_capacity'])}")

config3 = f'''
{{
    "btsp_memory_dim": 2048,
    "btsp_sparsity": 0.04,
    "btsp_branches": 16,
    "btsp_tau_e_steps": 6,
    "btsp_theta": 0.3,
    "btsp_alpha_star": 0.02,
    "btsp_eta": 0.05,
    "btsp_homeostasis_freq": 50,
    "btsp_eps_0": 0.05,
    "btsp_adaptive": false,
    "_note": "Fixed gating for cliff observation"
}}
'''
print(f"   Config: {config3}")

print("\n" + "="*60 + "\n")

# Table 4: Parameter sensitivity examples
print("4. Parameter Sensitivity Examples")

sensitivity_cases = [
    ("Fast forgetting (more plastic)", {"tau_e_steps": 8, "theta": 0.2}),
    ("Slow forgetting (more stable)", {"tau_e_steps": 4, "theta": 0.5}),
    ("Higher sparsity", {"p_pre": 0.06}),
    ("Lower sparsity", {"p_pre": 0.03})
]

base_params = {"N": 4096, "M": 10, "eps_0": 0.05, "tau_e_steps": 6, 
               "theta": 0.3, "num_branches": 16, "p_pre": 0.04}

for name, overrides in sensitivity_cases:
    params = base_params.copy()
    params.update(overrides)
    result = compute_parameters(**params)
    
    print(f"   {name}:")
    for key, value in overrides.items():
        print(f"     {key} = {value}")
    print(f"     → p_gate = {result['p_gate']:.6f}")
    print(f"     → capacity = {result['theoretical_capacity']:.1f}")
    print()

print("=== Usage Instructions ===")
print("1. Copy the appropriate config into your .json file")
print("2. Run: python test_btsp.py  # to verify implementation")
print("3. Run: python btsp_analysis.py --mode validate_params --config your_config.json")
print("4. Run: python main.py --config your_config.json")
print("\n=== Parameter Tuning Guide ===")
print("If forgetting too fast → decrease p_gate (or increase tau_e_steps, decrease theta)")
print("If learning too slow → increase p_gate (or decrease tau_e_steps, increase theta)")
print("If capacity insufficient → increase N or decrease p_pre")
print("If want to see cliff → use config 3 and test up to 15-20 tasks")
