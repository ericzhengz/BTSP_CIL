#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test script for BTSP implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from models.btsp_mp import BTSPMemoryPool

def test_btsp_initialization():
    """Test basic BTSP initialization"""
    print("Testing BTSP initialization...")
    
    args = {
        "device": ["cuda:0" if torch.cuda.is_available() else "cpu"],
        "btsp_memory_dim": 1024,
        "btsp_sparsity": 0.04,
        "btsp_branches": 8,
        "btsp_tau_e_steps": 4,
        "btsp_theta": 0.3,
        "btsp_alpha_star": 0.02,
        "btsp_eta": 0.05,
        "btsp_homeostasis_freq": 50,
        "btsp_eps_0": 0.05,
        "btsp_adaptive": True,
        "backbone_type": "vit_btsp",
        "pretrained": True,
        "num_classes": 0
    }
    
    try:
        learner = BTSPMemoryPool(args)
        print(f"✅ BTSP initialized successfully")
        print(f"   Memory dimension: {learner.N}")
        print(f"   Branches: {learner.num_branches}")
        print(f"   T_eff: {learner.T_eff_steps:.2f} steps")
        print(f"   Beta (decay): {learner.beta:.4f}")
        
        return learner
    except Exception as e:
        print(f"❌ BTSP initialization failed: {e}")
        return None

def test_parameter_computation():
    """Test parameter computation functions"""
    print("\nTesting parameter computations...")
    
    from btsp_analysis import compute_p_flip_max, solve_p_gate, theoretical_capacity
    
    # Test cases
    test_cases = [
        {"M": 10, "eps_0": 0.05},
        {"M": 20, "eps_0": 0.05}, 
        {"M": 50, "eps_0": 0.05}
    ]
    
    for case in test_cases:
        p_flip_max = compute_p_flip_max(case["M"], case["eps_0"])
        p_gate = solve_p_gate(p_flip_max, 0.04, 6, 0.3)
        capacity = theoretical_capacity(4096, 0.04, p_flip_max)
        
        print(f"   M={case['M']:2d}: p_flip_max={p_flip_max:.6f}, p_gate={p_gate:.6f}, capacity={capacity:.1f}")

def test_memory_operations():
    """Test basic memory operations"""
    print("\nTesting memory operations...")
    
    learner = test_btsp_initialization()
    if learner is None:
        return
    
    # Test binary encoding
    fake_features = torch.randn(4, 768, device=learner._device)
    try:
        binary_codes = learner._encode_features_to_bits(fake_features)
        sparsity = binary_codes.float().mean().item()
        print(f"✅ Binary encoding: shape={binary_codes.shape}, sparsity={sparsity:.3f}")
    except Exception as e:
        print(f"❌ Binary encoding failed: {e}")
        return
    
    # Test eligibility trace update
    try:
        active_classes = [0, 1]
        x_bits_dict = {0: binary_codes[0], 1: binary_codes[1]}
        learner._update_eligibility_traces(active_classes, x_bits_dict)
        print(f"✅ Eligibility traces updated for classes {active_classes}")
    except Exception as e:
        print(f"❌ Eligibility trace update failed: {e}")
        return
    
    # Test memory initialization
    try:
        for class_id in active_classes:
            if class_id not in learner.memory_weights:
                learner.memory_weights[class_id] = torch.zeros(learner.N, dtype=torch.bool, device=learner._device)
        print(f"✅ Memory weights initialized for classes {active_classes}")
    except Exception as e:
        print(f"❌ Memory weight initialization failed: {e}")
        return
    
    # Test branch gating
    try:
        learner._apply_branch_gating(active_classes)
        print(f"✅ Branch gating applied")
    except Exception as e:
        print(f"❌ Branch gating failed: {e}")
        return
    
    # Test homeostasis
    try:
        learner._update_homeostasis()
        avg_occupancy = learner.branch_occupancy.mean().item()
        print(f"✅ Homeostasis updated, avg occupancy: {avg_occupancy:.4f}")
    except Exception as e:
        print(f"❌ Homeostasis update failed: {e}")

def main():
    print("=== BTSP Implementation Test ===")
    
    # Check dependencies
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    # Run tests
    test_parameter_computation()
    test_memory_operations()
    
    print("\n=== Test Summary ===")
    print("If all tests passed ✅, you can proceed to run:")
    print("  python btsp_analysis.py --mode validate_params --config exps/btsp_mp.json")
    print("  python main.py --config exps/btsp_mp.json")

if __name__ == "__main__":
    main()
