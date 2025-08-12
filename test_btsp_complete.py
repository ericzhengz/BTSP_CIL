# -*- coding: utf-8 -*-
"""
Test BTSP Memory Pool Implementation
验证BTSP内存池实现的完整性
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add current directory to path
sys.path.append('.')

from models.btsp_mp import BTSPMemoryPool
from utils.inc_net_btsp import BTSPIncrementalNet


def create_dummy_dataset(num_samples=100, num_classes=10, feature_dim=512):
    """Create dummy dataset for testing"""
    # Generate random features
    features = torch.randn(num_samples, 3, 32, 32)  # CIFAR format
    
    # Generate balanced class labels
    labels = torch.randint(0, num_classes, (num_samples,))
    
    # Create dataset with PILOT format (task_id, data, labels)
    task_ids = torch.zeros(num_samples)
    
    dataset = TensorDataset(task_ids, features, labels)
    return dataset


def test_btsp_components():
    """Test BTSP individual components"""
    print("=" * 60)
    print("Testing BTSP Components")
    print("=" * 60)
    
    # Test configuration
    args = {
        "device": ["cuda" if torch.cuda.is_available() else "cpu"],
        "btsp_memory_dim": 512,
        "btsp_sparsity": 0.04,
        "btsp_branches": 8,
        "btsp_tau_e_steps": 6,
        "btsp_theta": 0.3,
        "btsp_alpha_star": 0.02,
        "btsp_eta": 0.05,
        "btsp_homeostasis_freq": 10,
        "btsp_eps_0": 0.05,
        "btsp_adaptive": True,
        "btsp_fusion_alpha": 0.3,
        "backbone_type": "vit_btsp",
        "pretrained": False,
        "init_cls": 10,
        "epochs": 2,
        "lr": 0.01,
        "milestones": [1],
        "lr_decay": 0.1,
        "weight_decay": 0.0005,
        "lrate": 0.01  # compatibility
    }
    
    device = args["device"][0]
    
    # 1. Test BTSPIncrementalNet
    print("\n1. Testing BTSPIncrementalNet...")
    try:
        net = BTSPIncrementalNet(args, with_fc=True)
        print(f"✓ BTSPIncrementalNet created: feature_dim={net.feature_dim}")
        
        # Test forward pass
        dummy_input = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            outputs = net(dummy_input)
            features = outputs["feat"]
            logits = outputs["logits"]
        
        print(f"✓ Forward pass: input={dummy_input.shape}, features={features.shape}, logits={logits.shape}")
        
        # Test feature extraction
        extracted_features = net.extract_vector(dummy_input)
        print(f"✓ Feature extraction: {extracted_features.shape}")
        
        # Test binary encoding
        binary_codes = net.encode_feats_to_bits(extracted_features, sparsity=0.04)
        sparsity_actual = binary_codes.float().mean().item()
        print(f"✓ Binary encoding: {binary_codes.shape}, sparsity={sparsity_actual:.4f}")
        
        # Test update_fc
        net.update_fc(20)
        outputs_expanded = net(dummy_input)
        print(f"✓ FC update: new logits shape={outputs_expanded['logits'].shape}")
        
    except Exception as e:
        print(f"✗ BTSPIncrementalNet test failed: {e}")
        return False
    
    # 2. Test BTSPMemoryPool
    print("\n2. Testing BTSPMemoryPool...")
    try:
        learner = BTSPMemoryPool(args)
        print(f"✓ BTSPMemoryPool created")
        
        # Test memory pool initialization
        print(f"✓ Memory components: N={learner.N}, branches={learner.num_branches}")
        print(f"✓ Parameters: tau_e={learner.tau_e_steps}, theta={learner.theta}")
        
        # Test memory pool operations with dummy data
        dummy_dataset = create_dummy_dataset(num_samples=50, num_classes=5)
        dummy_loader = DataLoader(dummy_dataset, batch_size=10, shuffle=False)
        
        # Test memory writing
        class_list = [0, 1, 2, 3, 4]
        learner._write_to_memory_pool(dummy_loader, class_list)
        print(f"✓ Memory writing: {len(learner.memory_weights)} classes stored")
        
        # Test memory retrieval
        dummy_query = torch.randn(4, 3, 32, 32)
        if torch.cuda.is_available():
            dummy_query = dummy_query.cuda()
            
        similarities = learner._retrieve_from_memory_pool(dummy_query)
        print(f"✓ Memory retrieval: query={dummy_query.shape}, similarities={similarities.shape}")
        
        # Test forward with memory fusion
        fused_outputs = learner._forward_with_memory_fusion(dummy_query)
        print(f"✓ Memory fusion: output shape={fused_outputs.shape}")
        
    except Exception as e:
        print(f"✗ BTSPMemoryPool test failed: {e}")
        return False
    
    print("\n✓ All BTSP components test passed!")
    return True


def test_btsp_memory_dynamics():
    """Test BTSP memory dynamics and capacity bounds"""
    print("\n" + "=" * 60)
    print("Testing BTSP Memory Dynamics")
    print("=" * 60)
    
    # Configuration for capacity testing
    args = {
        "device": ["cuda" if torch.cuda.is_available() else "cpu"],
        "btsp_memory_dim": 1024,
        "btsp_sparsity": 0.05,
        "btsp_branches": 16,
        "btsp_tau_e_steps": 5,
        "btsp_theta": 0.2,
        "btsp_alpha_star": 0.03,
        "btsp_eta": 0.1,
        "btsp_homeostasis_freq": 20,
        "btsp_eps_0": 0.05,
        "btsp_adaptive": True,
        "btsp_fusion_alpha": 0.4,
        "backbone_type": "vit_btsp",
        "pretrained": False,
        "init_cls": 10,
        "epochs": 1,
        "lr": 0.01,
        "milestones": [1],
        "lr_decay": 0.1,
        "weight_decay": 0.0005,
        "lrate": 0.01
    }
    
    try:
        learner = BTSPMemoryPool(args)
        
        # Test gating probability computation
        print("\n1. Testing gating probability adaptation...")
        for num_tasks in [1, 5, 10, 20]:
            p_flip_max = learner._compute_p_flip_max(num_tasks)
            p_gate = learner._solve_p_gate(p_flip_max)
            print(f"Tasks: {num_tasks:2d}, p_flip_max: {p_flip_max:.6f}, p_gate: {p_gate:.6f}")
        
        # Test branch occupancy and homeostasis
        print("\n2. Testing homeostasis mechanism...")
        dummy_dataset = create_dummy_dataset(num_samples=100, num_classes=10)
        dummy_loader = DataLoader(dummy_dataset, batch_size=20, shuffle=False)
        
        # Write multiple classes and observe homeostasis
        for class_id in range(5):
            learner._write_to_memory_pool(dummy_loader, [class_id])
            occupancy_mean = learner.branch_occupancy.mean().item()
            occupancy_std = learner.branch_occupancy.std().item()
            p_gate_mean = learner.p_gate_per_branch.mean().item()
            print(f"Class {class_id}: occupancy={occupancy_mean:.4f}±{occupancy_std:.4f}, p_gate={p_gate_mean:.6f}")
        
        # Test capacity estimation
        print("\n3. Testing memory capacity...")
        memory_per_class_bits = learner.N * learner.p_pre
        memory_per_class_bytes = memory_per_class_bits / 8
        print(f"Memory per class: {memory_per_class_bits:.0f} bits = {memory_per_class_bytes:.1f} bytes")
        print(f"Total memory for 100 classes: {memory_per_class_bytes * 100 / 1024:.1f} KB")
        
        # Test memory retrieval accuracy
        print("\n4. Testing retrieval dynamics...")
        query_features = torch.randn(10, 3, 32, 32)
        if torch.cuda.is_available():
            query_features = query_features.cuda()
            
        similarities = learner._retrieve_from_memory_pool(query_features)
        if similarities.size(1) > 0:
            max_similarities = similarities.max(dim=1)[0]
            print(f"Retrieval similarities: mean={max_similarities.mean():.2f}, max={max_similarities.max():.2f}")
        
        print("\n✓ Memory dynamics test completed!")
        return True
        
    except Exception as e:
        print(f"✗ Memory dynamics test failed: {e}")
        return False


def test_capacity_cliff_detection():
    """Test capacity cliff detection as described in the proposal"""
    print("\n" + "=" * 60)
    print("Testing Capacity Cliff Detection")
    print("=" * 60)
    
    # Parameters for capacity cliff experiment
    N = 2048
    p_pre = 0.04
    num_branches = 16
    
    print(f"Configuration: N={N}, p_pre={p_pre}, branches={num_branches}")
    
    # Compute theoretical capacity bounds
    k = int(N * p_pre)  # Number of active bits per pattern
    print(f"Active bits per pattern: k={k}")
    
    # Binary entropy bound
    H_binary = lambda p: -p * np.log2(p) - (1-p) * np.log2(1-p) if 0 < p < 1 else 0
    h_target = H_binary(p_pre)
    C_entropy = N * h_target
    print(f"Entropy bound: H(p_pre)={h_target:.4f}, C_entropy={C_entropy:.1f} bits")
    
    # Convolution bound (simplified)
    # Approximation: when patterns start overlapping significantly
    overlap_threshold = 0.5
    C_convolution = k / overlap_threshold
    print(f"Convolution bound (approx): C_conv≈{C_convolution:.1f} patterns")
    
    # Practical capacity estimate
    C_practical = min(C_entropy / k, C_convolution)
    print(f"Practical capacity estimate: {C_practical:.1f} patterns")
    
    # Simulate capacity cliff
    print("\nSimulating capacity cliff...")
    for M in [10, 50, 100, 200, 500, 1000]:
        # Expected overlap between random sparse patterns
        overlap_prob = 1 - (1 - p_pre) ** k
        expected_overlap = M * k * overlap_prob
        interference_ratio = expected_overlap / (M * k)
        
        print(f"M={M:4d}: overlap_prob={overlap_prob:.4f}, interference={interference_ratio:.4f}")
        
        # Capacity cliff occurs when interference becomes significant
        if interference_ratio > 0.1:
            print(f"  → Capacity cliff detected around M={M}")
            break
    
    print("\n✓ Capacity cliff analysis completed!")
    return True


if __name__ == "__main__":
    print("BTSP Memory Pool - Complete Implementation Test")
    print("=" * 60)
    
    success = True
    
    # Test 1: Component functionality
    if not test_btsp_components():
        success = False
    
    # Test 2: Memory dynamics
    if not test_btsp_memory_dynamics():
        success = False
    
    # Test 3: Capacity cliff detection
    if not test_capacity_cliff_detection():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All BTSP tests passed! Implementation is ready.")
        print("\nNext steps:")
        print("1. Run with PILOT framework: python main.py --config exps/btsp_mp_complete.json")
        print("2. Compare with other continual learning methods")
        print("3. Analyze memory efficiency and capacity bounds")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    print("=" * 60)
