@echo off
REM BTSP Memory Pool - Complete Implementation Demo
REM 根据1.0.txt方案实现的完整BTSP系统演示

echo ============================================================
echo BTSP Memory Pool - Complete Implementation Demo
echo Based on Behavioral Timescale Synaptic Plasticity
echo ============================================================

echo.
echo Step 1: Testing BTSP components...
python test_btsp_complete.py
if %errorlevel% neq 0 (
    echo Component test failed! Please check the implementation.
    pause
    exit /b 1
)

echo.
echo Step 2: Running BTSP on CIFAR-100 incremental learning...
echo Configuration: 10 initial classes + 9 tasks of 10 classes each
echo Memory: 4096-dim binary vectors with 4%% sparsity
echo.

python main.py --config exps/btsp_mp_complete.json

echo.
echo Step 3: Analysis and comparison (optional)...
echo You can now compare BTSP with other methods like:
echo - python main.py --config exps/finetune.json
echo - python main.py --config exps/icarl.json
echo - python main.py --config exps/l2p.json

echo.
echo ============================================================
echo BTSP Demo completed!
echo.
echo Key features implemented:
echo - Sparse random flip codes (4%% sparsity)
echo - Eligibility traces with exponential decay
echo - Branch-level gating with adaptive probabilities
echo - Homeostasis mechanism for capacity regulation
echo - Information-theoretic capacity bounds
echo - No exemplar storage (only binary memory pools)
echo.
echo Memory efficiency:
echo - Per class: ~205 bytes (4096 bits at 4%% density)
echo - 100 classes: ~20 KB total
echo - Zero gradient interference between tasks
echo ============================================================

pause
