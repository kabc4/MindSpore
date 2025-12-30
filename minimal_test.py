# quick_test.py
import mindspore as ms
from mindspore import Tensor, context
import numpy as np

# Test MindSpore installation
print("Testing MindSpore installation...")
try:
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    x = Tensor(np.array([1.0, 2.0, 3.0]))
    y = x + 1
    print(f"✓ MindSpore working! Tensor: {y}")
except Exception as e:
    print(f"✗ MindSpore error: {e}")

# Test imports
print("\nTesting imports...")
try:
    from flash_attention import FlashAttention2, StandardAttention
    print("✓ FlashAttention modules imported successfully")
except Exception as e:
    print(f"✗ Import error: {e}")

# Simple functionality test
print("\nTesting simple attention...")
try:
    batch_size, seq_len, heads, dim = 1, 16, 2, 8
    Q = Tensor(np.ones((batch_size, seq_len, heads, dim), dtype=np.float32))
    K = Tensor(np.ones((batch_size, seq_len, heads, dim), dtype=np.float32))
    V = Tensor(np.ones((batch_size, seq_len, heads, dim), dtype=np.float32))
    
    std_attn = StandardAttention()
    output = std_attn(Q, K, V)
    print(f"✓ Standard attention works! Output shape: {output.shape}")
    
    fa2 = FlashAttention2(block_size=8)
    output = fa2(Q, K, V)
    print(f"✓ FlashAttention-2 works! Output shape: {output.shape}")
    
except Exception as e:
    print(f"✗ Function test error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Quick test complete!")
print("Run 'python run_all.py' for full implementation pipeline")