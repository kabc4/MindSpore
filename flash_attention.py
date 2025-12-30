
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer, XavierUniform
import numpy as np
import time

class FlashAttention2(nn.Cell):
    """Complete FlashAttention-2 implementation for MindSpore"""
    
    def __init__(self, dropout=0.0, causal=False, block_size=256):
        super().__init__()
        self.dropout = dropout
        self.causal = causal
        self.block_size = block_size
        self.dropout_op = nn.Dropout(p=dropout) if dropout > 0 else None
        
    def construct(self, Q, K, V):
        """
        FlashAttention-2 forward pass
        Args:
            Q: [batch_size, seq_len_q, num_heads, head_dim]
            K: [batch_size, seq_len_k, num_heads, head_dim]
            V: [batch_size, seq_len_k, num_heads, head_dim]
        Returns:
            O: [batch_size, seq_len_q, num_heads, head_dim]
        """
        batch_size, seq_len_q, num_heads, head_dim = Q.shape
        seq_len_k = K.shape[1]
        
        # Scale query
        scale = head_dim ** -0.5
        Q_scaled = Q * scale
        
        # Initialize output
        O = ops.zeros_like(Q)
        
        # Process query blocks
        for q_start in range(0, seq_len_q, self.block_size):
            q_end = min(q_start + self.block_size, seq_len_q)
            q_len = q_end - q_start
            
            # Get query block
            Q_block = Q_scaled[:, q_start:q_end]  # [batch, q_len, heads, dim]
            
            # Reshape for batch matmul: [batch*heads, q_len, dim]
            Q_block_reshaped = Q_block.transpose(0, 2, 1, 3).reshape(
                batch_size * num_heads, q_len, head_dim)
            
            # Initialize block statistics
            O_block = ops.zeros((batch_size * num_heads, q_len, head_dim), Q.dtype)
            L_block = ops.zeros((batch_size * num_heads, q_len, 1), ms.float32)
            M_block = ops.fill(ms.float32, (batch_size * num_heads, q_len, 1), -float('inf'))
            
            # Process key-value blocks
            for kv_start in range(0, seq_len_k, self.block_size):
                kv_end = min(kv_start + self.block_size, seq_len_k)
                kv_len = kv_end - kv_start
                
                K_block = K[:, kv_start:kv_end]  # [batch, kv_len, heads, dim]
                V_block = V[:, kv_start:kv_end]
                
                # Reshape K, V: [batch*heads, kv_len, dim]
                K_block_reshaped = K_block.transpose(0, 2, 1, 3).reshape(
                    batch_size * num_heads, kv_len, head_dim)
                V_block_reshaped = V_block.transpose(0, 2, 1, 3).reshape(
                    batch_size * num_heads, kv_len, head_dim)
                
                # Compute attention scores: [batch*heads, q_len, kv_len]
                # Q @ K^T where Q: [..., q_len, dim], K^T: [..., dim, kv_len]
                S_block = ops.matmul(Q_block_reshaped, K_block_reshaped.transpose(0, 2, 1))
                
                # Apply causal mask if needed
                if self.causal:
                    causal_mask = self._create_causal_mask(q_len, kv_len, q_start, kv_start)
                    S_block = S_block + causal_mask
                
                # Online softmax update
                M_new = ops.maximum(M_block, S_block.max(axis=-1, keepdims=True))
                
                exp_S = ops.exp(S_block - M_new)
                
                # Apply dropout if training
                if self.dropout > 0 and self.training and self.dropout_op is not None:
                    exp_S = self.dropout_op(exp_S)
                
                P_block = exp_S
                L_new = ops.exp(M_block - M_new) * L_block + P_block.sum(axis=-1, keepdims=True)
                
                # Update output block
                O_block = ops.exp(M_block - M_new) * O_block + ops.matmul(P_block, V_block_reshaped)
                
                # Update statistics
                M_block = M_new
                L_block = L_new
            
            # Normalize and reshape back
            O_block_normalized = O_block / L_block
            
            # Reshape back to [batch, heads, q_len, dim] -> [batch, q_len, heads, dim]
            O_block_final = O_block_normalized.reshape(
                batch_size, num_heads, q_len, head_dim).transpose(0, 2, 1, 3)
            
            O[:, q_start:q_end] = O_block_final
        
        return O
    
    def _create_causal_mask(self, q_len, kv_len, q_start, kv_start):
        """Create causal mask for attention"""
        # Create mask where position i can only attend to positions <= i
        row = ops.arange(q_len).reshape(-1, 1)
        col = ops.arange(kv_len).reshape(1, -1)
        
        # For causal attention
        mask = (row + q_start) < (col + kv_start)
        mask = mask.reshape(1, q_len, kv_len).astype(ms.float32)
        mask = mask * -1e9  # Large negative value for masked positions
        
        return mask


class StandardAttention(nn.Cell):
    """Standard attention implementation for comparison"""
    
    def __init__(self, dropout=0.0, causal=False):
        super().__init__()
        self.dropout = dropout
        self.causal = causal
        self.dropout_op = nn.Dropout(p=dropout) if dropout > 0 else None
        
    def construct(self, Q, K, V):
        batch_size, seq_len_q, num_heads, head_dim = Q.shape
        seq_len_k = K.shape[1]
        
        # Scale query
        scale = head_dim ** -0.5
        Q_scaled = Q * scale
        
        # Reshape for batch matmul: [batch*heads, seq_len, dim]
        Q_reshaped = Q_scaled.transpose(0, 2, 1, 3).reshape(
            batch_size * num_heads, seq_len_q, head_dim)
        K_reshaped = K.transpose(0, 2, 1, 3).reshape(
            batch_size * num_heads, seq_len_k, head_dim)
        V_reshaped = V.transpose(0, 2, 1, 3).reshape(
            batch_size * num_heads, seq_len_k, head_dim)
        
        # Compute attention scores: [batch*heads, seq_len_q, seq_len_k]
        attn_scores = ops.matmul(Q_reshaped, K_reshaped.transpose(0, 2, 1))
        
        # Apply causal mask if needed
        if self.causal:
            mask = ops.triu(
                ops.ones((seq_len_q, seq_len_k), ms.float32), 
                k=1
            ) * -1e9
            mask = mask.reshape(1, seq_len_q, seq_len_k)
            attn_scores = attn_scores + mask
        
        # Apply softmax
        attn_weights = ops.softmax(attn_scores, axis=-1)
        
        # Apply dropout
        if self.dropout > 0 and self.training and self.dropout_op is not None:
            attn_weights = self.dropout_op(attn_weights)
        
        # Compute output
        output = ops.matmul(attn_weights, V_reshaped)
        
        # Reshape back to original: [batch, seq_len_q, heads, dim]
        output = output.reshape(
            batch_size, num_heads, seq_len_q, head_dim
        ).transpose(0, 2, 1, 3)
        
        return output