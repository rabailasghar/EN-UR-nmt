import torch
import torch.nn as nn

class HydraAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(HydraAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        tgt_len, bsz, _ = query.size()
        src_len = key.size(0)

        q, k, v = self.qkv(query).chunk(3, dim=-1)
        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim)
        k = k.view(src_len, bsz * self.num_heads, self.head_dim)
        v = v.view(src_len, bsz * self.num_heads, self.head_dim)

        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)

        # Apply mask to the key
        if mask is not None:
            mask = mask.unsqueeze(0)  # Expand mask to match the source length
            mask = mask.unsqueeze(0).repeat(tgt_len, self.num_heads, 1, 1)
            k = k.masked_fill(mask, 0)

        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-1, -2))
        attn_scores = attn_scores / self.head_dim ** 0.5

        # Apply softmax activation
        attn_probs = nn.functional.softmax(attn_scores, dim=-1)

        # Dropout
        attn_probs = self.dropout(attn_probs)

        # Weighted sum of values
        weighted_sum = torch.matmul(attn_probs, v)

        # Reshape and project back to the original dimension
        weighted_sum = weighted_sum.view(tgt_len, bsz, self.num_heads, self.head_dim)
        weighted_sum = weighted_sum.transpose(0, 1).contiguous()
        weighted_sum = weighted_sum.view(bsz, tgt_len, self.embed_dim)

        # Project to the output dimension
        out = self.out_proj(weighted_sum)

        return out
