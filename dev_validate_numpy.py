"""
dev_validate_numpy.py -- INTERNAL DEV TOOL, not a deliverable.

Re-implements the mini-GPT forward pass (embeddings -> N transformer blocks
with causal multi-head self-attention -> layer norm -> MLP -> output head)
using only numpy, with no autograd and no torch dependency, purely to
verify tensor shapes and the causal-masking logic before the real
`mini_gpt.py` (PyTorch) implementation is handed off. This mirrors the
dependency-aware validation approach used in Assignment 1.

Run with: python3 dev_validate_numpy.py
"""
import numpy as np

rng = np.random.default_rng(0)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def layer_norm(x, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def causal_self_attention(x, n_head, Wqkv, Wproj, key_padding_mask=None):
    B, T, C = x.shape
    head_dim = C // n_head
    qkv = x @ Wqkv  # (B, T, 3C)
    q, k, v = np.split(qkv, 3, axis=-1)

    def split_heads(t):
        return t.reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)  # (B, nh, T, hd)

    q, k, v = split_heads(q), split_heads(k), split_heads(v)
    att = q @ k.transpose(0, 1, 3, 2) / np.sqrt(head_dim)  # (B, nh, T, T)

    causal = np.tril(np.ones((T, T), dtype=bool))
    att = np.where(causal[None, None, :, :], att, -np.inf)

    if key_padding_mask is not None:  # (B, T), 1 = real token, 0 = pad
        pad = key_padding_mask[:, None, None, :].astype(bool)  # (B,1,1,T)
        att = np.where(pad, att, -np.inf)

    att = softmax(att, axis=-1)
    att = np.nan_to_num(att)  # rows that were fully masked (shouldn't happen) -> 0
    y = att @ v  # (B, nh, T, hd)
    y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
    return y @ Wproj, att


def mini_gpt_forward(idx, tok_emb, pos_emb, blocks, ln_f_scale, head_W, n_head, attention_mask=None):
    """
    idx: (B, T) int array of token ids
    tok_emb: (vocab, C) embedding table
    pos_emb: (block_size, C) positional embedding table
    blocks: list of dicts with keys Wqkv, Wproj, Wfc, Wmlp_proj
    head_W: (C, vocab) output projection (tied or separate)
    """
    B, T = idx.shape
    x = tok_emb[idx] + pos_emb[None, :T, :]
    attn_maps = []
    for blk in blocks:
        a_out, att = causal_self_attention(layer_norm(x), n_head, blk["Wqkv"], blk["Wproj"], attention_mask)
        x = x + a_out
        h = gelu(layer_norm(x) @ blk["Wfc"])
        x = x + h @ blk["Wmlp_proj"]
        attn_maps.append(att)
    x = layer_norm(x) * ln_f_scale
    logits = x @ head_W
    return logits, attn_maps


def main():
    vocab, C, T, n_head, n_layer, B = 200, 32, 16, 4, 2, 3

    tok_emb = rng.normal(scale=0.02, size=(vocab, C))
    pos_emb = rng.normal(scale=0.02, size=(T, C))
    blocks = []
    for _ in range(n_layer):
        blocks.append({
            "Wqkv": rng.normal(scale=0.02, size=(C, 3 * C)),
            "Wproj": rng.normal(scale=0.02, size=(C, C)),
            "Wfc": rng.normal(scale=0.02, size=(C, 4 * C)),
            "Wmlp_proj": rng.normal(scale=0.02, size=(4 * C, C)),
        })
    ln_f_scale = np.ones(C)
    head_W = rng.normal(scale=0.02, size=(C, vocab))

    idx = rng.integers(0, vocab, size=(B, T))
    # Simulate padding: last 3 tokens of sample 0 are padding
    attention_mask = np.ones((B, T), dtype=int)
    attention_mask[0, -3:] = 0

    logits, attn_maps = mini_gpt_forward(idx, tok_emb, pos_emb, blocks, ln_f_scale, head_W, n_head, attention_mask)

    assert logits.shape == (B, T, vocab), f"bad logits shape: {logits.shape}"
    for att in attn_maps:
        assert att.shape == (B, n_head, T, T), f"bad attention shape: {att.shape}"
        # causal check: position i must not attend to position j > i
        upper = np.triu_indices(T, k=1)
        assert np.allclose(att[:, :, upper[0], upper[1]], 0.0), "causal mask leaked future tokens!"
        # each real query row must sum to ~1 over keys it's allowed to see
        row_sums = att.sum(axis=-1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), "attention rows do not sum to 1"
    # padding check: no query should attend to the masked-out key positions for sample 0
    assert np.allclose(attn_maps[0][0, :, :, -3:], 0.0), "padding mask leaked into attention!"

    print("OK: logits shape", logits.shape)
    print("OK: causal masking verified (no attention to future tokens)")
    print("OK: padding masking verified (no attention to padded key positions)")
    print("OK: attention rows normalize to 1 over allowed positions")
    print(f"Architecture check passed for n_layer={n_layer}, n_head={n_head}, n_embd={C}, block_size={T}")


if __name__ == "__main__":
    main()
