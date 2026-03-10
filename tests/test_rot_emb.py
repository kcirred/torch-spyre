# test_rope_spyre.py
import torch
from fms.modules.positions import RotaryEmbedding

# Import torch-spyre to trigger monkey-patching


def test_rotary_embedding_spyre_vs_cpu():
    """
    Test that Spyre RoPE produces same results as CPU implementation.
    """
    # Configuration
    batch_size = 2
    seq_len = 128
    num_heads = 8
    head_dim = 256

    # Create RotaryEmbedding
    rope = RotaryEmbedding(
        dim=head_dim,
        max_seq_len=512,
        ratio=10000.0,
    )

    # Create test inputs on CPU
    q_cpu = torch.randn(batch_size, seq_len, num_heads, head_dim).to(
        dtype=torch.float16
    )
    k_cpu = torch.randn(batch_size, seq_len, num_heads, head_dim).to(
        dtype=torch.float16
    )
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    # Run on CPU
    q_out_cpu, k_out_cpu = rope.adjusted_qk(q_cpu, k_cpu, position_ids)

    # Move to Spyre
    q_spyre = q_cpu.to("spyre")
    k_spyre = k_cpu.to("spyre")
    # position_ids_spyre = position_ids.to("spyre")

    # Run on Spyre
    q_out_spyre, k_out_spyre = rope.adjusted_qk(q_spyre, k_spyre, position_ids)

    # Compare results
    q_out_spyre_cpu = q_out_spyre.cpu()
    k_out_spyre_cpu = k_out_spyre.cpu()

    print(f"{q_out_cpu=}, {q_out_spyre_cpu=}")

    torch.testing.assert_close(
        q_out_cpu,
        q_out_spyre_cpu,
        rtol=1e-3,
        atol=1e-3,
        msg="Query outputs don't match between CPU and Spyre",
    )

    torch.testing.assert_close(
        k_out_cpu,
        k_out_spyre_cpu,
        rtol=1e-3,
        atol=1e-3,
        msg="Key outputs don't match between CPU and Spyre",
    )

    print("RoPE outputs match between CPU and Spyre")


if __name__ == "__main__":
    test_rotary_embedding_spyre_vs_cpu()
