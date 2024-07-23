# Install the newest triton version with
# pip install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"
import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from flash_attn.utils.benchmark import benchmark_all, benchmark_forward, benchmark_backward
from flash_attn.utils.benchmark import benchmark_fwd_bwd, benchmark_combined

from flash_attn import flash_attn_qkvpacked_func

try:
    from triton.ops.flash_attention import attention as attention_triton
except ImportError:
    attention_triton = None

try:
    import xformers.ops as xops
except ImportError:
    xops = None

try:
    import pandas as pd
except ImportError:
    pd = None

def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0


def attention_pytorch(qkv, dropout_p=0.0, causal=True):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
    """
    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    q = rearrange(q, 'b t h d -> (b h) t d')
    k = rearrange(k, 'b s h d -> (b h) d s')
    softmax_scale = 1.0 / math.sqrt(d)
    # Preallocate attn_weights for `baddbmm`
    scores = torch.empty(batch_size * nheads, seqlen, seqlen, dtype=qkv.dtype, device=qkv.device)
    scores = rearrange(torch.baddbmm(scores, q, k, beta=0, alpha=softmax_scale),
                       '(b h) t s -> b h t s', h=nheads)
    if causal:
        # "triu_tril_cuda_template" not implemented for 'BFloat16'
        # So we have to construct the mask in float
        causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
        # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
        scores = scores + causal_mask.to(dtype=scores.dtype)
    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    return output.to(dtype=qkv.dtype)


def time_fwd_bwd(func, *args, **kwargs):
    time_f, time_b = benchmark_fwd_bwd(func, *args, **kwargs)
    return time_f[1].mean, time_b[1].mean

def time_fwd(func, *args, **kwargs):
    time_f = benchmark_forward(func, *args, **kwargs)
    return time_f[1].mean

def time_bwd(func, *args, **kwargs):
    time_b = benchmark_backward(func, *args, **kwargs)
    return time_b[1].mean

repeats = 30
device = 'cuda'
dtype = torch.float16

bs_seqlen_vals = [(32, 512), (16, 1024), (8, 2048), (4, 4096), (2, 8192), (1, 16384)]
causal_vals = [False, True]
headdim_vals = [64, 128]
dim = 2048
dropout_p = 0.0

time_f = {}
time_b = {}
time_f_b = {}
speed_f = {}
speed_b = {}
speed_f_b = {}

def run_benchmark(methods=None, mode="fwd_bwd"):
    if methods is None:
        methods = (["Flash2", "Pytorch"]
                   + (["Triton"] if attention_triton is not None else [])
                   + (["xformers.c"] if xops is not None else [])
                   + (["xformers.f"] if xops is not None else []))
    
    results = []

    for causal in causal_vals:
        for headdim in headdim_vals:
            for batch_size, seqlen in bs_seqlen_vals:
                config = (causal, headdim, batch_size, seqlen)
                nheads = dim // headdim
                qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, device=device, dtype=dtype,
                                  requires_grad=True)
                
                row = {'Causal': causal, 'Dim of Head': headdim, "Num of Heads": nheads, 'Batch Size': batch_size, 'Seq Len': seqlen}
                
                for method in methods:
                    if method == "Flash2":
                        if mode in ["fwd_bwd", "fwd"]:
                            time_f[config, method] = time_fwd(
                                flash_attn_qkvpacked_func, qkv, dropout_p, causal=causal, repeats=repeats, verbose=False
                            )
                        if mode in ["fwd_bwd", "bwd"]:
                            time_b[config, method] = time_bwd(
                                flash_attn_qkvpacked_func, qkv, dropout_p, causal=causal, repeats=repeats, verbose=False
                            )
                    elif method == "Pytorch":
                        try:
                            qkv = qkv.detach().requires_grad_(True)
                            if mode in ["fwd_bwd", "fwd"]:
                                time_f[config, method] = time_fwd(
                                    attention_pytorch, qkv, dropout_p, causal=causal, repeats=repeats, verbose=False
                                )
                            if mode in ["fwd_bwd", "bwd"]:
                                time_b[config, method] = time_bwd(
                                    attention_pytorch, qkv, dropout_p, causal=causal, repeats=repeats, verbose=False
                                )
                        except:  # Skip if OOM
                            time_f[config, method] = float('nan')
                            time_b[config, method] = float('nan')
                    elif method == "Triton" and attention_triton is not None:
                        q, k, v = [torch.randn(batch_size, nheads, seqlen, headdim, device=device, dtype=dtype,
                                            requires_grad=True) for _ in range(3)]
                        try:
                            if mode in ["fwd_bwd", "fwd"]:
                                time_f[config, method] = time_fwd(
                                    attention_triton, q, k, v, causal, headdim**(-0.5),
                                    False, repeats=repeats, verbose=False
                                )
                            if mode in ["fwd_bwd", "bwd"]:
                                time_b[config, method] = time_bwd(
                                    attention_triton, q, k, v, causal, headdim**(-0.5),
                                    False, repeats=repeats, verbose=False
                                )
                                if mode == "fwd_bwd":
                                    try:
                                        b0 = time_bwd(
                                            attention_triton, q, k, v, causal, headdim**(-0.5),
                                            True, repeats=repeats, verbose=False
                                        )
                                        time_b[config, method] = min(time_b[config, method], b0)
                                    except:
                                        pass
                        except:
                            time_f[config, method] = float('nan')
                            time_b[config, method] = float('inf')
                    elif method in ["xformers.c", "xformers.f"] and xops is not None:
                        q, k, v = [torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype,
                                            requires_grad=True) for _ in range(3)]
                        op = (xops.fmha.cutlass.FwOp, xops.fmha.cutlass.BwOp) if method == "xformers.c" else (xops.fmha.flash.FwOp, xops.fmha.flash.BwOp)
                        if mode in ["fwd_bwd", "fwd"]:
                            time_f[config, method] = time_fwd(
                                xops.memory_efficient_attention, q, k, v,
                                attn_bias=xops.LowerTriangularMask() if causal else None,
                                op=op
                            )
                        if mode in ["fwd_bwd", "bwd"]:
                            time_b[config, method] = time_bwd(
                                xops.memory_efficient_attention, q, k, v,
                                attn_bias=xops.LowerTriangularMask() if causal else None,
                                op=op
                            )
                    
                    # Calculate speeds and add to the row
                    if mode in ["fwd_bwd", "fwd"]:
                        speed_f[config, method] = efficiency(
                            flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"),
                            time_f[config, method]
                        )
                        row[f'{method} fwd (TFLOPs/s)'] = speed_f[config, method]
                    
                    if mode in ["fwd_bwd", "bwd"]:
                        speed_b[config, method] = efficiency(
                            flops(batch_size, seqlen, headdim, nheads, causal, mode="bwd"),
                            time_b[config, method]
                        )
                        row[f'{method} bwd (TFLOPs/s)'] = speed_b[config, method]
                    
                    if mode == "fwd_bwd":
                        time_f_b[config, method] = time_f[config, method] + time_b[config, method]
                        speed_f_b[config, method] = efficiency(
                            flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd_bwd"),
                            time_f_b[config, method]
                        )
                        row[f'{method} fwd+bwd (TFLOPs/s)'] = speed_f_b[config, method]
                
                results.append(row)
                
                print(f"### causal={causal}, headdim={headdim}, nheads: {nheads}, batch_size={batch_size}, seqlen={seqlen} ###")
                for method in methods:
                    if mode == "fwd_bwd":
                        print(
                            f"{method} fwd: {speed_f[config, method]:.2f} TFLOPs/s, "
                            f"bwd: {speed_b[config, method]:.2f} TFLOPs/s, "
                            f"fwd + bwd: {speed_f_b[config, method]:.2f} TFLOPs/s"
                        )
                    elif mode == "fwd":
                        print(f"{method} fwd: {speed_f[config, method]:.2f} TFLOPs/s")
                    elif mode == "bwd":
                        print(f"{method} bwd: {speed_b[config, method]:.2f} TFLOPs/s")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Reorder columns
    column_order = ['Causal', 'Dim of Head', "Num of Heads", 'Batch Size', 'Seq Len']
    for method in methods:
        if mode in ["fwd_bwd", "fwd"]:
            column_order.append(f'{method} fwd (TFLOPs/s)')
        if mode in ["fwd_bwd", "bwd"]:
            column_order.append(f'{method} bwd (TFLOPs/s)')
        if mode == "fwd_bwd":
            column_order.append(f'{method} fwd+bwd (TFLOPs/s)')
    df = df[column_order]

    # Print DataFrame
    print("\nDataFrame Output:")
    print(df.to_string(index=False))

    # Save to CSV
    csv_filename = f'benchmark_results_{mode}.csv'
    df.to_csv(csv_filename, index=False)
    print(f"\nResults saved to {csv_filename}")

    return df

# Run the benchmark
df_result = run_benchmark(methods=["Flash2", "Pytorch"], mode="fwd")  # or "bwd" or "fwd_bwd"


# with open('flash2_attn_time.plk', 'wb') as fp:
#     pickle.dump((speed_f, speed_b, speed_f_b), fp, protocol=pickle.HIGHEST_PROTOCOL)
