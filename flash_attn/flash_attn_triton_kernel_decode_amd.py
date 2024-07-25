from typing import Optional
from flash_attn.flash_attn_triton_kernel_prefill_amd import MetaData
import pytest
import torch
import sys

import triton
import triton.language as tl

DEBUG = True

def _strides(x: torch.Tensor, *stride_names: str):
    if x is None:
        return {f"stride_{s}": 0 for i, s in enumerate(stride_names)}

    assert x.ndim == len(stride_names)
    return {f"stride_{s}": x.stride(i) for i, s in enumerate(stride_names)}


@triton.jit
def _fwd_kernel_splitK(
    Q,
    K,
    V,
    sm_scale,
    Out_splitK,  # [B, H, split_k, Mq, K]
    Metadata,  # [B, H, 2, split_k, M_ceil] contains [mi, li]
    K_new,
    V_new,
    Seq_len,
    stride_qz,
    stride_qm,
    stride_qg,
    stride_qh,
    stride_qd,
    stride_kz,
    stride_kn,
    stride_kg,
    stride_kh,
    stride_kd,
    stride_vz,
    stride_vn,
    stride_vg,
    stride_vh,
    stride_vd,
    stride_osk_zhg,
    stride_osk_s,
    stride_osk_m,
    stride_osk_d,
    stride_mzhg,
    stride_m2,
    stride_ms,
    stride_mm,
    stride_kn_z,
    stride_kn_n,
    stride_kn_g,
    stride_kn_h,
    stride_kn_d,
    stride_vn_z,
    stride_vn_n,
    stride_vn_g,
    stride_vn_h,
    stride_vn_d,
    Z,
    N_CTX_Q,
    N_CTX_K,
    N_CTX_NEW,
    BLOCK_N_PER_SPLIT,
    H: tl.constexpr,
    G: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BOUNDS_CHECKS_N: tl.constexpr,
    USE_SEQ_LEN: tl.constexpr,
    NEW_KV: tl.constexpr,
    PACKED_PER_VAL: tl.constexpr = 1,
    N_GROUPS: tl.constexpr = 1,
):
    """This kernel can accept non-quantized or int4-quantized keys/values.
    PACKED_PER_VAL determines the quantization type:
        - PACKED_PER_VAL == 1 means no quantization
        - PACKED_PER_VAL == 8 means 4-bit quantization (8 packed quantized values inside one int32)
    For the quantized case K/V should be int32 tensors.
    Quantization can be row-wise (when N_GROUPS = 1) or group-wise with N_GROUPS = 2, 4, or 8.
    Quantization coefficients are stored at the beginning of the row along the last dimension of K/V
    So K[B, H, M, :] has a form
    [   quant_coef0, quant_coef1, ...|
        group0_quant_value0, group0_quant_value1,... |
        group1_quant_value0, group1_quant_value1,...]
    where each quant_coef is an int32 which should be interpreted as 2 packed float16: scale and offset.

    """
    tl.static_assert(
        (PACKED_PER_VAL == 1 and tl.constexpr(K.dtype.element_ty != tl.int32))
        or (PACKED_PER_VAL == 8 and tl.constexpr(K.dtype.element_ty == tl.int32)),
        f"Only 4-bit quantization is supported, K/V should have dtype int32 in "
        f"the quantized case: {PACKED_PER_VAL=} {tl.constexpr(K.dtype)=} {tl.constexpr(K.dtype.element_ty)=}",
    )
    tl.static_assert(
        (((N_GROUPS == 1 or N_GROUPS == 2) or N_GROUPS == 4) or N_GROUPS == 8),
        "Number of quantization groups can be 1 (row-wise quantization), 2, 4, or 8.",
    )

    QUANTIZED: tl.constexpr = PACKED_PER_VAL > 1
    PACKED_D_PER_GROUP: tl.constexpr = BLOCK_DMODEL // PACKED_PER_VAL // N_GROUPS
    D_PER_GROUP: tl.constexpr = BLOCK_DMODEL // N_GROUPS

    start_m = tl.program_id(0)
    off_zhg = tl.program_id(1)
    off_z = off_zhg // (H * G)
    off_h = (off_zhg // G) % H
    off_g = off_zhg % G
    splitk_idx = tl.program_id(2)

    lo = splitk_idx * BLOCK_N_PER_SPLIT
    if USE_SEQ_LEN:
        cache_seqlen_last_idx = tl.load(Seq_len + off_z)
        kv_len = cache_seqlen_last_idx + 1
    else:
        kv_len = N_CTX_K
        cache_seqlen_last_idx = kv_len - 1
    hi = tl.minimum((splitk_idx + 1) * BLOCK_N_PER_SPLIT, kv_len)
    # print("kv_len:", kv_len)
    # print("lo:", lo)
    # print("hi:", hi)

    # calculate base offset
    k_base = K + off_h * stride_kh + off_z * stride_kz + off_g * stride_kg
    v_base = V + off_h * stride_vh + off_z * stride_vz + off_g * stride_vg

    # Copy new Keys and Values into Cache
    if NEW_KV:
        kn_base = K_new + off_h * stride_kn_h + off_z * stride_kn_z + off_g * stride_kn_g
        hi_n = N_CTX_NEW
        lo_n = 0

        # Copy new Keys
        Kn_block_ptr = tl.make_block_ptr(
            base=kn_base + stride_kn_d * QUANTIZED * N_GROUPS,
            shape=(PACKED_D_PER_GROUP, hi_n),
            strides=(stride_kn_d, stride_kn_n),
            offsets=(0, lo_n),
            block_shape=(PACKED_D_PER_GROUP, BLOCK_N),
            order=(0, 1),
            )
        
        # Create a block pointer for the destination in K
        K_dest_ptr = tl.make_block_ptr(
            base=k_base + stride_kd * QUANTIZED * N_GROUPS,
            shape=(PACKED_D_PER_GROUP, N_CTX_K),
            strides=(stride_kd, stride_kn),
            offsets=(0, cache_seqlen_last_idx),
            block_shape=(PACKED_D_PER_GROUP, BLOCK_N),
            order=(0, 1),
        )

        # Copy new key values to K
        for i in range(0, N_CTX_NEW, BLOCK_N):
            # Load from K_new
            k_new_block = tl.load(Kn_block_ptr, boundary_check=(1,))
            print("k_new_block:", k_new_block)
            
            # Store to K
            tl.store(K_dest_ptr, k_new_block, boundary_check=(1,))
            
            # Advance pointers
            Kn_block_ptr = tl.advance(Kn_block_ptr, (0, BLOCK_N))
            K_dest_ptr = tl.advance(K_dest_ptr, (0, BLOCK_N))



        # Copy new Values
        vn_base = V_new + off_h * stride_vn_h + off_z * stride_vn_z + off_g * stride_vn_g
        Vn_block_ptr = tl.make_block_ptr(
            base=vn_base + stride_vn_d * QUANTIZED * N_GROUPS,
            shape=(hi_n, PACKED_D_PER_GROUP),
            strides=(stride_vn_n, stride_vn_d),
            offsets=(lo_n, 0),
            block_shape=(BLOCK_N, PACKED_D_PER_GROUP),
            order=(1, 0),
        )

        # write this
        # Create a block pointer for the destination in V
        V_dest_ptr = tl.make_block_ptr(
            base=v_base + stride_vd * QUANTIZED * N_GROUPS,
            shape=(N_CTX_K, PACKED_D_PER_GROUP),
            strides=(stride_vn, stride_vd),
            offsets=(cache_seqlen_last_idx, 0),
            block_shape=(BLOCK_N, PACKED_D_PER_GROUP),
            order=(1, 0),
        )

        # Copy new value values to V
        for i in range(0, N_CTX_NEW, BLOCK_N):
            # Load from V_new
            v_new_block = tl.load(Vn_block_ptr, boundary_check=(0,))
            
            # Store to V
            tl.store(V_dest_ptr, v_new_block, boundary_check=(0,))
            
            # Advance pointers
            Vn_block_ptr = tl.advance(Vn_block_ptr, (BLOCK_N, 0))
            V_dest_ptr = tl.advance(V_dest_ptr, (BLOCK_N, 0))


    else:
        Kn_block_ptr = None
        Vn_block_ptr = None        

    Q_block_ptr = tl.make_block_ptr(
        base=Q + off_h * stride_qh + off_z * stride_qz + off_g * stride_qg,
        shape=(N_CTX_Q, D_PER_GROUP),
        strides=(stride_qm, stride_qd),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D_PER_GROUP),
        order=(1, 0),
    )

    # Additional shift by 1 along the last dimension in the quantized case, since
    # the first element along that dim contains packed quantization coefficients.
    K_block_ptr = tl.make_block_ptr(
        base=k_base + stride_kd * QUANTIZED * N_GROUPS,
        shape=(PACKED_D_PER_GROUP, hi),
        strides=(stride_kd, stride_kn),
        offsets=(0, lo),
        block_shape=(PACKED_D_PER_GROUP, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=v_base + stride_vd * QUANTIZED * N_GROUPS,
        shape=(hi, PACKED_D_PER_GROUP),
        strides=(stride_vn, stride_vd),
        offsets=(lo, 0),
        block_shape=(BLOCK_N, PACKED_D_PER_GROUP),
        order=(1, 0),
    )



    if QUANTIZED:
        # Pointers to quantization coefficients
        K_scale_shift_block_ptr = tl.make_block_ptr(
            base=k_base,
            shape=(1, hi),
            strides=(stride_kd, stride_kn),
            offsets=(0, lo),
            block_shape=(1, BLOCK_N),
            order=(0, 1),
        )
        V_scale_shift_block_ptr = tl.make_block_ptr(
            base=v_base,
            shape=(hi, 1),
            strides=(stride_vn, stride_vd),
            offsets=(lo, 0),
            block_shape=(BLOCK_N, 1),
            order=(1, 0),
        )
    else:
        K_scale_shift_block_ptr = None
        V_scale_shift_block_ptr = None

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    acc = tl.zeros([BLOCK_M, D_PER_GROUP], dtype=tl.float32)  # noqa: F821

    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(  # noqa: F821
        tl.advance(Q_block_ptr, (0, 0)), boundary_check=(0, ))
    q = (q * qk_scale).to(q.dtype)

    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        print("start_n:", start_n)
        print("BLOCK_N:", BLOCK_N)

        k, v = load_dequantize_k_v_group(
            K_block_ptr,
            V_block_ptr,
            K_scale_shift_block_ptr,
            V_scale_shift_block_ptr,
            BOUNDS_CHECKS_N,
            PACKED_PER_VAL,
            PACKED_D_PER_GROUP,
            Q.dtype.element_ty,
            0,
        )

        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)  # noqa: F821

        # TODO: This is slow, and only needed at the last iteration.
        # Maybe we can unroll the last iteration instead?
        if BOUNDS_CHECKS_N:
            qk = tl.where(tl.arange(0, BLOCK_N) < hi - start_n, qk, float("-inf"))
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])

        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        p = p.to(Q.dtype.element_ty)

        # -- scale and update acc --
        acc *= alpha[:, None]
        acc += tl.dot(p, v)
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        if PACKED_PER_VAL > 1:
            K_scale_shift_block_ptr = tl.advance(K_scale_shift_block_ptr, (0, BLOCK_N))
            V_scale_shift_block_ptr = tl.advance(V_scale_shift_block_ptr, (BLOCK_N, 0))

    # write back O
    O_block_ptr = tl.make_block_ptr(
        base=Out_splitK + off_zhg * stride_osk_zhg + splitk_idx * stride_osk_s,
        shape=(N_CTX_Q, D_PER_GROUP),
        strides=(stride_osk_m, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, D_PER_GROUP),
        order=(1, 0),
    )
    tl.store(
        tl.advance(O_block_ptr, (0, 0)),
        acc,
        boundary_check=(0, ),
    )
    # Write metadata for split-K reduction
    Metadata_ptr = (Metadata + off_zhg * stride_mzhg + splitk_idx * stride_ms + start_m * BLOCK_M +
                    tl.arange(0, BLOCK_M))
    tl.store(Metadata_ptr, m_i)
    tl.store(Metadata_ptr + stride_m2, l_i)


@triton.jit
def load_dequantize_k_v_group(
    K_block_ptr,
    V_block_ptr,
    K_scale_shift_block_ptr,
    V_scale_shift_block_ptr,
    BOUNDS_CHECKS_N: tl.constexpr,
    PACKED_PER_VAL: tl.constexpr,
    PACKED_D_PER_GROUP: tl.constexpr,
    dtype: tl.constexpr,
    group_id: tl.constexpr,
):
    #Load K/V for a given block. In case of int4-quantized K/V,
    # dequantize them after loading. If quantization is group-wise,
    # use group_id to advance the pointers to the current group.

    # Advance to the current quantization group
    K_block_ptr = tl.advance(K_block_ptr, (PACKED_D_PER_GROUP * group_id, 0))
    V_block_ptr = tl.advance(V_block_ptr, (0, PACKED_D_PER_GROUP * group_id))

    # -- load k, v --
    k = tl.load(K_block_ptr, boundary_check=(1, ) if BOUNDS_CHECKS_N else ())
    v = tl.load(V_block_ptr, boundary_check=(0, ) if BOUNDS_CHECKS_N else ())

    print("k:", k)
    print("v:", v)

    if PACKED_PER_VAL > 1:
        # K/V are quantized, load quantization coefficients and dequantize
        K_scale_shift_block_ptr = tl.advance(K_scale_shift_block_ptr, (group_id, 0))
        V_scale_shift_block_ptr = tl.advance(V_scale_shift_block_ptr, (0, group_id))

        k_scale_shift = tl.load(K_scale_shift_block_ptr, boundary_check=(1, ) if BOUNDS_CHECKS_N else ())
        v_scale_shift = tl.load(V_scale_shift_block_ptr, boundary_check=(0, ) if BOUNDS_CHECKS_N else ())

        k_scale, k_shift = cast_uint32_to_half2(k_scale_shift)
        v_scale, v_shift = cast_uint32_to_half2(v_scale_shift)
        v = dequantize(v, v_scale, v_shift, PACKED_PER_VAL).to(dtype)
        k_t = dequantize(
            tl.trans(k),
            tl.trans(k_scale),
            tl.trans(k_shift),
            PACKED_PER_VAL,
        ).to(dtype)
        k = tl.trans(k_t)
    return k, v


@triton.jit
def cast_uint32_to_half2(scale_shift):
    # Extract two float16 packed into one int32
    scale = scale_shift & 0xFFFF
    shift = scale_shift >> 16
    scale = scale.to(tl.uint16).to(tl.float16, bitcast=True)
    shift = shift.to(tl.uint16).to(tl.float16, bitcast=True)
    return scale, shift


@triton.jit
def dequantize(
    x_,
    scale,
    shift,
    PACKED_PER_VAL: tl.constexpr = 8,
):
    # PACKED_PER_VAL is the number of values packed into
    # each element x_. For example, for int4 quantization
    #and x_ of type int32, PACKED_PER_VAL is 8.

    BLOCK_N: tl.constexpr = x_.shape[0]
    BLOCK_DMODEL_PACKED: tl.constexpr = x_.shape[1]
    offsets = tl.arange(0, PACKED_PER_VAL) * 4
    quant_offset = (x_[:, None, :] >> offsets[None, :, None])  # (BLOCK_N, PACKED_PER_VAL, D // PACKED_PER_VAL)

    quant_offset = tl.view(quant_offset, (BLOCK_N, BLOCK_DMODEL_PACKED * PACKED_PER_VAL))
    # Trick - instead of converting int4 to float16 we view it as float16
    # and then multiply by 32768 * 512 == 2**24
    quant_offset = (quant_offset & 0xF).to(tl.uint16).to(tl.float16, bitcast=True)
    quant_offset = (quant_offset * 32768.0).to(tl.float16)
    scale_512 = scale * 512

    dequant = quant_offset * scale_512 + shift
    return dequant


@triton.jit
def _splitK_reduce(
    Out_splitK,  # [B, H, split_k, Mq, K]
    Metadata,  # [B, H, 2, split_k, M_ceil] contains [mi, li]
    Out,  # [B, H, M, K]
    LSE,  # [B, H, M]
    stride_osk_zhg,
    stride_osk_s,
    stride_osk_m,
    stride_osk_k,
    stride_mzhg,
    stride_m2,
    stride_ms,
    stride_mm,
    stride_oz,
    stride_oh,
    stride_og,
    stride_om,
    stride_ok,
    stride_lse_zhg,
    stride_lse_m,
    M_ceil: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    split_k: tl.constexpr,
    splitK_pow2: tl.constexpr,
    use_mask: tl.constexpr,
):
    off_zhg = tl.program_id(0)
    off_z = off_zhg // (H * G)
    off_h = (off_zhg // G) % H
    off_g = off_zhg % G
    off_m = tl.program_id(1)
    off_k = tl.program_id(2)

    # read  chunk
    spk_idx = tl.arange(0, splitK_pow2)
    kidx = tl.arange(0, BLOCK_SIZE)

    Metadata_ptr = (Metadata + stride_mzhg * off_zhg + spk_idx * stride_ms + off_m * stride_mm)

    o_ptr = (Out_splitK + off_zhg * stride_osk_zhg + stride_osk_m * off_m + off_k * BLOCK_SIZE +
             stride_osk_s * spk_idx[:, None] + kidx[None, :] * stride_osk_k)

    # read max values of each splitK
    if use_mask:
        spk_mask = spk_idx < split_k
        l_m = tl.load(Metadata_ptr, mask=spk_mask, other=float("-inf"))
        l_sum = tl.load(Metadata_ptr + stride_m2, mask=spk_mask, other=0.0)
        acc = tl.load(o_ptr, mask=spk_mask[:, None], other=0.0)
    else:
        l_m = tl.load(Metadata_ptr)
        l_sum = tl.load(Metadata_ptr + stride_m2)
        acc = tl.load(o_ptr)

    g_m = tl.max(l_m, axis=0)
    alpha = tl.math.exp2(l_m - g_m)

    # read sum
    l_sum *= alpha
    g_sum = tl.sum(l_sum, axis=0)
    acc = acc * alpha[:, None]
    acc_out = tl.sum(acc, axis=0) / g_sum
    Out_ptr = (Out + stride_oz * off_z + stride_oh * off_h + stride_og * off_g + stride_om * off_m +
               off_k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE))
    tl.store(Out_ptr, acc_out)
    l_ptrs = LSE + off_zhg * stride_lse_zhg + off_m
    tl.store(l_ptrs, (g_m + tl.math.log2(g_sum)) / 1.44269504)


def quantize_kv_int4(k: torch.Tensor, num_groups: int = 1) -> torch.Tensor:
    # Scale and shift are such that quantization linearly maps
    # int4 values range [0..15] to input values range min(k)..max(k)
    # individually for every row
    k = k.reshape(*k.shape[:-1], num_groups, k.shape[-1] // num_groups)
    max_vals = torch.max(k, dim=-1, keepdim=True).values
    min_vals = torch.min(k, dim=-1, keepdim=True).values
    scale_k: torch.Tensor = (max_vals - min_vals) / 15

    shift_k = torch.min(k, dim=-1, keepdim=True).values
    scale_k = scale_k.to(torch.float16)
    shift_k = shift_k.to(torch.float16)

    in_bytes = ((k - shift_k.expand(k.shape)) / scale_k.expand(k.shape)) + 0.5
    in_bytes = in_bytes.to(torch.uint8)
    in_int4 = in_bytes & 0xF
    in_int4_packed = in_int4[..., ::2] + (in_int4[..., 1::2] << 4)
    scale_shift = torch.concat([scale_k.view(torch.uint8), shift_k.view(torch.uint8)], dim=-1)
    k_quant = torch.concat(
        [
            scale_shift.flatten(start_dim=-2),
            in_int4_packed.flatten(start_dim=-2),
        ],
        dim=-1,
    ).view(torch.int16)
    return k_quant


def dequantize_kv_fp16(quant_k: torch.Tensor, num_groups: int = 1) -> torch.Tensor:
    k_i16 = quant_k.view(torch.int16)
    k_ui8 = k_i16.view(torch.uint8)

    ss_size = num_groups * 4
    scale_shift_ui8 = k_ui8[..., 0:ss_size]
    scale_shift_ui8 = scale_shift_ui8.reshape(*scale_shift_ui8.shape[:-1], num_groups, 4)
    scale = scale_shift_ui8[..., 0:2].view(torch.float16)
    shift = scale_shift_ui8[..., 2:4].view(torch.float16)

    kv_ui8 = k_ui8[..., ss_size:]
    k_ui8 = kv_ui8.reshape(*kv_ui8.shape[:-1], num_groups, -1)
    k1_i4 = k_ui8 & 0xF
    k2_i4 = (k_ui8 & 0xF0) >> 4
    k_shape = k1_i4.shape
    k1_f16 = k1_i4.to(torch.float16) * scale.expand(k_shape) + shift.expand(k_shape)
    k2_f16 = k2_i4.to(torch.float16) * scale.expand(k_shape) + shift.expand(k_shape)

    out = torch.empty((*k1_f16.shape[:-1], k1_f16.shape[-1] * 2), dtype=torch.float16, device=quant_k.device)
    out[..., ::2] = k1_f16
    out[..., 1::2] = k2_f16
    out = out.reshape(*k_shape[:-2], -1)

    return out


def get_split_k(B: int, G: int, H: int, Mk: int) -> int:
    """Heuristic for the number of splits"""
    bh = max(B * H, 1)  # NOTE: Handle B*h=0 case
    split_k = max(Mk, 1024) // bh
    max_chunk_size = 64
    while split_k > 0 and Mk / split_k < max_chunk_size:
        split_k = split_k // 2
    while B * H * G * split_k >= 1024:
        split_k = split_k // 2
    split_k = min(split_k, 512)
    split_k = max(split_k, 1)
    return split_k

class _attention(torch.autograd.Function):

    OPERATOR = _fwd_kernel_splitK
    SUPPORTED_DEVICES = {"cuda"}
    CUDA_MINIMUM_COMPUTE_CAPABILITY = (8, 0)
    SUPPORTED_DTYPES = {
        torch.half,
        torch.bfloat16,
    }
    SUPPORTED_MAX_K = 128
    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_BMGHK = True
    NAME = "triton_splitKF"

    @staticmethod
    def forward(cls, q, k, v, input_metadata):
        if DEBUG:
            print()
            print("attention_decode.forward")
            print("q:", q, q.shape)
            print("k:", k, k.shape)
            print("v:", v, v.shape)
            print("input_metadata:", input_metadata)

        # kernels expects "bsghd"
        if input_metadata.layout == "bhsd":
            q=q.permute(0, 2, 1, 3).unsqueeze(2)
            k=k.permute(0, 2, 1, 3).unsqueeze(2)
            v=v.permute(0, 2, 1, 3).unsqueeze(2)
            if input_metadata.new_kv:
                input_metadata.k_new = input_metadata.k_new.permute(0, 2, 1, 3).unsqueeze(2)
                input_metadata.v_new = input_metadata.v_new.permute(0, 2, 1, 3).unsqueeze(2) 
                
        elif input_metadata.layout == "bshd":
            q=q.unsqueeze(2)
            k=k.unsqueeze(2)
            v=v.unsqueeze(2)

            if input_metadata.new_kv:
                input_metadata.k_new = input_metadata.k_new.unsqueeze(2)
                input_metadata.v_new = input_metadata.v_new.unsqueeze(2) 
        elif input_metadata.layout == "bsghd":
            pass
        elif input_metadata.layout is None:
            raise ValueError("Layout not given")

        # context
        cls.SPLIT_K: Optional[int] = None
        cls.BLOCK_M = 16
        cls.BLOCK_N = 64

        cls.NUM_GROUPS = 1  # Default quantization is row-wise

        # attn_bias = inp.attn_bias
        if input_metadata.cache_seqlens is not None:
            seq_len = input_metadata.cache_seqlens
        else:
            seq_len = None

        # Transpose in the case of MQA/GQA
        mqa_swap_seqlen_head = False
        if k.shape[3] > 1 and k.stride(3) == 0 and v.stride(3) == 0:
            mqa_swap_seqlen_head = True
            assert q.shape[1] == 1
            q = q.transpose(1, 3)
            k = k[:, :, :, :1]
            v = v[:, :, :, :1]
        print("mqa_swap_seqlen_head:", mqa_swap_seqlen_head)

        batch_size, seqlen_k, group_k, head_k, dim_k = k.shape
        PACKED_PER_VAL = 1
        if k.dtype == torch.int32:
            # Quantized K/V
            PACKED_PER_VAL = 8
            dim_k = (k.shape[-1] - cls.NUM_GROUPS) * 8
        batch_size, seqlen_q, group_q, head_q, dim_q = q.shape
        assert dim_k == dim_q, f"Keys have head dim {dim_k} but queries have head dim {dim_q}"
        print(f"batch_size = {batch_size}, seqlen_q = {seqlen_q}, seqlen_k = {seqlen_k}, head_q = {head_q}, head_k = {head_k}, dim_q = {dim_q}, dim_kv = {dim_k}")

        BLOCK_M = cls.BLOCK_M
        BLOCK_N = cls.BLOCK_N
        if cls.SPLIT_K is not None:
            split_k = cls.SPLIT_K
        else:
            # Use heuristics
            split_k = get_split_k(batch_size, group_k, head_k, seqlen_k) # NOTE: should the split think about seqlens?
        if DEBUG:
            print("split_k:", split_k)

        seqlen_q_ceil = (seqlen_q + BLOCK_M - 1) // BLOCK_M * BLOCK_M
        o_splitk = torch.empty([batch_size * group_q * head_q, split_k, seqlen_q_ceil, dim_q], dtype=torch.float32, device=q.device)
        metadata = torch.empty([batch_size * group_q * head_q, 2, split_k, seqlen_q_ceil], dtype=torch.float32, device=q.device)
        lse = torch.empty((batch_size * group_q * head_q, seqlen_q), device=q.device, dtype=torch.float32)
        grid = (triton.cdiv(seqlen_q, BLOCK_M), batch_size * group_k * head_k, split_k)

        num_warps = 1
        split_size = (seqlen_k + split_k - 1) // split_k
        use_seq_len = seq_len is not None

        print(f"batch_size = {batch_size}, group_q = {group_q}, head_q = {head_q}, split_k = {split_k}, seqlen_q_ceil = {seqlen_q_ceil}, dim_q = {dim_q}, num_of_wgs = {group_q * group_q * head_q * split_k}")
        
        if DEBUG:
            print("q:", q, q.shape)
            print("k:", k, k.shape)
            print("v:", v, v.shape)
            print("sm_scale:", input_metadata.sm_scale)
            print("o_splitk:", o_splitk, o_splitk.shape)
            print("metadata:", metadata, metadata.shape)
            print("seq_len:", seq_len)
            print("lse:", lse)
            print("grid:", grid)
            print("split_size:", split_size)
            print("use_seq_len:", use_seq_len)


        _fwd_kernel_splitK[grid](
            Q=q,
            K=k,
            V=v,
            sm_scale=input_metadata.sm_scale,
            Out_splitK=o_splitk,
            Metadata=metadata,
            K_new = input_metadata.k_new,
            V_new = input_metadata.v_new,
            Seq_len=seq_len,
            **_strides(q, "qz", "qm", "qg", "qh", "qd"),
            **_strides(k, "kz", "kn", "kg", "kh", "kd"),
            **_strides(v, "vz", "vn", "vg", "vh", "vd"),
            **_strides(o_splitk, "osk_zhg", "osk_s", "osk_m", "osk_d"),
            **_strides(metadata, "mzhg", "m2", "ms", "mm"),
            **_strides(input_metadata.k_new, "kn_z", "kn_n", "kn_g", "kn_h", "kn_d"),
            **_strides(input_metadata.v_new, "vn_z", "vn_n", "vn_g", "vn_h", "vn_d"),
            Z=batch_size,
            H=head_q,
            G=group_q,
            N_CTX_Q=seqlen_q,
            N_CTX_K=seqlen_k,
            N_CTX_NEW=input_metadata.k_new.shape[1],
            BLOCK_N_PER_SPLIT=split_size,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_DMODEL=dim_k,
            BOUNDS_CHECKS_N=(split_size % BLOCK_N) > 0 or use_seq_len,
            USE_SEQ_LEN=use_seq_len,
            NEW_KV=input_metadata.new_kv,
            num_warps=num_warps,
            num_stages=1,
            PACKED_PER_VAL=PACKED_PER_VAL,
            N_GROUPS=cls.NUM_GROUPS if PACKED_PER_VAL > 1 else 1,
        )

        if mqa_swap_seqlen_head:
            out = torch.empty((batch_size, head_q, group_q, seqlen_q, dim_q), device=q.device, dtype=q.dtype).transpose(1, 3)
        else:
            out = torch.empty((batch_size, seqlen_q, group_q, head_q, dim_q), device=q.device, dtype=q.dtype)

        # Merge together
        splitK_pow2 = triton.next_power_of_2(split_k)
        use_mask = splitK_pow2 > split_k
        if batch_size * group_q * head_q * seqlen_q >= 512:
            k_block_num = 1
        else:
            k_block_num = 2
        assert out.shape[-1] % k_block_num == 0
        k_block_size = out.shape[-1] // k_block_num
        grid = (batch_size * group_q * head_q, seqlen_q, k_block_num)
        _splitK_reduce[grid](
            o_splitk, 
            metadata, 
            out, 
            lse, 
            **_strides(o_splitk, "osk_zhg", "osk_s", "osk_m", "osk_k"),
            **_strides(metadata, "mzhg", "m2", "ms", "mm"), 
            **_strides(out, "oz", "om", "og", "oh", "ok"),
            **_strides(lse, "lse_zhg", "lse_m"), 
            M_ceil=seqlen_q_ceil, 
            BLOCK_SIZE=k_block_size, 
            G=group_q, 
            H=head_q,
            # TODO: Tune num_warps
            split_k=split_k, 
            splitK_pow2=splitK_pow2, 
            use_mask=use_mask, 
            num_warps=4)

        lse = lse.reshape([batch_size, group_q, head_q, seqlen_q])
        if mqa_swap_seqlen_head:
            # H/M dimensions have been swapped
            out = out.transpose(1, 3)
            lse = lse.transpose(2, 3)
        if q.ndim == 4:
            # BMGHK -> BMHK
            assert group_q == 1
            out = out[:, :, 0]
            lse = lse[:, 0]
        if seqlen_k == 0:
            out.zero_()
        if mqa_swap_seqlen_head:
            out = out.reshape(batch_size, -1, seqlen_q * group_q, dim_q).transpose(1, 2).contiguous()
        else:
            out = out.reshape(batch_size, head_q * group_q, -1, dim_q).contiguous()


        # out is "bhsd"
        if input_metadata.layout == "bshd":
            out=out.permute(0, 2, 1, 3)

        return out


attention_decode = _attention.apply


def get_input_shapes():
    cases = [(max(1, 2**(16 - i)), 1, 2**i, 16, 1, 128)
             for i in range(8, 18)] + [(max(1, 2**(16 - i)), 1, 2**i, 16, 2, 128) for i in range(8, 18)]

    return cases


@pytest.mark.parametrize('batch_size, seqlen_q, seqlen_k, group_q, group_k, dim', get_input_shapes())
def test_op_fwd(batch_size, seqlen_q, seqlen_k, group_q, group_k, dim, dtype=torch.float16):
    print()
    print(f"batch_size = {batch_size}, seqlen_q = {seqlen_q}, seqlen_k = {seqlen_k}, group_q = {group_q}, group_k = {group_k}, dim = {dim}")
    torch.manual_seed(20)
    query_group_head_size = (group_q + group_k - 1) // group_k
    q = (torch.empty((batch_size, seqlen_q, group_k, query_group_head_size, dim), dtype=dtype,
                     device="cuda").normal_(mean=0., std=0.5).requires_grad_())
    k = (torch.empty((batch_size, seqlen_k, group_k, 1, dim), dtype=dtype,
                     device="cuda").normal_(mean=0.,
                                            std=0.5).requires_grad_()).expand(-1, -1, -1, query_group_head_size, -1)
    v = (torch.empty((batch_size, seqlen_k, group_k, 1, dim), dtype=dtype,
                     device="cuda").normal_(mean=0.,
                                            std=0.5).requires_grad_()).expand(-1, -1, -1, query_group_head_size, -1)
    scale = 1 / dim**0.5

    print("q:", q.shape)
    print("k:", k.shape)
    print("v:", v.shape)
    input_metadata = MetaData(sm_scale=scale)
    input_metadata.layout = "bsghd"
    tri_out = attention_decode(q, k, v, input_metadata)
    print("tri_out:", tri_out.shape)
    print()

    q = q.reshape([batch_size, seqlen_q, -1, dim]).permute(0, 2, 1, 3)
    k = k.reshape([batch_size, seqlen_k, -1, dim]).permute(0, 2, 1, 3)
    v = v.reshape([batch_size, seqlen_k, -1, dim]).permute(0, 2, 1, 3)
    print("q_ref:", q.shape)
    print("k_ref:", k.shape)
    print("v_ref:", v.shape)
    attn = (q @ k.transpose(-1, -2) * scale).softmax(-1)
    print("attn:", attn.shape)
    ref_out = attn @ v
    print("ref_out:", ref_out.shape)

    # compare
    torch.testing.assert_close(ref_out, tri_out, atol=1e-3, rtol=0)


@pytest.mark.parametrize('B, Mq, Mkv, Hq, Hkv, K', get_input_shapes())
def test_op_fwd_int4_kv(B, Mq, Mkv, Hq, Hkv, K, dtype=torch.float16):
    torch.manual_seed(2)
    q = (torch.empty((B, Mq, Hkv, (Hq + Hkv - 1) // Hkv, K), dtype=dtype,
                     device="cuda").normal_(mean=1.0, std=0.5).requires_grad_())
    k = (torch.empty((B, Mkv, Hkv, 1, K), dtype=dtype,
                     device="cuda").normal_(mean=1.0,
                                            std=0.5).requires_grad_()).expand(-1, -1, -1, (Hq + Hkv - 1) // Hkv, -1)
    v = (torch.empty((B, Mkv, Hkv, 1, K), dtype=dtype,
                     device="cuda").normal_(mean=1.0,
                                            std=0.5).requires_grad_()).expand(-1, -1, -1, (Hq + Hkv - 1) // Hkv, -1)

    num_groups = 1
    quant_k = (quantize_kv_int4(k, num_groups=num_groups).contiguous().view(torch.int32))
    quant_v = (quantize_kv_int4(v, num_groups=num_groups).contiguous().view(torch.int32))
    scale = 1 / K**0.5
    tri_out = attention_decode(q, quant_k, quant_v, scale)

    q = q.reshape([B, Mq, -1, K]).permute(0, 2, 1, 3)
    k = k.reshape([B, Mkv, -1, K]).permute(0, 2, 1, 3)
    v = v.reshape([B, Mkv, -1, K]).permute(0, 2, 1, 3)
    attn = (q @ k.transpose(-1, -2) * scale).softmax(-1)
    ref_out = attn @ v
    # compare
    torch.testing.assert_close(ref_out, tri_out, atol=2.1e-2, rtol=0)

    # since quantization introduces rounding error, use the
    # dequantized kv as inputs to the ref implementation to reduce
    # the tolerance to 1e-3
    dqk = dequantize_kv_fp16(quant_k, num_groups=num_groups)
    dqv = dequantize_kv_fp16(quant_v, num_groups=num_groups)
    dqk = dqk.reshape([B, Mkv, -1, K]).permute(0, 2, 1, 3)
    dqv = dqv.reshape([B, Mkv, -1, K]).permute(0, 2, 1, 3)
    dq_attn = (q @ dqk.transpose(-1, -2) * scale).softmax(-1)
    dq_ref_out = dq_attn @ dqv
    torch.testing.assert_close(dq_ref_out, tri_out, atol=1e-3, rtol=0)


def test_quantization():
    a = torch.randn((2, 4, 32), dtype=torch.float16, device='cuda')
    qa = quantize_kv_int4(a, num_groups=4)
    dqa = dequantize_kv_fp16(qa, num_groups=4)
    torch.testing.assert_close(a, dqa, atol=1.5e-1, rtol=1e-1)


try:
    FLASH_VER = 2
except BaseException:
    try:
        FLASH_VER = 1
    except BaseException:
        FLASH_VER = None
HAS_FLASH = FLASH_VER is not None

configs = []
for mode in ['fwd']:
    # for D_HEAD in [128]:
    for causal in [False]:
        configs.append(
            triton.testing.Benchmark(
                x_names=['B', 'Mq', 'Mkv', 'Hq', 'Hkv', 'K'], x_vals=get_input_shapes(), line_arg='provider',
                line_vals=['triton'] + (['flash'] if HAS_FLASH else []),
                line_names=['Triton'] + ([f'Flash-{FLASH_VER}'] if HAS_FLASH else []), styles=[('red', '-'),
                                                                                               ('blue', '-')],
                ylabel='ms', plot_name=f'fused-attention-d{128}-{mode}-causal={causal}', args={
                    # 'D_HEAD': D_HEAD,
                    'dtype': torch.float16, 'mode': mode, 'causal': causal
                }))


@triton.testing.perf_report(configs)
def bench_flash_attention(B, Mq, Mkv, Hq, Hkv, K, causal, mode, provider, dtype=torch.float16, device="cuda"):
    assert mode in ['fwd', 'bwd']
    warmup = 100
    rep = 400
    ms = 0
    if provider == "triton":
        q = torch.randn([B, Mq, Hkv, Hq // Hkv, K], device="cuda", dtype=dtype, requires_grad=False)
        k = torch.randn([B, Mkv, Hkv, 1, K], device="cuda", dtype=dtype,
                        requires_grad=False).expand(-1, -1, -1, Hq // Hkv, -1)
        v = torch.randn([B, Mkv, Hkv, 1, K], device="cuda", dtype=dtype,
                        requires_grad=False).expand(-1, -1, -1, Hq // Hkv, -1)

        sm_scale = 1.3
        fn = lambda: attention_decode(q, k, v, sm_scale)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    # flops_per_matmul = 2 * B * Hq * (Mq * K * Mkv + Mq * Mkv * K)
    # total_flops = 2 * flops_per_matmul
    # totalBytes = ((B * Mkv * Hkv * K * 2) + (B * Mq * Hq * K) + (B * Mq * Hq * K)) * 2

    # return totalBytes / ms * 1e-9
    return ms * 1000


def main():
    bench_flash_attention.run(save_path='.', print_data=True)


if __name__ == '__main__':
    sys.exit(main())