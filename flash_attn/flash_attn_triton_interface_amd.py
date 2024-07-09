import torch
import triton
from .flash_attn_triton_kernel_amd import MetaData, attention, get_shape_from_layout, _attn_bwd_preprocess, _attn_bwd
from .flash_attn_triton_decode_amd import attention_inference

DEBUG=False
DEBUG_VARLEN=False
DEBUG_KVCACHE=True


def fwd(q,
        k,
        v,
        o,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        return_softmax,
        gen_):
    if DEBUG:
        print("flash_attn_triton_amd.py::fwd")
        print("q:", q.shape)
        print("k:", k.shape)
        print("v:", v.shape)
        print("alibi_slopes:", alibi_slopes)
        print("dropout_p:", dropout_p)
        print("softmax_scale:", softmax_scale)
        print("causal:", causal)
        print("window_size_left:", window_size_left)
        print("window_size_right:", window_size_right)
        print("return_softmax:", return_softmax)
        print("gen_:", gen_)

    if dropout_p != 0.0:
        raise ValueError("dropout is not supported on HIP")

    if o is None:
        o = torch.empty_like(q)

    # Setup metadata
    input_metadata = MetaData(sm_scale=softmax_scale)
    input_metadata.max_seqlens_q = q.shape[1]
    input_metadata.max_seqlens_k = k.shape[1]
    input_metadata.layout = "bshd"

    batch, nheads_q, nheads_k, head_size = get_shape_from_layout(q, k, input_metadata)
    
    if causal:
        input_metadata.need_causal()
    
    # if bias is not None:
    #     input_metadata.need_bias(bias, batch, nheads_q, input_metadata.max_seqlens_q, input_metadata.max_seqlens_k)
    
    if alibi_slopes is not None:
        input_metadata.need_alibi(alibi_slopes, batch, nheads_q)
    
    if dropout_p > 0.0:
        input_metadata.need_dropout(dropout_p, return_softmax)
    
    # Check arguments
    input_metadata.check_args(q, k, v, o)
    
    # Perform the forward attention computation
    tri_out, encoded_softmax = attention(q, k, v, o, input_metadata)

    softmax_lse = encoded_softmax
    softmax_p = encoded_softmax

    return tri_out, q , k , v, o, softmax_lse, softmax_p, torch.get_rng_state()

def varlen_fwd(
        q, 
        k, 
        v, 
        o,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        block_table_,
        alibi_slopes,\
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        zero_tensors,
        causal,
        window_size_left,
        window_size_right,
        return_softmax,
        gen_):
    
    if DEBUG_VARLEN:
        print("flash_attn_triton_amd.py::varlen_fwd")
        print("q:", q.shape)
        print("k:", k.shape)
        print("v:", v.shape)
        print("cu_seqlens_q:", cu_seqlens_q)
        print("cu_seqlens_k:", cu_seqlens_k)
        print("block_table_:", block_table_)
        print("alibi_slopes:", alibi_slopes)
        print("max_seqlen_q:", max_seqlen_q)
        print("max_seqlen_k:", max_seqlen_k)
        print("dropout_p:", dropout_p)
        print("softmax_scale:", softmax_scale)
        print("zero_tensors:", zero_tensors)
        print("causal:", causal)
        print("window_size_left:", window_size_left)
        print("window_size_right:", window_size_right)
        print("return_softmax:", return_softmax)
        print("gen_:", gen_)

    if dropout_p != 0.0:
        raise ValueError("dropout is not supported on HIP")
    
    if o is None:
        o = torch.empty_like(q)

    # Setup metadata
    input_metadata = MetaData(sm_scale=softmax_scale)
    input_metadata.set_varlen_params(cu_seqlens_q, cu_seqlens_k)

    # get shapes
    batch, nheads_q, nheads_k, head_size = get_shape_from_layout(q, k, input_metadata)

    if causal:
        input_metadata.need_causal()
    
    # if bias is not None:
    #     input_metadata.need_bias(bias, batch, nheads_q, q.shape[2], k.shape[2])
    
    if alibi_slopes is not None:
        input_metadata.need_alibi(alibi_slopes, batch, nheads_q)
    
    if dropout_p > 0.0:
        input_metadata.need_dropout(dropout_p, return_softmax)
    
    # Check arguments
    input_metadata.check_args(q, k, v, o)

    # Perform the forward attention computation
    tri_out, encoded_softmax = attention(q, k, v, o, input_metadata)

    softmax_lse = encoded_softmax
    softmax_p = encoded_softmax

    return tri_out, q , k , v, o, softmax_lse, softmax_p, torch.get_rng_state()


def unpage_inplace(paged_cache, block_table, seqlen_k):
    if DEBUG_KVCACHE:
        print("paged_cache before:", paged_cache, paged_cache.shape)
        print("block_table:", block_table, block_table.shape)
    
    # Extract dimensions
    num_blocks, block_size, nheads_k, d = paged_cache.shape
    batch_size, num_blocks_per_batch = block_table.shape
    seqlen_k_inferred = num_blocks_per_batch * block_size

    # Create a temporary buffer to hold the data during rearrangement
    temp_buffer = torch.empty_like(paged_cache[:batch_size * num_blocks_per_batch])

    # Rearrange the cache in-place
    for i in range(batch_size):
        for j, block_idx in enumerate(block_table[i]):
            if block_idx != -1:  # Ignore padding blocks
                start = i * num_blocks_per_batch + j
                temp_buffer[start] = paged_cache[block_idx]

    # Copy the rearranged data back to paged_cache
    paged_cache[:batch_size * num_blocks_per_batch] = temp_buffer

    # Reshape paged_cache to the unpaged shape
    paged_cache = paged_cache.view(batch_size, seqlen_k_inferred, nheads_k, d)

    # Truncate paged_cache to seqlen_k in-place
    paged_cache = paged_cache[:, :seqlen_k, :, :]

    if DEBUG_KVCACHE:
        print("paged_cache after:", paged_cache, paged_cache.shape)

    return paged_cache

def unpage(paged_cache, block_table, seqlen_k):
    if DEBUG_KVCACHE:
        print("paged_cache:", paged_cache, paged_cache.shape)
        print("block_table:", block_table, block_table.shape)
    
    # Extract dimensions
    num_blocks, block_size, nheads_k, d = paged_cache.shape
    batch_size, num_blocks_per_batch = block_table.shape
    seqlen_k_inferred = num_blocks_per_batch * block_size

    # Initialize the unpaged cache
    unpaged_cache = torch.zeros((batch_size, seqlen_k_inferred, nheads_k, d), 
                                dtype=paged_cache.dtype, 
                                device=paged_cache.device)

    # Reconstruct the unpaged cache
    for i in range(batch_size):
        for j, block_idx in enumerate(block_table[i]):
            if block_idx != -1:  # Ignore padding blocks
                start = j * block_size
                end = min((j + 1) * block_size, seqlen_k_inferred)
                unpaged_cache[i, start:end] = paged_cache[block_idx, :end-start]

    if DEBUG_KVCACHE:
        print("unpaged_cache:", unpaged_cache, unpaged_cache.shape)

    return unpaged_cache[:, :seqlen_k, :, :]

def update_cache_inplace(cache, new_seq, cache_seqlens):
    # if DEBUG_KVCACHE:
    #     print("k_cache before:", k_cache)
    #     print("k:", k)
    #     print("cache_seqlens:", cache_seqlens)

    # Ensure cache and new_seq are 4D tensors
    assert cache.dim() == 4 and new_seq.dim() == 4, "cache and new_seq should be 4D tensors (B, S, H, D)"
    
    # Ensure cache and new_seq have compatible dimensions
    assert cache.shape[0] == new_seq.shape[0], "Batch sizes don't match"
    assert cache.shape[2] == new_seq.shape[2], "Number of heads don't match"
    assert cache.shape[3] == new_seq.shape[3], "Head dimensions don't match"
    
    batch_size, seqlen_k, nheads, d = cache.shape
    seqlen_new = new_seq.shape[1]
    
    # Create a mask for updating
    arange = torch.arange(seqlen_k, device=cache.device).unsqueeze(0)
    cache_seqlens_expanded = cache_seqlens.unsqueeze(1)
    update_mask = torch.logical_and(
        cache_seqlens_expanded <= arange,
        arange < cache_seqlens_expanded + seqlen_new
    )
    
    # Update the cache in-place with new_seq where the mask is True
    cache[update_mask] = new_seq.view(-1, nheads, d)

    # if DEBUG_KVCACHE:
    #     print("k_cache after:", k_cache)
    return

def updated_paged_cache_inplace(paged_cache, new_seqs, cache_seqlens, block_table, debug_print=False):
    debug_print = DEBUG_KVCACHE and debug_print
    
    if debug_print:
        print("paged_cache before:", paged_cache, paged_cache.shape)
        print("new_seqs:", new_seqs, new_seqs.shape)
        print("cache_seqlens:", cache_seqlens, cache_seqlens.shape)
        print("block_table:", block_table, block_table.shape)

    # Extract dimensions
    num_blocks, block_size, nheads, d = paged_cache.shape
    batch_size, seqlen_new, nheads, d = new_seqs.shape
    batch_size, num_blocks_per_batch = block_table.shape

    # Iterate through the batch and update the cache
    for i in range(batch_size):
        seq_blocks = block_table[i]
        valid_blocks = seq_blocks[seq_blocks != -1]
        new_seq = new_seqs[i]
        start_idx = cache_seqlens[i]

        if debug_print:
            print("seq_blocks:", seq_blocks)
            print("valid_blocks:", valid_blocks)
            print("new_seq:", new_seq)
            print("start_idx:", start_idx)
        
        for j, block_idx in enumerate(valid_blocks):
            block_start = j * block_size
            block_end = min((j + 1) * block_size, start_idx + seqlen_new)
            if debug_print:
                print("block_start:", block_start)
                print("block_end:", block_end)
            
            # Calculate the range of indices to update in this block
            update_start = max(start_idx - block_start, 0)
            update_end = min(block_end - block_start, block_size)

            if debug_print:
                print("update_start:", update_start)
                print("update_end:", update_end)
            
            if update_end > update_start:
                # Update the cache for this block
                paged_cache[block_idx, update_start:update_end] = new_seq[block_start + update_start - start_idx:block_start + update_end - start_idx]

    if debug_print:
        print("paged_cache after:", paged_cache)
    
    return paged_cache


def fwd_kvcache(
        q,
        k_cache,
        v_cache,
        k,
        v,
        cache_seqlens,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        block_table,
        alibi_slopes,
        out,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        rotary_interleaved,
        num_splits):
    
    if DEBUG_KVCACHE:
        print()
        print("flash_attn_triton_amd.py::fwd_kvcache")
        print("q:", q.shape)
        print("k_cache:", k_cache, k_cache.shape)
        print("v_cache:", v_cache, v_cache.shape)
        print("k:", k, k.shape if k is not None else None)
        print("v:", v, v.shape if v is not None else None)
        print("cache_seqlens:", cache_seqlens, cache_seqlens.size())
        print("rotary_cos:", rotary_cos)
        print("rotary_sin:", rotary_sin)
        print("cache_batch_idx:", cache_batch_idx)
        print("block_table:", block_table, block_table.shape if block_table is not None else None)
        print("alibi_slopes:", alibi_slopes)
        print("out:", out)
        print("softmax_scale:", softmax_scale)
        print("causal:", causal)
        print("window_size_left:", window_size_left)
        print("window_size_right:", window_size_right)
        print("rotary_interleaved:", rotary_interleaved)
        print("num_splits:", num_splits)
    
    if out is None:
        out = torch.empty_like(q)

    if True:
        q_input = q
        input_metadata = MetaData(sm_scale=softmax_scale)


        # new kv
        if k is not None and v is not None:
            if block_table is not None:
                updated_paged_cache_inplace(k_cache, k, cache_seqlens, block_table)
                updated_paged_cache_inplace(v_cache, v, cache_seqlens, block_table)
            else:
                update_cache_inplace(k_cache, k, cache_seqlens)
                update_cache_inplace(v_cache, v, cache_seqlens)
            
            # fill metadata
            input_metadata.new_kv = True
            input_metadata.seqlen_new = k.shape[1]

        # paged attention
        if block_table is not None:
            unpage_seqlen_k = max(cache_seqlens) + 1 # infer the seqlen_k if paged cache
            k_cache = unpage(k_cache, block_table, unpage_seqlen_k)
            v_cache = unpage(v_cache, block_table, unpage_seqlen_k)
            
        # index into cache
        if cache_batch_idx is not None:
            k_input = k_cache[cache_batch_idx,:,:,:]
            v_input = v_cache[cache_batch_idx,:,:,:]
        else:
            k_input = k_cache
            v_input = v_cache

        # set metadata
        seqlen_q = q_input.shape[1]
        seqlen_k = k_input.shape[1]
        input_metadata.max_seqlens_q = seqlen_q
        input_metadata.max_seqlens_k = seqlen_k
        input_metadata.layout = "bshd"
        

        batch, nheads_q, nheads_k, head_size = get_shape_from_layout(q_input, k_input, input_metadata)
        
        if causal:
            input_metadata.need_causal()
        
        if alibi_slopes is not None:
            input_metadata.need_alibi(alibi_slopes, batch, nheads_q)

        # cache seqlens (seqlens in kvcache) (b x 1)
        input_metadata.cache_seqlens = cache_seqlens

        if out is None:
            out = torch.empty_like(q)
    
        # Check arguments
        input_metadata.check_args(q_input, k_input, v_input, out)

        # Perform the forward attention computation
        tri_out, encoded_softmax = attention(q_input, k_input, v_input, out, input_metadata)

        softmax_lse = encoded_softmax
        softmax_p = encoded_softmax
    else:
        q_input=q.unsqueeze(3)
        k_input=k_cache.unsqueeze(3)
        v_input=v_cache.unsqueeze(3)
        
        tri_out = attention_inference(q_input, k_input, v_input, softmax_scale)

    if DEBUG_KVCACHE:
        print()
        print("tri_out:", tri_out.shape)


    return tri_out, None


def bwd(dout, q, k, v, out, softmax_lse, dq, dk, dv, alibi_slopes, dropout_p, softmax_scale,  causal, window_size_left,
        window_size_right, deterministic, gen_, rng_state):
    if DEBUG:
        print("flash_attn_triton_amd.py::bwd")
        print("q:", q.shape)
        print("k:", k.shape)
        print("v:", v.shape)
        print("softmax_lse:", softmax_lse)
        print("dq:", dq.shape)
        print("dk:", dk.shape)
        print("dv:", dv.shape)
        print("alibi_slopes:", alibi_slopes)
        print("dropout_p:", dropout_p)
        print("softmax_scale:", softmax_scale)
        print("causal:", causal)
        print("window_size_left:", window_size_left)
        print("window_size_right:", window_size_right)
        print("deterministic:", deterministic)
        print("gen_:", gen_)
        print("rng_state:", rng_state)
 
    if out is None:
        out = torch.empty_like(q)

    # Ensure the tensors have requires_grad=True
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    out.requires_grad_()

    # Create metadata object
    metadata = MetaData(sm_scale=softmax_scale)
    metadata.max_seqlens_q = q.shape[1]
    metadata.max_seqlens_k = k.shape[1]
    metadata.layout = "bshd"

    if metadata == 'bshd':
        q = q.transpose(1, 2).clone()
        k = k.transpose(1, 2).clone()
        v = v.transpose(1, 2).clone()

    batch = q.shape[0]
    nheads_q = q.shape[1]
    BLOCK_DMODEL = q.shape[3]
    
    # Setup metadata
    if causal:
        metadata.need_causal()
    
    # if bias is not None:
    #     metadata.need_bias(bias, q.shape[0], q.shape[1], q.shape[2], k.shape[2])

    return_softmax = True
    if alibi_slopes is not None:
        metadata.need_alibi(alibi_slopes, batch, nheads_q)
    
    if dropout_p > 0.0:
        metadata.need_dropout(dropout_p, return_softmax)
    
    # Check arguments
    metadata.check_args(q, k, v, out)

    # write your own version backward
    M = torch.empty((batch, nheads_q, metadata.max_seqlens_q), device=q.device, dtype=torch.float32) # this passed from 

    if torch.version.hip is not None:
        BLOCK = 64
    else:
        BLOCK = 128
    o = out
    do = dout
    sm_scale = softmax_scale
    assert do.is_contiguous()
    assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
    seqlen_q = q.shape[2]
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    BATCH, N_CTX, N_HEAD = q.shape[:3]
    PRE_BLOCK = 128
    # NUM_WARPS, NUM_STAGES = 4, 1
    BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 64, 64, 32
    BLK_SLICE_FACTOR = 2
    RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
    arg_k = k
    arg_k = arg_k * (sm_scale * RCP_LN2)
    if DEBUG:
        print("N_CTX:", N_CTX)
    # assert N_CTX % PRE_BLOCK == 0

    delta = torch.empty_like(M)
    _, Lk, _ = q.shape[-1], k.shape[-1], v.shape[-1]
    # padded_head = (Lk != ctx.BLOCK_DMODEL)
    grid_preprocess = (triton.cdiv(do.shape[2], BLOCK), do.shape[1], do.shape[0])
    _attn_bwd_preprocess[grid_preprocess](
        o,
        do,
        delta,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        do.stride(3),
        seqlen_q,
        head_dim=Lk,
        BLOCK_M=BLOCK,
        D_HEAD=BLOCK_DMODEL,
    )
    grid = lambda META: (triton.cdiv(N_CTX, META['BLOCK_N1']), 1, BATCH * N_HEAD)
    _attn_bwd[grid](
        q,
        arg_k,
        v,
        sm_scale,
        alibi_slopes,
        do,
        dq,
        dk,
        dv,
        M,
        delta,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        N_HEAD,
        N_CTX,
        BLOCK_DMODEL= BLOCK_DMODEL,
        BLOCK_M1=BLOCK_M1,
        BLOCK_N1=BLOCK_N1,
        BLOCK_M2=BLOCK_M2,
        BLOCK_N2=BLOCK_N2,
        BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
        USE_ALIBI=False if alibi_slopes is None else True,
    )

    return dq, dk, dv, None


def varlen_bwd(dout, q, k, v, out, softmax_lse, dq, dk, dv, *args, **kwargs):
    pass