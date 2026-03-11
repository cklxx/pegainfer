use criterion::{BenchmarkId, Criterion, Throughput};
use pegainfer::ops;
use pegainfer::tensor::{DeviceContext, DeviceVec};

use super::common::{
    ATTN_SEQ_LEN, CONV_KERNEL_SIZE, EPS, HEAD_DIM_256, KV_HEADS_256, LINEAR_KEY_DIM,
    LINEAR_KEY_HEADS, LINEAR_VALUE_DIM, LINEAR_VALUE_HEADS, MAX_SEQ_LEN, Q_HEADS_256,
    ROPE_THETA_QWEN35, ROTARY_DIM_256, configure_group, decode_meta, device_vec, f32_slice,
    iter_sync, positive_device_vec, rope_cache, zero_f32_slice,
};

pub fn bench_qwen35_state_ops(c: &mut Criterion) {
    let conv_channels =
        LINEAR_KEY_HEADS * LINEAR_KEY_DIM * 2 + LINEAR_VALUE_HEADS * LINEAR_VALUE_DIM;
    let scale = 1.0 / (HEAD_DIM_256 as f32).sqrt();

    let mut group = c.benchmark_group("ops_qwen35_state");
    configure_group(&mut group);

    group.throughput(Throughput::Elements(conv_channels as u64));
    group.bench_function("conv1d_decode_into", |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let conv_x = device_vec(&ctx, conv_channels).expect("failed to allocate conv input");
        let conv_weight = device_vec(&ctx, conv_channels * CONV_KERNEL_SIZE)
            .expect("failed to allocate conv weight");
        let mut conv_state = DeviceVec::zeros(&ctx, conv_channels * (CONV_KERNEL_SIZE - 1))
            .expect("failed to allocate conv state");
        let mut conv_out =
            DeviceVec::zeros(&ctx, conv_channels).expect("failed to allocate conv out");
        iter_sync(b, &ctx, || {
            ops::conv1d_decode_into(
                &ctx,
                &conv_x,
                &conv_weight,
                &mut conv_state,
                &mut conv_out,
                CONV_KERNEL_SIZE,
            )
            .expect("conv1d_decode_into failed");
        });
    });

    group.throughput(Throughput::Elements((conv_channels * ATTN_SEQ_LEN) as u64));
    group.bench_function(BenchmarkId::new("conv1d_prefill_into", ATTN_SEQ_LEN), |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let x_seq =
            device_vec(&ctx, conv_channels * ATTN_SEQ_LEN).expect("failed to allocate conv seq");
        let conv_weight = device_vec(&ctx, conv_channels * CONV_KERNEL_SIZE)
            .expect("failed to allocate conv weight");
        let mut conv_state_prefill = DeviceVec::zeros(&ctx, conv_channels * (CONV_KERNEL_SIZE - 1))
            .expect("failed to allocate prefill conv state");
        let mut conv_prefill_out = DeviceVec::zeros(&ctx, conv_channels * ATTN_SEQ_LEN)
            .expect("failed to allocate prefill conv out");
        iter_sync(b, &ctx, || {
            ops::conv1d_prefill_into(
                &ctx,
                &x_seq,
                &conv_weight,
                &mut conv_state_prefill,
                &mut conv_prefill_out,
                conv_channels,
                ATTN_SEQ_LEN,
                CONV_KERNEL_SIZE,
            )
            .expect("conv1d_prefill_into failed");
        });
    });

    group.throughput(Throughput::Elements(
        (LINEAR_VALUE_HEADS * LINEAR_KEY_DIM * LINEAR_VALUE_DIM) as u64,
    ));
    group.bench_function("gated_delta_rule_decode_into", |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let qkv = device_vec(&ctx, conv_channels).expect("failed to allocate qkv");
        let b_proj = device_vec(&ctx, LINEAR_VALUE_HEADS).expect("failed to allocate b_proj");
        let a_proj = device_vec(&ctx, LINEAR_VALUE_HEADS).expect("failed to allocate a_proj");
        let dt_bias =
            positive_device_vec(&ctx, LINEAR_VALUE_HEADS).expect("failed to allocate dt_bias");
        let a_log = f32_slice(&ctx, LINEAR_VALUE_HEADS).expect("failed to allocate a_log");
        let mut state =
            zero_f32_slice(&ctx, LINEAR_VALUE_HEADS * LINEAR_KEY_DIM * LINEAR_VALUE_DIM)
                .expect("failed to allocate recurrent state");
        let mut recurrent_out = DeviceVec::zeros(&ctx, LINEAR_VALUE_HEADS * LINEAR_VALUE_DIM)
            .expect("failed to allocate recurrent out");
        iter_sync(b, &ctx, || {
            ops::gated_delta_rule_decode_into(
                &ctx,
                &qkv,
                &b_proj,
                &a_proj,
                &dt_bias,
                &a_log,
                &mut state,
                &mut recurrent_out,
                LINEAR_KEY_HEADS,
                LINEAR_VALUE_HEADS,
                LINEAR_KEY_DIM,
                LINEAR_VALUE_DIM,
            )
            .expect("gated_delta_rule_decode_into failed");
        });
    });

    group.throughput(Throughput::Elements((Q_HEADS_256 * HEAD_DIM_256) as u64));
    group.bench_function(
        BenchmarkId::new("fused_attention_hd256_decode_into", ATTN_SEQ_LEN),
        |b| {
            let ctx = DeviceContext::new().expect("failed to create CUDA context");
            let q_full = device_vec(&ctx, Q_HEADS_256 * HEAD_DIM_256 * 2)
                .expect("failed to allocate q_full");
            let k_full =
                device_vec(&ctx, KV_HEADS_256 * HEAD_DIM_256).expect("failed to allocate k_full");
            let v_full =
                device_vec(&ctx, KV_HEADS_256 * HEAD_DIM_256).expect("failed to allocate v_full");
            let q_norm =
                positive_device_vec(&ctx, HEAD_DIM_256).expect("failed to allocate q_norm");
            let k_norm =
                positive_device_vec(&ctx, HEAD_DIM_256).expect("failed to allocate k_norm");
            let (cos_cache, sin_cache) =
                rope_cache(&ctx, MAX_SEQ_LEN, ROTARY_DIM_256, ROPE_THETA_QWEN35)
                    .expect("failed to create rope cache");
            let current_pos = ATTN_SEQ_LEN - 1;
            let decode_meta = decode_meta(&ctx, 7, current_pos, ATTN_SEQ_LEN)
                .expect("failed to allocate decode meta");
            let cache_len = KV_HEADS_256 * MAX_SEQ_LEN * HEAD_DIM_256;
            let mut k_cache =
                DeviceVec::zeros(&ctx, cache_len).expect("failed to allocate k cache");
            let mut v_cache =
                DeviceVec::zeros(&ctx, cache_len).expect("failed to allocate v cache");
            let mut attn_out = DeviceVec::zeros(&ctx, Q_HEADS_256 * HEAD_DIM_256)
                .expect("failed to allocate attention out");
            iter_sync(b, &ctx, || {
                ops::fused_attention_hd256_decode_into(
                    &ctx,
                    &q_full,
                    &k_full,
                    &v_full,
                    &q_norm,
                    &k_norm,
                    &cos_cache,
                    &sin_cache,
                    &decode_meta,
                    &mut k_cache,
                    &mut v_cache,
                    &mut attn_out,
                    Q_HEADS_256,
                    KV_HEADS_256,
                    ROTARY_DIM_256,
                    scale,
                    EPS,
                )
                .expect("fused_attention_hd256_decode_into failed");
            });
        },
    );

    group.throughput(Throughput::Elements((Q_HEADS_256 * HEAD_DIM_256) as u64));
    group.bench_function(
        BenchmarkId::new("fused_attention_hd256_single_token_into", ATTN_SEQ_LEN),
        |b| {
            let ctx = DeviceContext::new().expect("failed to create CUDA context");
            let q_full = device_vec(&ctx, Q_HEADS_256 * HEAD_DIM_256 * 2)
                .expect("failed to allocate q_full");
            let k_full =
                device_vec(&ctx, KV_HEADS_256 * HEAD_DIM_256).expect("failed to allocate k_full");
            let v_full =
                device_vec(&ctx, KV_HEADS_256 * HEAD_DIM_256).expect("failed to allocate v_full");
            let q_norm =
                positive_device_vec(&ctx, HEAD_DIM_256).expect("failed to allocate q_norm");
            let k_norm =
                positive_device_vec(&ctx, HEAD_DIM_256).expect("failed to allocate k_norm");
            let (cos_cache, sin_cache) =
                rope_cache(&ctx, MAX_SEQ_LEN, ROTARY_DIM_256, ROPE_THETA_QWEN35)
                    .expect("failed to create rope cache");
            let current_pos = ATTN_SEQ_LEN - 1;
            let cos_pos = cos_cache.view(current_pos * ROTARY_DIM_256, ROTARY_DIM_256);
            let sin_pos = sin_cache.view(current_pos * ROTARY_DIM_256, ROTARY_DIM_256);
            let cache_len = KV_HEADS_256 * MAX_SEQ_LEN * HEAD_DIM_256;
            let mut k_cache =
                DeviceVec::zeros(&ctx, cache_len).expect("failed to allocate k cache");
            let mut v_cache =
                DeviceVec::zeros(&ctx, cache_len).expect("failed to allocate v cache");
            let mut attn_out = DeviceVec::zeros(&ctx, Q_HEADS_256 * HEAD_DIM_256)
                .expect("failed to allocate attention out");
            iter_sync(b, &ctx, || {
                ops::fused_attention_hd256_single_token_into(
                    &ctx,
                    &q_full,
                    &k_full,
                    &v_full,
                    &q_norm,
                    &k_norm,
                    &cos_pos,
                    &sin_pos,
                    &mut k_cache,
                    &mut v_cache,
                    &mut attn_out,
                    Q_HEADS_256,
                    KV_HEADS_256,
                    current_pos,
                    ATTN_SEQ_LEN,
                    ROTARY_DIM_256,
                    scale,
                    EPS,
                )
                .expect("fused_attention_hd256_single_token_into failed");
            });
        },
    );

    group.finish();
}
