use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput};
use pegainfer::ops;
use pegainfer::tensor::{DeviceContext, DeviceVec};

use super::common::{
    EPS, HEAD_DIM_128, INTERMEDIATE_DIM, OUT_DIM, ROPE_THETA_QWEN3, VECTOR_DIM, configure_group,
    device_matrix, device_vec, iter_sync, positive_device_vec, rope_cache,
};

pub fn bench_elementwise_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("ops_elementwise");
    configure_group(&mut group);

    group.throughput(Throughput::Elements((OUT_DIM * VECTOR_DIM) as u64));
    group.bench_function(BenchmarkId::new("gemv", VECTOR_DIM), |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let matrix = device_matrix(&ctx, OUT_DIM, VECTOR_DIM).expect("failed to allocate matrix");
        let x = device_vec(&ctx, VECTOR_DIM).expect("failed to allocate x");
        let mut gemv_out = DeviceVec::zeros(&ctx, OUT_DIM).expect("failed to allocate gemv out");
        iter_sync(b, &ctx, || {
            ops::gemv(&ctx, &matrix, &x, &mut gemv_out).expect("gemv failed");
        });
    });

    group.throughput(Throughput::Elements(VECTOR_DIM as u64));
    group.bench_function(BenchmarkId::new("rms_norm_into", VECTOR_DIM), |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let rms_x = device_vec(&ctx, VECTOR_DIM).expect("failed to allocate rms x");
        let rms_weight =
            positive_device_vec(&ctx, VECTOR_DIM).expect("failed to allocate rms weight");
        let mut rms_out = DeviceVec::zeros(&ctx, VECTOR_DIM).expect("failed to allocate rms out");
        iter_sync(b, &ctx, || {
            ops::rms_norm_into(&ctx, &rms_x, &rms_weight, EPS, &mut rms_out)
                .expect("rms_norm_into failed");
        });
    });

    group.throughput(Throughput::Elements((INTERMEDIATE_DIM * VECTOR_DIM) as u64));
    group.bench_function(BenchmarkId::new("fused_mlp_into", VECTOR_DIM), |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let x = device_vec(&ctx, VECTOR_DIM).expect("failed to allocate x");
        let gate_proj = device_matrix(&ctx, INTERMEDIATE_DIM, VECTOR_DIM)
            .expect("failed to allocate gate proj");
        let up_proj =
            device_matrix(&ctx, INTERMEDIATE_DIM, VECTOR_DIM).expect("failed to allocate up proj");
        let down_proj = device_matrix(&ctx, VECTOR_DIM, INTERMEDIATE_DIM)
            .expect("failed to allocate down proj");
        let mut act = DeviceVec::zeros(&ctx, INTERMEDIATE_DIM).expect("failed to allocate act");
        let mut mlp_out = DeviceVec::zeros(&ctx, VECTOR_DIM).expect("failed to allocate mlp out");
        iter_sync(b, &ctx, || {
            ops::fused_mlp_into(
                &ctx,
                &x,
                &gate_proj,
                &up_proj,
                &down_proj,
                &mut act,
                &mut mlp_out,
            )
            .expect("fused_mlp_into failed");
        });
    });

    group.throughput(Throughput::Elements(HEAD_DIM_128 as u64));
    group.bench_function(BenchmarkId::new("rope", HEAD_DIM_128), |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let rope_x = device_vec(&ctx, HEAD_DIM_128).expect("failed to allocate rope x");
        let (rope_cos, rope_sin) = rope_cache(&ctx, 1, HEAD_DIM_128, ROPE_THETA_QWEN3)
            .expect("failed to create rope cache");
        iter_sync(b, &ctx, || {
            let out = ops::rope(&ctx, &rope_x, &rope_cos, &rope_sin).expect("rope failed");
            black_box(out);
        });
    });

    group.throughput(Throughput::Elements(VECTOR_DIM as u64));
    group.bench_function(BenchmarkId::new("add", VECTOR_DIM), |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let add_a = device_vec(&ctx, VECTOR_DIM).expect("failed to allocate add lhs");
        let add_b = device_vec(&ctx, VECTOR_DIM).expect("failed to allocate add rhs");
        iter_sync(b, &ctx, || {
            let out = ops::add(&ctx, &add_a, &add_b).expect("add failed");
            black_box(out);
        });
    });

    group.throughput(Throughput::Elements(VECTOR_DIM as u64));
    group.bench_function(BenchmarkId::new("add_inplace", VECTOR_DIM), |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let mut add_a = device_vec(&ctx, VECTOR_DIM).expect("failed to allocate add lhs");
        let add_b = device_vec(&ctx, VECTOR_DIM).expect("failed to allocate add rhs");
        iter_sync(b, &ctx, || {
            ops::add_inplace(&ctx, &mut add_a, &add_b).expect("add_inplace failed");
        });
    });

    group.throughput(Throughput::Elements(VECTOR_DIM as u64));
    group.bench_function(
        BenchmarkId::new("fused_add_rms_norm_into", VECTOR_DIM),
        |b| {
            let ctx = DeviceContext::new().expect("failed to create CUDA context");
            let mut hidden = device_vec(&ctx, VECTOR_DIM).expect("failed to allocate hidden");
            let residual = device_vec(&ctx, VECTOR_DIM).expect("failed to allocate residual");
            let fused_weight =
                positive_device_vec(&ctx, VECTOR_DIM).expect("failed to allocate fused weight");
            let mut fused_out =
                DeviceVec::zeros(&ctx, VECTOR_DIM).expect("failed to allocate fused out");
            iter_sync(b, &ctx, || {
                ops::fused_add_rms_norm_into(
                    &ctx,
                    &mut hidden,
                    &residual,
                    &fused_weight,
                    EPS,
                    &mut fused_out,
                )
                .expect("fused_add_rms_norm_into failed");
            });
        },
    );

    group.throughput(Throughput::Elements((VECTOR_DIM * 4) as u64));
    group.bench_function(BenchmarkId::new("argmax", VECTOR_DIM * 4), |b| {
        let ctx = DeviceContext::new().expect("failed to create CUDA context");
        let argmax_x = device_vec(&ctx, VECTOR_DIM * 4).expect("failed to allocate argmax input");
        iter_sync(b, &ctx, || {
            let token = ops::argmax(&ctx, &argmax_x).expect("argmax failed");
            black_box(token);
        });
    });

    group.finish();
}
