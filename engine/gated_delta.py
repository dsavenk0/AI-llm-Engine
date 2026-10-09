# Gated delta rule recurrence (GatedDeltaNet) as a Metal kernel.
# Kernel source from mlx-lm (Copyright © Apple Inc., MIT) — see engine/NOTICE.
from functools import partial

import mlx.core as mx
import mlx.nn as nn


@partial(mx.compile, shapeless=True)
def compute_g(A_log, a, dt_bias):
    return mx.exp(-mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias))


_SOURCE = """
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;

    auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
    y += b_idx * T * Hv * Dv + hv_idx * Dv;

    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;

    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;

    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
    }

    auto g_ = g + b_idx * T * Hv;
    auto beta_ = beta + b_idx * T * Hv;

    for (int t = 0; t < T; ++t) {
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] * g_[hv_idx];
        kv_mem += state[i] * k_[s_idx];
      }
      kv_mem = simd_sum(kv_mem);
      auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];
      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] + k_[s_idx] * delta;
        out += state[i] * q_[s_idx];
      }
      out = simd_sum(out);
      if (thread_index_in_simdgroup == 0) {
        y[dv_idx] = static_cast<InT>(out);
      }
      q_ += Hk * Dk;
      k_ += Hk * Dk;
      v_ += Hv * Dv;
      y += Hv * Dv;
      g_ += Hv;
      beta_ += Hv;
    }
    for (int i = 0; i < n_per_t; ++i) {
      o_state[n_per_t * dk_idx + i] = static_cast<StT>(state[i]);
    }
"""

_kernel = mx.fast.metal_kernel(
    name="engine_gated_delta_step",
    input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
    output_names=["y", "state_out"],
    source=_SOURCE,
)


def gated_delta_update(q, k, v, a, b, A_log, dt_bias, state=None):
    """q, k: [B,T,Hk,Dk]  v: [B,T,Hv,Dv]  a, b: [B,T,Hv]  state: [B,Hv,Dv,Dk] fp32."""
    return gated_delta_kernel(q, k, v, compute_g(A_log, a, dt_bias), mx.sigmoid(b), state)


def gated_delta_kernel(q, k, v, g, beta, state=None):
    """Recurrence with precomputed decay g and write strength beta ([B,T,Hv])."""
    B, T, Hk, Dk = k.shape
    Hv, Dv = v.shape[2:]
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    return _kernel(
        inputs=[q, k, v, g, beta, state, T],
        template=[("InT", q.dtype), ("StT", state.dtype),
                  ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
        grid=(32, Dv, B * Hv),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, Hv, Dv), state.shape],
        output_dtypes=[q.dtype, state.dtype],
    )
