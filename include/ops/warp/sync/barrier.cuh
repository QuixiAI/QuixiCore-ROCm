/**
 * @file
 * @brief Synchronization primitives for gfx1250.
 *
 * gfx1250 replaces the unified `s_waitcnt` with per-counter waits and exposes
 * a split barrier that lets a warp signal "data ready" before waiting on
 * the corresponding consumer barrier. This header wraps every primitive
 * behind a clean `kittens::sync::*` API; the actual instructions are emitted
 * via clang `__builtin_amdgcn_*` builtins where available, and inline
 * assembly otherwise (per-counter waits other than `asynccnt`/`tensorcnt`
 * are not yet exposed as clang builtins as of LLVM 23).
 */

#pragma once

#ifdef KITTENS_UDNA1

#include "../../../common/common.cuh"

namespace kittens {
namespace sync {

/* ----------  SPLIT BARRIER (BLOCK-WIDE)  ---------- */

/**
 * @brief Signal a block-wide split barrier.
 *
 * Lowers to `s_barrier_signal -1`. May be issued from any warp and returns
 * immediately; only `wait()` blocks until every warp in the block has signalled.
 */
__device__ __forceinline__ void arrive() {
    __builtin_amdgcn_s_barrier_signal(-1);
}

/**
 * @brief Wait on a block-wide split barrier.
 *
 * Lowers to `s_barrier_wait -1`. Blocks until every warp in the block has
 * called `arrive()` since the last completion of this barrier.
 */
__device__ __forceinline__ void wait() {
    __builtin_amdgcn_s_barrier_wait(-1);
}

/**
 * @brief Block-wide barrier (signal + wait).
 *
 * Semantically equivalent to `__syncthreads()`. Prefer the split form
 * (`arrive()` followed by independent work followed by `wait()`) when the
 * window between signalling and waiting can be filled with non-dependent
 * instructions.
 */
__device__ __forceinline__ void sync() {
    arrive();
    wait();
}

/* ----------  GFX12+ PER-COUNTER WAITS  ---------- */
//
// Each gfx1250 wait counter is 6 bits; the `N` template parameter is the
// maximum number of in-flight ops that may remain after the wait. The default
// `N=0` drains the counter completely (semantically a full sync of that class).
// Use a non-zero `N` to keep a `K`-deep pipeline running, draining one slot
// at a time as new ops are issued.

/**
 * @brief Wait for outstanding global (and texture) loads, leaving up to N in flight.
 *
 * Lowers to `s_wait_loadcnt N`. Required after a `global_load_async_to_lds`
 * or any `global_load_*` whose results are about to be read.
 *
 * @note Clang 23 does not yet expose `__builtin_amdgcn_s_wait_loadcnt`;
 *       we emit the instruction directly with `N` as an immediate operand.
 */
template<int N = 0>
__device__ __forceinline__ void wait_load() {
    static_assert(N >= 0 && N < 64, "loadcnt is 6-bit; max 63");
    asm volatile("s_wait_loadcnt %0" :: "i"(N) : "memory");
}

/**
 * @brief Wait for outstanding global stores, leaving up to N in flight.
 *
 * Lowers to `s_wait_storecnt N`.
 */
template<int N = 0>
__device__ __forceinline__ void wait_store() {
    static_assert(N >= 0 && N < 64, "storecnt is 6-bit; max 63");
    asm volatile("s_wait_storecnt %0" :: "i"(N) : "memory");
}

/**
 * @brief Wait for outstanding LDS (DS_*) operations, leaving up to N in flight.
 *
 * Lowers to `s_wait_dscnt N`. Required between LDS writes (or `ds_load_b*`
 * issues) and a dependent VALU/WMMA consumer.
 */
template<int N = 0>
__device__ __forceinline__ void wait_ds() {
    static_assert(N >= 0 && N < 64, "dscnt is 6-bit; max 63");
    asm volatile("s_wait_dscnt %0" :: "i"(N) : "memory");
}

/**
 * @brief Wait for outstanding kernel-message ops, leaving up to N in flight.
 *
 * Lowers to `s_wait_kmcnt N`.
 */
template<int N = 0>
__device__ __forceinline__ void wait_km() {
    static_assert(N >= 0 && N < 64, "kmcnt is 6-bit; max 63");
    asm volatile("s_wait_kmcnt %0" :: "i"(N) : "memory");
}

/**
 * @brief Wait for outstanding async global->LDS transfers, leaving up to N in flight.
 *
 * Lowers to `s_wait_asynccnt N`. Drains anything started by
 * `__builtin_amdgcn_(global|cluster)_load_async_to_lds_*`.
 */
template<int N = 0>
__device__ __forceinline__ void wait_async() {
    static_assert(N >= 0 && N < 64, "asynccnt is 6-bit; max 63");
    __builtin_amdgcn_s_wait_asynccnt(N);
}

/**
 * @brief Wait for outstanding TDM transfers, leaving up to N in flight.
 *
 * Lowers to `s_wait_tensorcnt N`. Drains anything started by
 * `__builtin_amdgcn_tensor_load_to_lds` or `tensor_store_from_lds`.
 *
 * @code
 *   load_tdm(buf[0], ...);
 *   load_tdm(buf[1], ...);
 *   load_tdm(buf[2], ...);
 *   for (int k = 0; k + 3 < K; ++k) {
 *       sync::wait_tensor<2>();           // drain one slot, two stay in flight
 *       consume(buf[k % 3]);
 *       load_tdm(buf[k % 3], ...);
 *   }
 *   sync::wait_tensor<0>();               // drain the tail
 * @endcode
 */
template<int N = 0>
__device__ __forceinline__ void wait_tensor() {
    static_assert(N >= 0 && N < 64, "tensorcnt is 6-bit; max 63");
    __builtin_amdgcn_s_wait_tensorcnt(N);
}

/**
 * @brief Memory fence covering both global loads and LDS ops.
 *
 * Convenience for the common "producer side" pattern: ensure all in-flight
 * loads have settled into LDS before signalling consumers.
 */
__device__ __forceinline__ void fence() {
    wait_load<0>();
    wait_ds<0>();
}

/* ----------  LDS BARRIER CELLS (FOR TDM AUTO-ARRIVE)  ---------- */
//
// gfx1250 hosts a 64-bit barrier cell in LDS (per the gfx1250 spec):
//   bits  0..31  pending count   (decrements on arrive)
//   bit      32  phase           (flips when pending reaches 0)
//   bits 33..63  init count      (reload value at phase flip)
//
// A TDM descriptor can set `atomic_barrier_enable` + `atomic_barrier_address`
// so the hardware emits `DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64` on completion.
// The consumer then waits on the cell's phase flip instead of draining the
// global `tensorcnt`.

/**
 * @brief 64-bit LDS barrier cell.
 *
 * Allocate one (or more) as `__shared__ kittens::sync::barrier_lds bar;`
 * and prime once with `init_barrier(&bar.state, count)` before any
 * `load_tdm_arrive` referencing this cell. `count` is the number of arrivals
 * required to flip the phase.
 */
struct alignas(8) barrier_lds { uint64_t state; };

/// @brief Initialize an LDS barrier cell to expect `count` arrivals per phase.
__device__ __forceinline__ void init_barrier(uint64_t* bar, uint32_t count) {
    *bar = uint64_t(count) | (uint64_t(count) << 33);
}

/**
 * @brief Block on `bar` until its phase bit matches `expected_phase`.
 *
 * The hardware can wake sleeping waves on a phase flip; `s_sleep 1` yields
 * the SIMD between polls so this is not a busy spin. Callers maintain a
 * parity bit per barrier and pass `expected_phase = (phase ^= 1)` each
 * time they wait.
 */
__device__ __forceinline__ void wait_barrier(uint64_t* bar, int expected_phase) {
    const uint32_t lds_addr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(bar));
    while (true) {
        uint64_t v;
        asm volatile("ds_load_b64 %0, %1 offset:0"
            : "=v"(v) : "v"(lds_addr) : "memory");
        if (int((v >> 32) & 1) == expected_phase) break;
        __builtin_amdgcn_s_sleep(1);
    }
}

/**
 * @brief Arrive at an LDS barrier cell from an async-ordered path.
 *
 * Lowers to `DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64`. Use this to manually
 * arrive at a cell (the auto-arrive form is encoded in the TDM descriptor
 * via `load_tdm_arrive`).
 */
__device__ __forceinline__ void async_barrier_arrive(uint64_t* lds_counter) {
    uintptr_t lds_uint = reinterpret_cast<uintptr_t>(lds_counter);
    __builtin_amdgcn_ds_atomic_async_barrier_arrive_b64(
        reinterpret_cast<long __attribute__((address_space(3)))*>(lds_uint));
}

} // namespace sync
} // namespace kittens

#endif // KITTENS_UDNA1
