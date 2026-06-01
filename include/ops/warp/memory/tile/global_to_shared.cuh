/**
 * @file
 * @brief Functions for transferring data directly between global and shared memory and back.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

template<int axis, bool assume_aligned,
         ducks::st::all ST, ducks::gl::all GL,
         ducks::coord::tile COORD = coord<ST>,
         int N_THREADS = WARP_THREADS>
__device__ inline void load(ST& dst, const GL& src, const COORD& idx)
{
    using T = typename ST::dtype;

    constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_warp = bytes_per_thread * kittens::WARP_THREADS;
    constexpr int memcpy_per_tile = ST::rows * ST::cols * sizeof(T) / (bytes_per_thread * N_THREADS);
    static_assert(ST::rows * ST::cols * sizeof(T) >= bytes_per_warp, "shared tile must be at least 1024 bytes");
    
    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    const int laneid = kittens::laneid();
    const int warpid = kittens::warpid() % num_warps;

    const int row_stride = src.template stride<axis>();

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    T* global_ptr = (T*)&src[unit_coord];
    i32x4 srsrc = make_srsrc(global_ptr, row_stride * ST::rows * sizeof(T));

    const uintptr_t lds_base = reinterpret_cast<uintptr_t>(&dst.data[0]) + (warpid * bytes_per_warp);

    if constexpr (memcpy_per_tile > 0) {

        #pragma unroll
        for (int i = 0; i < memcpy_per_tile; i++) {

            const int lane_byte_offset = (laneid * bytes_per_thread) + (warpid * bytes_per_warp) + (i * num_warps * bytes_per_warp);
            const int subtile_id = lane_byte_offset / ST::underlying_subtile_bytes;
            const int subtile_row = subtile_id / ST::underlying_subtiles_per_row;
            const int subtile_col = subtile_id % ST::underlying_subtiles_per_row;
            const int subtile_lane_byte_offset = lane_byte_offset % ST::underlying_subtile_bytes;

            const int row = subtile_lane_byte_offset / ST::underlying_subtile_row_bytes;
            const int col = (subtile_lane_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);

            const uint32_t swizzled_shared_byte_offset = dst.swizzle({row, col});

            const int swizzled_global_row = (swizzled_shared_byte_offset / ST::underlying_subtile_row_bytes) + subtile_row * ST::underlying_subtile_rows;
            const int swizzled_global_col = (swizzled_shared_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T) + subtile_col * ST::underlying_subtile_cols;
            const uint32_t swizzled_global_byte_offset = (swizzled_global_row * row_stride + swizzled_global_col) * sizeof(T);

            uintptr_t lds_addr = lds_base + (i * num_warps * bytes_per_warp);
            as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);

            llvm_amdgcn_raw_buffer_load_lds(
                srsrc, // buffer resource
                lds_ptr,
                bytes_per_thread,
                swizzled_global_byte_offset,
                0, 
                0, // instruction offset
                static_cast<int>(coherency::cache_all)); // cache coherency
        }
    }
    // there are leftover loads that need to be handled here
    if constexpr (memcpy_per_tile * (bytes_per_thread * N_THREADS) != ST::rows * ST::cols * sizeof(T)) {

        constexpr int leftover_bytes = ST::rows * ST::cols * sizeof(T) - memcpy_per_tile * (bytes_per_thread * N_THREADS);
        constexpr int leftover_threads = leftover_bytes / bytes_per_thread;
        constexpr int leftover_warps = leftover_threads / kittens::WARP_THREADS;

        if (warpid < leftover_warps) {
            const int lane_byte_offset = (laneid * bytes_per_thread) + (warpid * bytes_per_warp) + (memcpy_per_tile * num_warps * bytes_per_warp);
            const int subtile_id = lane_byte_offset / ST::underlying_subtile_bytes;
            const int subtile_row = subtile_id / ST::underlying_subtiles_per_row;
            const int subtile_col = subtile_id % ST::underlying_subtiles_per_row;
            const int subtile_lane_byte_offset = lane_byte_offset % ST::underlying_subtile_bytes;

            const int row = subtile_lane_byte_offset / ST::underlying_subtile_row_bytes;
            const int col = (subtile_lane_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);

            const uint32_t swizzled_shared_byte_offset = dst.swizzle({row, col});

            const int swizzled_global_row = (swizzled_shared_byte_offset / ST::underlying_subtile_row_bytes) + subtile_row * ST::underlying_subtile_rows;
            const int swizzled_global_col = (swizzled_shared_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T) + subtile_col * ST::underlying_subtile_cols;
            const uint32_t swizzled_global_byte_offset = (swizzled_global_row * row_stride + swizzled_global_col) * sizeof(T);

            uintptr_t lds_addr = lds_base + (memcpy_per_tile * num_warps * bytes_per_warp);
            as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);

            llvm_amdgcn_raw_buffer_load_lds(
                srsrc, // buffer resource
                lds_ptr,
                bytes_per_thread,
                swizzled_global_byte_offset,
                0, 
                0, // instruction offset
                static_cast<int>(coherency::cache_all)); // cache coherency
        }
    }
}

template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx) {
    load<2, false, ST, GL, COORD, WARP_THREADS>(dst, src, idx);
}

template<int axis, bool assume_aligned,
         ducks::st::all ST, ducks::gl::all GL,
         int N_THREADS = WARP_THREADS>
__device__ inline void prefill_swizzled_offsets(
    ST& dst, const GL& src, uint32_t* swizzled_offsets)
{
    using T = typename ST::dtype;
 
    constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_warp = bytes_per_thread * kittens::WARP_THREADS;
    constexpr int memcpy_per_tile =  ST::rows * ST::cols * sizeof(T) / (bytes_per_thread * N_THREADS);
    static_assert(ST::rows * ST::cols * sizeof(T) >= bytes_per_warp, "shared tile must be at least 1024 bytes");

    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    const int laneid = kittens::laneid();
    const int warpid = kittens::warpid() % num_warps;

    const int row_stride = src.template stride<axis>();

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {
        const int lane_byte_offset = (laneid * bytes_per_thread) + (warpid * bytes_per_warp) + (i * num_warps * bytes_per_warp);
        const int subtile_id = lane_byte_offset / ST::underlying_subtile_bytes;
        const int subtile_row = subtile_id / ST::underlying_subtiles_per_row;
        const int subtile_col = subtile_id % ST::underlying_subtiles_per_row;
        const int subtile_lane_byte_offset = lane_byte_offset % ST::underlying_subtile_bytes;

        int row = subtile_lane_byte_offset / ST::underlying_subtile_row_bytes;
        int col = (subtile_lane_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);
        const uint32_t swizzled_shared_byte_offset = dst.swizzle({row, col});

        const int swizzled_global_row = (swizzled_shared_byte_offset / ST::underlying_subtile_row_bytes) + subtile_row * ST::underlying_subtile_rows;
        const int swizzled_global_col = (swizzled_shared_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T) + subtile_col * ST::underlying_subtile_cols;
        const uint32_t swizzled_global_byte_offset = (swizzled_global_row * row_stride + swizzled_global_col) * sizeof(T);
        swizzled_offsets[i] = swizzled_global_byte_offset;
    }

    // there are leftover loads that need to be handled here
    if constexpr (memcpy_per_tile * (bytes_per_thread * N_THREADS) != ST::rows * ST::cols * sizeof(T)) {

        constexpr int leftover_bytes = ST::rows * ST::cols * sizeof(T) - memcpy_per_tile * (bytes_per_thread * N_THREADS);
        constexpr int leftover_threads = leftover_bytes / bytes_per_thread;
        constexpr int leftover_warps = leftover_threads / kittens::WARP_THREADS;

        if (warpid < leftover_warps) {
            const int lane_byte_offset = (laneid * bytes_per_thread) + (warpid * bytes_per_warp) + (memcpy_per_tile * num_warps * bytes_per_warp);
            const int subtile_id = lane_byte_offset / ST::underlying_subtile_bytes;
            const int subtile_row = subtile_id / ST::underlying_subtiles_per_row;
            const int subtile_col = subtile_id % ST::underlying_subtiles_per_row;
            const int subtile_lane_byte_offset = lane_byte_offset % ST::underlying_subtile_bytes;

            const int row = subtile_lane_byte_offset / ST::underlying_subtile_row_bytes;
            const int col = (subtile_lane_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);

            const uint32_t swizzled_shared_byte_offset = dst.swizzle({row, col});

            const int swizzled_global_row = (swizzled_shared_byte_offset / ST::underlying_subtile_row_bytes) + subtile_row * ST::underlying_subtile_rows;
            const int swizzled_global_col = (swizzled_shared_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T) + subtile_col * ST::underlying_subtile_cols;
            const uint32_t swizzled_global_byte_offset = (swizzled_global_row * row_stride + swizzled_global_col) * sizeof(T);

            swizzled_offsets[memcpy_per_tile] = swizzled_global_byte_offset;
        }
    }
}

template<int axis, bool assume_aligned,
         ducks::st::all ST, ducks::gl::all GL,
         ducks::coord::tile COORD = coord<ST>,
         int N_THREADS = WARP_THREADS>
__device__ inline void load(ST& dst, const GL& src, const COORD& idx, const uint32_t* swizzled_offsets)
{
    using T = typename ST::dtype;

    constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    constexpr int bytes_per_warp = bytes_per_thread * kittens::WARP_THREADS;
    constexpr int elements_per_warp = bytes_per_warp / sizeof(T);
    constexpr int memcpy_per_tile = ST::rows * ST::cols * sizeof(T) / (bytes_per_thread * N_THREADS);
    static_assert(ST::rows * ST::cols * sizeof(T) >= bytes_per_warp, "shared tile must be at least 1024 bytes");
    
    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    const int warpid = kittens::warpid() % num_warps;

    const int row_stride = src.template stride<axis>();
    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    T* global_ptr = (T*)&src[unit_coord];
    i32x4 srsrc = make_srsrc(global_ptr, row_stride * ST::rows * sizeof(T));

    const T* lds_base = &dst.data[0] + (warpid * elements_per_warp);

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; i++) {

        const T* lds_elem_ptr = lds_base + (i * num_warps * elements_per_warp);
        uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_elem_ptr);
        as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);

        llvm_amdgcn_raw_buffer_load_lds(
            srsrc, // buffer resource
            lds_ptr,
            bytes_per_thread,
            swizzled_offsets[i],
            0, 
            0, // instruction offset
            static_cast<int>(coherency::cache_all)); // cache coherency
    }

    // there are leftover loads that need to be handled here
    if constexpr (memcpy_per_tile * (bytes_per_thread * N_THREADS) != ST::rows * ST::cols * sizeof(T)) {

        constexpr int leftover_bytes = ST::rows * ST::cols * sizeof(T) - memcpy_per_tile * (bytes_per_thread * N_THREADS);
        constexpr int leftover_threads = leftover_bytes / bytes_per_thread;
        constexpr int leftover_warps = leftover_threads / kittens::WARP_THREADS;

        if (warpid < leftover_warps) {

            const T* lds_elem_ptr = lds_base + (memcpy_per_tile * num_warps * elements_per_warp);
            uintptr_t lds_addr = reinterpret_cast<uintptr_t>(lds_elem_ptr);
            as3_uint32_ptr lds_ptr = (as3_uint32_ptr)(lds_addr);

            llvm_amdgcn_raw_buffer_load_lds(
                srsrc, // buffer resource
                lds_ptr,
                bytes_per_thread,
                swizzled_offsets[memcpy_per_tile],
                0, 
                0, // instruction offset
                static_cast<int>(coherency::cache_all)); // cache coherency
        }
    }
}

template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx, const uint32_t* swizzled_offsets) {
    load<2, false, ST, GL, COORD, WARP_THREADS>(dst, src, idx, swizzled_offsets);
}


using as3_uint32_ptr = __attribute__((address_space(3))) unsigned int*;
inline __device__ __forceinline__ uint32_t to_sgpr_u32(uint32_t x) {
    x = __builtin_amdgcn_readfirstlane(x); // make uniform
    asm volatile("" : "+s"(x));            // keep in SGPR class
    return x;
}

template<int axis, bool assume_aligned, ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD = coord<ST>, int N_THREADS = WARP_THREADS>
__attribute__((always_inline)) 
__device__ __forceinline__ void load(ST& dst, const GL& src, const COORD& idx,
                                const uint32_t* __restrict__ swizzled_offsets,
                                i32x4 SRD,
                                const void* base_ptr, const uint32_t lds_base)
{
    using T = typename ST::dtype;
    static_assert(sizeof(T) == 2 || sizeof(T) == 1, "only supporting 16 and 8-bit dtypes");

    constexpr int bytes_per_thread = 16;
    constexpr int bytes_per_memcpy = bytes_per_thread * N_THREADS;
    constexpr int memcpy_per_tile  = (ST::rows * ST::cols * sizeof(T)) / bytes_per_memcpy;
    static_assert(bytes_per_memcpy % 16 == 0, "LDS bump must be 16-aligned");

    constexpr int elem_per_thread = bytes_per_thread / sizeof(T);
    constexpr int elem_per_warp   = elem_per_thread * kittens::WARP_THREADS;

    // ---- compute per-tile base pointer and scalar offset (SOFF) ----
    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    T* __restrict__ gptr = (T*)&src[unit_coord];

    uint32_t SOFF = to_sgpr_u32(static_cast<uint32_t>(
    reinterpret_cast<const char*>(gptr) - reinterpret_cast<const char*>(base_ptr)
    ));

    // // ---- LDS base (byte address) as SGPR (wave-uniform) ----
    // const int num_warps = N_THREADS / kittens::WARP_THREADS;
    // const int wid = warpid() % num_warps;
    // uint32_t lds_base = to_sgpr_u32(static_cast<uint32_t>(
    // reinterpret_cast<uintptr_t>(&dst.data[0]) + wid * elem_per_warp * sizeof(T)
    // ));

    // ---- SGPR cursor we bump each iteration (no new readfirstlane) ----
    uint32_t lds_cur = lds_base;
    asm volatile("" : "+s"(lds_cur)); 

    #pragma unroll
    for (int i = 0; i < memcpy_per_tile; ++i) {
        int32_t lds_byte = lds_cur;                 // still SGPR
        asm volatile("" : "+s"(lds_byte));           // keep it SGPR at the use

        asm volatile("s_mov_b32 m0, %0" :: "s"(lds_byte));
        llvm_amdgcn_raw_buffer_load_lds(
            SRD, 
            (as3_uint32_ptr)0, 
            16, 
            swizzled_offsets[i], 
            SOFF, 
            0,
            static_cast<int>(coherency::cache_all)
        );

        // SGPR bump (compiler emits s_add_u32)
        lds_cur += bytes_per_memcpy;
    }
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx, const uint32_t* __restrict__ swizzled_offsets, i32x4 srd, const void* base_ptr, uint32_t lds_base) {
    load<2, false, ST, GL, COORD, WARP_THREADS>(dst, src, idx, swizzled_offsets, srd, base_ptr, lds_base);
}

/**
 * @brief Stores data from a shared memory tile into global memory.
 *
 * @tparam ST The type of the shared tile.
 * @param[out] dst The destination global memory array.
 * @param[in] src The source shared memory tile.
 * @param row_stride[in] The stride between rows in the destination array.
 */

template<int axis, bool assume_aligned, 
        ducks::st::all ST, ducks::gl::all GL, 
        ducks::coord::tile COORD=coord<ST>, int N_THREADS=WARP_THREADS>
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {

    using T = typename ST::dtype;
    using U = typename GL::dtype;

    static_assert(std::is_same_v<T, U>, "T and U must be the same type");
    static_assert(!std::is_same_v<T, fp8e4m3>, "Unsupported type for store");

    constexpr int bytes_per_thread = ST::underlying_subtile_bytes_per_thread;
    constexpr int elems_per_thread = bytes_per_thread / sizeof(T);
    constexpr int bytes_per_warp = bytes_per_thread * kittens::WARP_THREADS;
    constexpr int memcpy_per_tile =  ST::rows * ST::cols * sizeof(T) / (bytes_per_thread * N_THREADS);

    constexpr int num_warps = N_THREADS / kittens::WARP_THREADS;
    const int laneid = kittens::laneid();
    const int warpid = kittens::warpid() % num_warps;

    const int row_stride = dst.template stride<axis>();

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    uintptr_t dst_ptr = reinterpret_cast<uintptr_t>(&dst[unit_coord]);
    uintptr_t src_ptr = reinterpret_cast<uintptr_t>(&src.data[0]);

    if constexpr (memcpy_per_tile > 0) {

        #pragma unroll
        for (int i = 0; i < memcpy_per_tile; i++) {
            const int lane_byte_offset = (laneid * bytes_per_thread) + (warpid * bytes_per_warp) + (i * num_warps * bytes_per_warp);
            const int subtile_id = lane_byte_offset / ST::underlying_subtile_bytes;
            const int subtile_row = subtile_id / ST::underlying_subtiles_per_row;
            const int subtile_col = subtile_id % ST::underlying_subtiles_per_row;
            const int subtile_lane_byte_offset = lane_byte_offset % ST::underlying_subtile_bytes;

            const int row = subtile_lane_byte_offset / ST::underlying_subtile_row_bytes;
            const int col = (subtile_lane_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);
            const uint32_t swizzled_shared_byte_offset = src.swizzle({row, col});

            const int swizzled_global_row = (swizzled_shared_byte_offset / ST::underlying_subtile_row_bytes) + subtile_row * ST::underlying_subtile_rows;
            const int swizzled_global_col = (swizzled_shared_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T) + subtile_col * ST::underlying_subtile_cols;
            const uint32_t swizzled_global_byte_offset = (swizzled_global_row * row_stride + swizzled_global_col) * sizeof(T);

            U* dst_elem_ptr = (U*)(dst_ptr + swizzled_global_byte_offset);
            T* src_elem_ptr = (T*)(src_ptr + lane_byte_offset);

            #pragma unroll
            for (int j = 0; j < elems_per_thread; j++) {
                dst_elem_ptr[j] = kittens::base_types::convertor<U, T>::convert(src_elem_ptr[j]);
            }
        }
    }

    if constexpr (memcpy_per_tile * (bytes_per_thread * N_THREADS) != ST::rows * ST::cols * sizeof(T)) {

        constexpr int leftover_bytes = ST::rows * ST::cols * sizeof(T) - memcpy_per_tile * (bytes_per_thread * N_THREADS);
        constexpr int leftover_threads = leftover_bytes / bytes_per_thread;
        constexpr int leftover_warps = leftover_threads / kittens::WARP_THREADS;

        if (warpid < leftover_warps) {
            const int lane_byte_offset = (laneid * bytes_per_thread) + (warpid * bytes_per_warp) + (memcpy_per_tile * num_warps * bytes_per_warp);
            const int subtile_id = lane_byte_offset / ST::underlying_subtile_bytes;
            const int subtile_row = subtile_id / ST::underlying_subtiles_per_row;
            const int subtile_col = subtile_id % ST::underlying_subtiles_per_row;
            const int subtile_lane_byte_offset = lane_byte_offset % ST::underlying_subtile_bytes;

            const int row = subtile_lane_byte_offset / ST::underlying_subtile_row_bytes;
            const int col = (subtile_lane_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T);
            const uint32_t swizzled_shared_byte_offset = src.swizzle({row, col});

            const int swizzled_global_row = (swizzled_shared_byte_offset / ST::underlying_subtile_row_bytes) + subtile_row * ST::underlying_subtile_rows;
            const int swizzled_global_col = (swizzled_shared_byte_offset % ST::underlying_subtile_row_bytes) / sizeof(T) + subtile_col * ST::underlying_subtile_cols;
            const uint32_t swizzled_global_byte_offset = (swizzled_global_row * row_stride + swizzled_global_col) * sizeof(T);

            U* dst_elem_ptr = (U*)(dst_ptr + swizzled_global_byte_offset);
            T* src_elem_ptr = (T*)(src_ptr + lane_byte_offset);

            #pragma unroll
            for (int j = 0; j < elems_per_thread; j++) {
                dst_elem_ptr[j] = kittens::base_types::convertor<U, T>::convert(src_elem_ptr[j]);
            }
        }
    }
}
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD=coord<ST>>
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    store<2, false, ST, GL, COORD, WARP_THREADS>(dst, src, idx);
}

/* ========================================================================== *
 * gfx1250 raw-pointer global <-> LDS transfers
 *
 * The gfx1250 family relies on `global_load_async_to_lds_*`,
 * `cluster_load_async_to_lds_*`, and `tensor_load_to_lds` (TDM) which all take
 * raw 16B-aligned LDS pointers and emit single 16B requests per lane (or per
 * warp, for TDM). Padded LDS layouts further complicate offset math, so the
 * gfx1250 path takes raw `T*` LDS slabs plus a compile-time padding descriptor.
 * Kernels allocate the slab via `shared_allocator::allocate_in<segment<I>>`
 * and call these helpers to populate it.
 *
 * Lives in `kittens::g2s::` to avoid colliding with the CDNA `load(ST&, ...)`
 * overloads in the parent `kittens::` namespace.
 * ========================================================================== */
#ifdef KITTENS_UDNA1

/**
 * @brief Compile-time LDS padding descriptor for gfx1250.
 *
 * Encodes the "insert `AMOUNT` pad elements every `INTERVAL` elements" rule
 * used to break bank-conflict hotspots on gfx1250 LDS. The defaults
 * (interval = 128 bf16 = 256 B, amount = 8 bf16 = 16 B) match the
 * recommended layout for 16-bit operands.
 */
template<int INTERVAL = 128, int AMOUNT = 8>
struct lds_padded {
    static constexpr int interval = INTERVAL;
    static constexpr int amount   = AMOUNT;
    __device__ __host__ __forceinline__ static constexpr int padded(int flat) {
        return flat + (flat / INTERVAL) * AMOUNT;
    }
    static constexpr int padded_elems(int total) {
        return total + (total / INTERVAL) * AMOUNT;
    }
};

/// @brief Padding descriptor for unpadded LDS layouts.
struct lds_nopad {
    static constexpr int interval = 0;
    static constexpr int amount   = 0;
    __device__ __host__ __forceinline__ static constexpr int padded(int flat) { return flat; }
    static constexpr int padded_elems(int total) { return total; }
};

/// @brief Default LDS padding for bf16 GEMMs on gfx1250.
using lds_pad_default = lds_padded<128, 8>;

namespace g2s {
namespace detail {
/// @brief 16B (`int4`) vector types tagged with the address spaces the
///        gfx1250 `*_load_async_to_lds_b128` builtins require.
using i32x4_vec   = int __attribute__((__vector_size__(16)));
using i32x4_gvec  = int __attribute__((__vector_size__(16))) __attribute__((address_space(1)));
using i32x4_lvec  = int __attribute__((__vector_size__(16))) __attribute__((address_space(3)));

/**
 * @brief Subtile-major flat index helper for gfx1250 LDS layouts.
 *
 * Mirrors the indexing used by `ops/warp/memory/tile/shared_to_register.cuh`
 * gfx1250 path: rows/cols within a subtile are stored row-major, subtiles are
 * laid out row-major across the tile.
 */
template<int ROWS, int COLS, int SUB_ROWS, int SUB_COLS>
__device__ __forceinline__ int subtile_flat(int flat) {
    constexpr int sub_elems    = SUB_ROWS * SUB_COLS;
    constexpr int subs_per_row = COLS / SUB_COLS;
    const int subtile_id = flat / sub_elems;
    const int local_idx  = flat % sub_elems;
    const int local_row  = local_idx / SUB_COLS;
    const int local_col  = local_idx % SUB_COLS;
    const int sub_r      = subtile_id / subs_per_row;
    const int sub_c      = subtile_id % subs_per_row;
    return sub_r * SUB_ROWS * COLS
         + sub_c * SUB_COLS
         + local_row * COLS
         + local_col;
}
} // namespace detail (g2s)

/**
 * @brief Cooperative register-mediated global -> LDS tile copy (gfx1250 baseline).
 *
 * Plain `global_load` -> VGPR -> `ds_store` path. Use this when no async
 * intrinsic is available or for correctness baselines. The `Pad` parameter
 * controls the per-element LDS placement; pass `lds_nopad` for flat layouts.
 */
template<typename Pad = lds_nopad, int ROWS = 0, int COLS = 0, int N_THREADS = WARP_THREADS,
         typename T, ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
__device__ inline void load(T* __restrict__ lds_dst, const GL& src, const COORD& idx,
                            int row_stride)
{
    static_assert(ROWS > 0 && COLS > 0, "ROWS and COLS must be specified");
    constexpr int total_elems = ROWS * COLS;
    const int tid = threadIdx.x;
    // The COORD is interpreted as tile-index coordinates `{b, d, tile_row, tile_col}`
    // -- convert to element coordinates by multiplying the trailing two by ROWS/COLS.
    const int gr_base = idx.r * ROWS;
    const int gc_base = idx.c * COLS;
    const T* base = src.raw_ptr
                  + (((int64_t(idx.b) * src.depth() + idx.d) * src.rows() + gr_base)
                     * src.cols() + gc_base);

    #pragma unroll
    for (int i = tid; i < total_elems; i += N_THREADS) {
        const int row = i / COLS;
        const int col = i % COLS;
        // Subtile-major LDS layout (rows of 16x32 subtiles).
        const int lds_flat = detail::subtile_flat<ROWS, COLS, 16, 32>(i);
        lds_dst[Pad::padded(lds_flat)] = base[row * row_stride + col];
    }
}

/**
 * @brief Cooperative async global -> LDS tile copy on gfx1250.
 *
 * Lowers to `global_load_async_to_lds_b128` (single-WG) when `cluster_mask == 0`,
 * and to `cluster_load_async_to_lds_b128` (multicast) when non-zero. Each lane
 * issues one 16-byte transfer; the warp covers `8 * N_THREADS` elements per
 * iteration. Drain with `kittens::sync::wait_async()` before consuming.
 *
 * @tparam Pad      LDS padding descriptor.
 * @tparam ROWS,COLS  Tile shape (elements).
 * @tparam N_THREADS  Number of threads participating in the load.
 * @param  lds_dst    16B-aligned LDS pointer (typically `bf16*`).
 * @param  src        Global tile descriptor.
 * @param  idx        Tile coordinate inside `src`.
 * @param  row_stride Element stride between rows in `src`.
 * @param  cluster_mask `M0` cluster multicast mask (0 for single-WG, non-zero for CGA).
 */
template<typename Pad = lds_nopad, int ROWS = 0, int COLS = 0, int N_THREADS = WARP_THREADS,
         typename T, ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
__device__ inline void load_async(T* __restrict__ lds_dst, const GL& src, const COORD& idx,
                                  int row_stride, uint32_t cluster_mask = 0)
{
    static_assert(ROWS > 0 && COLS > 0, "ROWS and COLS must be specified");
    static_assert(sizeof(T) * 8 == 16, "load_async issues one b128 (16B) per lane");
    constexpr int elems_per_load = 16 / sizeof(T);
    constexpr int total_elems    = ROWS * COLS;
    const int tid = threadIdx.x;
    const int gr_base = idx.r * ROWS;
    const int gc_base = idx.c * COLS;
    const T* base = src.raw_ptr
                  + (((int64_t(idx.b) * src.depth() + idx.d) * src.rows() + gr_base)
                     * src.cols() + gc_base);

    #pragma unroll
    for (int i = tid * elems_per_load; i < total_elems;
         i += N_THREADS * elems_per_load)
    {
        const int row = i / COLS;
        const int col = i % COLS;
        const int lds_flat = detail::subtile_flat<ROWS, COLS, 16, 32>(i);

        // The gfx1250 async-to-LDS builtins want address-space-qualified
        // pointers (AS(1) global, AS(3) LDS). `reinterpret_cast` cannot add
        // an address space, so route through `uintptr_t` + a C-style cast,
        // matching the pattern used elsewhere in this file for AS(3).
        uintptr_t g_uint = reinterpret_cast<uintptr_t>(base + row * row_stride + col);
        uintptr_t l_uint = reinterpret_cast<uintptr_t>(lds_dst + Pad::padded(lds_flat));
        auto* g_ptr = (detail::i32x4_gvec*)(g_uint);
        auto* l_ptr = (detail::i32x4_lvec*)(l_uint);

        if (cluster_mask) {
            __builtin_amdgcn_cluster_load_async_to_lds_b128(
                g_ptr, l_ptr, 0, 0, static_cast<int>(cluster_mask));
        } else {
            __builtin_amdgcn_global_load_async_to_lds_b128(g_ptr, l_ptr, 0, 0);
        }
    }
}

/**
 * @brief Hardware tile DMA (TDM) global -> LDS load on gfx1250.
 *
 * Issues a single `tensor_load_to_lds` instruction whose D# descriptor
 * encodes the 2D tile shape, source tensor extents, row stride, and optional
 * LDS padding. The instruction is issued by every lane of the wave (`EXEC`
 * is ignored) but only `wave 0` of the group should call this -- TDM unit
 * arbitration is per-SIMD-pair, not per-lane.
 *
 * Drain with `kittens::sync::wait_tdm()`.
 *
 * @tparam Pad      LDS padding descriptor.
 * @tparam ROWS,COLS  Tile shape (elements).
 * @param  lds_dst     16B-aligned LDS pointer.
 * @param  src         Global tile descriptor.
 * @param  idx         Tile coordinate.
 * @param  row_stride  Source row stride (elements).
 * @param  cluster_mask Optional `workgroup_mask` (0 for single-WG, non-zero
 *                     to switch the load into `CLUSTER_LOAD_ASYNC` micro-ops).
 */
namespace detail {

using v4u32 = unsigned int __attribute__((ext_vector_type(4)));
using v8u32 = unsigned int __attribute__((ext_vector_type(8)));

/**
 * @brief Build the 12-DWord TDM D# (groups 0 + 1) for a 2D tile transfer.
 *
 * Encapsulates the bit-packing shared by `load_tdm` and `load_tdm_arrive`.
 * `bar_lds_addr` is the LDS byte address of a `barrier_lds` cell when the
 * caller wants the TDM unit to auto-arrive at completion (sets the
 * `atomic_barrier_enable` bit and stuffs the address into group 1). Pass 0
 * for the no-barrier path.
 */
template<typename Pad, int ROWS, int COLS, typename T>
__device__ __forceinline__ void build_tdm_d_2d(
    v4u32& g0, v8u32& g1,
    const T* base, T* lds_dst,
    int tensor_rows, int tensor_cols, int row_stride,
    uint32_t cluster_mask, uint32_t bar_lds_addr)
{
    // ---- Group 0: count, lds_addr, global_addr, type ----
    const uint32_t lds_addr = static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(lds_dst));
    const uint64_t gaddr    = reinterpret_cast<uint64_t>(base);

    g0[0] = 1u;                                                  // count
    g0[1] = lds_addr;
    g0[2] = static_cast<uint32_t>(gaddr);
    g0[3] = (static_cast<uint32_t>(gaddr >> 32) & 0x01FFFFFFu) | (2u << 30);

    // ---- Group 1: data_size, padding, dims, stride, optional barrier ----
    // data_size encoded as log2(bytes_per_element).
    constexpr uint32_t data_size_enc = (sizeof(T) == 1) ? 0
                                     : (sizeof(T) == 2) ? 1
                                     : (sizeof(T) == 4) ? 2
                                     : 3;
    constexpr uint32_t pad_enable   = (Pad::interval > 0) ? 1u : 0u;
    constexpr uint32_t pad_int_enc  = (Pad::interval > 0)
        ? ( __builtin_ctz(Pad::interval * sizeof(T) / 4) ) : 0;
    constexpr uint32_t pad_amt_enc  = (Pad::amount > 0)
        ? ( (Pad::amount * sizeof(T) / 4) - 1 ) : 0;

    // atomic_barrier_enable lives at bit 18 of group 1 word 0
    // (per the MI400 TDM D# layout: w0 = multicast_mask[15:0],
    // data_size[17:16], atomic_barrier_enable[18], iterate_enable[19],
    // pad_enable[20], pad_interval[24:22], pad_amount[31:25]).
    const uint32_t atomic_bar_enable = (bar_lds_addr != 0) ? (1u << 18) : 0u;

    uint32_t w0 = (data_size_enc << 16)
                | (pad_enable    << 20)
                |  atomic_bar_enable
                | (pad_int_enc   << 22)
                | (pad_amt_enc   << 25)
                | (cluster_mask  & 0xFFFFu);

    const uint32_t tdim0    = static_cast<uint32_t>(tensor_cols);
    const uint32_t tdim1    = static_cast<uint32_t>(tensor_rows);
    const uint32_t tiledim0 = static_cast<uint32_t>(COLS);
    const uint32_t tiledim1 = static_cast<uint32_t>(ROWS);

    // barrier_addr occupies w1[15:0]; tensor_dim0 lo16 occupies w1[31:16].
    uint32_t w1 = (bar_lds_addr & 0xFFFFu) | (tdim0 << 16);
    uint32_t w2 = (tdim0 >> 16) | (tdim1 << 16);
    uint32_t w3 = (tdim1 >> 16) | (tiledim0 << 16);
    uint32_t w4 = tiledim1;

    const uint64_t stride0 = static_cast<uint64_t>(
        static_cast<uint32_t>(row_stride * sizeof(T)));
    uint32_t w5 = static_cast<uint32_t>(stride0);
    uint32_t w6 = static_cast<uint32_t>(stride0 >> 32);
    uint32_t w7 = 0;

    g1[0] = w0; g1[1] = w1; g1[2] = w2; g1[3] = w3;
    g1[4] = w4; g1[5] = w5; g1[6] = w6; g1[7] = w7;
}

} // namespace detail (g2s)

template<typename Pad = lds_nopad, int ROWS = 0, int COLS = 0,
         typename T, ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
__device__ inline void load_tdm(T* __restrict__ lds_dst, const GL& src, const COORD& idx,
                                int tensor_rows, int tensor_cols, int row_stride,
                                uint32_t cluster_mask = 0)
{
    static_assert(ROWS > 0 && COLS > 0, "ROWS and COLS must be specified");

    const int gr_base = idx.r * ROWS;
    const int gc_base = idx.c * COLS;
    const T* base = src.raw_ptr
                  + (((int64_t(idx.b) * src.depth() + idx.d) * src.rows() + gr_base)
                     * src.cols() + gc_base);

    detail::v4u32 g0;
    detail::v8u32 g1;
    detail::build_tdm_d_2d<Pad, ROWS, COLS, T>(
        g0, g1, base, lds_dst, tensor_rows, tensor_cols, row_stride,
        cluster_mask, /*bar_lds_addr=*/ 0);

    detail::v4u32 g2 = {0, 0, 0, 0};
    detail::v4u32 g3 = {0, 0, 0, 0};
    __builtin_amdgcn_tensor_load_to_lds(g0, g1, g2, g3, 0);
}

/**
 * @brief TDM load that auto-arrives at an LDS barrier on completion.
 * @experimental
 *
 * Sets `atomic_barrier_enable` in the D# so the TDM unit emits a
 * `DS_ATOMIC_ASYNC_BARRIER_ARRIVE_B64` on `bar` after the transfer retires.
 * The consumer waits on `bar`'s phase flip via
 * `kittens::sync::wait_barrier(bar, phase)` instead of draining the global
 * `tensorcnt`, leaving unrelated TDM transfers in flight.
 *
 * The barrier must be primed via `kittens::sync::init_barrier(bar, count)`
 * before the first call referencing it. `count` is the number of
 * `load_tdm_arrive` invocations that target this barrier per phase.
 *
 * @note The D# bit positions for `atomic_barrier_enable` (`w0` bit 18) and
 * `atomic_barrier_address` (`w1[15:0]`) match the field table documented
 * in the Triton AMD backend (third_party/amd/lib/TritonAMDGPUToLLVM/
 * TDMUtility.cpp lines 224-264). The Triton lowering itself does not use
 * the D# auto-arrive path -- it follows `load_tdm` with an explicit
 * `wait_tdm()` + `async_barrier_arrive()` sequence (see
 * `gemm_tdm_arrive.cpp` for that pattern). This overload is provided for
 * runtimes that model TDM auto-arrive natively; on simulators that don't,
 * use the explicit-arrive pattern instead.
 *
 * @param bar  Pointer to a 64-bit LDS barrier counter (a `sync::barrier_lds`
 *             cell). Must point at LDS storage; must be 8-byte aligned.
 */
template<typename Pad = lds_nopad, int ROWS = 0, int COLS = 0,
         typename T, ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
__device__ inline void load_tdm_arrive(
    T* __restrict__ lds_dst, const GL& src, const COORD& idx,
    int tensor_rows, int tensor_cols, int row_stride,
    uint64_t* bar, uint32_t cluster_mask = 0)
{
    static_assert(ROWS > 0 && COLS > 0, "ROWS and COLS must be specified");

    const int gr_base = idx.r * ROWS;
    const int gc_base = idx.c * COLS;
    const T* base = src.raw_ptr
                  + (((int64_t(idx.b) * src.depth() + idx.d) * src.rows() + gr_base)
                     * src.cols() + gc_base);

    const uint32_t bar_lds_addr = static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(bar));

    detail::v4u32 g0;
    detail::v8u32 g1;
    detail::build_tdm_d_2d<Pad, ROWS, COLS, T>(
        g0, g1, base, lds_dst, tensor_rows, tensor_cols, row_stride,
        cluster_mask, bar_lds_addr);

    detail::v4u32 g2 = {0, 0, 0, 0};
    detail::v4u32 g3 = {0, 0, 0, 0};
    __builtin_amdgcn_tensor_load_to_lds(g0, g1, g2, g3, 0);
}

/**
 * @brief Cooperative L2 prefetch for an upcoming tile.
 *
 * Lowers to `__builtin_amdgcn_global_prefetch` issued from every participating
 * lane. The hint = 0 selects the default cache policy.
 */
template<int ROWS = 0, int COLS = 0, int N_THREADS = WARP_THREADS,
         typename T, ducks::gl::all GL, ducks::coord::tile COORD = coord<>>
__device__ inline void prefetch_l2(const GL& src, const COORD& idx, int row_stride)
{
    static_assert(ROWS > 0 && COLS > 0, "ROWS and COLS must be specified");
    constexpr int elems_per_pf = 16 / sizeof(T);
    constexpr int total_elems  = ROWS * COLS;
    const int tid = threadIdx.x;
    const int gr_base = idx.r * ROWS;
    const int gc_base = idx.c * COLS;
    const T* base = src.raw_ptr
                  + (((int64_t(idx.b) * src.depth() + idx.d) * src.rows() + gr_base)
                     * src.cols() + gc_base);

    #pragma unroll
    for (int i = tid * elems_per_pf; i < total_elems;
         i += N_THREADS * elems_per_pf)
    {
        const int row = i / COLS;
        const int col = i % COLS;
        const T* addr = base + row * row_stride + col;
        __builtin_amdgcn_global_prefetch(
            (const void __attribute__((address_space(1)))*)addr, 0);
    }
}

} // namespace g2s
#endif // KITTENS_UDNA1
}
