/**
 * @file
 * @brief Group (collaborative warp) ops for loading shared tiles from and storing to global memory. 
 */

template<int axis, bool assume_aligned, typename ST, kittens::ducks::gl::all GL, kittens::ducks::coord::tile COORD=coord<ST>,
         std::enable_if_t<kittens::detail::is_st<std::remove_cvref_t<ST>>::value, int> = 0>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx) {
    kittens::load<axis, assume_aligned, ST, GL, COORD, GROUP_THREADS>(dst, src, idx);
}
template<typename ST, kittens::ducks::gl::all GL, kittens::ducks::coord::tile COORD=coord<ST>,
         std::enable_if_t<kittens::detail::is_st<std::remove_cvref_t<ST>>::value, int> = 0> // default case
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx) {
    kittens::load<2, false, ST, GL, COORD, GROUP_THREADS>(dst, src, idx);
}
template<int axis, bool assume_aligned, typename ST, kittens::ducks::gl::all GL,
         std::enable_if_t<kittens::detail::is_st<std::remove_cvref_t<ST>>::value, int> = 0>
__device__ static inline void prefill_swizzled_offsets(ST &, const GL &, uint32_t *) {
}
template<typename ST, kittens::ducks::gl::all GL,
         std::enable_if_t<kittens::detail::is_st<std::remove_cvref_t<ST>>::value, int> = 0>
__device__ static inline void prefill_swizzled_offsets(ST &, const GL &, uint32_t *) {
}
template<int axis, bool assume_aligned, typename ST, kittens::ducks::gl::all GL, kittens::ducks::coord::tile COORD=coord<ST>,
         std::enable_if_t<kittens::detail::is_st<std::remove_cvref_t<ST>>::value, int> = 0>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx, const uint32_t *) {
    kittens::load<axis, assume_aligned, ST, GL, COORD, GROUP_THREADS>(dst, src, idx);
}
template<typename ST, kittens::ducks::gl::all GL, kittens::ducks::coord::tile COORD=coord<ST>,
         std::enable_if_t<kittens::detail::is_st<std::remove_cvref_t<ST>>::value, int> = 0>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx, const uint32_t *) {
    kittens::load<2, false, ST, GL, COORD, GROUP_THREADS>(dst, src, idx);
}
template<int axis, bool assume_aligned, typename ST, kittens::ducks::gl::all GL, kittens::ducks::coord::tile COORD=coord<ST>,
         std::enable_if_t<kittens::detail::is_st<std::remove_cvref_t<ST>>::value, int> = 0>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx, const uint32_t *, i32x4, const void*, uint32_t) {
    kittens::load<axis, assume_aligned, ST, GL, COORD, GROUP_THREADS>(dst, src, idx);
}
template<typename ST, kittens::ducks::gl::all GL, kittens::ducks::coord::tile COORD=coord<ST>,
         std::enable_if_t<kittens::detail::is_st<std::remove_cvref_t<ST>>::value, int> = 0>
__device__ static inline void load(ST &dst, const GL &src, const COORD &idx, const uint32_t *, i32x4, const void*, uint32_t) {
    kittens::load<2, false, ST, GL, COORD, GROUP_THREADS>(dst, src, idx);
}
template<int axis, bool assume_aligned, typename ST, kittens::ducks::gl::all GL, kittens::ducks::coord::tile COORD=coord<ST>,
         std::enable_if_t<kittens::detail::is_st<std::remove_cvref_t<ST>>::value, int> = 0>
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    kittens::store<axis, assume_aligned, ST, GL, COORD, GROUP_THREADS>(dst, src, idx);
}
template<typename ST, kittens::ducks::gl::all GL, kittens::ducks::coord::tile COORD=coord<ST>,
         std::enable_if_t<kittens::detail::is_st<std::remove_cvref_t<ST>>::value, int> = 0> // default case
__device__ static inline void store(const GL &dst, const ST &src, const COORD &idx) {
    kittens::store<2, false, ST, GL, COORD, GROUP_THREADS>(dst, src, idx);
}
