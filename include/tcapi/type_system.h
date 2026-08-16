// tcapi/type_system.h
// Mirrors tcapi_numpy/ten_kind.py + tcapi_numpy/context.py.
// Defines the compile-time TCAPI type system (Table I, Appendix C.1):
// the concrete TenT tensor families backed by ITensor, the tensor_traits
// specializations that map each family to its element/index types, and the
// backend execution context object.
#pragma once

#include "itensor/all.h"
#include <complex>
#include <vector>
#include <string>
#include <utility>
#include <unordered_map>
#include <functional>
#include <cstddef>

namespace tcapi {

// -----------------------------------------------------------------------------
// Auxiliary container aliases (Table II, Appendix C.1)
// -----------------------------------------------------------------------------

template<typename T>
using List = std::vector<T>;

template<typename T, typename U>
using Pair = std::pair<T, U>;

template<typename T, typename U>
using Map = std::unordered_map<T, U>;

// Borrowed-reference wrapper used where a list of tensors is passed by
// reference without copying (concatenate, stack, linear_combine).
template<typename T>
using CRef = std::reference_wrapper<const T>;

// -----------------------------------------------------------------------------
// Backend execution context
// -----------------------------------------------------------------------------

/// Backend execution context for the ITensor TCF.
/// Concrete type bound to tensor_traits<TenT>::context_handle_t.
struct ItensorContext
{
    bool active = false;
    bool is_active() const noexcept { return active; }
};

// -----------------------------------------------------------------------------
// tensor_traits<TenT> — compile-time type system (Table I)
// -----------------------------------------------------------------------------

/// Primary template — must be specialized per concrete tensor type TenT.
template<typename TenT>
struct tensor_traits;

struct ItensorReal;
struct ItensorCplx;

/// tensor_traits specialization for the real-valued ITensor family (64-bit).
template<>
struct tensor_traits<ItensorReal>
{
    using tent             = itensor::ITensor;
    using order_t          = int;
    using bond_dim_t       = long;
    using bond_idx_t       = int;
    using bond_label_t     = int;
    using shape_t          = List<bond_dim_t>;
    using ten_size_t       = std::size_t;
    using elem_t           = double;
    using elem_coor_t      = long;
    using elem_coors_t     = List<elem_coor_t>;
    using real_t           = double;
    using real_ten_t       = itensor::ITensor;
    using cplx_t           = std::complex<double>;
    using cplx_ten_t       = itensor::ITensor;
    using context_handle_t = ItensorContext;
};

/// tensor_traits specialization for the complex-valued ITensor family (64-bit).
template<>
struct tensor_traits<ItensorCplx>
{
    using tent             = itensor::ITensor;
    using order_t          = int;
    using bond_dim_t       = long;
    using bond_idx_t       = int;
    using bond_label_t     = int;
    using shape_t          = List<bond_dim_t>;
    using ten_size_t       = std::size_t;
    using elem_t           = std::complex<double>;
    using elem_coor_t      = long;
    using elem_coors_t     = List<elem_coor_t>;
    using real_t           = double;
    using real_ten_t       = itensor::ITensor;
    using cplx_t           = std::complex<double>;
    using cplx_ten_t       = itensor::ITensor;
    using context_handle_t = ItensorContext;
};

// --- Alias templates -------------------------------------------------

template<typename TenT> using tent_t           = typename tensor_traits<TenT>::tent;
template<typename TenT> using order_t          = typename tensor_traits<TenT>::order_t;
template<typename TenT> using shape_t          = typename tensor_traits<TenT>::shape_t;
template<typename TenT> using bond_dim_t       = typename tensor_traits<TenT>::bond_dim_t;
template<typename TenT> using bond_idx_t       = typename tensor_traits<TenT>::bond_idx_t;
template<typename TenT> using bond_label_t     = typename tensor_traits<TenT>::bond_label_t;
template<typename TenT> using ten_size_t       = typename tensor_traits<TenT>::ten_size_t;
template<typename TenT> using elem_t           = typename tensor_traits<TenT>::elem_t;
template<typename TenT> using elem_coor_t      = typename tensor_traits<TenT>::elem_coor_t;
template<typename TenT> using elem_coors_t     = typename tensor_traits<TenT>::elem_coors_t;
template<typename TenT> using real_t           = typename tensor_traits<TenT>::real_t;
template<typename TenT> using real_ten_t       = typename tensor_traits<TenT>::real_ten_t;
template<typename TenT> using cplx_t           = typename tensor_traits<TenT>::cplx_t;
template<typename TenT> using cplx_ten_t       = typename tensor_traits<TenT>::cplx_ten_t;
template<typename TenT> using context_handle_t = typename tensor_traits<TenT>::context_handle_t;

namespace detail {

template<typename TenT>
using bond_idx_pairs_t = List<Pair<bond_idx_t<TenT>, bond_idx_t<TenT>>>;

template<typename TenT>
using bond_idx_elem_coor_pair_map_t =
    Map<bond_idx_t<TenT>, Pair<elem_coor_t<TenT>, elem_coor_t<TenT>>>;

} // namespace detail

} // namespace tcapi
