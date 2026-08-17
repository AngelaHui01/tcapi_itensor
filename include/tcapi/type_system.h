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

template<typename T>
using List = std::vector<T>;

template<typename T, typename U>
using Pair = std::pair<T, U>;

template<typename T, typename U>
using Map = std::unordered_map<T, U>;

template<typename T>
using CRef = std::reference_wrapper<const T>;

struct ItensorContext
{
    bool active = false;
    bool is_active() const noexcept { return active; }
};

template<typename TenT>
struct tensor_traits;

struct ItensorReal;
struct ItensorCplx;

template<>
struct tensor_traits<ItensorReal>
{
    using ten_t            = itensor::ITensor;
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

template<>
struct tensor_traits<ItensorCplx>
{
    using ten_t            = itensor::ITensor;
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

template<typename TenT> using ten_t           = typename tensor_traits<TenT>::ten_t;
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
using bond_idx_elem_coor_pair_map =
    Map<bond_idx_t<TenT>, Pair<elem_coor_t<TenT>, elem_coor_t<TenT>>>;

} // namespace detail

} // namespace tcapi