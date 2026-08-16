// tcapi/queries.h
// Mirrors tcapi_numpy/queries.py.
// Sec. C2a — metadata queries: order, shape, size, size_bytes, get_elem.
#pragma once

#include "itensor/all.h"
#include <cstddef>

#include "tcapi/type_system.h"
#include "tcapi/detail.h"

namespace tcapi {

// --- order ---------------------------------------------------------------
// Sec. C2a — number of bonds (order) of a tensor.
template<typename TenT>
order_t<TenT> order(const context_handle_t<TenT>& ctx, const tent_t<TenT>& a)
{
    detail::ensure_active<TenT>(ctx);
    return itensor::order(a);
}

// --- shape ---------------------------------------------------------------
// Sec. C2a — per-bond dimensions, in bond order.
template<typename TenT>
shape_t<TenT> shape(const context_handle_t<TenT>& ctx, const tent_t<TenT>& a)
{
    detail::ensure_active<TenT>(ctx);
    shape_t<TenT> dims;
    for(auto const& I : itensor::inds(a)) dims.push_back(itensor::dim(I));
    return dims;
}

// --- size ----------------------------------------------------------------
// Sec. C2a — total number of elements (product of all bond dimensions).
template<typename TenT>
ten_size_t<TenT> size(const context_handle_t<TenT>& ctx, const tent_t<TenT>& a)
{
    detail::ensure_active<TenT>(ctx);
    ten_size_t<TenT> n = 1;
    for(auto const& I : itensor::inds(a)) n *= itensor::dim(I);
    return n;
}

// --- size_bytes ----------------------------------------------------------
// Sec. C2a — storage size in bytes of the (dense) element array.
template<typename TenT>
std::size_t size_bytes(const context_handle_t<TenT>& ctx, const tent_t<TenT>& a)
{
    detail::ensure_active<TenT>(ctx);
    return static_cast<std::size_t>(size<TenT>(ctx, a)) * sizeof(elem_t<TenT>);
}

// --- get_elem ------------------------------------------------------------
// Sec. C2a — read the element at the given (0-based) coordinates.
template<typename TenT>
elem_t<TenT> get_elem(const context_handle_t<TenT>& ctx,
                      const tent_t<TenT>& a,
                      const elem_coors_t<TenT>& coors)
{
    detail::ensure_active<TenT>(ctx);
    auto is = itensor::inds(a);
    auto ivs = detail::to_ivs<TenT>(is, coors);
    return detail::get_elem_impl<TenT>(a, ivs);
}

} // namespace tcapi
