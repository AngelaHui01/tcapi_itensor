#pragma once

#include "itensor/all.h"
#include <cstddef>

#include "tcapi/type_system.h"
#include "tcapi/detail.h"

namespace tcapi {

template<typename TenT>
order_t<TenT> order(const context_handle_t<TenT>& ctx, const ten_t<TenT>& a)
{
    detail::ensure_active<TenT>(ctx);
    return itensor::order(a);
}

template<typename TenT>
shape_t<TenT> shape(const context_handle_t<TenT>& ctx, const ten_t<TenT>& a)
{
    detail::ensure_active<TenT>(ctx);
    shape_t<TenT> dims;
    for(auto const& I : itensor::inds(a)) dims.push_back(itensor::dim(I));
    return dims;
}

template<typename TenT>
ten_size_t<TenT> size(const context_handle_t<TenT>& ctx, const ten_t<TenT>& a)
{
    detail::ensure_active<TenT>(ctx);
    ten_size_t<TenT> n = 1;
    for(auto const& I : itensor::inds(a)) n *= itensor::dim(I);
    return n;
}

template<typename TenT>
std::size_t size_bytes(const context_handle_t<TenT>& ctx, const ten_t<TenT>& a)
{
    detail::ensure_active<TenT>(ctx);
    return static_cast<std::size_t>(size<TenT>(ctx, a)) * sizeof(elem_t<TenT>);
}

template<typename TenT>
elem_t<TenT> get_elem(const context_handle_t<TenT>& ctx,
                      const ten_t<TenT>& a,
                      const elem_coors_t<TenT>& coors)
{
    detail::ensure_active<TenT>(ctx);
    auto is = itensor::inds(a);
    auto ivs = detail::to_ivs<TenT>(is, coors);
    return detail::get_elem_impl<TenT>(a, ivs);
}

} // namespace tcapi