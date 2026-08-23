#pragma once

#include "itensor/all.h"
#include <utility>

#include "tcapi/type_system.h"
#include "tcapi/detail.h"
#include "tcapi/queries.h"

namespace tcapi {

template<typename TenT>
ten_t<TenT> allocate(context_handle_t<TenT>& ctx, const shape_t<TenT>& shape)
{
    detail::ensure_active<TenT>(ctx);
    auto is = detail::make_indices<TenT>(shape);
    ten_t<TenT> A(is);

    if(is.size() > 0)
    {
        std::vector<itensor::IndexVal> ivs;
        ivs.reserve(is.size());
        for(std::size_t b = 0; b < is.size(); ++b)
            ivs.emplace_back(is[b], 1);

        detail::set_elem_impl<TenT>(A, ivs, elem_t<TenT>{0});
    }
    return A;
}

template<typename TenT>
ten_t<TenT> zeros(context_handle_t<TenT>& ctx, const shape_t<TenT>& shape)
{
    detail::ensure_active<TenT>(ctx);
    auto A = allocate<TenT>(ctx, shape);
    auto is = itensor::inds(A);
    detail::for_each_coordinate<TenT>(shape, [&](const elem_coors_t<TenT>& coors)
    {
        detail::set_elem_impl<TenT>(A, detail::to_ivs<TenT>(is, coors), elem_t<TenT>{0});
    });
    return A;
}

template<typename TenT>
ten_t<TenT> fill(context_handle_t<TenT>& ctx,
                  const shape_t<TenT>& shape,
                  elem_t<TenT> v)
{
    detail::ensure_active<TenT>(ctx);
    auto A = allocate<TenT>(ctx, shape);
    auto is = itensor::inds(A);
    detail::for_each_coordinate<TenT>(shape, [&](const elem_coors_t<TenT>& coors)
    {
        detail::set_elem_impl<TenT>(A, detail::to_ivs<TenT>(is, coors), v);
    });
    return A;
}

template<typename TenT, typename RandNumGen>
ten_t<TenT> random(context_handle_t<TenT>& ctx,
                    const shape_t<TenT>& shape,
                    RandNumGen& gen)
{
    detail::ensure_active<TenT>(ctx);
    auto A = allocate<TenT>(ctx, shape);
    auto is = itensor::inds(A);
    detail::for_each_coordinate<TenT>(shape, [&](const elem_coors_t<TenT>& coors)
    {
        elem_t<TenT> v = static_cast<elem_t<TenT>>(gen());
        detail::set_elem_impl<TenT>(A, detail::to_ivs<TenT>(is, coors), v);
    });
    return A;
}

template<typename TenT>
ten_t<TenT> eye(context_handle_t<TenT>& ctx, bond_dim_t<TenT> n)
{
    detail::ensure_active<TenT>(ctx);
    itensor::Index i(static_cast<int>(n), "b0");
    ten_t<TenT> A(i, itensor::prime(i));
    for(int k = 1; k <= n; ++k)
        A.set(i(k), itensor::prime(i)(k), elem_t<TenT>{1});
    return A;
}

template<typename TenT>
ten_t<TenT> copy(context_handle_t<TenT>& ctx, const ten_t<TenT>& orig)
{
    detail::ensure_active<TenT>(ctx);
    return orig * real_t<TenT>{1};
}

template<typename TenT>
ten_t<TenT> move(context_handle_t<TenT>& ctx, ten_t<TenT>& from)
{
    detail::ensure_active<TenT>(ctx);
    ten_t<TenT> moved(std::move(from));
    from = ten_t<TenT>{};
    return moved;
}

template<typename TenT>
void clear(context_handle_t<TenT>& ctx, ten_t<TenT>& a)
{
    detail::ensure_active<TenT>(ctx);
    a = ten_t<TenT>{};
}

template<typename TenT, typename RandomIt, typename Func>
ten_t<TenT> assign_from_range(context_handle_t<TenT>& ctx,
                               const shape_t<TenT>& shape,
                               RandomIt first,
                               Func coors2idx)
{
    detail::ensure_active<TenT>(ctx);
    auto A = allocate<TenT>(ctx, shape);
    auto is = itensor::inds(A);
    detail::for_each_coordinate<TenT>(shape, [&](const elem_coors_t<TenT>& coors)
    {
        auto idx = coors2idx(coors);
        elem_t<TenT> v = *(first + idx);
        detail::set_elem_impl<TenT>(A, detail::to_ivs<TenT>(is, coors), v);
    });
    return A;
}

} // namespace tcapi