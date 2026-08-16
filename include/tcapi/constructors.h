// tcapi/constructors.h
// Mirrors tcapi_numpy/constructors.py.
// Sec. C2b — tensor construction and destruction: allocate, zeros, fill,
// random, eye, copy, move, clear, assign_from_range.
#pragma once

#include "itensor/all.h"
#include <utility>

#include "tcapi/type_system.h"
#include "tcapi/detail.h"
#include "tcapi/queries.h"

namespace tcapi {

// --- allocate -----------------------------------------------------------
// Sec. C2b — allocate a zero-filled tensor of the given shape.
// Storage is eagerly zero-filled to avoid null-store reads (ITensor lazily
// allocates on set(), not on construction).
template<typename TenT>
tent_t<TenT> allocate(const context_handle_t<TenT>& ctx, const shape_t<TenT>& shape)
{
    detail::ensure_active<TenT>(ctx);
    auto is = detail::make_indices<TenT>(shape);
    tent_t<TenT> A(is);

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

// --- zeros --------------------------------------------------------------
// Sec. C2b — same as allocate; provided as a spec-named convenience.
template<typename TenT>
tent_t<TenT> zeros(const context_handle_t<TenT>& ctx, const shape_t<TenT>& shape)
{
    return allocate<TenT>(ctx, shape);
}

// --- fill ---------------------------------------------------------------
// Sec. C2b — allocate and fill every element with the scalar v.
template<typename TenT>
tent_t<TenT> fill(const context_handle_t<TenT>& ctx,
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

// --- random -------------------------------------------------------------
// Sec. C2b — fill a tensor of the given shape by drawing each element from
// the supplied generator (a callable producing elem_t<TenT> values).
// Per the spec, gen is passed by non-const lvalue reference (RandNumGen&),
// so generators that mutate internal state work as specified.
template<typename TenT, typename RandNumGen>
tent_t<TenT> random(const context_handle_t<TenT>& ctx,
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

// --- eye ----------------------------------------------------------------
// Sec. C2b — N x N identity tensor (second-order).
template<typename TenT>
tent_t<TenT> eye(const context_handle_t<TenT>& ctx, bond_dim_t<TenT> n)
{
    detail::ensure_active<TenT>(ctx);
    itensor::Index i(static_cast<int>(n), "b0");
    tent_t<TenT> A(i, itensor::prime(i));
    for(int k = 1; k <= n; ++k)
        A.set(i(k), itensor::prime(i)(k), elem_t<TenT>{1});
    return A;
}

// --- copy ---------------------------------------------------------------
// Sec. C2b — deep copy of a tensor.
template<typename TenT>
tent_t<TenT> copy(const context_handle_t<TenT>& ctx, const tent_t<TenT>& orig)
{
    detail::ensure_active<TenT>(ctx);
    return tent_t<TenT>(orig);
}

// --- move ---------------------------------------------------------------
// Sec. C2b — C++-only ownership transfer: moves storage out of `from`.
template<typename TenT>
tent_t<TenT> move(const context_handle_t<TenT>& ctx, tent_t<TenT>& from)
{
    detail::ensure_active<TenT>(ctx);
    tent_t<TenT> moved(std::move(from));
    from = tent_t<TenT>{};
    return moved;
}

// --- clear --------------------------------------------------------------
// Sec. C2b — C++-only ownership utility: releases the tensor's storage.
template<typename TenT>
void clear(const context_handle_t<TenT>& ctx, tent_t<TenT>& a)
{
    detail::ensure_active<TenT>(ctx);
    a = tent_t<TenT>{};
}

// --- assign_from_range --------------------------------------------------
// Sec. C2b — build a tensor of the given shape by pulling element values
// from the range [first, first + size) using coors2idx to map each
// coordinate tuple to a flat index.
template<typename TenT, typename RandomIt, typename Func>
tent_t<TenT> assign_from_range(const context_handle_t<TenT>& ctx,
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
