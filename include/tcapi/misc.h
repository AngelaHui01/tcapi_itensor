// tcapi/misc.h
// Mirrors tcapi_numpy/misc.py.
// Sec. C2f — context lifecycle, version, show, close, convert, to_range.
#pragma once

#include "itensor/all.h"
#include <iostream>
#include <string>
#include <type_traits>

#include "tcapi/type_system.h"
#include "tcapi/detail.h"
#include "tcapi/queries.h"
#include "tcapi/constructors.h"

namespace tcapi {

// --- create_context / destroy_context -----------------------------------
// The context handle is a plain (default-constructible) value object; these
// are the only operations that transition it between active states.
// The spec templatizes these on ContextHandleT (not TenT), so they deduce
// the concrete handle type directly from the argument.

template<typename ContextHandleT>
void create_context(ContextHandleT& ctx)
{
    ctx.active = true;
}

template<typename ContextHandleT>
void destroy_context(ContextHandleT& ctx)
{
    if(!ctx.active)
        throw std::runtime_error(
            "destroy_context called on an already-destroyed context.");
    ctx.active = false;
}

// --- version ------------------------------------------------------------
// Sec. C2f — backend version string. No ctx argument (mirrors numpy::version).

template<typename TenT>
std::string version()
{
    return "1.0";
}

// --- show ---------------------------------------------------------------
// Sec. C2f — print shape/dtype/order summary plus the tensor itself.

template<typename TenT>
void show(const context_handle_t<TenT>& ctx, const tent_t<TenT>& a)
{
    detail::ensure_active<TenT>(ctx);
    auto sh = shape<TenT>(ctx, a);
    const char* dtype = std::is_same<elem_t<TenT>, itensor::Real>::value ? "Real" : "Cplx";
    std::cout << "Tensor | order=" << order<TenT>(ctx, a) << " | shape=(";
    for(std::size_t i = 0; i < sh.size(); ++i)
        std::cout << sh[i] << (i + 1 < sh.size() ? "," : "");
    std::cout << ") | dtype=" << dtype << "\n" << a << "\n";
}

// --- close --------------------------------------------------------------
// Sec. C2f — elementwise max-abs-difference closeness. Shapes must match
// (returns false otherwise); epsilon must be >= 0. Uses get_elem so off-
// diagonal zeros of diagonal tensors are respected logically.

template<typename TenT>
bool close(const context_handle_t<TenT>& ctx,
           const tent_t<TenT>& a, const tent_t<TenT>& b,
           real_t<TenT> epsilon)
{
    detail::ensure_active<TenT>(ctx);
    if(epsilon < real_t<TenT>{0})
        throw std::invalid_argument("close: epsilon must be >= 0.");
    if(shape<TenT>(ctx, a) != shape<TenT>(ctx, b))
        return false;

    auto sh = shape<TenT>(ctx, a);
    if(sh.empty()) return true; // zeroth-order tensors have size 1
    ten_size_t<TenT> total = size<TenT>(ctx, a);
    if(total == 0) return true;

    real_t<TenT> maxdiff{0};
    bool first = true;
    cplx_t<TenT> v;
    (void)v;
    detail::for_each_coordinate<TenT>(sh, [&](const elem_coors_t<TenT>& coors)
    {
        auto da = static_cast<cplx_t<TenT>>(get_elem<TenT>(ctx, a, coors));
        auto db = static_cast<cplx_t<TenT>>(get_elem<TenT>(ctx, b, coors));
        auto diff = std::abs(da - db);
        if(first || diff > maxdiff) maxdiff = diff;
        first = false;
    });
    return maxdiff <= epsilon;
}

// --- convert ------------------------------------------------------------
// Sec. C2f — elementwise element-family conversion across contexts (the two
// contexts may belong to different backends). Same-kind is a deep copy; on
// success t2 is fully constructed. Complex -> real keeps the real part.

template<typename Ten1T, typename Ten2T>
void convert(const context_handle_t<Ten1T>& ctx1, const tent_t<Ten1T>& t1,
             const context_handle_t<Ten2T>& ctx2, tent_t<Ten2T>& t2)
{
    detail::ensure_active<Ten1T>(ctx1);
    detail::ensure_active<Ten2T>(ctx2);

    using E1 = elem_t<Ten1T>;
    using E2 = elem_t<Ten2T>;

    auto sh = shape<Ten1T>(ctx1, t1);
    t2 = allocate<Ten2T>(ctx2, sh);
    auto is2 = itensor::inds(t2);

    detail::for_each_coordinate<Ten1T>(sh, [&](const elem_coors_t<Ten1T>& coors)
    {
        E1 v1 = get_elem<Ten1T>(ctx1, t1, coors);
        E2 v2;
        if constexpr(std::is_same<E2, itensor::Real>::value)
        {
            if constexpr(std::is_same<E1, itensor::Real>::value)
                v2 = v1;
            else
                v2 = std::real(v1);
        }
        else
        {
            if constexpr(std::is_same<E1, itensor::Real>::value)
                v2 = E2(v1, 0.0);
            else
                v2 = E2(v1);
        }
        detail::set_elem_impl<Ten2T>(t2, detail::to_ivs<Ten2T>(is2, coors), v2);
    });
}

// --- to_range -----------------------------------------------------------
// Sec. C2f — write ALL logical elements (including off-diagonal zeros of
// diagonal tensors) into the range starting at first, ordered by coors2idx.

template<typename TenT, typename RandomIt, typename Func>
void to_range(const context_handle_t<TenT>& ctx,
              const tent_t<TenT>& a, RandomIt first, Func coors2idx)
{
    detail::ensure_active<TenT>(ctx);
    auto sh = shape<TenT>(ctx, a);
    ten_size_t<TenT> total = size<TenT>(ctx, a);

    detail::for_each_coordinate<TenT>(sh, [&](const elem_coors_t<TenT>& coors)
    {
        auto idx = coors2idx(coors);
        if(idx < 0 || static_cast<ten_size_t<TenT>>(idx) >= total)
            throw std::out_of_range(
                "to_range: coors2idx returned an index out of range.");
        *(first + idx) = get_elem<TenT>(ctx, a, coors);
    });
}

} // namespace tcapi