// tcapi/detail.h
// Internal helpers shared across the TCAPI modules: context validation,
// index construction, coordinate iteration, and IndexVal dispatch.
// These are NOT part of the public TCAPI surface.
#pragma once

#include "itensor/all.h"
#include <stdexcept>
#include <vector>
#include <string>
#include <utility>

#include "tcapi/type_system.h"

namespace tcapi {
namespace detail {

/// Validates that the context is active before any public operation.
template<typename TenT>
inline void ensure_active(const context_handle_t<TenT>& ctx)
{
    if(!ctx.is_active())
        throw std::runtime_error(
            "TCAPI: operation called on a destroyed context. Call create_context first."
        );
}

inline itensor::IndexSet make_indices_from_dims(const std::vector<long>& dims,
                                                const std::string& base_tag = "b")
{
    std::vector<itensor::Index> inds;
    inds.reserve(dims.size());
    for(std::size_t b = 0; b < dims.size(); ++b)
        inds.emplace_back(static_cast<int>(dims[b]), base_tag + std::to_string(b));
    return itensor::IndexSet(inds);
}

template<typename TenT>
itensor::IndexSet make_indices(const shape_t<TenT>& shape,
                               const std::string& base_tag = "b")
{
    std::vector<long> dims(shape.begin(), shape.end());
    return make_indices_from_dims(dims, base_tag);
}

/// Recursively visit every zero-based coordinate tuple for a given shape.
template<typename TenT, typename Func>
void for_each_coordinate(const shape_t<TenT>& shape, Func&& f)
{
    elem_coors_t<TenT> coors(shape.size(), 0);
    std::function<void(std::size_t)> rec = [&](std::size_t axis)
    {
        if(axis == shape.size()) { f(coors); return; }
        for(elem_coor_t<TenT> i = 0; i < static_cast<elem_coor_t<TenT>>(shape[axis]); ++i)
        {
            coors[axis] = i;
            rec(axis + 1);
        }
    };
    rec(0);
}

template<typename TenT>
std::vector<itensor::IndexVal> to_ivs(const itensor::IndexSet& is,
                                      const elem_coors_t<TenT>& coors)
{
    std::vector<itensor::IndexVal> ivs;
    ivs.reserve(is.size());
    for(std::size_t b = 0; b < is.size(); ++b)
        ivs.emplace_back(is[b], coors[b] + 1); // ITensor is 1-based
    return ivs;
}

// -----------------------------------------------------------------------------
// Element get/set via ITensor's vector-of-IndexVal accessors
// (eltC/set overloads taking a std::vector<IndexVal>). Unlike the variadic
// elt/eltC/set forms, these accept any tensor order (including rank 0), so no
// runtime arity dispatch table is needed.
// -----------------------------------------------------------------------------

template<typename TenT>
inline elem_t<TenT>
get_elem_impl(const tent_t<TenT>& a, const std::vector<itensor::IndexVal>& ivs)
{
    if constexpr (std::is_same_v<elem_t<TenT>, std::complex<double>>)
        return a.eltC(ivs);
    else
        return a.eltC(ivs).real();
}

template<typename TenT>
inline void
set_elem_impl(tent_t<TenT>& a, const std::vector<itensor::IndexVal>& ivs,
              itensor::Cplx val)
{
    a.set(ivs, val);
}

} // namespace detail
} // namespace tcapi
