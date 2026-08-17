#pragma once

#include "itensor/all.h"
#include <algorithm>
#include <utility>
#include <functional>
#include <type_traits>

#include "tcapi/type_system.h"
#include "tcapi/detail.h"
#include "tcapi/queries.h"
#include "tcapi/constructors.h"

namespace tcapi {

template<typename TenT>
void set_elem(const context_handle_t<TenT>& ctx,
              ten_t<TenT>& a,
              const elem_coors_t<TenT>& coors,
              elem_t<TenT> el)
{
    detail::ensure_active<TenT>(ctx);
    auto is = itensor::inds(a);
    auto ivs = detail::to_ivs<TenT>(is, coors);
    detail::set_elem_impl<TenT>(a, ivs, el);
}

namespace detail {

template<typename TenT>
void reshape_impl(const context_handle_t<TenT>& ctx,
                  const ten_t<TenT>& in,
                  const shape_t<TenT>& new_shape,
                  ten_t<TenT>& out)
{
    detail::ensure_active<TenT>(ctx);
    auto old_shape = shape<TenT>(ctx, in);
    ten_size_t<TenT> n = size<TenT>(ctx, in);

    ten_size_t<TenT> new_n = 1;
    for(auto d : new_shape) new_n *= static_cast<ten_size_t<TenT>>(d);
    if(new_n != n)
        throw std::invalid_argument("reshape: total element count mismatch between old and new shape.");

    std::vector<elem_t<TenT>> flat(n);
    ten_size_t<TenT> flat_idx = 0;
    for_each_coordinate<TenT>(old_shape, [&](const elem_coors_t<TenT>& coors)
    {
        flat[flat_idx++] = tcapi::get_elem<TenT>(ctx, in, coors);
    });

    out = tcapi::allocate<TenT>(ctx, new_shape);
    auto is_out = itensor::inds(out);
    flat_idx = 0;
    for_each_coordinate<TenT>(new_shape, [&](const elem_coors_t<TenT>& coors)
    {
        elem_t<TenT> v = flat[flat_idx++];
        detail::set_elem_impl<TenT>(out, to_ivs<TenT>(is_out, coors), v);
    });
}

template<typename TenT>
void transpose_impl(const context_handle_t<TenT>& ctx,
                    const ten_t<TenT>& in,
                    const List<bond_idx_t<TenT>>& new_order,
                    ten_t<TenT>& out)
{
    detail::ensure_active<TenT>(ctx);
    auto is_in = itensor::inds(in);
    std::vector<itensor::Index> permuted;
    permuted.reserve(is_in.size());
    for(auto idx : new_order) permuted.push_back(is_in[idx]);

    out = itensor::ITensor(itensor::IndexSet(permuted));
    auto old_shape = tcapi::shape<TenT>(ctx, in);
    for_each_coordinate<TenT>(old_shape, [&](const elem_coors_t<TenT>& coors)
    {
        elem_coors_t<TenT> new_coors(coors.size());
        for(std::size_t b = 0; b < new_order.size(); ++b)
            new_coors[b] = coors[new_order[b]];
        elem_t<TenT> v = tcapi::get_elem<TenT>(ctx, in, coors);
        detail::set_elem_impl<TenT>(out, to_ivs<TenT>(itensor::inds(out), new_coors), v);
    });
}

} // namespace detail

template<typename TenT>
void reshape(const context_handle_t<TenT>& ctx,
             const ten_t<TenT>& in,
             const shape_t<TenT>& new_shape,
             ten_t<TenT>& out)
{
    detail::reshape_impl<TenT>(ctx, in, new_shape, out);
}

template<typename TenT>
void reshape(const context_handle_t<TenT>& ctx,
             ten_t<TenT>& inout,
             const shape_t<TenT>& new_shape)
{
    ten_t<TenT> out;
    detail::reshape_impl<TenT>(ctx, inout, new_shape, out);
    inout = std::move(out);
}

template<typename TenT>
void transpose(const context_handle_t<TenT>& ctx,
               const ten_t<TenT>& in,
               const List<bond_idx_t<TenT>>& new_order,
               ten_t<TenT>& out)
{
    detail::transpose_impl<TenT>(ctx, in, new_order, out);
}

template<typename TenT>
void transpose(const context_handle_t<TenT>& ctx,
               ten_t<TenT>& inout,
               const List<bond_idx_t<TenT>>& new_order)
{
    ten_t<TenT> out;
    detail::transpose_impl<TenT>(ctx, inout, new_order, out);
    inout = std::move(out);
}

template<typename TenT>
void cplx_conj(const context_handle_t<TenT>& ctx,
               const ten_t<TenT>& in,
               ten_t<TenT>& out)
{
    detail::ensure_active<TenT>(ctx);
    if constexpr (std::is_same_v<elem_t<TenT>, std::complex<double>>)
        out = itensor::conj(in);
    else
        out = copy<TenT>(ctx, in);
}

template<typename TenT>
void cplx_conj(const context_handle_t<TenT>& ctx, ten_t<TenT>& inout)
{
    if constexpr (std::is_same_v<elem_t<TenT>, std::complex<double>>)
        inout = itensor::conj(inout);
}

template<typename TenT, typename Func>
void for_each(const context_handle_t<TenT>& ctx,
              const ten_t<TenT>& in,
              ten_t<TenT>& out,
              Func f)
{
    detail::ensure_active<TenT>(ctx);
    auto dims = shape<TenT>(ctx, in);
    out = allocate<TenT>(ctx, dims);
    auto is = itensor::inds(out);
    detail::for_each_coordinate<TenT>(dims, [&](const elem_coors_t<TenT>& coors)
    {
        elem_t<TenT> el = get_elem<TenT>(ctx, in, coors);
        el = f(el);
        detail::set_elem_impl<TenT>(out, detail::to_ivs<TenT>(is, coors), el);
    });
}

template<typename TenT, typename Func>
void for_each(const context_handle_t<TenT>& ctx, ten_t<TenT>& inout, Func f)
{
    detail::ensure_active<TenT>(ctx);
    auto dims = shape<TenT>(ctx, inout);
    auto is = itensor::inds(inout);
    detail::for_each_coordinate<TenT>(dims, [&](const elem_coors_t<TenT>& coors)
    {
        elem_t<TenT> el = get_elem<TenT>(ctx, inout, coors);
        if constexpr(std::is_void_v<decltype(f(el))>)
            f(el);              // (1) f(elem_t&): mutates el by reference
        else
            el = f(el);         // non-spec convenience: f returns new value
        detail::set_elem_impl<TenT>(inout, detail::to_ivs<TenT>(is, coors), el);
    });
}

template<typename TenT, typename Func>
void for_each(const context_handle_t<TenT>& ctx, const ten_t<TenT>& in, Func f)
{
    detail::ensure_active<TenT>(ctx);
    auto dims = shape<TenT>(ctx, in);
    detail::for_each_coordinate<TenT>(dims, [&](const elem_coors_t<TenT>& coors)
    {
        f(get_elem<TenT>(ctx, in, coors));
    });
}

template<typename TenT, typename Func>
void for_each_with_coors(const context_handle_t<TenT>& ctx,
                         const ten_t<TenT>& in,
                         ten_t<TenT>& out,
                         Func f)
{
    detail::ensure_active<TenT>(ctx);
    auto dims = shape<TenT>(ctx, in);
    out = allocate<TenT>(ctx, dims);
    auto is = itensor::inds(out);
    detail::for_each_coordinate<TenT>(dims, [&](const elem_coors_t<TenT>& coors)
    {
        elem_t<TenT> el = get_elem<TenT>(ctx, in, coors);
        el = f(el, coors);
        detail::set_elem_impl<TenT>(out, detail::to_ivs<TenT>(is, coors), el);
    });
}

template<typename TenT, typename Func>
void for_each_with_coors(const context_handle_t<TenT>& ctx,
                         ten_t<TenT>& inout,
                         Func f)
{
    detail::ensure_active<TenT>(ctx);
    auto dims = shape<TenT>(ctx, inout);
    auto is = itensor::inds(inout);
    detail::for_each_coordinate<TenT>(dims, [&](const elem_coors_t<TenT>& coors)
    {
        elem_t<TenT> el = get_elem<TenT>(ctx, inout, coors);
        if constexpr(std::is_void_v<decltype(f(el, coors))>)
            f(el, coors);       // (1) f(elem_t&, coors): mutates el by reference
        else
            el = f(el, coors);  // non-spec convenience: f returns new value
        detail::set_elem_impl<TenT>(inout, detail::to_ivs<TenT>(is, coors), el);
    });
}

template<typename TenT, typename Func>
void for_each_with_coors(const context_handle_t<TenT>& ctx,
                         const ten_t<TenT>& in,
                         Func f)
{
    detail::ensure_active<TenT>(ctx);
    auto dims = shape<TenT>(ctx, in);
    detail::for_each_coordinate<TenT>(dims, [&](const elem_coors_t<TenT>& coors)
    {
        f(get_elem<TenT>(ctx, in, coors), coors);
    });
}

template<typename TenT>
ten_t<TenT> concatenate(const context_handle_t<TenT>& ctx,
                         const List<CRef<ten_t<TenT>>>& ins,
                         bond_idx_t<TenT> concat_bdidx)
{
    detail::ensure_active<TenT>(ctx);
    auto base_shape = shape<TenT>(ctx, ins[0].get());
    bond_dim_t<TenT> total = 0;
    for(auto const& t : ins) total += shape<TenT>(ctx, t.get())[concat_bdidx];

    shape_t<TenT> out_shape = base_shape;
    out_shape[concat_bdidx] = total;
    auto out = allocate<TenT>(ctx, out_shape);
    auto is_out = itensor::inds(out);

    bond_dim_t<TenT> offset = 0;
    for(auto const& t : ins)
    {
        auto t_shape = shape<TenT>(ctx, t.get());
        detail::for_each_coordinate<TenT>(t_shape, [&](const elem_coors_t<TenT>& coors)
        {
            elem_coors_t<TenT> out_coors = coors;
            out_coors[concat_bdidx] += offset;
            elem_t<TenT> v = get_elem<TenT>(ctx, t.get(), coors);
            detail::set_elem_impl<TenT>(out, detail::to_ivs<TenT>(is_out, out_coors), v);
        });
        offset += t_shape[concat_bdidx];
    }
    return out;
}

template<typename TenT>
ten_t<TenT> stack(const context_handle_t<TenT>& ctx,
                   const List<CRef<ten_t<TenT>>>& ins,
                   bond_idx_t<TenT> stack_bdidx)
{
    detail::ensure_active<TenT>(ctx);
    if(ins.empty())
        throw std::invalid_argument("stack: ins must be non-empty.");

    auto base_shape = shape<TenT>(ctx, ins[0].get());
    shape_t<TenT> out_shape;
    for(std::size_t b = 0; b < base_shape.size(); ++b)
    {
        if(static_cast<bond_idx_t<TenT>>(b) == stack_bdidx)
            out_shape.push_back(static_cast<bond_dim_t<TenT>>(ins.size()));
        out_shape.push_back(base_shape[b]);
    }
    if(static_cast<std::size_t>(stack_bdidx) == base_shape.size())
        out_shape.push_back(static_cast<bond_dim_t<TenT>>(ins.size()));

    auto out = allocate<TenT>(ctx, out_shape);
    auto is_out = itensor::inds(out);

    for(std::size_t k = 0; k < ins.size(); ++k)
    {
        detail::for_each_coordinate<TenT>(base_shape, [&](const elem_coors_t<TenT>& coors)
        {
            elem_coors_t<TenT> out_coors;
            for(std::size_t b = 0; b < coors.size(); ++b)
            {
                if(static_cast<bond_idx_t<TenT>>(b) == stack_bdidx)
                    out_coors.push_back(static_cast<elem_coor_t<TenT>>(k));
                out_coors.push_back(coors[b]);
            }
            if(static_cast<std::size_t>(stack_bdidx) == coors.size())
                out_coors.push_back(static_cast<elem_coor_t<TenT>>(k));

            elem_t<TenT> v = get_elem<TenT>(ctx, ins[k].get(), coors);
            detail::set_elem_impl<TenT>(out, detail::to_ivs<TenT>(is_out, out_coors), v);
        });
    }
    return out;
}

namespace detail {

template<typename TenT>
void expand_impl(const context_handle_t<TenT>& ctx,
                 const ten_t<TenT>& in,
                 const Map<bond_idx_t<TenT>, bond_dim_t<TenT>>& incmap,
                 ten_t<TenT>& out)
{
    detail::ensure_active<TenT>(ctx);
    auto old_shape = shape<TenT>(ctx, in);
    shape_t<TenT> new_shape = old_shape;
    for(auto const& kv : incmap)
        new_shape[kv.first] += kv.second;

    out = allocate<TenT>(ctx, new_shape);
    auto is_out = itensor::inds(out);
    for_each_coordinate<TenT>(old_shape, [&](const elem_coors_t<TenT>& coors)
    {
        elem_t<TenT> v = tcapi::get_elem<TenT>(ctx, in, coors);
        detail::set_elem_impl<TenT>(out, to_ivs<TenT>(is_out, coors), v);
    });
}

} // namespace detail

template<typename TenT>
void expand(const context_handle_t<TenT>& ctx,
            ten_t<TenT>& inout,
            const Map<bond_idx_t<TenT>, bond_dim_t<TenT>>& bond_idx_increment_map)
{
    ten_t<TenT> out;
    detail::expand_impl<TenT>(ctx, inout, bond_idx_increment_map, out);
    inout = std::move(out);
}

template<typename TenT>
void expand(const context_handle_t<TenT>& ctx,
            const ten_t<TenT>& in,
            const Map<bond_idx_t<TenT>, bond_dim_t<TenT>>& bond_idx_increment_map,
            ten_t<TenT>& out)
{
    detail::expand_impl<TenT>(ctx, in, bond_idx_increment_map, out);
}

namespace detail {

template<typename TenT>
void shrink_impl(const context_handle_t<TenT>& ctx,
                 const ten_t<TenT>& in,
                 const bond_idx_elem_coor_pair_map<TenT>& ranges,
                 ten_t<TenT>& out)
{
    detail::ensure_active<TenT>(ctx);
    auto old_shape = shape<TenT>(ctx, in);
    std::size_t r = old_shape.size();

    std::vector<elem_coor_t<TenT>> lo(r), hi(r);
    for(std::size_t b = 0; b < r; ++b)
    {
        auto it = ranges.find(static_cast<bond_idx_t<TenT>>(b));
        if(it != ranges.end())
        {
            lo[b] = it->second.first;
            hi[b] = it->second.second;
        }
        else
        {
            lo[b] = 0;
            hi[b] = old_shape[b];
        }
    }

    shape_t<TenT> new_shape(r);
    for(std::size_t b = 0; b < r; ++b)
        new_shape[b] = static_cast<bond_dim_t<TenT>>(hi[b] - lo[b]);

    out = allocate<TenT>(ctx, new_shape);
    auto is_out = itensor::inds(out);
    for_each_coordinate<TenT>(new_shape, [&](const elem_coors_t<TenT>& new_coors)
    {
        elem_coors_t<TenT> old_coors(r);
        for(std::size_t b = 0; b < r; ++b) old_coors[b] = new_coors[b] + lo[b];
        elem_t<TenT> v = tcapi::get_elem<TenT>(ctx, in, old_coors);
        detail::set_elem_impl<TenT>(out, to_ivs<TenT>(is_out, new_coors), v);
    });
}

} // namespace detail

template<typename TenT>
void shrink(const context_handle_t<TenT>& ctx,
            ten_t<TenT>& inout,
            const detail::bond_idx_elem_coor_pair_map<TenT>& bdidx_elcoor_pair_map)
{
    ten_t<TenT> out;
    detail::shrink_impl<TenT>(ctx, inout, bdidx_elcoor_pair_map, out);
    inout = std::move(out);
}

template<typename TenT>
void shrink(const context_handle_t<TenT>& ctx,
            const ten_t<TenT>& in,
            const detail::bond_idx_elem_coor_pair_map<TenT>& bdidx_elcoor_pair_map,
            ten_t<TenT>& out)
{
    detail::shrink_impl<TenT>(ctx, in, bdidx_elcoor_pair_map, out);
}

namespace detail {

template<typename TenT>
void extract_sub_impl(const context_handle_t<TenT>& ctx,
                      const ten_t<TenT>& in,
                      const List<Pair<elem_coor_t<TenT>, elem_coor_t<TenT>>>& coor_pairs,
                      ten_t<TenT>& out)
{
    detail::ensure_active<TenT>(ctx);
    std::size_t r = coor_pairs.size();
    shape_t<TenT> new_shape(r);
    for(std::size_t b = 0; b < r; ++b)
        new_shape[b] = static_cast<bond_dim_t<TenT>>(coor_pairs[b].second - coor_pairs[b].first);

    out = allocate<TenT>(ctx, new_shape);
    auto is_out = itensor::inds(out);
    for_each_coordinate<TenT>(new_shape, [&](const elem_coors_t<TenT>& new_coors)
    {
        elem_coors_t<TenT> old_coors(r);
        for(std::size_t b = 0; b < r; ++b) old_coors[b] = new_coors[b] + coor_pairs[b].first;
        elem_t<TenT> v = tcapi::get_elem<TenT>(ctx, in, old_coors);
        detail::set_elem_impl<TenT>(out, to_ivs<TenT>(is_out, new_coors), v);
    });
}

} // namespace detail

template<typename TenT>
void extract_sub(const context_handle_t<TenT>& ctx,
                 ten_t<TenT>& inout,
                 const List<Pair<elem_coor_t<TenT>, elem_coor_t<TenT>>>& coor_pairs)
{
    ten_t<TenT> out;
    detail::extract_sub_impl<TenT>(ctx, inout, coor_pairs, out);
    inout = std::move(out);
}

template<typename TenT>
void extract_sub(const context_handle_t<TenT>& ctx,
                 const ten_t<TenT>& in,
                 const List<Pair<elem_coor_t<TenT>, elem_coor_t<TenT>>>& coor_pairs,
                 ten_t<TenT>& out)
{
    detail::extract_sub_impl<TenT>(ctx, in, coor_pairs, out);
}

namespace detail {

template<typename TenT>
void replace_sub_impl(const context_handle_t<TenT>& ctx,
                      const ten_t<TenT>& in,
                      const ten_t<TenT>& sub,
                      const elem_coors_t<TenT>& begin_pt,
                      ten_t<TenT>& out)
{
    detail::ensure_active<TenT>(ctx);
    out = copy<TenT>(ctx, in);
    auto sub_shape = shape<TenT>(ctx, sub);
    auto is_out = itensor::inds(out);

    for_each_coordinate<TenT>(sub_shape, [&](const elem_coors_t<TenT>& sub_coors)
    {
        elem_coors_t<TenT> out_coors(sub_coors.size());
        for(std::size_t b = 0; b < sub_coors.size(); ++b)
            out_coors[b] = sub_coors[b] + begin_pt[b];
        elem_t<TenT> v = tcapi::get_elem<TenT>(ctx, sub, sub_coors);
        detail::set_elem_impl<TenT>(out, to_ivs<TenT>(is_out, out_coors), v);
    });
}

} // namespace detail

template<typename TenT>
void replace_sub(const context_handle_t<TenT>& ctx,
                 ten_t<TenT>& inout,
                 const ten_t<TenT>& sub,
                 const elem_coors_t<TenT>& begin_pt)
{
    ten_t<TenT> out;
    detail::replace_sub_impl<TenT>(ctx, inout, sub, begin_pt, out);
    inout = std::move(out);
}

template<typename TenT>
void replace_sub(const context_handle_t<TenT>& ctx,
                 const ten_t<TenT>& in,
                 const ten_t<TenT>& sub,
                 const elem_coors_t<TenT>& begin_pt,
                 ten_t<TenT>& out)
{
    detail::replace_sub_impl<TenT>(ctx, in, sub, begin_pt, out);
}

template<typename TenT>
real_ten_t<TenT> real(const context_handle_t<TenT>& ctx, const ten_t<TenT>& in)
{
    detail::ensure_active<TenT>(ctx);
    auto sh = shape<TenT>(ctx, in);
    auto is_in = itensor::inds(in);

    std::vector<itensor::Index> real_inds;
    for(auto const& I : is_in) real_inds.push_back(I);
    itensor::ITensor out{itensor::IndexSet(real_inds)};

    detail::for_each_coordinate<TenT>(sh, [&](const elem_coors_t<TenT>& coors)
    {
        auto el = get_elem<TenT>(ctx, in, coors);
        double rv;
        if constexpr (std::is_same_v<elem_t<TenT>, std::complex<double>>)
            rv = el.real();
        else
            rv = static_cast<double>(el);
        detail::set_elem_impl<TenT>(out, detail::to_ivs<TenT>(itensor::inds(out), coors), rv);
    });
    return out;
}

template<typename TenT>
real_ten_t<TenT> imag(const context_handle_t<TenT>& ctx, const ten_t<TenT>& in)
{
    detail::ensure_active<TenT>(ctx);
    auto sh = shape<TenT>(ctx, in);
    auto is_in = itensor::inds(in);

    std::vector<itensor::Index> real_inds;
    for(auto const& I : is_in) real_inds.push_back(I);
    itensor::ITensor out{itensor::IndexSet(real_inds)};

    detail::for_each_coordinate<TenT>(sh, [&](const elem_coors_t<TenT>& coors)
    {
        double iv = 0.0;
        if constexpr (std::is_same_v<elem_t<TenT>, std::complex<double>>)
            iv = get_elem<TenT>(ctx, in, coors).imag();
        detail::set_elem_impl<TenT>(out, detail::to_ivs<TenT>(itensor::inds(out), coors), iv);
    });
    return out;
}

template<typename TenT>
cplx_ten_t<TenT> to_cplx(const context_handle_t<TenT>& ctx, const ten_t<TenT>& in)
{
    detail::ensure_active<TenT>(ctx);
    if constexpr (std::is_same_v<elem_t<TenT>, std::complex<double>>)
        return copy<TenT>(ctx, in);
    else
    {
        auto sh = shape<TenT>(ctx, in);
        auto is_in = itensor::inds(in);

        std::vector<itensor::Index> inds_vec;
        for(auto const& I : is_in) inds_vec.push_back(I);
        itensor::ITensor out{itensor::IndexSet(inds_vec)};

        detail::for_each_coordinate<TenT>(sh, [&](const elem_coors_t<TenT>& coors)
        {
            std::complex<double> v(static_cast<double>(get_elem<TenT>(ctx, in, coors)), 0.0);
            detail::set_elem_impl<TenT>(out, detail::to_ivs<TenT>(itensor::inds(out), coors), v);
        });
        return out;
    }
}

} // namespace tcapi