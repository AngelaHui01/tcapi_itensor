// tcapi/linalg.h
// Mirrors tcapi_numpy/linalg.py.
// Sec. C2e — linear algebra: norm, diag, normalize, scale, trace, exp,
// inverse, contract, linear_combine, svd, trunc_svd, qr, lq, eigvals,
// eigvalsh, eig, eigh.
#pragma once

#include "itensor/all.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <string_view>
#include <tuple>
#include <utility>

#include "tcapi/type_system.h"
#include "tcapi/detail.h"
#include "tcapi/queries.h"
#include "tcapi/constructors.h"

namespace tcapi {

// --- norm ---------------------------------------------------------------
// Sec. C2e — Frobenius norm.
template<typename TenT>
real_t<TenT> norm(const context_handle_t<TenT>& ctx, const tent_t<TenT>& a)
{
    detail::ensure_active<TenT>(ctx);
    return itensor::norm(a);
}

// --- contract (shared-index convenience) --------------------------------
// Sec. C2e — sum over all indices shared by a and b (ITensor implicit
// contraction). Convenience overload; the label-based forms below are the
// primary spec API.
template<typename TenT>
tent_t<TenT> contract(const context_handle_t<TenT>& ctx,
                      const tent_t<TenT>& a,
                      const tent_t<TenT>& b)
{
    detail::ensure_active<TenT>(ctx);
    return a * b;
}

// --- contract with explicit bond labels --------------------------------
// Sec. C2e — einsum-style: labels shared between bdlabs_a/bdlabs_b are
// summed; labels present in bdlabs_c are kept as free bonds of the output
// in that exact bond order. The result is written into the output tensor c;
// the C++ API guarantees c may alias a, b, or both (the computation is done
// into a fresh tensor and then assigned to c). Implemented via direct
// coordinate-summation so it works regardless of how a and b's Index objects
// were built.

namespace detail {

template<typename TenT, typename Lab>
tent_t<TenT> contract_labeled(const context_handle_t<TenT>& ctx,
                              const tent_t<TenT>& a,
                              const List<Lab>& bdlabs_a,
                              const tent_t<TenT>& b,
                              const List<Lab>& bdlabs_b,
                              const List<Lab>& bdlabs_c)
{
    auto shape_a = shape<TenT>(ctx, a);
    auto shape_b = shape<TenT>(ctx, b);

    auto lab_pos = [](const List<Lab>& labs, const Lab& lab)
    {
        auto it = std::find(labs.begin(), labs.end(), lab);
        return it == labs.end() ? static_cast<std::size_t>(-1)
                                : static_cast<std::size_t>(it - labs.begin());
    };

    auto dim_of = [&](const Lab& lab) -> bond_dim_t<TenT>
    {
        auto ia = lab_pos(bdlabs_a, lab);
        if(ia != static_cast<std::size_t>(-1)) return shape_a[ia];
        auto ib = lab_pos(bdlabs_b, lab);
        if(ib != static_cast<std::size_t>(-1)) return shape_b[ib];
        throw std::invalid_argument("contract: output label not present in either input.");
    };

    // Validate: shared labels must have matching dimensions and must not
    // also be advertised as free (output) labels.
    for(auto const& lab : bdlabs_a)
    {
        auto ib = lab_pos(bdlabs_b, lab);
        if(ib == static_cast<std::size_t>(-1)) continue;
        auto ia = lab_pos(bdlabs_a, lab);
        if(shape_a[ia] != shape_b[ib])
            throw std::invalid_argument("contract: shared bond label dimension mismatch.");
        if(lab_pos(bdlabs_c, lab) != static_cast<std::size_t>(-1))
            throw std::invalid_argument("contract: shared bond label also in output labels.");
    }

    // Labels to sum over: every label appearing in either input that is not a
    // free (output) label. This includes the labels shared by a and b as well
    // as any label present in exactly one input (summed per einsum semantics).
    List<Lab> sum_labs;
    auto append_unique = [&](const Lab& lab)
    {
        if(lab_pos(sum_labs, lab) == static_cast<std::size_t>(-1))
            sum_labs.push_back(lab);
    };
    for(auto const& lab : bdlabs_a)
        if(lab_pos(bdlabs_c, lab) == static_cast<std::size_t>(-1))
            append_unique(lab);
    for(auto const& lab : bdlabs_b)
        if(lab_pos(bdlabs_c, lab) == static_cast<std::size_t>(-1))
            append_unique(lab);

    shape_t<TenT> out_shape;
    for(auto const& lab : bdlabs_c) out_shape.push_back(dim_of(lab));

    auto c = allocate<TenT>(ctx, out_shape);
    auto is_c = itensor::inds(c);

    shape_t<TenT> sum_shape;
    for(auto const& lab : sum_labs) sum_shape.push_back(dim_of(lab));

    detail::for_each_coordinate<TenT>(out_shape, [&](const elem_coors_t<TenT>& out_coors)
    {
        Map<Lab, elem_coor_t<TenT>> free_val;
        for(std::size_t k = 0; k < bdlabs_c.size(); ++k)
            free_val[bdlabs_c[k]] = out_coors[k];

        elem_t<TenT> acc{0};
        if(sum_labs.empty())
        {
            elem_coors_t<TenT> ca(bdlabs_a.size()), cb(bdlabs_b.size());
            for(std::size_t k = 0; k < bdlabs_a.size(); ++k) ca[k] = free_val[bdlabs_a[k]];
            for(std::size_t k = 0; k < bdlabs_b.size(); ++k) cb[k] = free_val[bdlabs_b[k]];
            acc = get_elem<TenT>(ctx, a, ca) * get_elem<TenT>(ctx, b, cb);
        }
        else
        {
            detail::for_each_coordinate<TenT>(sum_shape, [&](const elem_coors_t<TenT>& s_coors)
            {
                Map<Lab, elem_coor_t<TenT>> full_val = free_val;
                for(std::size_t k = 0; k < sum_labs.size(); ++k)
                    full_val[sum_labs[k]] = s_coors[k];

                elem_coors_t<TenT> ca(bdlabs_a.size()), cb(bdlabs_b.size());
                for(std::size_t k = 0; k < bdlabs_a.size(); ++k) ca[k] = full_val[bdlabs_a[k]];
                for(std::size_t k = 0; k < bdlabs_b.size(); ++k) cb[k] = full_val[bdlabs_b[k]];
                acc += get_elem<TenT>(ctx, a, ca) * get_elem<TenT>(ctx, b, cb);
            });
        }

        detail::set_elem_impl<TenT>(c, detail::to_ivs<TenT>(is_c, out_coors), acc);
    });
    return c;
}

} // namespace detail

// Integer bond labels.
template<typename TenT>
void contract(const context_handle_t<TenT>& ctx,
              const tent_t<TenT>& a, const List<bond_label_t<TenT>>& bdlabs_a,
              const tent_t<TenT>& b, const List<bond_label_t<TenT>>& bdlabs_b,
              tent_t<TenT>& c, const List<bond_label_t<TenT>>& bdlabs_c)
{
    detail::ensure_active<TenT>(ctx);
    c = detail::contract_labeled<TenT, bond_label_t<TenT>>(
        ctx, a, bdlabs_a, b, bdlabs_b, bdlabs_c);
}

// String labels ("ijk"-style, one char per bond).
template<typename TenT>
void contract(const context_handle_t<TenT>& ctx,
              const tent_t<TenT>& a, const std::string_view bdlabs_a,
              const tent_t<TenT>& b, const std::string_view bdlabs_b,
              tent_t<TenT>& c, const std::string_view bdlabs_c)
{
    detail::ensure_active<TenT>(ctx);
    auto to_labs = [](const std::string_view s)
    {
        List<bond_label_t<TenT>> labs;
        for(char ch : s) labs.push_back(static_cast<bond_label_t<TenT>>(ch));
        return labs;
    };
    c = detail::contract_labeled<TenT, bond_label_t<TenT>>(
        ctx, a, to_labs(bdlabs_a), b, to_labs(bdlabs_b), to_labs(bdlabs_c));
}

// --- diag ---------------------------------------------------------------
// Sec. C2e — (1) 1st-order input -> 2nd-order diagonal tensor.
//            (2) 2nd-order input -> 1st-order diagonal vector.

template<typename TenT>
void diag(const context_handle_t<TenT>& ctx, const tent_t<TenT>& in, tent_t<TenT>& out)
{
    detail::ensure_active<TenT>(ctx);
    auto ord = order<TenT>(ctx, in);
    auto in_shape = shape<TenT>(ctx, in);

    if(ord == 1)
    {
        bond_dim_t<TenT> N = in_shape[0];
        out = allocate<TenT>(ctx, shape_t<TenT>{N, N});
        auto is = itensor::inds(out);
        for(bond_dim_t<TenT> k = 0; k < N; ++k)
        {
            elem_coors_t<TenT> in_coors{k};
            elem_t<TenT> v = get_elem<TenT>(ctx, in, in_coors);
            detail::set_elem_impl<TenT>(out, detail::to_ivs<TenT>(is, elem_coors_t<TenT>{k, k}), v);
        }
    }
    else if(ord == 2)
    {
        bond_dim_t<TenT> M = in_shape[0], N = in_shape[1];
        bond_dim_t<TenT> r = std::min(M, N);
        out = allocate<TenT>(ctx, shape_t<TenT>{r});
        auto is = itensor::inds(out);
        for(bond_dim_t<TenT> k = 0; k < r; ++k)
        {
            elem_t<TenT> v = get_elem<TenT>(ctx, in, elem_coors_t<TenT>{k, k});
            detail::set_elem_impl<TenT>(out, detail::to_ivs<TenT>(is, elem_coors_t<TenT>{k}), v);
        }
    }
    else
    {
        throw std::invalid_argument("diag: input must be 1st- or 2nd-order.");
    }
}

// In-place overload (numpy-style): diag(inout) replaces inout in place.
template<typename TenT>
void diag(const context_handle_t<TenT>& ctx, tent_t<TenT>& inout)
{
    tent_t<TenT> out;
    diag<TenT>(ctx, inout, out);
    inout = std::move(out);
}

// --- scale --------------------------------------------------------------
// Sec. C2e — in-place and out-of-place scalar multiplication.

template<typename TenT>
void scale(const context_handle_t<TenT>& ctx, tent_t<TenT>& inout, elem_t<TenT> s)
{
    detail::ensure_active<TenT>(ctx);
    inout *= s;
}

template<typename TenT>
void scale(const context_handle_t<TenT>& ctx,
           const tent_t<TenT>& in, elem_t<TenT> s, tent_t<TenT>& out)
{
    detail::ensure_active<TenT>(ctx);
    out = in * s;
}

// --- normalize ----------------------------------------------------------
// Sec. C2e — rescales inout in-place to unit Frobenius norm; returns the
// original norm. Out-of-place form writes into out.

template<typename TenT>
real_t<TenT> normalize(const context_handle_t<TenT>& ctx, tent_t<TenT>& inout)
{
    detail::ensure_active<TenT>(ctx);
    real_t<TenT> n = itensor::norm(inout);
    if(n == real_t<TenT>{0})
        throw std::runtime_error("normalize: cannot normalise a zero tensor.");
    inout /= n;
    return n;
}

template<typename TenT>
real_t<TenT> normalize(const context_handle_t<TenT>& ctx,
                       const tent_t<TenT>& in, tent_t<TenT>& out)
{
    detail::ensure_active<TenT>(ctx);
    out = in;
    return normalize<TenT>(ctx, out);
}

// --- trace --------------------------------------------------------------
// Sec. C2e — partial trace summing over the bond pairs in bdidx_pairs.

namespace detail {

template<typename TenT>
void trace_impl(const context_handle_t<TenT>& ctx,
                const tent_t<TenT>& in,
                const bond_idx_pairs_t<TenT>& bdidx_pairs,
                tent_t<TenT>& out)
{
    detail::ensure_active<TenT>(ctx);
    auto old_shape = shape<TenT>(ctx, in);
    std::size_t r = old_shape.size();

    std::vector<bool> is_traced(r, false);
    for(auto const& pr : bdidx_pairs)
    {
        if(old_shape[pr.first] != old_shape[pr.second])
            throw std::invalid_argument("trace: paired bonds must have matching dimension.");
        is_traced[pr.first] = true;
        is_traced[pr.second] = true;
    }

    List<bond_idx_t<TenT>> kept_bonds;
    for(std::size_t b = 0; b < r; ++b)
        if(!is_traced[b]) kept_bonds.push_back(static_cast<bond_idx_t<TenT>>(b));

    shape_t<TenT> out_shape;
    for(auto b : kept_bonds) out_shape.push_back(old_shape[b]);

    out = allocate<TenT>(ctx, out_shape);
    auto is_out = itensor::inds(out);

    shape_t<TenT> trace_shape;
    for(auto const& pr : bdidx_pairs) trace_shape.push_back(old_shape[pr.first]);

    for_each_coordinate<TenT>(out_shape, [&](const elem_coors_t<TenT>& out_coors)
    {
        elem_t<TenT> acc{0};
        for_each_coordinate<TenT>(trace_shape, [&](const elem_coors_t<TenT>& trace_coors)
        {
            elem_coors_t<TenT> full(r);
            for(std::size_t k = 0; k < kept_bonds.size(); ++k)
                full[kept_bonds[k]] = out_coors[k];
            for(std::size_t k = 0; k < bdidx_pairs.size(); ++k)
            {
                full[bdidx_pairs[k].first]  = trace_coors[k];
                full[bdidx_pairs[k].second] = trace_coors[k];
            }
            acc += tcapi::get_elem<TenT>(ctx, in, full);
        });

        if(!out_shape.empty())
            detail::set_elem_impl<TenT>(out, to_ivs<TenT>(is_out, out_coors), acc);
        else
            out.set(acc); // rank-0 tensor
    });
}

} // namespace detail

template<typename TenT>
void trace(const context_handle_t<TenT>& ctx,
           tent_t<TenT>& inout,
           const detail::bond_idx_pairs_t<TenT>& bdidx_pairs)
{
    tent_t<TenT> out;
    detail::trace_impl<TenT>(ctx, inout, bdidx_pairs, out);
    inout = std::move(out);
}

template<typename TenT>
void trace(const context_handle_t<TenT>& ctx,
           const tent_t<TenT>& in,
           const detail::bond_idx_pairs_t<TenT>& bdidx_pairs,
           tent_t<TenT>& out)
{
    detail::trace_impl<TenT>(ctx, in, bdidx_pairs, out);
}

// --- linear_combine -----------------------------------------------------
// Sec. C2e — forms sum_i coefs[i] * ins[i]. Inputs are borrowed
// (List<CRef<TenT>>); no copies are taken.

template<typename TenT>
tent_t<TenT> linear_combine(const context_handle_t<TenT>& ctx,
                            const List<CRef<tent_t<TenT>>>& ins,
                            const List<elem_t<TenT>>& coefs)
{
    detail::ensure_active<TenT>(ctx);
    if(ins.size() != coefs.size())
        throw std::invalid_argument("linear_combine: ins and coefs must have the same length.");
    if(ins.empty())
        throw std::invalid_argument("linear_combine: ins must be non-empty.");

    auto ref_inds = itensor::inds(ins[0].get());
    auto out = ins[0].get() * coefs[0];

    for(std::size_t k = 1; k < ins.size(); ++k)
    {
        auto k_inds = itensor::inds(ins[k].get());
        if(k_inds.size() != ref_inds.size())
            throw std::invalid_argument("linear_combine: shape mismatch among inputs.");

        auto term = ins[k].get();
        term = itensor::replaceInds(term, k_inds, ref_inds);
        out += coefs[k] * term;
    }
    return out;
}

template<typename TenT>
tent_t<TenT> linear_combine(const context_handle_t<TenT>& ctx,
                            const List<CRef<tent_t<TenT>>>& ins)
{
    List<elem_t<TenT>> coefs(ins.size(), elem_t<TenT>{1});
    return linear_combine<TenT>(ctx, ins, coefs);
}

// --- eig-family shared helper -------------------------------------------
// Matricizes a by treating the first num_of_bds_as_row bonds as the row
// side, combining each side into a single Index via ITensor combiners.

namespace detail {

template<typename TenT>
struct Matricized
{
    itensor::ITensor Cr, Cc;
    itensor::Index   cr, cc;
    itensor::ITensor M; // a * Cr * Cc over (cr, cc) with scale absorbed
};

template<typename TenT>
Matricized<TenT>
matricize(const context_handle_t<TenT>& ctx,
          const tent_t<TenT>& a,
          order_t<TenT> num_of_bds_as_row,
          const char* fname)
{
    Matricized<TenT> res;
    detail::ensure_active<TenT>(ctx);
    auto is = itensor::inds(a);
    if(num_of_bds_as_row < 1 ||
       num_of_bds_as_row >= static_cast<order_t<TenT>>(is.size()))
        throw std::invalid_argument(std::string(fname) +
            ": invalid num_of_bds_as_row (must satisfy 1 <= k < r).");
    std::vector<itensor::Index> row_inds, col_inds;
    for(int b = 0; b < static_cast<int>(is.size()); ++b)
        (b < num_of_bds_as_row ? row_inds : col_inds).push_back(is[b]);
    auto comb_r = itensor::combiner(row_inds, {"IndexName=", "cr"});
    auto comb_c = itensor::combiner(col_inds, {"IndexName=", "cc"});
    res.Cr = std::get<0>(comb_r);
    res.cr = std::get<1>(comb_r);
    res.Cc = std::get<0>(comb_c);
    res.cc = std::get<1>(comb_c);
    res.M = a * res.Cr * res.Cc;
    return res;
}

} // namespace detail

// --- exp -----------------------------------------------------------------
// Sec. C2e — general matrix exponential exp(in_out); the tensor is
// matricized by treating the first num_of_bds_as_row bonds as the row side
// and must be square. Matches tcapi_numpy, which uses scipy.linalg.expm on
// the matricized tensor (arbitrary real/complex matrices, NOT Hermitian-only).
// ITensor implementation: dense itensor::expMatrix (Padé approximation with
// scaling and squaring on itensor::Mat<Real>/Mat<Cplx>), unmatricized back
// to the input bond layout. Valid for symmetric <-> nonsymmetric input
// regardless of whether the input indices are prime-level paired.

namespace detail {

template<typename TenT>
tent_t<TenT> exp_impl(const context_handle_t<TenT>& ctx,
                      const tent_t<TenT>& a, order_t<TenT> num_of_bds_as_row)
{
    detail::ensure_active<TenT>(ctx);
    auto mx = detail::matricize<TenT>(ctx, a, num_of_bds_as_row, "exp");
    int n = itensor::dim(mx.cr);
    if(n != itensor::dim(mx.cc))
        throw std::invalid_argument("exp: matricized tensor must be square.");

    if constexpr(std::is_same<elem_t<TenT>, itensor::Real>::value)
    {
        itensor::Matrix Mmat(n, n);
        for(int i = 1; i <= n; ++i)
            for(int j = 1; j <= n; ++j)
                Mmat(i - 1, j - 1) = itensor::eltC(mx.M, mx.cr(i), mx.cc(j)).real();
        itensor::Matrix E = itensor::expMatrix(Mmat, itensor::Real{1});
        itensor::ITensor Et{mx.cc, mx.cr};
        for(int r = 1; r <= n; ++r)
            for(int c = 1; c <= n; ++c)
                Et.set(mx.cc(r), mx.cr(c), E(r - 1, c - 1));
        return Et * itensor::dag(mx.Cc) * itensor::dag(mx.Cr);
    }
    else
    {
        itensor::CMatrix Mmat(n, n);
        for(int i = 1; i <= n; ++i)
            for(int j = 1; j <= n; ++j)
                Mmat(i - 1, j - 1) = itensor::eltC(mx.M, mx.cr(i), mx.cc(j));
        itensor::CMatrix E = itensor::expMatrix(Mmat, itensor::Cplx{1});
        itensor::ITensor Et{mx.cc, mx.cr};
        for(int r = 1; r <= n; ++r)
            for(int c = 1; c <= n; ++c)
                Et.set(mx.cc(r), mx.cr(c), E(r - 1, c - 1));
        return Et * itensor::dag(mx.Cc) * itensor::dag(mx.Cr);
    }
}

} // namespace detail

template<typename TenT>
void exp(const context_handle_t<TenT>& ctx,
         tent_t<TenT>& inout, order_t<TenT> num_of_bds_as_row)
{
    detail::ensure_active<TenT>(ctx);
    tent_t<TenT> out = detail::exp_impl<TenT>(ctx, inout, num_of_bds_as_row);
    inout = std::move(out);
}

template<typename TenT>
void exp(const context_handle_t<TenT>& ctx,
         const tent_t<TenT>& in, order_t<TenT> num_of_bds_as_row, tent_t<TenT>& out)
{
    detail::ensure_active<TenT>(ctx);
    out = detail::exp_impl<TenT>(ctx, in, num_of_bds_as_row);
}

// --- inverse ------------------------------------------------------------
// Sec. C2e — matrix inverse via matricization + Gauss-Jordan elimination.
// Requires the tensor to matricize to a square matrix.

namespace detail {

/// Gauss-Jordan elimination computing the inverse of the n x n matrix M,
/// with partial pivoting. Works in real or complex arithmetic.
template<typename ElemT>
itensor::Mat<ElemT> gauss_jordan_inverse(const itensor::Mat<ElemT>& M, int n)
{
    itensor::Mat<ElemT> Aug(n, 2 * n);
    for(int i = 0; i < n; ++i)
    {
        for(int j = 0; j < n; ++j) Aug(i, j) = M(i, j);
        for(int j = 0; j < n; ++j) Aug(i, n + j) = (i == j) ? ElemT{1} : ElemT{0};
    }

    for(int col = 0; col < n; ++col)
    {
        int piv = col;
        double best = std::abs(Aug(col, col));
        for(int r = col + 1; r < n; ++r)
        {
            double cand = std::abs(Aug(r, col));
            if(cand > best) { best = cand; piv = r; }
        }
        if(best < 1e-14)
            throw std::runtime_error("inverse: matrix is singular or near-singular.");

        if(piv != col)
            for(int j = 0; j < 2 * n; ++j)
                std::swap(Aug(col, j), Aug(piv, j));

        ElemT diagv = Aug(col, col);
        for(int j = 0; j < 2 * n; ++j)
            Aug(col, j) /= diagv;

        for(int r = 0; r < n; ++r)
        {
            if(r == col) continue;
            ElemT factor = Aug(r, col);
            if(factor == ElemT{0}) continue;
            for(int j = 0; j < 2 * n; ++j)
                Aug(r, j) -= factor * Aug(col, j);
        }
    }

    itensor::Mat<ElemT> Minv(n, n);
    for(int i = 0; i < n; ++i)
        for(int j = 0; j < n; ++j)
            Minv(i, j) = Aug(i, n + j);
    return Minv;
}

template<typename TenT>
tent_t<TenT> inverse_impl(const context_handle_t<TenT>& ctx,
                          const tent_t<TenT>& a, order_t<TenT> num_of_bds_as_row)
{
    detail::ensure_active<TenT>(ctx);
    auto is = itensor::inds(a);
    if(num_of_bds_as_row < 1 ||
       num_of_bds_as_row >= static_cast<order_t<TenT>>(is.size()))
        throw std::invalid_argument("inverse: invalid num_of_bds_as_row (must satisfy 1 <= k < r).");

    // --- Split indices into "row" and "column" groups based on num_of_bds_as_row.
    std::vector<itensor::Index> row_inds, col_inds;
    for(int b = 0; b < static_cast<int>(is.size()); ++b)
        (b < num_of_bds_as_row ? row_inds : col_inds).push_back(is[b]);

    // --- Combine each group into a single Index via ITensor combiners.
    auto comb_r = itensor::combiner(row_inds, {"IndexName=", "cr"});
    auto comb_c = itensor::combiner(col_inds, {"IndexName=", "cc"});
    itensor::ITensor Cr = std::get<0>(comb_r);
    itensor::Index   ir = std::get<1>(comb_r);
    itensor::ITensor Cc = std::get<0>(comb_c);
    itensor::Index   ic = std::get<1>(comb_c);

    // --- Matricize: M now has exactly indices (ir, ic).
    auto M = a * Cr * Cc;

    int n = itensor::dim(ir);
    if(n != itensor::dim(ic))
        throw std::invalid_argument("inverse: matricized tensor must be square.");

    if constexpr (std::is_same<elem_t<TenT>, itensor::Real>::value)
    {
        itensor::Matrix Mmat(n, n);
        for(int i = 1; i <= n; ++i)
            for(int j = 1; j <= n; ++j)
                Mmat(i - 1, j - 1) = itensor::eltC(M, ir(i), ic(j)).real();
        itensor::Matrix Minv = gauss_jordan_inverse(Mmat, n);
        itensor::ITensor Minv_t{ic, ir};
        for(int r = 1; r <= n; ++r)
            for(int c = 1; c <= n; ++c)
                Minv_t.set(ic(r), ir(c), Minv(r - 1, c - 1));
        return Minv_t * itensor::dag(Cc) * itensor::dag(Cr);
    }
    else
    {
        itensor::CMatrix Mmat(n, n);
        for(int i = 1; i <= n; ++i)
            for(int j = 1; j <= n; ++j)
                Mmat(i - 1, j - 1) = itensor::eltC(M, ir(i), ic(j));
        itensor::CMatrix Minv = gauss_jordan_inverse(Mmat, n);
        itensor::ITensor Minv_t{ic, ir};
        for(int r = 1; r <= n; ++r)
            for(int c = 1; c <= n; ++c)
                Minv_t.set(ic(r), ir(c), Minv(r - 1, c - 1));
        return Minv_t * itensor::dag(Cc) * itensor::dag(Cr);
    }
}

} // namespace detail

// In-place (numpy-style): inverse(inout, num_of_bds_as_row).
template<typename TenT>
void inverse(const context_handle_t<TenT>& ctx,
             tent_t<TenT>& inout, order_t<TenT> num_of_bds_as_row)
{
    tent_t<TenT> out = detail::inverse_impl<TenT>(ctx, inout, num_of_bds_as_row);
    inout = std::move(out);
}

// Out-of-place convenience returning the inverse.
template<typename TenT>
tent_t<TenT> inverse(const context_handle_t<TenT>& ctx,
                     const tent_t<TenT>& a, order_t<TenT> num_of_bds_as_row)
{
    return detail::inverse_impl<TenT>(ctx, a, num_of_bds_as_row);
}

// Out-of-place convenience writing the inverse into `out`.
template<typename TenT>
void inverse(const context_handle_t<TenT>& ctx,
             const tent_t<TenT>& a, order_t<TenT> num_of_bds_as_row,
             tent_t<TenT>& out)
{
    detail::ensure_active<TenT>(ctx);
    out = detail::inverse_impl<TenT>(ctx, a, num_of_bds_as_row);
}

// --- svd -----------------------------------------------------------------
// Sec. C2e — singular value decomposition. a is matricized by treating the
// first num_of_bds_as_row bonds as the row side. Outputs (u, sigma, v_dag)
// via output parameters:
//   u {d_0..d_{k-1}, kappa}, sigma {kappa,kappa} (real diagonal,
//   non-increasing), v_dag {kappa, d_k..d_{r-1}}.
// ITensor's svd returns V over (col..., link); V's stored element at
// (col,link) is already the (unconjugated) right-singular factor so that
// A = U*S*V holds, i.e. v_dag = permute(V, {link, col...}).

template<typename TenT>
void svd(const context_handle_t<TenT>& ctx,
         const tent_t<TenT>& a,
         order_t<TenT> num_of_bds_as_row,
         tent_t<TenT>& u,
         real_ten_t<TenT>& sigma,
         tent_t<TenT>& v_dag)
{
    detail::ensure_active<TenT>(ctx);
    auto is = itensor::inds(a);
    if(num_of_bds_as_row < 1 ||
       num_of_bds_as_row >= static_cast<order_t<TenT>>(is.size()))
        throw std::invalid_argument("svd: invalid num_of_bds_as_row (must satisfy 1 <= k < r).");
    std::vector<itensor::Index> row_inds, col_inds;
    for(int b = 0; b < static_cast<int>(is.size()); ++b)
        (b < num_of_bds_as_row ? row_inds : col_inds).push_back(is[b]);

    itensor::ITensor U{itensor::IndexSet(row_inds)};
    itensor::ITensor S, V;
    itensor::svd(a, U, S, V);

    // v_dag must be (kappa, d_k..d_{n-1}).
    std::vector<itensor::Index> v_order;
    v_order.push_back(itensor::commonIndex(S, V));
    v_order.insert(v_order.end(), col_inds.begin(), col_inds.end());
    itensor::ITensor vdag = itensor::permute(V, itensor::IndexSet(v_order));

    u = U;
    sigma = S;
    v_dag = vdag;
}

// --- trunc_svd -----------------------------------------------------------
// Sec. C2e — truncated SVD.
// (1) simple form: (ctx, a, k, u, sigma, v_dag, trunc_err, chi_max, s_min)
// (2) full form:   (..., trunc_err, chi_min, chi_max, target_trunc_err, s_min)
// Truncation error = sum_{i=chi}^{kappa-1} s_i^2 / sum_i s_i^2.
// Strategy: drop s_i < s_min; keep at least chi_min survivors (never restore
// dropped values); increase chi until err <= target_trunc_err or
// chi == chi_max.

template<typename TenT>
void trunc_svd(const context_handle_t<TenT>& ctx,
               const tent_t<TenT>& a,
               order_t<TenT> num_of_bds_as_row,
               tent_t<TenT>& u,
               real_ten_t<TenT>& sigma,
               tent_t<TenT>& v_dag,
               real_t<TenT>& trunc_err,
               bond_dim_t<TenT> chi_min,
               bond_dim_t<TenT> chi_max,
               real_t<TenT> target_trunc_err,
               real_t<TenT> s_min)
{
    detail::ensure_active<TenT>(ctx);
    auto is = itensor::inds(a);
    if(num_of_bds_as_row < 1 ||
       num_of_bds_as_row >= static_cast<order_t<TenT>>(is.size()))
        throw std::invalid_argument("trunc_svd: invalid num_of_bds_as_row (must satisfy 1 <= k < r).");
    if(chi_min < 0 || chi_max < chi_min)
        throw std::invalid_argument("trunc_svd: invalid chi_min/chi_max range.");

    // Full (untruncated) SVD to get the full spectrum first.
    std::vector<itensor::Index> row_inds, col_inds;
    for(int b = 0; b < static_cast<int>(is.size()); ++b)
        (b < num_of_bds_as_row ? row_inds : col_inds).push_back(is[b]);

    itensor::ITensor U{itensor::IndexSet(row_inds)};
    itensor::ITensor S, V;
    itensor::svd(a, U, S, V);

    // Read the full (descending) singular values from the diagonal of S.
    // S's two bond indices are the (distinct) U-link and V-link objects,
    // so the diagonal is read with the two commonIndex links, not with the
    // primed U-link.
    itensor::Index link  = itensor::commonIndex(U, S);
    itensor::Index vlink = itensor::commonIndex(S, V);
    int kappa = itensor::dim(link);
    List<real_t<TenT>> svals(kappa);
    for(int i = 1; i <= kappa; ++i)
        svals[i - 1] = itensor::elt(S, link(i), vlink(i));

    real_t<TenT> total2{0};
    for(auto s : svals) total2 += s * s;

    // Determine chi: drop s_i < s_min, keep at least chi_min survivors.
    int chi = 0;
    for(int i = 0; i < kappa; ++i)
    {
        if(svals[i] < s_min) break;          // descending order
        ++chi;
    }
    if(chi < chi_min) chi = chi_min;
    if(chi > kappa) chi = kappa;
    if(chi > chi_max) chi = static_cast<int>(chi_max);

    // Increase chi until relative error <= target_trunc_err or chi == chi_max.
    if(target_trunc_err >= real_t<TenT>{0})
    {
        while(chi < kappa && chi < static_cast<int>(chi_max))
        {
            real_t<TenT> kept2{0};
            for(int i = 0; i < chi; ++i) kept2 += svals[i] * svals[i];
            real_t<TenT> err = (total2 > 0) ? (total2 - kept2) / total2 : real_t<TenT>{0};
            if(err <= target_trunc_err) break;
            ++chi;
        }
    }

    // Truncate to rank chi via ITensor's built-in cut (keeps the largest chi
    // singular values; Cutoff=0 ensures only the bond-dim cap applies). A
    // fresh truncated SVD is used because this ITensor version has no
    // Index-slicing helpers (itensor::slice/replaceIndex).
    itensor::ITensor U_trunc{itensor::IndexSet(row_inds)};
    itensor::ITensor S_trunc, V_trunc;
    itensor::svd(a, U_trunc, S_trunc, V_trunc,
                 {"MaxDim=", chi, "Cutoff=", real_t<TenT>{0}});

    std::vector<itensor::Index> v_order;
    v_order.push_back(itensor::commonIndex(S_trunc, V_trunc));
    v_order.insert(v_order.end(), col_inds.begin(), col_inds.end());
    itensor::ITensor vdag = itensor::permute(V_trunc, itensor::IndexSet(v_order));

    // Truncation error = sum of discarded s_i^2 / total.
    real_t<TenT> kept2{0};
    for(int i = 0; i < chi; ++i) kept2 += svals[i] * svals[i];
    trunc_err = (total2 > 0) ? (total2 - kept2) / total2 : real_t<TenT>{0};

    u = U_trunc;
    sigma = S_trunc;
    v_dag = vdag;
}

template<typename TenT>
void trunc_svd(const context_handle_t<TenT>& ctx,
               const tent_t<TenT>& a,
               order_t<TenT> num_of_bds_as_row,
               tent_t<TenT>& u,
               real_ten_t<TenT>& sigma,
               tent_t<TenT>& v_dag,
               real_t<TenT>& trunc_err,
               bond_dim_t<TenT> chi_max,
               real_t<TenT> s_min)
{
    trunc_svd<TenT>(ctx, a, num_of_bds_as_row, u, sigma, v_dag, trunc_err,
                    0, chi_max, real_t<TenT>{-1}, s_min);
}

// --- qr ------------------------------------------------------------------
// Sec. C2e — thin QR. a (row..., col...) -> q {d_0..d_{k-1}, rho},
// r {rho, d_k..}, rho = min(I, J).

template<typename TenT>
void qr(const context_handle_t<TenT>& ctx,
        const tent_t<TenT>& a,
        order_t<TenT> num_of_bds_as_row,
        tent_t<TenT>& q,
        tent_t<TenT>& r)
{
    detail::ensure_active<TenT>(ctx);
    auto is = itensor::inds(a);
    if(num_of_bds_as_row < 1 ||
       num_of_bds_as_row > static_cast<order_t<TenT>>(is.size()))
        throw std::invalid_argument("qr: invalid num_of_bds_as_row.");
    std::vector<itensor::Index> row_inds;
    for(int b = 0; b < num_of_bds_as_row; ++b) row_inds.push_back(is[b]);

    itensor::IndexSet Qis(row_inds);
    auto [Q, R] = itensor::qr(a, Qis);
    q = Q;
    r = R;
}

// --- lq ------------------------------------------------------------------
// Sec. C2e — thin LQ decomposition: a = l * q with
// l {d_0..d_{k-1}, rho} lower-triangular, q {rho, d_k..} with orthonormal
// rows. Implemented via QR of the (conjugate-)transposed tensor:
// a^T = q' * r'  =>  a = (r')^T * (q')^T, so l = dag(r') over (row..., rho)
// and q = dag(q') over (rho, col...).

template<typename TenT>
void lq(const context_handle_t<TenT>& ctx,
        const tent_t<TenT>& a,
        order_t<TenT> num_of_bds_as_row,
        tent_t<TenT>& l,
        tent_t<TenT>& q)
{
    detail::ensure_active<TenT>(ctx);
    auto is = itensor::inds(a);
    if(num_of_bds_as_row < 1 ||
       num_of_bds_as_row > static_cast<order_t<TenT>>(is.size()))
        throw std::invalid_argument("lq: invalid num_of_bds_as_row.");
    std::vector<itensor::Index> row_inds, col_inds;
    for(int b = 0; b < static_cast<int>(is.size()); ++b)
        (b < num_of_bds_as_row ? row_inds : col_inds).push_back(is[b]);

    // Transpose a so that the column side of the original becomes the "row"
    // side of the QR: a_T over (col..., row...) with a_T[col...,row...] =
    // a[row...,col...].
    itensor::IndexSet is_T;
    {
        std::vector<itensor::Index> inds_T;
        inds_T.insert(inds_T.end(), col_inds.begin(), col_inds.end());
        inds_T.insert(inds_T.end(), row_inds.begin(), row_inds.end());
        is_T = itensor::IndexSet(inds_T);
    }
    itensor::ITensor aT(is_T);
    {
        auto sh = shape<TenT>(ctx, a);
        detail::for_each_coordinate<TenT>(sh, [&](const elem_coors_t<TenT>& coors)
        {
            elem_coors_t<TenT> coors_T(coors.size());
            for(int b = 0; b < num_of_bds_as_row; ++b)
                coors_T[b + col_inds.size()] = coors[b];
            for(std::size_t b = 0; b < col_inds.size(); ++b)
                coors_T[b] = coors[b + num_of_bds_as_row];
            auto el = get_elem<TenT>(ctx, a, coors);
            detail::set_elem_impl<TenT>(aT, detail::to_ivs<TenT>(is_T, coors_T), el);
        });
    }

    // QR of the transpose: q' over (col..., rho), r' over (rho, row...).
    itensor::IndexSet Qis_T(col_inds);
    auto [qp, rp] = itensor::qr(aT, Qis_T);

    // l = dag(r') over (row..., rho); q = dag(q') over (rho, col...).
    auto lfull = itensor::dag(rp);
    std::vector<itensor::Index> l_order = row_inds;
    l_order.push_back(itensor::commonIndex(qp, rp));
    l = itensor::permute(lfull, itensor::IndexSet(l_order));

    auto qfull = itensor::dag(qp);
    std::vector<itensor::Index> q_order;
    q_order.push_back(itensor::commonIndex(qp, rp));
    q_order.insert(q_order.end(), col_inds.begin(), col_inds.end());
    q = itensor::permute(qfull, itensor::IndexSet(q_order));
}

// --- eigall ------------------------------------------------------------
// Sec. C2e — general (non-Hermitian) eigendecomposition A'*V = V*Lambda.
// lambda_mat is a complex diagonal tensor {I,I}; v is complex with shape
// {d_0..d_{k-1}, I} (right eigenvectors expanded in the row basis).
// Uses the dense-matrix itensor::eigen routine (dgeev/zgeev); eigenvalues
// are complex in general, order is unspecified (mirrors numpy's eig).

template<typename TenT>
void eig(const context_handle_t<TenT>& ctx,
         const tent_t<TenT>& a,
         order_t<TenT> num_of_bds_as_row,
         cplx_ten_t<TenT>& lambda_mat,
         cplx_ten_t<TenT>& v)
{
    auto mx = detail::matricize<TenT>(ctx, a, num_of_bds_as_row, "eig");
    int n = itensor::dim(mx.cr);
    if(n != itensor::dim(mx.cc))
        throw std::invalid_argument("eig: matricized tensor must be square.");

    itensor::Matrix Vr(n, n), Vi(n, n);
    itensor::Vector dr(n), di(n);
    if constexpr(std::is_same<elem_t<TenT>, itensor::Real>::value)
    {
        itensor::Matrix Mmat(n, n);
        for(int i = 1; i <= n; ++i)
            for(int j = 1; j <= n; ++j)
                Mmat(i - 1, j - 1) = itensor::elt(mx.M, mx.cr(i), mx.cc(j));
        itensor::eigen(Mmat, Vr, Vi, dr, di);
    }
    else
    {
        itensor::CMatrix Mmat(n, n);
        for(int i = 1; i <= n; ++i)
            for(int j = 1; j <= n; ++j)
                Mmat(i - 1, j - 1) = itensor::eltC(mx.M, mx.cr(i), mx.cc(j));
        itensor::eigen(Mmat, Vr, Vi, dr, di);
    }

    auto eig = itensor::Index(n, "eig");
    itensor::ITensor L{eig, itensor::prime(eig)};
    for(int k = 1; k <= n; ++k)
        L.set(eig(k), itensor::prime(eig)(k),
              itensor::Cplx(dr(k - 1), di(k - 1)));
    lambda_mat = L;

    // Right eigenvectors as columns, expanded in the combined row basis,
    // then uncombined back to the original row bonds.
    itensor::ITensor Vt{mx.cr, eig};
    for(int r = 1; r <= n; ++r)
        for(int k = 1; k <= n; ++k)
            Vt.set(mx.cr(r), eig(k),
                   itensor::Cplx(Vr(r - 1, k - 1), Vi(r - 1, k - 1)));
    v = Vt * itensor::dag(mx.Cr);
}

// --- eigvals -----------------------------------------------------------
// Sec. C2e — eigenvalues only (complex 1st-order output w).

template<typename TenT>
void eigvals(const context_handle_t<TenT>& ctx,
             const tent_t<TenT>& a,
             order_t<TenT> num_of_bds_as_row,
             cplx_ten_t<TenT>& w)
{
    cplx_ten_t<TenT> lambda_mat, v;
    eig<TenT>(ctx, a, num_of_bds_as_row, lambda_mat, v);
    auto is = itensor::inds(lambda_mat);
    bond_dim_t<TenT> n = itensor::dim(is[0]);
    auto eigidx = is[0];
    w = itensor::ITensor{eigidx};
    for(int k = 1; k <= n; ++k)
        w.set(eigidx(k), itensor::eltC(lambda_mat, is[0](k), is[1](k)));
}

// --- eigh --------------------------------------------------------------
// Sec. C2e — Hermitian eigendecomposition. lambda_mat is a REAL diagonal
// tensor {I,I} with eigenvalues ASCENDING; v has the same element type as
// the input and shape {d_0..d_{k-1}, I}.

template<typename TenT>
void eigh(const context_handle_t<TenT>& ctx,
          const tent_t<TenT>& a,
          order_t<TenT> num_of_bds_as_row,
          real_ten_t<TenT>& lambda_mat,
          tent_t<TenT>& v)
{
    auto mx = detail::matricize<TenT>(ctx, a, num_of_bds_as_row, "eigh");
    int n = itensor::dim(mx.cr);
    if(n != itensor::dim(mx.cc))
        throw std::invalid_argument("eigh: matricized tensor must be square.");

    // Explicit Hermiticity/symmetry validation: itensor::diagHermitian only
    // reads the upper triangle silently, so reject non-Hermitian inputs with
    // a clean std::invalid_argument rather than producing garbage or aborting.
    {
        itensor::Real asym = 0, scale = 0;
        for(int i = 1; i <= n; ++i)
            for(int j = 1; j <= n; ++j)
            {
                auto mij = itensor::eltC(mx.M, mx.cr(i), mx.cc(j));
                auto mji = itensor::eltC(mx.M, mx.cr(j), mx.cc(i));
                asym = std::max(asym, std::abs(mij - std::conj(mji)));
                scale = std::max(scale, std::abs(mij));
            }
        if(asym > itensor::Real(1e-8) * (1 + scale))
            throw std::invalid_argument("eigh: input matrix is not Hermitian/symmetric.");
    }

    itensor::Vector d(n);
    int ord = n;
    (void)ord;

    // U holds eigenvectors as columns (descending eigenvalues from
    // itensor::diagHermitian); we rebuild in ascending order afterwards.
    if constexpr(std::is_same<elem_t<TenT>, itensor::Real>::value)
    {
        itensor::Matrix Mmat(n, n), U(n, n);
        for(int i = 1; i <= n; ++i)
            for(int j = 1; j <= n; ++j)
                Mmat(i - 1, j - 1) = itensor::elt(mx.M, mx.cr(i), mx.cc(j));
        itensor::diagHermitian(Mmat, U, d);

        std::vector<int> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        std::sort(perm.begin(), perm.end(),
                  [&](int a, int b) { return d(a) < d(b); });

        auto eig = itensor::Index(n, "eig");
        itensor::ITensor L{eig, itensor::prime(eig)};
        for(int k = 1; k <= n; ++k)
            L.set(eig(k), itensor::prime(eig)(k), d(perm[k - 1]));
        lambda_mat = L;

        itensor::ITensor Vt{mx.cr, eig};
        for(int r = 1; r <= n; ++r)
            for(int k = 1; k <= n; ++k)
                Vt.set(mx.cr(r), eig(k), U(r - 1, perm[k - 1]));
        v = Vt * itensor::dag(mx.Cr);
    }
    else
    {
        itensor::CMatrix Mmat(n, n), U(n, n);
        for(int i = 1; i <= n; ++i)
            for(int j = 1; j <= n; ++j)
                Mmat(i - 1, j - 1) = itensor::eltC(mx.M, mx.cr(i), mx.cc(j));
        itensor::diagHermitian(Mmat, U, d);

        std::vector<int> perm(n);
        std::iota(perm.begin(), perm.end(), 0);
        std::sort(perm.begin(), perm.end(),
                  [&](int a, int b) { return d(a) < d(b); });

        auto eig = itensor::Index(n, "eig");
        itensor::ITensor L{eig, itensor::prime(eig)};
        for(int k = 1; k <= n; ++k)
            L.set(eig(k), itensor::prime(eig)(k), d(perm[k - 1]));
        lambda_mat = L;

        itensor::ITensor Vt{mx.cr, eig};
        for(int r = 1; r <= n; ++r)
            for(int k = 1; k <= n; ++k)
                Vt.set(mx.cr(r), eig(k), U(r - 1, perm[k - 1]));
        v = Vt * itensor::dag(mx.Cr);
    }
}

// --- eigvalsh ----------------------------------------------------------
// Sec. C2e — Hermitian eigenvalues only (real 1st-order output w, ascending).

template<typename TenT>
void eigvalsh(const context_handle_t<TenT>& ctx,
              const tent_t<TenT>& a,
              order_t<TenT> num_of_bds_as_row,
              real_ten_t<TenT>& w)
{
    real_ten_t<TenT> lambda_mat;
    tent_t<TenT> v;
    eigh<TenT>(ctx, a, num_of_bds_as_row, lambda_mat, v);
    auto is = itensor::inds(lambda_mat);
    bond_dim_t<TenT> n = itensor::dim(is[0]);
    auto eigidx = is[0];
    w = itensor::ITensor{eigidx};
    for(int k = 1; k <= n; ++k)
        w.set(eigidx(k), itensor::elt(lambda_mat, is[0](k), is[1](k)));
}

} // namespace tcapi