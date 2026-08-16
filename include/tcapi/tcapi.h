// tcapi/tcapi.h
// Single aggregating header for the C++ TCAPI implementation on ITensor,
// mirroring tcapi_numpy/__init__.py.
//
// Include this single header to get the full public API in namespace tcapi:
// a single header "tcapi/tcapi.h" as required by the spec.
#pragma once

#include "tcapi/type_system.h"      // auxiliary types, tensor_traits, context
#include "tcapi/detail.h"           // internal helpers (not public API)
#include "tcapi/queries.h"          // order, shape, size, size_bytes, get_elem
#include "tcapi/constructors.h"     // allocate, zeros, assign_from_range, fill,
                                    // random, eye, copy, move, clear
#include "tcapi/io.h"               // save, load
#include "tcapi/manipulation.h"     // set_elem, transpose, reshape, cplx_conj,
                                    // to_cplx, real, imag, expand, shrink,
                                    // extract_sub, replace_sub, concatenate,
                                    // stack, for_each, for_each_with_coors
#include "tcapi/linalg.h"           // norm, diag, normalize, scale, trace,
                                    // contract, svd, trunc_svd, inverse,
                                    // linear_combine, exp, lq, eigvals,
                                    // eigvalsh, eigh, eig, qr
#include "tcapi/misc.h"             // create_context, destroy_context, version,
                                    // show, close, convert, to_range
#include "tcapi/diagnostics.h"      // TCAPI_VERBOSE handling