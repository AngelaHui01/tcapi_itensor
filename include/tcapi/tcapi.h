#pragma once

#include "tcapi/type_system.h"      // auxiliary types, tensor_traits, context
#include "tcapi/detail.h"           // internal helpers (not public API)
#include "tcapi/queries.h"          // order, shape, size, size_bytes, get_elem
#include "tcapi/constructors.h"     // allocate, zeros, assign_from_range, fill,
#include "tcapi/io.h"               // save, load
#include "tcapi/manipulation.h"     // set_elem, transpose, reshape, cplx_conj,
#include "tcapi/linalg.h"           // norm, diag, normalize, scale, trace,
#include "tcapi/misc.h"             // create_context, destroy_context, version,
#include "tcapi/diagnostics.h"      // TCAPI_VERBOSE handling