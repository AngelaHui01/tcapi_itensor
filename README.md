# TCAPI - ITensor Backend

[TCAPI (Tensor Computing API)](https://arxiv.org/abs/2512.23917) (v2, arXiv:2512.23917v2) is a common tensor-computing interface for tensor-network applications. The specification is also available at [tensorcomputingapi.github.io](https://tensorcomputingapi.github.io/). This repository provides a C++ backend implementation of TCAPI using the [ITensor library](https://itensor.org/).

The project is named `tcapi_itensor`; the C++ namespace is `tcapi`.

## Requirements

- **C++17** or later
- **ITensor** (ITensor v3, built with `-std=c++17`)
- **CMake** (3.16 or later)
- A C++17-capable compiler (e.g., GCC 9+, Clang 10+, or recent MSVC)

## Quick Start

### 1. Clone the Repository

```bash
git clone <repo-url>
cd tcapi_itensor
```

### 2. Install Dependencies

ITensor v3 is vendored in-tree under `itensor/` and treated as an external
dependency: it keeps its own Make-based build workflow, and the TCAPI CMake
build links against the resulting static library in `itensor/lib`. If the
library is absent (e.g. a fresh clone), build ITensor first:

```bash
make -C itensor
```

### 3. Configure and Build (CMake)

```bash
cmake -S . -B build
cmake --build build
```

### 4. Run Tests

```bash
ctest --test-dir build --output-on-failure
```

## TRG Example

The Tensor Renormalization Group example for the two-dimensional classical
Ising model is in `examples/trg.cc`. It follows the ITensor TRG tutorial while
using the TCAPI operations for allocation, truncated SVD, contraction, and
normalization.

```bash
./build/trg                 # T=3.0, max bond dimension=8, 6 scales
./build/trg 2.269185 12 6   # near the critical temperature
```

## Usage Example

```cpp
#include <tcapi/tcapi.h>
#include <itensor/all.h>

using TenT = itensor::ITensor;
using CtxT = tcapi::context_handle_t<TenT>;

int main() {
    CtxT ctx;
    tcapi::create_context(ctx);

    auto a = tcapi::fill<TenT>(ctx, {2, 3}, 2.0);
    auto b = tcapi::eye<TenT>(ctx, 3);

    auto shp = tcapi::shape<TenT>(ctx, a);
    // shp is {2, 3}

    auto nrm = tcapi::norm<TenT>(ctx, b);
    // nrm == sqrt(3)

    tcapi::destroy_context(ctx);
    return 0;
}
```

## API Coverage

This backend implements dense-tensor versions of the main TCAPI categories:

| Category | Functions |
|---|---|
| Context and metadata | `create_context`, `destroy_context`, `version` |
| Queries | `order`, `shape`, `size`, `size_bytes`, `get_elem` |
| Construction | `allocate`, `zeros`, `assign_from_range`, `fill`, `random`, `eye`, `copy`, `move`, `clear` |
| I/O | `save`, `load` using ITensor's HDF5 or binary serialization |
| Manipulation | `set_elem`, `transpose`, `reshape`, `cplx_conj`, `to_cplx`, `real`, `imag`, `expand`, `shrink`, `extract_sub`, `replace_sub`, `concatenate`, `stack`, `for_each`, `for_each_with_coors` |
| Linear algebra | `norm`, `diag`, `normalize`, `scale`, `trace`, `contract`, `linear_combine`, `exp`, `inverse`, `svd`, `trunc_svd`, `qr`, `lq`, `eigvals`, `eigvalsh`, `eig`, `eigh` |
| Utilities | `show`, `close`, `convert`, `to_range` |
| Diagnostics | `TCAPI_VERBOSE` environment variable for runtime call diagnostics — level `1` prints TCAPI function names and compact argument summaries, level `2` also prints measured wall-clock time per wrapped TCAPI call |

Design notes:

- Tensor storage follows ITensor's native dense `ITensor` storage.
- Most mutating-style TCAPI routines are implemented as in-place operations where natural in ITensor; out-of-place variants follow the TCAPI Python spec semantics.
- Symmetry-aware (QN) storage, GPU execution, and automatic differentiation are not implemented in this backend.

## Linear Algebra Example

Tensor contractions are backed by ITensor's `operator*` and index-matching semantics:

```cpp
#include <tcapi/tcapi.h>
#include <itensor/all.h>

using TenT = itensor::ITensor;
using CtxT = tcapi::context_handle_t<TenT>;

int main() {
    CtxT ctx;
    tcapi::create_context(ctx);

    auto identity = tcapi::eye<TenT>(ctx, 3);
    auto matrix = tcapi::fill<TenT>(ctx, {3, 3}, 2.0);

    // contract with explicit index labels (string or non-string);
    // the result is written into the output tensor c (c may alias a/b)
    TenT product;
    tcapi::contract<TenT>(ctx, identity, "ij", matrix, "jk", product, "ik");
    // product should be numerically close to matrix

    // svd writes its factors via output parameters; sigma is a real diagonal
    // tensor of singular values
    TenT u, vdag;
    tcapi::real_ten_t<TenT> sigma;
    tcapi::svd<TenT>(ctx, matrix, 1, u, sigma, vdag);

    tcapi::destroy_context(ctx);
    return 0;
}
```

For tensor decompositions, `num_of_bds_as_row` controls how many leading tensor axes are grouped into the matrix row index before applying ITensor's linear algebra routines.

## Examples

Tensor-network example applications (e.g., a TRG demo and an iTEBD demo) are planned but not yet part of this repository. When added, they will be built by the CMake project as additional executables.

## Changes for Spec v2

This repository was updated from the original TCAPI spec (v1) to
[arXiv:2512.23917v2](https://arxiv.org/abs/2512.23917). Notable breaking
and non-breaking changes are listed below.

### Breaking changes

- **`trunc_svd` argument order** — the full overload now reads
  `(..., trunc_err, chi_max, s_min, chi_min, target_trunc_err)`
  per the v2 spec. The short overload `(..., trunc_err, chi_max, s_min)`
  is unchanged.
- **`concatenate`, `stack`, `linear_combine` now take
  `const List<CRef<TenT>>&`** — callers supply inputs via
  `std::cref()` references. Forwarding overloads accepting
  `std::initializer_list<CRef<TenT>>` are provided for braceless
  template deduction.

### Other changes

- **Zeroth-order (scalar) tensors** — shape `{}`, order `0`, size `1`.
  All constructors, queries, manipulation and linalg routines now
  handle or explicitly reject scalar inputs.
- **Diagonal tensor storage** — `eye`, `svd`/`trunc_svd` sigma return
  ITensor's compact diagITensor storage. Every TCAPI routine that
  writes elements (e.g. `set_elem`) densifies transparently on write.
- **`supported_ten_types_t`** — a `std::tuple` of every concrete tensor
  type the backend provides (`ItensorReal`, `ItensorCplx`).
- **Copy/move contract** — `tcapi::copy` (deep copy) and
  `tcapi::move` (ownership transfer) are the portable alternatives to
  plain copy construction. All library code uses them.
- **iTEBD example** — default coupling changed to `g = 1.0` (critical
  TFIM). Energies now use `-4/π` as the exact reference.

## Supported tensor types

This backend supports two concrete types:
- `tcapi::ItensorReal` (element type `double`)
- `tcapi::ItensorCplx` (element type `std::complex<double>`)

Both are listed in `tcapi::supported_ten_types_t`. Only double precision
is supported; the ITensor v3 library provides `Real` (`double`) and
`Cplx` (`std::complex<double>`) natively.

## Development

Useful commands:

```bash
# Clean build
rm -rf build && cmake -S . -B build && cmake --build build

# Run the complete test suite
ctest --test-dir build --output-on-failure
```

## Related Projects

- [tcapi_numpy](https://github.com/<your-org>/tcapi_numpy): Pure Python/NumPy backend for TCAPI.
- [TCAPI paper (v2)](https://arxiv.org/abs/2512.23917): Tensor Computing Interface specification and benchmarks.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
