# TCAPI - ITensor Backend

[TCAPI (Tensor Computing API)](https://arxiv.org/abs/2512.23917) is a common tensor-computing interface for tensor-network applications. This repository provides a C++ backend implementation of TCAPI using the [ITensor library](https://itensor.org/).

The project is named `tcapi_itensor`; the C++ namespace is `tcapi`.

## Requirements

- **C++14** or later
- **ITensor** (latest C++14-compatible release)
- **CMake 3.14+** for building
- A C++ compiler with sufficient C++14 support (e.g., GCC 7+, Clang 5+, or recent MSVC)

## Quick Start

### 1. Clone the Repository

```bash
git clone <repo-url>
cd tcapi_itensor
```

### 2. Install Dependencies

Ensure ITensor is installed and discoverable by CMake (e.g., via `CMAKE_PREFIX_PATH` or a system-wide install).

### 3. Build the Project

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

### 4. Run Tests

```bash
ctest --output-on-failure
```

or:

```bash
./test_tcapi
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
| Context and metadata | `create_context`, `destroy_context`, `version`, `ten_kind` |
| Queries | `order`, `shape`, `size`, `size_bytes`, `get_elem` |
| Construction | `allocate`, `zeros`, `assign_from_range`, `fill`, `random`, `eye`, `copy` |
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
    tcapi::tent_t<TenT> product;
    tcapi::contract<TenT>(ctx, identity, "ij", matrix, "jk", product, "ik");
    // product should be numerically close to matrix

    // svd writes its factors via output parameters; sigma is a real diagonal
    // tensor of singular values
    tcapi::tent_t<TenT> u, vdag;
    tcapi::real_ten_t<TenT> sigma;
    tcapi::svd<TenT>(ctx, matrix, 1, u, sigma, vdag);

    tcapi::destroy_context(ctx);
    return 0;
}
```

For tensor decompositions, `num_of_bds_as_row` controls how many leading tensor axes are grouped into the matrix row index before applying ITensor's linear algebra routines.

## Examples

The `example/` directory contains tensor-network demonstrations built on the TCAPI ITensor backend:

- `example/trg.cc`: Levin-Nave TRG for the infinite square-lattice Ising model
- `example/itebd_tfim.cc`: imaginary-time iTEBD for the 1D transverse-field Ising model

Build and run them with:

```bash
cmake --build . --target example_trg
cmake --build . --target example_itebd_tfim

./example_trg
./example_itebd_tfim
```

The iTEBD example is intentionally heavier than the basic unit tests.

## Diagnostics

Set `TCAPI_VERBOSE` to print runtime call diagnostics:

```bash
export TCAPI_VERBOSE=1
./test_tcapi

export TCAPI_VERBOSE=2
./example_itebd_tfim
```

Verbose levels:

- `0`: silent default
- `1`: print TCAPI function names and compact argument summaries
- `2`: also print measured wall-clock time per wrapped TCAPI call

## Development

Useful commands:

```bash
# Configure with debug build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Run the complete test suite
ctest --output-on-failure

# Run a focused test executable
./test_tcapi

# Build examples
cmake --build . --target example_trg
cmake --build . --target example_itebd_tfim
```

## Related Projects

- [tcapi_numpy](https://github.com/<your-org>/tcapi_numpy): Pure Python/NumPy backend for TCAPI.
- [TCAPI paper](https://arxiv.org/abs/2512.23917): Tensor Computing Interface specification and benchmarks.

## License

This project is licensed under the [Apache License 2.0](LICENSE).