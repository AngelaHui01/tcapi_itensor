# AGENTS.md — Conventions and AI Workflow for tcapi_itensor

This file encodes project conventions and the intended AI-assisted workflow for the `tcapi_itensor` repository. It is meant to be read by AI coding agents (and human contributors) to ensure consistent, spec-faithful implementations.

## Project Overview

- **Goal:** Implement the C++ TCAPI specification on top of the ITensor library, with semantics matching the TCAPI paper and the Python/NumPy backend (`tcapi_numpy`).
- **Language/Standard:** C++17
- **Build system:** CMake (TCAPI project); ITensor is built separately by its own Make workflow and linked as an external dependency
- **Backend:** ITensor v3 (`itensor/` is vendored in-tree, pre-built libs in `itensor/lib`)
- **Namespace:** `tcapi`
- **Primary tensor type:** `itensor::ITensor` (dense storage)

## Repository Layout

```text
tcapi_itensor/
├── CMakeLists.txt
├── README.md
├── AGENTS.md
├── LICENSE
├── itensor/                # vendored ITensor v3 library (built via its own Make workflow)
├── include/
│   └── tcapi/
│       ├── tcapi.h          # single public header; include this to get all of tcapi
│       ├── type_system.h    # auxiliary types, tensor_traits<TenT>, aliases, context
│       ├── detail.h         # internal helpers (ensure_active, coordinate iteration)
│       ├── queries.h        # order, shape, size, size_bytes, get_elem
│       ├── constructors.h   # allocate, zeros, assign_from_range, random, eye, fill, copy, move, clear
│       ├── io.h             # save, load
│       ├── manipulation.h   # set_elem, reshape, transpose, cplx_conj, to_cplx, real, imag,
│       │                    # expand, shrink, extract_sub, replace_sub, concatenate, stack,
│       │                    # for_each, for_each_with_coors
│       ├── linalg.h         # contract, diag, norm, normalize, scale, trace, linear_combine,
│       │                    # exp, inverse, svd, trunc_svd, qr, lq, eig, eigvals, eigh, eigvalsh
│       ├── misc.h           # create_context, destroy_context, version, show, close, convert, to_range
│       └── diagnostics.h    # TCAPI_VERBOSE environment-variable diagnostics
├── examples/               # examples/trg.cc (TRG for 2D Ising) -> binary build/trg
└── tests/                  # test_<area>.cc + tc_test_util.h; binaries built into build/
```

The implementation is header-only (templates in `include/tcapi/`); there is no `src/`.

## Build & Test

- `cmake -S . -B build` — configure the CMake project out-of-source into `build/`.
- `cmake --build build` — builds all test binaries AND the `trg` example.
- `ctest --test-dir build --output-on-failure` — runs every registered test. All tests currently pass.
- Single test: `cmake --build build --target test_linalg && ./build/test_linalg`.
- `rm -rf build` — clean build.

### Build gotchas

- ITensor is an external dependency: it is built with its own Make workflow (`make -C itensor`), and the TCAPI CMake project links against the pre-built `itensor/lib/libitensor.a`. Configure fails with a hint if that library is missing.
- CMake builds strictly out-of-source into `build/` (gitignored); nothing is written into `tests/` or `examples/`. Do not stage `build/` or any `*.o` files in commits.

## Coding Conventions

### Naming and Signatures

- All public TCAPI functions must:
  - Live in the `tcapi` namespace.
  - Match the function names, argument order, and return types from the TCAPI spec (web specification; Appendix C of the paper).
  - Use template parameter `typename TenT` for the tensor type.
  - Use the traits-forwarding aliases instead of spelling out `typename tensor_traits<TenT>::...`: `ten_t<TenT>`, `order_t<TenT>`, `shape_t<TenT>`, `bond_dim_t<TenT>`, `bond_idx_t<TenT>`, `bond_label_t<TenT>`, `ten_size_t<TenT>`, `elem_t<TenT>`, `elem_coor_t<TenT>`, `elem_coors_t<TenT>`, `real_t<TenT>`, `real_ten_t<TenT>`, `cplx_t<TenT>`, `cplx_ten_t<TenT>`, `context_handle_t<TenT>`.
  - The concrete tensor type is spelled `ten_t<TenT>` (member type `tcapi::tensor_traits<TenT>::ten_t`), **not** `tent_t` or any other name.
  - Take `const context_handle_t<TenT>& ctx` as the first argument.
- Do **not** rename functions to match ITensor's native API (e.g., do not call it `expHermitianWrapper`). The public name must be `exp`.
- Example signature pattern:

  ```cpp
  template<typename TenT>
  void exp(const context_handle_t<TenT>& ctx,
           ten_t<TenT>& inout,
           order_t<TenT> num_of_bds_as_row);
  ```

### Context Handling

- Every public TCAPI function must call `ensure_active(ctx)` as its first statement (it throws `std::runtime_error` on a destroyed context).
- `create_context` and `destroy_context` are the only functions that construct/destroy the underlying context state.

### Error Handling

- Use `std::invalid_argument` for shape mismatches, invalid `num_of_bds_as_row`, non-square matrices where required.
- Use `std::runtime_error` for backend failures (e.g., ITensor throws, singular matrices in inversion, zero tensor in normalize).
- Error messages start with the function name, e.g. `throw std::invalid_argument("eigh: matricized tensor must be square.");`

### Comments

- Do **not** write docstrings, documentation comment blocks, or doxygen (`///`) comments in function files. In particular, do not add:
  - `// --- <function> ---` section headers describing the following function.
  - `// Sec. C2x — ...` spec-citation comments.
  - Multi-line description comments above functions or classes.
- Function files contain only code, includes, and `#pragma once` (plus at most a one-line `// tcapi/<name>.h` file marker).
- Keep short inline comments only where a non-obvious implementation detail genuinely needs explanation.

## Test Conventions

- Tests live in the standalone files `tests/test_*.cc` (`test_constructors`, `test_contract`, `test_manipulation`, `test_linalg`, `test_io_misc`, `test_errors`, `test_examples`), built by the CMake project into binaries in `build/`.
- Test function naming: `test_<function_name>()`, e.g. `test_exp()`, `test_eigh()`.
- Each test should:
  - Create a context via `create_context`.
  - Call the TCAPI function under test.
  - Use helpers from `tests/tc_test_util.h`: `CHECK`, `CHECK_APPROX`, `CHECK_THROW`, `CHECK_SHAPE`, `CHECK_ALL_CLOSE`, `approx`, `all_close`, `at`.
  - Destroy the context via `destroy_context`.
  - Be registered in `main()` in the same file via `tc_test::run_test(...)`.
- Where possible, mirror the numeric values and shapes from the `tcapi_numpy` tests so that results are directly comparable across backends.

## AI-Assisted Porting Workflow

This repository is part of a broader effort to use AI agents for cross-language porting of tensor-network code. The high-level workflow is:

1. **Inputs to the AI system**
   - Source implementation: e.g., `tcapi_numpy` (Python/NumPy).
   - Target spec: TCAPI function list and signatures (the web specification, https://tensorcomputingapi.github.io/, and Appendix C of the paper).
   - Target backend docs: ITensor v3 C++ API documentation.
   - This `AGENTS.md` file (conventions, layout, test style, comment rules).

2. **Typical AI tasks**

   When asked to implement or update a function, the AI should:

   - Read the corresponding NumPy implementation (if available).
   - Read the TCAPI spec signature and semantics.
   - Generate a C++ implementation that:
     - Matches the TCAPI signature exactly (names, argument order, return types).
     - Uses ITensor primitives internally but does not expose ITensor-specific naming or semantics in the public API.
     - Follows the conventions in this file (naming, errors, comment rules).
   - Generate a matching `test_<fn>()` in `tests/test_<area>.cc`, mirroring the NumPy test logic.

   Example prompt pattern:

   > "Implement `tcapi::eigh` for the ITensor backend. Use the Python/NumPy implementation in `tcapi_numpy/linalg.py` and the TCAPI spec (web spec / Appendix C2e) as references. Follow the conventions in `AGENTS.md`: same function signature spelled with `ten_t<TenT>` and the traits aliases, error messages starting with 'eigh:', no doc comments, and a `test_eigh()` in `tests/test_linalg.cc` that mirrors `test_eigh` from the NumPy test suite."

3. **Human role**

   - Define the spec and conventions (this file).
   - Review AI-generated code for:
     - Correctness (tests pass).
     - Faithfulness to TCAPI semantics (no silent changes in behavior).
     - Consistency with existing code style (including the no-docstrings rule).
   - Capture successful patterns (prompt templates, checklists) back into this file or a separate `docs/ai_workflow.md`.

4. **Desired outcome**

   - A complete, tested C++ ITensor backend for TCAPI.
   - A documented, reproducible workflow for AI-assisted porting between Python/NumPy and C++/ITensor.
   - A template that other physicists can adapt for porting their own tensor codes across languages/frameworks.

## Dos and Don'ts

### Dos

- Do match TCAPI function signatures from the spec exactly.
- Do call `ensure_active(ctx)` at the start of every public function.
- Do write tests that mirror the NumPy backend's tests where possible.
- Do keep ITensor-specific details internal; the public API should look like TCAPI, not ITensor.
- Do update this file if you discover new conventions or patterns that should be enforced.

### Don'ts

- Don't change function names or argument order to "fit ITensor better."
- Don't silently alter semantics (e.g., changing how `num_of_bds_as_row` is interpreted).
- Don't introduce backend-specific behavior that breaks compatibility with the TCAPI spec.
- Don't add docstrings or documentation comment blocks to function files.
- Don't add heavy dependencies beyond ITensor and standard C++ without discussion.
- Don't commit build artifacts (`build/`, `*.o` files) without explicit request.
