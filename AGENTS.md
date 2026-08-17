# AGENTS.md — Conventions and AI Workflow for tcapi_itensor

This file encodes project conventions and the intended AI-assisted workflow for the `tcapi_itensor` repository. It is meant to be read by AI coding agents (and human contributors) to ensure consistent, spec-faithful implementations.

## Project Overview

- **Goal:** Implement the C++ TCAPI specification on top of the ITensor library, with semantics matching the TCAPI paper and the Python/NumPy backend (`tcapi_numpy`).
- **Language/Standard:** C++17
- **Build system:** Make (via ITensor v3's `this_dir.mk` / `options.mk`); ITensor itself requires `-std=c++17`
- **Backend:** ITensor v3 (`itensor/` is vendored in-tree)
- **Namespace:** `tcapi`
- **Primary tensor type:** `itensor::ITensor` (dense storage)

## Repository Layout

```text
tcapi_itensor/
├── Makefile
├── README.md
├── AGENTS.md
├── LICENSE
├── itensor/                # vendored ITensor v3 library
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
└── tests/                   # standalone unit test suite (tests/test_*.cc -> binaries in tests/)
    └── tc_test_util.h       # shared test helpers (CHECK, approx, at, ...)
```

The implementation is header-only (templates in `include/tcapi/`); there is no `src/` or CMake build. If you are generating or modifying files, keep this structure unless there is a compelling reason to change it.

## Coding Conventions

### C++17

- The codebase and ITensor require C++17. You may use C++17 features (`if constexpr`, `std::string_view`, structured bindings, etc.).
- Build and run the full test suite with `make test-all` (default `make` builds the standalone test binaries).

### Naming and Signatures

- All public TCAPI functions must:
  - Live in the `tcapi` namespace.
  - Match the function names, argument order, and return types from the TCAPI spec (web specification; Appendix C of the paper).
  - Use template parameter `typename TenT` for the tensor type.
  - Use the traits-forwarding aliases instead of spelling out `typename tensor_traits<TenT>::...`: `ten_t<TenT>`, `order_t<TenT>`, `shape_t<TenT>`, `bond_dim_t<TenT>`, `bond_idx_t<TenT>`, `bond_label_t<TenT>`, `ten_size_t<TenT>`, `elem_t<TenT>`, `elem_coor_t<TenT>`, `elem_coors_t<TenT>`, `real_t<TenT>`, `real_ten_t<TenT>`, `cplx_t<TenT>`, `cplx_ten_t<TenT>`, `context_handle_t<TenT>`.
  - The concrete tensor type is spelled `ten_t<TenT>` (member type `tcapi::tensor_traits<TenT>::ten_t`), **not** `tent_t` or any other name.
  - Take `const context_handle_t<TenT>& ctx` as the first argument.
- Example signature pattern:

  ```cpp
  template<typename TenT>
  void exp(const context_handle_t<TenT>& ctx,
           ten_t<TenT>& inout,
           order_t<TenT> num_of_bds_as_row);
  ```

- Do **not** rename functions to match ITensor's native API (e.g., do not call it `expHermitianWrapper`). The public name must be `exp`.

### Error Handling

- Use `std::invalid_argument` for:
  - Shape mismatches.
  - Invalid `num_of_bds_as_row`.
  - Non-square matrices where required.
- Use `std::runtime_error` for:
  - Backend failures (e.g., ITensor throws, singular matrices in inversion).
- Error messages should be concise and start with the function name, e.g.:

  ```cpp
  throw std::invalid_argument("eigh: row/col index count mismatch.");
  ```

### Context Handling

- Every public TCAPI function must call `ensure_active(ctx)` (or equivalent) as its first statement to validate the context.
- `create_context` and `destroy_context` are the only functions that construct/destroy the underlying context state.

### Comments

- Do **not** write docstrings, documentation comment blocks, or doxygen (`///`) comments in function files. In particular, do not add:
  - `// --- <function> ---` section headers describing the following function.
  - `// Sec. C2x — ...` spec-citation comments.
  - Multi-line description comments above functions or classes.
- Function files contain only code, includes, and `#pragma once` (plus at most a one-line `// tcapi/<name>.h` file marker).
- Keep short inline comments only where a non-obvious implementation detail genuinely needs explanation.

## Test Conventions

- Tests live in the standalone files `tests/test_*.cc`, built by `make` into binaries in `tests/`. Run the full suite with `make test-all`.
- Test naming: `test_<function_name>()`, e.g. `test_exp()`, `test_eigh()`.
- Each test should:
  - Create a context via `create_context`.
  - Call the TCAPI function under test.
  - Use the `CHECK` macro or simple comparisons from `tests/tc_test_util.h` to check numeric results.
  - Destroy the context via `destroy_context`.
  - Be registered in `main()` in the same file.

Where possible, mirror the numeric values and shapes from the `tcapi_numpy` tests so that results are directly comparable across backends.

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

## Future Directions

Possible extensions (not required for the initial implementation):

- Support for symmetry-aware (QN) ITensors.
- GPU execution via ITensor's GPU backends.
- Automatic differentiation support (if/when TCAPI defines an AD API).
- Additional example applications (e.g., DMRG, PEPS, quantum circuit simulation).

These can be added later without changing the core TCAPI interface.