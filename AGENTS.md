# AGENTS.md — Conventions and AI Workflow for tcapi_itensor

This file encodes project conventions and the intended AI-assisted workflow for the `tcapi_itensor` repository. It is meant to be read by AI coding agents (and human contributors) to ensure consistent, spec-faithful implementations.

## Project Overview

- **Goal:** Implement the C++ TCAPI specification on top of the ITensor library, with semantics matching the TCAPI paper and the Python/NumPy backend (`tcapi_numpy`).
- **Language/Standard:** C++14
- **Build system:** CMake 3.14+
- **Backend:** ITensor (latest C++14-compatible release)
- **Namespace:** `tcapi`
- **Primary tensor type:** `itensor::ITensor` (dense storage)

## Repository Layout

```text
tcapi_itensor/
├── CMakeLists.txt
├── README.md
├── AGENTS.md
├── LICENSE
├── include/
│   └── tcapi/
│       ├── tcapi.h          # Main public header (function declarations, templates)
│       ├── context.h        # Context handle and management
│       ├── dispatch.h       # Type traits and aliases (elem_t, shape_t, etc.)
│       └── ...              # Other public headers as needed
├── src/
│   ├── context.cc
│   ├── construct.cc         # allocate, zeros, fill, eye, random, etc.
│   ├── core.cc              # reshape, transpose, concatenate, etc.
│   ├── linalg.cc            # contract, svd, eig, eigh, exp, etc.
│   ├── combine.cc           # linear_combine, stack
│   ├── io.cc                # save, load
│   └── utils.cc             # show, close, convert, to_range, diagnostics
├── tests/
│   ├── CMakeLists.txt
│   ├── test_construct.cc
│   ├── test_core.cc
│   ├── test_linalg.cc
│   ├── test_combine.cc
│   └── test_main.cc         # test harness / main()
└── example/
    ├── CMakeLists.txt
    ├── trg.cc
    └── itebd_tfim.cc
```

If you are generating or modifying files, keep this structure unless there is a compelling reason to change it.

## Coding Conventions

### Naming and Signatures

- All public TCAPI functions must:
  - Live in the `tcapi` namespace.
  - Match the function names and argument order from the TCAPI spec (Appendix C of the paper).
  - Use template parameter `typename TenT` for the tensor type.
  - Take `context_handle_t<TenT>& ctx` as the first argument.
- Example signature pattern:

  ```cpp
  template<typename TenT>
  void exp(const context_handle_t<TenT>& ctx,
           tent_t<TenT>& inout,
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

### Documentation

- Every public function must have a doc comment that:
  - Briefly describes the operation.
  - Cites the relevant TCAPI section (e.g., "Sec. C2e — Hermitian eigensolver").
  - Notes any ITensor-specific implementation details or deviations (if any).

Example:

```cpp
// --- exp -----------------------------------------------------------------
// Sec. C2e — matrix exponential via Hermitian eigendecomposition.
// For ITensor backend, this uses itensor::expHermitian internally.
template<typename TenT>
void exp(const context_handle_t<TenT>& ctx,
         tent_t<TenT>& inout,
         order_t<TenT> num_of_bds_as_row);
```

## Test Conventions

- Tests live in `tests/` and are built into a single `test_tcapi` executable (or one executable per test file if you prefer).
- Each TCAPI function category has a corresponding test file:
  - `test_construct.cc` → `allocate`, `zeros`, `fill`, `eye`, `random`, `copy`.
  - `test_core.cc` → `reshape`, `transpose`, `concatenate`, etc.
  - `test_linalg.cc` → `contract`, `svd`, `eig`, `eigh`, `exp`, etc.
  - `test_combine.cc` → `linear_combine`, `stack`.
- Test naming: `test_<function_name>()`, e.g. `test_exp()`, `test_eigh()`.
- Each test should:
  - Create a context via `create_context`.
  - Call the TCAPI function under test.
  - Use `assert` or a simple test macro to check numeric results.
  - Destroy the context via `destroy_context`.
  - Print `"<test_name> passed\n"` on success.

Where possible, mirror the numeric values and shapes from the `tcapi_numpy` tests so that results are directly comparable across backends.

## AI-Assisted Porting Workflow

This repository is part of a broader effort to use AI agents for cross-language porting of tensor-network code. The high-level workflow is:

1. **Inputs to the AI system**
   - Source implementation: e.g., `tcapi_numpy` (Python/NumPy).
   - Target spec: TCAPI function list and signatures (Appendix C of the paper).
   - Target backend docs: ITensor C++ API documentation.
   - This `AGENTS.md` file (conventions, layout, test style).

2. **Typical AI tasks**

   When asked to implement or update a function, the AI should:

   - Read the corresponding NumPy implementation (if available).
   - Read the TCAPI spec signature and semantics.
   - Generate a C++ implementation that:
     - Matches the TCAPI signature exactly.
     - Uses ITensor primitives internally but does not expose ITensor-specific naming or semantics in the public API.
     - Follows the conventions in this file (naming, errors, docs).
   - Generate a matching test function in the appropriate `test_*.cc` file, mirroring the NumPy test logic.

   Example prompt pattern:

   > "Implement `tcapi::eigh` for the ITensor backend. Use the Python/NumPy implementation in `tcapi_numpy/linalg.py` and the TCAPI spec (Appendix C2e) as references. Follow the conventions in `AGENTS.md`: same function signature, error messages starting with 'eigh:', and a test in `test_linalg.cc` that mirrors `test_eigh` from the NumPy test suite."

3. **Human role**

   - Define the spec and conventions (this file).
   - Review AI-generated code for:
     - Correctness (tests pass).
     - Faithfulness to TCAPI semantics (no silent changes in behavior).
     - Consistency with existing code style.
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
- Don't add heavy dependencies beyond ITensor and standard C++ without discussion.

## Future Directions

Possible extensions (not required for the initial implementation):

- Support for symmetry-aware (QN) ITensors.
- GPU execution via ITensor's GPU backends.
- Automatic differentiation support (if/when TCAPI defines an AD API).
- Additional example applications (e.g., DMRG, PEPS, quantum circuit simulation).

These can be added later without changing the core TCAPI interface.