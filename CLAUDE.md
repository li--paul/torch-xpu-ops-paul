# torch-xpu-ops — Agent Guide

## What this is

torch-xpu-ops provides XPU (Intel GPU) operator implementations for PyTorch ATen.
It is a **submodule** of PyTorch (`third_party/torch-xpu-ops`) — no standalone build.
Most operators are SYCL kernels; linear algebra uses oneMKL; conv/gemm use oneDNN
(live in `aten/src/ATen/native/mkldnn/xpu/` in the PyTorch repo, not here).

## Before touching any operator

Read the corresponding upstream PyTorch source. Do NOT guess from memory:

| This file | Upstream reference |
|-----------|------------------|
| `src/ATen/native/xpu/<Op>.cpp` | `aten/src/ATen/native/<Op>.cpp` |
| `src/ATen/native/xpu/sycl/<Op>Kernels.cpp` | `aten/src/ATen/native/cuda/<Op>.cu` |
| Dispatch registration | `aten/src/ATen/native/native_functions.yaml` |

Use `gh`, WebFetch, or sub-agents to pull files from `pytorch/pytorch`.

## Repository layout

```
src/ATen/native/xpu/          # Dispatch wrappers only (no SYCL)
src/ATen/native/xpu/sycl/     # Kernel implementations
src/ATen/native/nested/xpu/   # Nested tensor XPU
src/comm/                     # Shared utility headers
src/xccl/                     # XCCL distributed backend
test/xpu/                     # Pytest-based XPU tests
test/regressions/             # Regression tests
test/sycl/                    # SYCL C++ unit tests (CMake, BUILD_TEST=ON)
test/repro/                   # Mandatory reproducer dir for AI-generated PRs
tools/linter/                 # Lint adapter scripts
agent_space_xpu/              # Git-ignored scratch directory
```

XPU dispatch keys: `XPU`, `SparseXPU`, `SparseCsrXPU`, `NestedTensorXPU`.

## Build

Only builds as part of PyTorch. From the PyTorch root:

```bash
pip install -e . -v --no-build-isolation
```

Key env vars: `USE_XPU=ON` (default), `USE_XCCL=ON` (default >= PT2.8),
`TORCH_XPU_ARCH_LIST` (comma-sep arch list, omit → JIT compile at runtime).

**Dev pin override**: PyTorch pins torch-xpu-ops in `third_party/xpu.txt`.
To work on your PR branch:

```bash
cd <pytorch_root>/third_party
git clone <your-fork> torch-xpu-ops
cd torch-xpu-ops && git checkout <branch>
git rev-parse HEAD > ../xpu.txt   # local only, do NOT commit
```

**Debug/RelWithDebInfo** auto-sets `BUILD_SEPARATE_OPS=1` (faster TU-scoped builds)
and disables SYCL-TLA (excessive memory with debug info).

## Test

Run via **pytest** (not `python -m pytest`). Standard CI options:
`-v --timeout 600 --timeout_method=thread --dist worksteal`.

```bash
pytest test/xpu/test_ops_xpu.py -v -k "test_foo"
pytest test/regressions/
cd test/xpu && python run_test_with_skip.py  # CI-style with skip lists
```

To verify XPU availability: `python -c "import torch; print(torch.xpu.is_available())"`.

**AI-generated PRs must include** either `test/repro/test_*.py` files (pytest format)
or pytest commands in the PR body (lines starting with `pytest`).

## Lint

```bash
lintrunner init                          # first time
lintrunner -a -m origin/main             # auto-fix changed files
lintrunner --skip CLANGTIDY,CLANGFORMAT  # skip clang linters for speed
```

Active linters: FLAKE8, RUFF, CLANGFORMAT, CLANGTIDY, MYPY, SHELLCHECK, CMAKE,
plus custom grep checks (quoted includes, deprecated macros, trailing spaces, etc.).

Before committing: run `lintrunner -a`.

## C++ code conventions

**File split**: dispatch in `src/ATen/native/xpu/<Op>.cpp`, kernel in
`src/ATen/native/xpu/sycl/<Op>Kernels.{cpp,h}`. Dispatch files use `REGISTER_XPU_DISPATCH`.

**Includes**: `#include <...>` always (never quotes). Order: ATen → ATen ops →
SYCL headers → comm. Use `#pragma once`.

**Namespaces**: `namespace at::native::xpu { ... }` in kernels,
`namespace at { namespace native { ... } }` in dispatchers. Always add closing comment.

**SYCL functor pattern** (lambdas don't work with SYCL):

```cpp
template <typename opmath_t>
struct MyFunctor {
  opmath_t operator()(opmath_t a, opmath_t b) const { return a + alpha_ * b; }
  MyFunctor(opmath_t alpha) : alpha_(alpha) {}
private:
  opmath_t alpha_;
};

void my_kernel(TensorIteratorBase& iter, const Scalar& alpha) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kHalf, kBFloat16, kBool, iter.common_dtype(), "my_xpu", [&]() {
        using opmath_t = opmath_type<scalar_t>;
        opmath_gpu_kernel_with_scalars<scalar_t>(
            iter, MyFunctor<opmath_t>(alpha.to<opmath_t>()));
      });
}
```

**Error handling**: `TORCH_CHECK` for input validation, `TORCH_INTERNAL_ASSERT` for
invariants. No `AT_ERROR`.

**Lint-enforced replacements**:
- `[[maybe_unused]]` not `C10_UNUSED`
- `[[nodiscard]]` not `C10_NODISCARD`
- `c10::call_once`/`c10::once_flag` not `std::`

## Python code conventions

- Max line length: 120 (ruff/flake8). Formatter: ruff-format + usort (test files).
- Test files wrap upstream PyTorch tests via `XPUPatchForImport` and `instantiate_device_type_tests`.
- `type: ignore` must be qualified: `# type: ignore[attr-defined]`.
- `# noqa` must be qualified: `# noqa: F401`.
- Use `torch._dynamo.config.patch` as decorator/context manager (not manual save/restore).
- Fix B950 in multi-line strings by putting `# noqa: B950` on the terminating `"""` line.

## Git workflow

**ghstack** is used. Identify via detached HEAD or `ghstack-source-id` trailer.
- Don't amend or push directly unless asked.
- `ghstack --no-stack` for single-commit PRs; full `ghstack` for stacks.
- Preserve `Pull-Request:` and `ghstack-source-id:` trailers on rewrite.
- Disclose AI-assisted authorship in commit messages.

## CI labels (used in `pull.yml`)

Set via PR labels: `disable_all`, `disable_ut`, `disable_e2e`, `disable_distributed`,
`disable_win`, `disable_build`, `windows_ci`, `ai_generated` (enables reproducer check).

## Gotchas

- `.ci/docker/` changes trigger full Docker rebuild (content-hashed). Avoid touching it
  when builds are broken upstream.
- **SYCL-TLA** only works on Linux with GCC >= 13.0. Disabled in Debug/RelWithDebInfo builds.
- `$IS_XPU_CI=1` enables `-Werror=unused-variable` in CMake.
- License header required: `Copyright 2020-2026 Intel Corporation` + Apache 2.0.
- Skills available under `.claude/skills/` — use them for PR review, issue handling,
  build setup, ASM extraction, and more.

## Related instruction files

- `CLAUDE.md` — verbose copy of this file (legacy, may drift)
- `.github/copilot-instructions.md` — points to `CLAUDE.md`
- `.github/instructions/` — Copilot-specific rules per path pattern
