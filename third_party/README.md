# Vendored build dependencies (macOS / CPU-only spconv)

spconv has no macOS wheels, and the current upstream `spconv` and `cumm` do not build on macOS. These are source
snapshots of the forks that do. They are plain copies, not submodules, so nothing has to be cloned. On Linux + CUDA,
use the prebuilt `spconv-cu120` wheel instead (see `QUICKSTART.md`).

| Directory | Source | Commit | Licence | Change vs upstream |
|---|---|---|---|---|
| `ccimport/` | [tilmantroester/ccimport](https://github.com/tilmantroester/ccimport) `macos_support` | `58bd65e` | MIT | Links Python extensions with `-undefined dynamic_lookup` on macOS (upstream 0.4.2 + 12 lines). |
| `cumm/` | [tilmantroester/cumm](https://github.com/tilmantroester/cumm) `macos_cpu_only` | `cb47c0d` | Apache-2.0 | `CPU_ONLY=1` at install time writes `cumm/__build_settings__.py` for a CPU-only build (upstream 0.4.7 + 26 lines). |
| `spconv/` | [tilmantroester/spconv](https://github.com/tilmantroester/spconv) `macos_cpu_only` | `e6a5906` | Apache-2.0 | macOS CPU build: compiler flags in `spconv/build.py`, CPU kernels and ops fixes (upstream v2.3.5 + 3 commits). |

Removed from the snapshots to save space: `spconv/test/data/`, `spconv/example/libspconv/benchmark-pc.jarr`,
`spconv/docs/*.pdf`. Nothing else was changed. Each directory keeps its own `LICENSE`.

Install (order matters; `--no-build-isolation` stops pip fetching a CUDA cumm from PyPI to build spconv):

```bash
pip install -e third_party/ccimport --no-deps
CPU_ONLY=1 pip install -e third_party/cumm
CPU_ONLY=1 pip install -e third_party/spconv --no-build-isolation
python -c "import spconv.pytorch"   # first import JIT-compiles the C++ into third_party/*/build (gitignored)
```
