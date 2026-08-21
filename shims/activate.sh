# Source this before any generate/train run: puts the kernels shim (shims/kernels) on PYTHONPATH.
# See environment/ENVIRONMENT.md "REQUIRED workaround: the kernels shim".
_here="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PYTHONPATH="${_here}:${PYTHONPATH:-}"
