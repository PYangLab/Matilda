# basilisk environment that bundles the upstream Matilda python dependencies.
#
# Version notes:
#  * Upstream environment_matilda.yaml pins python 3.7 / torch 1.9.1 / cu11.1.
#    torch 1.9.1 cannot drive Ada GPUs (RTX 4090, sm_89), and python 3.7 is EOL,
#    so we move to a modern, still-compatible stack. The model code uses only
#    stable nn / DataLoader / optim APIs, so a newer torch runs it unchanged.
#  * torch from PyPI ships CUDA and falls back to CPU when no GPU is present, so a
#    single pin gives GPU on capable hosts and CPU elsewhere (e.g. Bioconductor
#    build machines). "Same model quality" is validated as bit-identical parity in the
#    SAME env (see inst/scripts/parity_check.R).

#' @importFrom basilisk BasiliskEnvironment
matilda_env <- basilisk::BasiliskEnvironment(
  envname  = "env-matilda",
  pkgname  = "matilda",
  packages = c(           # conda-forge
    "python=3.9",
    "numpy=1.23.5",
    "pandas=1.5.3",
    "scipy=1.10.1",
    "h5py=3.8.0"
  ),
  pip = c(                # PyPI (torch ships CUDA; CPU fallback when no GPU)
    "torch==2.1.2",
    "torchvision==0.16.2",
    "captum==0.7.0",
    "scanpy==1.9.6",
    "tqdm==4.66.1"
  )
)
