# The TEA-seq demo dataset is ~75 MB (too large to ship inside the package), so
# the tutorial downloads it on first use and caches it locally.

# Download URL of the demo tarball; override with options(matilda.demo_url = ).
.matilda_demo_url <- "https://github.com/PYangLab/Matilda/releases/download/demo-data/matilda_teaseq_demo.tar.gz"

#' Download the TEA-seq demo dataset used by the tutorial.
#'
#' Downloads a small tarball — the native Matilda \code{.h5} train/test files plus
#' the same training data as \code{.h5ad}, 10x and a Seurat \code{.rds} — caches it
#' under the user cache directory, and returns the local directory. Re-running uses
#' the cache instead of downloading again.
#'
#' @param dest cache directory (default: a per-user cache dir).
#' @param url download URL of the tarball (default: \code{getOption("matilda.demo_url")}).
#' @param force re-download even if the data is already cached.
#' @return Path to the unpacked \code{matilda_teaseq_demo} directory: the native
#'   \code{.h5}/\code{.csv} files plus a \code{formats/} subfolder
#'   (\code{.h5ad}, 10x, \code{.rds}).
#' @examples
#' \donttest{
#'   data_dir <- matilda_download_example()
#'   list.files(data_dir)
#' }
#' @export
matilda_download_example <- function(dest = tools::R_user_dir("matilda", "cache"),
                                     url = getOption("matilda.demo_url", .matilda_demo_url),
                                     force = FALSE) {
  data_dir <- file.path(dest, "matilda_teaseq_demo")
  if (dir.exists(data_dir) && !force) return(data_dir)
  dir.create(dest, recursive = TRUE, showWarnings = FALSE)
  tgz <- file.path(dest, "matilda_teaseq_demo.tar.gz")
  message("Downloading the Matilda TEA-seq demo (~75 MB) to ", dest, " ...")
  utils::download.file(url, tgz, mode = "wb")
  utils::untar(tgz, exdir = dest)
  unlink(tgz)
  data_dir
}
