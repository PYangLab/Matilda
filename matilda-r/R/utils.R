# Small internal helpers shared across the package.

#' Path to the vendored python package shipped in inst/python/matilda.
#' @keywords internal
.pkg_py <- function() {
  p <- system.file("python", "matilda", package = "matilda")
  if (!nzchar(p)) stop("vendored python not found (inst/python/matilda)")
  p
}

#' Matilda mode from which modalities are present (mirrors the CLI's NULL logic).
#' @keywords internal
.mode_of <- function(has_adt, has_atac) {
  if (has_adt && has_atac) "TEAseq"
  else if (!has_adt && has_atac) "SHAREseq"
  else if (has_adt && !has_atac) "CITEseq"
  else "rna_only"
}

#' Number of cells in any supported input.
#' @keywords internal
.ncells <- function(x) {
  if (methods::is(x, "SummarizedExperiment")) ncol(x)
  else if (methods::is(x, "MultiAssayExperiment")) nrow(MultiAssayExperiment::colData(x))
  else if (is.list(x) && !is.null(x$rna)) ncol(as.matrix(x$rna))
  else ncol(as.matrix(x))
}

#' Feature names stored in a Matilda .h5 (matrix/features).
#' @keywords internal
.h5_features <- function(path) as.character(rhdf5::h5read(path, "matrix/features"))
