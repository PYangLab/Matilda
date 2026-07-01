# Object -> the exact .h5 / .csv layout the vendored python expects.
# Mirrors qc/data_to_h5.R (write_h5 / write_csv) byte-for-byte so that
# util.read_h5_data / read_fs_label read them back identically.

#' Write a genes x cells matrix to Matilda's .h5 layout.
#'
#' Upstream writes \code{writeHDF5Array(t(exprs), "matrix/data")} (cells x
#' features on disk) plus features = rownames, barcodes = colnames;
#' \code{read_h5_data} transposes it back. We replicate that exactly with
#' \pkg{HDF5Array} to avoid the R/HDF5 column-major transpose trap.
#'
#' @param x genes x cells matrix (rownames = features, colnames = cells).
#' @param path output .h5 file.
#' @keywords internal
.write_h5_matilda <- function(x, path) {
  x <- as.matrix(x)
  if (is.null(rownames(x))) rownames(x) <- paste0("feature_", seq_len(nrow(x)))
  if (is.null(colnames(x))) colnames(x) <- paste0("cell_", seq_len(ncol(x)))
  if (file.exists(path)) file.remove(path)
  rhdf5::h5createFile(path)
  rhdf5::h5createGroup(path, "matrix")
  HDF5Array::writeHDF5Array(t(x), path, name = "matrix/data")
  rhdf5::h5write(rownames(x), path, "matrix/features")
  rhdf5::h5write(colnames(x), path, "matrix/barcodes")
  rhdf5::h5closeAll()
  invisible(path)
}

#' Write cell-type labels as the CSV \code{read_fs_label} expects.
#'
#' Mirrors upstream \code{write_csv}: \code{write.csv(labels)} yields a header
#' row plus labels in column index 1, which \code{read_fs_label} reads via
#' \code{iloc[1:, 1]} and factorises alphabetically with \code{pd.Categorical}.
#'
#' @param labels character/factor vector of cell types.
#' @param path output .csv file.
#' @keywords internal
.write_cty_csv <- function(labels, path) {
  labels <- as.character(labels)
  names(labels) <- NULL
  if (file.exists(path)) file.remove(path)
  utils::write.csv(labels, file = path)
  invisible(path)
}
