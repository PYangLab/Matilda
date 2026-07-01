# The lightweight S3 model handle, stored inside the object (Seurat-style).

#' @keywords internal
new_matilda_model <- function(checkpoint, mode, label_levels, features, dims, hyper,
                              label_col = NULL) {
  structure(list(checkpoint = checkpoint, mode = mode, label_levels = label_levels,
                 features = features, dims = dims, hyper = hyper, label_col = label_col),
            class = "matilda_model")
}

#' Print a Matilda model handle.
#' @param x a \code{matilda_model}.
#' @param ... unused.
#' @return \code{x}, invisibly.
#' @examples
#' \donttest{
#'   sce <- matilda_train(matilda_example_sce(), label = "cell_type", epochs = 2L)
#'   print(matilda_model(sce))
#' }
#' @export
print.matilda_model <- function(x, ...) {
  cat("Matilda model (", x$mode, ")\n", sep = "")
  k <- length(x$label_levels)
  cat("  cell types : ", k, " [",
      paste(utils::head(x$label_levels, 4), collapse = ", "),
      if (k > 4) ", ..." else "", "]\n", sep = "")
  cat("  latent dim : ", x$dims$z_dim, "\n", sep = "")
  cat("  features   : rna=", x$dims$nfeatures_rna,
      if (!is.null(x$dims$nfeatures_adt))  paste0(" adt=",  x$dims$nfeatures_adt)  else "",
      if (!is.null(x$dims$nfeatures_atac)) paste0(" atac=", x$dims$nfeatures_atac) else "",
      "\n", sep = "")
  cat("  trained    : epochs=", x$hyper$epochs, " seed=", x$hyper$seed, "\n", sep = "")
  invisible(x)
}

#' Extract the Matilda model stored in (or passed as) an object.
#'
#' @param object a \link[SingleCellExperiment]{SingleCellExperiment} /
#'   \link[MultiAssayExperiment]{MultiAssayExperiment} trained with
#'   \code{\link{matilda_train}}, or a \code{matilda_model} itself.
#' @return a \code{matilda_model}, or \code{NULL} if none is stored.
#' @examples
#' sce <- matilda_example_sce()
#' \donttest{
#'   sce <- matilda_train(sce, label = "cell_type", epochs = 2L)
#'   matilda_model(sce)
#' }
#' @export
matilda_model <- function(object) {
  if (inherits(object, "matilda_model")) return(object)
  if (methods::is(object, "SummarizedExperiment") ||
      methods::is(object, "MultiAssayExperiment")) {
    return(S4Vectors::metadata(object)$matilda)
  }
  NULL
}

#' @keywords internal
.store_model <- function(object, model) {
  S4Vectors::metadata(object)$matilda <- model
  object
}

#' Resolve which model a task should use: explicit reference, else the object's own.
#' @keywords internal
.resolve_model <- function(x, reference) {
  if (!is.null(reference)) {
    m <- matilda_model(reference)
    if (is.null(m)) stop("`reference` holds no trained Matilda model.")
    return(m)
  }
  m <- matilda_model(x)
  if (is.null(m)) {
    stop("No trained Matilda model found. Train first with matilda_train(), ",
         "or pass reference = <a trained object or model>.")
  }
  m
}
