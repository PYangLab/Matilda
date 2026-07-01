#' matilda: R interface to Matilda multi-task single-cell multimodal learning
#'
#' Trains a multimodal VAE + classifier (RNA / ADT / ATAC) and runs
#' classification, dimension reduction, feature selection and simulation. The
#' upstream PyTorch implementation is vendored under \code{inst/python/matilda}
#' and driven via \pkg{reticulate} inside a \pkg{basilisk} environment. Results
#' are written back into the input \link[SingleCellExperiment]{SingleCellExperiment}.
#'
#' @keywords internal
"_PACKAGE"
