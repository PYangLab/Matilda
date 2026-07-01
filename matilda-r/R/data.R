#' Locate the TEA-seq demo dataset (train/test .h5 + label .csv).
#'
#' Resolves a local copy of the demo via \code{options(matilda.demo=)} or the
#' \code{MATILDA_DEMO} environment variable. To download and cache the public
#' demo data, use \code{\link{matilda_download_example}}.
#'
#' @param dir directory holding the demo files; \code{NULL} auto-resolves.
#' @return a named character vector of the eight file paths.
#' @examples
#' demo_dir <- getOption("matilda.demo")  # where the demo is resolved from
#' \donttest{
#'   paths <- matilda_example_teaseq()
#' }
#' @export
matilda_example_teaseq <- function(dir = NULL) {
  if (is.null(dir)) {
    dir <- getOption("matilda.demo",
                     Sys.getenv("MATILDA_DEMO", ""))
  }
  files <- c(train_rna = "train_rna.h5", train_adt = "train_adt.h5",
             train_atac = "train_atac.h5", train_cty = "train_cty.csv",
             test_rna = "test_rna.h5", test_adt = "test_adt.h5",
             test_atac = "test_atac.h5", test_cty = "test_cty.csv")
  paths <- stats::setNames(file.path(dir, files), names(files))
  miss <- !file.exists(paths)
  if (any(miss)) {
    stop("TEA-seq demo files not found in '", dir, "': ",
         paste(files[miss], collapse = ", "),
         ". Set options(matilda.demo=) or MATILDA_DEMO.")
  }
  paths
}

#' A tiny multimodal SingleCellExperiment for examples and quick trials.
#'
#' Builds a small synthetic RNA + ADT + ATAC \link[SingleCellExperiment]{SingleCellExperiment}
#' with imbalanced cell types, suitable for trying the Matilda workflow.
#'
#' @param n_cells number of cells.
#' @return a SingleCellExperiment with altExps "ADT"/"ATAC" and a colData "cell_type".
#' @examples
#' sce <- matilda_example_sce()
#' sce
#' @export
matilda_example_sce <- function(n_cells = 60) {
  mk <- function(nf, lam, pre) {
    matrix(stats::rpois(nf * n_cells, lam), nf, n_cells,
           dimnames = list(paste0(pre, seq_len(nf)), paste0("cell", seq_len(n_cells))))
  }
  rna <- mk(50, 5, "gene"); adt <- mk(8, 8, "adt"); atac <- mk(40, 3, "peak")
  sizes <- c(round(n_cells * 0.5), round(n_cells * 0.33))
  sizes <- c(sizes, n_cells - sum(sizes))
  ct <- factor(rep(c("A", "B", "C"), times = sizes))
  sce <- SingleCellExperiment::SingleCellExperiment(assays = list(counts = rna))
  SingleCellExperiment::altExp(sce, "ADT") <-
    SummarizedExperiment::SummarizedExperiment(list(counts = adt))
  SingleCellExperiment::altExp(sce, "ATAC") <-
    SummarizedExperiment::SummarizedExperiment(list(counts = atac))
  SummarizedExperiment::colData(sce)$cell_type <- ct
  sce
}
