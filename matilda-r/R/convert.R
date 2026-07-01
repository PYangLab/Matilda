# Normalise any supported input to a modality list of genes x cells matrices.

#' @keywords internal
.resolve_label <- function(label, cd, n) {
  if (is.null(label)) return(NULL)
  if (length(label) == 1L && is.character(label)) {
    if (!label %in% colnames(cd)) {
      stop("label column '", label, "' not found in colData.")
    }
    return(cd[[label]])
  }
  if (length(label) != n) {
    stop("label vector length (", length(label), ") != number of cells (", n, ").")
  }
  label
}

#' Convert SCE / MAE / Seurat / list to list(rna, adt, atac, cty, mode).
#'
#' SCE: RNA in assay `rna` (default "counts"); ADT/ATAC as altExps named
#' `adt`/`atac`. MAE: experiments named `rna`/`adt`/`atac`, assumed column-aligned.
#'
#' @keywords internal
.as_modalities <- function(x, label, rna = "counts", adt = "ADT", atac = "ATAC") {
  if (inherits(x, "Seurat")) {
    if (!requireNamespace("Seurat", quietly = TRUE)) {
      stop("Seurat input requires the Seurat package.")
    }
    x <- Seurat::as.SingleCellExperiment(x)
  }
  asm <- function(obj, which) as.matrix(SummarizedExperiment::assay(obj, which))

  if (methods::is(x, "SingleCellExperiment")) {
    rmat <- asm(x, rna)
    ae <- SingleCellExperiment::altExpNames(x)
    amat <- if (!is.null(adt)  && adt  %in% ae) asm(SingleCellExperiment::altExp(x, adt),  1L) else NULL
    tmat <- if (!is.null(atac) && atac %in% ae) asm(SingleCellExperiment::altExp(x, atac), 1L) else NULL
    cty  <- .resolve_label(label, SummarizedExperiment::colData(x), ncol(x))
  } else if (methods::is(x, "MultiAssayExperiment")) {
    nm <- names(x)
    rmat <- as.matrix(MultiAssayExperiment::assay(x, rna))
    amat <- if (!is.null(adt)  && adt  %in% nm) as.matrix(MultiAssayExperiment::assay(x, adt))  else NULL
    tmat <- if (!is.null(atac) && atac %in% nm) as.matrix(MultiAssayExperiment::assay(x, atac)) else NULL
    cty  <- .resolve_label(label, MultiAssayExperiment::colData(x), .ncells(x))
  } else if (is.list(x)) {
    rmat <- as.matrix(x$rna)
    amat <- if (!is.null(x$adt))  as.matrix(x$adt)  else NULL
    tmat <- if (!is.null(x$atac)) as.matrix(x$atac) else NULL
    cty  <- label
  } else {
    stop("Unsupported input class: ", paste(class(x), collapse = "/"))
  }

  if (is.null(cty)) {
    stop("No cell-type labels: supply `label` (a colData column name or a vector).")
  }
  if (anyNA(cty)) {
    stop("cell-type labels contain missing values (NA); drop or annotate those cells ",
         "before training -- Matilda has no 'unlabelled' class.")
  }
  list(rna = rmat, adt = amat, atac = tmat, cty = as.character(cty),
       mode = .mode_of(!is.null(amat), !is.null(tmat)))
}
