# A tiny deterministic multimodal SCE for pure-R + fast integration tests.
# Classes are intentionally IMBALANCED (~50/33/17%) so the upstream
# median-balancing augmentation has a single, deterministic anchor cell type.
toy_sce <- function(n_rna = 40, n_adt = 6, n_atac = 30, n_cells = 24, seed = 1) {
  set.seed(seed)
  rna <- matrix(stats::rpois(n_rna * n_cells, 5), n_rna, n_cells,
                dimnames = list(paste0("gene", seq_len(n_rna)),
                                paste0("cell", seq_len(n_cells))))
  adt <- matrix(stats::rpois(n_adt * n_cells, 8), n_adt, n_cells,
                dimnames = list(paste0("adt", seq_len(n_adt)), colnames(rna)))
  atac <- matrix(stats::rpois(n_atac * n_cells, 3), n_atac, n_cells,
                 dimnames = list(paste0("peak", seq_len(n_atac)), colnames(rna)))
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
