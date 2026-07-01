# Parse the scripts' file outputs back into R objects.

#' Parse per-cell predictions from accuracy_each_cell.txt.
#'
#' Upstream writes one tab-separated line per cell with fields: cell ID,
#' real cell type, predicted cell type, probability. When --cty is absent the
#' "real cell type" field is omitted.
#'
#' @return data.frame(cell, real, predicted, prob)
#' @keywords internal
.parse_classification <- function(path) {
  ln <- readLines(path, warn = FALSE)
  ln <- ln[nzchar(trimws(ln))]
  clean <- function(s) trimws(gsub("[\t]+", "", s))
  predicted <- clean(sub("\\s*probability:.*$", "",
                         sub("^.*predicted cell type:\\s*", "", ln)))
  real <- ifelse(grepl("real cell type:", ln),
                 clean(sub("\\s*predicted cell type:.*$", "",
                           sub("^.*real cell type:\\s*", "", ln))),
                 NA_character_)
  prob <- suppressWarnings(as.numeric(sub("^.*probability:\\s*", "", ln)))
  data.frame(cell = seq_along(ln) - 1L, real = real, predicted = predicted,
             prob = prob, stringsAsFactors = FALSE)
}

#' Read the latent embedding (latent_space.csv): cells x z_dim matrix.
#' @keywords internal
.read_latent <- function(path) {
  df <- utils::read.csv(path, row.names = 1, check.names = FALSE)
  as.matrix(df)
}

#' Read per-cell-type importance tables (fs.celltype_*.csv) into a tidy frame.
#' @return data.frame(celltype, feature, importance)
#' @keywords internal
.read_markers <- function(dir) {
  fs <- list.files(dir, pattern = "^fs\\.celltype_.*\\.csv$", full.names = TRUE)
  if (!length(fs)) return(data.frame(celltype = character(), feature = character(),
                                     importance = numeric()))
  parts <- lapply(fs, function(f) {
    ct <- sub("^fs\\.celltype_(.*)\\.csv$", "\\1", basename(f))
    # Feature names can repeat across modalities (e.g. a gene in RNA and in the
    # ATAC gene-activity), so read the index as a plain column rather than
    # row.names (which must be unique).
    d <- utils::read.csv(f, check.names = FALSE)
    data.frame(celltype = ct, feature = as.character(d[[1]]),
               importance = d[[2]], stringsAsFactors = FALSE)
  })
  out <- do.call(rbind, parts)
  out[order(out$celltype, -out$importance), , drop = FALSE]
}

#' Read simulated + real cells (sim_data_*.csv / real_data_*.csv / *_label.csv).
#' @return list(sim = list(rna,adt,atac,label), real = list(rna,adt,atac,label))
#' @keywords internal
.read_sim <- function(dir) {
  rd <- function(n) {
    p <- file.path(dir, n)
    if (file.exists(p)) t(as.matrix(utils::read.csv(p, row.names = 1, check.names = FALSE))) else NULL
  }
  rl <- function(n) {
    p <- file.path(dir, n)
    if (file.exists(p)) as.character(utils::read.csv(p, row.names = 1)[[1]]) else NULL
  }
  list(
    sim  = list(rna = rd("sim_data_rna.csv"),  adt = rd("sim_data_adt.csv"),
                atac = rd("sim_data_atac.csv"), label = rl("sim_label.csv")),
    real = list(rna = rd("real_data_rna.csv"), adt = rd("real_data_adt.csv"),
                atac = rd("real_data_atac.csv"), label = rl("real_label.csv"))
  )
}
