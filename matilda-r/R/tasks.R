# Downstream tasks: data-first, object-carries-model, results written back.

#' @keywords internal
.is_se <- function(x) {
  methods::is(x, "SummarizedExperiment") || methods::is(x, "MultiAssayExperiment")
}

#' Shared task runner: stage a run dir, seed the model, write inputs, run the script.
#' @keywords internal
.run_task <- function(model, x, label, flags, query, assay, adt_exp, atac_exp, device) {
  n <- .ncells(x)
  lab <- if (is.null(label)) rep(model$label_levels[1], n) else label
  mods <- .as_modalities(x, label = lab, rna = assay, adt = adt_exp, atac = atac_exp)
  rundir <- .stage_rundir(); .seed_rundir_model(rundir, model)
  td <- file.path(rundir, "data")
  pth <- function(nm) file.path(td, paste0(nm, ".h5"))
  .write_h5_matilda(mods$rna, pth("rna"))
  if (!is.null(mods$adt))  .write_h5_matilda(mods$adt,  pth("adt"))
  if (!is.null(mods$atac)) .write_h5_matilda(mods$atac, pth("atac"))
  ctyf <- file.path(td, "cty.csv"); .write_cty_csv(mods$cty, ctyf)
  args <- c("--rna", pth("rna"), "--cty", ctyf,
            if (!is.null(mods$adt))  c("--adt",  pth("adt")),
            if (!is.null(mods$atac)) c("--atac", pth("atac")),
            if (query) c("--query", "True"),
            "--z_dim", model$dims$z_dim, "--hidden_rna", model$dims$hidden_rna,
            if (!is.null(model$dims$nfeatures_adt))  c("--hidden_adt",  model$dims$hidden_adt),
            if (!is.null(model$dims$nfeatures_atac)) c("--hidden_atac", model$dims$hidden_atac),
            "--seed", model$hyper$seed, flags)
  script <- if (model$mode == "rna_only") "main_matilda_rna_task.py" else "main_matilda_task.py"
  .matilda_run(script, as.character(args), rundir, device = device)
  list(rundir = rundir, out = file.path(rundir, "output"),
       sub = if (query) "query" else "reference")
}

#' @keywords internal
.write_back_pred <- function(x, pred, prob) {
  if (methods::is(x, "SummarizedExperiment")) {
    SummarizedExperiment::colData(x)$matilda_pred <- pred
    SummarizedExperiment::colData(x)$matilda_prob <- prob
    return(x)
  }
  if (methods::is(x, "MultiAssayExperiment")) {
    MultiAssayExperiment::colData(x)$matilda_pred <- pred
    MultiAssayExperiment::colData(x)$matilda_prob <- prob
    return(x)
  }
  list(pred = pred, prob = prob)
}

#' @keywords internal
.write_back_latent <- function(x, L) {
  if (methods::is(x, "SingleCellExperiment")) {
    # latent rows are in input cell order (task DataLoader uses shuffle=FALSE)
    rownames(L) <- colnames(x)
    SingleCellExperiment::reducedDim(x, "MATILDA") <- L
    return(x)
  }
  if (methods::is(x, "SummarizedExperiment") || methods::is(x, "MultiAssayExperiment")) {
    S4Vectors::metadata(x)$MATILDA <- L
    return(x)
  }
  list(latent = L)
}

#' @keywords internal
.build_sim <- function(s) {
  if (is.null(s$rna)) return(s)
  sce <- SingleCellExperiment::SingleCellExperiment(assays = list(counts = s$rna))
  if (!is.null(s$adt)) {
    SingleCellExperiment::altExp(sce, "ADT") <-
      SummarizedExperiment::SummarizedExperiment(list(counts = s$adt))
  }
  if (!is.null(s$atac)) {
    SingleCellExperiment::altExp(sce, "ATAC") <-
      SummarizedExperiment::SummarizedExperiment(list(counts = s$atac))
  }
  if (!is.null(s$label)) SummarizedExperiment::colData(sce)$label <- s$label
  sce
}

#' Run one or more Matilda tasks in a single call (object in, enriched object out).
#'
#' The combinable counterpart of the single-task verbs (\code{matilda_classify} /
#' \code{matilda_reduce} / \code{matilda_markers} / \code{matilda_simulate}), which are thin
#' wrappers over it. Mirrors the Python \code{matilda.task()}: enable any combination of
#' tasks and they run in a single engine pass (the model loads once). Results are written
#' back into the returned object — classification to \code{colData$matilda_pred} /
#' \code{$matilda_prob}, dim_reduce to \code{reducedDim "MATILDA"}, fs to
#' \code{metadata$matilda_markers} (a data.frame), simulation to
#' \code{metadata$matilda_simulation} (a SingleCellExperiment whose \code{metadata(.)$real}
#' holds the real reference cells). For a plain matrix list, a named results list is returned.
#'
#' (Unlike Python's \code{task()}, which returns a \code{TaskResult}, the R form returns the
#' enriched object — the idiomatic Bioconductor pattern.)
#'
#' @param x SCE/MAE (with a model, or a query) / matrix list.
#' @param reference a trained object/model to use; \code{NULL} = use \code{x}'s own.
#' @param classification,dim_reduce,fs,simulation task flags; any combination may be TRUE.
#' @param fs_method "IntegratedGradient" (default) or "Saliency".
#' @param simulation_ct cell type to simulate (\code{NULL} = all types).
#' @param simulation_num number of cells to simulate.
#' @param label optional cell-type labels (a colData column name or a vector); required for
#'   \code{fs} / \code{simulation}, optional ground truth for \code{classification}.
#' @param assay,adt_exp,atac_exp assay/altExp selectors.
#' @param device "auto"/"cpu"/"cuda".
#' @return \code{x} enriched with the requested results (a named list for matrix input).
#' @examples
#' sce <- matilda_example_sce()
#' \donttest{
#'   sce <- matilda_train(sce, label = "cell_type", epochs = 2L)
#'   sce <- matilda_task(sce, classification = TRUE, dim_reduce = TRUE)
#' }
#' @export
matilda_task <- function(x, reference = NULL,
                         classification = FALSE, dim_reduce = FALSE, fs = FALSE,
                         simulation = FALSE,
                         fs_method = c("IntegratedGradient", "Saliency"),
                         simulation_ct = NULL, simulation_num = 100L, label = NULL,
                         assay = "counts", adt_exp = "ADT", atac_exp = "ATAC",
                         device = c("auto", "cpu", "cuda")) {
  fs_method <- match.arg(fs_method); device <- match.arg(device)
  if (!(classification || dim_reduce || fs || simulation)) {
    stop("matilda_task(): enable at least one of classification/dim_reduce/fs/simulation.")
  }
  model <- .resolve_model(x, reference)
  lbl <- label
  if ((fs || simulation) && is.null(lbl)) lbl <- model$label_col
  if ((fs || simulation) && is.null(lbl)) {
    stop("fs/simulation need cell-type labels; pass label= (a colData column name or a vector).")
  }
  ct <- if (is.null(simulation_ct)) "-1" else as.character(simulation_ct)
  flags <- c(
    if (classification) c("--classification", "True"),
    if (dim_reduce)     c("--dim_reduce", "True"),
    if (fs)             c("--fs", "True", "--fs_method", fs_method),
    if (simulation)     c("--simulation", "True", "--simulation_ct", ct,
                          "--simulation_num", as.character(as.integer(simulation_num))))
  r <- .run_task(model, x, lbl, flags, query = !is.null(reference),
                 assay, adt_exp, atac_exp, device)
  on.exit(unlink(r$rundir, recursive = TRUE), add = TRUE)
  obj <- .is_se(x); res <- list()
  if (classification) {
    df <- .parse_classification(
      file.path(r$out, "classification", model$mode, r$sub, "accuracy_each_cell.txt"))
    if (obj) x <- .write_back_pred(x, df$predicted, df$prob)
    else res$classification <- list(pred = df$predicted, prob = df$prob)
  }
  if (dim_reduce) {
    L <- .read_latent(file.path(r$out, "dim_reduce", model$mode, r$sub, "latent_space.csv"))
    if (obj) x <- .write_back_latent(x, L) else res$latent <- L
  }
  if (fs) {
    mk <- .read_markers(file.path(r$out, "marker", model$mode, r$sub))
    if (obj) S4Vectors::metadata(x)$matilda_markers <- mk else res$markers <- mk
  }
  if (simulation) {
    sim <- .read_sim(file.path(r$out, "simulation_result", model$mode, r$sub))
    out <- .build_sim(sim$sim)
    if (methods::is(out, "SingleCellExperiment")) S4Vectors::metadata(out)$real <- sim$real
    if (obj) S4Vectors::metadata(x)$matilda_simulation <- out else res$simulation <- out
  }
  if (obj) x else res
}

#' Classify cells with a trained Matilda model (automatic feature reconciliation).
#'
#' Labels \code{x} against a reference, deciding from the feature overlap whether it can reuse
#' the reference's model or must retrain — so the call is the same whether or not the panels
#' match. When \code{x} already carries its own model (no \code{reference}) it simply classifies
#' \code{x}. When a \code{reference} is given: if \code{x} carries every feature the model was
#' trained on (equal panel, or a superset) the query is sliced to the model's features and the
#' model is **reused** (no retrain); if \code{x} is missing some of them (the common
#' cross-dataset case) the per-modality **intersection** of reference and query is taken
#' (reference order, real values only, no zero-padding), a model is **retrained** on it, and the
#' query is classified. \code{metadata(.)$matilda_retrained} records which path ran, and
#' \code{metadata(.)$matilda_common_features} the per-modality feature counts kept (both paths).
#'
#' @inheritParams matilda_task
#' @param label reference cell-type label (a \code{colData} column name or vector). Needed only
#'   when a retrain is required (the query misses model features, or the reference is not yet
#'   trained). When \code{reference} is \code{NULL}, these are \code{x}'s own labels.
#' @param query_label optional ground-truth labels for the query (adds the accuracy report).
#' @param epochs,seed training options used only when a retrain is required.
#' @return \code{x} with \code{colData$matilda_pred}/\code{$matilda_prob}, or a list for matrices.
#' @examples
#' sce <- matilda_example_sce()
#' \donttest{
#'   sce <- matilda_train(sce, label = "cell_type", epochs = 2L)
#'   sce <- matilda_classify(sce)
#' }
#' @export
matilda_classify <- function(x, reference = NULL, label = NULL, query_label = NULL,
                             assay = "counts", adt_exp = "ADT", atac_exp = "ATAC",
                             epochs = 30L, seed = 1L, device = c("auto", "cpu", "cuda")) {
  device <- match.arg(device)
  if (!is.null(reference) && methods::is(x, "SingleCellExperiment")) {
    m <- matilda_model(reference)
    rfeat <- if (!is.null(m)) m$features
             else if (methods::is(reference, "SingleCellExperiment"))
               .sce_features(reference, adt_exp, atac_exp) else NULL
    covered <- !is.null(rfeat) && .covers(rfeat, .sce_features(x, adt_exp, atac_exp))
    if (!covered) {
      # query misses some reference features -> intersect + retrain on the overlap
      if (!methods::is(reference, "SingleCellExperiment")) {
        stop("matilda_classify(): the query is missing model features, so it must retrain on ",
             "the reference∩query intersection — pass `reference` as the labelled ",
             "SingleCellExperiment (not a bare model) so its data is available.")
      }
      refmods <- names(.sce_features(reference, adt_exp, atac_exp))
      qmods   <- names(.sce_features(x, adt_exp, atac_exp))
      for (mod in setdiff(union(refmods, qmods), intersect(refmods, qmods)))
        warning(sprintf("modality '%s' is present in only one of reference/query; it is dropped (only modalities present in both are used).", mod))
      ix <- .intersect_sce(reference, x, adt_exp, atac_exp)
      fit <- matilda_train(ix$reference, label = label, assay = assay, adt_exp = adt_exp,
                           atac_exp = atac_exp, epochs = epochs, seed = seed, device = device)
      out <- matilda_task(ix$query, reference = fit, classification = TRUE,
                          label = query_label, assay = assay, adt_exp = adt_exp,
                          atac_exp = atac_exp, device = device)
      if (.is_se(out)) {
        S4Vectors::metadata(out)$matilda_common_features <- ix$common
        S4Vectors::metadata(out)$matilda_retrained <- TRUE
      }
      return(out)
    }
    # covered -> reuse: train the reference if it has no model yet, slice the query to the
    # model's features (drop extras, fix order), then classify.
    ref <- reference
    if (is.null(m)) {
      ref <- matilda_train(reference, label = label, assay = assay, adt_exp = adt_exp,
                           atac_exp = atac_exp, epochs = epochs, seed = seed, device = device)
      m <- matilda_model(ref)
    }
    x <- .slice_to_features(x, m$features, adt_exp, atac_exp)
    out <- matilda_task(x, reference = ref, classification = TRUE, label = query_label,
                        assay = assay, adt_exp = adt_exp, atac_exp = atac_exp, device = device)
    if (.is_se(out)) {                                  # report on both paths (mirror Python)
      S4Vectors::metadata(out)$matilda_common_features <- lengths(Filter(length, m$features))
      S4Vectors::metadata(out)$matilda_retrained <- FALSE
    }
    return(out)
  }
  if (!is.null(reference) && !methods::is(x, "SingleCellExperiment")) {
    # MAE / matrix-list queries can't be auto-reconciled here (intersect/slice are SCE-only);
    # warn so a mismatched panel isn't silently mispredicted by the positional engine.
    warning("matilda_classify(): automatic feature reconciliation runs only for a ",
            "SingleCellExperiment query; for a ", class(x)[1], " query, ensure its features ",
            "already match the reference model (mismatched panels are not reconciled here).")
  }
  out <- matilda_task(x, reference = reference, classification = TRUE, label = label,
                      assay = assay, adt_exp = adt_exp, atac_exp = atac_exp, device = device)
  if (.is_se(x)) out else out$classification
}

#' Project cells into the Matilda integrated latent space.
#'
#' Single-task wrapper over \code{\link{matilda_task}}.
#'
#' @inheritParams matilda_task
#' @return \code{x} with \code{reducedDim "MATILDA"}, or a list for matrices.
#' @examples
#' sce <- matilda_example_sce()
#' \donttest{
#'   sce <- matilda_train(sce, label = "cell_type", epochs = 2L)
#'   sce <- matilda_reduce(sce)
#' }
#' @export
matilda_reduce <- function(x, reference = NULL, label = NULL,
                           assay = "counts", adt_exp = "ADT", atac_exp = "ATAC",
                           device = c("auto", "cpu", "cuda")) {
  device <- match.arg(device)
  out <- matilda_task(x, reference = reference, dim_reduce = TRUE, label = label,
                      assay = assay, adt_exp = adt_exp, atac_exp = atac_exp, device = device)
  if (.is_se(x)) out else list(latent = out$latent)
}

#' Per-cell-type feature importance (markers) via integrated gradients / saliency.
#'
#' Single-task wrapper over \code{\link{matilda_task}}.
#'
#' @inheritParams matilda_task
#' @param method "IntegratedGradient" (default) or "Saliency".
#' @return data.frame(celltype, feature, importance).
#' @examples
#' sce <- matilda_example_sce()
#' \donttest{
#'   sce <- matilda_train(sce, label = "cell_type", epochs = 2L)
#'   mk <- matilda_markers(sce)
#' }
#' @export
matilda_markers <- function(x, reference = NULL, label = NULL,
                            method = c("IntegratedGradient", "Saliency"),
                            assay = "counts", adt_exp = "ADT", atac_exp = "ATAC",
                            device = c("auto", "cpu", "cuda")) {
  method <- match.arg(method); device <- match.arg(device)
  out <- matilda_task(x, reference = reference, fs = TRUE, fs_method = method, label = label,
                      assay = assay, adt_exp = adt_exp, atac_exp = atac_exp, device = device)
  if (.is_se(x)) S4Vectors::metadata(out)$matilda_markers else out$markers
}

#' Simulate cells for a cell type (or all types) from a trained model.
#'
#' Single-task wrapper over \code{\link{matilda_task}}.
#'
#' @inheritParams matilda_task
#' @param celltype cell-type name to simulate; \code{NULL} = all types.
#' @param n number of cells to simulate.
#' @return a SingleCellExperiment of simulated cells. \code{metadata(.)$real}
#'   holds the real reference cells Matilda used, as a list of per-modality
#'   matrices (\code{rna}/\code{adt}/\code{atac}) plus \code{label}, in the same
#'   feature space as the simulation (for real-vs-simulated UMAPs).
#' @examples
#' sce <- matilda_example_sce()
#' \donttest{
#'   sce <- matilda_train(sce, label = "cell_type", epochs = 2L)
#'   sim <- matilda_simulate(sce, celltype = "A", n = 20L)
#' }
#' @export
matilda_simulate <- function(x, reference = NULL, celltype = NULL, n = 100L, label = NULL,
                             assay = "counts", adt_exp = "ADT", atac_exp = "ATAC",
                             device = c("auto", "cpu", "cuda")) {
  device <- match.arg(device)
  out <- matilda_task(x, reference = reference, simulation = TRUE,
                      simulation_ct = celltype, simulation_num = n, label = label,
                      assay = assay, adt_exp = adt_exp, atac_exp = atac_exp, device = device)
  if (.is_se(x)) S4Vectors::metadata(out)$matilda_simulation else out$simulation
}

#' Run Matilda tasks from file paths (mirrors main_matilda_task.py / main_matilda_rna_task.py); writes to outdir.
#'
#' Power-user / validation entry point: drives the unchanged task script on raw
#' .h5/.csv inputs and copies the produced \code{output/} tree to \code{outdir}.
#'
#' @param model a \code{matilda_model}.
#' @param rna,adt,atac,cty input file paths.
#' @param classification,fs,dim_reduce,simulation,query task flags.
#' @param fs_method,simulation_ct,simulation_num task options.
#' @param outdir directory to copy results into.
#' @param device "auto"/"cpu"/"cuda".
#' @return the output directory, invisibly.
#' @examples
#' \donttest{
#'   f <- matilda_example_teaseq()
#'   m <- matilda_train_files(f["train_rna"], f["train_adt"], f["train_atac"],
#'                            f["train_cty"], epochs = 2L)
#'   matilda_task_files(m, f["test_rna"], f["test_adt"], f["test_atac"],
#'                      f["test_cty"], classification = TRUE, query = TRUE,
#'                      outdir = tempfile())
#' }
#' @export
matilda_task_files <- function(model, rna, adt = NULL, atac = NULL, cty,
                               classification = FALSE, fs = FALSE, dim_reduce = FALSE,
                               simulation = FALSE, query = FALSE,
                               fs_method = "IntegratedGradient", simulation_ct = -1,
                               simulation_num = 100L, outdir = ".",
                               device = c("auto", "cpu", "cuda")) {
  device <- match.arg(device)
  rundir <- .stage_rundir(); on.exit(unlink(rundir, recursive = TRUE), add = TRUE)
  .seed_rundir_model(rundir, model)
  args <- c("--rna", rna, "--cty", cty,
            if (!is.null(adt))  c("--adt",  adt),
            if (!is.null(atac)) c("--atac", atac),
            if (classification) c("--classification", "True"),
            if (fs) c("--fs", "True", "--fs_method", fs_method),
            if (dim_reduce) c("--dim_reduce", "True"),
            if (simulation) c("--simulation", "True", "--simulation_ct", as.character(simulation_ct),
                              "--simulation_num", as.character(as.integer(simulation_num))),
            if (query) c("--query", "True"),
            "--z_dim", model$dims$z_dim, "--hidden_rna", model$dims$hidden_rna,
            if (!is.null(model$dims$nfeatures_adt))  c("--hidden_adt",  model$dims$hidden_adt),
            if (!is.null(model$dims$nfeatures_atac)) c("--hidden_atac", model$dims$hidden_atac),
            "--seed", model$hyper$seed)
  script <- if (model$mode == "rna_only") "main_matilda_rna_task.py" else "main_matilda_task.py"
  .matilda_run(script, as.character(args), rundir, device = device)
  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
  files <- list.files(file.path(rundir, "output"), full.names = TRUE)
  if (length(files)) file.copy(files, outdir, recursive = TRUE)
  invisible(normalizePath(outdir))
}

#' Per-modality feature names of an SCE ({rna, adt, atac}; adt/atac only if the altExp exists).
#' @keywords internal
.sce_features <- function(sce, adt_exp = "ADT", atac_exp = "ATAC") {
  f <- list(rna = rownames(sce))
  ae <- SingleCellExperiment::altExpNames(sce)
  if (adt_exp %in% ae)  f$adt  <- rownames(SingleCellExperiment::altExp(sce, adt_exp))
  if (atac_exp %in% ae) f$atac <- rownames(SingleCellExperiment::altExp(sce, atac_exp))
  f
}

#' TRUE iff the query supplies every feature (per modality) the model was trained on.
#' @keywords internal
.covers <- function(rfeat, qfeat) {
  for (m in names(rfeat)) {
    rf <- rfeat[[m]]
    if (is.null(rf)) next
    qf <- qfeat[[m]]
    if (is.null(qf) || !all(rf %in% qf)) return(FALSE)
  }
  TRUE
}

#' Slice an SCE to the model's exact features + order (drop extra features / altExps).
#' @keywords internal
.slice_to_features <- function(sce, feats, adt_exp = "ADT", atac_exp = "ATAC") {
  s <- sce[feats$rna, ]
  keep <- c(if (!is.null(feats$adt)) adt_exp, if (!is.null(feats$atac)) atac_exp)
  for (ae in SingleCellExperiment::altExpNames(s)) {
    if (ae %in% keep) {
      sub <- if (ae == adt_exp) feats$adt else feats$atac
      SingleCellExperiment::altExp(s, ae) <- SingleCellExperiment::altExp(sce, ae)[sub, ]
    } else {
      SingleCellExperiment::altExp(s, ae) <- NULL
    }
  }
  s
}

#' @keywords internal
.intersect_sce <- function(reference, query, adt_exp = "ADT", atac_exp = "ATAC") {
  if (!methods::is(reference, "SingleCellExperiment") ||
      !methods::is(query, "SingleCellExperiment")) {
    stop("matilda_classify() needs SingleCellExperiment reference and query (with feature rownames).")
  }
  rc <- intersect(rownames(reference), rownames(query))          # reference order
  if (!length(rc)) stop("reference and query share no common RNA features.")
  ref_s <- reference[rc, ]; qry_s <- query[rc, ]
  common <- c(rna = length(rc))
  shared <- intersect(SingleCellExperiment::altExpNames(reference),
                      SingleCellExperiment::altExpNames(query))
  for (ae in shared) {
    a <- intersect(rownames(SingleCellExperiment::altExp(reference, ae)),
                   rownames(SingleCellExperiment::altExp(query, ae)))
    if (!length(a)) stop(sprintf("reference and query share no common features in altExp '%s'.", ae))
    SingleCellExperiment::altExp(ref_s, ae) <- SingleCellExperiment::altExp(reference, ae)[a, ]
    SingleCellExperiment::altExp(qry_s, ae) <- SingleCellExperiment::altExp(query, ae)[a, ]
    common[ae] <- length(a)
  }
  for (ae in setdiff(SingleCellExperiment::altExpNames(ref_s), shared))
    SingleCellExperiment::altExp(ref_s, ae) <- NULL
  for (ae in setdiff(SingleCellExperiment::altExpNames(qry_s), shared))
    SingleCellExperiment::altExp(qry_s, ae) <- NULL
  list(reference = ref_s, query = qry_s, common = common)
}
