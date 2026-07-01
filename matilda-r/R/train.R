# Training: Seurat-style object-in/object-out, plus a path-based driver.

#' Train Matilda on single-cell multimodal data.
#'
#' Trains the multimodal VAE + classifier. For a
#' \link[SingleCellExperiment]{SingleCellExperiment} or
#' \link[MultiAssayExperiment]{MultiAssayExperiment} the trained model is stored
#' inside the object (\code{metadata(x)$matilda}) and the object is returned, so
#' it pipes straight into \code{\link{matilda_reduce}} / \code{\link{matilda_classify}}.
#' For plain matrices it returns a \code{matilda_model}.
#'
#' @param x an SCE/MAE/Seurat object, or \code{NULL} to use the matrix arguments.
#' @param label colData column name (or a vector) of cell types (object input).
#' @param rna,adt,atac genes x cells matrices (used only when \code{x} is \code{NULL}).
#' @param cty cell-type labels vector (used only when \code{x} is \code{NULL}).
#' @param assay,adt_exp,atac_exp assay / altExp selectors for SCE/MAE input.
#' @param batch_size,epochs,lr,z_dim,hidden_rna,hidden_adt,hidden_atac,augmentation,seed
#'   training hyperparameters (defaults match upstream Matilda).
#' @param device one of "auto", "cpu", "cuda".
#' @return the input object with the model stored, or a \code{matilda_model} for matrices.
#' @examples
#' sce <- matilda_example_sce()
#' \donttest{
#'   sce <- matilda_train(sce, label = "cell_type", epochs = 2L)
#' }
#' @export
matilda_train <- function(x = NULL, label = NULL,
                          rna = NULL, adt = NULL, atac = NULL, cty = NULL,
                          assay = "counts", adt_exp = "ADT", atac_exp = "ATAC",
                          batch_size = 64L, epochs = 30L, lr = 0.02, z_dim = 100L,
                          hidden_rna = 185L, hidden_adt = 30L, hidden_atac = 185L,
                          augmentation = TRUE, seed = 1L,
                          device = c("auto", "cpu", "cuda")) {
  device <- match.arg(device)
  hp <- list(batch_size = batch_size, epochs = epochs, lr = lr, z_dim = z_dim,
             hidden_rna = hidden_rna, hidden_adt = hidden_adt, hidden_atac = hidden_atac,
             augmentation = augmentation, seed = seed, device = device)
  if (is.null(x)) {
    if (is.null(rna)) stop("Provide an object `x`, or matrices via rna=/adt=/atac= and cty=.")
    if (is.null(cty)) stop("Provide `cty` (cell-type labels) for matrix input.")
    mods <- list(rna = as.matrix(rna),
                 adt  = if (!is.null(adt))  as.matrix(adt)  else NULL,
                 atac = if (!is.null(atac)) as.matrix(atac) else NULL,
                 cty = as.character(cty),
                 mode = .mode_of(!is.null(adt), !is.null(atac)))
    return(.train_mods(mods, hp))
  }
  mods <- .as_modalities(x, label = label, rna = assay, adt = adt_exp, atac = atac_exp)
  model <- .train_mods(mods, hp)
  model$label_col <- if (length(label) == 1L && is.character(label)) label else NULL
  .store_model(x, model)
}

#' Write modalities to temp files and call the path-based trainer.
#' @keywords internal
.train_mods <- function(mods, hp) {
  td <- tempfile("matilda_in_"); dir.create(td)
  on.exit(unlink(td, recursive = TRUE), add = TRUE)
  pth <- function(n) file.path(td, paste0(n, ".h5"))
  .write_h5_matilda(mods$rna, pth("rna"))
  if (!is.null(mods$adt))  .write_h5_matilda(mods$adt,  pth("adt"))
  if (!is.null(mods$atac)) .write_h5_matilda(mods$atac, pth("atac"))
  ctyf <- file.path(td, "cty.csv"); .write_cty_csv(mods$cty, ctyf)
  matilda_train_files(
    rna  = pth("rna"),
    adt  = if (!is.null(mods$adt))  pth("adt")  else NULL,
    atac = if (!is.null(mods$atac)) pth("atac") else NULL,
    cty  = ctyf,
    batch_size = hp$batch_size, epochs = hp$epochs, lr = hp$lr, z_dim = hp$z_dim,
    hidden_rna = hp$hidden_rna, hidden_adt = hp$hidden_adt, hidden_atac = hp$hidden_atac,
    augmentation = hp$augmentation, seed = hp$seed, device = hp$device)
}

#' Train Matilda from .h5 / .csv file paths (mirrors main_matilda_train.py / main_matilda_rna_train.py).
#'
#' @param rna,adt,atac paths to per-modality .h5 files (adt/atac may be NULL).
#' @param cty path to the cell-type label .csv.
#' @inheritParams matilda_train
#' @return a \code{matilda_model}.
#' @examples
#' defaults <- formals(matilda_train_files)[c("epochs", "lr", "z_dim")]
#' \donttest{
#'   f <- matilda_example_teaseq()
#'   m <- matilda_train_files(f["train_rna"], f["train_adt"], f["train_atac"],
#'                            f["train_cty"], epochs = 2L)
#' }
#' @export
matilda_train_files <- function(rna, adt = NULL, atac = NULL, cty,
                                batch_size = 64L, epochs = 30L, lr = 0.02, z_dim = 100L,
                                hidden_rna = 185L, hidden_adt = 30L, hidden_atac = 185L,
                                augmentation = TRUE, seed = 1L,
                                device = c("auto", "cpu", "cuda")) {
  device <- match.arg(device)
  rundir <- .stage_rundir(); on.exit(unlink(rundir, recursive = TRUE), add = TRUE)
  rna_only <- is.null(adt) && is.null(atac)
  script <- if (rna_only) "main_matilda_rna_train.py" else "main_matilda_train.py"
  args <- c("--rna", rna, "--cty", cty,
            if (!is.null(adt))  c("--adt",  adt),
            if (!is.null(atac)) c("--atac", atac),
            "--batch_size", batch_size, "--epochs", epochs, "--lr", lr,
            "--z_dim", z_dim, "--hidden_rna", hidden_rna,
            # the RNA-only script doesn't define --hidden_adt/--hidden_atac, so
            # only pass them when that modality is present.
            if (!is.null(adt))  c("--hidden_adt",  hidden_adt),
            if (!is.null(atac)) c("--hidden_atac", hidden_atac),
            "--seed", seed)
  # upstream argparse type=bool: any non-empty string is True, "" is False.
  if (!augmentation) args <- c(args, "--augmentation", "")
  .matilda_run(script, as.character(args), rundir, device = device)

  mode <- .mode_of(!is.null(adt), !is.null(atac))
  ckf <- file.path(rundir, "trained_model", mode, "model_best.pth.tar")
  if (!file.exists(ckf)) stop("Training produced no checkpoint at ", ckf)
  ck <- readBin(ckf, "raw", n = file.info(ckf)$size)
  real_cty <- file.path(rundir, "main", "real_cty.csv")
  if (file.exists(real_cty)) {
    levels <- as.character(utils::read.csv(real_cty, header = FALSE)[[1]])
  } else {
    # The RNA-only train script does not emit real_cty.csv. Derive the same
    # ordering read_fs_label uses: pd.Categorical sorts unique labels
    # lexicographically (byte order) -> radix sort, locale-independent.
    lab <- as.character(utils::read.csv(cty, header = FALSE)[[2]])[-1]
    levels <- sort(unique(lab), method = "radix")
  }
  feats <- list(rna  = .h5_features(rna),
                adt  = if (!is.null(adt))  .h5_features(adt)  else NULL,
                atac = if (!is.null(atac)) .h5_features(atac) else NULL)
  dims <- list(z_dim = z_dim, hidden_rna = hidden_rna, hidden_adt = hidden_adt,
               hidden_atac = hidden_atac,
               nfeatures_rna  = length(feats$rna),
               nfeatures_adt  = if (!is.null(feats$adt))  length(feats$adt)  else NULL,
               nfeatures_atac = if (!is.null(feats$atac)) length(feats$atac) else NULL,
               classify_dim = length(levels))
  new_matilda_model(ck, mode, levels, feats, dims,
                    list(epochs = epochs, lr = lr, batch_size = batch_size,
                         seed = seed, augmentation = augmentation))
}
