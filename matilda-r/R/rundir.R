# Stage an isolated run directory mirroring the upstream repo layout, so the
# scripts' cwd-relative paths resolve without ever touching the user's wd.
#
#   <rundir>/main           <- cwd for the script; real_cty.csv lands here
#   <rundir>/trained_model  <- ../trained_model/<mode>/model_best.pth.tar
#   <rundir>/output         <- ../output/<task>/<mode>/<ref|query>/...
#   <rundir>/data           <- temp input .h5 / .csv we write

#' @keywords internal
.stage_rundir <- function() {
  rd <- tempfile("matilda_run_")
  # Pre-create the per-mode trained_model subdirs: the upstream RNA-only train
  # script hardcodes a torch.save() to ../trained_model/RNAseq/ that it never
  # mkdir's, so it would crash without this.
  dirs <- c("main", "output", "data",
            file.path("trained_model",
                      c("TEAseq", "CITEseq", "SHAREseq", "rna_only", "RNAseq")))
  for (d in dirs) dir.create(file.path(rd, d), recursive = TRUE, showWarnings = FALSE)
  rd
}

#' Seed a run dir with a trained model so task scripts can load it.
#' Writes the checkpoint to ../trained_model/<mode>/ and real_cty.csv to cwd.
#' @keywords internal
.seed_rundir_model <- function(rundir, model) {
  mdir <- file.path(rundir, "trained_model", model$mode)
  dir.create(mdir, recursive = TRUE, showWarnings = FALSE)
  writeBin(model$checkpoint, file.path(mdir, "model_best.pth.tar"))
  # real_cty.csv: one label per line, no header/index (matches train's to_csv);
  # quoting keeps labels with commas intact for pandas read_csv(header=None).
  utils::write.table(data.frame(model$label_levels), file.path(rundir, "main", "real_cty.csv"),
                     sep = ",", row.names = FALSE, col.names = FALSE, quote = TRUE)
  invisible(rundir)
}
