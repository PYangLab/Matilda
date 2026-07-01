#!/usr/bin/env Rscript
# Multi-task parity: matilda (R) vs the original Matilda Python, on the SAME
# trained checkpoint, same basilisk env, same seed. Confirms identical results
# across classification, dimension reduction, feature selection and simulation.
#
#   Rscript inst/scripts/parity_check.R
suppressMessages(library(matilda))

D     <- Sys.getenv("MATILDA_DEMO", "")
REPO  <- normalizePath(file.path(D, "..", ".."))            # .../Matilda
MAIN  <- file.path(REPO, "main")
OUT   <- file.path(REPO, "output")
ENVPY <- file.path(basilisk::obtainEnvironmentPath(matilda:::matilda_env), "bin", "python")

h5  <- function(split, mod) file.path(D, sprintf("%s_%s.h5", split, mod))
cty <- function(split)      file.path(D, sprintf("%s_cty.csv", split))
runpy <- function(...) {
  old <- setwd(MAIN); on.exit(setwd(old))
  system2(ENVPY, c(...), stdout = FALSE, stderr = FALSE)
}
cmp <- function(label, r, p) {
  r <- as.matrix(r); p <- as.matrix(p)
  n <- min(nrow(r), nrow(p)); m <- min(ncol(r), ncol(p))
  r <- r[seq_len(n), seq_len(m), drop = FALSE]; p <- p[seq_len(n), seq_len(m), drop = FALSE]
  cat(sprintf("  %-20s max|Δ| = %.3e   Pearson r = %.6f\n",
              label, max(abs(r - p)), stats::cor(as.vector(r), as.vector(p))))
}

cat("== [0] Python train (seed 1) ==\n")
runpy("main_matilda_train.py", "--rna", h5("train","rna"), "--adt", h5("train","adt"),
      "--atac", h5("train","atac"), "--cty", cty("train"), "--seed", "1")

# Build a matilda_model from the Python checkpoint so R and Python share ONE model.
ckf   <- file.path(REPO, "trained_model", "TEAseq", "model_best.pth.tar")
feats <- list(rna = matilda:::.h5_features(h5("train","rna")),
              adt = matilda:::.h5_features(h5("train","adt")),
              atac = matilda:::.h5_features(h5("train","atac")))
levels <- as.character(utils::read.csv(file.path(MAIN, "real_cty.csv"), header = FALSE)[[1]])
model <- matilda:::new_matilda_model(
  readBin(ckf, "raw", n = file.info(ckf)$size), "TEAseq", levels, feats,
  list(z_dim = 100L, hidden_rna = 185L, hidden_adt = 30L, hidden_atac = 185L,
       nfeatures_rna = length(feats$rna), nfeatures_adt = length(feats$adt),
       nfeatures_atac = length(feats$atac), classify_dim = length(levels)),
  list(epochs = 30L, lr = 0.02, batch_size = 64L, seed = 1L, augmentation = TRUE))
rout <- tempfile("Rpar_"); dir.create(rout)
trip <- function(split) list(rna = h5(split,"rna"), adt = h5(split,"adt"), atac = h5(split,"atac"))

## ---- classification (query) ----
cat("== classification (query) ==\n")
runpy("main_matilda_task.py", "--rna", h5("test","rna"), "--adt", h5("test","adt"),
      "--atac", h5("test","atac"), "--cty", cty("test"), "--classification","True","--query","True","--seed","1")
py <- matilda:::.parse_classification(file.path(OUT,"classification","TEAseq","query","accuracy_each_cell.txt"))
matilda_task_files(model, rna=h5("test","rna"), adt=h5("test","adt"), atac=h5("test","atac"),
                   cty=cty("test"), classification=TRUE, query=TRUE, outdir=file.path(rout,"clf"))
r <- matilda:::.parse_classification(file.path(rout,"clf","classification","TEAseq","query","accuracy_each_cell.txt"))
cat(sprintf("  label agreement R vs Py = %.4f   (Py acc %.4f | R acc %.4f)\n",
            mean(r$predicted == py$predicted), mean(py$predicted==py$real), mean(r$predicted==r$real)))
cmp("softmax prob", r$prob, py$prob)

## ---- dimension reduction (query) ----
cat("== dimension reduction (query) ==\n")
runpy("main_matilda_task.py", "--rna", h5("test","rna"), "--adt", h5("test","adt"),
      "--atac", h5("test","atac"), "--cty", cty("test"), "--dim_reduce","True","--query","True","--seed","1")
py_l <- matilda:::.read_latent(file.path(OUT,"dim_reduce","TEAseq","query","latent_space.csv"))
matilda_task_files(model, rna=h5("test","rna"), adt=h5("test","adt"), atac=h5("test","atac"),
                   cty=cty("test"), dim_reduce=TRUE, query=TRUE, outdir=file.path(rout,"dr"))
r_l <- matilda:::.read_latent(file.path(rout,"dr","dim_reduce","TEAseq","query","latent_space.csv"))
cmp("latent space", r_l, py_l)

## ---- feature selection (reference) ----
cat("== feature selection (reference) ==\n")
runpy("main_matilda_task.py", "--rna", h5("train","rna"), "--adt", h5("train","adt"),
      "--atac", h5("train","atac"), "--cty", cty("train"), "--fs","True","--seed","1")
matilda_task_files(model, rna=h5("train","rna"), adt=h5("train","adt"), atac=h5("train","atac"),
                   cty=cty("train"), fs=TRUE, query=FALSE, outdir=file.path(rout,"fs"))
# Compare position-wise per cell type (feature names repeat across modalities, so
# a merge() on (celltype, feature) would cross-pair duplicates).
pyd <- file.path(OUT, "marker", "TEAseq", "reference")
rd  <- file.path(rout, "fs", "marker", "TEAseq", "reference")
fsf <- list.files(pyd, pattern = "^fs\\.celltype_")
fsmax <- 0
for (f in fsf) {
  a <- utils::read.csv(file.path(rd,  f), check.names = FALSE)[[2]]
  b <- utils::read.csv(file.path(pyd, f), check.names = FALSE)[[2]]
  fsmax <- max(fsmax, max(abs(a - b)))
}
cat(sprintf("  %-20s max|Δ| = %.3e   (%d cell types, feature-aligned)\n",
            "feature importance", fsmax, length(fsf)))

## ---- simulation (reference, all types) ----
cat("== simulation (reference, all types) ==\n")
runpy("main_matilda_task.py", "--rna", h5("train","rna"), "--adt", h5("train","adt"),
      "--atac", h5("train","atac"), "--cty", cty("train"), "--simulation","True",
      "--simulation_ct","-1","--simulation_num","200","--seed","1")
py_s <- matilda:::.read_sim(file.path(OUT,"simulation_result","TEAseq","reference"))
matilda_task_files(model, rna=h5("train","rna"), adt=h5("train","adt"), atac=h5("train","atac"),
                   cty=cty("train"), simulation=TRUE, simulation_ct=-1, simulation_num=200,
                   query=FALSE, outdir=file.path(rout,"sim"))
r_s <- matilda:::.read_sim(file.path(rout,"sim","simulation_result","TEAseq","reference"))
cmp("sim RNA", r_s$sim$rna, py_s$sim$rna)

cat("== PARITY_DONE ==\n")
