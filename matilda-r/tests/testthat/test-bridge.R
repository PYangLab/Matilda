test_that(".stage_rundir builds the repo-relative layout", {
  rd <- matilda:::.stage_rundir()
  expect_true(dir.exists(file.path(rd, "main")))
  expect_true(dir.exists(file.path(rd, "trained_model")))
  expect_true(dir.exists(file.path(rd, "output")))
  expect_true(dir.exists(file.path(rd, "data")))
})

test_that("bridge runs the unchanged train script and writes a checkpoint", {
  skip_if_no_matilda_env(); skip_if_no_demo()
  d <- demo_dir()
  rd <- matilda:::.stage_rundir()
  matilda:::.matilda_run(
    "main_matilda_train.py",
    c("--rna", file.path(d, "train_rna.h5"),
      "--adt", file.path(d, "train_adt.h5"),
      "--atac", file.path(d, "train_atac.h5"),
      "--cty", file.path(d, "train_cty.csv"),
      "--epochs", "2"),
    rundir = rd)
  expect_true(file.exists(file.path(rd, "trained_model", "TEAseq", "model_best.pth.tar")))
  expect_true(file.exists(file.path(rd, "main", "real_cty.csv")))
})
