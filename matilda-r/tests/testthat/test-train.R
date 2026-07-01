test_that("matilda_train_files trains on the TEAseq demo and returns a model", {
  skip_if_no_matilda_env(); skip_if_no_demo()
  d <- demo_dir()
  m <- matilda_train_files(
    rna = file.path(d, "train_rna.h5"), adt = file.path(d, "train_adt.h5"),
    atac = file.path(d, "train_atac.h5"), cty = file.path(d, "train_cty.csv"),
    epochs = 2L, seed = 1L)
  expect_s3_class(m, "matilda_model")
  expect_equal(m$mode, "TEAseq")
  expect_gte(length(m$label_levels), 2L)
  expect_gt(length(m$checkpoint), 0L)
})

test_that("matilda_train stores the model inside an SCE", {
  skip_if_no_matilda_env()
  sce <- toy_sce(n_rna = 50, n_cells = 60)
  sce <- matilda_train(sce, label = "cell_type", epochs = 2L, z_dim = 16L)
  expect_false(is.null(matilda_model(sce)))
  expect_equal(matilda_model(sce)$mode, "TEAseq")
  expect_equal(matilda_model(sce)$label_col, "cell_type")
})
