test_that("classify + reduce write results back into the SCE", {
  skip_if_no_matilda_env()
  sce <- toy_sce(n_cells = 60)
  sce <- matilda_train(sce, label = "cell_type", epochs = 2L, z_dim = 16L)
  sce <- matilda_classify(sce)
  expect_true("matilda_pred" %in% colnames(SummarizedExperiment::colData(sce)))
  expect_equal(length(SummarizedExperiment::colData(sce)$matilda_pred), ncol(sce))
  sce <- matilda_reduce(sce)
  expect_true("MATILDA" %in% SingleCellExperiment::reducedDimNames(sce))
  expect_equal(nrow(SingleCellExperiment::reducedDim(sce, "MATILDA")), ncol(sce))
})

test_that("markers returns a tidy frame and simulate returns an SCE", {
  skip_if_no_matilda_env()
  sce <- toy_sce(n_cells = 60)
  sce <- matilda_train(sce, label = "cell_type", epochs = 2L, z_dim = 16L)
  mk <- matilda_markers(sce, method = "IntegratedGradient")
  expect_true(all(c("celltype", "feature", "importance") %in% names(mk)))
  expect_true(all(c("A", "B", "C") %in% mk$celltype))
  sim <- matilda_simulate(sce, celltype = "A", n = 20L)
  expect_s4_class(sim, "SingleCellExperiment")
  expect_gt(ncol(sim), 0L)
})

test_that("query workflow uses an explicit reference (matched panel -> reuse)", {
  skip_if_no_matilda_env()
  ref <- toy_sce(n_cells = 60, seed = 1)
  ref <- matilda_train(ref, label = "cell_type", epochs = 2L, z_dim = 16L)
  query <- toy_sce(n_cells = 30, seed = 2)
  query <- matilda_classify(query, reference = ref)
  expect_equal(length(SummarizedExperiment::colData(query)$matilda_pred), ncol(query))
  expect_false(isTRUE(S4Vectors::metadata(query)$matilda_retrained))   # reused, no retrain
})

test_that("classify reconciles features (different panel -> intersect + retrain)", {
  skip_if_no_matilda_env()
  ref <- toy_sce(n_cells = 60, seed = 1)
  ref <- matilda_train(ref, label = "cell_type", epochs = 2L, z_dim = 16L)
  query <- toy_sce(n_cells = 30, seed = 2)
  query <- query[5:nrow(query), ]                       # drop some RNA features -> different panel
  out <- matilda_classify(query, reference = ref, label = "cell_type", epochs = 2L)
  expect_true(isTRUE(S4Vectors::metadata(out)$matilda_retrained))
  expect_true(!is.null(S4Vectors::metadata(out)$matilda_common_features))
  expect_equal(length(SummarizedExperiment::colData(out)$matilda_pred), ncol(out))
})
