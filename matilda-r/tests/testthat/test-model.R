make_model <- function() {
  matilda:::new_matilda_model(
    checkpoint = as.raw(c(1, 2, 3, 4)), mode = "TEAseq",
    label_levels = c("A", "B", "C"),
    features = list(rna = c("g1", "g2"), adt = "a1", atac = "p1"),
    dims = list(z_dim = 100L, hidden_rna = 185L, hidden_adt = 30L, hidden_atac = 185L,
                nfeatures_rna = 2L, nfeatures_adt = 1L, nfeatures_atac = 1L, classify_dim = 3L),
    hyper = list(epochs = 30L, lr = 0.02, batch_size = 64L, seed = 1L, augmentation = TRUE),
    label_col = "cell_type")
}

test_that("matilda_model round-trips through saveRDS", {
  m <- make_model()
  expect_s3_class(m, "matilda_model")
  f <- tempfile(fileext = ".rds"); saveRDS(m, f); m2 <- readRDS(f)
  expect_identical(m2$checkpoint, as.raw(c(1, 2, 3, 4)))
  expect_identical(m2$label_levels, c("A", "B", "C"))
  expect_output(print(m), "Matilda model")
})

test_that("matilda_model() stores into and extracts from an SCE", {
  sce <- toy_sce(); m <- make_model()
  sce <- matilda:::.store_model(sce, m)
  expect_equal(matilda_model(sce)$mode, "TEAseq")
  expect_identical(matilda_model(m), m)         # passthrough on a model
  expect_null(matilda_model(toy_sce()))         # none stored
})

test_that(".resolve_model prefers reference, else the object's own, else errors", {
  sce <- toy_sce(); m <- make_model()
  trained <- matilda:::.store_model(toy_sce(), m)
  expect_equal(matilda:::.resolve_model(sce, reference = trained)$mode, "TEAseq")
  expect_equal(matilda:::.resolve_model(trained, reference = NULL)$mode, "TEAseq")
  expect_error(matilda:::.resolve_model(sce, reference = NULL), "No trained Matilda model")
})
