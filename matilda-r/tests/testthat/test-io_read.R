test_that(".parse_classification reads per-cell predictions (with real label)", {
  f <- tempfile()
  writeLines(c(
    "cell ID:  0 \t \t real cell type: A \t \t predicted cell type: B \t \t probability: 0.77",
    "cell ID:  1 \t \t real cell type: B \t \t predicted cell type: B \t \t probability: 0.53"), f)
  df <- matilda:::.parse_classification(f)
  expect_equal(nrow(df), 2L)
  expect_equal(df$predicted, c("B", "B"))
  expect_equal(df$real, c("A", "B"))
  expect_equal(df$prob, c(0.77, 0.53))
})

test_that(".parse_classification handles the no-real-label form", {
  f <- tempfile()
  writeLines("cell ID:  0 \t \t predicted cell type: NK \t \t probability: 0.91", f)
  df <- matilda:::.parse_classification(f)
  expect_equal(df$predicted, "NK")
  expect_true(is.na(df$real))
  expect_equal(df$prob, 0.91)
})

test_that(".read_latent reads the latent matrix (cells x z)", {
  f <- tempfile(fileext = ".csv")
  utils::write.csv(data.frame(feature_0 = c(0.1, 0.2), feature_1 = c(0.3, 0.4),
                              row.names = c("cell_0", "cell_1")), f)
  L <- matilda:::.read_latent(f)
  expect_equal(dim(L), c(2L, 2L))
  expect_equal(unname(L[2, 2]), 0.4)
})

test_that(".read_markers tidies per-celltype importance files", {
  d <- tempfile(); dir.create(d)
  utils::write.csv(data.frame(`importance score` = c(0.5, 0.2),
                              row.names = c("g1", "g2"), check.names = FALSE),
                   file.path(d, "fs.celltype_A.csv"))
  mk <- matilda:::.read_markers(d)
  expect_true(all(c("celltype", "feature", "importance") %in% names(mk)))
  expect_equal(mk$celltype[1], "A")
  expect_equal(mk$feature[1], "g1")            # sorted by descending importance
})
