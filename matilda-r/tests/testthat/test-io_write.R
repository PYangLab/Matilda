test_that(".write_h5_matilda writes t(x) to matrix/data with features/barcodes", {
  m <- matrix(c(1, 2, 3, 4, 5, 6), 3, 2,
              dimnames = list(c("g1", "g2", "g3"), c("c1", "c2")))
  f <- tempfile(fileext = ".h5")
  matilda:::.write_h5_matilda(m, f)
  # read_h5_data does np.array(matrix/data).transpose(); on disk it's cells x features.
  d <- as.matrix(HDF5Array::HDF5Array(f, "matrix/data"))
  expect_equal(dim(d), c(2L, 3L))
  expect_equal(unname(d), unname(t(m)))
  expect_equal(as.character(rhdf5::h5read(f, "matrix/features")), c("g1", "g2", "g3"))
  expect_equal(as.character(rhdf5::h5read(f, "matrix/barcodes")), c("c1", "c2"))
})

test_that(".write_cty_csv puts labels in column index 1 (read_fs_label convention)", {
  f <- tempfile(fileext = ".csv")
  matilda:::.write_cty_csv(c("B", "A", "B"), f)
  # pandas read_csv(header=None) then iloc[1:, 1]
  raw <- utils::read.csv(f, header = FALSE, stringsAsFactors = FALSE)
  expect_equal(as.character(raw[[2]][-1]), c("B", "A", "B"))
})
