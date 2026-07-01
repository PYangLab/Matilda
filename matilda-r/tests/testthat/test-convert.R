test_that(".as_modalities extracts rna/adt/atac/cty from an SCE (TEAseq)", {
  sce <- toy_sce()
  m <- matilda:::.as_modalities(sce, label = "cell_type",
                                rna = "counts", adt = "ADT", atac = "ATAC")
  expect_equal(dim(m$rna), c(40L, 24L))
  expect_equal(dim(m$adt), c(6L, 24L))
  expect_equal(dim(m$atac), c(30L, 24L))
  expect_equal(m$mode, "TEAseq")
  expect_equal(m$cty, as.character(SummarizedExperiment::colData(sce)$cell_type))
})

test_that(".as_modalities infers CITEseq from a matrix list", {
  sce <- toy_sce()
  m <- matilda:::.as_modalities(
    list(rna = SingleCellExperiment::counts(sce),
         adt = SummarizedExperiment::assay(SingleCellExperiment::altExp(sce, "ADT"))),
    label = SummarizedExperiment::colData(sce)$cell_type)
  expect_equal(m$mode, "CITEseq")
  expect_null(m$atac)
})

test_that(".as_modalities errors on a bad label column", {
  expect_error(matilda:::.as_modalities(toy_sce(), label = "nope"), "not found")
})
