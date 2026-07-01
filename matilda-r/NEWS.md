# matilda 0.99.0

* Initial version: an R interface to Matilda (multi-task learning from
  single-cell multimodal omics: RNA, ADT, ATAC), wrapping the upstream PyTorch
  implementation via `basilisk` + `reticulate` so users never manage Python.
* Seurat-style API operating on `SingleCellExperiment` / `MultiAssayExperiment`:
  `matilda_train()`, `matilda_classify()`, `matilda_reduce()`,
  `matilda_markers()`, `matilda_simulate()`; results are written back into the
  object and the trained model is stored in `metadata()`.
* Validated on the TEA-seq demo: query classification accuracy is identical to
  the original Python implementation run in the same environment.
