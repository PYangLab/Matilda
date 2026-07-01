skip_if_no_matilda_env <- function() {
  # Gate on an explicit opt-in so pure-R test runs never trigger the (slow)
  # basilisk env build. Set MATILDA_RUN_INTEGRATION=1 to run integration tests.
  testthat::skip_if_not(
    nzchar(Sys.getenv("MATILDA_RUN_INTEGRATION")),
    "integration tests off (set MATILDA_RUN_INTEGRATION=1)"
  )
}

demo_dir <- function() {
  Sys.getenv("MATILDA_DEMO", "")
}

skip_if_no_demo <- function() {
  testthat::skip_if_not(
    file.exists(file.path(demo_dir(), "train_rna.h5")),
    "TEAseq demo not found"
  )
}
