# The bridge: run an UNCHANGED vendored Matilda script inside the basilisk env,
# with sys.argv patched and the working directory set so the scripts' repo-
# relative paths (../trained_model, ../output, real_cty.csv) resolve.

#' Run a vendored Matilda script with patched argv inside the basilisk env.
#'
#' @param script file name under inst/python/matilda, e.g. "main_matilda_train.py".
#' @param args   character vector of CLI tokens (already coerced to strings).
#' @param rundir staged run directory from \code{.stage_rundir()}; cwd = <rundir>/main.
#' @param device one of "auto", "cpu", "cuda"; "cpu" hides CUDA from torch.
#' @keywords internal
.matilda_run <- function(script, args, rundir, device = c("auto", "cpu", "cuda")) {
  device <- match.arg(device)
  proc <- basilisk::basiliskStart(matilda_env)
  on.exit(basilisk::basiliskStop(proc), add = TRUE)
  basilisk::basiliskRun(
    proc,
    fun = function(script, args, maindir, pkgpy, device) {
      sys <- reticulate::import("sys", convert = FALSE)
      os  <- reticulate::import("os",  convert = FALSE)
      if (identical(device, "cpu")) {
        os$environ$update(reticulate::dict(CUDA_VISIBLE_DEVICES = ""))
      }
      pkgpy <- normalizePath(pkgpy)
      sys$path$insert(0L, pkgpy)          # resolve `import util`, `from learn... import`
      oldwd <- os$getcwd()
      os$chdir(maindir)                    # ../trained_model, ../output, real_cty.csv
      # The vendored scripts open output text files (accuracy_each_*.txt) but
      # never close them; the long-lived reticulate process therefore leaves
      # their buffers unflushed (small outputs vanish, large ones lose the final
      # buffer). Flush every open writable file after the run, and always
      # restore the cwd (the scripts chdir-relative output would otherwise leave
      # the process inside a deleted run directory).
      flush_py <- paste(
        "import io, gc",
        "for _o in gc.get_objects():",
        "    try:",
        "        if isinstance(_o, io.IOBase) and (not _o.closed) and _o.writable():",
        "            _o.flush()",
        "    except Exception:",
        "        pass",
        sep = "\n")
      on.exit({
        try(reticulate::py_run_string(flush_py), silent = TRUE)
        try(os$chdir(oldwd), silent = TRUE)
      }, add = TRUE)
      sys$argv <- c(script, args)
      reticulate::py_run_file(file.path(pkgpy, script), local = FALSE, convert = FALSE)
      invisible(NULL)
    },
    script  = script,
    args    = as.character(args),
    maindir = normalizePath(file.path(rundir, "main")),
    pkgpy   = .pkg_py(),
    device  = device
  )
  invisible(rundir)
}
