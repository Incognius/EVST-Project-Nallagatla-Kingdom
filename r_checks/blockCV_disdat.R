.libPaths(c(normalizePath("r_checks/lib"), .libPaths()))
suppressPackageStartupMessages({library(blockCV); library(sf); library(disdat)})
regions <- c("AWT", "CAN", "NSW", "NZ", "SA", "SWI")
cats <- list(CAN = "ontveg", NSW = "vegsys", NZ = c("age", "toxicats"), SWI = "calc")
out <- list()
for (r in regions) {
  bg <- disBg(r)
  crs <- if (r %in% c("AWT", "NZ", "SWI")) disCRS(r, format = "EPSG") else 4326
  pts <- st_as_sf(bg, coords = c("x", "y"), crs = crs)
  covs <- setdiff(names(bg), c("spid", "x", "y", "occ", "group", "siteid", cats[[r]]))
  for (v in covs) {
    rr <- tryCatch(cv_spatial_autocor(x = pts, column = v, plot = FALSE, progress = FALSE)$range,
                   error = function(e) NA)
    out[[length(out) + 1]] <- data.frame(region = r, covariate = v, range_blockCV = rr)
  }
  cat(r, "done\n")
}
write.csv(do.call(rbind, out), "results/R_blockCV_ranges.csv", row.names = FALSE)
