.libPaths(c(normalizePath("r_checks/lib"), .libPaths()))
suppressMessages(library(sampbias))
set.seed(1)
g <- read.csv("data/processed/sampbias_grid_5km.csv")
x <- data.frame(id = seq_len(nrow(g)), record_count = g$count, g[, setdiff(names(g), "count")])
t0 <- Sys.time()
fit <- sampbias:::.RunSampBias(x = x, rescale_distances = 1, iterations = 1e5, burnin = 2e4)
post <- fit[, grep("^w_", names(fit))]
out <- data.frame(factor = sub("^w_", "", names(post)),
                  w_mean = colMeans(post),
                  w_q025 = apply(post, 2, quantile, 0.025),
                  w_q975 = apply(post, 2, quantile, 0.975))
out$q_mean <- mean(fit$q)
write.csv(out, "results/R_sampbias_wg.csv", row.names = FALSE)
print(out)
print(Sys.time() - t0)
