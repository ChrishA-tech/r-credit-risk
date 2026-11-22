# Lightweight checks on the analysis outputs (run in CI after analysis.R).
stopifnot(file.exists("metrics.json"))
m <- readLines("metrics.json")
val <- function(key) as.numeric(sub(".*: ", "", sub(",", "", grep(key, m, value = TRUE)[1])))

auc_gam <- val('"auc_gam"')
cost_opt <- val('"cost_optimal"')
cost_naive <- val('"cost_naive_0_5"')

stopifnot("champion AUC should clear 0.70" = auc_gam >= 0.70)
stopifnot("cost-sensitive threshold must beat the naive 0.5 cutoff" = cost_opt < cost_naive)
stopifnot("ROC figure produced" = file.exists("figures/roc_comparison.png"))
stopifnot("cost figure produced" = file.exists("figures/cost_threshold.png"))
cat("All checks passed: GAM AUC =", auc_gam,
    "| cost", cost_opt, "<", cost_naive, "(naive)\n")
