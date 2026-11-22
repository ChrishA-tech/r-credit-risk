# Credit-risk modelling on the German Credit dataset (UCI, 1,000 loans).
# Compares a logistic baseline, L1-regularised logistic (glmnet) and a GAM
# (mgcv) with non-linear effects, then chooses a COST-SENSITIVE decision
# threshold using the standard 5:1 cost matrix (approving a bad loan costs
# five times rejecting a good one). Reproducible end to end.

suppressMessages({
  library(dplyr); library(ggplot2); library(glmnet)
  library(pROC); library(mgcv); library(broom); library(scales)
})
set.seed(42)
dir.create("figures", showWarnings = FALSE)

# ---- 1. Load + prepare -------------------------------------------------------
raw <- read.csv("data/german_credit.csv", stringsAsFactors = TRUE)
# Target: 1 = bad credit (the risk we model), 0 = good.
df <- raw %>% mutate(bad = as.integer(credit_risk == 0)) %>% select(-credit_risk)
cat(sprintf("Loans: %d | bad rate: %.1f%%\n", nrow(df), 100 * mean(df$bad)))

# Stratified 70/30 train/test split
idx <- unlist(lapply(split(seq_len(nrow(df)), df$bad),
                     function(i) sample(i, floor(0.7 * length(i)))))
train <- df[idx, ]; test <- df[-idx, ]

# ---- 2. EDA figure -----------------------------------------------------------
p_eda <- df %>%
  mutate(bad = factor(bad, labels = c("good", "bad"))) %>%
  ggplot(aes(x = duration, y = amount, colour = bad)) +
  geom_point(alpha = 0.5) +
  scale_colour_manual(values = c(good = "#1f5f8b", bad = "#c0392b")) +
  scale_y_continuous(labels = comma) +
  labs(title = "Loan amount vs duration by credit outcome",
       x = "Duration (months)", y = "Amount (DM)", colour = "Outcome") +
  theme_minimal(base_size = 12)
ggsave("figures/eda_amount_duration.png", p_eda, width = 7, height = 4.5, dpi = 130)

# ---- 3. Models ---------------------------------------------------------------
# (a) Logistic baseline
f <- bad ~ status + duration + credit_history + purpose + amount + savings +
  employment_duration + installment_rate + personal_status_sex + other_debtors +
  present_residence + property + age + other_installment_plans + housing +
  number_credits + job + people_liable + telephone + foreign_worker
m_logit <- glm(f, data = train, family = binomial)

# (b) L1-regularised logistic (glmnet) with CV-selected lambda
mm <- function(d) model.matrix(f, data = d)[, -1]
x_tr <- mm(train); x_te <- mm(test)
cvfit <- cv.glmnet(x_tr, train$bad, family = "binomial", alpha = 1, nfolds = 10)

# (c) GAM with smooth terms on the continuous risk drivers
m_gam <- gam(bad ~ s(duration) + s(amount) + s(age) + status + credit_history +
             savings + purpose + employment_duration + installment_rate +
             other_debtors + property + other_installment_plans + housing +
             foreign_worker,
             data = train, family = binomial, method = "REML")

# ---- 4. Test-set probabilities + ROC/AUC ------------------------------------
p_logit <- predict(m_logit, test, type = "response")
p_lasso <- as.numeric(predict(cvfit, x_te, s = "lambda.min", type = "response"))
p_gam   <- as.numeric(predict(m_gam, test, type = "response"))

auc <- function(p) as.numeric(pROC::auc(pROC::roc(test$bad, p, quiet = TRUE)))
aucs <- c(Logistic = auc(p_logit), `LASSO (glmnet)` = auc(p_lasso), `GAM (mgcv)` = auc(p_gam))
cat("\nTest AUC:\n"); print(round(aucs, 3))

# ROC overlay
roc_df <- bind_rows(lapply(names(aucs), function(nm) {
  p <- switch(nm, Logistic = p_logit, `LASSO (glmnet)` = p_lasso, `GAM (mgcv)` = p_gam)
  r <- pROC::roc(test$bad, p, quiet = TRUE)
  data.frame(model = sprintf("%s (AUC %.3f)", nm, aucs[[nm]]),
             fpr = 1 - r$specificities, tpr = r$sensitivities)
}))
p_roc <- ggplot(roc_df, aes(fpr, tpr, colour = model)) +
  geom_abline(linetype = "dashed", colour = "grey60") +
  geom_path(linewidth = 0.9) +
  labs(title = "ROC on held-out test set", x = "False positive rate",
       y = "True positive rate", colour = NULL) +
  theme_minimal(base_size = 12) + theme(legend.position = c(0.65, 0.2))
ggsave("figures/roc_comparison.png", p_roc, width = 7, height = 4.8, dpi = 130)

# ---- 5. Cost-sensitive threshold (the banking decision) ----------------------
# Cost matrix: approving a BAD loan (false negative) = 5; rejecting a GOOD loan
# (false positive) = 1. Choose the P(bad) cutoff minimising expected cost.
best <- bestp <- NULL
champ <- names(which.max(aucs))
p_best_model <- switch(champ, Logistic = p_logit, `LASSO (glmnet)` = p_lasso, `GAM (mgcv)` = p_gam)
grid <- seq(0.05, 0.95, 0.01)
costs <- sapply(grid, function(t) {
  pred_bad <- p_best_model >= t
  fn <- sum(!pred_bad & test$bad == 1)   # bad loan approved
  fp <- sum(pred_bad  & test$bad == 0)   # good loan rejected
  5 * fn + 1 * fp
})
t_star <- grid[which.min(costs)]
pred_bad <- p_best_model >= t_star
default_thresh_cost <- { pb <- p_best_model >= 0.5
  5 * sum(!pb & test$bad == 1) + 1 * sum(pb & test$bad == 0) }
cat(sprintf("\nChampion: %s\n", champ))
cat(sprintf("Cost-optimal threshold P(bad) >= %.2f\n", t_star))
cat(sprintf("Expected cost: %d (vs %d at the naive 0.50 cutoff) -> %.0f%% lower\n",
            min(costs), default_thresh_cost,
            100 * (default_thresh_cost - min(costs)) / default_thresh_cost))

cost_df <- data.frame(threshold = grid, cost = costs)
p_cost <- ggplot(cost_df, aes(threshold, cost)) +
  geom_line(colour = "#1f5f8b", linewidth = 0.9) +
  geom_vline(xintercept = t_star, linetype = "dashed", colour = "#c0392b") +
  annotate("text", x = t_star + 0.02, y = max(costs),
           label = sprintf("optimal %.2f", t_star), hjust = 0, colour = "#c0392b") +
  labs(title = "Expected cost vs decision threshold (5:1 cost matrix)",
       x = "P(bad) cutoff", y = "Expected cost on test set") +
  theme_minimal(base_size = 12)
ggsave("figures/cost_threshold.png", p_cost, width = 7, height = 4.5, dpi = 130)

# ---- 6. Persist metrics ------------------------------------------------------
metrics <- list(
  n = nrow(df), bad_rate = round(mean(df$bad), 3),
  auc = round(aucs, 3), champion = champ,
  cost_optimal_threshold = t_star,
  cost_optimal = min(costs), cost_naive_0.5 = default_thresh_cost)
writeLines(jsonlite_like <- paste0(
  "{\n",
  sprintf('  "n": %d,\n', metrics$n),
  sprintf('  "bad_rate": %.3f,\n', metrics$bad_rate),
  sprintf('  "auc_logistic": %.3f,\n', aucs[["Logistic"]]),
  sprintf('  "auc_lasso": %.3f,\n', aucs[["LASSO (glmnet)"]]),
  sprintf('  "auc_gam": %.3f,\n', aucs[["GAM (mgcv)"]]),
  sprintf('  "champion": "%s",\n', champ),
  sprintf('  "cost_optimal_threshold": %.2f,\n', t_star),
  sprintf('  "cost_optimal": %d,\n', min(costs)),
  sprintf('  "cost_naive_0_5": %d\n', default_thresh_cost),
  "}\n"), "metrics.json")
cat("\nWrote metrics.json and figures/.\n")
