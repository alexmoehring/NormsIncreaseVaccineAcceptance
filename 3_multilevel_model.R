# Title     : TODO
# Objective : TODO
# Created by: moehring
# Created on: 1/21/2021

# module load sloan/R/3.5.1
# module load sloan/R/CRAN/3.5

library(lme4)
library(ggplot2)
library(estimatr)
library(lmtest)
library(sandwich)
library(broom.mixed)
library(ggplot2)
library(dplyr)
library(aod)

color_map <- list("Broad"="#093469", "Narrow"="#92c2fc")
set.seed(43525)

num_cores <- 3
num_boots <- 20  # Did 500 in figure for paper, but this takes awhile on a desktop computer
data_path <- './data/'
out_path <- './norms_experiment/figs'
options(mc.cores = num_cores)
df <- read.csv(file.path(data_path, "analysis_data.csv"))

# filter non-vaccine behaviors
df <- df[(df$behavior == "vaccine") | (df$behavior == "control"), ]

ov <- 'future_vaccine'
# filter rows w/ missing treatment or outcome
df <- df[!is.na(df[ov]), ]
df <- df[!is.na(df['treatment']), ]

covariates <- c('factor(vaccine_accept_pre_both_imputed)', 'norms_dist_pre_both_imputed', 'norms_vaccine_pre_both_imputed',
                 'norms_masks_pre_both_imputed', 'prevention_distancing_pre_both_imputed',
                 'prevention_mask_pre_both_imputed', 'effect_mask_pre_both_imputed',
                 'distancing_importance_pre_both_imputed', 'factor(period)')

# first center covariates
xs <- data.frame(model.matrix(formula(paste("~", paste(covariates, collapse="+"))), df))
xs <- xs[colnames(xs) != "X.Intercept."]

# center and add to df
covariates <- c()
for(c in colnames(xs)){
  df[paste(c, "_c", sep="")] <- xs[[c]] - weighted.mean(xs[[c]], weights=df[["weight_full"]])
  covariates <- append(covariates, paste(c, "_c", sep=""))
  print(weighted.mean(df[[paste(c, "_c", sep="")]], weights=df$weight_full))
}

df['treatment_high'] <- as.numeric(df$treatment == 'high')
df['treatment_low'] <- as.numeric(df$treatment == 'low')
for (c in covariates){
  covariates <- append(covariates, paste("treatment_high", c, sep=":"))
  covariates <- append(covariates, paste("treatment_low", c, sep=":"))
  df[paste("treatment_high", c, sep=":")] <- df[[c]] * df[["treatment_high"]]
  df[paste("treatment_low", c, sep=":")] <- df[[c]] * df[["treatment_low"]]
}
print(covariates)


# lme_model <- lmer(formula = future_vaccine ~ 1 + treatment + (1 + treatment | country_pre), data=df)
# summary(lme_model)
ols_reg_form <- "future_vaccine ~ 1 + treatment_high + treatment_low +  %s"
ols_reg_form <- sprintf(ols_reg_form, paste(covariates, collapse="+"))
reg_form <- paste(ols_reg_form, "(1 + treatment_high + treatment_low | country_pre)", sep=" + ")

# ols
m <- lm(formula(ols_reg_form), df, weights=df$weight_full)
coeftest(m, vcov=vcovHC(m, type="HC2"))

# ols w/ countries
fe_reg_form <- "future_vaccine ~ 0 + country_pre + country_pre:treatment_high + country_pre:treatment_low + %s"
fe_reg_form <- sprintf(fe_reg_form, paste(covariates, collapse="+"))
m_fe <- lm(
  formula(fe_reg_form),
  data=df,
  weights=df$weight_full
)
coeftest(m_fe, vcov=vcovHC(m_fe, type="HC2"))

lme_model <- lmer(
  formula=formula(reg_form),
  data=df,
  REML=TRUE,
  verbose=2,
  weights=df$weight_full
)

merBoot <- bootMer(
  lme_model,
  FUN=function(m) bind_rows(tidy(m), tidy(m, effects="ran_coefs"))$estimate,
  nsim=num_boots, parallel="multicore", ncpus=num_cores, verbose=TRUE,
  use.u=TRUE, type="parametric"
)
lme_results <- data.frame(bind_rows(tidy(lme_model), tidy(lme_model, effects="ran_coefs")))
# lme_results$lower_boot <- apply(merBoot$t, 2, function(x) as.numeric(quantile(x, probs=.025)))
# lme_results$upper_boot <- apply(merBoot$t, 2, function(x) as.numeric(quantile(x, probs=.975)))

# first plot w/ covariates centered at the grand-mean
grand_mean_lme <- lme_results
grand_mean_lme$se <- apply(merBoot$t, 2, sd)
grand_mean_lme$lower <- grand_mean_lme$estimate - 1.96 * grand_mean_lme$se
grand_mean_lme$upper <- grand_mean_lme$estimate + 1.96 * grand_mean_lme$se


grand_mean_lme <- grand_mean_lme[grand_mean_lme$term == "treatment_high" | grand_mean_lme$term == "treatment_low", ]
grand_mean_lme[is.na(grand_mean_lme$level), "level"] <- "Average"

# now add average treatment value to this df
grand_mean_lme$treatment_value = -1
for(country in unique(grand_mean_lme$level)){
    print(country)

    if(country == "Average"){
        tmp <- df
    }
    else{
        tmp <- df[df$country == country, ]
    }
    country_ixs <- grand_mean_lme$level == country

    high_val <- mean(tmp$high_value_vaccine)
    low_val <- mean(tmp$low_value_vaccine)

    high_ixs = country_ixs & (grand_mean_lme$term == "treatment_high")
    low_ixs = country_ixs & (grand_mean_lme$term == "treatment_low")
    stopifnot(sum(high_ixs) == 1)
    stopifnot(sum(low_ixs) == 1)
    grand_mean_lme[high_ixs, "treatment_value"] <- high_val
    grand_mean_lme[low_ixs, "treatment_value"] <- low_val
}

# now get values centered at country mean
country_mean_lme <- grand_mean_lme
country_mean_boots <- merBoot$t[, lme_results$term == "treatment_high" | lme_results$term == "treatment_low"]
# add interaction terms times country means
row_ix <- 0
for(row in rownames(country_mean_lme)){
  row_ix <- row_ix + 1
  country <- country_mean_lme[row, "level"]
  if (is.na(country) | country == "Average"){
    next
  }
  # get relevant lme_results
  tmp_coefs <- lme_results[!is.na(lme_results$level) & lme_results$level == country, ]

  # get country data
  cdf <- df[df$country == country, ]

  # now loop through these, and add country means
  for(inner_row in rownames(tmp_coefs)){
    term <- tmp_coefs[inner_row, "term"]
    if(!grepl("treatment_", term)){
      next
    }
    if (term == "treatment_high" | term == "treatment_low"){
      next
    }
    coef <- tmp_coefs[inner_row, "estimate"]

    # get covariate name
    cov_name <- strsplit(term, ":")[[1]][[2]]
    cov_mean <- weighted.mean(cdf[[cov_name]], cdf[["weight_full"]])

    # add to estimate
    country_mean_lme[row, "estimate"] <- country_mean_lme[row, "estimate"] + cov_mean * coef

    # also need to add to bootstrapped samples
    country_mean_boots[, row_ix] <- country_mean_boots[, row_ix] + cov_mean * merBoot$t[, as.numeric(inner_row)]
  }
}
country_mean_lme$se <- apply(country_mean_boots, 2, sd)
country_mean_lme$lower <- country_mean_lme$estimate - 1.96 * country_mean_lme$se
country_mean_lme$upper <- country_mean_lme$estimate + 1.96 * country_mean_lme$se

lme_results_to_plot <- list("centered_grand_mean"=grand_mean_lme, "centered_country_mean"=country_mean_lme)
# add in covariances for joint test
for(center_type in names(lme_results_to_plot)){
    print(center_type)
    coef_df <- lme_results_to_plot[[center_type]]
    if(center_type == "centered_country_mean"){
        boots <- country_mean_boots
    }
    else{
        boots <- merBoot$t[, lme_results$term == "treatment_high" | lme_results$term == "treatment_low"]
    }
    coef_df$low_high_cov = -1000
    coef_df$low_var = -1000
    coef_df$high_var = -1000
    for(country in unique(coef_df$level)){
        if(is.na(country)){
            next
        }
        low_ixs <- (coef_df$level == country) & (coef_df$term == "treatment_low") & (!is.na(coef_df$level))
        high_ixs <- (coef_df$level == country) & (coef_df$term == "treatment_high") & (!is.na(coef_df$level))
        low_vals <- boots[, low_ixs]
        high_vals <- boots[, high_ixs]
        c <- cov(low_vals, high_vals)
        v_low <- var(low_vals)
        v_high <- var(high_vals)
        country_ixs <- (coef_df$level == country) & !is.na(coef_df$level)
        coef_df[country_ixs, "low_high_cov"] <- c
        coef_df[country_ixs, "low_var"] <- v_low
        coef_df[country_ixs, "high_var"] <- v_high
    }
    lme_results_to_plot[[center_type]] <- coef_df
}


for(center_type in names(lme_results_to_plot)){
  print(center_type)
  tmp_lme_results <- lme_results_to_plot[[center_type]]

  broad_estimates <- tmp_lme_results[tmp_lme_results$term == "treatment_high", "estimate"]
  narrow_estimates <- tmp_lme_results[tmp_lme_results$term == "treatment_low", "estimate"]
  print(sprintf("Variance of REs for Broad: %s, SD %s", var(broad_estimates), sd(broad_estimates)))
  print(sprintf("Variance of REs for Narrow: %s, SD %s", var(narrow_estimates), sd(narrow_estimates)))

  # now create new factor variable
  tmp <- tmp_lme_results[tmp_lme_results$term == "treatment_high", ]
  country_order <- tmp[order(tmp$estimate), ]$level
  country_order <- country_order[country_order != "Average"]
  country_order <- append(country_order, "Average")
  tmp_lme_results$country <- factor(tmp_lme_results$level, levels=country_order)
  tmp_lme_results$treatment <- "Narrow"
  tmp_lme_results[tmp_lme_results$term == "treatment_high", "treatment"] <- "Broad"
  tmp_lme_results$is_average <- tmp_lme_results$level == "Average"

  ggplot(data=tmp_lme_results, aes(x=country, y=estimate, ymin=lower, ymax=upper, colour=treatment, shape=is_average)) +
    geom_point(position=position_dodge(width=0.6), size=2.5) +
    geom_errorbar(position=position_dodge(width=0.6), width=0.0) +
    geom_hline(yintercept=0) +
    coord_flip() +
    scale_color_manual(values=c(color_map$Broad, color_map$Narrow)) +
    theme_classic() +
    guides(shape=FALSE) +
    labs(colour="", y="Effect on vaccine acceptance scale", x="")

  ggsave(
    file.path(out_path, sprintf("country_coefficients_%s.pdf", center_type)),
    plot = last_plot()
  )

  write.csv(tmp_lme_results, file.path(out_path, sprintf("country_coefficients_%s.csv", center_type)))

  # now plot scatter plot of te with treatment number
  mod <- lm("estimate ~ treatment_value*treatment", tmp_lme_results)
  print(summary(mod))
  mod <- lm("estimate ~ treatment_value", tmp_lme_results)
  print(summary(mod))

  ggplot(data=tmp_lme_results, aes(x=treatment_value, y=estimate, colour=treatment, shape=is_average)) +
    geom_point(position=position_dodge(width=0.6), size=2.5) +
#     geom_errorbar(position=position_dodge(width=0.6), width=0.0) +
    geom_hline(yintercept=0) +
#     coord_flip() +
    geom_smooth(method=lm) +
    scale_color_manual(values=c(color_map$Broad, color_map$Narrow)) +
    theme_classic() +
    guides(shape=FALSE) +
    labs(colour="", y="Effect on vaccine acceptance scale", x="Average treatment value")

      ggsave(
        file.path(out_path, sprintf("country_coefficients_scatter_%s.pdf", center_type)),
        plot = last_plot()
      )

}

# Now test if coefficients are sig different from 0 (both individually and jointly)
coefs <- lme_results_to_plot[["centered_country_mean"]]
num_significant <- list("0.05"=0, "0.005"=0)
num_significant_grand_mean <- list("0.05"=0, "0.005"=0)
grand_mean <- c(coefs[(coefs$level == "Average" & coefs$term == "treatment_low"), "estimate"], coefs[(coefs$level == "Average" & coefs$term == "treatment_high"), "estimate"])
for(country in unique(coefs$level)){
    if(country == "Average"){
        next
    }
    tmp <- coefs[coefs$level == country, ]
    low_row <- tmp[tmp$term == "treatment_low", ]
    high_row <- tmp[tmp$term == "treatment_high", ]

    b <- c(low_row[1, "estimate"], high_row[1, "estimate"])

    sigma <- matrix(c(low_row[1, "low_var"], low_row[1, "low_high_cov"], low_row[1, "low_high_cov"], low_row[1, "high_var"]), nrow=2, ncol=2)
    test <- wald.test(Sigma=sigma, b=b, L=matrix(c(1,1), nrow=1, ncol=2))
    pval <- test$result$chi2[["P"]]

    for(sig in names(num_significant)){
        if(pval < as.numeric(sig)){
            num_significant[[sig]] = num_significant[[sig]] + 1
        }
    }

    test_grand_mean <- wald.test(Sigma=sigma, b=b-grand_mean, L=matrix(c(1,1), nrow=1, ncol=2))
    pval <- test_grand_mean$result$chi2[["P"]]
    for(sig in names(num_significant_grand_mean)){
        if(pval < as.numeric(sig)){
        print(country)
            num_significant_grand_mean[[sig]] = num_significant_grand_mean[[sig]] + 1
        }
    }
}
print(num_significant)
print(num_significant_grand_mean)