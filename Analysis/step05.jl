log_lik = pointwise_log_likelihoods(
    recovery_model_1.model,
    recovery_chains_model_1
)

loo = psis_loo(log_lik)
loo.estimates