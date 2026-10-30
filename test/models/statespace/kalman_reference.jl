# Closed-form reference for the linear Gaussian state space models in this folder,
# computed without message passing:
#
#   x[0] ~ N(prior_mean, prior_cov)
#   x[t] = A * x[t - 1] + u + w[t],  w[t] ~ N(0, Q)
#   y[t] = C * x[t] + v[t],          v[t] ~ N(0, R)
#
# The forward pass is the Kalman filter, the backward pass is the Rauch-Tung-Striebel
# smoother and the log evidence `log p(y[1:n])` is the sum of the one-step-ahead
# predictive log densities. Scalars and vectors/matrices both work, so `Q = 0` covers a
# deterministic transition such as `x[t] ~ x[t - 1] + c`.
#
# Belief propagation is exact on these chain-structured models, so RxInfer's marginals
# must match the smoothed posteriors and its Bethe free energy must equal `-log_evidence`.

function gaussian_logpdf(residual, covariance)
    return -(logdet(2π * covariance) + dot(residual, covariance \ residual)) / 2
end

function kalman_rts_reference(y, prior_mean, prior_cov, A, u, C, Q, R)
    n = length(y)
    predicted_means = Vector{typeof(prior_mean)}(undef, n)
    predicted_covs = Vector{typeof(prior_cov)}(undef, n)
    filtered_means = Vector{typeof(prior_mean)}(undef, n)
    filtered_covs = Vector{typeof(prior_cov)}(undef, n)
    log_evidence = 0.0

    mean_prev, cov_prev = prior_mean, prior_cov
    for t in 1:n
        predicted_mean = A * mean_prev + u
        predicted_cov = A * cov_prev * A' + Q
        innovation = y[t] - C * predicted_mean
        innovation_cov = C * predicted_cov * C' + R
        log_evidence += gaussian_logpdf(innovation, innovation_cov)

        gain = predicted_cov * C' / innovation_cov
        filtered_mean = predicted_mean + gain * innovation
        filtered_cov = predicted_cov - gain * innovation_cov * gain'
        filtered_cov = (filtered_cov + filtered_cov') / 2

        predicted_means[t], predicted_covs[t] = predicted_mean, predicted_cov
        filtered_means[t], filtered_covs[t] = filtered_mean, filtered_cov
        mean_prev, cov_prev = filtered_mean, filtered_cov
    end

    smoothed_means = copy(filtered_means)
    smoothed_covs = copy(filtered_covs)
    for t in (n - 1):-1:1
        smoother_gain = filtered_covs[t] * A' / predicted_covs[t + 1]
        smoothed_means[t] =
            filtered_means[t] +
            smoother_gain * (smoothed_means[t + 1] - predicted_means[t + 1])
        smoothed_cov =
            filtered_covs[t] +
            smoother_gain * (smoothed_covs[t + 1] - predicted_covs[t + 1]) * smoother_gain'
        smoothed_covs[t] = (smoothed_cov + smoothed_cov') / 2
    end

    return (
        smoothed_means = smoothed_means,
        smoothed_covs = smoothed_covs,
        log_evidence = log_evidence,
    )
end
