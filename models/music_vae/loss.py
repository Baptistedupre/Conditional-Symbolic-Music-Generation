import torch
from torch.nn.functional import binary_cross_entropy
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence


def ELBO_Loss(pred, mu, sigma, target, beta=1.0):

    likelihood = -binary_cross_entropy(pred, target, reduction='sum')

    mu_prior, sigma_prior = torch.zeros_like(mu), torch.ones_like(sigma)
    
    # Ensure sigma values are strictly positive
    epsilon = 1e-5
    sigma = sigma.clamp_min(epsilon)
    sigma_prior = sigma_prior.clamp_min(epsilon)
    
    p, q = Normal(mu_prior, sigma_prior), Normal(mu, sigma)

    kl_div = kl_divergence(q, p)

    elbo = torch.mean(likelihood) - (beta * torch.mean(kl_div))

    return elbo
