import torch
import torch.nn as nn
import torch.distributions as D

ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)
LOG2PI = math.log(2 * math.pi)

def bivariate_gaussian_probability(sigma, mu, pho, target):
    """Returns the probability of `target` given MoG parameters `sigma` and `mu`.
    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        pho (BXG)
        target (BxO): A batch of target. B is the batch size and I is the number of
            input dimensions.
    Returns:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """
    target = target.unsqueeze(1).expand_as(sigma)
    sigma_1 = sigma[:,:,0]
    sigma_2 = sigma[:,:,1]
    mu_1 = mu[:,:,0]
    mu_2 = mu[:,:,1]
    target_1 = target[:,:,0]
    target_2 = target[:,:,1]

    pho_inverse = 1-pho.pow(2)


    ret = (-LOG2PI - torch.log(sigma_1) - torch.log(sigma_2) 
        - 0.5*torch.log(pho_inverse) 
        - 0.5 * torch.pow((target_1 – mu1) / sigma1, 2)/pho_inverse
        - 0.5 * torch.pow((target_2 – mu2) / sigma2, 2)/pho_inverse
        + pho/pho_inverse * (target_1 – mu1) * (target_2 – mu2) / (sigma1*sigma2)
    )

    return torch.prod(ret, 2)



class MDN(nn.Module):
    """A mixture density network layer
    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.
    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions
    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.
    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """

    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians

        self.pi = nn.Sequential(
            nn.Linear(in_features, num_gaussians),
            nn.Softmax(dim=1)
        )

        self.sigma = nn.Linear(in_features, out_features * num_gaussians *2)
        self.sigma_act = nn.ELU()

        self.mu = nn.Linear(in_features, out_features * num_gaussians * 2)

        self.pho = nn.Linear(in_features, out_features * num_gaussians)
        self.pho_act = nn.Tanh()

    def mdn_loss(pi, mu, sigma, pho, target):
        mix_prob = torch.log(pi) + bivariate_gaussian_probability(sigma, mu, pho, target)
        log_prob = torch.logsumexp(mix_prob, dim = -1)
        loss = torch.mean(-log_prob)

        if para_regu:
            sigma_l1_reg = 0
            pi_l1_reg = 0
            mu_l1_reg = 0
            pho_l1_reg = 0

            sigma_params = torch.cat( [x.view(-1) for x in self.sigma.parameters()] )
            pi_l1_reg = lambda_sigma * torch.norm(sigma_params, p = 2)

            pi_params = torch.cat( [x.view(-1) for x in self.pi.parameters()] )
            pi_l1_reg = lambda_pi * torch.norm(pi_params, p = 2)

            mu_params = torch.cat( [x.view(-1) for x in self.mu.parameters()] )
            mu_l1_reg = lambda_mu * torch.norm(mu_params, p = 2)

            pho_params = torch.cat( [x.view(-1) for x in self.pho.parameters()] )
            pho_l1_reg = lambda_pho * torch.norm(pho_params, p = 2)

            loss = loss + sigma_l1_reg + pi_l1_reg + mu_l1_reg + pho_l1_reg 

        return loss
        
    def forward(self, minibatch):
        pi = self.pi(minibatch)
        sigma = self.sigma_act(self.sigma(minibatch)) + 1 + 1e-15
        sigma = sigma.view(-1, self.num_gaussians, self.out_features*2)

        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features,)

        pho = self.pho_act(self.pho(minibatch))
        return pi, sigma, mu


    def sampling(self, pi, mu, sigma, pho):
        mix = D.Categorical(pi)
        pis = categorical.sample().unsqueeze(1)