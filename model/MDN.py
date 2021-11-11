import torch
import torch.nn as nn
import torch.distributions as D
import math 

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
    #print(target.unsqueeze(1).shape)
    #print(sigma.shape)
    target = target.unsqueeze(1).expand_as(sigma)
    
    sigma_1 = sigma[:,:,0]
    sigma_2 = sigma[:,:,1]
    mu1 = mu[:,:,0]
    mu2 = mu[:,:,1]
    target_1 = target[:,:,0]
    target_2 = target[:,:,1]
    pho_inverse = 1-pho.pow(2) + 1e-15

    sigma_index = (sigma <= 0).nonzero() #9.0150
    pho_index =  (pho_inverse == 0).nonzero()

    if sigma_index.size(0) > 0:  print(sigma[sigma_index[0][0], sigma_index[0][1]])
    if pho_index.size(0) > 0:  print(pho[pho_index[0][0], pho_index[0][1]])

    ret = (-LOG2PI - torch.log(sigma_1) - torch.log(sigma_2) 
        - 0.5*torch.log(pho_inverse) 
        - 0.5 * torch.pow((target_1-mu1) / sigma_1, 2)/pho_inverse
        - 0.5 * torch.pow((target_2 - mu2) / sigma_2, 2)/pho_inverse
        + pho/pho_inverse * (target_1 - mu1) * (target_2 - mu2) / (sigma_1*sigma_2)
    )

    return ret



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
        
    def forward(self, minibatch):
        pi = self.pi(minibatch)
        sigma = self.sigma_act(self.sigma(minibatch)) + 1 + 1e-15
        
        sigma = sigma.view(-1, self.num_gaussians, self.out_features*2)

        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features*2)

        pho0 = self.pho(minibatch)
        pho = self.pho_act(pho0)
        return pi, sigma, mu, pho

def mdn_loss(pi, mu, sigma, pho, target, para_regu = False, lambda_sigma =0 , lambda_pi=0, lambda_mu=0, lambda_pho=0):
    #print(bivariate_gaussian_probability(sigma, mu, pho, target))
    
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

def sampling_test(pi, mu, sigma, pho):
    batch_size = pi.size(0)
    result = torch.zeros((batch_size, 2))
    for i in range(0, batch_size):
        pi_i = pi[i, ...]
        mu_i = mu[i, ...]
        sigma_i = sigma[i, ...]
        pho_i = pho[i, ...]
        mix = D.Categorical(pi_i)
        idx = mix.sample().item()
        sigma_matrix = torch.Tensor(
                [   [sigma_i[idx,0]**2, pho_i[idx]*sigma_i[idx,0]*sigma_i[idx,1]], 
            [pho_i[idx]*sigma_i[idx,0]*sigma_i[idx,1], sigma_i[idx,1]**2  ]  ] )
        m = D.MultivariateNormal(mu_i[idx], sigma_matrix)
        data = m.sample() 
        result[i, :] = data
    return result      


def sampling(pi, mu, sigma, pho, n ):
    mix = D.Categorical(pi)
    indexes = mix.sample((n, ))

    sigma_matrix = []
    for i in range(0, mu.size(0)):
        sigma_matrix.append(
            torch.Tensor(
                [     [sigma[i,0]**2, pho[i]*sigma[i,0]*sigma[i,1]], 
            [pho[i]*sigma[i,0]*sigma[i,1], sigma[i,1]**2] 
            ] )
        )
    
    sample = []

    for i in range(0, indexes.size(0)):
        idx = indexes[i].item()
        m = D.MultivariateNormal(mu_matrix[idx], sigma_matrix[idx])
        data = m.sample()
        sample.append(data)
    
    sample = np.array(sample)
    sample_avg = np.mean(sample, axis = 0)

def return_expecation_value(pi, mu):
    expectation = torch.zeros(2)
    for i in range(0, mu.size(0)):
        expectation += pi[i]*mu[i]
    return expectation.numpy()
