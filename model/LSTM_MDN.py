import torch
import torch.nn as nn
from model.MDN import MDN, mdn_loss, bivariate_gaussian_probability
from model.LSTM_model import myLSTM


class LSTM_MDN(nn.Module):
    def __init__(self, Config, device):
        super(LSTM_MDN, self).__init__()
        self.Config = Config
        self.rnn = myLSTM(Config, device)
        self.mdn = MDN(Config["input_features"], Config["out_features"], Config["num_gaussians"])
        
        self.para_regu = Config["para_regu"] 
        self.lambda_sigma =Config["lambda_sigma"]
        self.lambda_pi=Config["lambda_pi"]
        self.lambda_mu=Config["lambda_mu"]
        self.lambda_pho=Config["lambda_pho"]

    def loss_calculator(self, pi, mu, sigma, pho, target):
        mix_prob = torch.log(pi) + bivariate_gaussian_probability(sigma, mu, pho, target)
        log_prob = torch.logsumexp(mix_prob, dim = -1)
        loss = torch.mean(-log_prob)
        if self.para_regu:
            sigma_l1_reg = 0
            pi_l1_reg = 0
            mu_l1_reg = 0
            pho_l1_reg = 0

            sigma_params = torch.cat( [x.view(-1) for x in self.mdn.sigma.parameters()] )
            pi_l1_reg = self.lambda_sigma * torch.norm(sigma_params, p = 2)

            pi_params = torch.cat( [x.view(-1) for x in self.mdn.pi.parameters()] )
            pi_l1_reg = self.lambda_pi * torch.norm(pi_params, p = 2)

            mu_params = torch.cat( [x.view(-1) for x in self.mdn.mu.parameters()] )
            mu_l1_reg = self.lambda_mu * torch.norm(mu_params, p = 2)

            pho_params = torch.cat( [x.view(-1) for x in self.mdn.pho.parameters()] )
            pho_l1_reg = self.lambda_pho * torch.norm(pho_params, p = 2)

            loss = loss + sigma_l1_reg + pi_l1_reg + mu_l1_reg + pho_l1_reg 
        return loss 

    def forward(self, minibatch, targets = None):
        rnn_out = self.rnn(minibatch, targets = None)
        pi, sigma, mu, pho = self.mdn(rnn_out)
        out = pi, sigma, mu, pho 
        if (targets is not None):
            loss = self.loss_calculator(pi, mu, sigma, pho, targets)
            '''
            if loss <= 0: 
                print(loss)
                raise KeyboardInterrupt
            '''
            return out, loss
        else:
            return out