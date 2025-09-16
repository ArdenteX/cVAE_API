import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

def reparameterize(mu, log_var):
    # reparameterize
    std = torch.exp(0.5 * log_var)
    epsilon = torch.randn_like(std)
    z = mu + std * epsilon
    return z


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResidualBlock, self).__init__()

        self.shortcut = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.shortcut(x)


class Encoder(nn.Module):
    def __init__(self, i_dim, z_dim, c_dim, num_hidden, mode='condition'):

        super(Encoder, self).__init__()
        self.i_dim = i_dim
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.num_hidden = num_hidden
        self.mode = mode

        self.root_layer = nn.Sequential(
            nn.Linear(self.i_dim + self.c_dim if mode == 'condition' else self.i_dim, self.num_hidden * 2),
            nn.SiLU(),
            nn.Linear(self.num_hidden * 2, self.num_hidden * 2),
            nn.SiLU(),
            nn.Linear(self.num_hidden * 2, self.num_hidden),
            nn.BatchNorm1d(self.num_hidden),
            nn.SiLU()
        )

        # q(z|x)
        self.z_x_mu = nn.Linear(self.num_hidden, self.z_dim)
        self.z_x_logvar = nn.Linear(self.num_hidden, self.z_dim)

        self.shortcut = ResidualBlock(in_dim=self.i_dim, out_dim=self.num_hidden)

        # # q(z|y)
        # self.z_y_mu = nn.Linear(self.num_hidden, self.o_dim)
        # self.z_y_logvar = nn.Linear(self.num_hidden, self.o_dim)
        #
        # # space y reflects to space latent
        # self.fc_l = nn.Linear(self.o_dim, self.z_dim)

    def forward(self, x, y=None):
        """
        Encoder Forward of cVAE

        Parameters:
            x(tensor): Original data
            y(tensor): Observation error data

        Returns:
            mu(tensor): mean of latent distribution
            log var: log variance of latent distribution
        """
        short_cut = self.shortcut(x)

        if x.shape[-1] != self.i_dim + self.c_dim and self.mode == 'condition':
            h = torch.cat([x, y], dim=1)

        else:
            h = x

        h = self.root_layer(h)
        h += short_cut
        h = nn.functional.silu(h)
        # q(z|x)
        l_z_mu = self.z_x_mu(h)
        l_z_logvar = self.z_x_logvar(h)
        z = reparameterize(l_z_mu, l_z_logvar)

        # # q(z|y)
        # p_y_mu = self.z_y_mu(h)
        # p_y_logvar = self.z_y_logvar(h)
        # y = reparameterize(p_y_mu, p_y_logvar)
        #
        # # y reflect to l
        # y_r_l = self.fc_l(y)

        return l_z_mu, l_z_logvar, z


class Decoder(nn.Module):
    def __init__(self, z_dim, c_dim=None, out_dim=None, num_hidden=None, mode='condition'):
        super(Decoder, self).__init__()

        self.z_dim = z_dim
        self.c_dim = c_dim
        self.out_dim = out_dim
        self.num_hidden = num_hidden
        self.mode = mode

        if self.mode == 'condition':
            self.fc_z1 = nn.Linear(self.z_dim + self.c_dim, self.num_hidden * 2)

        else:
            self.fc_z1 = nn.Linear(self.z_dim, self.num_hidden * 2)

        self.fc_z2 = nn.Linear(self.num_hidden * 2, self.num_hidden * 2)

        self.fc_z3 = nn.Linear(self.num_hidden * 2, self.num_hidden)

        self.fc_out = nn.Linear(self.num_hidden, self.out_dim)

    def forward(self, z, y=None):
        """
        Decoder Forward of cVAE

        Parameters:
            z(tensor): latent mean of posterior distribution
            y(tensor): Observation error data

        Returns:
            h(tensor): mean of posterior distribution
        """
        h = self.fc_z1(torch.cat([z, y], dim=1) if self.mode == 'condition' else z)
        h = nn.functional.silu(h)
        h = self.fc_z2(h)
        h = nn.functional.silu(h)
        h = self.fc_z3(h)
        h = nn.functional.silu(h)
        h = self.fc_out(h)
        return h


class cVAE(nn.Module):
    def __init__(self, i_dim=None, c_dim=None, z_dim=None, o_dim=None, num_hidden=None, mode='condition'):
        super(cVAE, self).__init__()
        self.i_dim = i_dim
        self.z_dim = z_dim
        self.o_dim = o_dim
        self.c_dim = c_dim
        self.num_hidden = num_hidden
        self.mode = mode

        self.encoder = Encoder(self.i_dim, self.z_dim, self.c_dim, self.num_hidden, self.mode)
        self.decoder = Decoder(self.z_dim, self.c_dim, self.i_dim, self.num_hidden, self.mode)

    def forward(self, x, y=None):
        """
        cVAE Forward

        Parameters:
            x(tensor): input data contain original data or original data and observation error,
            e.g. input size = 4 then x (batch size, input size) which only contain original data,
            x (batch size, input size * 2) which contain original data and observation error.

            y(tensor): observation error, it can be set as None when x contain those.

        Returns:
            h(tensor): mean of posterior distribution
            mu(tensor): mean of latent distribution
            log var: log variance of latent distribution
        """

        if x.shape[-1] < self.i_dim and y is None and self.mode == 'condition':
            raise ValueError("The set input dim: {}, however check the input data x's dimension: {}"
                             ", please introduce the noise data to x or y".format(self.i_dim, x.shape[-1]))

        if x.shape[-1] == self.i_dim + self.c_dim and self.mode == 'condition':
            y = x[:, 4:]

        l_z_mu, l_z_logvar, z = self.encoder(x, y)
        recon_x = self.decoder(z, y if self.mode == 'condition' else None)

        return recon_x, l_z_mu, l_z_logvar

    def pred_distribution_inference(self, X, times, device):
        condition = []
        pred_distribution = []
        with tqdm(X, 'points') as points:
            for x in points:

                x_ = np.repeat(x.reshape(1, -1), times, axis=0)
                # x_ = cvae_m_x_50th.transform(x_)


                fake_y = np.random.normal(0, 1, (times, 7 * self.i_dim))
                # fake_y = np.repeat(fake_y, times, axis=0)

                curr_prediction = self.decoder(torch.from_numpy(fake_y).to(torch.float).to(device),
                                               torch.from_numpy(x_).to(torch.float).to(device))

                curr_prediction = curr_prediction.detach().cpu().numpy()
                condition.append(x_)
                pred_distribution.append(curr_prediction)

        condition = np.vstack(condition)
        pred_distribution = np.vstack(pred_distribution)

        return condition, pred_distribution

    def posterior_distribution_inference(self, c, pred_d, device):
        posterior_distribution = self.encoder(torch.from_numpy(pred_d).to(torch.float).to(device),
                                              torch.from_numpy(c).to(torch.float).to(device))

        z = posterior_distribution[-1].detach().cpu().numpy()
        return z



class CombineLoss(nn.Module):
    def __init__(self):
        super(CombineLoss, self).__init__()
        self.re_mse = nn.MSELoss()

    def reconstruction_loss(self, recon_x, x):
        return self.re_mse(recon_x, x)

    def KLD_Loss(self, l_z_mu, l_z_logvar):
        return -0.5 * torch.sum(1 + l_z_logvar - torch.square(l_z_mu) - l_z_logvar.exp())

    def NLLLoss(self, p_y_mu, p_y_logvar, y):
        z_score = torch.divide(torch.square((p_y_mu - y)), torch.exp(p_y_logvar))

        normal_loglik = 0.5 * z_score + 0.5 * p_y_logvar

        # loglik = -torch.logsumexp(normal_loglik, dim=-1)
        loglik = torch.logsumexp(normal_loglik, dim=-1)

        return loglik.mean()

    def forward(self, recon_x, x, l_z_mu, l_z_logvar, beta=1.0, auto_adj=(), eta=0.01):
        """
        Combination of loss function
            Reconstruction loss: reconstruction x, x
            KLD loss: latent z mu, latent z log var

        Parameters:
            recon_x(tensor): Generated data
            x(tensor): Observation
            l_z_mu(tensor): Mean of latent distribution
            l_z_logvar(tensor): Log variance of latent distribution
            beta(float): Beta-VAE
            auto_adj(tuple): Auto adjust kld in a range
            eta: Degree of adjust

        Returns:
            loss(tensor): loss combination with NLLLoss and KLD

        """
        # Reconstruction Loss
        recon_loss = self.reconstruction_loss(recon_x, x)

        # KLD
        kld = self.KLD_Loss(l_z_mu, l_z_logvar)

        # NLLLoss
        # nllloss = self.NLLLoss(p_y_mu, p_y_logvar, y)

        # print("recon_loss: {}, kld: {}, nllloss: {}".format(recon_loss, kld, nllloss))

        kl_ratio = beta * kld / (recon_loss + beta * kld)

        if auto_adj:
            if auto_adj[0] > kl_ratio:
                beta = beta * (1 + eta * (auto_adj[0] - kl_ratio))

            if auto_adj[1] < kl_ratio:
                beta = beta * (1 - eta * (kl_ratio - auto_adj[1]))

        return torch.mean(recon_loss + beta * kld), recon_loss, kld
