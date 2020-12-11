# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import torch
import torch.nn as nn
import torchvision.models as models

from .layers import build_mlp



class VidEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, long_term, hidden_dims=None, norm='none', act='relu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.long_term = long_term

        if hidden_dims is None:
            layers = [input_dim * 2, output_dim]
            self.encoder1 = None
            self.mu_head = nn.Linear(layers[0], output_dim)
            self.logvar_head = nn.Linear(layers[0], output_dim)
        else:
            layers = (input_dim * 2,) + hidden_dims
            self.encoder1 = build_mlp(layers, batch_norm=norm, activation=act, final_nonlinearity=True)
            self.mu_head = nn.Linear(layers[-1], output_dim)
            self.logvar_head = nn.Linear(layers[-1], output_dim)

    def forward(self, vid_batch):
        """
        Called during training. 1. encode to u 2. resample
        :return: obj_z: (Dt, V, D), kl_loss: criterion
        """
        obj_z, kl_loss, ori_z, src_feats = self._forward(vid_batch, True)
        return obj_z, kl_loss, ori_z, src_feats

    def no_sample(self, vid_batch):
        """
        Called during realistic testing. 1. encode u. 2. NO sample
        :return: obj_z: (Dt, V, D), kl_loss: criterion
        """
        obj_z, kl_loss, ori_z, src_feats = self._forward(vid_batch, False)
        return obj_z, kl_loss, ori_z, src_feats

    def _forward(self, vid_batch, sample=True):
        raise NotImplementedError

    def batch_sample(self, V, time_len, seed, image):
        """
        Called during inception. Don't encode. sample from N(0, 1)
        :param V:
        :param time_len:
        :param seed:
        :return:
        """
        if seed is not None:
            torch.manual_seed(seed)
        u = torch.randn(V, self.output_dim).unsqueeze(0).cuda()  # (1, V, D)
        img_z = self.generate_z_from_u(u, time_len, V)
        return img_z

    def reparameterize(self, mu, logvar, sample):
        """generate sample from N(mu, var). or no sample"""
        if self.training and sample:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)  ## (0, 1)
            return eps.mul(std).add_(mu)
            # return mu
        else:
            # mu is encoded by (src, dst) of the current data point.
            return mu

    def mu_logvar(self, input_feat):
        out = input_feat
        if self.encoder1 is not None:
            out = self.encoder1(input_feat)
        mu = self.mu_head(out)
        logvar = self.logvar_head(out)

        # mu = nn.functional.normalize(mu)
        return mu, logvar

    def generate_z_from_u(self, u, dt, V):
        raise NotImplementedError


class TrajHierarchy(VidEncoder):
    def __init__(self, feat_dim_list, output_dim, dt, show_length, long_term, hidden_dims, norm, act):
        """under construction!!!"""
        assert len(feat_dim_list) == 1
        input_dim = feat_dim_list[0]
        super().__init__(input_dim, output_dim, long_term, hidden_dims, norm, act)
        self.dt = dt
        self.show_length = show_length

        self.lstm_layers = 1
        self.lstm = nn.LSTM(output_dim, output_dim, self.lstm_layers)

        self.mylstm = nn.LSTM(512, 512, self.lstm_layers)

        resent = models.resnet18(pretrained=True)
        modules = list(resent.children())[:-1]
        self.image_tower = nn.Sequential(*modules)

    def _forward(self, vid, sample=True):
        dt, V, _, C, H, W = vid['image'].size()
        dst_image = vid['image'][self.long_term].squeeze(1)

        # src_image = vid['image'][0].squeeze(1)
        # src_feat = self.image_tower(src_image).view(V, -1)
        # self.show_length = 4
        src_feats = []
        for i in range(self.dt):
            src_image = vid['image'][i].squeeze(1)
            src_feat = self.image_tower(src_image).view(V, -1)
            src_feats.append(src_feat)

        src_feat, _ = self.mylstm(torch.stack(src_feats[:4]))
        src_feat = torch.sum(src_feat, dim=0)

        dst_feat = self.image_tower(dst_image).view(V, -1)  ## (1, 512)
        src_dst = torch.cat([src_feat, dst_feat], dim=-1)
        long_u_mu, logvar = self.mu_logvar(src_dst)
        reparam = self.reparameterize(long_u_mu, logvar, sample).unsqueeze(0)  # (1, V, D)

        # (Dt, V, D)
        img_z = self.generate_z_from_u(reparam, dt-self.show_length+1, V)
        kl_loss = -0.5 * torch.mean(1 + logvar - long_u_mu.pow(2) - logvar.exp())
        return img_z, kl_loss, long_u_mu, src_feats

    def generate_z_from_u(self, u, dt, V):
        """
        :param u: (1, V, D)
        :return: (dt, V, D)
        """
        h0 = torch.zeros(u.size()).to(u)
        zeros = torch.zeros(u.size()).to(u).expand(dt, V, self.output_dim)
        # [1, V, 8], [dt, V, 8], [1, V, 8]
        img_z, (h0, c0) = self.lstm(zeros, (h0, u))
        return img_z


def kl_criterion(mu1, logvar1, mu2, logvar2):
    sigma1 = logvar1.mul(0.5).exp()
    sigma2 = logvar2.mul(0.5).exp()
    kld = torch.log(sigma2 / sigma1) + (torch.exp(logvar1) + (mu1 - mu2) ** 2) / (2 * torch.exp(logvar2)) - 1 / 2
    return kld.mean()



EncoderFactory = {
    'traj': TrajHierarchy,
    # 'noZ': ImageNoZ,
    # 'fp': ImageFixPrior,
    # 'lp': ImageLearnedPrior,
}
