import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding, TokenEmbedding
from .ctc import ConceptTransformer
from .ctc import OTmodel


class EncoderLayer(nn.Module):
    def __init__(self, args, d_model, d_ff=None, dropout=0.1, activation="relu", win_size=20, output_attention=20,
                 n_heads=1, device='cpu'):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        attention = AttentionLayer(
            AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention,
                             device=device),
            d_model, n_heads)
        self.args = args
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, device=device)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, device=device)
        self.norm1 = nn.LayerNorm(d_model, device=device)
        self.norm2 = nn.LayerNorm(d_model, device=device)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, 0


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        series_list = []
        concept_att_list = []
        for attn_layer in self.attn_layers:
            x, series, concept_att = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            concept_att_list.append(concept_att)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, concept_att_list


class Decoder(nn.Module):
    def __init__(self, args, device, model_state='train'):
        super(Decoder, self).__init__()
        self.args = args
        self.device = device
        self.OTmodel = OTmodel(
            args=self.args, device=self.device, model_state=model_state
        )

        self.ma_mean_fc = nn.Linear(self.args.d_vae, self.args.d_model_concept)
        self.ma_logvar_fc = nn.Linear(self.args.d_vae, self.args.d_model_concept)

        # self.fc_alphas = nn.Linear(self.args.d_vae, self.args.d_c)

        self.z_mean_fc = nn.Linear(self.args.d_vae + self.args.d_model_concept, self.args.d_vae)
        self.z_logvar_fc = nn.Linear(self.args.d_vae + self.args.d_model_concept, self.args.d_vae)

        self.encode_h1 = nn.Linear(self.args.d_vae, self.args.d_vae)
        self.encode_h2 = nn.Linear(self.args.d_vae, self.args.d_vae)

        self.decoer_z_mean = nn.Sequential(
            nn.Linear(self.args.d_c, self.args.d_vae),
            nn.ReLU(),
            nn.Linear(self.args.d_vae, self.args.d_vae)
        )
        self.decoer_z_logvar = nn.Sequential(
            nn.Linear(self.args.d_c, self.args.d_vae),
            nn.ReLU(),
            nn.Linear(self.args.d_vae, self.args.d_vae)
        )

        self.decoer_x_mean = nn.Sequential(
            nn.Linear(self.args.d_vae, self.args.d_vae),
            nn.ReLU(),
            nn.Linear(self.args.d_vae, self.args.d_vae)
        )

        self.res_pass = nn.Conv1d(self.args.T, self.args.T, kernel_size=1, stride=1, bias=True)

    def forward(self, h):

        # Encoder inference
        # h0 = h
        h1 = self.encode_h1(h)
        h2 = self.encode_h2(h1)

        # h0 = h
        # h1 = h
        # h2 = h

        # h0 = torch.zeros(h.size()).to(self.device)
        # h1 = torch.zeros(h.size()).to(self.device)
        # h2 = h

        # out, OTloss, loss_mse = self.OTmodel(h2)
        # out = h

        # temp_out = h1

        ma_mean = self.ma_mean_fc(h2)
        ma_logvar = self.ma_logvar_fc(h2)
        ma_sample = self.reparameterize(ma_mean, ma_logvar)

        # alphas_out = self.fc_alphas(temp_out)
        # alphas_out = F.softmax(alphas_out, dim=-1)
        # alhpa_sample = self.sample_gumbel_softmax(alphas_out)

        h_box = torch.cat([h1, ma_sample], dim=2)

        z_mean = self.z_mean_fc(h_box)
        z_logvar = self.z_logvar_fc(h_box)
        z_sample = self.reparameterize(z_mean, z_logvar)

        # Decoder
        temp_out_dir = self.OTmodel(ma_sample)

        out = temp_out_dir['embadding']
        OTloss = temp_out_dir['loss']
        loss_mse = temp_out_dir['loss_mse']

        # out = alhpa_sample

        # # Res pass controling the proportion between memory and prior
        # out = out + self.res_pass(alhpa_sample)

        # out_box = torch.cat([out, alhpa_sample], dim=2)
        out_box = out

        decoder_z_mean_out = self.decoer_z_mean(out_box)
        decoder_z_logvar_out = self.decoer_z_logvar(out_box)

        decoder_x_mean_out = self.decoer_x_mean(z_sample)

        KL_c = self.KL_loss_normal(ma_mean, ma_logvar)
        KL_z = self.KL_loss(z_mean, z_logvar, decoder_z_mean_out, decoder_z_logvar_out)

        # z_sample = out
        # return decoder_x_mean_out, KL_c, KL_z,  OTloss, loss_mse
        return {
            'decoder_x_mean_out': decoder_x_mean_out,
            'KL_c': KL_c,
            'KL_z': KL_z,
            'OTloss': OTloss,
            'loss_mse': loss_mse,
            'p_xj': temp_out_dir['p_xj'],
            'theta': temp_out_dir['theta']
        }

    def sample_gumbel_softmax(self, alpha):
        """
        Samples from a gumbel-softmax distribution using the reparameterization
        trick.

        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the gumbel-softmax distribution. Shape (N, D)
        """
        EPS = 1e-12

        unif = torch.rand(alpha.size())
        unif = unif.to(self.device)
        gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
        # Reparameterize to create gumbel softmax sample
        log_alpha = torch.log(alpha + EPS)
        logit = (log_alpha + gumbel) / self.args.softmax_t
        return F.softmax(logit, dim=-1)

        # if self.training:
        #     # Sample from gumbel distribution
        #     unif = torch.rand(alpha.size())
        #     unif = unif.to(self.device)
        #     gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
        #     # Reparameterize to create gumbel softmax sample
        #     log_alpha = torch.log(alpha + EPS)
        #     logit = (log_alpha + gumbel) / self.args.softmax_t
        #     return F.softmax(logit, dim=-1)
        # else:
        #
        #     return alpha

        # if self.training:
        #     # Sample from gumbel distribution
        #     unif = torch.rand(alpha.size())
        #     unif = unif.to(self.device)
        #     gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
        #     # Reparameterize to create gumbel softmax sample
        #     log_alpha = torch.log(alpha + EPS)
        #     logit = (log_alpha + gumbel) / self.args.softmax_t
        #     return F.softmax(logit, dim=-1)
        # else:
        #     # In reconstruction mode, pick most likely sample
        #     _, max_alpha = torch.max(alpha, dim=-1)
        #     one_hot_samples = torch.zeros(alpha.size()).view(-1, alpha.size()[-1])
        #     # On axis 1 of one_hot_samples, scatter the value 1 at indices
        #     # max_alpha. Note the view is because scatter_ only accepts 2D
        #     # tensors.
        #     one_hot_samples.scatter_(1, max_alpha.view(-1, 1).data.cpu(), 1)
        #
        #     one_hot_samples = one_hot_samples.to(self.device)
        #     return one_hot_samples.view(alpha.size())

    def reparameterize(self, posterior_mean, posterior_logvar):
        posterior_var = posterior_logvar.exp()
        # take sample

        if self.training:
            eps = torch.zeros_like(posterior_var).normal_()
            z = posterior_mean + posterior_var.sqrt() * eps  # reparameterization
        else:
            z = posterior_mean
        # z = posterior_mean
        return z

    def KL_loss_normal(self, posterior_mean, posterior_logvar):
        KL = -0.5 * torch.mean(1 - posterior_mean ** 2 + posterior_logvar -
                               torch.exp(posterior_logvar), dim=1)
        return torch.mean(KL)

    def KL_loss(self, p_m, p_lv, q_m, q_lv):
        q_div_p = torch.div(q_lv.exp(), p_lv.exp())
        kl = 0.5 * torch.mean((q_m - p_m) ** 2 / p_lv.exp() + (q_div_p - 1 - torch.log(q_div_p)), dim=1)
        return torch.mean(kl)

    def KL_discrete_loss(self, alpha):
        """
        Calculates the KL divergence between a categorical distribution and a
        uniform categorical distribution.

        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the categorical or gumbel-softmax distribution.
            Shape (N, D)
        """
        EPS = 1e-12
        disc_dim = int(alpha.size()[-1])
        log_dim = torch.Tensor([np.log(disc_dim)])

        log_dim = log_dim.to(self.device)
        # Calculate negative entropy of each row
        neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=-1)
        # Take mean of negative entropy across batch
        mean_neg_entropy = torch.mean(neg_entropy)
        # KL loss of alpha with uniform categorical variable
        kl_loss = log_dim + mean_neg_entropy
        return kl_loss


class AnomalyTransformer(nn.Module):
    def __init__(self, args, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True, device='cpu', model_state='train'):
        super(AnomalyTransformer, self).__init__()
        self.state = model_state
        self.output_attention = output_attention
        self.args = args
        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout, device)

        # Encoder
        self.encoder = Encoder(
            [EncoderLayer(args=args, d_model=d_model, d_ff=d_ff, dropout=dropout, activation=activation,
                          win_size=win_size,
                          output_attention=win_size, n_heads=n_heads, device=device) for l in range(e_layers)],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.decoder = Decoder(args, device, self.state)
        self.projection = nn.Linear(self.args.d_vae * self.args.T, c_out * self.args.T, bias=True, device=device)

        self.en2de = nn.Linear(self.args.d_model, self.args.d_vae, bias=True, device=device)

        self.losses = LossFunctions()

    def loss_fn(self, x, cate_posterior, c_mean_prior, x_mu, x_logsigma):
        batch_size = x.size(0)

        loglikelihood = self.losses.log_normal(x.float(), x_mu.float(), torch.from_numpy(np.array(1)))

        # if -loglikelihood > 0:
        #     a = self.losses.log_normal(x.float(), x_mu.float(), torch.pow(torch.exp(x_logsigma.float()), 2))
        # else:
        #     a = self.losses.log_normal(x.float(), x_mu.float(), torch.pow(torch.exp(x_logsigma.float()), 2))

        # return (-loglikelihood) / batch_size
        return -loglikelihood

    def loss_fn_mse(self, x, x_mu):
        loss = F.mse_loss(x, x_mu)
        return loss

    def forward(self, x):
        enc_out = self.embedding(x)
        enc_out, series, concept_att = self.encoder(enc_out)
        enc_out = self.en2de(enc_out)
        temp_out_dir = self.decoder(enc_out)
        enc_out = temp_out_dir['decoder_x_mean_out']
        KL_ma = temp_out_dir['KL_c']
        KL_z = temp_out_dir['KL_z']
        otloss = temp_out_dir['OTloss']
        loss_mse = temp_out_dir['loss_mse']

        enc_out = self.projection(enc_out.view(enc_out.size()[0], -1))
        # enc_out = self.projection2(enc_out.view(enc_out.size()[0], -1))

        enc_out = enc_out.view(enc_out.size()[0], self.args.T, -1)

        Other = {
            'p_xj': temp_out_dir['p_xj'],
            'theta':  temp_out_dir['theta'],

        }
        if self.output_attention:
            return enc_out, series, otloss, loss_mse, KL_ma, KL_z, Other
            # return enc_out, series, concept_att, 0, 0
        else:
            return enc_out  # [B, L, D]


class LossFunctions:
    eps = 1e-8

    def log_normal(self, x, mu, var):
        """Logarithm of normal distribution with mean=mu and variance=var
           log(x|μ, σ^2) = loss = -0.5 * Σ log(2π) + log(σ^2) + ((x - μ)/σ)^2

        Args:
           x: (array) corresponding array containing the input
           mu: (array) corresponding array containing the mean
           var: (array) corresponding array containing the variance

        Returns:
           output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
        """
        if self.eps > 0.0:
            var = var + self.eps
        # return -0.5 * torch.sum(
        #     np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var)
        return -0.5 * torch.mean(
            np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var)

    def entropy(self, logits, targets):
        """Entropy loss
            loss = (1/n) * -Σ targets*log(predicted)

         Args:
            logits: (array) corresponding array containing the logits of the categorical variable
            real: (array) corresponding array containing the true labels

         Returns:
            output: (array/float) depending on average parameters the result will be the mean
                                of all the sample losses or an array with the losses per sample
        """
        log_q = F.log_softmax(logits, dim=-1)
        return -torch.mean(torch.sum(targets * log_q, dim=-1))
