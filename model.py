import torch
import torch.nn as nn

from modules import Encoder, CategoricalDecoder
from loss import ELBO_Loss


class MusicVAE(nn.Module):
    def __init__(self, input_size, output_size, latent_dim,
                 hidden_size=2048,
                 decoder_hidden_size=1024,
                 decoder_num_layers=2,
                 encoder_num_layers=2,
                 segment_length=32,
                 teacher_forcing_ratio=0.5,
                 device='cpu'):
        super(MusicVAE, self).__init__()

        self.encoder = Encoder(input_size,
                               latent_dim,
                               hidden_size,
                               encoder_num_layers)

        self.decoder = CategoricalDecoder(output_size=output_size,
                                          latent_dim=latent_dim,
                                          decoder_hidden_size=decoder_hidden_size, # noqa E501
                                          decoder_num_layers=decoder_num_layers, # noqa E501
                                          segment_length=segment_length,
                                          teacher_forcing_ratio=teacher_forcing_ratio, # noqa E501
                                          device=device)

    def forward(self, x):
        mu, sigma, z = self.encoder(x)
        out = self.decoder(z)

        return out, mu, sigma, z



if __name__ == '__main__':
    model = MusicVAE(input_size=90,
                     output_size=90,
                     latent_dim=512)
    input = torch.rand(16, 32, 90)
    out, mu, sigma, z = model(input)
    print(ELBO_Loss(input, mu, sigma, out))
