import torch
import torch.nn as nn

from layers import Linear, LSTM, BiLSTM


class Encoder(nn.Module):
    def __init__(self, input_size, latent_dim,
                 features_size=13,
                 hidden_size=2048,
                 num_layers=2):
        super(Encoder, self).__init__()

        self.bilstm_layer = BiLSTM(input_size, hidden_size, num_layers) # noqa 501
        self.fc_mu = Linear(hidden_size * 2 + features_size, latent_dim)
        self.fc_sigma = Linear(hidden_size * 2 + features_size, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x, features=None):

        x, _ = self.bilstm_layer(x)
        x = x[:, -1, :]

        if features is not None:
            x = torch.cat([x, features], dim=-1)

        mu = self.fc_mu(x)
        raw_sigma = self.fc_sigma(x)
        sigma = torch.log(torch.exp(raw_sigma) + 1)
        # Clamp sigma to avoid division by zero later
        sigma = sigma.clamp_min(1e-5)

        with torch.no_grad():
            epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon

        return mu, sigma, z


class CategoricalDecoder(nn.Module):
    def __init__(self, output_size, latent_dim,
                 features_size=13,
                 projection_hidden_size=512,
                 decoder_hidden_size=1024,
                 decoder_num_layers=2,
                 segment_length=32,
                 teacher_forcing_ratio=0.5,
                 device='cpu'):
        super(CategoricalDecoder, self).__init__()
        self.device = device
        self.decoder_num_layers = decoder_num_layers
        self.projection_hidden_size = projection_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.segment_length = segment_length
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.projection_layer = nn.Sequential(
            Linear(latent_dim + features_size, projection_hidden_size),
            nn.Tanh()
        )

        self.decoder = LSTM(projection_hidden_size + output_size,
                            decoder_hidden_size,
                            decoder_num_layers)

        self.decoder_fc = nn.Sequential(
            Linear(decoder_hidden_size, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, z, features=None, target=None):
        batch_size = z.size(0)
        if features is not None:
            z = torch.cat([z, features], dim=-1)

        z = self.projection_layer(z)

        z = z.unsqueeze(1).repeat(1, self.decoder_num_layers, 1)
        prev_note, out = torch.zeros(batch_size, self.decoder_num_layers, self.output_size).to(self.device), torch.zeros(batch_size, self.segment_length, self.output_size).to(self.device) # noqa 501
        state_dec = self.init_hidden_decoder(batch_size)
        if target is not None:
            use_teacher_forcing = self.use_teacher_forcing()

        if target is not None and use_teacher_forcing:
            decoder_input = torch.cat([target, z], dim=-1)
            out, _ = self.decoder(decoder_input, state_dec)
            out = self.decoder_fc(out)
        else:
            for note_idx in range(self.segment_length):
                decoder_input = torch.cat([prev_note, z], dim=-1)
                prev_note, state_dec = self.decoder(decoder_input, state_dec)
                prev_note = self.decoder_fc(prev_note)
                out[:, note_idx, :] = prev_note[:, -1, :]
        return out

    def use_teacher_forcing(self):
        return torch.rand(1).item() < self.teacher_forcing_ratio

    def init_hidden_decoder(self, batch_size):
        return (torch.zeros(self.decoder_num_layers, batch_size, self.decoder_hidden_size).to(self.device), # noqa 501
                torch.zeros(self.decoder_num_layers, batch_size, self.decoder_hidden_size).to(self.device)) # noqa 501


class HierarchicalDecoder(nn.Module):
    def __init__(self, latent_dim, output_size,
                 conductor_hidden_size=1024,
                 conductor_output_size=512,
                 conductor_num_layers=2,
                 decoder_hidden_size=1024,
                 decoder_num_layers=2,
                 num_segments=16,
                 segment_length=16,
                 max_sequence_length=256,
                 teacher_forcing_ratio=0.5):
        super(HierarchicalDecoder, self).__init__()
        self.conductor_num_layers = conductor_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.conductor_hidden_size = conductor_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.num_segments = num_segments
        self.segment_length = segment_length
        self.max_sequence_length = max_sequence_length
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.projection_layer = nn.Sequential(
            Linear(latent_dim, conductor_hidden_size),
            nn.Tanh()
        )

        self.conductor = LSTM(conductor_hidden_size,
                              conductor_hidden_size,
                              conductor_num_layers) # noqa 501

        self.conductor_fc = nn.Sequential(
            Linear(conductor_hidden_size, conductor_output_size),
            Linear(conductor_output_size, conductor_output_size),
            nn.Tanh()
        )

        self.decoder = LSTM(conductor_output_size + output_size,
                            decoder_hidden_size,
                            decoder_num_layers) # noqa 501

        self.decoder_fc = nn.Sequential(
            Linear(decoder_hidden_size, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, z, target=None):
        batch_size = z.size(0)

        z = self.projection_layer(z)  # (batch_size, latent_dim)
        z = z.unsqueeze(1).repeat(1, self.conductor_num_layers, 1) # (batch_size, conductor_num_layers, conductor_hidden_size) # noqa 501
        prev_note, out = torch.zeros(batch_size, self.decoder_num_layers, self.output_size), torch.zeros(batch_size, self.max_sequence_length, self.output_size) # noqa 501
        state, state_dec = self.init_hidden_conductor(batch_size), self.init_hidden_decoder(batch_size) # noqa 501

        if target is not None:
            eps = torch.rand(1).item()
            use_teacher_forcing = eps < self.teacher_forcing_ratio

        for sequence_idx in range(self.num_segments):
            embedding, state = self.conductor(z, state) # (batch_size, conductor_output_size) # noqa 501
            embedding = self.conductor_fc(embedding) # (batch_size, conductor_output_size) # noqa 501

            if target is not None and use_teacher_forcing:
                embedding = embedding.expand(batch_size, self.segment_length, self.conductor_output_size) # noqa 501
                idx = range(sequence_idx * self.segment_length, (sequence_idx + 1) * self.segment_length) # noqa 501
                decoder_input = torch.cat(target[:, idx, :], embedding, dim=-1)
                prev_note, state = self.decoder(decoder_input, state_dec)
                prev_note = self.decoder_fc(prev_note)
                out[:, idx, :] = prev_note
                prev_note = prev_note.unsqueeze(1)
            else:
                for note_idx in range(self.segment_length):
                    decoder_input = torch.cat([prev_note, embedding], dim=-1) # noqa 501
                    prev_note, state_dec = self.decoder(decoder_input, state_dec) # noqa 501
                    prev_note = self.decoder_fc(prev_note)

                    idx = sequence_idx * self.segment_length + note_idx
                    out[:, idx, :] = prev_note.squeeze(1)
        return out

    def use_teacher_forcing(self):
        return torch.rand(1).item() < self.teacher_forcing_ratio

    def init_hidden_conductor(self, batch_size):
        return (torch.zeros(batch_size, self.conductor_num_layers, self.conductor_hidden_size), # noqa 501
                torch.zeros(batch_size, self.conductor_num_layers, self.conductor_hidden_size)) # noqa 501

    def init_hidden_decoder(self, batch_size):
        return (torch.zeros(batch_size, self.decoder_num_layers, self.decoder_hidden_size), # noqa 501
                torch.zeros(batch_size, self.decoder_num_layers, self.decoder_hidden_size)) # noqa 501


if __name__ == '__main__':
    encoder = Encoder(90, 512)
    decoder = CategoricalDecoder(90, 512)
    seq_length = 32
    batch_size = 3
    input = torch.rand(batch_size, seq_length, 90)
    features = torch.rand(batch_size, 13)
    mu, sigma, z = encoder(input, features)
    out = decoder(z, features)
    print(out)
