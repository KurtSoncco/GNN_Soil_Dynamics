import torch
import torch.nn as nn

# Define the Encoder LSTM
class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=False):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(x.device)
        out, (h1, c1) = self.lstm(x, (h0, c0))
        return out, (h1, c1)

class DecoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional_encoder=False):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional_encoder = bidirectional_encoder
        self.num_directions = 2 if bidirectional_encoder else 1
        self.lstm = nn.LSTM(input_size, hidden_size * self.num_directions, num_layers, batch_first=True)

    def forward(self, x, hidden):
        if self.bidirectional_encoder:
            # Concatenate the hidden states from both directions
            h0, c0 = hidden
            h0 = h0.view(self.num_layers, self.num_directions, -1, self.hidden_size)
            c0 = c0.view(self.num_layers, self.num_directions, -1, self.hidden_size)
            h0 = torch.cat((h0[:, 0, :, :], h0[:, 1, :, :]), dim=2)
            c0 = torch.cat((c0[:, 0, :, :], c0[:, 1, :, :]), dim=2)
            h0 = h0.view(self.num_layers, -1, self.hidden_size * 2)
            c0 = c0.view(self.num_layers, -1, self.hidden_size * 2)
            hidden = (h0, c0)
        out, hidden = self.lstm(x, hidden)
        return out, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, mlp, device="cpu"):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.mlp = mlp.to(device)

    def forward(self, X, print_shapes=False):
        target_length = X.shape[1]

        # Encode the input sequence
        encoder_output, (h1, c1) = self.encoder(X)
        if print_shapes:
            print("Encoder output shape:", encoder_output.shape)
            print("Encoder hidden state shape:", h1.shape)
            print("Encoder cell state shape:", c1.shape)

        # Use the last hidden state of the encoder as the initial input for the decoder
        decoder_input = encoder_output[:, -1].unsqueeze(1)
        decoder_outputs = []
        decoder_hidden = (h1, c1)
        for t in range(target_length):  # Use the target length from X.shape[1]
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_outputs.append(self.mlp(decoder_output))
            decoder_input = decoder_output  # Use the current output as the next input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        if print_shapes:
            print("Output shape after MLP:", decoder_outputs.shape)
        return decoder_outputs
