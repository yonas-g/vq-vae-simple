import torch
import torch.nn as nn

from torchsummary import summary

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

batch_size = 8
time_dim = 300
num_channels = 80

data = torch.rand(batch_size, time_dim, num_channels).to(device) # [B,T,D]
print(data.shape)


class VQVAE(nn.Module):
    def __init__(self, enc_hidden_dim=80, codebook_size=10, vq_hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=codebook_size, embedding_dim=vq_hidden_dim)
        self.embedding.weight.data.uniform_(-1/codebook_size, 1/vq_hidden_dim)
        
        self.enc_conv1 = nn.Conv1d(enc_hidden_dim, 64, kernel_size=5)
        self.enc_conv2 = nn.Conv1d(64, vq_hidden_dim, kernel_size=5)

        self.dec_conv1 = nn.ConvTranspose1d(vq_hidden_dim, 64, kernel_size=5)
        self.dec_conv2 = nn.ConvTranspose1d(64, enc_hidden_dim, kernel_size=5)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)
        out = self.enc_conv1(x)
        z_e = self.enc_conv2(out) # [B, vq_hidden_dim, T]

        # L2 distance
        dist = torch.cdist(z_e.permute(0, 2, 1), self.embedding.weight) # [B,T,K] - > per each time, the distance of each T from the K codebooks
        indices = torch.argmin(dist, dim=-1).to(device) # [B,T]
        # z_q = self.embedding.weight[indices] # [B, T, vq_hidden_dim]
        z_q = self.embedding(indices) 
        
        out = self.dec_conv1(z_q.permute(0, 2, 1))
        out = self.dec_conv2(out)

        out = out.permute(0, 2, 1)
        return out


if __name__ == "__main__":
    model = VQVAE(80, 3, 128).to(device)
    # summary(model, input_size=(40, 80)) #[T, C_in]
    out = model(data)