# vq-vae

Simplest experimentation implementation of VQ-VAE from https://arxiv.org/pdf/1711.00937

## LOSS CALCULATION
During loss calculation, we do reconstruction loss, commitment loss and codebook loss. 
1. Reconstruction loss — regular decoder output vs original input
2. Codebook loss — update embedding/codebook only. Gradient flows into z_q → updates the codebook
3. Commitment loss — update encoder only. Gradient flows into z_e → updates the encoder

```python
# z_q: [B, T, D], z_e: [B, D, T] → permute z_e
# 
loss_recon = F.mse_loss(out, x_orig)
loss_codebook = F.mse_loss(z_q, z_e.detach())
loss_commit = F.mse_loss(z_e, z_q.detach())
# Total loss
loss = loss_recon + loss_codebook + beta * loss_commit
```
