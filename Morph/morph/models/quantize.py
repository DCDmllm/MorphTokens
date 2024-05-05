import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class VectorQuantizer(nn.Module):
    def __init__(self, hidden_size=32, beta=0.25, n_e=8192):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_e = n_e
        embeddings = nn.Embedding(self.n_e, self.hidden_size)
        embeddings.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        self.embedding_weight = nn.Parameter(embeddings.weight)
        self.embedding_weight.requires_grad = False
        
        self.beta = beta
        self.n_e = self.embedding_weight.size(0)
        self.e_dim = self.embedding_weight.size(-1)
        
    
    def forward(self, z):
        bz = z.shape[0]
        z_flattened = z.view(-1, self.e_dim)
        
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding_weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding_weight, 'n d -> d n'))
        
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)
        
        z_q = torch.matmul(min_encodings, self.embedding_weight).view(z.shape)


        # reshape min_encoding_indicies
        min_encoding_indices = min_encoding_indices.reshape(bz, -1).detach() + 32000

        z_q = z_q.detach()
        z_q = z_q.reshape(bz, -1, z_q.shape[-1])
    
        return z_q, min_encoding_indices

    
    def get_codebook_entry(self, indices, shape=None):
        bz = indices.shape[0]
        indicie_flattend = indices.view(-1)
        
        min_encodings = torch.zeros(indicie_flattend.shape[0], self.n_e).to(indicie_flattend)
        min_encodings.scatter_(1, indicie_flattend[:,None], 1)
        
        z_q = torch.matmul(min_encodings.float(), self.embedding_weight)
        
        z_q = z_q.reshape(bz, -1, z_q.shape[-1])

        return z_q