import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # positional encoding
        self.pos_encoding = nn.Embedding(max_seq_len, embed_dim)
        # transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # output projection
        self.output_proj = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, x):
        # get token embeddings
        tok_emb = self.embedding(x)
        # create position indices and get position embeddings
        positions = torch.arange(x.size(1), device=x.device)
        pos_emb = self.pos_encoding(positions)
        # add them together
        x = tok_emb + pos_emb
        # pass through transformer blocks
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        # pass through output projection
        x = self.transformer(x, mask=mask, is_causal=True)
        logits = self.output_proj(x)
        return logits
