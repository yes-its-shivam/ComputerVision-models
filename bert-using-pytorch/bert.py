import torch.nn as nn

from transformer import TransformerBlock
from .embedding import BERTEmbedding


class BERT(nn.Module):

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        vocab_size: vocab_size of total words
        hidden: BERT model hidden size
        n_layers: numbers of Transformer blocks/layers
        attn_heads: number of attention heads(multihead/singlehead)
        dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # in paper they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi_layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, segment_info):
        # attention masking for padded token
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        x = self.embedding(x, segment_info)
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x