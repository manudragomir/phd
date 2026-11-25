import torch
import torch.nn as nn

from .transformer import Block
from .embeddings import PatchEmbedding, TokenPosition2DEmbedding, TokenPositionEmbedding


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        in_channels,
        emb_size,
        num_attention_heads,
        dropout,
        num_transformer_blocks,
        n_classes,
        pool="cls",
        pos_embed="1d",
    ):
        super().__init__()

        assert image_size % patch_size == 0
        assert emb_size % num_attention_heads == 0
        if pool not in {"cls", "mean", "patch0", "rand"}:
            raise ValueError("pool must be 'cls', 'mean', 'patch0', or 'rand'")
        if pos_embed not in {"1d", "2d"}:
            raise ValueError("pos_embed must be '1d' or '2d'")
        self.pool = pool
        self.pos_embed = pos_embed

        # 1. Patch embedding
        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            emb_size=emb_size
        )

        # 2. CLS token + positional embeddings
        grid_size = image_size // patch_size
        num_patches = grid_size ** 2
        if pos_embed == "1d":
            self.token_position_embedding = TokenPositionEmbedding(
                num_patches=num_patches,
                emb_size=emb_size
            )
        else:
            self.token_position_embedding = TokenPosition2DEmbedding(
                grid_size=grid_size,
                emb_size=emb_size
            )

        # 3. Transformer encoder blocks
        self.blocks = nn.Sequential(
            *[
                Block(
                    num_heads=num_attention_heads,
                    emb_size=emb_size,
                    dropout=dropout
                )
                for _ in range(num_transformer_blocks)
            ]
        )

        # 4. Final LayerNorm
        self.norm = nn.LayerNorm(emb_size)

        # 5. Classification head
        self.classifier = nn.Linear(emb_size, n_classes)

    def encode_tokens(self, x):
        # x: (B, C, H, W)

        x = self.patch_embedding(x)
        # (B, P, E)

        x = self.token_position_embedding(x)
        # (B, P + 1, E)

        x = self.blocks(x)
        # (B, P + 1, E)

        x = self.norm(x)
        # (B, P + 1, E). Index 0 is the CLS token.
        return x

    def encode(self, x):
        # CLS token, one vector per image
        return self.encode_tokens(x)[:, 0, :]
        # (B, E)

    def pool_tokens(self, tokens):
        # tokens: (B, P + 1, E). Index 0 is the CLS token.
        if self.pool == "cls":
            return tokens[:, 0, :]
        if self.pool == "mean":
            return tokens[:, 1:, :].mean(dim=1)
        if self.pool == "rand":
            # One patch token per image. Index 0 is the class token.
            batch, length, _ = tokens.shape
            index = torch.randint(1, length, (batch,), device=tokens.device)
            return tokens[torch.arange(batch, device=tokens.device), index]
        return tokens[:, 1, :]

    def forward(self, x):
        embedding = self.pool_tokens(self.encode_tokens(x))
        logits = self.classifier(embedding)
        # (B, n_classes)
        return logits