import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, 
                image_size,
                patch_size, 
                in_channels,
                emb_size):
        super().__init__()

        assert image_size % patch_size == 0

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.emb_size = emb_size

        self.conv = nn.Conv2d(in_channels=in_channels, 
                              out_channels=emb_size, 
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        
        x_out = self.conv(x)
        # print(x_out.shape) # B x E x (H / K) x (W / K) = B x E x Pd x Pd
        
        x_out_flatten = x_out.flatten(start_dim=2)
        #print(x_out_flatten.shape) # B x E x P

        x_permuted = x_out_flatten.permute(0, 2, 1)
        
        return x_permuted # B x P x E
        

class TokenPositionEmbedding(nn.Module):
    def __init__(self, num_patches, emb_size):
        super().__init__()

        self.num_patches = num_patches
        self.emb_size = emb_size

        # LEARNABLE CLS token
        # 1 shared parameter, 1 token, emb_size features
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, emb_size)
        )
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        # EMBEDDINGS for positions
        # num_patches + 1 includes the CLS token
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, emb_size)
        )
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

    def forward(self, x):
        # x: B x P x E
        B, P, E = x.shape

        assert P == self.num_patches
        assert E == self.emb_size

        # Step 1: use the same CLS token for each image
        cls_each_image = self.cls_token.expand(B, -1, -1)

        # Step 2: concatenate CLS at the beginning
        x = torch.cat([cls_each_image, x], dim=1)

        # Step 3: add positional embeddings
        # Broadcasting across the batch dimension :)
        x = x + self.pos_embedding

        # B x (P + 1) x E
        return x


class TokenPosition2DEmbedding(nn.Module):
    def __init__(self, grid_size, emb_size):
        super().__init__()

        self.grid_size = grid_size
        self.emb_size = emb_size
        self.num_patches = grid_size * grid_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        # One vector per row i, one vector per column j.
        # Patch (i, j) gets row_embed[i] + col_embed[j].
        self.row_embed = nn.Parameter(torch.zeros(grid_size, emb_size))
        self.col_embed = nn.Parameter(torch.zeros(grid_size, emb_size))
        nn.init.normal_(self.row_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.col_embed, mean=0.0, std=0.02)

    def forward(self, x):
        # x: B x P x E, row-major, matching PatchEmbedding.flatten
        B, P, E = x.shape
        g = self.grid_size

        assert P == self.num_patches
        assert E == self.emb_size

        # (g, 1, E) + (1, g, E) -> (g, g, E)
        # Flatten with i slowest, so token k = i * g + j gets pos[i, j].
        pos = self.row_embed[:, None, :] + self.col_embed[None, :, :]
        pos = pos.reshape(1, P, E)
        x = x + pos

        cls_each_image = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_each_image, x], dim=1)

        # B x (P + 1) x E
        return x

