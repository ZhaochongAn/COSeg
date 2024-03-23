import torch
import torch.nn as nn

from einops import rearrange, repeat

from torch_points3d.core.common_modules import FastBatchNorm1d
from torch_points3d.modules.KPConv.kernels import KPConvLayer


class KPConvResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        prev_grid_size,
        sigma=1.0,
        negative_slope=0.2,
        bn_momentum=0.02,
    ):
        super().__init__()
        d_2 = out_channels // 4
        activation = nn.LeakyReLU(negative_slope=negative_slope)
        self.unary_1 = torch.nn.Sequential(
            nn.Linear(in_channels, d_2, bias=False),
            FastBatchNorm1d(d_2, momentum=bn_momentum),
            activation,
        )
        self.unary_2 = torch.nn.Sequential(
            nn.Linear(d_2, out_channels, bias=False),
            FastBatchNorm1d(out_channels, momentum=bn_momentum),
            activation,
        )
        self.kpconv = KPConvLayer(
            d_2, d_2, point_influence=prev_grid_size * sigma, add_one=False
        )
        self.bn = FastBatchNorm1d(out_channels, momentum=bn_momentum)
        self.activation = activation

        if in_channels != out_channels:
            self.shortcut_op = torch.nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=False),
                FastBatchNorm1d(out_channels, momentum=bn_momentum),
            )
        else:
            self.shortcut_op = nn.Identity()

    def forward(self, feats, xyz, batch, neighbor_idx):
        # feats: [N, C]
        # xyz: [N, 3]
        # batch: [N,]
        # neighbor_idx: [N, M]
        shortcut = feats.clone()
        feats = self.unary_1(feats)
        feats = self.kpconv(xyz, xyz, neighbor_idx, feats)
        feats = self.unary_2(feats)
        shortcut = self.shortcut_op(shortcut)
        feats = feats + shortcut
        return feats


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class FullAttention(nn.Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(
                ~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]),
                float("-inf"),
            )

        # Compute the attention and the weighted average
        softmax_temp = 1.0 / queries.size(3) ** 0.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values):
        """Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = (
            torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length
        )

        return queried_values.contiguous()


class AttentionLayer(nn.Module):
    def __init__(
        self, hidden_dim, guidance_dim, nheads=8, attention_type="linear"
    ):
        super().__init__()
        self.nheads = nheads
        self.q = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)

        if attention_type == "linear":
            self.attention = LinearAttention()
        elif attention_type == "full":
            self.attention = FullAttention()
        else:
            raise NotImplementedError

    def forward(self, x, guidance):
        """
        Arguments:
            x: B, L, C
            guidance: B, L, C
        """
        q = (
            self.q(torch.cat([x, guidance], dim=-1))
            if guidance is not None
            else self.q(x)
        )
        k = (
            self.k(torch.cat([x, guidance], dim=-1))
            if guidance is not None
            else self.k(x)
        )
        v = self.v(x)

        q = rearrange(q, "B L (H D) -> B L H D", H=self.nheads)
        k = rearrange(k, "B S (H D) -> B S H D", H=self.nheads)
        v = rearrange(v, "B S (H D) -> B S H D", H=self.nheads)

        out = self.attention(q, k, v)
        out = rearrange(out, "B L H D -> B L (H D)")
        return out


class ClassTransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        guidance_dim=64,
        nheads=8,
        attention_type="linear",
    ) -> None:
        super().__init__()
        self.attention = AttentionLayer(
            hidden_dim,
            guidance_dim,
            nheads=nheads,
            attention_type=attention_type,
        )
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.norm_xbg = nn.LayerNorm([2, hidden_dim])
        self.base_merge = nn.Conv1d(2, 1, kernel_size=1, bias=False)
        self.base_merge.weight = nn.Parameter(
            torch.tensor([[1.0], [0.0]]).reshape_as(self.base_merge.weight)
        )

    def forward(self, x, base_pred, guidance):
        """
        Arguments:
            x: B, C, T, N
            guidance: B, T, C
            base_pred: N, 1
        """
        B, _, _, N = x.size()
        x_pool = rearrange(x, "B C T N -> (B N) T C")

        if base_pred is not None:
            x_bg = x_pool[:, :1].clone()
            x_pool[:, :1] = self.base_merge(
                self.norm_xbg(
                    torch.cat(
                        [
                            x_bg,
                            base_pred.unsqueeze(-1).repeat(
                                1, 1, x_bg.shape[-1]
                            ),
                        ],
                        dim=1,
                    )
                )
            )  # N, 2, C

        if guidance is not None:
            guidance = repeat(guidance, "B T C -> (B N) T C", N=N)

        x_pool = x_pool + self.attention(self.norm1(x_pool), guidance)
        x_pool = x_pool + self.MLP(self.norm2(x_pool))

        x_pool = rearrange(x_pool, "(B N) T C -> B C T N", N=N)

        x = x + x_pool  # Residual
        return x


class SpatialTransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        guidance_dim=64,
        nheads=8,
        attention_type="linear",
    ) -> None:
        super().__init__()
        self.attention = AttentionLayer(
            hidden_dim,
            guidance_dim,
            nheads=nheads,
            attention_type=attention_type,
        )
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, guidance):
        """
        Arguments:
            x: B, C, T, N
            guidance: B, N, C
        """
        B, _, T, N = x.size()

        x_pool = rearrange(x, "B C T N -> (B T) N C")
        if guidance is not None:
            guidance = repeat(guidance, "B N C -> (B T) N C", T=T)

        x_pool = x_pool + self.attention(
            self.norm1(x_pool), guidance
        )  # Attention
        x_pool = x_pool + self.MLP(self.norm2(x_pool))  # MLP

        x_pool = rearrange(x_pool, "(B T) N C -> B C T N", T=T)

        x = x + x_pool  # Residual
        return x


class AggregatorLayer(nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        text_guidance_dim=512,
        appearance_guidance=512,
        nheads=4,
        attention_type="linear",
    ) -> None:
        super().__init__()
        self.spatial_attention = SpatialTransformerLayer(
            hidden_dim,
            appearance_guidance,
            nheads=nheads,
            attention_type=attention_type,
        )

        self.class_attention = ClassTransformerLayer(
            hidden_dim,
            text_guidance_dim,
            nheads=nheads,
            attention_type=attention_type,
        )

    def forward(
        self, x, basept_guidance, appearance_guidance=None, text_guidance=None
    ):
        """
        Arguments:
            x: B C T N
        """
        x = self.spatial_attention(x, appearance_guidance)
        x = self.class_attention(x, basept_guidance, text_guidance)
        return x


class MLPWithoutResidual(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.fc2 = nn.Linear(4 * hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
