import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from hmr4d.configs import MainStore, builds

from hmr4d.network.base_arch.transformer.encoder_rope import EncoderRoPEBlock
from hmr4d.network.base_arch.transformer.layer import zero_module
from hmr4d.network.base_arch.embeddings.pe import PositionalEncoding

from hmr4d.utils.net_utils import length_to_mask
from timm.models.vision_transformer import Mlp


class TimestepEmbedder(nn.Module):
    """Diffusion timestep embedder (sinusoidal lookup -> MLP), batch-first.

    Mirrors GEM-X's TimestepEmbedder but returns (B, 1, latent) so it broadcasts
    over the sequence dim of a batch-first (B, L, latent) tensor (VITH is
    batch-first throughout; GEM-X's original .permute(1,0,2) targets seq-first).
    """

    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder
        self.time_embed = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, timesteps):
        # timesteps: (B,) integer steps. pe: (max_len, 1, latent) -> (B, 1, latent)
        pe = self.sequence_pos_encoder.pe[timesteps.long()]
        return self.time_embed(pe)  # (B, 1, latent)


class ConditionEmbedder(nn.Module):
    """Embed raw conditions (obs / cliffcam / cam_angvel / imgseq) -> (B, L, latent).

    GEM-X moves condition embedding OUT of the per-step denoiser so f_cond is built
    ONCE and reused across all DDIM steps, and so CFG can build a separate
    unconditional f_uncond. This holds the embedders that previously lived inside
    NetworkEncoderRoPE; the math is byte-for-byte identical to the old in-denoiser
    path. `drop_obs` (per-sample bool, (B,)) forces obs all-invisible -> used to
    build the CFG unconditional branch (∅ = drop the 2D-pose condition).
    """

    def __init__(self, latent_dim, cliffcam_dim=3, cam_angvel_dim=6, imgseq_dim=1024,
                 obs_num_joints=14, dropout=0.1):
        super().__init__()
        self.obs_num_joints = obs_num_joints
        self.cliffcam_dim = cliffcam_dim
        self.cam_angvel_dim = cam_angvel_dim
        self.imgseq_dim = imgseq_dim

        # obs (2D pose) main token: 2 -> 32 per joint, then MLP to latent
        self.learned_pos_linear = nn.Linear(2, 32)
        self.learned_pos_params = nn.Parameter(torch.randn(obs_num_joints, 32), requires_grad=True)
        self.embed_noisyobs = Mlp(
            obs_num_joints * 32, hidden_features=latent_dim * 2, out_features=latent_dim, drop=dropout
        )

        self.cliffcam_embedder = nn.Sequential(
            nn.Linear(cliffcam_dim, latent_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )
        if cam_angvel_dim > 0:
            self.cam_angvel_embedder = nn.Sequential(
                nn.Linear(cam_angvel_dim, latent_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                zero_module(nn.Linear(latent_dim, latent_dim)),
            )
        if imgseq_dim > 0:
            self.imgseq_embedder = nn.Sequential(
                nn.LayerNorm(imgseq_dim),
                zero_module(nn.Linear(imgseq_dim, latent_dim)),
            )

    def forward(self, obs=None, f_cliffcam=None, f_cam_angvel=None,
                f_imgseq=None, f_imgseq_mask=None, drop_obs=None):
        B, L, J, C = obs.shape
        assert J == self.obs_num_joints and C == 3

        obs = obs.clone()
        visible_mask = obs[..., [2]] > 0.5  # (B, L, J, 1)
        obs[~visible_mask[..., 0]] = 0  # set low-conf to all zeros
        if drop_obs is not None:
            # CFG unconditional: drop the 2D-pose condition for selected samples
            # -> all joints invisible -> obs token = learned "invisible" embedding.
            visible_mask = visible_mask & (~drop_obs.view(B, 1, 1, 1))
        f_obs = self.learned_pos_linear(obs[..., :2])  # (B, L, J, 32)
        f_obs = f_obs * visible_mask + self.learned_pos_params.repeat(B, L, 1, 1) * ~visible_mask
        x = self.embed_noisyobs(f_obs.view(B, L, -1))  # (B, L, latent)

        x = x + self.cliffcam_embedder(f_cliffcam)
        if hasattr(self, "cam_angvel_embedder"):
            x = x + self.cam_angvel_embedder(f_cam_angvel)
        if f_imgseq is not None and hasattr(self, "imgseq_embedder"):
            f_img = self.imgseq_embedder(f_imgseq)
            if f_imgseq_mask is not None:
                m = f_imgseq_mask.to(device=f_img.device, dtype=torch.bool)
                if m.ndim == 1:
                    m = m[:, None, None]
                elif m.ndim == 2:
                    m = m[:, :, None]
                f_img = f_img * m
            x = x + f_img
        return x  # (B, L, latent)


class NetworkEncoderRoPE(nn.Module):
    def __init__(
        self,
        # x
        output_dim=151,
        xt_dim=189,  # diffusion: dim of the noisy motion x_t fed in (== output_dim)
        max_len=120,
        # condition
        cliffcam_dim=3,
        cam_angvel_dim=6,
        imgseq_dim=1024,
        obs_num_joints=14,
        # intermediate
        latent_dim=512,
        num_layers=12,
        num_heads=8,
        mlp_ratio=4.0,
        # output
        pred_cam_dim=3,
        static_conf_dim=6,
        # training
        dropout=0.1,
        real_embed=False,
        # other
        avgbeta=True,
    ):
        super().__init__()

        # input
        self.output_dim = output_dim
        self.xt_dim = xt_dim
        self.max_len = max_len

        # condition
        self.cliffcam_dim = cliffcam_dim
        self.cam_angvel_dim = cam_angvel_dim
        self.imgseq_dim = imgseq_dim
        self.obs_num_joints = obs_num_joints

        # intermediate
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.real_embed = bool(real_embed)

        # ===== build model ===== #
        # NOTE: condition embedders (obs/cliffcam/cam_angvel/imgseq) now live in
        # ConditionEmbedder (owned by the Pipeline), GEM-X style — built once and
        # reused across DDIM steps, and reusable for the CFG unconditional branch.
        # The *_dim attrs above are kept so the Pipeline can size ConditionEmbedder.

        # Transformer
        self.blocks = nn.ModuleList(
            [
                EncoderRoPEBlock(self.latent_dim, self.num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(self.num_layers)
            ]
        )

        # Diffusion: timestep embedder + a linear fusing the noisy motion x_t into
        # the condition embedding (GEM-X style). These are inert in the regression
        # path (forward called with xt=None / timesteps=None).
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        if self.real_embed:
            # Keep a separate MLP from the diffusion-timestep MLP. Both use the
            # same fixed sinusoidal lookup, but represent different semantics.
            self.embed_real = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.add_cond_linear = nn.Linear(self.xt_dim + self.latent_dim, self.latent_dim)

        # Output heads
        self.final_layer = Mlp(self.latent_dim, out_features=self.output_dim)
        self.pred_cam_head = pred_cam_dim > 0  # keep extra_output for easy-loading old ckpt
        if self.pred_cam_head:
            self.pred_cam_head = Mlp(self.latent_dim, out_features=pred_cam_dim)
            self.register_buffer("pred_cam_mean", torch.tensor([1.0606, -0.0027, 0.2702]), False)
            self.register_buffer("pred_cam_std", torch.tensor([0.1784, 0.0956, 0.0764]), False)

        self.static_conf_head = static_conf_dim > 0
        if self.static_conf_head:
            self.static_conf_head = Mlp(self.latent_dim, out_features=static_conf_dim)

        self.avgbeta = avgbeta

    def set_pred_cam_stats(self, mean, std):
        if not hasattr(self, "pred_cam_mean") or not hasattr(self, "pred_cam_std"):
            raise RuntimeError("pred_cam_head is disabled, cannot set pred_cam statistics.")

        mean = torch.as_tensor(mean, dtype=self.pred_cam_mean.dtype, device=self.pred_cam_mean.device)
        std = torch.as_tensor(std, dtype=self.pred_cam_std.dtype, device=self.pred_cam_std.device).clamp(min=1e-6)
        if mean.shape != self.pred_cam_mean.shape or std.shape != self.pred_cam_std.shape:
            raise ValueError(
                f"pred_cam stats shape mismatch: mean {tuple(mean.shape)}, std {tuple(std.shape)}, "
                f"expected {tuple(self.pred_cam_mean.shape)}"
            )

        self.pred_cam_mean.copy_(mean)
        self.pred_cam_std.copy_(std)

    @staticmethod
    def _apply_real_prefix(xt, gt_motion, real):
        """Replace the first ``real`` noisy frames with exact GT motion.

        ``real`` may be one scalar shared by the batch or a ``(B,)`` tensor so
        each training sample can independently choose its observed prefix.
        ``real=0`` is an exact no-op and preserves the original diffusion path.
        """
        if real is None:
            return xt
        if xt is None or gt_motion is None:
            raise ValueError("real requires both xt and gt_motion")
        if xt.shape != gt_motion.shape:
            raise ValueError(
                f"xt/gt_motion shape mismatch: {tuple(xt.shape)} vs {tuple(gt_motion.shape)}"
            )

        B, L, _ = xt.shape
        real = torch.as_tensor(real, device=xt.device, dtype=torch.long).reshape(-1)
        if real.numel() == 1:
            real = real.expand(B)
        if real.numel() != B:
            raise ValueError(f"real must be scalar or shape ({B},), got {tuple(real.shape)}")
        if bool(((real < 0) | (real >= L)).any().item()):
            raise ValueError(f"real must be in [0, {L - 1}], got {real.tolist()}")

        prefix_mask = torch.arange(L, device=xt.device)[None, :] < real[:, None]
        return torch.where(prefix_mask[..., None], gt_motion.to(xt), xt)

    def forward(
        self,
        f_cond=None,
        length=None,
        xt=None,
        timesteps=None,
        real=None,
        gt_motion=None,
    ):
        """
        Args:
            f_cond: (B, L, latent) precomputed condition embedding (from ConditionEmbedder).
            length: (B), valid length of the sequence.
            xt: (B, L, xt_dim) noisy motion x_t (diffusion). None -> regression path.
            timesteps: (B,) diffusion timestep indices. None -> regression path.
            real: scalar or (B,) number of leading frames replaced by exact GT.
            gt_motion: (B, L, xt_dim) normalized robot GT used for that prefix.
        """
        x = f_cond
        B, L, _ = x.shape

        xt = self._apply_real_prefix(xt, gt_motion, real)

        # Diffusion fusion (GEM-X style). Regression path: xt/timesteps None -> skip,
        # so `x` stays the pure condition embedding (identical to the old behaviour).
        if timesteps is not None:
            x = x + self.embed_timestep(timesteps)  # (B,1,latent) broadcasts over L
        if real is not None:
            if not self.real_embed:
                raise ValueError("real was provided but network.real_embed is disabled")
            real_for_embed = torch.as_tensor(real, device=x.device, dtype=torch.long).reshape(-1)
            if real_for_embed.numel() == 1:
                real_for_embed = real_for_embed.expand(B)
            if real_for_embed.numel() != B:
                raise ValueError(
                    f"real must be scalar or shape ({B},), got {tuple(real_for_embed.shape)}"
                )
            x = x + self.embed_real(real_for_embed)  # (B,1,latent) broadcasts over L
        if xt is not None:
            x = self.add_cond_linear(torch.cat([x, xt], dim=-1))  # fuse noisy motion

        # Setup length and make padding mask
        assert B == length.size(0)
        pmask = ~length_to_mask(length, L)  # (B, L)

        if L > self.max_len:
            attnmask = torch.ones((L, L), device=x.device, dtype=torch.bool)
            for i in range(L):
                min_ind = max(0, i - self.max_len // 2)
                max_ind = min(L, i + self.max_len // 2)
                max_ind = max(self.max_len, max_ind)
                min_ind = min(L - self.max_len, min_ind)
                attnmask[i, min_ind:max_ind] = False
        else:
            attnmask = None

        # Transformer
        for block in self.blocks:
            x = block(x, attn_mask=attnmask, tgt_key_padding_mask=pmask)

        # Output
        sample = self.final_layer(x)  # (B, L, C)
        if self.avgbeta:
            betas = (sample[..., 126:136] * (~pmask[..., None])).sum(1) / length[:, None].clamp(min=1)  # (B, C)
            betas = repeat(betas, "b c -> b l c", l=L)
            sample = torch.cat([sample[..., :126], betas, sample[..., 136:]], dim=-1)

        # Output (extra)
        pred_cam = None
        if self.pred_cam_head:
            pred_cam = self.pred_cam_head(x)
            pred_cam = pred_cam * self.pred_cam_std + self.pred_cam_mean
            torch.clamp_min_(pred_cam[..., 0], 0.25)  # min_clamp s to 0.25 (prevent negative prediction)

        static_conf_logits = None
        if self.static_conf_head:
            static_conf_logits = self.static_conf_head(x)  # (B, L, C')

        output = {
            "pred_context": x,
            "pred_x": sample,
            "pred_x_start": sample,  # diffusion: model predicts x_0 (== pred_x)
            "pred_cam": pred_cam,
            "static_conf_logits": static_conf_logits,
        }
        return output


# Add to MainStore
group_name = "network/offline"
MainStore.store(
    name="relative_transformer",
    node=builds(NetworkEncoderRoPE, populate_full_signature=True),
    group=group_name,
)
