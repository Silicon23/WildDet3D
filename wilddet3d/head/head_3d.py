"""3D detection head."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import batched_nms, nms
from vis4d.op.layer.attention import MultiheadAttention
from vis4d.op.layer.transformer import FFN, get_clones
from vis4d.op.layer.weight_init import xavier_init

from wilddet3d.ops.box2d import bbox_cxcywh_to_xyxy
from wilddet3d.ops.ray import generate_rays, rsh_cart_8
from wilddet3d.ops.mlp import MLP
from wilddet3d.ops.util import flat_interpolate

from .coder_3d import Det3DCoder


def convert_grounding_to_cls_scores(
    logits: Tensor, positive_maps: dict[int, list[int, int]]
) -> Tensor:
    """Convert logits to class scores."""
    assert len(positive_maps) == logits.shape[0]  # batch size

    scores = torch.zeros(
        logits.shape[0], logits.shape[1], len(positive_maps[0])
    ).to(logits.device)
    if positive_maps is not None:
        if all(x == positive_maps[0] for x in positive_maps):
            # only need to compute once
            positive_map = positive_maps[0]
            for label_j in positive_map:
                scores[:, :, label_j - 1] = logits[
                    :, :, torch.LongTensor(positive_map[label_j])
                ].mean(-1)
        else:
            for i, positive_map in enumerate(positive_maps):
                for label_j in positive_map:
                    scores[i, :, label_j - 1] = logits[
                        i, :, torch.LongTensor(positive_map[label_j])
                    ].mean(-1)
    return scores


class Det3DHead(nn.Module):
    """3D detection head.

    Args:
        embed_dims: Embedding dimension for the head.
        num_decoder_layer: Number of decoder layers.
        num_reg_fcs: Number of fully connected layers in regression branch.
        as_two_stage: Whether to use two-stage detection.
        box_coder: 3D box coder for encoding/decoding.
        depth_output_scales: Scale factor for depth embedding dims.
        use_camera_prompt: Whether to use camera/ray prompt branch.
            Set to False when using ray-aware depth backends (UniDepthV2, DetAny3D)
            since their depth_latents already incorporate ray information.
            Set to True for non-ray-aware backends (UniDepthHead v1).
        use_depth_prompt: Whether to use depth prompt branch.
            Set to False for ablation: only use depth via encoder fusion.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_decoder_layer: int = 6,
        num_reg_fcs: int = 2,
        as_two_stage: bool = True,
        box_coder: Det3DCoder | None = None,
        depth_output_scales: int = 1,
        depth_latent_dim: int | None = None,
        use_camera_prompt: bool = True,
        use_depth_prompt: bool = True,
        use_temporal_prompt: bool = False,
        traj_token_dim: int = 256,
        use_layer_bias: bool = False,
        use_temporal_kv_norm: bool = False,
    ) -> None:
        """Initialize the 3D detection head.

        Args:
            depth_latent_dim: Dimension of depth latents from geometry backend.
                If provided, uses this directly. If None, computes from
                depth_output_scales as embed_dims // 2**depth_output_scales.
            use_temporal_prompt: Track C. Adds a per-layer temporal prompt
                (cloned Prompt3DQueryLayer) gated by a zero-init LayerScale, and
                enables the output-space residual on ``temporal_box_12d`` in
                ``single_forward``. Backward compatible: when False (default) or
                when temporal inputs are None, the head behaves exactly as the
                pretrained WildDet3D head.
            traj_token_dim: Dim of the trajectory-encoder tokens fed to the
                temporal prompt (256 for Flavor 1).
        """
        super().__init__()
        self.embed_dims = embed_dims
        self.use_camera_prompt = use_camera_prompt
        self.use_depth_prompt = use_depth_prompt
        self.use_temporal_prompt = use_temporal_prompt
        self.use_layer_bias = use_layer_bias
        self.use_temporal_kv_norm = use_temporal_kv_norm

        self.num_pred_layer = (
            num_decoder_layer + 1 if as_two_stage else num_decoder_layer
        )
        self.as_two_stage = as_two_stage

        self.box_coder = box_coder or Det3DCoder()

        reg_branch = self._get_reg_branch(num_reg_fcs, self.box_coder.reg_dims)
        self.reg_branches = get_clones(reg_branch, self.num_pred_layer)

        # 3D confidence branch (predicts 3D-aware objectness score)
        conf_branch = self._get_conf_branch(num_reg_fcs)
        self.conf_branches = get_clones(conf_branch, self.num_pred_layer)

        # Camera prompt branch (only created if use_camera_prompt is True)
        if self.use_camera_prompt:
            project_rays, prompt_camera = self._get_condition_branch(
                input_dims=81, expansion=4, embed_dims=embed_dims
            )
            self.project_rays = get_clones(project_rays, self.num_pred_layer)
            self.prompt_camera = get_clones(prompt_camera, self.num_pred_layer)
        else:
            self.project_rays = None
            self.prompt_camera = None

        # Depth prompt branch (only created if use_depth_prompt is True)
        if self.use_depth_prompt:
            # Use depth_latent_dim directly if provided, else compute from depth_output_scales
            if depth_latent_dim is not None:
                depth_embed_dims = depth_latent_dim
            else:
                depth_embed_dims = embed_dims // 2**depth_output_scales
            project_depth, prompt_depth = self._get_condition_branch(
                depth_embed_dims, expansion=4, embed_dims=embed_dims
            )
            self.project_depth = get_clones(project_depth, self.num_pred_layer)
            self.prompt_depth = get_clones(prompt_depth, self.num_pred_layer)
        else:
            self.project_depth = None
            self.prompt_depth = None

        # Temporal prompt branch (Track C). Cloned per prediction layer, gated
        # by a per-layer zero-init LayerScale so it is identity at init.
        if self.use_temporal_prompt:
            project_temporal, prompt_temporal = self._get_condition_branch(
                input_dims=traj_token_dim, expansion=4, embed_dims=embed_dims
            )
            self.project_temporal = get_clones(
                project_temporal, self.num_pred_layer
            )
            self.prompt_temporal = get_clones(
                prompt_temporal, self.num_pred_layer
            )
            self.temporal_gate = nn.ParameterList(
                [
                    nn.Parameter(torch.zeros(embed_dims))
                    for _ in range(self.num_pred_layer)
                ]
            )
            # Optional LayerNorm on the projected temporal KV before the cross-
            # attn. The project_* MLP normalizes its INPUT but not its OUTPUT;
            # the from-scratch temporal projection can drift to a ~100x-too-large
            # KV scale vs the pretrained depth/ray KVs. This LN guarantees a
            # well-conditioned, unit-scale KV regardless of the learned weights.
            if self.use_temporal_kv_norm:
                self.temporal_kv_norm = nn.ModuleList(
                    [nn.LayerNorm(embed_dims) for _ in range(self.num_pred_layer)]
                )
            else:
                self.temporal_kv_norm = None
        else:
            self.project_temporal = None
            self.prompt_temporal = None
            self.temporal_gate = None
            self.temporal_kv_norm = None

        # Per-layer learned additive bias (Track C v11_with_bias). Tests the
        # "constant feature" hypothesis: replace the prompt_temporal collapse
        # with the minimal-cost equivalent — a per-layer additive bias on the
        # hidden state. Zero-init keeps identity at init.
        if self.use_layer_bias:
            self.layer_bias = nn.ParameterList(
                [nn.Parameter(torch.zeros(embed_dims))
                 for _ in range(self.num_pred_layer)]
            )
        else:
            self.layer_bias = None

        self._init_weights()

    def _get_reg_branch(
        self, num_reg_fcs: int, reg_dims: int
    ) -> nn.Sequential:
        """Get the regression branch."""
        reg_branch = []
        for _ in range(num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, reg_dims))
        return nn.Sequential(*reg_branch)

    def _get_conf_branch(self, num_reg_fcs: int) -> nn.Sequential:
        """Get the 3D confidence branch (output dim = 1)."""
        conf_branch = []
        for _ in range(num_reg_fcs):
            conf_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            conf_branch.append(nn.ReLU())
        conf_branch.append(nn.Linear(self.embed_dims, 1))
        return nn.Sequential(*conf_branch)

    def _get_condition_branch(
        self, input_dims: int, expansion: int, embed_dims: int
    ) -> tuple[nn.Module, nn.Module]:
        """Get the condition branch."""
        project_layer = MLP(
            input_dims, expansion=expansion, output_dim=embed_dims
        )

        prompt_layer = Prompt3DQueryLayer(embed_dims)

        return project_layer, prompt_layer

    def _init_weights(self) -> None:
        """Initialize weights of the Deformable DETR head."""
        for m in self.reg_branches:
            xavier_init(m, distribution="uniform")
        for m in self.conf_branches:
            xavier_init(m, distribution="uniform")

    def zero_init_reg_residual(self) -> None:
        """Zero-init the final Linear of every reg_branch (Track C).

        Makes the per-layer regression an output-space *residual* on top of the
        temporal-encoder box: with ``reg_output = reg_branches[i](h) +
        temporal_box_12d`` and a zero final layer, the head outputs exactly the
        temporal prior at init and learns image-grounded corrections from there.

        Call this AFTER loading pretrained 3D-head weights (otherwise
        ``load_state_dict`` overwrites the zeros). The temporal prompt itself is
        already identity-at-init via its zero ``temporal_gate``.
        """
        for m in self.reg_branches:
            nn.init.zeros_(m[-1].weight)
            nn.init.zeros_(m[-1].bias)

    def warm_start_temporal_from_depth(self) -> None:
        """Initialize the temporal prompt branch from the pretrained depth branch.

        ``project_temporal``/``prompt_temporal`` are structurally identical to
        ``project_depth``/``prompt_depth`` (both ``MLP(256->1024->256)`` +
        ``Prompt3DQueryLayer(256)``). Copying the pretrained depth weights gives
        the otherwise-from-scratch temporal branch a well-conditioned starting
        point (sub-unit projected KV) instead of letting it drift into the
        ~100x-too-large KV regime. The zero-init ``temporal_gate`` still makes
        the branch identity-at-init; this only sets where training starts from
        once the gate opens. Call AFTER loading pretrained head weights.
        """
        assert self.use_temporal_prompt and self.use_depth_prompt, (
            "warm-start needs both temporal and depth prompt branches"
        )
        for i in range(self.num_pred_layer):
            self.project_temporal[i].load_state_dict(
                self.project_depth[i].state_dict()
            )
            self.prompt_temporal[i].load_state_dict(
                self.prompt_depth[i].state_dict()
            )

    def get_camera_embeddings(
        self,
        intrinsics: Tensor,
        image_shape: tuple[int, int],
        downsample: int = 16,
    ) -> Tensor:
        """Get the camera embeddings.

        Args:
            intrinsics: Camera intrinsics [B, 3, 3]. Should match the space
                where depth_latents were computed (may be adjusted for DINOv2).
            image_shape: Image (H, W) in the same space as intrinsics.
            downsample: Downsample factor for ray grid (8 or 16).
                Must match depth_latents resolution.

        Returns:
            ray_embeddings: [B, H//downsample * W//downsample, 81]
        """
        rays, _ = generate_rays(intrinsics, image_shape)

        rays = F.normalize(
            flat_interpolate(
                rays,
                old=image_shape,
                new=(image_shape[0] // downsample, image_shape[1] // downsample),
            ),
            dim=-1,
        )

        return rsh_cart_8(rays)

    def single_forward(
        self,
        layer_id: int,
        hidden_state: Tensor,
        ray_embeddings: Tensor | None,
        depth_latents: Tensor | None = None,
        temporal_tokens: Tensor | None = None,
        temporal_box_12d: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Single layer forward pass of the 3D detection head.

        Args:
            layer_id: Index of the decoder layer.
            hidden_state: Query hidden states [B, num_queries, embed_dims].
            ray_embeddings: Ray embeddings [B, H*W, 81]. Only used if use_camera_prompt=True.
            depth_latents: Depth latent features [B, H*W, depth_embed_dims].
            temporal_tokens: Track C trajectory-encoder tokens
                [B, n_tok, traj_token_dim] (n_tok=1 for Flavor 1). Cross-attended
                by the per-layer temporal prompt, gated by a zero-init LayerScale.
            temporal_box_12d: Track C output-space residual base, the temporal
                box re-encoded to the 12-d coder space against this frame's
                predicted 2D box. Broadcastable to [B, num_queries, reg_dims].

        Returns:
            Tuple of (reg_output, conf_output):
            - reg_output: 3D box regression [B, num_queries, reg_dims]
            - conf_output: 3D confidence logits [B, num_queries, 1]
        """
        # Camera-aware 3D queries (only if use_camera_prompt is True)
        if self.use_camera_prompt and ray_embeddings is not None:
            ray_embedding = self.project_rays[layer_id](ray_embeddings)
            hidden_state = self.prompt_camera[layer_id](
                hidden_state, ray_embedding, ray_embedding
            )

        # Depth-aware 3D queries (only if use_depth_prompt is True)
        if self.use_depth_prompt and depth_latents is not None:
            proj_depth_latents = self.project_depth[layer_id](depth_latents)
            hidden_state = self.prompt_depth[layer_id](
                hidden_state, proj_depth_latents, proj_depth_latents
            )

        # Temporal-aware 3D queries (Track C). Zero-init gate -> identity at init.
        if self.use_temporal_prompt and temporal_tokens is not None:
            proj_temporal = self.project_temporal[layer_id](temporal_tokens)
            if self.temporal_kv_norm is not None:
                proj_temporal = self.temporal_kv_norm[layer_id](proj_temporal)
            updated = self.prompt_temporal[layer_id](
                hidden_state, proj_temporal, proj_temporal
            )
            hidden_state = hidden_state + self.temporal_gate[layer_id] * (
                updated - hidden_state
            )

        # Per-layer learned bias (Track C v11_with_bias). Replaces the
        # collapsed-to-constant temporal cross-attn at ~256 params/layer.
        if self.use_layer_bias:
            hidden_state = hidden_state + self.layer_bias[layer_id]

        reg_output = self.reg_branches[layer_id](hidden_state)
        # Output-space residual on the temporal prior (Track C). With the final
        # reg Linear zero-init (see zero_init_reg_residual), reg_output is the
        # temporal prior at init; otherwise it is a correction on top of it.
        if temporal_box_12d is not None:
            reg_output = reg_output + temporal_box_12d
        conf_output = self.conf_branches[layer_id](hidden_state)

        return reg_output, conf_output

    def forward(
        self,
        hidden_states: Tensor,
        ray_embeddings: Tensor | None,
        depth_latents: Tensor | None = None,
        temporal_tokens: Tensor | None = None,
        temporal_box_12d: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass of the 3D detection head.

        Args:
            hidden_states: Query hidden states [num_layers, B, num_queries, embed_dims].
            ray_embeddings: Ray embeddings [B, H*W, 81]. Can be None if use_camera_prompt=False.
            depth_latents: Depth latent features [B, H*W, depth_embed_dims].
            temporal_tokens: Track C temporal tokens [B, n_tok, traj_token_dim],
                shared across decoder layers. None for the base head.
            temporal_box_12d: Track C residual base [B, num_queries, reg_dims]
                (or broadcastable), shared across decoder layers. None for base.

        Returns:
            Tuple of (stacked_reg, stacked_conf):
            - stacked_reg: [num_layers, B, num_queries, reg_dims]
            - stacked_conf: [num_layers, B, num_queries, 1]
        """
        all_layers_outputs_3d = []
        all_layers_conf_3d = []

        for layer_id in range(hidden_states.shape[0]):
            hidden_state = hidden_states[layer_id]

            reg_output, conf_output = self.single_forward(
                layer_id,
                hidden_state,
                ray_embeddings,
                depth_latents,
                temporal_tokens=temporal_tokens,
                temporal_box_12d=temporal_box_12d,
            )

            all_layers_outputs_3d.append(reg_output)
            all_layers_conf_3d.append(conf_output)

        return torch.stack(all_layers_outputs_3d), torch.stack(all_layers_conf_3d)


class Prompt3DQueryLayer(nn.Module):
    """Prompt 3D object query Layer."""

    def __init__(self, embed_dims: int = 256) -> None:
        """Init."""
        super().__init__()
        self.self_attn = MultiheadAttention(
            embed_dims=256, num_heads=8, batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dims)

        self.cross_attn = MultiheadAttention(
            embed_dims=256, num_heads=1, batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dims)

        self.ffn = FFN(embed_dims)

        self.norm3 = nn.LayerNorm(embed_dims)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_pos: Tensor | None = None,
    ) -> Tensor:
        """Forward."""
        # self attention
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
        )
        query = self.norm1(query)

        # cross attention
        query = self.cross_attn(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
        )
        query = self.norm2(query)

        # FFN
        query = self.ffn(query)
        query = self.norm3(query)

        return query


class RoI2Det3D:
    """Convert RoI to 3D Detection."""

    def __init__(
        self,
        nms: bool = False,
        max_per_img: int = 300,
        class_agnostic_nms: bool = False,
        score_threshold: float = 0.0,
        iou_threshold: float = 0.5,
        box_coder: Det3DCoder | None = None,
    ) -> None:
        """Create an instance of RoI2Det3D."""
        self.nms = nms
        self.max_per_img = max_per_img
        self.class_agnostic_nms = class_agnostic_nms
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

        self.box_coder = box_coder or Det3DCoder()

    def __call__(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        token_positive_maps: dict[int, list[int]] | None,
        img_shape: tuple[int, int],
        ori_shape: tuple[int, int],
        bbox_3d_pred: Tensor,
        intrinsics: Tensor,
        padding: list[int] | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Transform the bbox head output into bbox results."""
        assert len(cls_score) == len(bbox_pred)  # num_queries

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])

        if token_positive_maps is not None:
            cls_score = convert_grounding_to_cls_scores(
                logits=cls_score.sigmoid()[None],
                positive_maps=[token_positive_maps],
            )[0]

            k = min(self.max_per_img, cls_score.view(-1).shape[0])
            if k == 0:
                device = cls_score.device
                return (
                    torch.zeros(0, 4, device=device),
                    torch.zeros(0, device=device),
                    torch.zeros(0, dtype=torch.long, device=device),
                    torch.zeros(0, 10, device=device),
                )
            scores, indexes = cls_score.view(-1).topk(k)
            num_classes = cls_score.shape[-1]
            det_labels = indexes % num_classes
            bbox_index = indexes // num_classes
            det_bboxes = det_bboxes[bbox_index]
            bbox_3d_pred = bbox_3d_pred[bbox_index]

            # Remove low scoring boxes
            if self.score_threshold > 0.0:
                mask = scores > self.score_threshold
                det_bboxes = det_bboxes[mask]
                det_labels = det_labels[mask]
                scores = scores[mask]
                bbox_3d_pred = bbox_3d_pred[mask]

            if self.nms:
                if self.class_agnostic_nms:
                    keep = nms(det_bboxes, scores, self.iou_threshold)
                else:
                    keep = batched_nms(
                        det_bboxes, scores, det_labels, self.iou_threshold
                    )

                det_bboxes = det_bboxes[keep]
                det_labels = det_labels[keep]
                scores = scores[keep]
                bbox_3d_pred = bbox_3d_pred[keep]
        else:
            cls_score = cls_score.sigmoid()
            scores, _ = cls_score.max(-1)
            scores, indexes = scores.topk(self.max_per_img)
            det_bboxes = det_bboxes[indexes]
            bbox_3d_pred = bbox_3d_pred[indexes]
            det_labels = scores.new_zeros(scores.shape, dtype=torch.long)

        if bbox_3d_pred.numel() == 0:
            return (
                det_bboxes,
                scores,
                det_labels,
                bbox_3d_pred.new_empty((0, 10)),
            )

        det_bboxes3d = self.box_coder.decode(
            det_bboxes, bbox_3d_pred, intrinsics
        )

        # Remove padding when input_hw is affected by padding
        if padding is not None:
            det_bboxes[:, 0] -= padding[0]
            det_bboxes[:, 1] -= padding[2]
            det_bboxes[:, 2] -= padding[0]
            det_bboxes[:, 3] -= padding[2]

            scales = [
                (img_shape[1] - padding[0] - padding[1]) / ori_shape[1],
                (img_shape[0] - padding[2] - padding[3]) / ori_shape[0],
            ]

        else:
            scales = [img_shape[1] / ori_shape[1], img_shape[0] / ori_shape[0]]

        # Rescale to original shape
        det_bboxes /= det_bboxes.new_tensor(scales).repeat((1, 2))

        return det_bboxes, scores, det_labels, det_bboxes3d
