from typing import Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
from model.stratified_transformer import Stratified
from model.common import MLPWithoutResidual, KPConvResBlock, AggregatorLayer

import torch_points_kernels as tp
from util.logger import get_logger
from lib.pointops2.functions import pointops


class COSeg(nn.Module):
    def __init__(self, args):
        super(COSeg, self).__init__()
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.n_subprototypes = args.n_subprototypes
        self.n_queries = args.n_queries
        self.n_classes = self.n_way + 1
        self.args = args
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([0.1] + [1 for _ in range(self.n_way)]),
            ignore_index=args.ignore_label,
        )
        self.criterion_base = nn.CrossEntropyLoss(
            ignore_index=args.ignore_label
        )

        args.patch_size = args.grid_size * args.patch_size
        args.window_size = [
            args.patch_size * args.window_size * (2**i)
            for i in range(args.num_layers)
        ]
        args.grid_sizes = [
            args.patch_size * (2**i) for i in range(args.num_layers)
        ]
        args.quant_sizes = [
            args.quant_size * (2**i) for i in range(args.num_layers)
        ]

        if args.data_name == "s3dis":
            self.base_classes = 6
            if args.cvfold == 1:
                self.base_class_to_pred_label = {
                    0: 1,
                    3: 2,
                    4: 3,
                    8: 4,
                    10: 5,
                    11: 6,
                }
            else:
                self.base_class_to_pred_label = {
                    1: 1,
                    2: 2,
                    5: 3,
                    6: 4,
                    7: 5,
                    9: 6,
                }
        else:
            self.base_classes = 10
            if args.cvfold == 1:
                self.base_class_to_pred_label = {
                    2: 1,
                    3: 2,
                    5: 3,
                    6: 4,
                    7: 5,
                    10: 6,
                    12: 7,
                    13: 8,
                    14: 9,
                    19: 10,
                }
            else:
                self.base_class_to_pred_label = {
                    1: 1,
                    4: 2,
                    8: 3,
                    9: 4,
                    11: 5,
                    15: 6,
                    16: 7,
                    17: 8,
                    18: 9,
                    20: 10,
                }

        if self.main_process():
            self.logger = get_logger(args.save_path)

        self.encoder = Stratified(
            args.downsample_scale,
            args.depths,
            args.channels,
            args.num_heads,
            args.window_size,
            args.up_k,
            args.grid_sizes,
            args.quant_sizes,
            rel_query=args.rel_query,
            rel_key=args.rel_key,
            rel_value=args.rel_value,
            drop_path_rate=args.drop_path_rate,
            concat_xyz=args.concat_xyz,
            num_classes=self.args.classes // 2 + 1,
            ratio=args.ratio,
            k=args.k,
            prev_grid_size=args.grid_size,
            sigma=1.0,
            num_layers=args.num_layers,
            stem_transformer=args.stem_transformer,
            backbone=True,
            logger=get_logger(args.save_path),
        )

        self.feat_dim = args.channels[2]

        self.visualization = args.vis

        self.lin1 = nn.Sequential(
            nn.Linear(self.n_subprototypes, self.feat_dim),
            nn.ReLU(inplace=True),
        )

        self.kpconv = KPConvResBlock(
            self.feat_dim, self.feat_dim, 0.04, sigma=2
        )

        self.cls = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(self.feat_dim, self.n_classes),
        )

        self.bk_ffn = nn.Sequential(
            nn.Linear(self.feat_dim + self.feat_dim // 2, 4 * self.feat_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4 * self.feat_dim, self.feat_dim),
        )

        if self.args.data_name == "s3dis":
            agglayers = 2
        else:
            agglayers = 4

        print(f"use agglayers {agglayers}")
        self.agglayers = nn.ModuleList(
            [
                AggregatorLayer(
                    hidden_dim=self.feat_dim,
                    text_guidance_dim=0,
                    appearance_guidance=0,
                    nheads=4,
                    attention_type="linear",
                )
                for _ in range(agglayers)
            ]
        )

        if self.n_way == 1:
            self.class_reduce = nn.Sequential(
                nn.LayerNorm(self.feat_dim),
                nn.Conv1d(self.n_classes, 1, kernel_size=1),
                nn.ReLU(inplace=True),
            )
        else:
            self.class_reduce = MLPWithoutResidual(
                self.feat_dim * (self.n_way + 1), self.feat_dim
            )

        self.bg_proto_reduce = MLPWithoutResidual(
            self.n_subprototypes * self.n_way, self.n_subprototypes
        )

        self.init_weights()

        self.register_buffer(
            "base_prototypes", torch.zeros(self.base_classes, self.feat_dim)
        )

    def init_weights(self):
        for name, m in self.named_parameters():
            if "class_attention.base_merge" in name:
                continue
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)

    def main_process(self):
        return not self.args.multiprocessing_distributed or (
            self.args.multiprocessing_distributed
            and self.args.rank % self.args.ngpus_per_node == 0
        )

    def forward(
        self,
        support_offset: torch.Tensor,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_offset: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        epoch: int,
        support_base_y: Optional[torch.Tensor] = None,
        query_base_y: Optional[torch.Tensor] = None,
        sampled_classes: Optional[np.array] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the COSEG model.

        Args:
            support_offset: Offset of each scene in support inputs. Shape (N_way*K_shot).
            support_x: Support point cloud input with shape (N_support, in_channels).
            support_y: Support masks with shape (N_support).
            query_offset: Offset of each scene in support inputs. Shape (N_way).
            query_x: Query point cloud input with shape (N_query, in_channels).
            query_y: Query labels with shape (N_query).
            epoch: Current epoch.
            support_base_y: Support base class labels with shape (N_support).
            query_base_y: Query base class labels with shape (N_query).
            sampled_classes: Sampled classes in current episode. Shape (N_way).

        Returns:
            query_pred: Predicted class logits for query point clouds. Shape: (1, n_way+1, N_query).
            loss: Forward loss value.
        """

        # get downsampled support features
        (
            support_feat,  # N_s, C
            support_x_low,  # N_s, 3
            support_offset,
            support_y_low,  # N_s
            _,
            support_base_y,  # N_s
        ) = self.getFeatures(
            support_x, support_offset, support_y, support_base_y
        )
        assert support_y_low.shape[0] == support_x_low.shape[0]
        # split support features and coords into list according to offset
        support_offset = support_offset[:-1].long().cpu()
        support_feat = torch.tensor_split(support_feat, support_offset)
        support_x_low = torch.tensor_split(support_x_low, support_offset)
        if support_base_y is not None:
            support_base_y = torch.tensor_split(support_base_y, support_offset)

        # get prototypes
        fg_mask = support_y_low
        bg_mask = torch.logical_not(support_y_low)
        fg_mask = torch.tensor_split(fg_mask, support_offset)
        bg_mask = torch.tensor_split(bg_mask, support_offset)

        # For k_shot, extract N_pt/k_shot per shot
        fg_prototypes = self.getPrototypes(
            support_x_low,
            support_feat,
            fg_mask,
            k=self.n_subprototypes // self.k_shot,
        )  # N_way*N_pt, C
        bg_prototype = self.getPrototypes(
            support_x_low,
            support_feat,
            bg_mask,
            k=self.n_subprototypes // self.k_shot,
        )  # N_way*N_pt, C

        # reduce the number of bg_prototypes to n_subprototypes when N_way > 1
        if bg_prototype.shape[0] > self.n_subprototypes:
            bg_prototype = self.bg_proto_reduce(
                bg_prototype.permute(1, 0)
            ).permute(1, 0)

        sparse_embeddings = torch.cat(
            [bg_prototype, fg_prototypes]
        )  # (N_way+1)*N_pt, C

        # get downsampled query features
        (
            query_feat,  # N_q, C
            query_x_low,  # N_q, 3
            query_offset_low,
            query_y_low,  # N_q
            q_base_pred,  # N_q, N_base_classes
            query_base_y,  # N_q
        ) = self.getFeatures(query_x, query_offset, query_y, query_base_y)

        # split query features into list according to offset
        query_offset_low_cpu = query_offset_low[:-1].long().cpu()
        query_feat = torch.tensor_split(query_feat, query_offset_low_cpu)
        query_x_low_list = torch.tensor_split(
            query_x_low, query_offset_low_cpu
        )
        if query_base_y is not None:
            query_base_y_list = torch.tensor_split(
                query_base_y, query_offset_low_cpu
            )

        # update base prototypes
        if self.training:
            for base_feat, base_y in zip(
                list(query_feat) + list(support_feat),
                list(query_base_y_list) + list(support_base_y),
            ):
                cur_baseclsses = base_y.unique()
                cur_baseclsses = cur_baseclsses[
                    cur_baseclsses != 0
                ]  # remove background
                for class_label in cur_baseclsses:
                    class_mask = base_y == class_label
                    class_features = (
                        base_feat[class_mask].sum(dim=0) / class_mask.sum()
                    ).detach()  # C
                    # use the current features to update when the base prototype is initialized
                    if torch.all(self.base_prototypes[class_label - 1] == 0):
                        self.base_prototypes[class_label - 1] = class_features
                    else:  # exponential moving average
                        self.base_prototypes[class_label - 1] = (
                            self.base_prototypes[class_label - 1] * 0.995
                            + class_features * 0.005
                        )
            # mask out the base porotype for current target classes which should not be considered as background
            mask_list = [
                self.base_class_to_pred_label[base_cls] - 1
                for base_cls in sampled_classes
            ]
            base_mask = self.base_prototypes.new_ones(
                (self.base_prototypes.shape[0]), dtype=torch.bool
            )
            base_mask[mask_list] = False
            base_avail_pts = self.base_prototypes[base_mask]
            assert len(base_avail_pts) == self.base_classes - self.n_way
        else:
            base_avail_pts = self.base_prototypes

        query_pred = []
        for i, q_feat in enumerate(query_feat):
            # get base guidance, warm up for 1 epoch
            if epoch < 1:
                base_guidance = None
            else:
                base_similarity = F.cosine_similarity(
                    q_feat[:, None, :],  # N_q, 1, C
                    base_avail_pts[None, :, :],  # 1, N_base_classes, C
                    dim=2,
                )  # N_q, N_base_classes
                # max similarity for each query point as the base guidance
                base_guidance = base_similarity.max(dim=1, keepdim=True)[
                    0
                ]  # N_q, 1

            correlations = F.cosine_similarity(
                q_feat[:, None, :],
                sparse_embeddings[None, :, :],  # 1, (N_way+1)*N_pt, C
                dim=2,
            )  # N_q, (N_way+1)*N_pt
            correlations = (
                self.lin1(
                    correlations.view(
                        correlations.shape[0], self.n_way + 1, -1
                    )  # N_q, (N_way+1), N_pt
                )
                .permute(2, 1, 0)
                .unsqueeze(0)
            )  # 1, C, N_way+1, N_q

            for layer in self.agglayers:
                correlations = layer(
                    correlations, base_guidance
                )  # 1, C, N_way+1, N_q

            correlations = (
                correlations.squeeze(0).permute(2, 1, 0).contiguous()
            )  # N_q, N_way+1, C

            # reduce the class dimension
            if self.n_way == 1:
                correlations = self.class_reduce(correlations).squeeze(
                    1
                )  # N_q, C
            else:
                correlations = self.class_reduce(
                    correlations.view(correlations.shape[0], -1)
                )  # N_q, C

            # kpconv layer
            coord = query_x_low_list[i]  # N_q, 3
            batch = torch.zeros(
                correlations.shape[0], dtype=torch.int64, device=coord.device
            )
            sigma = 2.0
            radius = 2.5 * self.args.grid_size * sigma
            neighbors = tp.ball_query(
                radius,
                self.args.max_num_neighbors,
                coord,
                coord,
                mode="partial_dense",
                batch_x=batch,
                batch_y=batch,
            )[
                0
            ]  # N_q, max_num_neighbors
            correlations = self.kpconv(
                correlations, coord, batch, neighbors.clone()
            )  # N_q, C

            # classification layer
            out = self.cls(correlations)  # N_q, n_way+1
            query_pred.append(out)

        query_pred = torch.cat(query_pred)  # N_q, n_way+1

        assert not torch.any(
            torch.isnan(query_pred)
        ), "torch.any(torch.isnan(query_pred))"
        loss = self.criterion(query_pred, query_y_low)
        if query_base_y is not None:
            loss += self.criterion_base(q_base_pred, query_base_y.cuda())

        final_pred = (
            pointops.interpolation(
                query_x_low,
                query_x[:, :3].cuda().contiguous(),
                query_pred.contiguous(),
                query_offset_low,
                query_offset.cuda(),
            )
            .transpose(0, 1)
            .unsqueeze(0)
        )  # 1, n_way+1, N_query

        # wandb visualization
        if self.visualization:
            self.vis(
                query_offset,
                query_x,
                query_y,
                support_offset,
                support_x,
                support_y,
                final_pred,
            )

        return final_pred, loss

    def getFeatures(self, ptclouds, offset, gt, query_base_y=None):
        """
        Get the features of one point cloud from backbone network.

        Args:
            ptclouds: Input point clouds with shape (N_pt, 6), where N_pt is the number of points.
            offset: Offset tensor with shape (b), where b is the number of query scenes.
            gt: Ground truth labels. shape (N_pt).
            query_base_y: Optional base class labels for input point cloud. shape (N_pt).

        Returns:
            feat: Features from backbone with shape (N_down, C), where C is the number of channels.
            coord: Point coords. Shape (N_down, 3).
            offset: Offset for each scene. Shape (b).
            gt: Ground truth labels. Shape (N_down).
            base_pred: Base class predictions from backbone. Shape (N_down, N_base_classes).
            query_base_y: Base class labels for input point cloud. Shape (N_down).
        """
        coord, feat = (
            ptclouds[:, :3].contiguous(),
            ptclouds[:, 3:6].contiguous(),  # rgb color
        )  # (N_pt, 3), (N_pt, 3)

        offset_ = offset.clone()
        offset_[1:] = offset_[1:] - offset_[:-1]
        batch = torch.cat(
            [torch.tensor([ii] * o) for ii, o in enumerate(offset_)], 0
        ).long()  # N_pt

        sigma = 1.0
        radius = 2.5 * self.args.grid_size * sigma
        batch = batch.to(coord.device)
        neighbor_idx = tp.ball_query(
            radius,
            self.args.max_num_neighbors,
            coord,
            coord,
            mode="partial_dense",
            batch_x=batch,
            batch_y=batch,
        )[
            0
        ]  # (N_pt, max_num_neighbors)

        coord, feat, offset, gt = (
            coord.cuda(non_blocking=True),
            feat.cuda(non_blocking=True),
            offset.cuda(non_blocking=True),
            gt.cuda(non_blocking=True),
        )
        batch = batch.cuda(non_blocking=True)
        neighbor_idx = neighbor_idx.cuda(non_blocking=True)
        assert batch.shape[0] == feat.shape[0]

        if self.args.concat_xyz:
            feat = torch.cat([feat, coord], 1)  # N_pt, 6
        # downsample the input point clouds
        feat, coord, offset, gt, base_pred, query_base_y = self.encoder(
            feat, coord, offset, batch, neighbor_idx, gt, query_base_y
        )  # (N_down, C_bc) (N_down, 3) (b), (N_down), (N_down, N_base_classes), (N_down)

        feat = self.bk_ffn(feat)  # N_down, C
        return feat, coord, offset, gt, base_pred, query_base_y

    def getPrototypes(self, coords, feats, masks, k=100):
        """
        Extract k prototypes for each scene.

        Args:
            coords: Point coordinates. List of (N_pt, 3).
            feats: Point features. List of (N_pt, C).
            masks: Target class masks. List of (N_pt).
            k: Number of prototypes extracted in each shot (default: 100).

        Return:
            prototypes: Shape (n_way*k_shot*k, C).
        """
        prototypes = []
        for i in range(0, self.n_way * self.k_shot):
            coord = coords[i][:, :3]  # N_pt, 3
            feat = feats[i]  # N_pt, C
            mask = masks[i].bool()  # N_pt

            coord_mask = coord[mask]
            feat_mask = feat[mask]
            protos = self.getMutiplePrototypes(
                coord_mask, feat_mask, k
            )  # k, C
            prototypes.append(protos)

        prototypes = torch.cat(prototypes)  # n_way*k_shot*k, C
        return prototypes

    def getMutiplePrototypes(self, coord, feat, num_prototypes):
        """
        Extract k prototypes using furthest point samplling

        Args:
            coord: Point coordinates. Shape (N_pt, 3)
            feat: Point features. Shape (N_pt, C).
            num_prototypes: Number of prototypes to extract.
        Return:
            prototypes: Extracted prototypes. Shape: (num_prototypes, C).
        """
        # when the number of points is less than the number of prototypes, pad the points with zero features
        if feat.shape[0] <= num_prototypes:
            no_feats = feat.new_zeros(
                1,
                self.feat_dim,
            ).expand(num_prototypes - feat.shape[0], -1)
            feat = torch.cat([feat, no_feats])
            return feat

        # sample k seeds  by Farthest Point Sampling
        fps_index = pointops.furthestsampling(
            coord,
            torch.cuda.IntTensor([coord.shape[0]]),
            torch.cuda.IntTensor([num_prototypes]),
        ).long()  # (num_prototypes,)

        # use the k seeds as initial centers and compute the point-to-seed distance
        num_prototypes = len(fps_index)
        farthest_seeds = feat[fps_index]  # (num_prototypes, feat_dim)
        distances = torch.linalg.norm(
            feat[:, None, :] - farthest_seeds[None, :, :], dim=2
        )  # (N_pt, num_prototypes)

        # clustering the points to the nearest seed
        assignments = torch.argmin(distances, dim=1)  # (N_pt,)

        # aggregating each cluster to form prototype
        prototypes = torch.zeros(
            (num_prototypes, self.feat_dim), device="cuda"
        )
        for i in range(num_prototypes):
            selected = torch.nonzero(assignments == i).squeeze(
                1
            )  # (N_selected,)
            selected = feat[selected, :]  # (N_selected, C)
            if (
                len(selected) == 0
            ):  # exists same prototypes (coord not same), points are assigned to the prior prototype
                # simple use the seed as the prototype here
                prototypes[i] = feat[fps_index[i]]
                if self.main_process():
                    self.logger.info("len(selected) == 0")
            else:
                prototypes[i] = selected.mean(0)  # (C,)

        return prototypes

    def vis(
        self,
        query_offset,
        query_x,
        query_y,
        support_offset,
        support_x,
        support_y,
        final_pred,
    ):
        query_offset_cpu = query_offset[:-1].long().cpu()
        query_x_splits = torch.tensor_split(query_x, query_offset_cpu)
        query_y_splits = torch.tensor_split(query_y, query_offset_cpu)
        vis_pred = torch.tensor_split(final_pred, query_offset_cpu, dim=-1)
        vis_mask = torch.tensor_split(support_y, support_offset)

        sp_nps, sp_fgs = [], []
        for i, support_x_split in enumerate(
            torch.tensor_split(support_x, support_offset)
        ):
            sp_np = (
                support_x_split.detach().cpu().numpy()
            )  # num_points, in_channels
            sp_np[:, 3:6] = sp_np[:, 3:6] * 255.0
            sp_fg = np.concatenate(
                (
                    sp_np[:, :3],
                    vis_mask[i].unsqueeze(-1).detach().cpu().numpy(),
                ),
                axis=-1,
            )
            sp_nps.append(sp_np)
            sp_fgs.append(sp_fg)

        qu_s, qu_gts, qu_pds = [], [], []
        for i, query_x_split in enumerate(query_x_splits):
            qu = (
                query_x_split.detach().cpu().numpy()
            )  # num_points, in_channels
            qu[:, 3:6] = qu[:, 3:6] * 255.0
            result_tensor = torch.where(
                query_y_splits[i] == 255,
                torch.tensor(0, device=query_y.device),
                query_y_splits[i],
            )
            qu_gt = np.concatenate(
                (
                    qu[:, :3],
                    result_tensor.unsqueeze(-1).detach().cpu().numpy(),
                ),
                axis=-1,
            )
            q_prd = np.concatenate(
                (
                    qu[:, :3],
                    vis_pred[i]
                    .squeeze(0)
                    .max(0)[1]
                    .unsqueeze(-1)
                    .detach()
                    .cpu()
                    .numpy(),
                ),
                axis=-1,
            )

            qu_s.append(qu)
            qu_gts.append(qu_gt)
            qu_pds.append(q_prd)

        wandb.log(
            {
                "Support": [
                    wandb.Object3D(sp_nps[i]) for i in range(len(sp_nps))
                ],
                "Support_fg": [
                    wandb.Object3D(sp_fgs[i]) for i in range(len(sp_fgs))
                ],
                "Query": [wandb.Object3D(qu_s[i]) for i in range(len(qu_s))],
                "Query_pred": [
                    wandb.Object3D(qu_pds[i]) for i in range(len(qu_pds))
                ],
                "Query_GT": [
                    wandb.Object3D(qu_gts[i]) for i in range(len(qu_gts))
                ],
            }
        )
