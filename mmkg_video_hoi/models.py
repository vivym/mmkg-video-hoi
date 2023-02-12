from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .slowfast.models import optimizer as optim
from .slowfast.models.build import build_model
from .slowfast.utils.parser import load_config


@dataclass
class Args:
    cfg_file: str
    opts: str


class MMVideoHOI(nn.Module):
    def __init__(self):
        super().__init__()

        args = Args(
            cfg_file=str(Path(__file__).parent / "configs" / "config.yaml"),
            opts=None,
        )
        self.cfg = load_config(args)
        self.model = build_model(self.cfg)

    def run_train(self):
        optimizer = optim.construct_optimizer(
            self.model, self.cfg
        )

    @torch.no_grad()
    def inference(self, inputs, meta):
        cfg = self.cfg

        trajectories = human_poses = trajectory_boxes = skeleton_imgs = trajectory_box_masks = None
        if cfg.MODEL.USE_TRAJECTORIES:
            trajectories = meta['trajectories']
        if cfg.MODEL.USE_HUMAN_POSES:
            human_poses = meta['human_poses']
        if cfg.DETECTION.ENABLE_TOI_POOLING or cfg.MODEL.USE_TRAJECTORY_CONV:
            trajectory_boxes = meta['trajectory_boxes']
        if cfg.MODEL.USE_SPA_CONF:
            skeleton_imgs = meta['skeleton_imgs']
            trajectory_box_masks = meta['trajectory_box_masks']
        preds, action_labels, bbox_pair_ids, gt_bbox_pair_ids = self.model(
            inputs,
            meta["boxes"],
            meta['proposal_classes'],
            meta['proposal_lengths'],
            meta['action_labels'],
            meta['obj_classes'],
            meta['obj_classes_lengths'],
            trajectories=trajectories,
            human_poses=human_poses,
            trajectory_boxes=trajectory_boxes,
            skeleton_imgs=skeleton_imgs,
            trajectory_box_masks=trajectory_box_masks,
        )

        preds_score = F.sigmoid(preds).cpu()
        preds = preds_score >= 0.5 # Convert scores into 'True' or 'False'
        action_labels = action_labels.cpu()
        boxes = meta["boxes"].cpu()
        obj_classes = meta['obj_classes'].cpu()
        # obj_classes_lengths = meta['obj_classes_lengths'].cpu()
        bbox_pair_ids = bbox_pair_ids.cpu()
        gt_bbox_pair_ids = gt_bbox_pair_ids.cpu()
        # hopairs = hopairs # .cpu()
        proposal_scores = meta['proposal_scores'].cpu()
        gt_boxes = meta['gt_boxes'].cpu()
        proposal_classes = meta['proposal_classes'].cpu()

        return preds_score, preds, proposal_scores, proposal_classes


if __name__ == "__main__":
    MMVideoHOI()
