import json
import os

import cv2
import evo
import numpy as np
import torch
import copy 

from evo.tools import log
log.configure_logging(verbose=False, debug=True, silent=False)
from evo.tools.settings import SETTINGS
SETTINGS.plot_backend = 'Agg'
from evo.tools import plot
plot.apply_settings(SETTINGS)

from evo.core import metrics, trajectory, sync
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
# from evo.tools import plot
# from evo.tools.plot import PlotMode
# from evo.tools.settings import SETTINGS
from matplotlib import pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import wandb
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.logging_utils import Log


# def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False):
#     ## Plot
#     traj_ref = PosePath3D(poses_se3=poses_gt)
#     traj_est = PosePath3D(poses_se3=poses_est)
#     traj_est_aligned = trajectory.align_trajectory(
#         traj_est, traj_ref, correct_scale=monocular
#     )

#     ## RMSE
#     pose_relation = metrics.PoseRelation.translation_part
#     data = (traj_ref, traj_est_aligned)
#     ape_metric = metrics.APE(pose_relation)
#     ape_metric.process_data(data)
#     ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
#     ape_stats = ape_metric.get_all_statistics()
#     Log("RMSE ATE \[m]", ape_stat, tag="Eval")

#     with open(
#         os.path.join(plot_dir, "stats_{}.json".format(str(label))),
#         "w",
#         encoding="utf-8",
#     ) as f:
#         json.dump(ape_stats, f, indent=4)

#     plot_mode = evo.tools.plot.PlotMode.xy
#     fig = plt.figure()
#     ax = evo.tools.plot.prepare_axis(fig, plot_mode)
#     ax.set_title(f"ATE RMSE: {ape_stat}")
#     evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
#     evo.tools.plot.traj_colormap(
#         ax,
#         traj_est_aligned,
#         ape_metric.error,
#         plot_mode,
#         min_map=ape_stats["min"],
#         max_map=ape_stats["max"],
#     )
#     ax.legend()
#     plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))), dpi=90)

#     return ape_stat


# def eval_ate(frames, kf_ids, save_dir, iterations, final=False, monocular=False):
#     trj_data = dict()
#     latest_frame_idx = kf_ids[-1] + 2 if final else kf_ids[-1] + 1
#     trj_id, trj_est, trj_gt = [], [], []
#     trj_est_np, trj_gt_np = [], []

#     def gen_pose_matrix(R, T):
#         pose = np.eye(4)
#         pose[0:3, 0:3] = R.cpu().numpy()
#         pose[0:3, 3] = T.cpu().numpy()
#         return pose

#     for kf_id in kf_ids:
#         kf = frames[kf_id]
#         pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
#         pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))

#         trj_id.append(frames[kf_id].uid)
#         trj_est.append(pose_est.tolist())
#         trj_gt.append(pose_gt.tolist())

#         trj_est_np.append(pose_est)
#         trj_gt_np.append(pose_gt)

#     trj_data["trj_id"] = trj_id
#     trj_data["trj_est"] = trj_est
#     trj_data["trj_gt"] = trj_gt

#     plot_dir = os.path.join(save_dir, "plot")
#     mkdir_p(plot_dir)

#     label_evo = "final" if final else "{:04}".format(iterations)
#     with open(
#         os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
#     ) as f:
#         json.dump(trj_data, f, indent=4)

#     ate = evaluate_evo(
#         poses_gt=trj_gt_np,
#         poses_est=trj_est_np,
#         plot_dir=plot_dir,
#         label=label_evo,
#         monocular=monocular,
#     )
#     wandb.log({"frame_idx": latest_frame_idx, "ate": ate})
#     return ate


def compute_metric(data: tuple, metric: str, pose_relation: metrics.PoseRelation):
    if metric == 'APE':
        pe_metric = metrics.APE(pose_relation)
    elif metric == 'RPE':
        # normal mode
        delta = 1
        delta_unit = Unit.frames
        all_pairs = False  # activate
        pe_metric = metrics.RPE(pose_relation=pose_relation, delta=delta, delta_unit=delta_unit, all_pairs=all_pairs)
    else:
        assert False, "Metric not identified: {}".format(metric)

    pe_metric.process_data(data)
    pe_stat = pe_metric.get_statistic(metrics.StatisticsType.rmse)
    pe_stats = pe_metric.get_all_statistics()
    return pe_stat, pe_stats, pe_metric


def evaluate_evo_full(poses_gt, poses_est, plot_dir, label, monocular=False):
    # sincronize data & align 
    max_diff = 0.01
    traj_ref, traj_est = PosePath3D(poses_se3=poses_gt), PosePath3D(poses_se3=poses_est)
    # traj_ref, traj_est = sync.associate_trajectories(poses_gt, poses_est, max_diff)
    traj_est_aligned = copy.deepcopy(traj_est)
    traj_est_aligned.align(traj_ref, correct_scale=False, correct_only_scale=False)

    ## RMSE
    metrics_list = ["APE", "RPE"]
    pose_relations = [metrics.PoseRelation.translation_part]#, metrics.PoseRelation.rotation_angle_rad]
    data = (traj_ref, traj_est_aligned)
    stats_full = {}
    stat_full = {}
    for metric in metrics_list:
        curr_stats = {}
        curr_stat = {}
        for pose_relation in pose_relations:
            pe_stat, pe_stats, pe_metric = compute_metric(data, metric, pose_relation)
            curr_stats[str(pose_relation)] = pe_stats
            curr_stat[str(pose_relation)] = pe_stat
            # log out 
            Log(f"RMSE {metric}/{str(pose_relation)} [m]", pe_stat, tag="Eval")
            # plot 
            plot_mode = plot.PlotMode.xy
            fig = plt.figure()
            ax = plot.prepare_axis(fig, plot_mode)
            plot.traj(ax, plot_mode, traj_ref, '--', "gray", "reference")
            plot.traj_colormap(ax, traj_est_aligned, pe_metric.error, 
                            plot_mode, min_map=pe_stats["min"], max_map=pe_stats["max"])
            ax.legend()
            plt.show()
            plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))), dpi=90)
        stats_full[metric] = curr_stats
        stat_full[metric] = curr_stat

    with open(
        os.path.join(plot_dir, "stats_{}.json".format(str(label))),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(stats_full, f, indent=4)

    return stat_full


def eval_trajectory(frames, kf_ids, save_dir, iterations, final=False, monocular=False):
    trj_data = dict()
    latest_frame_idx = kf_ids[-1] + 2 if final else kf_ids[-1] + 1
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose

    for kf_id in kf_ids:
        kf = frames[kf_id]
        pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
        pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))

        trj_id.append(frames[kf_id].uid)
        trj_est.append(pose_est.tolist())
        trj_gt.append(pose_gt.tolist())

        trj_est_np.append(pose_est)
        trj_gt_np.append(pose_gt)

    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    plot_dir = os.path.join(save_dir, "plot")
    mkdir_p(plot_dir)

    label_evo = "final" if final else "{:04}".format(iterations)
    with open(
        os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trj_data, f, indent=4)

    stat_full = evaluate_evo_full(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        plot_dir=plot_dir,
        label=label_evo,
        monocular=monocular,
    )
    wandb.log({"frame_idx": latest_frame_idx, "ate_trans": stat_full["APE"][str(metrics.PoseRelation.translation_part)]})
    # wandb.log({"frame_idx": latest_frame_idx, "ate_rot": stat_full["APE"][str(metrics.PoseRelation.rotation_angle_rad)]})
    wandb.log({"frame_idx": latest_frame_idx, "rte_trans": stat_full["RPE"][str(metrics.PoseRelation.translation_part)]})
    # wandb.log({"frame_idx": latest_frame_idx, "rte_rot": stat_full["RPE"][str(metrics.PoseRelation.rotation_angle_rad)]})

    return stat_full


def eval_rendering(
    frames,
    gaussians,
    dataset,
    save_dir,
    pipe,
    background,
    kf_indices,
    iteration="final",
):
    interval = 5
    img_pred, img_gt, saved_frame_idx = [], [], []
    end_idx = len(frames) - 1 if iteration == "final" or "before_opt" else iteration
    psnr_array, ssim_array, lpips_array = [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")
    for idx in range(0, end_idx, interval):
        if idx in kf_indices:
            continue
        saved_frame_idx.append(idx)
        frame = frames[idx]
        gt_image, _, _ = dataset[idx]

        rendering = render(frame, gaussians, pipe, background)["render"]
        image = torch.clamp(rendering, 0.0, 1.0)

        gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        img_pred.append(pred)
        img_gt.append(gt)

        mask = gt_image > 0

        psnr_score = psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))

    Log(
        f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}',
        tag="Eval",
    )

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )
    return output


def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
