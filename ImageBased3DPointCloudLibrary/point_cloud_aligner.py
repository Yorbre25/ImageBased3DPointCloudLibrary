import open3d as o3d
import numpy as np
import warnings


def check_inputs(source, target, work_with_n):
    if not isinstance(source, o3d.geometry.PointCloud):
        raise TypeError("source must be of type open3d.geometry.PointCloud")
    if not isinstance(target, o3d.geometry.PointCloud):
        raise TypeError("target must be of type open3d.geometry.PointCloud")
    if not isinstance(work_with_n, int):
        raise TypeError("work_with_n must be of type int")
    if work_with_n <= 10000:
        raise TypeError("work_with_n should be greater than 10000")


def align_point_clouds(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    work_with_n_points: int = 150000,
):
    check_inputs(source, target, work_with_n_points)

    results_dict = {}
    voxel_size = calc_voxel_size(source, target, work_with_n_points)
    results_dict["voxel_size"] = voxel_size

    radius_feature = voxel_size * 5
    radius_normal = voxel_size * 2
    results_dict["radius_feature"] = radius_feature
    results_dict["radius_normal"] = radius_normal

    source_down, source_fpfh = preprocess_point_cloud(
        source,
        voxel_size=voxel_size,
        radius_feature=radius_feature,
        radius_normal=radius_normal,
        results_dict=results_dict,
        key="source",
    )
    target_down, target_fpfh = preprocess_point_cloud(
        target,
        voxel_size=voxel_size,
        radius_feature=radius_feature,
        radius_normal=radius_normal,
        results_dict=results_dict,
        key="target",
    )

    distance_threshold = voxel_size * 1.5
    results_dict["distance_threshold"] = distance_threshold
    result = execute_global_registration(
        source_down=source_down,
        target_down=target_down,
        source_fpfh=source_fpfh,
        target_fpfh=target_fpfh,
        distance_threshold=distance_threshold,
    )

    results_dict["correspondece_set_size"] = len(np.asarray(result.correspondence_set))
    results_dict["fitness"] = result.fitness
    results_dict["inlier_rmse"] = result.inlier_rmse
    return results_dict, result.transformation


def calc_voxel_size(source, target, work_with_n_points):
    initial_voxel_size = 0.1
    voxel_size_source = compute_voxel_size(
        source, initial_voxel_size, work_with_n_points
    )
    voxel_size_target = compute_voxel_size(
        target, initial_voxel_size, work_with_n_points
    )
    return np.mean([voxel_size_source, voxel_size_target])


def compute_voxel_size(point_cloud, initial_voxel_size, target_points):
    voxel_size = initial_voxel_size
    downsampled = point_cloud
    while len(downsampled.points) > target_points:
        voxel_size += 0.01
        downsampled = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    return voxel_size


def preprocess_point_cloud(
    pcd, voxel_size, radius_feature, radius_normal, results_dict, key
):
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

    radius_normal = voxel_size * 2
    # Noramls are used to calculate the FPFH features
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )

    results_dict[key] = {
        "initial_length": len(pcd.points),
        "downsampled_lenght": len(pcd_down.points),
    }

    return pcd_down, pcd_fpfh


def execute_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, distance_threshold
):

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[  # Pruning, points that pass the pruning will be subject to RANSAC
            # Checking if the edeges of source and target are about 0.9 of each other
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            # Checking if the distance between the points is less than the threshold
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )  # max_iter, confidence
    return result
