include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
  map_frame = "map",
  tracking_frame = "base_link",       -- 修改为IMU数据的实际发布帧
  published_frame = "odom",           -- 从odom发布位姿
  odom_frame = "odom",
  provide_odom_frame = false,         -- 因为已有odom，所以不需要提供
  publish_frame_projected_to_2d = false,
  use_odometry = true,                
  use_nav_sat = false,
  use_landmarks = false,
  num_laser_scans = 0,
  num_multi_echo_laser_scans = 0,
  num_subdivisions_per_laser_scan = 1,
  num_point_clouds = 1,
  lookup_transform_timeout_sec = 0.5, -- 增加超时时间，避免TF查找失败
  submap_publish_period_sec = 0.3,
  pose_publish_period_sec = 5e-3,
  trajectory_publish_period_sec = 30e-3,
  rangefinder_sampling_ratio = 1.,
  odometry_sampling_ratio = 1.,
  fixed_frame_pose_sampling_ratio = 1.,
  imu_sampling_ratio = 1.,
  landmarks_sampling_ratio = 1.,
}

MAP_BUILDER.use_trajectory_builder_3d = true
MAP_BUILDER.num_background_threads = 7

TRAJECTORY_BUILDER_3D.num_accumulated_range_data = 1

TRAJECTORY_BUILDER_3D.min_range = 0.01
TRAJECTORY_BUILDER_3D.max_range = 100.0
TRAJECTORY_BUILDER_3D.voxel_filter_size = 0.15

-- TRAJECTORY_BUILDER_3D.high_resolution_adaptive_voxel_filter.max_length = 2.0
-- TRAJECTORY_BUILDER_3D.high_resolution_adaptive_voxel_filter.min_num_points = 150
-- TRAJECTORY_BUILDER_3D.low_resolution_adaptive_voxel_filter.max_length = 4.0
-- TRAJECTORY_BUILDER_3D.low_resolution_adaptive_voxel_filter.min_num_points = 200

-- TRAJECTORY_BUILDER_3D.use_online_correlative_scan_matching = true
-- TRAJECTORY_BUILDER_3D.real_time_correlative_scan_matcher.linear_search_window = 0.15
-- TRAJECTORY_BUILDER_3D.real_time_correlative_scan_matcher.angular_search_window = math.rad(1.)

-- TRAJECTORY_BUILDER_3D.ceres_scan_matcher.occupied_space_weight_0 = 1.
-- TRAJECTORY_BUILDER_3D.ceres_scan_matcher.occupied_space_weight_1 = 6.
-- TRAJECTORY_BUILDER_3D.ceres_scan_matcher.translation_weight = 5.
-- TRAJECTORY_BUILDER_3D.ceres_scan_matcher.rotation_weight = 4e2
-- TRAJECTORY_BUILDER_3D.ceres_scan_matcher.only_optimize_yaw = false

-- -- 回环检测参数优化
-- POSE_GRAPH.constraint_builder.sampling_ratio = 0.3
-- POSE_GRAPH.optimization_problem.ceres_solver_options.max_num_iterations = 10
-- POSE_GRAPH.constraint_builder.min_score = 0.55
-- POSE_GRAPH.constraint_builder.global_localization_min_score = 0.6
-- POSE_GRAPH.optimization_problem.huber_scale = 1e1

-- -- 增加额外参数以提高点云匹配性能
-- TRAJECTORY_BUILDER_3D.submaps.high_resolution = 0.10
-- TRAJECTORY_BUILDER_3D.submaps.low_resolution = 0.45
-- TRAJECTORY_BUILDER_3D.submaps.num_range_data = 160

return options 