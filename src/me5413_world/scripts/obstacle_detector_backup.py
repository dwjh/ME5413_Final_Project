#!/usr/bin/env python3
import rospy
import numpy as np
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener, TransformException
import tf2_geometry_msgs
import cv2

class CubeDetector3D:
    def __init__(self):
        rospy.init_node("cube_detector_3d")
        rospy.loginfo("初始化立方体检测器...")

        self.bridge = CvBridge()
        self.depth_image = None
        self.camera_info = None
        self.got_camera_info = False

        # 等待tf树准备就绪
        rospy.sleep(1.0)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        # 订阅相机信息以获取正确的参数
        self.caminfo_sub = rospy.Subscriber("/front/depth/camera_info", CameraInfo, self.camera_info_callback)

        # 等待获取相机信息
        while not self.got_camera_info and not rospy.is_shutdown():
            rospy.loginfo_throttle(1, "等待相机参数...")
            rospy.sleep(0.1)

        # 订阅点云和深度图像
        self.pc_sub = rospy.Subscriber("/mid/points", PointCloud2, self.cloud_callback)
        self.depth_sub = rospy.Subscriber("/front/depth/image_raw", Image, self.depth_callback)
        
        # 发布检测结果
        self.center_pub = rospy.Publisher("/block_center", PointStamped, queue_size=10)
        self.marker_pub = rospy.Publisher("/block_marker", Marker, queue_size=10)

        self.map_frame = "map"
        self.sensor_frame = "velodyne"  # 点云的frame
        self.camera_frame = "front_frame_optical"  # 相机的frame
        self.block_size = 0.8  # 方块边长

        # 检测参数
        self.min_height = 0.1
        self.max_height = 0.8
        self.max_range = 5.0
        self.min_points = 20
        self.voxel_size = 0.08
        self.cluster_eps = 0.35
        self.cluster_min_points = 8

        # 跟踪参数
        self.tracked_objects = {}  # {track_id: {'position': [x,y,z], 'last_update': time, 'score': float, 'history': [], 'map_position': None}}
        self.permanent_blocks = {}  # 将在检测过程中填充
        
        # 跟踪和更新参数
        self.iou_threshold = 0.2           # 降低IOU阈值使匹配更容易
        self.track_timeout = rospy.Duration(2.0)  # 跟踪超时时间
        self.min_track_score = 0.2         # 最小跟踪分数
        self.permanent_score_threshold = 0.95  # 永久保存阈值
        self.next_track_id = 0
        self.next_permanent_id = 0
        self.history_size = 10             # 历史记录大小
        self.position_threshold = 0.5       # 位置变化阈值
        self.update_weight = 0.3           # 新观测的权重
        self.max_update_distance = 0.3      # 允许更新的最大距离
        self.min_observations = 5          # 计算稳定位置所需的最小观测数
        self.confidence_threshold = 0.8    # 高置信度阈值
        self.max_variance = 0.1           # 最大允许方差

        rospy.loginfo("立方体检测器初始化完成")

    def camera_info_callback(self, msg):
        if not self.got_camera_info:
            self.camera_info = msg
            self.got_camera_info = True
            self.caminfo_sub.unregister()
            rospy.loginfo("已获取相机参数")

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            rospy.loginfo_throttle(5, "✅ 深度图像接收正常")
        except Exception as e:
            rospy.logwarn(f"深度图像转换失败: {e}")

    def calculate_box_iou(self, center1, center2):
        """计算两个立方体的IOU"""
        half_size = self.block_size / 2.0
        
        # 计算两个框的边界
        box1_min = center1 - half_size
        box1_max = center1 + half_size
        box2_min = center2 - half_size
        box2_max = center2 + half_size

        # 计算交集
        intersection_min = np.maximum(box1_min, box2_min)
        intersection_max = np.minimum(box1_max, box2_max)
        
        if np.any(intersection_max < intersection_min):
            return 0.0

        # 计算体积
        intersection_volume = np.prod(intersection_max - intersection_min)
        box_volume = self.block_size ** 3
        
        # 计算IOU
        iou = intersection_volume / (2 * box_volume - intersection_volume)
        return iou

    def update_permanent_block(self, block_id, new_position, current_time):
        """更新永久方块的位置和状态"""
        block = self.permanent_blocks[block_id]
        old_position = block['map_position']
        
        # 更新历史观测
        block.setdefault('observation_history', []).append(new_position)
        if len(block['observation_history']) > self.history_size:
            block['observation_history'].pop(0)
        
        # 计算位置方差
        if len(block['observation_history']) >= 3:
            positions = np.array(block['observation_history'])
            block['variance'] = np.mean(np.var(positions, axis=0))
        
        # 根据观测次数和方差动态调整权重
        update_count = block.get('update_count', 0)
        base_weight = self.update_weight
        
        # 方差越大，权重越小
        if 'variance' in block:
            variance_factor = max(0.1, 1.0 - block['variance'] / self.max_variance)
            base_weight *= variance_factor
        
        # 观测次数越多，权重越小
        if update_count > self.min_observations:
            count_factor = 1.0 / (1 + np.log10(update_count - self.min_observations + 1))
            base_weight *= count_factor
        
        # 计算新位置
        block['map_position'] = (1 - base_weight) * old_position + base_weight * new_position
        block['update_count'] = update_count + 1
        block['last_update'] = current_time
        
        # 更新稳定位置（使用中值）
        if len(block['observation_history']) >= self.min_observations:
            positions = np.array(block['observation_history'])
            block['stable_position'] = np.median(positions, axis=0)
        
        # 更新置信度
        if 'variance' in block and block['update_count'] >= self.min_observations:
            confidence = min(1.0, max(0.0, 1.0 - block['variance'] / self.max_variance))
            block['confidence'] = confidence
        
        rospy.loginfo(f"更新永久方块 {block_id}：更新次数={block['update_count']}, " + 
                     f"方差={block.get('variance', 'N/A'):.3f}, 置信度={block.get('confidence', 'N/A'):.3f}")

    def update_tracks(self, detected_centers):
        """更新跟踪状态"""
        current_time = rospy.Time.now()
        
        # 首先，检查新检测到的中心点是否可以用来更新永久方块
        for center in detected_centers:
            try:
                # 转换到地图坐标系
                point_msg = PointStamped()
                point_msg.header.frame_id = self.sensor_frame
                point_msg.header.stamp = current_time
                point_msg.point.x = center[0]
                point_msg.point.y = center[1]
                point_msg.point.z = center[2]
                
                point_map = self.tf_buffer.transform(point_msg, self.map_frame, rospy.Duration(1.0))
                center_map = np.array([point_map.point.x, point_map.point.y, point_map.point.z])
                
                # 检查是否可以更新任何永久方块
                best_block_id = None
                min_distance = float('inf')
                
                for block_id, block_info in self.permanent_blocks.items():
                    distance = np.linalg.norm(center_map - block_info['map_position'])
                    if distance < self.max_update_distance and distance < min_distance:
                        min_distance = distance
                        best_block_id = block_id
                
                if best_block_id is not None:
                    self.update_permanent_block(best_block_id, center_map, current_time)
                        
            except TransformException:
                continue

        # 删除超时的跟踪
        expired_tracks = []
        permanent_candidates = []  # 用于存储要转为永久方块的跟踪ID
        
        for track_id, track_info in self.tracked_objects.items():
            # 检查是否达到永久保存阈值
            if track_info['score'] >= self.permanent_score_threshold and track_info.get('map_position') is not None:
                permanent_candidates.append((track_id, track_info))
                continue
                
            if (current_time - track_info['last_update']) > self.track_timeout:
                if track_info['score'] > 0.7:  # 如果是高置信度的跟踪对象，延长超时时间
                    continue
                expired_tracks.append(track_id)

        # 处理要永久保存的方块
        for track_id, track_info in permanent_candidates:
            # 检查是否与现有永久方块重叠
            is_overlapping = False
            new_map_pos = track_info['map_position']
            
            for existing_block in self.permanent_blocks.values():
                existing_map_pos = existing_block['map_position']
                # 计算两个方块中心点之间的距离
                distance = np.linalg.norm(new_map_pos - existing_map_pos)
                
                # 如果距离小于方块大小，认为是重叠的
                if distance < self.block_size:
                    is_overlapping = True
                    rospy.loginfo(f"方块 {track_id} 与现有永久方块重叠，跳过保存")
                    break
            
            if not is_overlapping:
                self.permanent_blocks[self.next_permanent_id] = {
                    'position': track_info['position'],
                    'map_position': track_info['map_position'],
                    'update_count': 1,
                    'last_update': current_time,
                    'confidence': 1.0,
                    'observation_history': [track_info['position']],
                    'stable_position': track_info['position'],
                    'variance': 0.0
                }
                self.next_permanent_id += 1
                rospy.loginfo(f"方块 {track_id} 已永久保存到地图中")
            
            expired_tracks.append(track_id)  # 无论是否重叠，都从跟踪列表中移除

        # 删除过期的跟踪
        for track_id in expired_tracks:
            del self.tracked_objects[track_id]

        # 匹配检测结果与现有跟踪
        matched_detections = set()
        matched_tracks = set()

        for center in detected_centers:
            best_iou = 0
            best_track_id = None
            min_distance = float('inf')
            distance_track_id = None

            # 寻找最佳匹配
            for track_id, track_info in self.tracked_objects.items():
                if track_id in matched_tracks:
                    continue
                    
                iou = self.calculate_box_iou(center, track_info['position'])
                distance = np.linalg.norm(center - track_info['position'])

                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id
                
                if distance < min_distance and distance < self.position_threshold:
                    min_distance = distance
                    distance_track_id = track_id

            matched_track_id = best_track_id if best_track_id is not None else distance_track_id

            if matched_track_id is not None:
                # 更新现有跟踪
                track = self.tracked_objects[matched_track_id]
                track['position'] = center
                track['last_update'] = current_time
                track['score'] = min(1.0, track['score'] + 0.1)
                
                # 更新历史位置
                track.setdefault('history', []).append(center)
                if len(track['history']) > self.history_size:
                    track['history'].pop(0)
                
                # 尝试更新地图坐标系中的位置
                try:
                    point_msg = PointStamped()
                    point_msg.header.frame_id = self.sensor_frame
                    point_msg.header.stamp = current_time
                    point_msg.point.x = center[0]
                    point_msg.point.y = center[1]
                    point_msg.point.z = center[2]
                    
                    point_map = self.tf_buffer.transform(point_msg, self.map_frame, rospy.Duration(1.0))
                    track['map_position'] = np.array([point_map.point.x, point_map.point.y, point_map.point.z])
                except TransformException:
                    pass

                matched_detections.add(tuple(center))
                matched_tracks.add(matched_track_id)
            else:
                # 创建新跟踪
                self.tracked_objects[self.next_track_id] = {
                    'position': center,
                    'last_update': current_time,
                    'score': 0.3,
                    'history': [center],
                    'map_position': None
                }
                self.next_track_id += 1

        # 更新未匹配的跟踪
        for track_id in self.tracked_objects:
            if track_id not in matched_tracks:
                track = self.tracked_objects[track_id]
                # 如果有稳定的地图位置，降低惩罚力度
                if track.get('map_position') is not None and track['score'] > 0.7:
                    track['score'] = max(0.5, track['score'] - 0.05)
                else:
                    track['score'] = max(0, track['score'] - 0.1)

    def cloud_callback(self, msg):
        try:
            # 读取点云数据
            cloud_points = list(pc2.read_points(msg, field_names=["x", "y", "z"], skip_nans=True))
            points_np = np.array(cloud_points, dtype=np.float32)
            
            if len(points_np) == 0:
                rospy.logwarn_throttle(1, "点云数据为空")
                return

            # 限制扫描区域
            dist = np.linalg.norm(points_np[:, :2], axis=1)
            height_mask = (points_np[:, 2] > self.min_height) & (points_np[:, 2] < self.max_height)
            range_mask = dist < self.max_range
            points_np = points_np[height_mask & range_mask]

            if len(points_np) < self.min_points:
                rospy.logwarn_throttle(1, "点数过少")
                return

            # 点云处理
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_np)
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=2.5)
            points_np = np.asarray(pcd.points)

            if len(points_np) < self.min_points:
                rospy.logwarn_throttle(1, "下采样后点数过少")
                return

            # 聚类
            labels = np.array(pcd.cluster_dbscan(
                eps=self.cluster_eps,
                min_points=self.cluster_min_points,
                print_progress=False
            ))

            if labels.size == 0 or labels.max() < 0:
                rospy.logwarn_throttle(1, "未找到任何聚类")
                return

            # 处理检测到的聚类
            detected_centers = []
            for cluster_id in range(labels.max() + 1):
                cluster_pts = points_np[labels == cluster_id]
                if len(cluster_pts) < self.min_points:
                    continue

                # 计算边界框
                min_pt = cluster_pts.min(axis=0)
                max_pt = cluster_pts.max(axis=0)
                size = max_pt - min_pt
                
                # 验证大小
                if not (0.3 < size[0] < 1.2 and 0.3 < size[1] < 1.2 and 0.3 < size[2] < 1.2):
                    continue

                center = (min_pt + max_pt) / 2.0
                detected_centers.append(center)

            # 更新跟踪状态
            self.update_tracks(detected_centers)

            # 发布跟踪结果和永久方块
            self.publish_all_blocks(rospy.Time.now())

        except Exception as e:
            rospy.logerr(f"点云处理错误: {e}")

    def publish_all_blocks(self, current_time):
        """发布所有跟踪的方块和永久保存的方块"""
        # 发布跟踪中的方块
        for track_id, track_info in self.tracked_objects.items():
            if track_info['score'] < self.min_track_score:
                continue

            # 发布中心点
            point_msg = PointStamped()
            point_msg.header.stamp = current_time
            point_msg.header.frame_id = self.sensor_frame
            point_msg.point.x = track_info['position'][0]
            point_msg.point.y = track_info['position'][1]
            point_msg.point.z = track_info['position'][2]

            try:
                # 转换到map坐标系
                point_map = self.tf_buffer.transform(point_msg, self.map_frame, rospy.Duration(1.0))
                self.publish_marker(point_map.point, track_id, track_info['score'])
                self.center_pub.publish(point_map)
            except TransformException as e:
                rospy.logwarn_throttle(1, f"坐标转换失败: {e}")

        # 发布永久保存的方块
        for block_id, block_info in self.permanent_blocks.items():
            point = PointStamped()
            point.header.stamp = current_time
            point.header.frame_id = self.map_frame
            
            # 使用稳定位置（如果可用）
            if 'stable_position' in block_info and block_info['update_count'] >= self.min_observations:
                map_pos = block_info['stable_position']
            else:
                map_pos = block_info['map_position']
                
            point.point.x = map_pos[0]
            point.point.y = map_pos[1]
            point.point.z = map_pos[2]
            
            # 发布永久方块的marker
            marker = Marker()
            marker.header.frame_id = self.map_frame
            marker.header.stamp = current_time
            marker.ns = "permanent_block"
            marker.id = block_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position = point.point
            
            angle = 20.0 * np.pi / 180.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = np.sin(angle/2)
            marker.pose.orientation.w = np.cos(angle/2)
            
            marker.scale.x = self.block_size
            marker.scale.y = self.block_size
            marker.scale.z = self.block_size
            
            # 根据置信度调整颜色
            confidence = block_info.get('confidence', 0.5)
            if confidence > self.confidence_threshold:
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0  # 高置信度为蓝色
            else:
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 1.0  # 低置信度为紫色
            
            marker.color.a = min(0.8, max(0.3, confidence))  # 透明度随置信度变化
            marker.lifetime = rospy.Duration(0)  # 永不过期
            
            self.marker_pub.publish(marker)
            self.center_pub.publish(point)

    def publish_marker(self, point, track_id, score):
        try:
            marker = Marker()
            marker.header.frame_id = self.map_frame
            marker.header.stamp = rospy.Time.now()
            marker.ns = "block"
            marker.id = track_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position = point
            
            angle = 20.0 * np.pi / 180.0  # 转换为弧度
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = np.sin(angle/2)
            marker.pose.orientation.w = np.cos(angle/2)
            
            marker.scale.x = self.block_size
            marker.scale.y = self.block_size
            marker.scale.z = self.block_size
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = min(0.7, max(0.3, score))
            marker.lifetime = rospy.Duration(0.2)
            self.marker_pub.publish(marker)
        except Exception as e:
            rospy.logerr(f"发布marker时出错: {e}")

if __name__ == "__main__":
    try:
        detector = CubeDetector3D()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
