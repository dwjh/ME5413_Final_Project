#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped, Point
import sensor_msgs.point_cloud2 as pc2
from tf2_ros import TransformListener, Buffer, TransformException
import tf2_geometry_msgs

class PointCloudObstacleDetector:
    def __init__(self):
        rospy.init_node('pointcloud_obstacle_detector')
        
        # 订阅点云数据
        self.lidar_sub = rospy.Subscriber("/mid/points", PointCloud2, self.lidar_callback)
        # 发布障碍物标记
        self.marker_pub = rospy.Publisher('/detected_3d_obstacles', MarkerArray, queue_size=10)
        # 发布障碍物中心点
        self.center_pub = rospy.Publisher("/obstacle_centers", MarkerArray, queue_size=10)
        # 发布清除区域
        self.clearing_pub = rospy.Publisher("/clearing_areas", MarkerArray, queue_size=10)

        # TF2初始化
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)
        
        # 设置坐标系
        self.lidar_frame = "velodyne"  # 激光雷达坐标系
        self.base_frame = "base_link"  # 机器人基座坐标系
        self.odom_frame = "odom"      # 里程计坐标系
        self.map_frame = "map"        # 地图坐标系
        
        # 参数设置
        self.search_radius = 5.0  # 搜索半径（米），与obstacle_range一致
        self.cluster_tolerance = 0.3  # 聚类容差，调小以提高精度
        self.min_cluster_size = 5  # 最小聚类点数，调小以检测更小的障碍物
        self.obstacles = {}  # 存储检测到的障碍物
        self.next_obstacle_id = 0
        self.obstacle_timeout = 0.5  # 障碍物超时时间（秒）
        self.min_height = 0.0  # 最小障碍物高度，与costmap配置一致
        self.max_height = 0.8  # 最大障碍物高度，与costmap配置一致
        
        rospy.loginfo("3D点云障碍物检测器已启动")

    def lidar_callback(self, msg):
        try:
            # 读取点云数据并过滤
            points_list = []
            gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            
            # 过滤距离和高度范围内的点
            for p in gen:
                x, y, z = p
                distance = np.sqrt(x**2 + y**2)  # 只考虑水平距离
                if distance <= self.search_radius and self.min_height <= z <= self.max_height:
                    points_list.append((x, y, z))
            
            if not points_list:
                return
                
            # 转换点云到map坐标系
            points_map = self.transform_points_to_map(points_list, msg.header.stamp)
            if points_map is None:
                return
                
            # 获取传感器在map坐标系下的位置
            sensor_position = self.get_sensor_position(msg.header.stamp)
            if sensor_position is None:
                return
                
            # 简单聚类处理
            clusters = self.cluster_points(points_map)
            
            # 更新和发布障碍物
            self.update_obstacles(clusters, msg.header.stamp)
            
            # 发布清除区域
            self.publish_clearing_areas(sensor_position, points_map, msg.header.stamp)
            
        except Exception as e:
            rospy.logerr(f"处理点云数据错误: {str(e)}")

    def transform_points_to_map(self, points, timestamp):
        try:
            latest_time = rospy.Time(0)  # 使用0表示最新的可用转换
            
            # 直接获取从velodyne到map的转换
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.lidar_frame,
                latest_time,
                rospy.Duration(3.0)  # 增加等待时间以确保转换可用
            )
            
            points_map = []
            for point in points:
                # 创建点的消息
                point_stamped = PointStamped()
                point_stamped.header.frame_id = self.lidar_frame
                point_stamped.header.stamp = latest_time
                point_stamped.point.x = point[0]
                point_stamped.point.y = point[1]
                point_stamped.point.z = point[2]
                
                try:
                    # 直接转换到map坐标系
                    point_map = self.tf_buffer.transform(point_stamped, self.map_frame)
                    points_map.append([point_map.point.x, point_map.point.y, point_map.point.z])
                except (TransformException, rospy.ROSException) as e:
                    continue  # 跳过转换失败的点
                
            return np.array(points_map) if points_map else None
            
        except (TransformException, rospy.ROSException) as e:
            rospy.logwarn_throttle(1.0, f"坐标转换失败: {str(e)}")
            return None

    def get_sensor_position(self, timestamp):
        """获取传感器在map坐标系下的位置"""
        try:
            # 直接获取从velodyne到map的转换
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.lidar_frame,
                rospy.Time(0),
                rospy.Duration(3.0)  # 增加等待时间以确保转换可用
            )
            
            # 创建一个点来表示传感器位置
            point = PointStamped()
            point.header.frame_id = self.lidar_frame
            point.header.stamp = rospy.Time(0)
            point.point.x = 0
            point.point.y = 0
            point.point.z = 0
            
            # 直接转换到map坐标系
            point_map = self.tf_buffer.transform(point, self.map_frame)
            
            return [
                point_map.point.x,
                point_map.point.y,
                point_map.point.z
            ]
        except (TransformException, rospy.ROSException) as e:
            rospy.logwarn_throttle(1.0, f"获取传感器位置失败: {str(e)}")
            return None

    def cluster_points(self, points):
        clusters = []
        processed = set()
        
        for i, point in enumerate(points):
            if i in processed:
                continue
                
            cluster = []
            queue = [i]
            
            while queue:
                idx = queue.pop(0)
                if idx not in processed:
                    processed.add(idx)
                    cluster.append(points[idx])
                    
                    # 查找邻近点
                    for j, other_point in enumerate(points):
                        if j not in processed and j not in queue:
                            if np.linalg.norm(points[idx] - other_point) < self.cluster_tolerance:
                                queue.append(j)
            
            if len(cluster) >= self.min_cluster_size:
                clusters.append(np.array(cluster))
        
        return clusters

    def update_obstacles(self, clusters, timestamp):
        current_time = rospy.Time.now().to_sec()
        
        # 清理过期的障碍物
        self.obstacles = {k: v for k, v in self.obstacles.items() 
                         if (current_time - v['last_seen'] < self.obstacle_timeout)}
        
        # 更新障碍物信息
        marker_array = MarkerArray()
        center_marker_array = MarkerArray()
        
        for cluster in clusters:
            center = np.mean(cluster, axis=0)
            size = np.max(cluster, axis=0) - np.min(cluster, axis=0)
            
            # 创建或更新障碍物
            obstacle_id = self.next_obstacle_id
            self.obstacles[obstacle_id] = {
                'center': center,
                'size': size,
                'last_seen': current_time
            }
            self.next_obstacle_id += 1
            
            # 创建障碍物标记
            marker = self.create_obstacle_marker(obstacle_id, center, size)
            marker_array.markers.append(marker)
            
            # 创建中心点标记
            center_marker = self.create_center_marker(obstacle_id, center)
            center_marker_array.markers.append(center_marker)
        
        # 发布标记
        if marker_array.markers:
            self.marker_pub.publish(marker_array)
            self.center_pub.publish(center_marker_array)

    def create_obstacle_marker(self, obstacle_id, center, size):
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "obstacles"
        marker.id = obstacle_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        marker.pose.position.x = center[0]
        marker.pose.position.y = center[1]
        marker.pose.position.z = center[2]
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = max(size[0], 0.1)
        marker.scale.y = max(size[1], 0.1)
        marker.scale.z = max(size[2], 0.1)
        
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.6
        
        return marker

    def create_center_marker(self, obstacle_id, center):
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "centers"
        marker.id = obstacle_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = center[0]
        marker.pose.position.y = center[1]
        marker.pose.position.z = center[2]
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        return marker

    def publish_clearing_areas(self, sensor_position, points_map, timestamp):
        """发布清除区域的可视化标记"""
        marker_array = MarkerArray()
        
        # 为每个点创建一个射线标记
        for i, point in enumerate(points_map):
            marker = Marker()
            marker.header.frame_id = self.map_frame
            marker.header.stamp = timestamp
            marker.ns = "clearing_rays"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            
            # 添加射线的起点（传感器位置）和终点（障碍物点）
            start = Point()
            start.x = sensor_position[0]
            start.y = sensor_position[1]
            start.z = sensor_position[2]
            
            end = Point()
            end.x = point[0]
            end.y = point[1]
            end.z = point[2]
            
            marker.points = [start, end]
            
            # 设置射线的外观
            marker.scale.x = 0.01  # 线宽
            marker.color.a = 0.3  # 透明度
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            
            marker_array.markers.append(marker)
        
        if marker_array.markers:
            self.clearing_pub.publish(marker_array)

if __name__ == "__main__":
    try:
        detector = PointCloudObstacleDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass