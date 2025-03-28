#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PointStamped
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
from tf.transformations import quaternion_matrix
from tf2_ros import TransformListener, Buffer, TransformException
import tf2_geometry_msgs

class BridgeDetector:
    def __init__(self):
        rospy.init_node('bridge_detector')
        
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/front/image_raw", Image, self.image_callback)
        self.lidar_sub = rospy.Subscriber("/mid/points", PointCloud2, self.lidar_callback)
        self.debug_pub = rospy.Publisher("/bridge_detection/debug_image", Image, queue_size=1)
        self.coord_pub = rospy.Publisher("/coordinate", PointStamped, queue_size=10)

        # TF2初始化
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        # 颜色阈值 (HSV)
        self.lower_orange = np.array([5, 100, 100])
        self.upper_orange = np.array([15, 255, 255])

        # 相机参数
        self.camera_matrix = np.array([
            [554.25, 0,     320.5],
            [0,     554.25, 256.5],
            [0,     0,      1]
        ])

        # 坐标变换参数
        self.translation = np.array([0.242, 0.000, -0.105])
        self.quaternion = np.array([-0.500, 0.500, -0.500, 0.500])
        self.rotation_matrix = quaternion_matrix(self.quaternion)[:3, :3]
        
        self.camera_frame = "front_camera_optical"
        self.odom_frame = "odom"

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            timestamp = msg.header.stamp
            
            best_bbox = self.detect_orange_object(cv_image)
            if best_bbox:
                x, y, w, h = best_bbox
                center_x, center_y = x + w//2, y + h//2

                if hasattr(self, 'lidar_data') and self.lidar_data:
                    lidar_coords = self.get_lidar_depth(center_x, center_y)
                    if lidar_coords:
                        camera_coords = self.lidar_to_camera(*lidar_coords)
                        odom_coords = self.camera_to_odom(camera_coords, timestamp)
                        
                        if odom_coords:
                            coord_msg = PointStamped()
                            coord_msg.header.stamp = timestamp
                            coord_msg.header.frame_id = self.odom_frame
                            coord_msg.point.x = odom_coords[0]
                            coord_msg.point.y = odom_coords[1]
                            coord_msg.point.z = odom_coords[2]
                            self.coord_pub.publish(coord_msg)

            cv2.imshow("Detection", cv_image)
            cv2.waitKey(1)
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
            
        except Exception as e:
            rospy.logerr(f"Image processing error: {str(e)}")

    def lidar_callback(self, msg):
        self.lidar_data = msg

    def detect_orange_object(self, cv_image):
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_orange, self.upper_orange)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_bbox = None
        min_y = float('inf')

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 200:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if x < 200:
                continue

            if y < min_y:
                min_y = y
                best_bbox = (x, y, w, h)

        if best_bbox:
            x, y, w, h = best_bbox
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return best_bbox

    def get_lidar_depth(self, pixel_x, pixel_y):
        if not hasattr(self, 'lidar_data') or self.lidar_data is None:
            return None

        gen = pc2.read_points(self.lidar_data, field_names=("x", "y", "z"), skip_nans=True)
        closest_point = None
        min_distance = float('inf')
        
        for p in gen:
            lidar_x, lidar_y, lidar_z = p
            camera_point = self.lidar_to_camera(lidar_x, lidar_y, lidar_z)
            camera_x, camera_y, camera_z = camera_point
            
            if camera_z <= 0:
                continue

            u = int(self.camera_matrix[0, 0] * camera_x / camera_z + self.camera_matrix[0, 2])
            v = int(self.camera_matrix[1, 1] * camera_y / camera_z + self.camera_matrix[1, 2])

            if abs(u - pixel_x) < 8 and abs(v - pixel_y) < 8:
                distance = np.sqrt(lidar_x**2 + lidar_y**2 + lidar_z**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_point = (lidar_x, lidar_y, lidar_z)
        
        return closest_point

    def lidar_to_camera(self, lidar_x, lidar_y, lidar_z):
        lidar_point = np.array([lidar_x, lidar_y, lidar_z])
        rotated_point = np.dot(self.rotation_matrix, lidar_point)
        camera_point = rotated_point + self.translation
        return camera_point

    def camera_to_odom(self, camera_point, timestamp):
        try:
            point_camera = PointStamped()
            point_camera.header.stamp = timestamp
            point_camera.header.frame_id = self.camera_frame
            point_camera.point.x = float(camera_point[0])
            point_camera.point.y = float(camera_point[1])
            point_camera.point.z = float(camera_point[2])
            
            point_odom = self.tf_buffer.transform(
                point_camera, 
                self.odom_frame,
                rospy.Duration(1.0)
            )
            return [point_odom.point.x, point_odom.point.y, point_odom.point.z]
            
        except (TransformException, rospy.ROSException) as e:
            rospy.logwarn_throttle(1.0, f"TF转换失败: {str(e)}")
            return None

if __name__ == "__main__":
    try:
        detector = BridgeDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass