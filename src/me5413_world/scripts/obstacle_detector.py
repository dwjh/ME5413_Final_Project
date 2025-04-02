#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import pytesseract
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener, TransformException
import tf2_geometry_msgs

class BlockDetector:
    def __init__(self):
        rospy.init_node("block_digit_detector")
        self.bridge = CvBridge()

        self.camera_matrix = None
        self.rgb_image = None
        self.depth_image = None
        self.got_camera_info = False
        self.obstacle_cache = {}
        self.max_obstacles = 11

        self.min_height = 0.2
        self.max_height = 2.0
        self.min_depth = 0.5
        self.max_depth = 5

        self.min_contour_area = 1000
        self.max_contour_area = 50000

        self.debug = True
        self.debug_publisher = rospy.Publisher("/debug_image", Image, queue_size=1)

        self.caminfo_sub = rospy.Subscriber("/front/rgb/camera_info", CameraInfo, self.caminfo_callback)
        self.rgb_sub = rospy.Subscriber("/front/rgb/image_raw", Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber("/front/depth/image_raw", Image, self.depth_callback)

        self.marker_pub = rospy.Publisher("/obstacle_markers", MarkerArray, queue_size=10)
        self.position_pub = rospy.Publisher("/obstacle_positions", MarkerArray, queue_size=10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        self.camera_frame = "front_frame_optical"
        self.map_frame = "map"

        self.timer = rospy.Timer(rospy.Duration(0.1), self.publish_markers)
        rospy.loginfo("Block Digit Detector Initialized")

    def show_debug_image(self, name, img):
        if self.debug:
            if len(img.shape) == 2:
                img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_display = img.copy()
            scale = 0.5
            width = int(img_display.shape[1] * scale)
            height = int(img_display.shape[0] * scale)
            img_display = cv2.resize(img_display, (width, height))
            cv2.imshow(name, img_display)
            cv2.waitKey(1)
            try:
                debug_msg = self.bridge.cv2_to_imgmsg(img_display, "bgr8")
                self.debug_publisher.publish(debug_msg)
            except Exception as e:
                rospy.logwarn(f"Debug image conversion failed: {e}")

    def caminfo_callback(self, msg):
        if not self.got_camera_info:
            self.camera_matrix = np.array(msg.K).reshape((3, 3))
            self.got_camera_info = True
            self.caminfo_sub.unregister()
            rospy.loginfo("Camera intrinsics received")

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.try_process_both_streams()

    def rgb_callback(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.rgb_stamp = msg.header.stamp
        self.try_process_both_streams()

    def try_process_both_streams(self):
        if not self.got_camera_info or self.rgb_image is None or self.depth_image is None:
            return
        self.process_images()

    def get_3d_point(self, u, v, depth):
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        return x, y, z

    def process_images(self):
        rgb_image = self.rgb_image.copy()
        
        # 1. 只保留合理深度范围内的点
        depth_mask = (self.depth_image > self.min_depth) & (self.depth_image < self.max_depth)
        depth_mask = depth_mask.astype(np.uint8) * 255
        self.show_debug_image("1. Initial Depth Mask", depth_mask)
        
        # 2. 寻找轮廓并切割地面
        contours, _ = cv2.findContours(depth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            margin = 10  # 增大margin以便更好地检查边缘
            if y + h_box + margin >= depth_mask.shape[0]:  # 如果碰到图像底部就跳过
                continue
            
            # 检查上边缘
            top = depth_mask[y:y+margin, x:x+w_box]
            # 检查左边缘
            left = depth_mask[y:y+h_box, x:x+margin]
            # 检查右边缘
            right = depth_mask[y:y+h_box, x+w_box-margin:x+w_box]
            
            # 如果三个边缘都比较暗（有很多0），说明这可能是一个立方体
            if (np.count_nonzero(top) < margin * w_box * 0.3 and  # 调大阈值到30%
                np.count_nonzero(left) < margin * h_box * 0.3 and 
                np.count_nonzero(right) < margin * h_box * 0.3):
                # 切掉这个物体下方的所有区域
                depth_mask[y+h_box:, x:x+w_box] = 0
        
        self.show_debug_image("2. After Ground Removal", depth_mask)
        
        # 3. 形态学操作来清理噪声
        kernel = np.ones((5,5), np.uint8)
        cleaned_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        
        self.show_debug_image("3. Final Mask", cleaned_mask)
        
        # 4. 在原图上显示结果
        debug_img = rgb_image.copy()
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(debug_img, contours, -1, (0,255,0), 2)
        self.show_debug_image("4. Result on RGB", debug_img)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area or area > self.max_contour_area:
                continue

            x, y, w_box, h_box = cv2.boundingRect(contour)
            roi_gray = rgb_image[y:y+h_box, x:x+w_box]
            roi_gray = cv2.equalizeHist(roi_gray)
            text = pytesseract.image_to_string(roi_gray, config='--psm 10 -c tessedit_char_whitelist=0123456789').strip()
            rospy.loginfo(f"OCR result: {text}")

            if not text.isdigit():
                continue

            digit = int(text)
            if digit >= self.max_obstacles:
                continue

            center_u = x + w_box // 2
            center_v = y + h_box // 2
            depth = np.median(self.depth_image[y:y+h_box, x:x+w_box])
            if not np.isfinite(depth) or depth <= 0:
                continue

            x_cam, y_cam, z_cam = self.get_3d_point(center_u, center_v, depth)
            if not (self.min_height < y_cam < self.max_height):
                continue

            point_camera = PointStamped()
            point_camera.header.stamp = self.rgb_stamp
            point_camera.header.frame_id = self.camera_frame
            point_camera.point.x = x_cam
            point_camera.point.y = y_cam
            point_camera.point.z = z_cam

            try:
                point_map = self.tf_buffer.transform(point_camera, self.map_frame, rospy.Duration(1.0))
                new_position = (point_map.point.x, point_map.point.y, point_map.point.z)
                if digit in self.obstacle_cache:
                    old = self.obstacle_cache[digit]
                    old_pos = (old.point.x, old.point.y, old.point.z)
                    dist = np.linalg.norm(np.array(new_position) - np.array(old_pos))
                    if dist < 0.1:
                        continue
                self.obstacle_cache[digit] = point_map
                rospy.loginfo(f"Obstacle {digit} at map: x={new_position[0]:.2f}, y={new_position[1]:.2f}, z={new_position[2]:.2f}")
            except (TransformException, rospy.ROSException) as e:
                rospy.logwarn(f"TF transform failed: {e}")

    def publish_markers(self, event):
        marker_array = MarkerArray()
        position_array = MarkerArray()
        for digit, point in self.obstacle_cache.items():
            text_marker = Marker()
            text_marker.header.frame_id = self.map_frame
            text_marker.header.stamp = rospy.Time.now()
            text_marker.ns = "obstacles_text"
            text_marker.id = digit
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position = point.point
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.4
            text_marker.color.r = 1.0
            text_marker.color.g = 0.6
            text_marker.color.b = 0.2
            text_marker.color.a = 1.0
            text_marker.text = str(digit)
            marker_array.markers.append(text_marker)

            position_marker = Marker()
            position_marker.header.frame_id = self.map_frame
            position_marker.header.stamp = rospy.Time.now()
            position_marker.ns = "obstacles_position"
            position_marker.id = digit
            position_marker.type = Marker.SPHERE
            position_marker.action = Marker.ADD
            position_marker.pose.position = point.point
            position_marker.pose.orientation.w = 1.0
            position_marker.scale.x = 0.2
            position_marker.scale.y = 0.2
            position_marker.scale.z = 0.2
            position_marker.color.r = 0.2
            position_marker.color.g = 0.8
            position_marker.color.b = 0.2
            position_marker.color.a = 0.8
            position_array.markers.append(position_marker)

        self.marker_pub.publish(marker_array)
        self.position_pub.publish(position_array)

if __name__ == "__main__":
    try:
        BlockDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
