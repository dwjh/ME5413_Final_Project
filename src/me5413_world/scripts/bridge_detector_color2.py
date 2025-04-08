#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from tf2_ros import TransformListener, Buffer, TransformException
import tf2_geometry_msgs
import rosgraph_msgs.msg

class BridgeDetector:
    def __init__(self):
        rospy.init_node('bridge_detector')

        # 等待时钟服务
        try:
            rospy.loginfo("Waiting for /clock...")
            rospy.wait_for_message('/clock', rosgraph_msgs.msg.Clock, timeout=5.0)
            rospy.loginfo("Clock received!")
        except rospy.ROSException:
            rospy.logwarn("No clock message received. Continuing anyway...")

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.depth_image = None
        self.got_camera_info = False

        self.caminfo_sub = rospy.Subscriber("/front/rgb/camera_info", CameraInfo, self.caminfo_callback)

        rate = rospy.Rate(1)
        while not self.got_camera_info and not rospy.is_shutdown():
            rospy.loginfo("Waiting for camera intrinsics...")
            rate.sleep()

        self.image_sub = rospy.Subscriber("/front/rgb/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/front/depth/image_raw", Image, self.depth_callback)

        self.debug_pub = rospy.Publisher("/bridge_detection/debug_image", Image, queue_size=1)
        self.coord_pub = rospy.Publisher("/coordinate", PointStamped, queue_size=10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        self.lower_orange = np.array([5, 100, 100])
        self.upper_orange = np.array([15, 255, 255])

        self.camera_frame = "front_frame_optical"
        self.map_frame = "map"

        rospy.loginfo("Bridge detector initialized!")

    def caminfo_callback(self, msg):
        if not self.got_camera_info:
            self.camera_matrix = np.array(msg.K).reshape((3, 3))
            rospy.loginfo(f"[BridgeDetector] Camera intrinsics loaded:\n{self.camera_matrix}")
            self.got_camera_info = True
            self.caminfo_sub.unregister()

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def image_callback(self, msg):
        try:
            if not self.got_camera_info:
                return

            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            timestamp = msg.header.stamp

            best_bbox, depth_value = self.detect_orange_object(cv_image)
            if best_bbox and depth_value is not None:
                x, y, w, h = best_bbox
                cx, cy = x + w // 2, y + h // 2

                depth = depth_value  # 米
                fx = self.camera_matrix[0, 0]
                fy = self.camera_matrix[1, 1]
                cx_cam = self.camera_matrix[0, 2]
                cy_cam = self.camera_matrix[1, 2]

                camera_x = (cx - cx_cam) * depth / fx
                camera_y = (cy - cy_cam) * depth / fy
                camera_z = depth

                rospy.loginfo(f"Depth in meters: {depth:.3f}m")
                rospy.loginfo(f"Pixel coordinates: ({cx}, {cy})")
                rospy.loginfo(f"Camera intrinsics: fx={fx}, fy={fy}, cx={cx_cam}, cy={cy_cam}")

                camera_coords = np.array([camera_x, camera_y, camera_z])
                odom_coords = self.camera_to_odom(camera_coords)
                if odom_coords:
                    self.publish_coord(odom_coords)
                    cv2.putText(cv_image,
                        f"X:{odom_coords[0]:.2f} Y:{odom_coords[1]:.2f} Z:{odom_coords[2]:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("Detection", cv_image)
            cv2.waitKey(1)
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

        except Exception as e:
            rospy.logerr(f"Image processing error: {str(e)}")

    def detect_orange_object(self, cv_image):
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_orange, self.upper_orange)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        cv2.imshow("Mask", mask)

        if self.depth_image is not None:
            depth_display = cv2.normalize(self.depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.imshow("Depth", depth_display)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rospy.loginfo(f"Found {len(contours)} contours")
        valid_candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if y < 10 or y > cv_image.shape[0] - 10:
                continue
            if x < 10 or x > cv_image.shape[1] - 10:
                continue

            aspect_ratio = float(w) / h
            if not (1.5 <= aspect_ratio <= 4.0):
                continue

            rospy.loginfo(f"Contour - Area: {area}, Ratio: {aspect_ratio:.2f}, Position: ({x}, {y}), Size: {w}x{h}")

            if self.depth_image is not None:
                center_x = x + w // 2
                center_y = y + h // 2
                center_depth = self.depth_image[center_y, center_x]
                rospy.loginfo(f"Center point depth: {center_depth:.3f}m")

                if not np.isfinite(center_depth) or center_depth <= 0:
                    sample_points = []
                    step = 5
                    for sy in range(y, y + h, step):
                        for sx in range(x, x + w, step):
                            if sy >= self.depth_image.shape[0] or sx >= self.depth_image.shape[1]:
                                continue
                            depth = self.depth_image[sy, sx]
                            if np.isfinite(depth) and 0 < depth < 10000:
                                sample_points.append(depth)
                    if sample_points:
                        avg_depth = np.median(sample_points)
                        rospy.loginfo(f"Using median depth from {len(sample_points)} valid samples: {avg_depth:.3f}m")
                        valid_candidates.append((x, y, w, h, avg_depth, aspect_ratio, area))
                    else:
                        rospy.loginfo(f"No valid depth found in target area at ({x}, {y})")
                        cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(cv_image, "No depth", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                else:
                    rospy.loginfo(f"Using center point depth: {center_depth:.3f}m")
                    valid_candidates.append((x, y, w, h, center_depth, aspect_ratio, area))

        if valid_candidates:
            primary_candidates = [c for c in valid_candidates if c[6] > 1000 and 1.8 <= c[5] <= 3.5]
            best_candidate = max(primary_candidates or valid_candidates, key=lambda x: x[6])
            best_bbox = best_candidate[:4]
            depth_value = best_candidate[4]
            x, y, w, h = best_bbox
            info_text = f"w:{w}px h:{h}px ratio:{best_candidate[5]:.2f} area:{best_candidate[6]:.0f} depth:{depth_value:.2f}m"
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(cv_image, info_text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            return best_bbox, depth_value
        return None, None

    def camera_to_odom(self, camera_point):
        try:
            point_camera = PointStamped()
            point_camera.header.stamp = rospy.Time(0)
            point_camera.header.frame_id = self.camera_frame
            point_camera.point.x = float(camera_point[0])
            point_camera.point.y = float(camera_point[1])
            point_camera.point.z = float(camera_point[2])

            rospy.loginfo(f"Attempting to transform point from {self.camera_frame} to {self.map_frame}")
            rospy.loginfo(f"Camera point: x={camera_point[0]:.3f}, y={camera_point[1]:.3f}, z={camera_point[2]:.3f}")

            if not self.tf_buffer.can_transform(
                self.map_frame, self.camera_frame, rospy.Time(0), rospy.Duration(1.0)
            ):
                rospy.logwarn(f"Transform from {self.camera_frame} to {self.map_frame} not available yet")
                return None

            camera_pose = self.tf_buffer.lookup_transform(
                self.map_frame, self.camera_frame, rospy.Time(0)
            )
            camera_pos = camera_pose.transform.translation
            rospy.loginfo(f"Camera position in map: x={camera_pos.x:.3f}, y={camera_pos.y:.3f}, z={camera_pos.z:.3f}")

            point_map = self.tf_buffer.transform(point_camera, self.map_frame, rospy.Duration(1.0))

            dx = point_map.point.x - camera_pos.x
            dy = point_map.point.y - camera_pos.y
            dz = point_map.point.z - camera_pos.z
            distance = np.sqrt(dx**2 + dy**2 + dz**2)

            rospy.loginfo(f"Distance from camera to point: {distance:.3f}m")

            if distance < 0.5 or distance > 10.0:
                rospy.logwarn(f"Transformed point too close or too far from camera: {distance:.3f}m")
                return None

            rospy.loginfo(f"Successfully transformed point to map frame: x={point_map.point.x:.3f}, y={point_map.point.y:.3f}, z={point_map.point.z:.3f}")
            return [point_map.point.x, point_map.point.y, point_map.point.z]

        except (TransformException, rospy.ROSException) as e:
            rospy.logwarn_throttle(1.0, f"TF transform failed: {str(e)}")
            return None

    def publish_coord(self, coords):
        try:
            coord_msg = PointStamped()
            current_time = rospy.get_rostime()
            coord_msg.header.stamp = current_time
            coord_msg.header.frame_id = self.map_frame
            coord_msg.point.x = coords[0]
            coord_msg.point.y = coords[1]
            coord_msg.point.z = coords[2]
            rospy.loginfo(f"Publishing coordinate at time {current_time.to_sec():.3f}")
            rospy.loginfo(f"Frame: {self.map_frame}, Coords: x={coords[0]:.3f}, y={coords[1]:.3f}, z={coords[2]:.3f}")
            self.coord_pub.publish(coord_msg)
        except Exception as e:
            rospy.logerr(f"Error publishing coordinate: {str(e)}")

if __name__ == "__main__":
    try:
        detector = BridgeDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
