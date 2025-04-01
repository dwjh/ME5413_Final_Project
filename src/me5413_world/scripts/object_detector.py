#!/usr/bin/env python3
import rospy
import cv2
import pytesseract
import numpy as np
import re
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from std_msgs.msg import String

class OCRNode:
    def __init__(self):
        rospy.init_node("ocr_node", anonymous=True)
        self.bridge = CvBridge()
        self.pub = rospy.Publisher("/ocr_result", String, queue_size=10)

        # 订阅相机数据
        rospy.Subscriber("/front/image_raw", Image, self.image_callback)
        rospy.Subscriber("/front/camera_info", CameraInfo, self.camera_info_callback)

        self.camera_matrix = None
        self.dist_coeffs = None
        rospy.loginfo("OCR Node Initialized. Waiting for camera info...")

    def camera_info_callback(self, msg):
        """获取相机内参"""
        self.camera_matrix = np.array(msg.K).reshape((3, 3))
        self.dist_coeffs = np.array(msg.D)

    def image_callback(self, msg):
        try:
            # 1. 读取ROS图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
            cv2.imshow("Original Image", cv_image)

            # 2. 去畸变（如果有相机参数）
            if self.camera_matrix is not None and self.dist_coeffs is not None:
                cv_image = cv2.undistort(cv_image, self.camera_matrix, self.dist_coeffs)
                cv2.imshow("Undistorted Image", cv_image)

            # 3. 预处理图像
            processed_image = self.preprocess_image(cv_image)
            
            if processed_image is not None:
                # OCR 识别
                raw_text = pytesseract.image_to_string(
                    processed_image,
                    config="--psm 10 --oem 3 -c tessedit_char_whitelist=123456789"
                )
                filtered_text = self.filter_single_digit(raw_text)

                # 直接打印原始识别结果和过滤后的结果
                print("\n" + "="*30)
                print("原始识别结果:", raw_text.strip())
                print("过滤后结果:", filtered_text)
                print("="*30 + "\n")

                if filtered_text:
                    self.pub.publish(filtered_text)

            # 5. 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                rospy.signal_shutdown("User exited.")

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def preprocess_image(self, image):
        """OCR前的图像预处理"""
        height, width = image.shape
        
        # 1. 边缘检测
        edges = cv2.Canny(image, 50, 150)
        
        # 2. 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 3. 筛选合适的矩形轮廓
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:  # 过滤太小的区域
                continue
            
            # 获取最小外接矩形
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # 计算矩形的宽高比
            width = rect[1][0]
            height = rect[1][1]
            ratio = max(width, height) / min(width, height)
            
            # 如果是近似正方形
            if 0.8 < ratio < 1.2:
                # 获取ROI，稍微缩小一点以去除边框
                x, y, w, h = cv2.boundingRect(cnt)
                # 向内缩进以去除边框
                margin = int(min(w, h) * 0.15)  # 缩进15%
                roi = image[y+margin:y+h-margin, x+margin:x+w-margin]
                
                # 显示ROI
                cv2.imshow("ROI", roi)
                
                # 增强对比度
                roi = cv2.convertScaleAbs(roi, alpha=1.5, beta=0)
                
                # 自适应二值化
                binary = cv2.adaptiveThreshold(
                    roi, 
                    255, 
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 
                    21,  # 增大块大小
                    10   # 增大常数
                )
                
                # 去噪
                kernel = np.ones((3,3), np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                
                # 调整大小，保持比例
                target_size = 128
                ratio = float(target_size) / max(binary.shape)
                new_size = tuple([int(dim * ratio) for dim in binary.shape])
                resized = cv2.resize(binary, (new_size[1], new_size[0]))
                
                # 添加白色边距
                final = np.full((target_size, target_size), 255, dtype=np.uint8)
                x_offset = (target_size - new_size[1]) // 2
                y_offset = (target_size - new_size[0]) // 2
                final[y_offset:y_offset+new_size[0], x_offset:x_offset+new_size[1]] = resized
                
                cv2.imshow("Processed", final)
                
                print("\n" + "="*30)
                print("检测到方框，处理中...")
                print("="*30 + "\n")
                
                return final
                
        return None

    def filter_single_digit(self, text):
        """过滤OCR结果，确保只有一个1-9的数字"""
        match = re.search(r'[1-9]', text)  # 查找 1-9 的单个字符
        return match.group(0) if match else None  # 返回匹配结果，如果没有则返回None

if __name__ == "__main__":
    try:
        OCRNode()
        rospy.spin()
        cv2.destroyAllWindows()
    except rospy.ROSInterruptException:
        pass


