#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Imu
from collections import deque
import numpy as np
import copy

class ImuBiasCorrector:
    def __init__(self):
        rospy.init_node('imu_bias_corrector', anonymous=True)

        # 原始 IMU 数据 topic（可通过参数设置）
        self.imu_topic = rospy.get_param('~imuTopic', '/imu/data')
        
        # 重力加速度（使用与LIO-SAM相同的值）
        self.gravity = rospy.get_param('~gravity', 9.805)

        # 发布校正后的 IMU 数据
        self.corrected_imu_pub = rospy.Publisher('/imu/data_corrected', Imu, queue_size=10)

        # 用于存储前 100 帧的偏置
        self.linear_acc_bias_queue = deque(maxlen=100)
        self.angular_vel_bias_queue = deque(maxlen=100)

        # 初始化偏置值
        self.linear_acc_bias = np.zeros(3)
        self.angular_vel_bias = np.zeros(3)

        # 是否完成初始校正
        self.bias_initialized = False

        # 订阅原始 IMU 数据
        rospy.Subscriber(self.imu_topic, Imu, self.imu_callback)
        rospy.loginfo("IMU Bias Corrector initialized, listening on %s", self.imu_topic)

    def imu_callback(self, msg):
        # 提取加速度和角速度
        lin_acc = np.array([msg.linear_acceleration.x,
                            msg.linear_acceleration.y,
                            msg.linear_acceleration.z])
        ang_vel = np.array([msg.angular_velocity.x,
                            msg.angular_velocity.y,
                            msg.angular_velocity.z])

        # 收集初始 100 帧用于计算偏置
        if not self.bias_initialized:
            self.linear_acc_bias_queue.append(lin_acc)
            self.angular_vel_bias_queue.append(ang_vel)

            if len(self.linear_acc_bias_queue) == 100:
                # 计算X和Y轴的偏置（应该接近0）
                self.linear_acc_bias[0] = np.mean(np.array(self.linear_acc_bias_queue)[:, 0])
                self.linear_acc_bias[1] = np.mean(np.array(self.linear_acc_bias_queue)[:, 1])
                
                # 计算Z轴偏置（需要考虑重力）
                # 假设Z轴向上，静止时Z值应该是-g（重力加速度）
                z_readings = np.array(self.linear_acc_bias_queue)[:, 2]
                z_mean = np.mean(z_readings)
                self.linear_acc_bias[2] = z_mean + self.gravity  # 加上重力是因为我们希望校正后的值为-g
                
                # 计算角速度偏置（静止时应该为0）
                self.angular_vel_bias = np.mean(np.array(self.angular_vel_bias_queue), axis=0)
                
                self.bias_initialized = True
                rospy.loginfo("IMU Bias initialized:")
                rospy.loginfo("  Linear Acc Bias (excluding gravity): %s", self.linear_acc_bias)
                rospy.loginfo("  Angular Vel Bias: %s", self.angular_vel_bias)
                rospy.loginfo("  Gravity compensation: %.3f", self.gravity)
            return  # 暂不发布

        # 复制 msg，防止原始消息被更改
        corrected_msg = copy.deepcopy(msg)

        # 校正线性加速度
        corrected_msg.linear_acceleration.x -= self.linear_acc_bias[0]
        corrected_msg.linear_acceleration.y -= self.linear_acc_bias[1]
        corrected_msg.linear_acceleration.z -= self.linear_acc_bias[2]

        # 校正角速度
        corrected_msg.angular_velocity.x -= self.angular_vel_bias[0]
        corrected_msg.angular_velocity.y -= self.angular_vel_bias[1]
        corrected_msg.angular_velocity.z -= self.angular_vel_bias[2]

        # orientation 通常由 IMU 提供，暂不直接修改（除非你有初始角度估计）
        # 可选加：用静止时加速度方向推估 roll/pitch 补正 orientation（更复杂）

        # 发布校正后的 IMU 消息
        self.corrected_imu_pub.publish(corrected_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = ImuBiasCorrector()
        node.run()
    except rospy.ROSInterruptException:
        pass
