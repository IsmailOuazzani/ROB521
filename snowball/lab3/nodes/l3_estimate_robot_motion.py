#!/usr/bin/env python3
from __future__ import division, print_function
import time

import numpy as np
import rospy
import tf_conversions
import tf2_ros
import rosbag
import rospkg
from math import cos, sin
import matplotlib.pyplot as plt
import copy

# msgs
from turtlebot3_msgs.msg import SensorState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Twist, TransformStamped, Transform, Quaternion
from std_msgs.msg import Empty

from utils import convert_pose_to_tf, euler_from_ros_quat, ros_quat_from_euler


ENC_TICKS = 4096
RAD_PER_TICK = 2 * np.pi / ENC_TICKS
WHEEL_RADIUS = 0.03255
BASELINE = 0.14645
INT32_MAX = 2**31


class WheelOdom:
    def __init__(self):
        # publishers, subscribers, tf broadcaster
        self.sensor_state_sub = rospy.Subscriber('/sensor_state', SensorState, self.sensor_state_cb, queue_size=1)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_cb, queue_size=1)
        self.wheel_odom_pub = rospy.Publisher('/wheel_odom', Odometry, queue_size=1)
        self.cmd_sub = rospy.Subscriber('/cmd_vel', Twist, self.cmd_cb, queue_size=1)
        self.tf_br = tf2_ros.TransformBroadcaster()

        # attributes
        self.odom = Odometry()
        self.odom.pose.pose.position.x = 1e10
        self.wheel_odom = Odometry()
        self.wheel_odom.header.frame_id = 'odom'
        self.wheel_odom.child_frame_id = 'wo_base_link'
        self.wheel_odom_tf = TransformStamped()
        self.wheel_odom_tf.header.frame_id = 'odom'
        self.wheel_odom_tf.child_frame_id = 'wo_base_link'
        self.pose = Pose()
        self.pose.orientation.w = 1.0
        self.twist = Twist()
        self.last_enc_l = None
        self.last_enc_r = None
        self.last_time = None
        self.true_x = []
        self.true_y = []
        self.true_time = []
        self.true_theta = []
        self.estimated_x = []
        self.estimated_y = []
        self.estimated_theta = []
        self.commands_dict = {"v_true":[],
                              "w_true":[],
                              "v_est":[],
                              "w_est":[]}
        self.commands = (0,0)

        # estimation parameters
        self.mu = np.array([0, 0, 0]).reshape(-1, 1) # assume zero origin
        self.cov = np.diag([1e-3, 1e-3, 1e-3])

        # rosbag
        rospack = rospkg.RosPack()
        path = rospack.get_path("rob521_lab3")
        self.bag = rosbag.Bag(path+"/motion_estimate.bag", 'w')

        # reset current odometry to allow comparison with this node
        reset_pub = rospy.Publisher('/reset', Empty, queue_size=1, latch=True)
        reset_pub.publish(Empty())
        while not rospy.is_shutdown() and (self.odom.pose.pose.position.x >= 1e-3 or self.odom.pose.pose.position.y >= 1e-3 or
               self.odom.pose.pose.orientation.z >= 1e-2):
            time.sleep(0.2)  # allow reset_pub to be ready to publish
        print('Robot o  dometry reset.')

        rospy.spin()
        self.bag.close()
        print("saving bag")
        
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))

        # True vs Estimated Path
        axes[0, 0].plot(self.true_x, self.true_y, label='True')
        axes[0, 0].plot(self.estimated_x, self.estimated_y, label='Estimated')
        axes[0, 0].set_xlabel("X")
        axes[0, 0].set_ylabel("Y")
        axes[0, 0].legend()
        axes[0, 0].set_title("True vs Estimated Path")

        # Time vs X
        axes[0, 1].plot(self.true_time, self.true_x, label='True X')
        axes[0, 1].plot(self.true_time, self.estimated_x, label='Estimated X')
        axes[0, 1].set_xlabel("Time")
        axes[0, 1].set_ylabel("X")
        axes[0, 1].legend()
        axes[0, 1].set_title("Time vs X")

        # Time vs Y
        axes[1, 0].plot(self.true_time, self.true_y, label='True Y')
        axes[1, 0].plot(self.true_time, self.estimated_y, label='Estimated Y')
        axes[1, 0].set_xlabel("Time")
        axes[1, 0].set_ylabel("Y")
        axes[1, 0].legend()
        axes[1, 0].set_title("Time vs Y")

        # Time vs Theta
        axes[1, 1].plot(self.true_time, self.true_theta, label='True Theta')
        axes[1, 1].plot(self.true_time, self.estimated_theta, label='Estimated Theta')
        axes[1, 1].set_xlabel("Time")
        axes[1, 1].set_ylabel("Theta")
        axes[1, 1].legend()
        axes[1, 1].set_title("Time vs Theta")

        # Time vs Linear Velocity (v)
        axes[2, 0].plot(self.true_time, self.commands_dict["v_true"], label="True v")
        # axes[2, 0].plot(self.true_time, self.commands_dict["v_est"], label="Estimated v")
        axes[2, 0].set_xlabel("Time")
        axes[2, 0].set_ylabel("Linear Velocity (v)")
        axes[2, 0].legend()
        axes[2, 0].set_title("Time vs Linear Velocity (v)")

        # Time vs Angular Velocity (w)
        axes[2, 1].plot(self.true_time, self.commands_dict["w_true"], label="True w")
        # axes[2, 1].plot(self.true_time, self.commands_dict["w_est"], label="Estimated w")
        axes[2, 1].set_xlabel("Time")
        axes[2, 1].set_ylabel("Angular Velocity (w)")
        axes[2, 1].legend()
        axes[2, 1].set_title("Time vs Angular Velocity (w)")

        plt.tight_layout()
        plt.show()


    def cmd_cb(self, cmd_msg):
        self.commands = (cmd_msg.linear.x, cmd_msg.angular.z)

    def B(self, theta_bar):
        return np.array([[cos(theta_bar), 0], [sin(theta_bar), 0], [0, 1]]) @ \
        np.array([[WHEEL_RADIUS /2, WHEEL_RADIUS /2 ], [WHEEL_RADIUS/(2*BASELINE), -WHEEL_RADIUS/(2*BASELINE)]])

    def safeDelPhi(self, a, b):
        #Need to check if the encoder storage variable has overflowed
        diff = np.int64(b) - np.int64(a)
        if diff < -np.int64(INT32_MAX): #Overflowed
            delPhi = (INT32_MAX - 1 - a) + (INT32_MAX + b) + 1
        elif diff > np.int64(INT32_MAX) - 1: #Underflowed
            delPhi = (INT32_MAX + a) + (INT32_MAX - 1 - b) + 1
        else:
            delPhi = b - a  
        return delPhi

    def sensor_state_cb(self, sensor_state_msg):
        # Callback for whenever a new encoder message is published
        # set initial encoder pose
        if self.last_enc_l is None:
            self.last_enc_l = sensor_state_msg.left_encoder #int32
            self.last_enc_r = sensor_state_msg.right_encoder #int32
            self.last_time = sensor_state_msg.header.stamp
            self.first_point = True
            self.start_time = sensor_state_msg.header.stamp
        else:
            if self.first_point:
                self.first_point = False
                self.start_odom_pose = self.odom
                self.mu = np.array([self.odom.pose.pose.position.x , self.odom.pose.pose.position.y, euler_from_ros_quat(self.odom.pose.pose.orientation)[2]]).reshape(-1, 1)

            self.topic_time = sensor_state_msg.header.stamp 
            # update calculated pose and twist with new data
            le = sensor_state_msg.left_encoder #int32
            re = sensor_state_msg.right_encoder #int32
            if self.topic_time - self.last_time > rospy.Duration(0.5):
                print("Time difference too large: %2.3f" % (self.topic_time - self.last_time).to_sec())

            # make sure to account for tick overflow
            del_left_encoder = self.safeDelPhi(self.last_enc_l, le)
            del_right_encoder = self.safeDelPhi(self.last_enc_r, re)
            # Update your odom estimates with the latest encoder measurements and populate the relevant area
            # of self.pose and self.twist with estimated position, heading and velocity
            theta_bar = self.mu[2]
            delta_phi = np.array([RAD_PER_TICK * (del_right_encoder), RAD_PER_TICK * (del_left_encoder)]).reshape(-1, 1)
            update = self.B(theta_bar) @ delta_phi
            estimated_inputs = update / (self.topic_time - self.last_time).to_sec()
            self.commands_dict["v_true"].append(self.commands[0])
            self.commands_dict["w_true"].append(self.commands[1])
            self.commands_dict["v_est"].append(float(estimated_inputs[0]))
            self.commands_dict["w_est"].append(float(estimated_inputs[1]))

            new_mu = self.mu + update
            deep_mu = copy.deepcopy(new_mu)
            # create deepcopy

            self.mu = deep_mu
            est_theta = float(self.mu[2])
            if est_theta > np.pi:
                est_theta -= 2*np.pi
            elif est_theta < -np.pi:
                est_theta += 2*np.pi
            self.estimated_theta.append(est_theta)

            mu_dot = (new_mu - self.mu)/(self.topic_time - self.last_time).to_sec()

            self.pose.position.x = self.mu[0]
            self.pose.position.y = self.mu[1]
            self.pose.orientation = ros_quat_from_euler([0, 0, new_mu[2]])

            self.twist.linear.x = mu_dot[0].item()
            self.twist.linear.y = mu_dot[1].item()
            self.twist.angular.z = mu_dot[2].item()

            # publish the updates as a topic and in the tf tree
            current_time = rospy.Time.now()
            self.wheel_odom_tf.header.stamp = current_time
            self.wheel_odom_tf.transform = convert_pose_to_tf(self.pose)
            self.tf_br.sendTransform(self.wheel_odom_tf)

            self.wheel_odom.header.stamp = current_time
            self.wheel_odom.pose.pose = self.pose
            self.wheel_odom.twist.twist = self.twist
            self.wheel_odom_pub.publish(self.wheel_odom)

            self.bag.write('odom_est', self.wheel_odom)
            self.bag.write('odom_onboard', self.odom)

            # modify the previous 
            self.last_enc_l = le
            self.last_enc_r = re
            self.last_time = self.topic_time

            # for testing against actual odom
            mu = self.mu
            print("Wheel Odom: x: %2.3f, y: %2.3f, t: %2.3f" % (
                self.pose.position.x, self.pose.position.y, mu[2].item()
            ))
            self.true_x.append(self.odom.pose.pose.position.x)
            self.true_y.append(self.odom.pose.pose.position.y)
            self.true_theta.append(euler_from_ros_quat(self.odom.pose.pose.orientation)[2])
            self.estimated_x.append(self.pose.position.x)
            self.estimated_y.append(self.pose.position.y)
            self.true_time.append((self.odom.header.stamp - self.start_time).to_sec())
            print("Turtlebot3 Odom: x: %2.3f, y: %2.3f, t: %2.3f" % (
                self.odom.pose.pose.position.x- self.start_odom_pose.pose.pose.position.x,
                self.odom.pose.pose.position.y- self.start_odom_pose.pose.pose.position.y,
                euler_from_ros_quat(self.odom.pose.pose.orientation)[2]
            ))

    def odom_cb(self, odom_msg):
        # get odom from turtlebot3 packages
        self.odom = odom_msg


    def plot(self, bag):
        data = {"odom_est":{"time":[], "data":[]}, 
                "odom_onboard":{'time':[], "data":[]}}
        for topic, msg, t in bag.read_messages(topics=['odom_est', 'odom_onboard']):
            print(msg)


if __name__ == '__main__':
    try:
        rospy.init_node('wheel_odometry')
        wheel_odom = WheelOdom()
    except rospy.ROSInterruptException:
        pass