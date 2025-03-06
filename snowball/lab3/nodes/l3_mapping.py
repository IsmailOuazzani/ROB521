#!/usr/bin/env python3
from __future__ import division, print_function

import numpy as np
import rospy
import tf2_ros
from skimage.draw import line as ray_trace
import rospkg
import matplotlib.pyplot as plt
from typing import List

# msgs
from nav_msgs.msg import (
    OccupancyGrid, # https://docs.ros.org/en/noetic/api/nav_msgs/html/msg/OccupancyGrid.html
    MapMetaData # https://docs.ros.org/en/noetic/api/nav_msgs/html/msg/MapMetaData.html
    )
from geometry_msgs.msg import TransformStamped # https://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/TransformStamped.html
from sensor_msgs.msg import LaserScan # https://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/LaserScan.html

from utils import convert_pose_to_tf, convert_tf_to_pose, euler_from_ros_quat, \
     tf_to_tf_mat, tf_mat_to_tf


ALPHA = 10 # default: 1
BETA = 1 # default: 1
MAP_DIM = (4, 4)
CELL_SIZE = .01
NUM_PTS_OBSTACLE = 3
SCAN_DOWNSAMPLE = 1
CONFIDENCE_THRESH = 5 # default: 0
LIKELIHOOD_CUTOFF = 50 # default: 100

class OccupancyGripMap:
    def __init__(self):
        # use tf2 buffer to access transforms between existing frames in tf tree
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_br = tf2_ros.TransformBroadcaster()

        # subscribers and publishers
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_cb, queue_size=1)
        self.map_pub = rospy.Publisher('/map', OccupancyGrid, queue_size=1)

        # attributes
        width = int(MAP_DIM[0] / CELL_SIZE); height = int(MAP_DIM[1] / CELL_SIZE)
        self.log_odds = np.zeros((width, height))
        self.np_map = np.ones((width, height), dtype=np.uint8) * -1  # -1 for unknown
        self.map_msg = OccupancyGrid()
        self.map_msg.info = MapMetaData()
        self.map_msg.info.resolution = CELL_SIZE
        self.map_msg.info.width = width
        self.map_msg.info.height = height

        # transforms
        self.base_link_scan_tf = self.tf_buffer.lookup_transform('base_link', 'base_scan', rospy.Time(0),
                                                            rospy.Duration(2.0))
        odom_tf = self.tf_buffer.lookup_transform('odom', 'base_link', rospy.Time(0), rospy.Duration(2.0)).transform

        # set origin to center of map
        rob_to_mid_origin_tf_mat = np.eye(4)
        rob_to_mid_origin_tf_mat[0, 3] = -width / 2 * CELL_SIZE
        rob_to_mid_origin_tf_mat[1, 3] = -height / 2 * CELL_SIZE
        odom_tf_mat = tf_to_tf_mat(odom_tf)
        self.map_msg.info.origin = convert_tf_to_pose(tf_mat_to_tf(odom_tf_mat.dot(rob_to_mid_origin_tf_mat)))

        # map to odom broadcaster
        self.map_odom_timer = rospy.Timer(rospy.Duration(0.1), self.broadcast_map_odom)
        self.map_odom_tf = TransformStamped()
        self.map_odom_tf.header.frame_id = 'map'
        self.map_odom_tf.child_frame_id = 'odom'
        self.map_odom_tf.transform.rotation.w = 1.0

        rospy.spin()
        plt.imshow(100-self.np_map, cmap='gray', vmin=0, vmax=100)
        rospack = rospkg.RosPack()
        path = rospack.get_path("rob521_lab3")
        plt.savefig(path+"/map.png")

    def broadcast_map_odom(self, e):
        self.map_odom_tf.header.stamp = rospy.Time.now()
        self.tf_br.sendTransform(self.map_odom_tf)

    def scan_cb(self, scan_msg):
        # read new laser data and populate map
        # get current odometry robot pose
        try:
            odom_tf = self.tf_buffer.lookup_transform('odom', 'base_scan', rospy.Time(0)).transform
        except tf2_ros.TransformException:
            rospy.logwarn('Pose from odom lookup failed. Using origin as odom.')
            odom_tf = convert_pose_to_tf(self.map_msg.info.origin)

        # get odom in frame of map
        odom_map_tf = tf_mat_to_tf(
            np.linalg.inv(tf_to_tf_mat(convert_pose_to_tf(self.map_msg.info.origin))).dot(tf_to_tf_mat(odom_tf))
        )
        odom_map = np.zeros(3)
        odom_map[0] = odom_map_tf.translation.x
        odom_map[1] = odom_map_tf.translation.y
        odom_map[2] = euler_from_ros_quat(odom_map_tf.rotation)[2]

        # YOUR CODE HERE!!! Loop through each measurement in scan_msg to get the correct angle and
        # x_start and y_start to send to your ray_trace_update function.

        # Convert the odom transform to the map frame.
        map_origin_tf = convert_pose_to_tf(self.map_msg.info.origin)
        map_origin_mat = tf_to_tf_mat(map_origin_tf)
        odom_mat = tf_to_tf_mat(odom_tf)
        robot_mat = np.linalg.inv(map_origin_mat).dot(odom_mat)
        robot_tf = tf_mat_to_tf(robot_mat)

        # Extract robot position and orientation.
        robot_x = robot_tf.translation.x
        robot_y = robot_tf.translation.y
        robot_yaw = euler_from_ros_quat(robot_tf.rotation)[2]

        # Use the robot's position as the laser scan origin.
        x_start = robot_x
        y_start = robot_y

        ranges: List[float] = scan_msg.ranges
        for i in range(0, len(ranges), SCAN_DOWNSAMPLE):
            beam_angle = scan_msg.angle_min + i * scan_msg.angle_increment
            angle = beam_angle + robot_yaw
            range_mes = ranges[i]

            # print(f"Angle: {angle}, Range: {range_mes}")
            # print(f"Range max: {scan_msg.range_max}")
            # print(f"Robot x: {robot_x}, Robot y: {robot_y}")
            
            if np.isnan(range_mes) or np.isinf(range_mes) or range_mes <= 0:
                continue

            # TODO: will most likely need to make a transform from the base_link to the base_scan
            self.ray_trace_update(
                map=self.np_map, 
                log_odds=self.log_odds, 
                x_start=x_start,
                y_start=y_start,
                angle=angle,
                range_mes=range_mes,
                range_max=scan_msg.range_max)
            

        # Note that the first lidar beam in the message points directly in front of the robot (x-axis of the robot), and each subsequent
        # beam moves in a counter-clockwise direction with an angle change equal to
        # scan_msg.angle_increment.

        # publish the message
        self.map_msg.info.map_load_time = rospy.Time.now()
        self.map_msg.data = self.np_map.flatten()
        self.map_pub.publish(self.map_msg)

    def ray_trace_update(self, map, log_odds, x_start, y_start, angle, range_mes, range_max):
        """
        A ray tracing grid update as described in the lab document.

        :param map: The numpy map.
        :param log_odds: The map of log odds values.
        :param x_start: The x starting point in the map coordinate frame (i.e. the x 'pixel' that the robot is in).
        :param y_start: The y starting point in the map coordinate frame (i.e. the y 'pixel' that the robot is in).
        :param angle: The ray angle relative to the x axis of the map.
        :param range_mes: The range of the measurement along the ray.
        :return: The numpy map and the log odds updated along a single ray.
        """
        # YOUR CODE HERE!!! You should modify the log_odds object and the numpy map based on the outputs from
        # ray_trace and the equations from class. Your numpy map must be an array of int8s with 0 to 100 representing
        # probability of occupancy, and -1 representing unknown.

        # Note: ray_trace will return the indices of the pixels in the map that belong to the LIDAR ray

        # Get the x and y end points of the ray
        x_end = x_start + range_mes * np.cos(angle)
        y_end = y_start + range_mes * np.sin(angle)

        # Discretize start and end positions to grid indices.
        x_start_idx = int(np.floor(x_start / CELL_SIZE))
        y_start_idx = int(np.floor(y_start / CELL_SIZE))
        x_end_idx = int(np.floor(x_end / CELL_SIZE))
        y_end_idx = int(np.floor(y_end / CELL_SIZE))

        # Get the indices of the pixels that the ray passes through
        rr, cc = ray_trace(y_start_idx, x_start_idx, y_end_idx, x_end_idx)


        # cc is the x values and rr is the y values
        # cc is sorted from closest to x_start to x_end

        for j in range(len(rr) - 1):
            r = rr[j]
            c = cc[j]
            if r < 0 or r >= self.np_map.shape[0] or c < 0 or c >= self.np_map.shape[1]:
                continue
            self.log_odds[r, c] -= BETA
            self.log_odds[r, c] = np.clip(self.log_odds[r, c], -LIKELIHOOD_CUTOFF, LIKELIHOOD_CUTOFF)
            if self.log_odds[r, c] > CONFIDENCE_THRESH:
                self.np_map[r, c] = 100
            elif self.log_odds[r, c] < -CONFIDENCE_THRESH:
                self.np_map[r, c] = 0
            else:
                self.np_map[r, c] = -1

        # If the measurement is less than the maximum range, update the endpoint as occupied.
        if range_mes < range_max:
            r_end = rr[-1]
            c_end = cc[-1]
            if 0 <= r_end < self.np_map.shape[0] and 0 <= c_end < self.np_map.shape[1]:
                self.log_odds[r_end, c_end] += ALPHA
                self.log_odds[r_end, c_end] = np.clip(self.log_odds[r_end, c_end], -LIKELIHOOD_CUTOFF, LIKELIHOOD_CUTOFF)
                if self.log_odds[r_end, c_end] > CONFIDENCE_THRESH:
                    self.np_map[r_end, c_end] = 100
                elif self.log_odds[r_end, c_end] < -CONFIDENCE_THRESH:
                    self.np_map[r_end, c_end] = 0
                else:
                    self.np_map[r_end, c_end] = -1

        return map, log_odds

    def log_odds_to_probability(self, values):
        # print(values)
        return np.exp(values) / (1 + np.exp(values))


if __name__ == '__main__':
    try:
        rospy.init_node('mapping')
        ogm = OccupancyGripMap()
    except rospy.ROSInterruptException:
        pass