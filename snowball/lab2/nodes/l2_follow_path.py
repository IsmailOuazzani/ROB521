#!/usr/bin/env python3
from __future__ import division, print_function
import os

import numpy as np
from scipy.linalg import block_diag
from scipy.spatial.distance import cityblock
import rospy
import tf2_ros
import matplotlib.pyplot as plt

# msgs
from geometry_msgs.msg import TransformStamped, Twist, PoseStamped
from nav_msgs.msg import Path, Odometry, OccupancyGrid
from visualization_msgs.msg import Marker

# ros and se2 conversion utils
import utils
from skimage.draw import disk



TRANS_GOAL_TOL = .2  # m, tolerance to consider a goal complete
ROT_GOAL_TOL = .3  # rad, tolerance to consider a goal complete
TRANS_VEL_OPTS = [0.025 , 0.075, 0.15] # m/s, max of real robot is .26
ROT_VEL_OPTS = np.linspace(-1.3, 1.3, 31)  # rad/s, max of real robot is 1.82
CONTROL_RATE = 5  # Hz, how frequently control signals are sent
CONTROL_HORIZON = 4  # seconds. if this is set too high and INTEGRATION_DT is too low, code will take a long time to run!
INTEGRATION_DT = .2  # s, delta t to propagate trajectories forward by
COLLISION_RADIUS = 0.225  # m, radius from base_link to use for collisions, min of 0.2077 based on dimensions of .281 x .306
ROT_DIST_MULT = .1  # multiplier to change effect of rotational distance in choosing correct control
OBS_DIST_MULT = .03  # multiplier to change the effect of low distance to obstacles on a path
MIN_TRANS_DIST_TO_USE_ROT = TRANS_GOAL_TOL  # m, robot has to be within this distance to use rot distance in cost
PATH_NAME = 'path.npy'  # saved path from l2_planning.py, should be in the same directory as this file

# here are some hardcoded paths to use if you want to develop l2_planning and this file in parallel
TEMP_HARDCODE_PATH = [[2, 0, 0], [2.75, -1, -np.pi/2], [2.75, -4, -np.pi/2], [2, -4.4, np.pi]]  # almost collision-free
# TEMP_HARDCODE_PATH = [[2, -.5, 0], [2.4, -1, -np.pi/2], [2.45, -3.5, -np.pi/2], [1.5, -4.4, np.pi]]  # some possible collisions

def unicycle_model(vel: np.ndarray, rot_vel: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Vectorized unicycle model."""
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    G = np.stack([cos_theta, np.zeros_like(theta), 
                  sin_theta, np.zeros_like(theta), 
                  np.zeros_like(theta), np.ones_like(theta)], axis=-1).reshape(-1, 3, 2)
    
    p = np.stack([vel, rot_vel], axis=-1)[..., np.newaxis]  # Shape: (num_opts, 2, 1)
    
    return (G @ p).squeeze(-1)  # Shape: (num_opts, 3)

class PathFollower():
    def __init__(self):
        # time full path
        self.path_follow_start_time = rospy.Time.now()

        # use tf2 buffer to access transforms between existing frames in tf tree
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.sleep(1.0)  # time to get buffer running

        # constant transforms
        self.map_odom_tf = self.tf_buffer.lookup_transform('map', 'odom', rospy.Time(0), rospy.Duration(2.0)).transform
        print(self.map_odom_tf)

        # subscribers and publishers
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.global_path_pub = rospy.Publisher('~global_path', Path, queue_size=1, latch=True)
        self.local_path_pub = rospy.Publisher('~local_path', Path, queue_size=1)
        self.collision_marker_pub = rospy.Publisher('~collision_marker', Marker, queue_size=1)

        # map
        map = rospy.wait_for_message('/map', OccupancyGrid)
        self.map_np = np.array(map.data).reshape(map.info.height, map.info.width)
        self.map_resolution = round(map.info.resolution, 5)
        self.map_origin = -utils.se2_pose_from_pose(map.info.origin)  # negative because of weird way origin is stored
        self.map_bounds = np.array([map.info.height, map.info.width])
        print(self.map_origin)
        self.map_nonzero_idxes = np.argwhere(self.map_np)
        print(map)


        # collisions
        self.collision_radius_pix = COLLISION_RADIUS / self.map_resolution
        self.collision_marker = Marker()
        self.collision_marker.header.frame_id = '/map'
        self.collision_marker.ns = '/collision_radius'
        self.collision_marker.id = 0
        self.collision_marker.type = Marker.CYLINDER
        self.collision_marker.action = Marker.ADD
        self.collision_marker.scale.x = COLLISION_RADIUS * 2
        self.collision_marker.scale.y = COLLISION_RADIUS * 2
        self.collision_marker.scale.z = 1.0
        self.collision_marker.color.g = 1.0
        self.collision_marker.color.a = 0.5

        # transforms
        self.map_baselink_tf = self.tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0), rospy.Duration(2.0))
        self.pose_in_map_np = np.zeros(3)
        self.pos_in_map_pix = np.zeros(2)
        self.update_pose()

        # path variables
        cur_dir = os.path.dirname(os.path.realpath(__file__))

        # to use the temp hardcoded paths above, switch the comment on the following two lines
        # self.path_tuples = np.load(os.path.join(cur_dir, PATH_NAME)).T
        self.path_tuples = np.array(TEMP_HARDCODE_PATH)

        self.path = utils.se2_pose_list_to_path(self.path_tuples, 'map')
        self.global_path_pub.publish(self.path)

        # goal
        self.cur_goal = np.array(self.path_tuples[0])
        self.cur_path_index = 0

        # trajectory rollout tools
        # self.all_opts is a Nx2 array with all N possible combinations of the t and v vels, scaled by integration dt
        self.all_opts = np.array(np.meshgrid(TRANS_VEL_OPTS, ROT_VEL_OPTS)).T.reshape(-1, 2)

        # if there is a [0, 0] option, remove it
        all_zeros_index = (np.abs(self.all_opts) < [0.001, 0.001]).all(axis=1).nonzero()[0]
        if all_zeros_index.size > 0:
            self.all_opts = np.delete(self.all_opts, all_zeros_index, axis=0)
        self.all_opts_scaled = self.all_opts * INTEGRATION_DT

        self.num_opts = self.all_opts_scaled.shape[0]
        self.horizon_timesteps = int(np.ceil(CONTROL_HORIZON / INTEGRATION_DT))

        self.rate = rospy.Rate(CONTROL_RATE)

        rospy.on_shutdown(self.stop_robot_on_shutdown)
        self.follow_path()

    def points_to_robot_circle(self, pixels: np.ndarray) -> np.ndarray:
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function
        all_pixels = []
        for pt in range(pixels.shape[0]):
            rr, cc = disk(radius=COLLISION_RADIUS/self.map_resolution, center=pixels[pt, :])
            all_pixels.append(np.vstack((rr,cc)))

        return np.hstack(all_pixels)   
    
    def collision_detected(self, robot_traj):
        # Check if the robot trajectory collides with the map
        
        # Transform points to robot pixel coors
        coords = self.points_to_robot_circle(robot_traj).reshape(2, -1) # disregard orientation
        col_indices, row_indices = coords[0], coords[1]
        # print(coords)
        # Ensure indices are within bounds
        valid_mask = (row_indices >= 0) & (row_indices < self.map_bounds[0]) & \
                    (col_indices >= 0) & (col_indices < self.map_bounds[1])
        
        # Filter out-of-bounds indices
        row_indices, col_indices = row_indices[valid_mask], col_indices[valid_mask]
        return not np.all(self.map_np[row_indices, col_indices] == 0)

    def follow_path(self):
        while not rospy.is_shutdown():
            # timing for debugging...loop time should be less than 1/CONTROL_RATE
            tic = rospy.Time.now()

            self.update_pose()
            self.check_and_update_goal()

            # start trajectory rollout algorithm
            local_paths = np.zeros([self.horizon_timesteps + 1, self.num_opts, 3])
            # setting all of the first points in the local paths to the current pose
            local_paths[0] = np.atleast_2d(self.pose_in_map_np).repeat(self.num_opts, axis=0)
            # print("TO DO: Propogate the trajectory forward, storing the resulting points in local_paths!")
            # Extract control inputs
            vel = np.array([opt[0] for opt in self.all_opts])  # Shape: (num_opts,)
            rot_vel = np.array([opt[1] for opt in self.all_opts])  # Shape: (num_opts,)

            for t in range(1, self.horizon_timesteps + 1):
                theta = local_paths[t - 1, :, 2]  # Extract theta for all options
                delta = INTEGRATION_DT * unicycle_model(vel, rot_vel, theta)  # Compute change in position
                local_paths[t] = local_paths[t - 1] + delta  # Apply update
            # check all trajectory points for collisions
            # first find the closest collision point in the map to each local path point
            local_paths_pixels = (self.map_origin[:2] + local_paths[:, :, :2]) / self.map_resolution
            bad_opts = []
            # plt.imshow(self.map_np)
            # for opt in range(local_paths_pixels.shape[1]):
            #     plt.scatter(local_paths_pixels[:, opt, 0], local_paths_pixels[:, opt, 1])
            # print("TO DO: Check the points in local_path_pixels for collisions")
            cost = np.inf
            best_opt = None
            for opt in range(local_paths_pixels.shape[1]):
                # plt.imshow(self.map_np)
                # plt.scatter(local_paths_pixels[:, opt, 0], local_paths_pixels[:, opt, 1])
                # plt.show()
                if self.collision_detected(local_paths_pixels[:, opt, :]):
                    print("Collision detected for option {opt}, removing from valid options.".format(opt=opt))
                    bad_opts.append(opt)
                    continue
                # Calculate the distance to the closest obstacle
                dist_to_goal = np.linalg.norm(local_paths[-1, opt, :2] - self.cur_goal[:2]) + np.abs(local_paths[-1, opt, 2] - self.cur_goal[2]) * ROT_DIST_MULT
                # Calculate the distance to the closest obstacle
                target_point = local_paths_pixels[-1, opt, :]  # Extract the point (x, y)
                distances = np.linalg.norm(self.map_nonzero_idxes - target_point, axis=1)  # Compute Euclidean distances
                min_distance = np.min(distances)  # Get the shortest distance
                new_cost = dist_to_goal - min_distance * OBS_DIST_MULT
                # Update the best option
                if new_cost < cost:
                    cost = new_cost
                    best_opt = opt

            if cost == np.inf:  # hardcoded recovery if all options have collision
                print("All options have collision, stopping robot.")
                control = [-.1, 0]
            else:
                control = self.all_opts[best_opt]
                self.local_path_pub.publish(utils.se2_pose_list_to_path(local_paths[:, best_opt], 'map'))

            # send command to robot
            self.cmd_pub.publish(utils.unicyle_vel_to_twist(control))

            # uncomment out for debugging if necessary
            print("Selected control: {control}, Loop time: {time}, Max time: {max_time}".format(
                control=control, time=(rospy.Time.now() - tic).to_sec(), max_time=1/CONTROL_RATE))

            self.rate.sleep()

    def update_pose(self):
        # Update numpy poses with current pose using the tf_buffer
        self.map_baselink_tf = self.tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0)).transform
        self.pose_in_map_np[:] = [self.map_baselink_tf.translation.x, self.map_baselink_tf.translation.y,
                                  utils.euler_from_ros_quat(self.map_baselink_tf.rotation)[2]]
        self.pos_in_map_pix = (self.map_origin[:2] + self.pose_in_map_np[:2]) / self.map_resolution
        self.collision_marker.header.stamp = rospy.Time.now()
        self.collision_marker.pose = utils.pose_from_se2_pose(self.pose_in_map_np)
        self.collision_marker_pub.publish(self.collision_marker)

    def check_and_update_goal(self):
        # iterate the goal if necessary
        dist_from_goal = np.linalg.norm(self.pose_in_map_np[:2] - self.cur_goal[:2])
        abs_angle_diff = np.abs(self.pose_in_map_np[2] - self.cur_goal[2])
        rot_dist_from_goal = min(np.pi * 2 - abs_angle_diff, abs_angle_diff)
        if dist_from_goal < TRANS_GOAL_TOL and rot_dist_from_goal < ROT_GOAL_TOL:
            rospy.loginfo("Goal {goal} at {pose} complete.".format(
                    goal=self.cur_path_index, pose=self.cur_goal))
            if self.cur_path_index == len(self.path_tuples) - 1:
                rospy.loginfo("Full path complete in {time}s! Path Follower node shutting down.".format(
                    time=(rospy.Time.now() - self.path_follow_start_time).to_sec()))
                rospy.signal_shutdown("Full path complete! Path Follower node shutting down.")
            else:
                self.cur_path_index += 1
                self.cur_goal = np.array(self.path_tuples[self.cur_path_index])
        else:
            rospy.logdebug("Goal {goal} at {pose}, trans error: {t_err}, rot error: {r_err}.".format(
                goal=self.cur_path_index, pose=self.cur_goal, t_err=dist_from_goal, r_err=rot_dist_from_goal
            ))

    def stop_robot_on_shutdown(self):
        self.cmd_pub.publish(Twist())
        rospy.loginfo("Published zero vel on shutdown.")


if __name__ == '__main__':
    try:
        rospy.init_node('path_follower', log_level=rospy.DEBUG)
        pf = PathFollower()
    except rospy.ROSInterruptException:
        pass