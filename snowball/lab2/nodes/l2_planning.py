#!/usr/bin/env python3
import numpy as np
import yaml
import pygame
import time
import pygame_utils
import matplotlib.image as mpimg
from skimage.draw import disk
from scipy.linalg import block_diag
from math import cos, sin, acos, pi
import matplotlib.pyplot as plt
import random
from typing import Tuple

NODE_CLOSENESS_TOL = 0.01
ITERATIONS = 10000
SCALE_FACTOR_VEL = 1
SCALE_FACTOR_ROT_VEL = 0.5

def load_map(filename):
    im = mpimg.imread("../maps/" + filename)
    if len(im.shape) > 2:
        im = im[:,:,0]
    im_np = np.array(im)  #Whitespace is true, black is false
    #im_np = np.logical_not(im_np)    
    return im_np

def acos_range(x: float) -> float:
    if x < 0:
        return acos(x) - pi
    else:
        return acos(x)


def load_map_yaml(filename):
    with open("../maps/" + filename, "r") as stream:
            map_settings_dict = yaml.safe_load(stream)
    return map_settings_dict

#Node for building a graph
class Node:
    def __init__(self, robot_pose: np.ndarray, parent_id, cost):
        self.robot_pose = robot_pose # A 3 by 1 vector [x, y, theta]
        self.parent_id = parent_id # The parent node id that leads to this node (There should only every be one parent in RRT)
        self.cost = cost # The cost to come to this node
        self.children_ids = [] # The children node ids of this node
        return

class SlidingWindowSampler:
    def __init__(self, map_size, window_size, overlap=0.5, total_samples=100):
        self.map_size = map_size  # (width, height)
        self.window_size = window_size  # (w, h)
        self.overlap = overlap  # Overlap percentage (0 to 1)
        self.window_positions = self._generate_window_positions()
        self.num_steps = total_samples // len(self.window_positions)
        self.current_step = 0

    def _generate_window_positions(self):
        """Create a grid of window positions with overlap."""
        step_x = int(self.window_size[0] * (1 - self.overlap))  # Move less than full width
        step_y = int(self.window_size[1] * (1 - self.overlap))  # Move less than full height

        # Ensure at least one step is taken
        step_x = max(1, step_x)
        step_y = max(1, step_y)

        positions = [
            (x, y)
            for x in range(0, self.map_size[0] - self.window_size[0] + 1, step_x)
            for y in range(0, self.map_size[1] - self.window_size[1] + 1, step_y)
        ]
        
        return positions

    def sample(self):
        """Sample `num_samples` points within the current window."""
        if not self.window_positions:
            raise ValueError("No window positions generated.")

        # Get the current window position
        window_x, window_y = self.window_positions[self.current_step // self.num_steps]

        # Generate random points within the window
        x = np.random.uniform(window_x, window_x + self.window_size[0])
        y = np.random.uniform(-window_y, -window_y - self.window_size[1])

        # Move to the next window after `num_steps` iterations
        self.current_step += 1
        if self.current_step // self.num_steps >= len(self.window_positions):
            self.current_step = 0  # Reset after covering the whole map

        return np.array([[x], [y]])

#Path Planner 
class PathPlanner:
    #A path planner capable of perfomring RRT and RRT*
    def __init__(self, map_filename, map_setings_filename, goal_point, stopping_dist, headless: bool = True):
        #Get map information
        self.occupancy_map = load_map(map_filename)
        self.map_shape = self.occupancy_map.shape
        self.map_settings_dict = load_map_yaml(map_setings_filename)

        #Get the metric bounds of the map
        self.bounds = np.zeros([2,2]) #m
        self.bounds[0, 0] = self.map_settings_dict["origin"][0]
        self.bounds[1, 0] = self.map_settings_dict["origin"][1]
        self.bounds[0, 1] = self.map_settings_dict["origin"][0] + self.map_shape[1] * self.map_settings_dict["resolution"]
        self.bounds[1, 1] = self.map_settings_dict["origin"][1] + self.map_shape[0] * self.map_settings_dict["resolution"]
        self.resolution = self.map_settings_dict["resolution"]
        self.origin = self.map_settings_dict["origin"]
        self.origin_pixel = np.zeros((2,1)) # array containing the pixel coordinates of the origin
        self.origin_pixel[0] = - self.origin[0] / self.resolution
        self.origin_pixel[1] = (self.map_shape[0]*self.resolution + self.origin[1]) / self.resolution
        self.origin = np.array(self.origin).reshape(3,1)
        #Robot information
        self.robot_radius = 0.22 #m
        self.vel_max = 0.5 #m/s (Feel free to change!)
        self.rot_vel_max = 0.2 #rad/s (Feel free to change!)

        #Goal Parameters
        self.goal_point = goal_point #m
        self.stopping_dist = stopping_dist #m

        #Trajectory Simulation Parameters
        self.timestep = 1 #s
        self.num_substeps = 5 #Number of substeps in the trajectory

        #Planning storage
        self.nodes = [Node(np.zeros((3,1)), -1, 0)]

        # Sampler
        self.sampler = SlidingWindowSampler((55,66), (10, 10), overlap=0.25, total_samples=int(ITERATIONS/2))
        self.sampler2 = SlidingWindowSampler((55,65), (15, 15), overlap=0.75, total_samples=int(ITERATIONS/2))

        #RRT* Specific Parameters
        self.lebesgue_free = np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] **2
        self.zeta_d = np.pi
        self.gamma_RRT_star = 2 * (1 + 1/2) ** (1/2) * (self.lebesgue_free / self.zeta_d) ** (1/2)
        self.gamma_RRT = self.gamma_RRT_star + .1
        self.epsilon = 2.5
        self.samples_so_far = 0

        self.headless = headless
        if not self.headless:
            #Pygame window for visualization
            self.window = pygame_utils.PygameWindow(
                map_filename, (1000, 1000), self.occupancy_map.shape, self.map_settings_dict, self.goal_point, self.stopping_dist)
        return

    #Functions required for RRT
    def sample_map_space(self) -> np.ndarray:
        #Return an [x,y] coordinate to drive the robot towards 
        # sampling in meters, from origin
        # random_x = np.random.uniform(0, self.bounds[0,1]*min(1,50*len(self.nodes)/ITERATIONS))  
        # random_y = np.random.uniform(self.bounds[1,0]*min(1,50*len(self.nodes)/ITERATIONS), self.bounds[1,1]*min(1,50*len(self.nodes)/ITERATIONS))
        # progressively expand the search space as the number of nodes increases
        # fac = min(1,5*len(self.nodes)/ITERATIONS)
        # random_x = np.random.uniform(-10*fac, 40*fac)
        # random_y = np.random.uniform(-45*fac, 15*fac)
        # random_r = np.random.uniform(0, 40)*min(1,10*len(self.nodes)/ITERATIONS)
        # random_theta = np.random.uniform(-np.pi, np.pi)
        # random_x = random_r * np.cos(random_theta)
        # random_y = random_r * np.sin(random_theta)
        # make sure the point is within the bounds
        # random_x = np.clip(random_x, self.bounds[0,0], self.bounds[0,1])
        # random_y = np.clip(random_y, self.bounds[1,0], self.bounds[1,1])
        # random_x = np.clip(random_x, -10, 40)
        # random_y = np.clip(random_y, -45, 15)
        print(f"Samples so far: {self.samples_so_far}")

        if self.samples_so_far < ITERATIONS/2:
            return self.sampler.sample() + np.array([[-5], [15]])
        else:
            return self.sampler2.sample() + np.array([[-5], [15]])


    def check_if_duplicate(self, pose: np.ndarray) -> bool:
        #Check if point is a duplicate of an already existing node
        for node in self.nodes:
            if np.allclose(node.robot_pose, pose, atol=NODE_CLOSENESS_TOL):
                return True
        return False
    
    def closest_node(self, point: np.ndarray) -> int:
        #Returns the index of the closest node
        closest_node = None
        closest_dist = np.inf
        for i, node in enumerate(self.nodes):
            dist = np.linalg.norm(node.robot_pose[:2] - point)
            angle_to_target = np.arctan2(point[1] - node.robot_pose[1], point[0] - node.robot_pose[0])
            angle_diff = np.arctan2(np.sin(angle_to_target - node.robot_pose[2]), np.cos(angle_to_target - node.robot_pose[2]))
            cost = dist + 1.5*abs(angle_diff)
            if cost < closest_dist:
                closest_dist = cost
                closest_node = i
        return closest_node
    
    def simulate_trajectory(self, node_i: Node, point_s: np.ndarray) -> np.ndarray:
        #Simulates the non-holonomic motion of the robot.
        #This function drives the robot from node_i towards point_s. This function does has many solutions!
        #node_i is a 3 by 1 vector [x;y;theta] this can be used to construct the SE(2) matrix T_{OI} in course notation
        #point_s is the sampled point vector [x; y]
        vel, rot_vel = self.robot_controller(node_i, point_s)

        robot_traj = self.trajectory_rollout(node_i, vel, rot_vel)
        return robot_traj
    
    def robot_controller(self, node_i: Node, point_s: np.ndarray) -> Tuple[float,float]:
        #This controller determines the velocities that will nominally move the robot from node i to node s
        #Max velocities should be enforced

        dt = self.timestep * self.num_substeps

        robot_pose = node_i.robot_pose[:2].flatten()
        theta = node_i.robot_pose[2]
        point_flat = point_s.flatten()

        dist = np.linalg.norm(robot_pose - point_flat)
        vel = (dist / dt).flatten() * SCALE_FACTOR_VEL
        angle_to_target = np.arctan2(point_flat[1] - robot_pose[1], point_flat[0] - robot_pose[0])
        angle_diff = np.arctan2(np.sin(angle_to_target - theta), np.cos(angle_to_target - theta))
        # if angle_diff > np.pi/2:
        #     vel = 0.001

        # Compute rotational velocity
        rot_vel = angle_diff / dt
        vel = float(min(vel, self.vel_max))
        rot_vel = float(np.sign(rot_vel) * min(abs(rot_vel), self.rot_vel_max))

        return vel, rot_vel
            
    def trajectory_rollout(self,  node_i: Node, vel: float, rot_vel: float):
        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points to check for collisions
        rollout = np.zeros((3, self.num_substeps))
        rollout[:,0] = node_i.robot_pose.flatten()
        
        # for i in range(1,self.num_substeps):
        #     # Forward Euler
        #     last_pos = rollout[:,i-1]
        #     q_dot = unicycle_model(vel=vel, rot_vel=rot_vel, theta=last_pos[2])
        #     rollout[:,i] = last_pos + (q_dot * self.timestep).flatten()

        # first perform the rotation in first step
        rollout[:,1] = rollout[:,0] + np.array([0, 0, rot_vel * self.timestep * self.num_substeps])
        for i in range(2,self.num_substeps):
            # Forward Euler
            last_pos = rollout[:,i-1]
            q_dot = unicycle_model(vel=vel, rot_vel=0, theta=last_pos[2])
            rollout[:,i] = last_pos + (q_dot * self.timestep).flatten()
    

        return rollout
    
    def point_to_cell(self, point: np.ndarray) -> np.ndarray:
        # Convert a series of [x,y] points in the map to the indices for the corresponding cell in the occupancy map
        # point is a 2 by N matrix of points of interest
        pixel_coords = np.zeros_like(point)
        for i in range(point.shape[1]):
            pixel_coords[0,i] = (point[0,i])/self.resolution + self.origin_pixel[0] 
            pixel_coords[1,i] = self.origin_pixel[1] - (point[1,i])/self.resolution
        return pixel_coords

    def points_to_robot_circle(self, points: np.ndarray) -> np.ndarray:
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function
        all_points = []
        mapped_pts = self.point_to_cell(points)
        for pt in range(mapped_pts.shape[1]):
            rr, cc = disk(radius=self.robot_radius/self.resolution, center=mapped_pts[:,pt])
            all_points.append(np.vstack((rr,cc)))
        return np.hstack(all_points)    

    #RRT* specific functions
    def ball_radius(self):
        #Close neighbor distance
        card_V = len(self.nodes)
        return min(self.gamma_RRT * (np.log(card_V) / card_V ) ** (1.0/2.0), self.epsilon)
    
    def connect_node_to_point(self, node_i, point_f):
        #Given two nodes find the non-holonomic path that connects them
        #Settings
        #node is a 3 by 1 node
        #point is a 2 by 1 point
        print("TO DO: Implement a way to connect two already existing nodes (for rewiring).")
        return np.zeros((3, self.num_substeps))
    
    @staticmethod
    def cost_to_go(trajectory_o: np.ndarray) -> float:
        #The cost to get to a node from lavalle 
        cost = 0.0
        for i in range(1, trajectory_o.shape[1]):
            cost += np.linalg.norm(trajectory_o[:2, i] - trajectory_o[:2, i-1])
        return cost
    
    def update_children(self, node_id):
        #Given a node_id with a changed cost, update all connected nodes with the new cost
        print("TO DO: Update the costs of connected nodes after rewiring.")
        return
    
    def collision_detected(self, robot_traj: np.ndarray) -> bool:
        # Check if the robot trajectory collides with the map
        coords = self.points_to_robot_circle(robot_traj[:2, :]).reshape(2,-1)
        col_indices, row_indices = coords[0], coords[1]
        # ensure indices are within bounds
        row_indices = np.clip(row_indices, 0, self.occupancy_map.shape[0]-1)
        col_indices = np.clip(col_indices, 0, self.occupancy_map.shape[1]-1)
        return np.any(self.occupancy_map[row_indices, col_indices] == 0)

    #Planner Functions
    def rrt_planning(self):
        #This function performs RRT on the given map and robot
        #You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
        for i in range(ITERATIONS): #Most likely need more iterations than this to complete the map!
            #Sample map space
            point = self.sample_map_space()
            self.samples_so_far += 1
            if not self.headless:
                self.window.add_point(point[:2].flatten(), radius = 2, color=(0,0,255))
            
            #Get the closest point
            closest_node_id = self.closest_node(point)
            closest_node = self.nodes[closest_node_id]

            #Simulate driving the robot towards the closest point
            trajectory_o = self.simulate_trajectory(closest_node, point)
            # Test for collision
            if self.collision_detected(trajectory_o):
                if not self.headless:
                    # print("Collision Detected!")
                    # find first collision point
                    # for i in range(1, trajectory_o.shape[1]):
                    #     if self.collision_detected(trajectory_o[:,:i]):
                    #         # set the trajectory to the point before the collision
                    #         trajectory_o[:,i:] = trajectory_o[:,i-1].reshape(3,1)
                    #         break
                    continue
                    # for i inrange(1, trajectory_o.shape[1]):
                    #     self.window.add_point(trajectory_o[:2,i], radius=2, color=(255,0,0))
            # plot trajectory

            if not self.headless:
                for i in range(1, trajectory_o.shape[1]):
                    self.window.add_point(trajectory_o[:2,i], radius=2, color=(0,255,0))
            # # ADD TRANSLATION FROM LAST THING
            # if not self.headless:
            #     plt.imshow(self.occupancy_map)
            #     # plot sampled point
            #     self.window.add_point(point.flatten(), radius=5, color=(0,0,255))

            #     point = self.point_to_cell(point.reshape(2,1))
            #     plt.scatter(point[0], point[1], color='b')
            #     for i in range(1, trajectory_o.shape[1]):
            #         # print(f"Trajectory point: {trajectory_o[:2,i]}")
            #         point_image_space = self.point_to_cell(trajectory_o[:2,i].reshape(2,1))
            #         plt.scatter(point_image_space[0], point_image_space[1], color='r')
            #         # print(f"Trajectory point in image space: {point_image_space}")
            #         self.window.add_point(trajectory_o[:2,i], radius=5, color=(255,0,0))
            #         # self.window.add_point_image_space(point_image_space, radius=5, color=(0,0,255))
            #     plt.savefig(f"trajectory_{i}.png")

            # check for duplicates

            if not self.check_if_duplicate(trajectory_o[:,-1]):
                # print("No duplicate")
                cost_of_trajectory = self.cost_to_go(trajectory_o)
                cost_to_come = closest_node.cost + cost_of_trajectory
                new_node = Node(robot_pose=trajectory_o[:,-1].reshape(3,1), parent_id=closest_node_id, cost=cost_to_come)


            # Update graph
            self.nodes.append(new_node)
            # time.sleep(0.1)
            #Check if goal has been reached
            if np.linalg.norm(new_node.robot_pose[:2] - self.goal_point) < self.stopping_dist:
                print("GOAL REACHED!")
                time.sleep(5)
                return self.nodes
            
        print("GOAL NOT REACHED!")

        return self.nodes
    
    def rrt_star_planning(self):
        #This function performs RRT* for the given map and robot        
        for i in range(1): #Most likely need more iterations than this to complete the map!
            #Sample
            point = self.sample_map_space()

            #Closest Node
            closest_node_id = self.closest_node(point)

           #Simulate driving the robot towards the closest point
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            #Check for collisions
            print("TO DO: Check for collision.")

            #Last node rewire
            print("TO DO: Last node rewiring")

            #Close node rewire
            print("TO DO: Near point rewiring")

            #Check for early end
            print("TO DO: Check for early end")
        return self.nodes
    
    def recover_path(self, node_id = -1):
        path = [self.nodes[node_id].robot_pose]
        current_node_id = self.nodes[node_id].parent_id
        while current_node_id > -1:
            path.append(self.nodes[current_node_id].robot_pose)
            current_node_id = self.nodes[current_node_id].parent_id
        path.reverse()
        return path


def unicycle_model(vel: float, rot_vel: float, theta: float) -> np.ndarray:
        G = np.array(
            [[cos(theta),0],
            [sin(theta),0],
            [0,1]]
        )
        p = np.array([[vel],[rot_vel]])
        return G @ p

def main():
    #Set map information
    map_filename = "myhal.png"
    map_setings_filename = "myhal.yaml"
    map_filename = "willowgarageworld_05res.png"
    map_setings_filename = "willowgarageworld_05res.yaml"


    #robot information
    goal_point = np.array([[42], [-44]]) #m
    stopping_dist = 0.5 #m

    #RRT precursor
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist, headless=False)
    nodes = path_planner.rrt_planning()
    print("Path Length: ", len(nodes))
    print(nodes[0].robot_pose)
    print(nodes[-1].robot_pose)
    path = path_planner.recover_path()
    print(len(path))
    # node_path_metric = np.hstack(path_planner.recover_path())

    #Leftover test functions
    # np.save("shortest_path.npy", node_path_metric)


if __name__ == '__main__':
    main()