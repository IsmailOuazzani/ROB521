#!/usr/bin/env python3
import numpy as np
import yaml
import pygame
import time
import pygame_utils
import matplotlib.image as mpimg
from skimage.draw import disk
from scipy.linalg import block_diag
from math import cos, sin, acos
import random

NODE_CLOSENESS_TOL = 0.01

def load_map(filename):
    im = mpimg.imread("../maps/" + filename)
    if len(im.shape) > 2:
        im = im[:,:,0]
    im_np = np.array(im)  #Whitespace is true, black is false
    #im_np = np.logical_not(im_np)    
    return im_np


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

        #Robot information
        self.robot_radius = 0.22 #m
        self.vel_max = 0.5 #m/s (Feel free to change!)
        self.rot_vel_max = 0.2 #rad/s (Feel free to change!)

        #Goal Parameters
        self.goal_point = goal_point #m
        self.stopping_dist = stopping_dist #m

        #Trajectory Simulation Parameters
        self.timestep = 1.0 #s
        self.num_substeps = 10

        #Planning storage
        self.nodes = [Node(np.zeros((3,1)), -1, 0)]

        #RRT* Specific Parameters
        self.lebesgue_free = np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] **2
        self.zeta_d = np.pi
        self.gamma_RRT_star = 2 * (1 + 1/2) ** (1/2) * (self.lebesgue_free / self.zeta_d) ** (1/2)
        self.gamma_RRT = self.gamma_RRT_star + .1
        self.epsilon = 2.5
        
        if not headless:
            #Pygame window for visualization
            self.window = pygame_utils.PygameWindow(
                "Path Planner", (1000, 1000), self.occupancy_map.shape, self.map_settings_dict, self.goal_point, self.stopping_dist)
        return

    #Functions required for RRT
    def sample_map_space(self) -> np.ndarray:
        #Return an [x,y] coordinate to drive the robot towards
        random_x = random.random() * (self.map_shape[0,1] - self.map_shape[0,0])
        random_y = random.random() * (self.map_shape[1,1] - self.map_shape[1,0])

        return np.array([[random_x],[random_y]])
    
    def check_if_duplicate(self, point: np.ndarray) -> bool:
        #Check if point is a duplicate of an already existing node
        for node in self.nodes:
            if np.allclose(node.robot_pose[:2], point, atol=NODE_CLOSENESS_TOL):
                return True
        return False
    
    def closest_node(self, point):
        #Returns the index of the closest node
        print("TO DO: Implement a method to get the closest node to a sapled point")
        return 0
    
    def simulate_trajectory(self, node_i, point_s):
        #Simulates the non-holonomic motion of the robot.
        #This function drives the robot from node_i towards point_s. This function does has many solutions!
        #node_i is a 3 by 1 vector [x;y;theta] this can be used to construct the SE(2) matrix T_{OI} in course notation
        #point_s is the sampled point vector [x; y]
        vel, rot_vel = self.robot_controller(node_i, point_s)

        robot_traj = self.trajectory_rollout(vel, rot_vel)
        return robot_traj
    
    def robot_controller(self, node_i: Node, point_s: np.ndarray) -> tuple[float,float]:
        #This controller determines the velocities that will nominally move the robot from node i to node s
        #Max velocities should be enforced

        dt = self.timestep * self.num_substeps

        robot_pose = node_i.robot_pose[:2].flatten()
        theta = node_i.robot_pose[2]
        point_flat = point_s.flatten()

        
        vel = (np.linalg.norm(robot_pose - point_flat) / dt).flatten()
        rot_vel = -(acos(
            np.dot(point_flat - robot_pose, np.array([[cos(theta)], [sin(theta)]]))/(np.linalg.norm(point_flat-robot_pose)))
        )/dt

        vel = float(min(vel, self.vel_max))
        rot_vel = float(np.sign(rot_vel) * min(abs(rot_vel), self.rot_vel_max))

        return vel,rot_vel
            
    def trajectory_rollout(self, vel: float, rot_vel: float):
        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points to check for collisions
        rollout = np.zeros((3, self.num_substeps))
        
        for i in range(1,self.num_substeps):
            # Forward Euler
            last_pos = rollout[:,i-1]
            q_dot = unicycle_model(vel=vel, rot_vel=rot_vel, theta=last_pos[2])
            rollout[:,i] = last_pos + (q_dot * self.timestep).flatten()

        return rollout
    
    def point_to_cell(self, point: np.ndarray) -> np.ndarray:
        #Convert a series of [x,y] points in the map to the indices for the corresponding cell in the occupancy map
        #point is a 2 by N matrix of points of interest
        c = self.map_settings_dict["resolution"]
        k = np.array([
            [c, 0, self.bounds[0,0]],
            [0, c, self.bounds[1,0]],
            [0, 0, 1]
        ])
        num_points = point.shape[1]
        ones = np.ones((1,num_points))
        homo_point = np.vstack((point,ones))

        transformed = k @ homo_point
        x_normalized = transformed[0, :] / transformed[2, :]
        y_normalized = transformed[1, :] / transformed[2, :]
        cell_indices = np.vstack((x_normalized, y_normalized))
        return cell_indices.astype(int)

    def points_to_robot_circle(self, points: np.ndarray) -> np.ndarray:
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function
        all_points = []
        mapped_pts = self.point_to_cell(points)
        for pt in range(mapped_pts.shape[1]):
            rr, cc = disk(radius=self.robot_radius, center=mapped_pts[:,pt])
            all_points.append(np.vstack((cc,rr)))
        return np.dstack(all_points)    

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
    
    def cost_to_come(self, trajectory_o):
        #The cost to get to a node from lavalle 
        print("TO DO: Implement a cost to come metric")
        return 0
    
    def update_children(self, node_id):
        #Given a node_id with a changed cost, update all connected nodes with the new cost
        print("TO DO: Update the costs of connected nodes after rewiring.")
        return

    #Planner Functions
    def rrt_planning(self):
        #This function performs RRT on the given map and robot
        #You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
        for i in range(1): #Most likely need more iterations than this to complete the map!
            #Sample map space
            point = self.sample_map_space()
            

            #Get the closest point
            closest_node_id = self.closest_node(point)

            #Simulate driving the robot towards the closest point
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            #Check for collisions
            print("TO DO: Check for collisions and add safe points to list of nodes.")
            
            #Check if goal has been reached
            print("TO DO: Check if at goal point.")
        return self.nodes
    
    def rrt_star_planning(self):
        #This function performs RRT* for the given map and robot        
        for i in range(1): #Most likely need more iterations than this to complete the map!
            #Sample
            point = self.sample_map_space()

            #Closest Node
            closest_node_id = self.closest_node(point)

            #Simulate trajectory
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            #Check for Collision
            print("TO DO: Check for collision.")

            #Last node rewire
            print("TO DO: Last node rewiring")

            #Close node rewire
            print("TO DO: Near point rewiring")

            #Check for early end
            print("TO DO: Check for early end")
        return self.nodes
    
    def recover_path(self, node_id = -1):
        path = [self.nodes[node_id].point]
        current_node_id = self.nodes[node_id].parent_id
        while current_node_id > -1:
            path.append(self.nodes[current_node_id].point)
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
    map_filename = "willowgarageworld_05res.png"
    map_setings_filename = "willowgarageworld_05res.yaml"

    #robot information
    goal_point = np.array([[10], [10]]) #m
    stopping_dist = 0.5 #m

    #RRT precursor
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)
    nodes = path_planner.rrt_star_planning()
    node_path_metric = np.hstack(path_planner.recover_path())

    #Leftover test functions
    np.save("shortest_path.npy", node_path_metric)


if __name__ == '__main__':
    main()
