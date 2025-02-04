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
ITERATIONS = 13000
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
    def __init__(self, robot_pose: np.ndarray, parent_id: int, cost_from_parent: float, cost: float, children_ids: list[int] = []):
        self.robot_pose = robot_pose # A 3 by 1 vector [x, y, theta]
        self.parent_id = parent_id # The parent node id that leads to this node (There should only every be one parent in RRT)
        self.cost = cost # The cost to come to this node
        self.children_ids = children_ids # The children node ids of this node
        self.cost_from_parent = cost_from_parent
        return
    
def update_children(nodes: list[Node], node_id: int):
        #Given a node_id with a changed cost, update all connected nodes with the new cost
        root = nodes[node_id]
        queue = [root]
        while queue:
            node = queue.pop(0)
            for child_id in node.children_ids:
                child: Node = nodes[child_id]
                child.cost = node.cost + child.cost_from_parent
                queue.append(child)
        return

class SlidingWindowSampler:
    def __init__(self, map_size, window_size, overlap=0.5, total_samples=100, switch=False):
        self.map_size = map_size  # (width, height)
        self.window_size = window_size  # (w, h)
        self.overlap = overlap  # Overlap percentage (0 to 1)
        self.switch = switch
        self.window_positions = self._generate_window_positions()
        self.num_steps = total_samples // len(self.window_positions)
        self.total_samples = total_samples
        self.current_step = 0

    def _generate_window_positions(self):
        """Create a grid of window positions with overlap."""
        step_x = int(self.window_size[0] * (1 - self.overlap))  # Move less than full width
        step_y = int(self.window_size[1] * (1 - self.overlap))  # Move less than full height

        # Ensure at least one step is taken
        step_x = max(1, step_x)
        step_y = max(1, step_y)

        if not self.switch:
            positions = [
                (x, y)
                for x in range(0, self.map_size[0] - self.window_size[0] + 1, step_x)
                for y in range(0, self.map_size[1] - self.window_size[1] + 1, step_y)
            ]
        else:
            positions = [
                (x, y)
                for y in range(0, self.map_size[1] - self.window_size[1] + 1, step_y)
                for x in range(0, self.map_size[0] - self.window_size[0] + 1, step_x)]
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
        self.timestep = 0.5 #s
        self.num_substeps = 10 #Number of substeps in the trajectory

        #Planning storage
        self.nodes = [Node(np.zeros((3,1)), -1, 0, 0)]

        # Sampler
        self.sampler = SlidingWindowSampler((50,60), (10, 10), overlap=0.2, total_samples=int(2000))
        self.sampler2 = SlidingWindowSampler((50,60), (10, 10), overlap=0.25, total_samples=int(2000), switch=True)
        self.sampler3 = SlidingWindowSampler((50,60), (2, 2), overlap=0.1, total_samples=int(2000))

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
        # print(f"Samples so far: {self.samples_so_far}")
        if self.samples_so_far < 2000:
            random_x = float(np.random.uniform(-5,50))
            random_y = float(np.random.uniform(-50,20))
        else:
            random_x = float(np.random.uniform(self.goal_point[0] - 2.5,self. goal_point[0] + 2.5))
            random_y = float(np.random.uniform(self.goal_point[1] - 2.5, self.goal_point[1] + 2.5))
        return np.array([[random_x], [random_y]])
        # return np.array([[x], [y]])
        # if self.samples_so_far < self.sampler.total_samples:
        #     return self.sampler.sample() + np.array([[-5], [15]])
        # elif self.samples_so_far < self.sampler.total_samples + 500:
        #     # sample in the goal region
        #     random_x = float(np.random.uniform(self.goal_point[0] - 2.5,self. goal_point[0] + 2.5))
        #     random_y = float(np.random.uniform(self.goal_point[1] - 2.5, self.goal_point[1] + 2.5))
        #     return np.array([[random_x], [random_y]])
        # elif self.samples_so_far < self.sampler2.total_samples + self.sampler.total_samples + 500:
        #     return self.sampler2.sample() + np.array([[-5], [15]])
        # else:
        #     return self.sampler3.sample() + np.array([[-5], [15]])


    def check_if_duplicate(self, pose: np.ndarray) -> bool:
        #Check if point is a duplicate of an already existing node
        for node in self.nodes:
            if np.allclose(node.robot_pose, pose, atol=NODE_CLOSENESS_TOL):
                return True
        return False
    
    @staticmethod
    def cost_of_trajectory(trajectory_o: np.ndarray) -> float:
        #The cost to get to a node from lavalle 
        cost = 0.0
        for i in range(1, trajectory_o.shape[1]):
            prev_pose = trajectory_o[:, i-1]
            cur_pose = trajectory_o[:, i]
            angle_to_target = np.arctan2(prev_pose[1] - cur_pose[1], prev_pose[0] - cur_pose[0])
            angle_diff = np.arctan2(np.sin(angle_to_target - cur_pose[2]), np.cos(angle_to_target - cur_pose[2]))
            cost += np.linalg.norm(trajectory_o[:2, i] - trajectory_o[:2, i-1])
            # cost += 1.5*abs(angle_diff)
        return cost
    
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

    def nodes_within_radius(self, point: np.ndarray, radius: float) -> list[int]:
        #Returns the indices of the nodes within a certain radius
        nodes_within_radius = []
        for i, node in enumerate(self.nodes):
            dist = np.linalg.norm(node.robot_pose[:2] - point)
            angle_to_target = np.arctan2(point[1] - node.robot_pose[1], point[0] - node.robot_pose[0])
            angle_diff = np.arctan2(np.sin(angle_to_target - node.robot_pose[2]), np.cos(angle_to_target - node.robot_pose[2]))
            cost = dist #+ 1.5*abs(angle_diff)
            if cost < radius:
                nodes_within_radius.append(i)
        return nodes_within_radius

    def k_closest_nodes(self, point: np.ndarray, k: int) -> np.ndarray:
        #Returns the indices of the k closest nodes
        closest_nodes = []
        for i, node in enumerate(self.nodes):
            dist = np.linalg.norm(node.robot_pose[:2] - point)
            angle_to_target = np.arctan2(point[1] - node.robot_pose[1], point[0] - node.robot_pose[0])
            angle_diff = np.arctan2(np.sin(angle_to_target - node.robot_pose[2]), np.cos(angle_to_target - node.robot_pose[2]))
            cost = dist + 1.5*abs(angle_diff)
            closest_nodes.append((i, cost))
        closest_nodes.sort(key=lambda x: x[1])
        return [node[0] for node in closest_nodes[:k]]
    
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
    
    def connect_node_to_point(self, node_i: Node, point_f: np.ndarray) -> np.ndarray:
        #Given two nodes find the non-holonomic path that connects them
        #Settings
        #node is a 3 by 1 node
        #point is a 2 by 1 point
        # diregard theta
        dist = np.linalg.norm(node_i.robot_pose - point_f.flatten())
        vel = self.vel_max
        final_t = dist / vel
        rollout = np.zeros((3, int(final_t / self.timestep)))
        rollout[0,:] = np.linspace(node_i.robot_pose[0], point_f[0], int(final_t / self.timestep)).flatten()
        rollout[1,:] = np.linspace(node_i.robot_pose[1], point_f[1], int(final_t / self.timestep)).flatten()

        return rollout
    
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
        new_node = self.nodes[0]    
        for i in range(ITERATIONS): #Most likely need more iterations than this to complete the map!
            #Sample map space
            point = self.sample_map_space()
            self.samples_so_far += 1
            if not self.headless:
                self.window.add_point(point[:2].flatten(), radius = 2, color=(0,0,255))
            

            closest_node_ids = self.k_closest_nodes(point, 5)
            for closest_node_id in closest_node_ids:
                closest_node = self.nodes[closest_node_id]

                #Simulate driving the robot towards the closest point
                trajectory_o = self.simulate_trajectory(closest_node, point)
                # Test for collision
                if self.collision_detected(trajectory_o):
                    continue

                if not self.check_if_duplicate(trajectory_o[:,-1]):
                    # print("No duplicate")
                    cost_of_trajectory = self.cost_of_trajectory(trajectory_o)
                    cost_to_come = closest_node.cost + cost_of_trajectory
                    new_node = Node(robot_pose=trajectory_o[:,-1].reshape(3,1), parent_id=closest_node_id, cost=cost_to_come, cost_from_parent=cost_of_trajectory)
                    closest_node.children_ids.append(len(self.nodes))
                    if not self.headless:
                        for i in range(1, trajectory_o.shape[1]):
                            self.window.add_point(trajectory_o[:2,i], radius=2, color=(0,255,0))
                    # Update graph
                    self.nodes.append(new_node)
                    break
            
            # time.sleep(0.1)
            #Check if goal has been reached
            if np.linalg.norm(new_node.robot_pose[:2] - self.goal_point) < self.stopping_dist:
                print("GOAL REACHED!")
                # Return the path
                while new_node.parent_id > -1:
                    self.window.add_point(new_node.robot_pose[:2].flatten(), radius=2, color=(255,0,0))
                    self.window.add_line(self.nodes[new_node.parent_id].robot_pose[:2].flatten(), new_node.robot_pose[:2].flatten(), width=2, color=(255,0,0))
                    new_node = self.nodes[new_node.parent_id]
                time.sleep(20)
                return self.nodes
            
        print("GOAL NOT REACHED!")

        return self.nodes
    
    def rrt_star_planning(self):
        new_node= self.nodes[0]
        #This function performs RRT* for the given map and robot        
        for i in range(ITERATIONS): #Most likely need more iterations than this to complete the map!
            #Sample
            point = self.sample_map_space()
            self.samples_so_far += 1
            # if not self.headless:
            #     self.window.add_point(point[:2].flatten(), radius = 2, color=(0,0,255))

            #Closest Node
            #TODO: combine closest node and nodes_within_radius and k_closest_nodes within the same O(n) loop
            closest_nodes_ids_k = self.k_closest_nodes(point, 5)
            for closest_node_id in closest_nodes_ids_k:
                closest_node = self.nodes[closest_node_id]

            #Simulate driving the robot towards the closest point
                trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id], point)
                #Check for collisions
                if self.collision_detected(trajectory_o) or self.check_if_duplicate(trajectory_o[:,-1]):
                    continue

                #Last node rewire
                cost_from_parent = self.cost_of_trajectory(trajectory_o)
                new_node_id = len(self.nodes)
                new_node = Node(
                    robot_pose=trajectory_o[:,-1].reshape(3,1),
                    parent_id=closest_node_id,
                    cost=self.nodes[closest_node_id].cost + cost_from_parent,
                    cost_from_parent=cost_from_parent
                )
                closest_node.children_ids.append(new_node_id)
                
                self.nodes.append(new_node)
                ball_radius = self.ball_radius()
                closest_nodes_ids = self.nodes_within_radius(point=new_node.robot_pose[:2], radius=ball_radius)
                # plot trajectory
                # if not self.headless:
                #     for i in range(1, trajectory_o.shape[1]):
                #         self.window.add_point(trajectory_o[:2,i], radius=2, color=(0,255,0))
                # print(f"Inital parent: {new_node.parent_id}")
                for node_id in closest_nodes_ids:
                    if node_id == closest_node_id:
                        continue
                    node = self.nodes[node_id]
                    trajectory_o = self.simulate_trajectory(node, new_node.robot_pose[:2])
                    if self.collision_detected(trajectory_o):
                        continue
                    cost = node.cost + self.cost_of_trajectory(trajectory_o)
                    if cost < new_node.cost:
                        current_parent = new_node.parent_id
                        self.nodes[current_parent].children_ids.remove(new_node_id)    
                        new_node.parent_id = node_id
                        node.children_ids.append(new_node_id)
                        new_node.cost = cost
                
                
                # final rewire
                if not self.headless:
                    trajectory_o = self.connect_node_to_point(new_node, self.nodes[new_node.parent_id].robot_pose[:2])
                    for i in range(1, trajectory_o.shape[1]):
                            self.window.add_point(trajectory_o[:2,i], radius=2, color=(0,255,0))
                # print(f"final cost: {new_node.cost}")

                #Close node rewire
                for node_id in closest_nodes_ids:
                    node = self.nodes[node_id]
                    trajectory_o = self.connect_node_to_point(new_node, node.robot_pose[:2])
                    trajectory_o[2,trajectory_o.shape[1]-1] = node.robot_pose[2]
                    if self.collision_detected(trajectory_o):
                        continue
                    cost_from_parent = self.cost_of_trajectory(trajectory_o)
                    cost = new_node.cost + cost_from_parent
                    if cost < node.cost:
                        # print("Rewiring")
                        current_parent = node.parent_id
                        self.nodes[current_parent].children_ids.remove(node_id)
                        node.parent_id = new_node_id
                        node.cost_from_parent = cost_from_parent
                        node.cost = cost
                        new_node.children_ids.append(node_id)
                        update_children(nodes=self.nodes, node_id=node_id) 
                        # plot trajectory
                        # if not self.headless:
                        #     for i in range(1, trajectory_o.shape[1]):
                        #         self.window.add_point(trajectory_o[:2,i], radius=2, color=(100,255,100))
                break
                # if not self.headless:
                #     self.window.add_point(new_node.robot_pose[:2].flatten(), radius=2, color=(0,0, 255))
            if np.linalg.norm(new_node.robot_pose[:2] - self.goal_point) < self.stopping_dist:
                print("GOAL REACHED!")
                # Return the path
                while new_node.parent_id > -1:
                    self.window.add_point(new_node.robot_pose[:2].flatten(), radius=2, color=(255,0,0))
                    self.window.add_line(self.nodes[new_node.parent_id].robot_pose[:2].flatten(), new_node.robot_pose[:2].flatten(), width=2, color=(255,0,0))
                    new_node = self.nodes[new_node.parent_id]
                time.sleep(20)
                return self.nodes
            
        print("GOAL NOT REACHED!")
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
    # goal_point = np.array([[16], [-7]]) #m 
    # goal_point = np.array([[10], [-15]]) #m
    # goal_point = np.array([[8], [0]])
    stopping_dist = 1 #m

    #RRT precursor
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist, headless=False)
    nodes = path_planner.rrt_star_planning()
    # nodes = path_planner.rrt_planning()
    print("Path Length: ", len(nodes))
    print("Samples so far: ", path_planner.samples_so_far)
    print(nodes[0].robot_pose)
    print(nodes[-1].robot_pose)
    path = path_planner.recover_path()
    print(len(path))
    # node_path_metric = np.hstack(path_planner.recover_path())

    #Leftover test functions
    # np.save("shortest_path.npy", node_path_metric)


if __name__ == '__main__':
    main()