#!/usr/bin/env python3
#Standard Libraries
import numpy as np
import yaml
import pygame_utils
import matplotlib.image as mpimg
from skimage.draw import disk
from math import cos, sin, acos, pi
import random
import matplotlib.pyplot as plt
from typing import Tuple, List
import time

NODE_CLOSENESS_TOL = 0.1
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


def load_map_yaml(filename):
    with open("../maps/" + filename, "r") as stream:
            map_settings_dict = yaml.safe_load(stream)
    return map_settings_dict

#Node for building a graph
class Node:
    def __init__(self, robot_pose: np.ndarray, parent_id: int, cost_from_parent: float, cost: float, children_ids: list[int]|None = None):
        self.robot_pose = robot_pose # A 3 by 1 vector [x, y, theta]
        self.parent_id = parent_id # The parent node id that leads to this node (There should only every be one parent in RRT)
        self.cost = cost # The cost to come to this node
        self.children_ids = children_ids if children_ids else [] # The children node ids of this node
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

def cost_of_trajectory(trajectory: np.ndarray) -> float:
    diffs = np.diff(trajectory[:2, :], axis=1)
    segment_costs = np.linalg.norm(diffs, axis=0)
    return segment_costs.sum()

class SlidingWindowSampler:
    def __init__(self, map_size, window_size, overlap=0.5, total_samples=100, switch=False):
        self.map_size = map_size  # (width, height)
        self.window_size = window_size  # (w, h)
        self.overlap = overlap  # Overlap percentage (0 to 1)
        self.switch = switch # Top down or left to right
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
    def __init__(self, map_filename, map_setings_filename, goal_point, stopping_dist, headless: bool = True, uniform_sampling: bool = False, point_and_shoot: bool = False):
        #Get map information
        self.occupancy_map = load_map(map_filename)
        self.map_shape = self.occupancy_map.shape
        self.map_settings_dict = load_map_yaml(map_setings_filename)

        #Get the metric bounds of the map
        self.bounds = np.zeros([2,2]) #m
        self.resolution = self.map_settings_dict["resolution"]
        self.bounds[0, 0] = self.map_settings_dict["origin"][0]
        self.bounds[1, 0] = self.map_settings_dict["origin"][1]
        self.bounds[0, 1] = self.map_settings_dict["origin"][0] + self.map_shape[1] * self.resolution
        self.bounds[1, 1] = self.map_settings_dict["origin"][1] + self.map_shape[0] * self.resolution

        self.origin = self.map_settings_dict["origin"]
        self.origin_pixel = np.zeros((2,1)) # array containing the pixel coordinates of the origin
        self.origin_pixel[0] = - self.origin[0] / self.resolution
        self.origin_pixel[1] = (self.map_shape[0]*self.resolution + self.origin[1]) / self.resolution

        # Sampler
        self.uniform_sampling = uniform_sampling
        self.sampler = SlidingWindowSampler((50,60), (10, 10), overlap=0.2, total_samples=int(2000))
        self.sampler2 = SlidingWindowSampler((50,60), (10, 10), overlap=0.25, total_samples=int(2000), switch=True)
        self.sampler3 = SlidingWindowSampler((50,60), (2, 2), overlap=0.15, total_samples=int(2000))

        #Robot information
        self.robot_radius = 0.25 #m
        self.vel_max = 0.5 #m/s (Feel free to change!)
        self.rot_vel_max = 0.2 #rad/s (Feel free to change!)

        #Goal Parameters
        self.goal_point = goal_point #m
        self.stopping_dist = stopping_dist #m

        #Trajectory Simulation Parameters
        self.timestep = 0.5 #s
        self.num_substeps = 10
        self.point_and_shoot = point_and_shoot

        #Planning storage
        self.nodes = [Node(np.zeros((3,1)), -1, 0, 0)]
        self.num_samples = 0

        #RRT* Specific Parameters
        self.lebesgue_free = np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] **2
        self.zeta_d = np.pi
        self.gamma_RRT_star = 2 * (1 + 1/2) ** (1/2) * (self.lebesgue_free / self.zeta_d) ** (1/2)
        self.gamma_RRT = self.gamma_RRT_star + .1
        self.epsilon = 2.5
        
        #Pygame window for visualization
        self.headless = headless
        if not self.headless:
            self.window = pygame_utils.PygameWindow(
                map_filename, (1000, 1000), self.occupancy_map.shape, self.map_settings_dict, self.goal_point, self.stopping_dist)
        return

    #Functions required for RRT
    def sample_map_space(self) -> np.ndarray:
        #Return an [x,y] coordinate to drive the robot towards
        self.num_samples += 1
        if self.uniform_sampling:
            if self.num_samples < 2000:
                random_x = float(np.random.uniform(-5,50))
                random_y = float(np.random.uniform(-45,15))
            else:
                random_x = float(np.random.uniform(self.goal_point[0] - 7.5,self. goal_point[0] + 2.5))
                random_y = float(np.random.uniform(self.goal_point[1] - 2.5, self.goal_point[1] + 2.5))
            return np.array([[random_x], [random_y]])
        else: # Smarter sampling, only works on the willowgarage map TODO: generalize
          if self.num_samples < self.sampler.total_samples:
            return self.sampler.sample() + np.array([[-5], [15]])
          elif self.num_samples < self.sampler.total_samples + 500:
              # sample in the goal region
              random_x = float(np.random.uniform(self.goal_point[0] - 7.5,  self.goal_point[0] + 2.5))
              random_y = float(np.random.uniform(self.goal_point[1] - 3, self.goal_point[1] + 3))
              return np.array([[random_x], [random_y]]) 
          elif self.num_samples < self.sampler2.total_samples + self.sampler.total_samples + 500:
              return self.sampler2.sample() + np.array([[-5], [15]])
          else:
              return self.sampler3.sample() + np.array([[-5], [15]])
    
    def is_duplicate(self, pose: np.ndarray) -> bool:
        #Check if point is a duplicate of an already existing node
        for node in self.nodes:
            if np.allclose(node.robot_pose, pose, atol=NODE_CLOSENESS_TOL):
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

        robot_traj = self.trajectory_rollout(node_i, vel, rot_vel)
        return robot_traj
    
    def robot_controller(self, node_i: Node, point_s: np.ndarray) -> tuple[float,float]:
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
            
    def trajectory_rollout(self, node: Node, vel: float, rot_vel: float) -> np.ndarray:
        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points to check for collisions
        rollout = np.zeros((3, self.num_substeps))
        rollout[:,0] = node.robot_pose.flatten()
        
        if self.point_and_shoot:
            rollout[:,1] = rollout[:,0] + np.array([0, 0, rot_vel * self.timestep * self.num_substeps])
            for i in range(2,self.num_substeps):
                # Forward Euler
                last_pos = rollout[:,i-1]
                q_dot = unicycle_model(vel=vel, rot_vel=0, theta=last_pos[2])
                rollout[:,i] = last_pos + (q_dot * self.timestep).flatten()
        else:
          for i in range(1,self.num_substeps):
              # Forward Euler
              last_pos = rollout[:,i-1]
              q_dot = unicycle_model(vel=vel, rot_vel=rot_vel, theta=last_pos[2])
              rollout[:,i] = last_pos + (q_dot * self.timestep).flatten()
        return rollout
    
    def point_to_cell(self, points: np.ndarray) -> np.ndarray:
        # points is expected to be a 2 x N array.
        pixel_coords = np.empty_like(points)
        pixel_coords[0, :] = points[0, :] / self.resolution + self.origin_pixel[0]
        pixel_coords[1, :] = self.origin_pixel[1] - points[1, :] / self.resolution
        return pixel_coords.astype(np.int32)


    def points_to_robot_circle(self, points: np.ndarray) -> np.ndarray:
        #Convert a series of [x,y] points to robot map footprints for collision detection
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
        dist = np.linalg.norm(node_i.robot_pose - point_f.flatten())
        vel = self.vel_max
        final_t = dist / vel
        rollout = np.zeros((3, int(final_t / self.timestep)))
        rollout[0,:] = np.linspace(node_i.robot_pose[0], point_f[0], int(final_t / self.timestep)).flatten()
        rollout[1,:] = np.linspace(node_i.robot_pose[1], point_f[1], int(final_t / self.timestep)).flatten()
        return rollout
    
    def is_collision_detected(self, trajectory: np.ndarray) -> bool:
        #Check if the trajectory is in collision with the map
        coords = self.points_to_robot_circle(trajectory[:2, :]).reshape(2,-1)
        col_indices, row_indices = coords[0], coords[1]
        # ensure indices are within bounds
        row_indices = np.clip(row_indices, 0, self.occupancy_map.shape[0]-1)
        col_indices = np.clip(col_indices, 0, self.occupancy_map.shape[1]-1)
        return bool(np.any(self.occupancy_map[row_indices, col_indices] == 0))
    
    def k_closest_nodes(self, point: np.ndarray, k: int) -> np.ndarray:
        point = point.flatten()
        positions = np.array([node.robot_pose[:2].flatten() for node in self.nodes])
        diff = positions - point  # shape: (n, 2)
        dists_sq = np.sum(diff**2, axis=1)
        k = min(k, len(self.nodes))
        k_smallest_indices = np.argpartition(dists_sq, k - 1)[:k]
        sorted_k_indices = k_smallest_indices[np.argsort(dists_sq[k_smallest_indices])]
        return sorted_k_indices
    
    def add_node(self, parent_node_id: int, new_node_id: int, new_node: Node):
        self.nodes.append(new_node)
        self.nodes[parent_node_id].children_ids.append(new_node_id)

    #Planner Functions
    def rrt_planning(self):
        #This function performs RRT on the given map and robot
        for i in range(ITERATIONS): 
            
            #Sample map space
            point = self.sample_map_space()
            if not self.headless:
                self.window.add_point(point[:2].flatten(), radius = 2, color=(0,0,255))

            #Get the closest points. We test with multiple points because we often have nodes that are close to both sides of a wall.
            closest_node_ids = self.k_closest_nodes(point=point, k=5)
            for closest_node_id in closest_node_ids:
                closest_node = self.nodes[closest_node_id]
                trajectory_from_closest_node = self.simulate_trajectory(closest_node, point)

                if self.is_collision_detected(trajectory_from_closest_node):
                    continue
                
                # # Skip to save O(n) time
                # if self.is_duplicate(trajectory_from_closest_node[:,-1]):
                #     continue
                
                new_node = Node(
                    robot_pose=trajectory_from_closest_node[:,-1].reshape(3,1),
                    parent_id=closest_node_id,
                    cost_from_parent=0, # RRT does not need cost
                    cost=0,
                )
                new_node_id = len(self.nodes)
                self.add_node(closest_node_id, new_node_id, new_node)
                if not self.headless:
                    for i in range(1, trajectory_from_closest_node.shape[1]):
                        self.window.add_point(trajectory_from_closest_node[:2,i], radius=2, color=(0,255,0))
                break #We only need to add one node per iteration
            
            # Logging
            if i % 100 == 0:
                print(f"Sampled {self.num_samples} points, num nodes: {len(self.nodes)}")

            #Check if goal has been reached
            if np.linalg.norm(self.nodes[-1].robot_pose[:2] - self.goal_point) < self.stopping_dist:
                print("Goal Reached!")
                # Return the path
                while new_node.parent_id > -1:
                    self.window.add_point(new_node.robot_pose[:2].flatten(), radius=2, color=(255,0,0))
                    self.window.add_line(self.nodes[new_node.parent_id].robot_pose[:2].flatten(), new_node.robot_pose[:2].flatten(), width=2, color=(255,0,0))
                    new_node = self.nodes[new_node.parent_id]
                time.sleep(20)
                return self.nodes
            
    def nodes_in_radius(self, node_id: int, radius: float) -> List[int]:
        # Return a list of node ids within a given radius of the node with id node_id
        node = self.nodes[node_id]
        positions = np.array([node.robot_pose[:2].flatten() for node in self.nodes])
        diff = positions - node.robot_pose[:2].flatten()
        dists_sq = np.sum(diff**2, axis=1)
        indices = np.where(dists_sq < radius**2)[0]
        return indices
    
    def change_parent(self, node_id: int, new_parent_id: int, new_cost_from_parent: float):
        # Change the parent of the node with id node_id to the node with id new_parent_id
        node = self.nodes[node_id]
        old_parent_id = node.parent_id
        node.parent_id = new_parent_id
        self.nodes[new_parent_id].children_ids.append(node_id)
        if old_parent_id != -1:
          self.nodes[old_parent_id].children_ids.remove(node_id)
        node.cost_from_parent = new_cost_from_parent
        node.cost = self.nodes[new_parent_id].cost + new_cost_from_parent
    
    def rrt_star_planning(self):
        #This function performs RRT* for the given map and robot        
        for i in range(ITERATIONS): 
            point = self.sample_map_space()
            if not self.headless:
                self.window.add_point(point[:2].flatten(), radius = 2, color=(0,0,255))

            #Closest Nodes
            closest_nodes_ids_k = self.k_closest_nodes(point, 5)
            for closest_node_id in closest_nodes_ids_k:
                closest_node: Node = self.nodes[closest_node_id]

                trajectory_from_closest_node = self.simulate_trajectory(self.nodes[closest_node_id], point)

                if self.is_collision_detected(trajectory_from_closest_node):
                    continue
              
              # # Commenting out to save O(n) time
                if self.is_duplicate(trajectory_from_closest_node[:,-1]):
                    continue

                trajectory_from_closest_node_cost = cost_of_trajectory(trajectory_from_closest_node) 
                new_node_id = len(self.nodes)
                new_node = Node(
                    robot_pose=trajectory_from_closest_node[:,-1].reshape(3,1),
                    parent_id=closest_node_id,
                    cost_from_parent=trajectory_from_closest_node_cost,
                    cost=closest_node.cost + trajectory_from_closest_node_cost,
                )
                self.add_node(parent_node_id=closest_node_id, 
                                new_node_id=new_node_id, 
                                new_node=new_node)
              
                # Rewiring
                ball_radius = self.ball_radius()
                nodes_ids_in_radius = self.nodes_in_radius(new_node_id, ball_radius)
                for node_id in nodes_ids_in_radius:
                    if node_id == new_node_id:
                        continue
                    node = self.nodes[node_id]
                    trajectory_from_node = self.simulate_trajectory(node, new_node.robot_pose[:2])
                    if self.is_collision_detected(trajectory_from_node):
                        continue
                    trajectory_from_node_cost = cost_of_trajectory(trajectory_from_node)
                    if node.cost + trajectory_from_node_cost < new_node.cost:
                        self.change_parent(node_id=new_node_id, new_parent_id=node_id, new_cost_from_parent=trajectory_from_node_cost)

                if not self.headless:
                        trajectory_o = self.connect_node_to_point(new_node, self.nodes[new_node.parent_id].robot_pose[:2])
                        for i in range(1, trajectory_o.shape[1]):
                                self.window.add_point(trajectory_o[:2,i], radius=2, color=(0,255,0))
                            
                # Close nodes rewire
                for node_id in nodes_ids_in_radius:
                    if node_id == new_node_id:
                        continue
                    node = self.nodes[node_id]
                    trajectory_from_node = self.connect_node_to_point(node, new_node.robot_pose[:2])
                    trajectory_from_node[2,-1] = node.robot_pose[2]
                    if self.is_collision_detected(trajectory_from_node):
                        continue
                    trajectory_from_node_cost = cost_of_trajectory(trajectory_from_node)
                    if new_node.cost + trajectory_from_node_cost < node.cost:
                        self.change_parent(node_id=node_id, new_parent_id=new_node_id, new_cost_from_parent=trajectory_from_node_cost)
                        update_children(nodes=self.nodes, node_id=node_id)
                        # Plor the new path
                        if not self.headless:
                            for i in range(1, trajectory_from_node.shape[1]):
                                self.window.add_point(trajectory_from_node[:2,i], radius=2, color=(255,0,255))
                break     

            # Logging
            if i % 100 == 0:
                print(f"Sampled {self.num_samples} points, num nodes: {len(self.nodes)}")

            #Check if goal has been reached
            if np.linalg.norm(self.nodes[-1].robot_pose[:2] - self.goal_point) < self.stopping_dist:
                print("Goal Reached!")
                while new_node.parent_id > -1:
                    self.window.add_point(new_node.robot_pose[:2].flatten(), radius=2, color=(255,0,0))
                    self.window.add_line(self.nodes[new_node.parent_id].robot_pose[:2].flatten(), new_node.robot_pose[:2].flatten(), width=2, color=(255,0,0))
                    new_node = self.nodes[new_node.parent_id]
                time.sleep(20)

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
    map_filename = "willowgarageworld_05res.png"
    map_setings_filename = "willowgarageworld_05res.yaml"

    #robot information
    # goal_point = np.array([[10], [10]]) #m
    goal_point = np.array([[42], [-44]]) #m
    stopping_dist = 0.5 #m

    #RRT
    path_planner = PathPlanner(map_filename, 
                               map_setings_filename, 
                               goal_point, 
                               stopping_dist, 
                               headless=False, 
                               uniform_sampling=False,
                               point_and_shoot=True)
    nodes = path_planner.rrt_planning()
    node_path_metric = np.hstack(path_planner.recover_path())

    #RRT*
    # path_planner = PathPlanner(map_filename, 
    #                            map_setings_filename, 
    #                            goal_point, 
    #                            stopping_dist, 
    #                            headless=False, 
    #                            uniform_sampling=False,
    #                            point_and_shoot=True)
    # nodes = path_planner.rrt_star_planning()
    # node_path_metric = np.hstack(path_planner.recover_path())

    #Leftover test functions
    np.save("shortest_path.npy", node_path_metric)


if __name__ == '__main__':
    main()
