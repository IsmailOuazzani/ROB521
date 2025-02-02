from .l2_planning import PathPlanner, Node
import numpy as np
from copy import deepcopy

PLANNER = PathPlanner(
        map_filename="myhal.png",
        map_setings_filename="myhal.yaml",
        goal_point=np.array([[3],[3]]),
        stopping_dist=0.1
)

def test_trajectory_rollout():
    
    vel=2.0
    rot_vel=0.0

    assert np.array_equal(
    PLANNER.trajectory_rollout(vel=vel, rot_vel=rot_vel),
    np.array([
        [ 0.,  2.,  4.,  6.,  8., 10., 12., 14., 16., 18.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]
    ])
    )

def test_controller():
    node = Node(
        robot_pose=np.array([[0],[0],[np.pi/2]]),
        parent_id=None,
        cost=0
    )
    point = np.array([[1], [0]])
    PLANNER.robot_controller(node_i=node, point_s=point)

def test_simulatee_traj():
    node = Node(
        robot_pose=np.array([[0],[0],[np.pi/2]]),
        parent_id=None,
        cost=0
    )
    point = np.array([[1], [0]])
    PLANNER.simulate_trajectory(node_i=node, point_s=point)

def test_duplicate_nodes():
    node = deepcopy(PLANNER.nodes[0]) # to modify if the planner isnt initialised with a first node anymore
    point = node.robot_pose[:2]
    assert PLANNER.check_if_duplicate(point) is True
    node.robot_pose[0] = np.inf
    assert PLANNER.check_if_duplicate(point) is False

def test_point_to_cell():
    origin = PLANNER.map_settings_dict["origin"][:2]
    origin = np.array([[origin[0]], [origin[1]]])
    mapped_pt = PLANNER.point_to_cell(np.array([[0],[0]]))
    assert np.array_equal(mapped_pt, origin.astype(int))