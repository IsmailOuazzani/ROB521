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

def test_get_nearest_node():
    node1 = Node(
        robot_pose=np.array([[10],[25],[np.pi/2]]),
        parent_id=None,
        cost=0
    )
    node2 = Node(
        robot_pose=np.array([[20],[25],[np.pi/2]]),
        parent_id=None,
        cost=0
    )
    PLANNER.nodes = [node1, node2]

    point1 = np.array([[9], [26]])
    assert PLANNER.closest_node(point1) == 0

    point2 = np.array([[19], [25]])
    assert PLANNER.closest_node(point2) == 1

def test_point_to_cell():
    origin = PLANNER.map_settings_dict["origin"][:2]
    origin = np.array([[origin[0]], [origin[1]]])
    mapped_pt = PLANNER.point_to_cell(np.array([[0],[0]]))
    assert np.array_equal(mapped_pt, origin.astype(int))

def test_sample_map_space():
    xrange = [PLANNER.bounds[0,0],PLANNER.bounds[0,1]]
    yrange = [PLANNER.bounds[1,0],PLANNER.bounds[1,1]]
    sample = PLANNER.sample_map_space()
    print(sample,xrange,yrange)
    assert xrange[1] >= sample[0] >= xrange[0]
    assert yrange[1] >= sample[1] >= yrange[0]
