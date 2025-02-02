from .l2_planning import PathPlanner, Node
import numpy as np

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
