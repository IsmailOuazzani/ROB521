from .l2_planning import PathPlanner, Node, update_children
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
        cost=0,
        cost_from_parent=0
    )
    point = np.array([[1], [0]])
    PLANNER.robot_controller(node_i=node, point_s=point)

def test_simulatee_traj():
    node = Node(
        robot_pose=np.array([[0],[0],[np.pi/2]]),
        parent_id=None,
        cost=0,
        cost_from_parent=0
    )
    point = np.array([[1], [0]])
    PLANNER.simulate_trajectory(node_i=node, point_s=point)

def test_point_to_cell():
    origin = PLANNER.map_settings_dict["origin"][:2]
    origin = np.array([[origin[0]], [origin[1]]])
    mapped_pt = PLANNER.point_to_cell(np.array([[0],[0]]))
    assert np.array_equal(mapped_pt, origin.astype(int))

def test_update_children():
    nodes = [
        Node(
            robot_pose=np.array([[0],[0],[0]]),
            parent_id=-1,
            cost=0,
            cost_from_parent=0,
            children_ids=[1]
        ),
        Node(
            robot_pose=np.array([[0],[0],[0]]),
            parent_id=0,
            cost=2,
            cost_from_parent=2,
            children_ids=[2]
        ),
        Node(
            robot_pose=np.array([[0],[0],[0]]),
            parent_id=1,
            cost=5,
            cost_from_parent=3,
            children_ids=[3]
        ),
        Node(
            robot_pose=np.array([[0],[0],[0]]),
            parent_id=2,
            cost=7,
            cost_from_parent=2
        )
    ]

    root_id = 1
    nodes[root_id].cost_from_parent = 1
    nodes[root_id].cost = 1
    update_children(nodes, root_id)
    assert nodes[2].cost == 4
    assert nodes[3].cost == 6


def test_k_closest_nodes():
  planner = PathPlanner(
        map_filename="myhal.png",
        map_setings_filename="myhal.yaml",
        goal_point=np.array([[3], [3]]),
        stopping_dist=0.1
    )
  planner.nodes = [
        Node(np.array([[0.0], [0.0], [0.0]]), -1, 0, 0),  # Node 0
        Node(np.array([[1.0], [0.0], [0.0]]), 0, 0, 0),     # Node 1
        Node(np.array([[0.0], [1.1], [0.0]]), 0, 0, 0),     # Node 2
        Node(np.array([[2.0], [2.0], [0.0]]), 0, 0, 0),     # Node 3
        Node(np.array([[0.5], [0.5], [0.0]]), 0, 0, 0)      # Node 4
    ]
  test_point = np.array([[0.0], [0.0]])
  np.testing.assert_array_equal(
        planner.k_closest_nodes(test_point, 3),
        np.array([0, 4, 1])
  )