from .l2_planning import PathPlanner, Node, update_children
import numpy as np

PLANNER = PathPlanner(
        map_filename="myhal.png",
        map_setings_filename="myhal.yaml",
        goal_point=np.array([[3],[3]]),
        stopping_dist=0.1
)

def test_trajectory_rollout():
    node = Node(
        robot_pose=np.array([[0],[0],[0]]),
        parent_id=None,
        cost=0,
        cost_from_parent=0
    )
    vel=2.0
    rot_vel=0.0

    assert np.array_equal(
    PLANNER.trajectory_rollout(node=node, vel=vel, rot_vel=rot_vel),
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




def setup_planner_for_collision_tests():
    """
    Create a PathPlanner instance with a dummy occupancy map and map settings.
    This bypasses file loading by overriding occupancy_map and map_settings_dict.
    """
    # We use dummy filenames because we override the loaded map.
    planner = PathPlanner(
        map_filename="myhal.png",
        map_setings_filename="myhal.yaml",
        goal_point=np.array([[0], [0]]),
        stopping_dist=0.1
    )
    # Create a simple 10x10 occupancy map with all free cells (value 1)
    dummy_map = np.ones((10, 10), dtype=np.uint8)
    planner.occupancy_map = dummy_map
    # Set map settings so that points are converted to indices in an easy way.
    # Here we use a resolution of 1.0 and an origin at (0, 0).
    planner.map_settings_dict = {"resolution": 1.0, "origin": [0, 0]}
    # Set the bounds to match the dummy map (10 columns and 10 rows)
    planner.bounds = np.array([[0, 10], [0, 10]])
    # Choose a robot_radius that, when divided by resolution, is an integer (for simplicity)
    planner.robot_radius = 1.0  # disk radius = 1
    return planner

def test_is_collision_detected_no_collision():
    """
    Test that a trajectory that stays in free space (all ones in the occupancy map)
    does not result in a collision.
    """
    planner = setup_planner_for_collision_tests()
    
    # Create a trajectory that stays at (3,3) for all timesteps.
    # The trajectory array has shape (3, N): first row is x, second is y, third is theta.
    trajectory = np.array([
         [3, 3, 3, 3],  # x-coordinates
         [3, 3, 3, 3],  # y-coordinates
         [0, 0, 0, 0]   # theta (unused in collision checking)
    ])
    
    # Since the occupancy map is all free, no collision should be detected.
    assert planner.is_collision_detected(trajectory) is False

def test_is_collision_detected_with_collision():
    """
    Test that a trajectory that passes over an obstacle cell (value 0 in occupancy_map)
    is correctly flagged as colliding.
    """
    planner = setup_planner_for_collision_tests()
    
    # Place an obstacle at cell (3,3). (Remember that after the coordinate conversion,
    # a point at (3,3) in the world becomes cell (3,3) when resolution=1.0 and origin=(0,0).)
    planner.occupancy_map[3, 3] = 0

    # Create a trajectory that stays at (3,3) so that its footprint (computed via a disk of radius 1)
    # will include the obstacle cell.
    trajectory = np.array([
         [3, 3, 3, 3],
         [3, 3, 3, 3],
         [0, 0, 0, 0]
    ])
    
    # Because the robot’s footprint now covers an obstacle cell, a collision should be detected.
    assert planner.is_collision_detected(trajectory) is True