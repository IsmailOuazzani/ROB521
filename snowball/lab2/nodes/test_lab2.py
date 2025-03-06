from .l2_planning import PathPlanner, Node, update_children, cost_of_trajectory
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
        [ 0.,  1.,  2.,  3.,  4., 5., 6., 7., 8., 9.],
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
    mapped_pt = PLANNER.point_to_cell(np.array([[0], [0]]))
    expected = PLANNER.origin_pixel.astype(int)
    assert np.array_equal(mapped_pt, expected)

def test_cost_of_trajectory():
    trajectory = np.array([[0, 1, 2], [0, 0, 0], [0, 0, 0]])
    assert cost_of_trajectory(trajectory) == 2


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
    planner = PathPlanner(
        map_filename="myhal.png",
        map_setings_filename="myhal.yaml",
        goal_point=np.array([[0], [0]]),
        stopping_dist=0.1
    )
    # Create a simple 10x10 occupancy map with all free cells (value 1)
    dummy_map = np.ones((10, 10), dtype=np.uint8)
    planner.occupancy_map = dummy_map
    planner.map_shape = dummy_map.shape
    # Use dummy settings: resolution 1.0 and origin at (0,0)
    planner.map_settings_dict = {"resolution": 1.0, "origin": [0, 0]}
    planner.bounds = np.array([[0, 10], [0, 10]])
    planner.robot_radius = 1.0  # disk radius = 1

    # Update dependent variables based on the new map settings.
    planner.resolution = planner.map_settings_dict["resolution"]
    planner.origin = planner.map_settings_dict["origin"]
    planner.origin_pixel = np.zeros((2, 1))
    planner.origin_pixel[0] = -planner.origin[0] / planner.resolution
    planner.origin_pixel[1] = (planner.map_shape[0] * planner.resolution + planner.origin[1]) / planner.resolution

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
    
    # Compute the footprint for a single point at (3,3)
    footprint = planner.points_to_robot_circle(np.array([[3], [3]]))
    # Ensure the footprint is 2 x N (each column is one [x; y] cell)
    footprint = footprint.reshape(2, -1)
    # In your is_collision_detected, the occupancy map is indexed as:
    #   row_indices = footprint[1] and col_indices = footprint[0]
    # So we select the first footprint cell (for example) as our obstacle location.
    obstacle_row = int(footprint[1, 0])
    obstacle_col = int(footprint[0, 0])
    
    # Place an obstacle at that cell.
    planner.occupancy_map[obstacle_row, obstacle_col] = 0

    # Create a trajectory that stays at (3,3) for all timesteps.
    trajectory = np.array([
         [3, 3, 3, 3],  # x-coordinates
         [3, 3, 3, 3],  # y-coordinates
         [0, 0, 0, 0]   # theta (unused in collision checking)
    ])
    
    # Now the footprint for (3,3) will include the forced obstacle.
    assert planner.is_collision_detected(trajectory) is True


def test_nodes_in_radius():
    import numpy as np
    # Create a dummy planner instance. (The file names here are dummies; the test
    # overrides the nodes, so the map loading is not used.)
    planner = PathPlanner(
        map_filename="myhal.png",
        map_setings_filename="myhal.yaml",
        goal_point=np.array([[0], [0]]),
        stopping_dist=0.1
    )

    # Overwrite the planner's nodes with our known test nodes.
    planner.nodes = [
        Node(np.array([[0.0], [0.0], [0.0]]), -1, 0, 0),  # Node 0 at (0, 0)
        Node(np.array([[1.0], [1.0], [0.0]]), 0, 0, 0),     # Node 1 at (1, 1)
        Node(np.array([[2.0], [2.0], [0.0]]), 0, 0, 0),     # Node 2 at (2, 2)
        Node(np.array([[0.0], [1.0], [0.0]]), 0, 0, 0),     # Node 3 at (0, 1)
        Node(np.array([[0.0], [2.0], [0.0]]), 0, 0, 0)      # Node 4 at (0, 2)
    ]

    # Using node 0 as the center, choose a radius of 1.5.
    # Expected nodes: 0, 1, and 3.
    expected_indices = np.array([0, 1, 3])
    
    # Call nodes_in_radius on node 0.
    indices = planner.nodes_in_radius(node_id=0, radius=1.5)
    
    # Since the function returns a numpy array of indices, we can use np.testing.assert_array_equal.
    # Sorting the result to ensure the order matches our expected order.
    np.testing.assert_array_equal(np.sort(indices), expected_indices)