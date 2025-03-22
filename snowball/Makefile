lab4-task2:
	@echo "Launching ROS and GUIs..."
	@bash -c ' \
		source /home/catkin_ws/devel/setup.bash; \
		trap "kill 0" EXIT; \
		roscore & \
		sleep 5; \
		roslaunch rob521_lab3 turtlebot3_world.launch & \
		roslaunch rob521_lab4 slam.launch & \
		roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch & \
		wait'
