## Install
Build docker file:
```
docker build -t rob521 .
```

To run the container:
```
xhost +
docker compose run --rm rob521
```

## Testing
Go to the `nodes` directory of the relevant package (for example, `snowball/lab2/nodes`) and run:
```
PYTHONPATH=$PWD pytest
```

## LAB 3

### Simulation

#### Task 1 - Vehicle Calibration 
4 terminals

First terminal:
```
catkin_make
source devel/setup.bash
roscore
```

Second terminal:
```
source devel/setup.bash
rosrun rob521_lab3 l3_estimate_wheel_radius.py
```

Fourth terminal:
```
source devel/setup.bash
roscd rob521_lab3/  
rosbag play sample_data.bag 
```

#### TASK 2 - Estimate Robot Motion using Wheel Encoder
3 terminals

run roscore
```
source devel/setup.bash
roscore
```

run the estimator
```
source devel/setup.bash
rosrun rob521_lab3 l3_estimate_robot_motion.py
```

launch rosbag
```
source devel/setup.bash
roscd rob521_lab3/  
rosbag play sample_data.bag --pause
```

#### TASK 3 - Construct an Occupancy Grid Map 

Within the container, in `/home/catkin_ws`, run:
```
catkin_make
rospack profile
source devel/setup.bash
roslaunch rob521_lab3 mapping_rviz.launch

```
Open a new terminal with `docker exec -it` and run:
```
source devel/setup.bash
roslaunch rob521_lab3 turtlebot3_world.launch
```
In a second terminal,  with `docker exec -it`  run:
```
source devel/setup.bash
rosrun rob521_lab3 l3_mapping.py
```
In a third terminal,  with `docker exec -it`  run:
```
source devel/setup.bash
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
```
In a 4th terminal,  with `docker exec -it`  run:
```
source devel/setup.bash
rosrun rob521_lab3 l3_mapping.py
```



If you run into an error importing `skimage`, run: 
```
pip install --upgrade numpy scikit-image
```