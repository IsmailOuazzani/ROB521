## Install
Build docker file:
```
docker build -t rob521 .
```

To run the container:
```
docker compose run --rm rob521
```

## Testing
Go to the `nodes` directory of the relevant package (for example, `snowball/lab2/nodes`) and run:
```
PYTHONPATH=$PWD pytest
```

## LAB 3

### TASK 2 - Simulation

#### Simulation using the rosbag
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