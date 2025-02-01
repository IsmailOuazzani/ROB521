FROM osrf/ros:noetic-desktop-full

RUN apt-get update -y
RUN apt-get install -y git tmux ros-noetic-joy ros-noetic-teleop-twist-joy \
    ros-noetic-teleop-twist-keyboard ros-noetic-laser-proc \
    ros-noetic-rgbd-launch ros-noetic-rosserial-arduino \
    ros-noetic-rosserial-python ros-noetic-rosserial-client \
    ros-noetic-rosserial-msgs ros-noetic-amcl ros-noetic-map-server \
    ros-noetic-move-base ros-noetic-urdf ros-noetic-xacro \
    ros-noetic-compressed-image-transport ros-noetic-rqt* \
    ros-noetic-rviz ros-noetic-gmapping \
    ros-noetic-navigation ros-noetic-interactive-markers 

RUN apt-get install -y ros-noetic-dynamixel-sdk \
                ros-noetic-turtlebot3-msgs \
                ros-noetic-turtlebot3

RUN mkdir -p /home/catkin_ws/src
WORKDIR /home/catkin_ws
RUN cd src && git clone -b noetic-devel \
https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git



RUN useradd -m -s /bin/bash mariodininjatartuga && \
    echo "source /opt/ros/noetic/setup.bash" >> /home/mariodininjatartuga/.bashrc && \
    echo "export TURTLEBOT3_MODEL=waffle_pi" >> /home/mariodininjatartuga/.bashrc && \
    chown -R mariodininjatartuga:mariodininjatartuga /home/catkin_ws

RUN usermod -aG sudo mariodininjatartuga && \
    echo "mariodininjatartuga ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER mariodininjatartuga
WORKDIR /home/catkin_ws