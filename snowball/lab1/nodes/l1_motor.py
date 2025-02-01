#!/usr/bin/env python3
import rospy
import math
from geometry_msgs.msg import Twist
from std_msgs.msg import String

def publisher_node():
    # Initialize the publisher to send messages to the 'cmd_vel' topic
    cmd_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)

    # Initialize the Twist message
    twist = Twist()

    # Define linear velocity (for moving forward)
    twist.linear.x = 0.1  # Speed in m/s
    # Define angular velocity (for rotating)
    twist.angular.z = 0.1  # Rotation speed in rad/s

    # Command to move forward 1 meter
    rospy.loginfo("Moving forward for 1 meter")
    rate = rospy.Rate(10)  # Set the frequency of publishing messages (10Hz)
    
    # Move forward for 1 meter, assuming the robot is moving at 0.1 m/s
    # Time needed to cover 1 meter: time = distance / speed = 1 / 0.1 = 10 seconds
    move_time = 10  # seconds
    start_time = rospy.get_time()
    while rospy.get_time() - start_time < move_time:
        cmd_pub.publish(twist)
        rate.sleep()

    # Stop the robot after moving forward
    twist.linear.x = 0.0  # Set linear speed to 0
    cmd_pub.publish(twist)
    rospy.loginfo("Stopping after moving 1 meter")
    rate.sleep()

    # Command to rotate 360 degrees (2 * pi radians)
    rospy.loginfo("Rotating 360 degrees clockwise")
    twist.angular.z = 0.1  # Set angular speed for rotation (rad/s)
    
    # Time needed to rotate 360 degrees: time = angle / angular velocity = 2*pi / 0.1 rad/s = 62.83 seconds
    rotate_time = 2 * math.pi / 0.1  # radians divided by angular velocity
    start_time = rospy.get_time()
    while rospy.get_time() - start_time < rotate_time:
        cmd_pub.publish(twist)
        rate.sleep()

    # Stop the robot after rotating
    twist.angular.z = 0.0  # Set angular speed to 0
    cmd_pub.publish(twist)
    rospy.loginfo("Stopped after completing 360-degree rotation")
    rate.sleep()

def main():
    try:
        rospy.init_node('motor')
        publisher_node()
    except rospy.ROSInterruptException:
        print("EXCEPPTIOOOONN")
        pass

if __name__ == "__main__":
    print("running this shit")
    main()