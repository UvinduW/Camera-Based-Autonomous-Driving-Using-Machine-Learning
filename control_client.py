#!/usr/bin/env python
from __future__ import print_function

import roslib
# roslib.load_manifest('my_package')
import sys
import rospy
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import pygame
from threading import Thread
from time import sleep
from cv_bridge import CvBridge, CvBridgeError
import StringIO
import sys
import os
import numpy as np
import socket
import pickle
import struct

#graph = tf.get_default_graph()

# NOTE: THIS CODE RUNS ON THE ROSBOT. NOT THE PC

# Thread to get commands from the networked computer
def command_stream():
    global quit_request

    control_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
    twist = Twist()
    twist.linear.y = 0
    twist.linear.z = 0
    twist.angular.x = 0
    twist.angular.y = 0

    # Connect to socket
    ip_addr = "0.0.0.0"
    port = 8001

    # Create UDP socket
    control_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    control_socket.bind((ip_addr, port))
    steer_scaling = 5.0
    while not quit_request:
        command = control_socket.recv(1024)

        if int(command) == 999:
            quit_request = 1
            command = str(00000000)
            rospy.signal_shutdown("end requested")
            break

        angle = int(command[1:4])
        angle = angle / 100.0
        if command[0] == "0":
            angle *= -1

        # Get throttle input from file name
        throttle = int(command[5:8])
        throttle = throttle / 255.0
        if command[4] == "1":
            throttle *= -1

        twist.linear.x = throttle
        twist.angular.z = angle
        control_pub.publish(twist)

        # print("Command: " + str(command) + "  |  Throttle: " + str(throttle) + "  |  Angle: " + str(angle))

    control_socket.close()


# Thread to send images to networked computer
# Source: https://gist.github.com/kittinan/e7ecefddda5616eab2765fdb2affed1b
def image_streamer(data):
    global camera_socket
    global camera_bridge
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    try:
        # Get camera input
        image = camera_bridge.imgmsg_to_cv2(data, "bgr8")
        image = cv2.resize(image, (320, 240))
        result, image = cv2.imencode('.jpg', image, encode_param)
        data = pickle.dumps(image, 0)
        size = len(data)
        camera_socket.sendall(struct.pack(">L", size) + data)

    except CvBridgeError as e:
        print(e)


def main(args):
    global camera_bridge
    global camera_socket
    global quit_request

    quit_request = 0

    # Create ROS node
    rospy.init_node('autonomous_controller', disable_signals=True)

    # Start receiving commands
    command_thread = Thread(target=command_stream)
    command_thread.start()

    # Create socket for camera stream
    camera_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    camera_socket.connect(('192.168.0.102', 8000))
    connection = camera_socket.makefile('wb')
    print("connected")

    # Start camera subscriber
    camera_bridge = CvBridge()
    camera_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, image_streamer)
    print("Started subscriber")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down camera")
    cv2.destroyAllWindows()
    quit_request = 1
    camera_socket.close()


if __name__ == '__main__':
    main(sys.argv)
