# Author: Uvindu Wijesinghe

import pygame
import socket
import cv2
import struct
import pickle
from time import time
from threading import Thread
from time import sleep
import numpy as np
from model import steering_network_model_2

# Global variables
quit_request = 0
autonomous_mode = 0
save_images = 0
image_number = 40000
command = "00000000"
autonomous_angle = 0
autonomous_throttle = 0

# Thread to get commands from gamepad and send gamepad/autonomous commands to ROSbot
def command_streamer():
    global quit_request
    global autonomous_mode
    global save_images
    global command

    # Initialise gamepad
    pygame.init()
    pygame.joystick.init()
    gamepad = pygame.joystick.Joystick(0)
    gamepad.init()

    # Map gamepad axes
    steer_axis = 0
    throttle_axis = 2
    A_button = 0
    B_button = 1
    X_button = 2
    Y_button = 3
    L_button = 4
    R_button = 5
    back_button = 6
    start_button = 7

    # Connect to socket
    local_ip = "0.0.0.0"
    rosbot_ip = "192.168.0.100"
    port = 8001

    # Create UDP socket
    control_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    control_socket.bind((local_ip, port))

    steer_base = 2

    while not quit_request:
        pygame.event.pump()

        # Get throttle
        throttle_input = gamepad.get_axis(throttle_axis)
        # Get steer input
        angle_input = gamepad.get_axis(steer_axis)



        # Check if forwards or reverse
        if throttle_input > 0:
            reverse = 1
        else:
            reverse = 0

        # Form throttle command, scaled between 0 and 255
        throttle = str(int((abs(throttle_input)) * 255))
        throttle = str(throttle).zfill(3)

        # Check if left or right
        if angle_input < 0:
            # left
            direction = 1
        else:
            # right
            direction = 0

        if abs(angle_input) < 0.06:
            # Force centre the stick if its resting position is near centre
            angle_input = 0

        if abs(throttle_input) < 0.01:
            # No steer commands if there is no throttle input
            steer = "000"
        else:
            # More steer at lower speeds and less steer at higher speeds for better control and avoid twitchy behaviour
            steer = int(abs(angle_input) * (steer_base - throttle_input) * 100)  # Max value: 300
            steer = str(steer).zfill(3)

        if autonomous_mode and (throttle_input) < -0.2:
            steer = str(abs(autonomous_angle)).zfill(3)
            if int(autonomous_angle) < 0:
                direction = 1
            else:
                direction = 0

            throttle = str(abs(autonomous_throttle)).zfill(3)
            if int(autonomous_throttle) < 0:
                throttle = 0


        command = str(direction) + steer + str(reverse) + throttle

        # Check gamepad inputs
        if gamepad.get_button(B_button):
            # Send quit request
            control_socket.sendto(str(999).encode(), (rosbot_ip, port))
            quit_request = 1

        # Gamepad "A" button to toggle autonomous mode
        if gamepad.get_button(A_button):
            autonomous_mode = 1 - autonomous_mode
            sleep(0.05)

        # Gamepad "X" button to toggle recording of images to file
        if gamepad.get_button(X_button):
            save_images = 1 - save_images
            sleep(0.05)

        # Send commands to ROSbot over socket
        control_socket.sendto(command.encode(), (rosbot_ip, port))

        sleep(0.1)


# Display the camera feed to the user
def feed_viewer():
    global image
    global quit_request
    global autonomous_mode
    global save_images

    while not quit_request:
        disp_image = np.copy(image) # Make copy of image to avoid altering the original

        # Display text indicating the control command being sent to ROSbot
        cv2.putText(disp_image, command, (200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        # Display text indicating whether images are being stored
        cv2.putText(disp_image, "Store Images: " + str(save_images), (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2, cv2.LINE_AA)
        # Display text indicating whether Autonomous Mode is active
        cv2.putText(disp_image, "Autonomous: " + str(autonomous_mode), (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                    2, cv2.LINE_AA)
        # Show image
        cv2.imshow('ImageWindow', disp_image)
        cv2.waitKey(1)


# Thread to receive images from ROSbot over socket
# Source: https://gist.github.com/kittinan/e7ecefddda5616eab2765fdb2affed1b
def image_stream():
    global quit_request
    global image
    global autonomous_mode

    # Setup and initialise socket
    print("Waiting for connection")
    image_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    local_ip = '0.0.0.0'
    port = 8000
    image_socket.bind((local_ip, port))
    image_socket.listen(10)
    conn, addr = image_socket.accept()

    # Get images from socket
    data = b""
    payload_size = struct.calcsize(">L")
    print("payload_size: {}".format(payload_size))

    first_run = 1

    while not quit_request:

        while len(data) < payload_size:
            data += conn.recv(4096)

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        while len(data) < msg_size:
            data += conn.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]

        frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        image = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        # Start viewer thread since first frame has been loaded
        if first_run:
            viewer_thread = Thread(target=feed_viewer)
            viewer_thread.start()
            first_run = 0

    image_socket.close()


# Thread to generate autonomous commands when mode is activated
def autonomous_commands():
    global autonomous_mode
    global quit_request
    global image
    global autonomous_angle
    global autonomous_throttle

    # Blank array to store image
    predict_image = np.empty([1, 240, 320, 3], dtype=np.float64)

    # Load steering model
    steering_model = steering_network_model_2()
    steering_model.load_weights("steer_model_99.h5")

    # Load throttle model
    throttle_model = steering_network_model_2()
    throttle_model.load_weights("throttle_model_20.h5")

    while not quit_request:
        # Predict steering and throttle if autonomous mode is active
        if autonomous_mode:
            predict_image[0] = image
            autonomous_angle = int((steering_model.predict(predict_image)[0] + 1)*600/2 - 300)

            autonomous_throttle = int((throttle_model.predict(predict_image)[0]) * 255 * 0.4)

            print("Autonomous Angle: " + str(autonomous_angle) + "  |  Autonomous Throttle: " + str(autonomous_throttle))
            sleep(0.01)
        else:
            # Large sleep delay if this thread is not being actively used to preserve system resources
            sleep(0.5)


# Thread to record images to file with command embedded in the file name
def image_recorder():
    global image
    global quit_request
    global save_images
    global image_number
    global command

    while not quit_request:
        if save_images:
            # Save images if save mode enabled
            cv2.imwrite('training_images/frame{:>010}_command-{}.jpg'.format(image_number, command),
                        image)
            # Increment number which is included in file name to ensure duplicate file names are not possible
            image_number += 1
            sleep(0.05)
        else:
            # Large sleep delay if this thread is not being actively used to preserve system resources
            sleep(0.5)


def main():
    # Receives images over socket
    image_thread = Thread(target=image_stream)
    image_thread.start()

    # Generates commands from gamepad and transmits them over socket
    command_thread = Thread(target=command_streamer)
    command_thread.start()

    # Generates commands from images received through neural network
    autonomous_thread = Thread(target=autonomous_commands)
    autonomous_thread.start()

    # Saves images and commands to file to collect training data
    saver_thread = Thread(target=image_recorder)
    saver_thread.start()


if __name__ == '__main__':
    main()
