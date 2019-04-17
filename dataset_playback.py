import numpy as np
import glob
import cv2

img_actual = cv2.imread('steering_wheel_image.jpg', 0)
rows, cols = img_actual.shape
smoothed_angle = 0
smoothed_angle_actual = 0

folder_name = "training_images/"
file_list = []
test_proportion = 0.2

# Read all images
image = np.empty([1, 240, 320, 3])
for filename in glob.glob(folder_name + "*.jpg"):
    image[0] = cv2.imread(filename)



    # Get actual angle from file name
    command = filename[len(folder_name) + 24:48]
    angle = int(command[1:4])
    angle = angle * 180 / 255
    if command[0] == "1":
        angle *= -1
    if angle == 0:
        # Avoid div by zero
        angle = 0.1

    # Get throttle input from file name
    throttle = int(command[5:8])
    if command[4] == "1":
        throttle *= -1



    # Smooth angle for animated steering wheel
    smoothed_angle_actual += 0.2 * pow(abs((angle - smoothed_angle_actual)), 2.0 / 3.0) * (
                angle - smoothed_angle_actual) / abs(angle - smoothed_angle_actual)
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle_actual, 1)
    modified_image = cv2.warpAffine(img_actual, rotation_matrix, (cols, rows))
    cv2.putText(modified_image, "Actual: " + str(round(angle, 0)), (int(cols / 2 - 70), 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255),
                lineType=cv2.LINE_AA)

    # Draw progress bar to represent throttle input
    top_left_x = 20
    top_left_y = 100
    bar_width = 20
    bar_height = 100
    # Border rectangle
    cv2.rectangle(image[0, :, :, :], (top_left_x, top_left_y), (top_left_x + bar_width, top_left_y + bar_height), (0, 255, 0), 3)
    # Fill rectangle
    cv2.rectangle(image[0, :, :, :], (top_left_x, top_left_y + int((255-throttle)*bar_height/255)), (top_left_x + bar_width, top_left_y + bar_height),
                  (0, 255, 0), -1)

    print("Command: " + str(command) + "  |  Throttle: " + str((255-throttle)))
    wheel = np.zeros((image[0, :, :, :].shape[0], 240, 3))
    modified_image = cv2.resize(modified_image, (240, 240))
    wheel[0:modified_image.shape[0], 0:modified_image.shape[1], 0] = modified_image
    wheel[0:modified_image.shape[0], 0:modified_image.shape[1], 1] = modified_image
    wheel[0:modified_image.shape[0], 0:modified_image.shape[1], 2] = modified_image
    # Display the footage and steering wheel
    wheel = np.reshape(wheel,(wheel.shape[0], wheel.shape[1], 3))
    visualisation = np.concatenate((image[0, :, :, :], wheel), axis=1)
    cv2.imshow("Driving Footage", visualisation / 255)

    cv2.waitKey(10)
