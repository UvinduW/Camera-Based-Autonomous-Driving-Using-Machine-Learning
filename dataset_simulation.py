from model import steering_network_model_2, steering_network_model_2
import numpy as np
import glob
import cv2

# Load steering angle predicting network
steering_model = steering_network_model_2()
steering_model.load_weights("Test_1/steer_model_30.h5")

# Load throttle predicting network
throttle_model = steering_network_model_2()
throttle_model.load_weights("throttle_model_99.h5")

# Load graphics and initialise variables for steering animation
steering_wheel_predicted = cv2.imread('steering_wheel_image.jpg', 0)
steering_wheel_actual = steering_wheel_predicted
wheel_rows, wheel_cols = steering_wheel_predicted.shape
smoothed_angle = 0
smoothed_angle_actual = 0

image_rows = 240
image_cols = 320
vis_scaling_factor = 1

folder_name = "training_images_2/"  # "training_images/"
file_list = []
test_proportion = 0.2

# Read all images
image = np.empty([1, 240, 320, 3])
for filename in glob.glob(folder_name + "*.jpg"):
    image[0] = cv2.imread(filename)

    # Predict angle
    predicted_angle = (steering_model.predict(image)[0] + 1)*600/2 - 300
    # Predict throttle
    # predicted_throttle = ((throttle_model.predict(image)[0] + 1)*510/2 - 255) * 255
    predicted_throttle = (throttle_model.predict(image)[0]) * 255

    # Smooth angle for animated steering wheel
    smoothed_angle = predicted_angle# += 0.2 * pow(abs((predicted_angle - smoothed_angle)), 2.0 / 3.0) * (predicted_angle - smoothed_angle) / abs(predicted_angle - smoothed_angle)
    rotation_matrix_predicted = cv2.getRotationMatrix2D((wheel_cols / 2, wheel_rows / 2), -smoothed_angle, 1)
    modified_wheel_predicted = cv2.warpAffine(steering_wheel_predicted, rotation_matrix_predicted, (wheel_cols, wheel_rows))
    cv2.putText(modified_wheel_predicted, "Predicted: " + str(round(int(predicted_angle), 2)), (int(wheel_cols / 2 - 90), 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

    # Get actual angle from file name
    command = filename[len(folder_name) + 24:-4]
    angle = int(command[1:4])
    #angle = angle * 180 / 255
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
    smoothed_angle_actual = angle# += 0.2 * pow(abs((angle - smoothed_angle_actual)), 2.0 / 3.0) * (angle - smoothed_angle_actual) / abs(angle - smoothed_angle_actual)
    rotation_matrix_actual = cv2.getRotationMatrix2D((wheel_cols / 2, wheel_rows / 2), -smoothed_angle_actual, 1)
    modified_wheel_actual = cv2.warpAffine(steering_wheel_actual, rotation_matrix_actual, (wheel_cols, wheel_rows))
    cv2.putText(modified_wheel_actual, "Actual: " + str(round(angle, 0)), (int(wheel_cols / 2 - 70), 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255),
                lineType=cv2.LINE_AA)

    # Combine the two images
    # First, process predicted wheel
    wheel_predicted = np.zeros((image[0, :, :, :].shape[0], image_rows, 3))
    modified_wheel_predicted = cv2.resize(modified_wheel_predicted, (image_rows, image_rows))
    wheel_predicted[0:modified_wheel_predicted.shape[0], 0:modified_wheel_predicted.shape[1], 0] = modified_wheel_predicted
    wheel_predicted[0:modified_wheel_predicted.shape[0], 0:modified_wheel_predicted.shape[1], 1] = modified_wheel_predicted
    wheel_predicted[0:modified_wheel_predicted.shape[0], 0:modified_wheel_predicted.shape[1], 2] = modified_wheel_predicted

    # Next process actual wheel
    wheel_actual = np.copy(wheel_predicted)
    modified_wheel_actual = cv2.resize(modified_wheel_actual, (image_rows, image_rows))
    wheel_actual[0:modified_wheel_actual.shape[0], 0:modified_wheel_actual.shape[1], 0] = modified_wheel_actual
    wheel_actual[0:modified_wheel_actual.shape[0], 0:modified_wheel_actual.shape[1], 1] = modified_wheel_actual
    wheel_actual[0:modified_wheel_actual.shape[0], 0:modified_wheel_actual.shape[1], 2] = modified_wheel_actual

    # Draw progress bar to represent throttle input
    top_left_x = 20
    top_left_y = 150
    bar_width = 10
    bar_height = 80
    # Actual throttle
    # Border rectangle
    cv2.rectangle(wheel_actual, (top_left_x, top_left_y), (top_left_x + bar_width, top_left_y + bar_height), (0, 255, 0), 3)
    # Fill rectangle
    cv2.rectangle(wheel_actual, (top_left_x, top_left_y + int((255-throttle)*bar_height/255)), (top_left_x + bar_width, top_left_y + bar_height),
                  (0, 255, 0), -1)

    # Predicted throttle
    cv2.rectangle(wheel_predicted, (top_left_x, top_left_y), (top_left_x + bar_width, top_left_y + bar_height), (0, 255, 0), 3)
    # Fill rectangle
    cv2.rectangle(wheel_predicted, (top_left_x, top_left_y + int((255-predicted_throttle)*bar_height/255)), (top_left_x + bar_width, top_left_y + bar_height),
                  (0, 255, 0), -1)

    # Now combine the two wheels and the footage
    visualisation = np.concatenate((image[0, :, :, :], wheel_predicted, wheel_actual), axis=1)

    vis_rows = visualisation.shape[0]
    vis_cols = visualisation.shape[1]

    visualisation = cv2.resize(visualisation, (int(vis_cols * vis_scaling_factor), int(vis_rows * vis_scaling_factor)))
    #vis_scaling_factor = 0

    # Display the visualisation
    cv2.imshow("Driving Simulation", visualisation/255)

    key = cv2.waitKey(1)
    if key == 113:
        break
    elif key == 43:
        vis_scaling_factor += 0.2

    elif key == 45:
        vis_scaling_factor -= 0.2

    elif key == -1:
        pass

    else:
        print(key)
