from model import network_model
import numpy as np
import glob
import cv2

cnn_model = network_model()
cnn_model.load_weights("drive_model_44.h5")
img = cv2.imread('steering_wheel_image.jpg',0)
img_actual = cv2.imread('steering_wheel_image.jpg',0)
rows,cols = img.shape
smoothed_angle = 0
smoothed_angle_actual = 0

folder_name = "training_images/"
file_list = []
test_proportion = 0.2

# Read all images
image = np.empty([1, 240, 320, 3])
for filename in glob.glob(folder_name + "*.jpg"):
    image[0] = cv2.imread(filename)

    cv2.imshow("Driving Footage", image[0,-150:, :, :]/255)
    # Predict angle
    degrees = cnn_model.predict(image)[0] * 180.0

    # Smooth angle for animated steering wheel
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.putText(dst, "Predicted: " + str(round(int(degrees), 2)), (int(cols/2 - 90), 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
    # Get actual angle from file name
    command = filename[len(folder_name) + 24:48]
    angle = int(command[1:4])
    angle = angle * 180 / 255
    if command[0] == "1":
        angle *= -1
    if angle == 0:
        # Avoid div by zero
        angle = 0.1

    # Smooth angle for animated steering wheel
    smoothed_angle_actual += 0.2 * pow(abs((angle - smoothed_angle_actual)), 2.0 / 3.0) * (angle - smoothed_angle_actual) / abs(angle - smoothed_angle_actual)
    M_actual = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle_actual,1)
    dst_actual = cv2.warpAffine(img_actual,M_actual,(cols,rows))
    cv2.putText(dst_actual, "Actual: " + str(round(angle, 0)), (int(cols / 2 - 70), 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255),
                lineType=cv2.LINE_AA)
    # Display the two steering wheels
    cv2.imshow("Predicted steering wheel", dst)
    cv2.imshow("Actual steering wheel", dst_actual)

    cv2.waitKey(10)