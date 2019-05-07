from model import steering_network_model_2
import numpy as np
import glob
import cv2

image_folder = "training_images/"
test_to_load = 1
model_number = 75

steering_model = steering_network_model_2()
steering_model.load_weights("Test_" + str(test_to_load) + "/steer_model_" + str(model_number) + ".h5")


image = np.empty([1, 240, 320, 3])  # Empty array to hold images
diff_array = np.zeros([401, 2])     # Zero-initialised array to store the differences between
count = 0
for i in range(401):
    diff_array[i, 0] = i-200

# Read all images
for filename in glob.glob(image_folder + "*.jpg"):
    image[0] = cv2.imread(filename)
    # Predict steering angle using model
    predicted_angle = (steering_model.predict(image)[0][0] + 1) * 600 / 2 - 300

    # Get actual steering angle by processing file name
    command = filename[len(image_folder) + 24:-4]
    actual_angle = int(command[1:4])
    if command[0] == "1":
        actual_angle *= -1
    if actual_angle == 0:
        count += 1
        continue
        # actual_angle = 1

    # Find percentage difference between predicted and actual angle
    diff = int(100*(predicted_angle.astype(float) - actual_angle)/actual_angle)

    # Limit max difference to +-200%
    if diff > 200:
        diff = 200
    elif diff < -200:
        diff = -200

    # Update array with values
    diff_array[(diff + 200), 1] += 1

    # Count of images processed
    count += 1

    # Print progress every 200 images and save results to CSV file (interval avoids slowing down the script)
    if count % 200 == 0:
        print("Progress: " + str(round(100 * count / 49008, 2)) + "%  |  Count: " + str(
            count) + "  |  Difference: " + str(diff) + "%  |  Actual: " + str(actual_angle) + "  |  Predicted: " + str(
            int(predicted_angle)))
        np.savetxt("Test_" + str(test_to_load) + "/accuracy_" + str(model_number) + ".csv", diff_array, delimiter=',')

# Save the final results to a CSV file
np.savetxt("Test_" + str(test_to_load) + "/accuracy_" + str(model_number) + ".csv", diff_array, delimiter=',')


