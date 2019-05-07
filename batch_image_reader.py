import numpy as np
import glob
import cv2
import math

folder_name = "training_images_2/"
file_list = []
test_proportion = 0.2

# Read all images
pos = 0
neg = 0
# Read all images from folder
for filename in glob.glob(folder_name + "*.jpg"):
    file_list.append(filename)  # Add image files to list
    command = filename[len(folder_name) + 24:-4]    # Extract the command from the file name
    angle = int(command[1:4])
    angle = angle * 180 / 255
    if command[0] == "1":
        angle *= -1

    if angle < 0:
        neg += 1
    else:
        pos += 1

# Print a count of how many left and right steering images there are
print("Pos: " + str(pos))
print("Neg: " + str(neg))

train_files = file_list[:int(len(file_list)*0.8)]
test_files = file_list[int(len(file_list)*0.2):]

print("Finished loading files")


# Returns the number of training images available
def number_of_training_images():
    return len(train_files)


# Read images from files into batches for training
def read_train_images(batch_size=-1, flip_rate=0.5, throttle_mode=0, zero_drop_pct=0.4):

    if batch_size < 0:
        batch_size = len(train_files)

    # Create an empty array to hold batch of images
    image_batch = np.zeros((batch_size, 240, 320, 3), dtype=np.float)
    output_batch = np.zeros(batch_size)

    file_count = len(train_files)
    while True:
        i = 0   # Tracks current batch size
        while i < batch_size:
            if i >= batch_size:
                # Break loop and yield array if it has reached the batch size
                break
            # Generate random number indicating which image to read
            file_id = np.random.randint(0, file_count)

            # Read image
            image = cv2.imread(train_files[file_id])

            # Read the drive command and extract steering angle input
            command = train_files[file_id][len(folder_name) + 24:-4]

            if command == "":
                # Skip image if command can't be read
                print("Null command")
                continue

            # If we're training model to predict throttle inputs
            if throttle_mode:
                # Get command from file name
                throttle = int(command[5:8])

                # Check if reverse
                if command[4] == "1":
                    throttle *= -1

                # Flip images randomly
                if np.random.random_sample() <= flip_rate:
                    image = cv2.flip(image, 1)

                if throttle < 1:
                    # Drop frames where there is no throttle input
                    continue
                output = throttle
                output = ((output + 255.0) / 510.0) * 2.0 - 1   # Scale throttle appropriately

            # If we're training model to predict steering inputs
            else:
                # Get command from file
                angle = int(command[1:4])

                # Set steering direction (left or right)
                if command[0] == "1":
                    angle *= -1

                # Flip images randomly
                if np.random.random_sample() <= flip_rate:
                    image = cv2.flip(image, 1)
                    angle = angle * -1.0

                if angle == 0:
                    # Drop some zero angle data
                    if np.random.random_sample() <= zero_drop_pct:
                        continue

                output = angle
                output = ((output + 300) / 600) * 2.0 - 1   # Scale steering input

            # Add image to batch
            image_batch[i] = image

            # Add command to batch
            output_batch[i] = output
            i += 1

        # Yield the batch of images and commands to callin
        yield image_batch, output_batch


# Read images from files into batches to test the trained neural network
# Same as previous function but no image augmentation is needed for test data
def read_test_images(batch_size=-1, throttle_mode=0):

    if batch_size < 0:
        batch_size = len(test_files)

    # Create an empty array to hold batch of images
    image_batch = np.zeros((batch_size, 240, 320, 3), dtype=np.float)
    output_batch = np.zeros(batch_size)

    file_count = len(test_files)
    image_number = 0
    while True:
        i = 0
        while i < batch_size:
            if i >= batch_size:
                break
            image = cv2.imread(test_files[image_number])
            command = test_files[image_number][len(folder_name) + 24:-4]

            if throttle_mode:
                throttle = int(command[5:8])

                if command[4] == "1":
                    throttle *= -1

                output = throttle
                output = ((output + 255.0) / 510.0) * 2.0 - 1
            else:
                angle = int(command[1:4])

                if command[0] == "1":
                    angle *= -1

                output = angle
                output = ((output + 300) / 600) * 2.0 - 1

            image_batch[i] = image
            output_batch[i] = output
            i += 1
            image_number += 1
            if image_number >= file_count:
                image_number = 0
        yield image_batch, output_batch


# Uncomment to test the functions:
# generator = read_train_images(10, 0, throttle_mode=0)
# neg = 0
# pos = 0
# zero = 0
#
# for images, angles in generator:
#     for pic, im_angle in zip(images, angles):
#         # cv2.imshow('a', pic/255.0)
#         # print(im_angle)
#         if im_angle < 0:
#             neg += 1
#         elif im_angle > 0:
#             pos += 1
#         else:
#             zero += 1
#         # print(pos/(pos+neg))
#         print("Pos: " + str(pos) + "  |  Neg: " + str(neg) + "  |  Zero: " + str(zero))
#         # key = cv2.waitKey(10)
#         # if key == 113:
#         #     break
