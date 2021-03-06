from batch_image_reader import read_test_images, read_train_images, number_of_training_images
from model import steering_network_model_2
import json
import matplotlib.pyplot as plt

throttle_mode = 0
plots_only = 1
cnn_model = steering_network_model_2()
batch_size = 10

train_generator = read_train_images(batch_size, 0, throttle_mode=throttle_mode)
test_generator = read_test_images(1000, throttle_mode=throttle_mode)
test_images, test_angles = next(test_generator)

train_image_count = number_of_training_images()

print("Starting training")
epochs = 100  # 100

# Create empty list to hold history
full_history = {'val_loss': list(), 'val_acc': list(), 'loss': list(), 'acc': list()}

for e in range(epochs):
    if plots_only:
        break

    print("Actual Epochs: " + str(e) + "/" + str(epochs))
    history = cnn_model.fit_generator(train_generator, validation_data=(test_images, test_angles),
                                      steps_per_epoch=train_image_count / batch_size, epochs=1, verbose=1)
    print("Saving Model")

    # Append new history to total history
    full_history['val_loss'].extend(history.history['val_loss'])
    full_history['val_acc'].extend(history.history['val_acc'])
    full_history['loss'].extend(history.history['loss'])
    full_history['acc'].extend(history.history['acc'])

    if throttle_mode:
        # Save model weights
        cnn_model.save("throttle_model_" + str(e) + ".h5")
        # Save stats for current epoch
        json.dump(history.history, open("throttle_history_" + str(e) + ".json", 'w'))
        # Update overall performance history
        json.dump(full_history, open("throttle_history_full.json", 'w'))

        print("Model saved as " + "throttle_model_" + str(e) + ".h5")
    else:
        # Save model weights
        cnn_model.save("steer_model_" + str(e) + ".h5")
        # Save stats for current epoch
        json.dump(history.history, open("steer_history_" + str(e) + ".json", 'w'))
        # Update overall performance history
        json.dump(full_history, open("steer_history_full.json", 'w'))

        print("Model saved as " + "steer_model_" + str(e) + ".h5")


print("Finished training")

if throttle_mode:
    history_dict = json.load(open('throttle_history_full.json', 'r'))
else:
    history_dict = json.load(open('steer_history_full.json', 'r'))

# list all data in history
print(history_dict.keys())
# summarize history for accuracy
plt.plot(history_dict['acc'])
#plt.plot(history_dict_2['acc'])
plt.plot(history_dict['val_acc'])
#plt.plot(history_dict_2['val_acc'])
plt.title('Model Accuracy Evolution')
plt.ylabel('accuracy')
plt.xlabel('epoch')
#plt.legend(['1000 batch - training', '10 batch - training', '1000 batch - testing', '10 batch - testing'], loc='lower right')
plt.legend(['10 batch - training', '10 batch - testing'], loc='lower right')
plt.show()

# summarize history for loss
plt.plot(history_dict['loss'])
#plt.plot(history_dict_2['loss'])
plt.plot(history_dict['val_loss'])
#plt.plot(history_dict_2['val_loss'])
plt.title('Model Loss Evolution')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.legend(['1000 batch - training', '10 batch - training', '1000 batch - testing', '10 batch - testing'], loc='upper right')
plt.legend(['10 batch - training', '10 batch - testing'], loc='upper right')

plt.show()
