from batch_image_reader import read_test_images, read_train_images, number_of_training_images
from model import network_model
import json
import matplotlib.pyplot as plt

throttle_mode = 1
plots_only = 0
cnn_model = network_model()
batch_size = 10


train_generator = read_train_images(batch_size, 0.9, throttle_mode=throttle_mode)
test_generator = read_test_images(1000, throttle_mode=throttle_mode)
test_images, test_angles = next(test_generator)

train_image_count = number_of_training_images()

print("Starting training")
epochs = 40  # 100

# Create empty list to hold history
full_history = {'val_loss': list(), 'val_acc': list(), 'loss': list(), 'acc': list()}

for e in range(epochs):
    if plots_only:
        break

    print("(Batch 1000) Actual Epochs: " + str(e) + "/" + str(epochs))
    history = cnn_model.fit_generator(train_generator, validation_data=(test_images, test_angles),
                                      steps_per_epoch=train_image_count / batch_size, epochs=1, verbose=1)
    print("Saving Model")

    # Append new history to total history
    full_history['val_loss'].extend(history.history['val_loss'])
    full_history['val_acc'].extend(history.history['val_acc'])
    full_history['loss'].extend(history.history['loss'])
    full_history['acc'].extend(history.history['acc'])

    # Save model weights
    cnn_model.save("drive_model_" + str(e) + ".h5")

    # Save stats for current epoch
    json.dump(history.history, open("history_"+str(e)+".json", 'w'))

    # Update overall performance history
    json.dump(full_history, open("history_f.json", 'w'))

    print("Model saved as " + "drive_model_" + str(e) + ".h5")


print("Finished training")

history_dict = json.load(open('history_f.json', 'r'))
history_dict_2 = json.load(open('2history_f.json', 'r'))

# list all data in history
print(history_dict.keys())
# summarize history for accuracy
plt.plot(history_dict['acc'])
plt.plot(history_dict_2['acc'])
plt.plot(history_dict['val_acc'])
plt.plot(history_dict_2['val_acc'])
plt.title('Model Accuracy Evolution')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['1000 batch - training', '10 batch - training', '1000 batch - testing', '10 batch - testing'], loc='lower right')
plt.show()

# summarize history for loss
plt.plot(history_dict['loss'])
plt.plot(history_dict_2['loss'])
plt.plot(history_dict['val_loss'])
plt.plot(history_dict_2['val_loss'])
plt.title('Model Loss Evolution')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['1000 batch - training', '10 batch - training', '1000 batch - testing', '10 batch - testing'], loc='upper right')
plt.show()

