# load dogs vs cats dataset, reshape and save to a new file
from os import listdir
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

# divide images into 2 sets
# https://stackoverflow.com/questions/60130918/how-to-split-images-into-test-and-train-set-using-my-own-data-in-tensorflow
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#flow_from_directory
# tutaj jest nawet przykład z klasami "dogs" i "cats"
image_classes=["cats", "dogs"]
image_generator = ImageDataGenerator(validation_split=0.2)
train_data_gen = image_generator.flow_from_directory(directory='train', classes=image_classes, 

                                                     subset='training')
val_data_gen = image_generator.flow_from_directory(directory='train', classes=image_classes, 
                                                   subset='validation')
# define location of dataset
# folder = 'train/'
# photos, labels = list(), list()
# # enumerate files in the directory
# for file in listdir(folder):
#  # determine class
#  output = 0.0
#  if file.startswith('dog'):
#  output = 1.0
#  # load image
#  photo = load_img(folder + file, target_size=(200, 200))
#  # convert to numpy array
#  photo = img_to_array(photo)
#  # store
#  photos.append(photo)
#  labels.append(output)
# # convert to a numpy arrays
# photos = asarray(photos)
# labels = asarray(labels)
# print(photos.shape, labels.shape)
# # save the reshaped photos
# save('dogs_vs_cats_photos.npy', photos)
# save('dogs_vs_cats_labels.npy', labels)