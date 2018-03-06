from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

#load weights
model.load_weights("best_model.h5")

# compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


# few parameters
dim_x = 224
dim_y = 224
batch_size = 32
n_samples = 10357

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(dim_x, dim_y),
        batch_size=batch_size,
        class_mode=None)

filenames = test_generator.filenames
nb_samples = len(filenames)
print(nb_samples)


predictions = model.predict_generator(test_generator, steps=(n_samples // batch_size) + 1, verbose=1)


print(predictions.shape)
# save the output as a Numpy array
np.save(open('predictions.npy', 'w'), predictions)

