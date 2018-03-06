from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from sklearn.model_selection import train_test_split

from keras.initializers import RandomNormal
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.pooling import MaxPooling2D

from keras.models import Sequential
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image


# Define few paths
data_dir = './data'
train_data_dir = './data/train'
test_data_dir = './data/test'


batch_size = 32
dim_x = 224
dim_y = 224
n_channels = 3
n_classes = 120
n_samples = 10222
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'data/train',  # this is the target directory
        target_size=(dim_x, dim_y),  # all images will be resized
        batch_size=batch_size,
        class_mode='categorical')

filenames = train_generator.filenames
nb_samples = len(filenames)
print(nb_samples)


def get_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(dim_x, dim_y, n_channels))
    x = base_model.output
    x = Flatten()(x)

    x = Dense(units=256,
              kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
              bias_initializer='zeros',
              kernel_regularizer=l2(0.001),
              bias_regularizer=None)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(rate=0.2)(x)

    x = Dense(units=256,
              kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
              bias_initializer='zeros',
              kernel_regularizer=l2(0.001),
              bias_regularizer=None)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(rate=0.3)(x)

    x = Dense(units=n_classes,
              kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
              bias_initializer='zeros',
              kernel_regularizer=l2(0.001),
              bias_regularizer=None)(x)
    predictions = Activation('softmax')(x)

    # create graph of new model
    model = Model(inputs=base_model.input, outputs=predictions)

    # freeze all convolutional base model layers
    for layer in base_model.layers:
        layer.trainable = False

    return model


# get the model
model = get_model()
# define the checkpoint
file_path = "best_model.h5"
checkpoint = ModelCheckpoint(file_path, monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# compile the model
opt = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)


model.fit_generator(
        train_generator,
        steps_per_epoch=n_samples // batch_size + 1,
        epochs=20,
        callbacks=callbacks_list,
        verbose=2)
