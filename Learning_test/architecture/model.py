from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense

class VGGnetSmall():
    def __init__(self, input_shape, num_classes, final_activation):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.final_activation = final_activation

    def build(self):
        model = Sequential()
        model.add(Conv2D(filters=6, kernel_size=(3, 3), input_shape=self.input_shape,
                     activation='relu'))
        model.add(MaxPooling2D())
        model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=16, activation='relu'))
        model.add(Dense(units=self.num_classes, activation=self.final_activation))

        # model = Sequential()
        # model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=self.input_shape, activation='relu'))
        # model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        # model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        # model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        # model.add(Flatten())
        # model.add(Dense(units=1024, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(units=self.num_classes, activation=self.final_activation))

        return model