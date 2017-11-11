import tensorflow as tf

conv_section = tf.keras.applications.VGG16(include_top=False, weights=None)
#conv_section.summary()
#
#conv_section.layers.pop()
#conv_section.layers.pop()
#conv_section.layers.pop()
#conv_section.summary()


# Can't pop because you can't rename the layers and the model expects them

# Can't start without top because you can't fucking set the Dense layer input_size

keras_vgg16 = tf.keras.models.Sequential()
keras_vgg16.add(conv_section)
keras_vgg16.add(tf.keras.layers.Flatten())
keras_vgg16.add(tf.keras.layers.Dense(256, activation="relu", input_shape=(None, 25088), name="fc1"))
keras_vgg16.add(tf.keras.layers.Dense(64, activation="relu", name="fc2"))
keras_vgg16.add(tf.keras.layers.Dense(9, activation="linear", name="predictions"))
keras_vgg16.summary()

keras_vgg16.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                      loss='mean_squared_error',
                      metric='accuracy')


