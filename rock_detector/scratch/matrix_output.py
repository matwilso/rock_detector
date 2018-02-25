import tensorflow as tf

# NEURAL NETWORK DEFINITION
# (tf.keras VGG16, random weights not trained on imagenet, custom fc head layers)
#conv_section = tf.keras.applications.VGG16(include_top=False, weights=None, input_shape=(224,224,3))
#
#keras_vgg16 = tf.keras.models.Sequential()
#keras_vgg16.add(conv_section)
#keras_vgg16.add(tf.keras.layers.Flatten())
##keras_vgg16.add(tf.keras.layers.Dense(256, activation='relu', input_shape=(None, 512*7*7), name='fc1'))
##keras_vgg16.add(tf.keras.layers.Dense(64, activation='relu', name='fc2'))
##keras_vgg16.add(tf.keras.layers.Dense(9, activation='linear', name='predictions'))
### Print summary on architecture
#keras_vgg16.summary()
##
### Input image (224, 224, 3)
#img_input = keras_vgg16.input
### Output vector of rock xy+height (9,)
##pred_tf = keras_vgg16.output


conv_section = tf.keras.applications.ResNet50(include_top=False, weights=None, input_shape=(224,224,3))
conv_section.layers.pop()
reshape = tf.keras.layers.Reshape((392, 256))
sigmoid = tf.keras.layers.Activation('sigmoid')
conv_out = sigmoid(reshape(conv_section.layers[-1].output))

resnet50 = tf.keras.models.Model(conv_section.input, conv_out)

#resnet50 = tf.keras.models.Sequential()
#resnet50.add(conv_section)
#resnet50.add(tf.keras.layers.Flatten())
resnet50.summary()


conv_section = tf.keras.applications.VGG16(include_top=False, weights=None, input_shape=(224,224,3))
conv_section.layers.pop()
reshape = tf.keras.layers.Reshape((392, 256))
conv_out = sigmoid(reshape(conv_section.layers[-1].output))
vgg16 = tf.keras.models.Model(conv_section.input, conv_out)
vgg16.summary()
import ipdb; ipdb.set_trace()


