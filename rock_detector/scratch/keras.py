import tensorflow as tf
from utils import *
import matplotlib

conv_section = tf.keras.applications.VGG16(include_top=False, weights=None, input_shape=(224,224,3))

conv_section.summary()

# Can't pop because you can't rename the layers and the model expects them

# Can't start without top because you can't fucking set the Dense layer input_size

keras_vgg16 = tf.keras.models.Sequential()
keras_vgg16.add(conv_section)
keras_vgg16.add(tf.keras.layers.Flatten())
keras_vgg16.add(tf.keras.layers.Dense(256, activation="relu", input_shape=(None, 25088), name="fc1"))
keras_vgg16.add(tf.keras.layers.Dense(64, activation="relu", name="fc2"))
keras_vgg16.add(tf.keras.layers.Dense(9, activation="linear", name="preds"))
keras_vgg16.summary()

#keras_vgg16.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
#                      loss='mean_squared_error',
#                      metric='accuracy')


cam_img =  preproc_image(matplotlib.image.imread("../assets/practice.jpg"))
cam_imgs = cam_img.reshape((1, 224, 224, 3))


predictions = keras_vgg16.output#.name
img_input = keras_vgg16.input#.name

object_positions = np.random.randn(1, 9)
ground_truths = tf.placeholder(tf.float32, shape=(None, 9), name="ground_truths")

# loss (sum of squares)
loss = tf.reduce_sum(tf.square(ground_truths - predictions)) 
# optimizer
optimizer = tf.train.AdamOptimizer(1e-4) # 1e-4 suggested from dom rand paper
train = optimizer.minimize(loss)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

##keras_vgg16.compile()
for i in range(10):
    print(sess.run([train, loss], {img_input : cam_imgs, ground_truths : object_positions}))



#est_vgg16 = tf.keras.estimator.model_to_estimator(keras_model=keras_vgg16)
