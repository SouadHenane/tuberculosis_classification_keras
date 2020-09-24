import tensorflow as tf 
from numpy import array
from common import readcsv, BASE_DIR, os

train_path = os.path.join(*[BASE_DIR, 'data', 'train.csv'])
test_path = os.path.join(*[BASE_DIR, 'data', 'test.csv'])

train = readcsv(train_path)[1:]
test = readcsv(test_path)[1:]

x_train = array([row[:-1] for row in train])
y_train = array([row[-1] for row in train])

x_test = array([row[:-1] for row in test])
y_test = array([row[-1] for row in test])

## model
model = tf.keras.models.Sequential()

## hidden layer
model.add(tf.keras.layers.Dense(300, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(50, activation = tf.nn.relu))

## output layer 
model.add(tf.keras.layers.Dense(5, activation = tf.nn.softmax))

## cost
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

## learning
model.fit(x_train, y_train, epochs = 3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print("Loss {} \nAccuracy {}".format(val_loss, val_acc))

model.save_weights('2_weights.h5')