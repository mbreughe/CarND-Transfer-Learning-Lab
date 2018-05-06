from keras.datasets import cifar10
from lenet import LeNet
from sklearn.utils import shuffle

import tensorflow as tf

def preprocess(x):
    scaled = ((x - 128.0) / 128.0)
    return scaled
if __name__ == "__main__":

    # Load the data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # y_train.shape is 2d, (50000, 1). While Keras is smart enough to handle this
    # it's a good idea to flatten the array.
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    from sklearn.model_selection import train_test_split
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=42, stratify = y_train)

    
    # Hyper params
    n_classes = 10
    rate = 0.001
    EPOCHS = 10
    BATCH_SIZE=128

    # Place holders for input and output
    keep_prob = tf.placeholder(tf.float32, name = "keep_prob")
    x = tf.placeholder(tf.float32, (None, 32, 32, 3), name = "input")
    y = tf.placeholder(tf.int32, (None), name = "labels")
    one_hot_y = tf.one_hot(y, n_classes)

    # Network architecture and optimizer
    logits = LeNet(preprocess(x), keep_prob, n_classes)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = rate, name="adam_optim")
    training_operation = optimizer.minimize(loss_operation)


    # Functions to calculate accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Run inference
    def evaluate(X_data, y_data):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
            accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples

    # Run training
    def train_model(X_train, y_train, X_valid, y_valid):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
                
            num_examples = len(X_train)

            #print([node.name for node in tf.get_default_graph().as_graph_def().node])
            
            print("Training...")
            print()
            for i in range(EPOCHS):
                X_train, y_train = shuffle(X_train, y_train)
                for offset in range(0, num_examples, BATCH_SIZE):
                    end = offset + BATCH_SIZE
                    batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                    sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

                print("EPOCH {} ...".format(i+1))
                train_accuracy = evaluate(X_train, y_train)
                validation_accuracy = evaluate(X_valid, y_valid)

                print("Train Accuracy = {:.3f}; Validation Accuracy = {:.3f}".format(train_accuracy, validation_accuracy))
                print()
        
    train_model(X_train, y_train, X_valid, y_valid)
