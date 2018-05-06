import pickle
import tensorflow as tf
#` TODO: import Keras layers you need here
from sklearn.utils import shuffle

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")
flags.DEFINE_string('n_classes', '', "Number of classes (10 for cifar, 43 for traffic")


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic
    EPOCHS = 20
    rate = 0.001
    BATCH_SIZE = 128
    n_classes = int(FLAGS.n_classes)

    n_features = X_train.shape[-1]
    mu = 0
    sigma = 0.1
    x = tf.placeholder(tf.float32, (None, 1, 1, n_features))
    reshape_x = tf.reshape(x, [-1, n_features])
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, n_classes)


    fc1_W = tf.Variable(tf.truncated_normal(shape = (n_features, 512), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(512))
    fc1 = tf.matmul(reshape_x, fc1_W) + fc1_b
    fc1 = tf.nn.relu(fc1)

    fc2_W = tf.Variable(tf.truncated_normal(shape = (512, 256), mean = mu, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros(256))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b
    fc2 = tf.nn.relu(fc2)
    
    fc3_W = tf.Variable(tf.truncated_normal(shape = (256, n_classes), mean = mu, stddev = sigma))
    fc3_b = tf.Variable(tf.zeros(n_classes))
    fc3 = tf.matmul(fc2, fc3_W) + fc3_b

    logits = fc3

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = rate, name="adam_optim")
    training_operation = optimizer.minimize(loss_operation)


    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(tf.reshape(one_hot_y, [-1, n_classes]), 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def evaluate(X_data, y_data):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
            accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples
    
	# T: train your model here
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        num_examples = len(X_train)

        print("Training...")
        print()
        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
                

            print("EPOCH {} ...".format(i+1))
            train_accuracy = evaluate(X_train, y_train)
            validation_accuracy = evaluate(X_val, y_val)

            print("Train Accuracy = {:.3f}; Validation Accuracy = {:.3f}".format(train_accuracy, validation_accuracy))
            print()


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
