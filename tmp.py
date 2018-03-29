

def LeNet(x, nclasses):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    # hyperparameters for weight initialization
    mu = 0
    sigma = 0.1
    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    # (32- 5 +1)/1 = 28 
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, nclasses), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(nclasses))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits

def NetworkPrediction(X, y, NetworkArchitecture, nclasses, Optimizer, grayscale=True):
    """
    Fit the neural network defined in NetworkArchitecture and output prediction accu
    Optimizer = tf.train.AdamOptimizer(learning_rate = learnrate)
    Optimizer = tf.train.GradientDescentOptimizer(learning_rate=learnrate) 

    """
    
    logits = NetworkArchitecture(X, nclasses, grayscale) 
    
    # Make output classes categorical
    one_hot_y = tf.one_hot(y, n_classes)
    
    # Loss function
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_y, logits=logits)
    loss = tf.reduce_mean(cross_entropy)
    
    # Train data with chosen Optimization algorithm  
    # This runs the optimization
    #Optimizer = tf.train.AdamOptimizer(learning_rate = learnrate)
    training = Optimizer.minimize(loss)

    #Returns the index with the largest value across axes of a tensor. 
    predicted = tf.argmax(logits, 1)
    actual = tf.argmax(one_hot_y, 1)
    
    correct_prediction = tf.equal(predicted, actual)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return logits, training, predicted, actual, accuracy



def evaluateAccuracy(x,y,x_data, y_data, accuracy_operation):
    """
    ev
    """
    num_examples = len(x_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = x_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

def TrainModel(x_train, y_train, x_valid, y_valid,  Architecture, Optimizer, grayscale = True, SAVE=True):
    """
    launch the tensor flow instance and call the optimizer
    """
    
    
        
    nchannel = 1 if grayscale else 3
        
    x = tf.placeholder(tf.float32, (None, img_size, img_size, nchannel))
    y = tf.placeholder(tf.int32, (None))
    #Optimizer = tf.train.AdamOptimizer(learning_rate = learnrate)
    
    #with tf.Graph().as_default(), tf.Session() as sess:

    logits, training, predicted, actual, accuracy = NetworkPrediction(x, y, Architecture, n_classes, 
                                                                          Optimizer, grayscale=grayscale)
      
    with tf.Session() as sess:


        # Variable initialization
        init = tf.global_variables_initializer()
        sess.run(init)

        
        saver = tf.train.Saver()
        
        #  training and validation accuracy over epochs, like such:
        accuracy_history = []

        # Record time elapsed for performance check
        last_time = time.time()
        train_start_time = time.time()

        num_examples = len(x_train)

        for i in range(EPOCHS):
            x_train, y_train = shuffle(x_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = x_train[offset:end], y_train[offset:end]
                sess.run(training, feed_dict={x: batch_x, y: batch_y})

            validation_accuracy = evaluateAccuracy(x,y,x_valid, y_valid, accuracy)
            accuracy_history.append(validation_accuracy)
            print("EPOCH {} ...".format(i+1))
            print("Validation Accuracy: %.4f, Elapsed time: %.2f sec" % (validation_accuracy, time.time()-last_time))
            print()
            last_time = time.time()

        total_time = time.time() - train_start_time
        print('Total elapsed time: %.2f sec (%.2f min)' % (total_time, total_time/60))


        if SAVE:
            save_path = saver.save(sess, MODEL_PATH)
            print('Trained model saved at: %s' % save_path)
            # Save accuracy history
            print('Accuracy history saved at accuracy_history.p')
            with open('accuracy_history.p', 'wb') as f:
                pickle.dump(accuracy_history, f)

    return saver, accuracy_history, logits, training, predicted, actual, accuracy, x, y 


