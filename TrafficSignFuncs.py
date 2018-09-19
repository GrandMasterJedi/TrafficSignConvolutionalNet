def gray(images):
    """
    Convert all images from RGB to gray using average of R, G, B values
    images shape: (n x w x l x 1)
    output shape: (n x w x l x 1)
    """
    gray = np.average(images, axis=3)
    gray = np.expand_dims(gray, axis=3)
    return gray

def normalize(images, m=128, s=128):
    """
    Normalize each image in the set images
    """
    #X = images.astype('float32')
    X = np.array(images, dtype=np.float32)
    
    out = (X - m) / s
    return out

def preprocess(images, grayscale=True):
    """
    """
    if grayscale:
        images = gray(images)
        
    nimages = normalize(images)
    return nimages


### Model Architecture
def MyNet1(x, nclasses, nchannel):
    """
    Architectire of the neural network,
    Augment the LeNet architecture with additional layer and allow nchannel as input.
    Logit is array of nchannel
    """
    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    # hyperparameters for weight initialization
    mu = 0
    sigma = 0.1
    
    ###############################
    # Layer 1: Convolutional. Input = 32x32xnchannel. Output = 28x28x9.
    # (32- 5 +1)/1 = 28 
    # Modification, allow possibility for 3 channel RGB data 
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, nchannel, 9), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(9))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x9. Output = 14x14x9
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    ###############################
    # Layer 2: Convolutional.  Input =  14x14x9. Output = 12x12x27.
    # (14-3+1)/1 = 12
    conv2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 9, 27), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(27))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # Activation.
    conv2 = tf.nn.relu(conv2)

    ###############################
    # Add: Layer 3. Convolution, input 12x12x27, output:  12-4+1 = 9x9x81
    conv3_W = tf.Variable(tf.truncated_normal(shape=(4, 4, 27, 81), mean = mu, stddev = sigma))
    conv3_b = tf.Variable(tf.zeros(81))
    conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_b
    
    # Add Activation
    conv3 = tf.nn.relu(conv3)
    
    # Add max Pooling: input = 9x9x81, output = (3x3x81) 
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='VALID')
    
    ###############################
    # Flatten. Input =3x3x81. Output = 729
    fc0   = flatten(conv3)
    
    fcdim = 729
    nflattenlayer = 1
    while (fcdim > (3*nclasses)):     
        # Layer 3+: Fully Connected. Input = 729. Output = nclasses.
        fcdim2 = int(fcdim/2.5) 
        fc1_W = tf.Variable(tf.truncated_normal(shape=(fcdim, fcdim2), mean = mu, stddev = sigma))
        fc1_b = tf.Variable(tf.zeros(fcdim2))
        fc0   = tf.matmul(fc0, fc1_W) + fc1_b
        # Activation
        fc0    = tf.nn.relu(fc0)
        fcdim =  fcdim2 
        nflattenlayer = nflattenlayer +1 

#     # Layer 4: Fully Connected. Input = 729. Output = 290.
#     fc1_W = tf.Variable(tf.truncated_normal(shape=(729, 290), mean = mu, stddev = sigma))
#     fc1_b = tf.Variable(tf.zeros(290))
#     fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
#     # Activation.
#     fc1    = tf.nn.relu(fc1)

#     # Layer 5: Fully Connected. Input = 290. Output = 110.
#     fc2_W  = tf.Variable(tf.truncated_normal(shape=(290, 110), mean = mu, stddev = sigma))
#     fc2_b  = tf.Variable(tf.zeros(84))
#     fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
#     # Activation.
#     fc2    = tf.nn.relu(fc2)
        
    # Last 6: Fully Connected. Input = 110. Output = nClasses
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(fcdim, nclasses), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(nclasses))
    logits = tf.matmul(fc0, fc3_W) + fc3_b
    print( "Number of full connections layer is ", nflattenlayer)
        
    
    return logits


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples