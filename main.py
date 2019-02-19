#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    
    
#global variables / hyper-parameters
EPOCHS = 30
BATCH_SIZE = 5
KEEP_PROB = 0.6
LEARNING_RATE = 0.0005
L2_REG = 1e-3
STD_DEV = 0.01
NUM_CLASSES = 2
IMAGE_SHAPE_KITI = (160, 576)  # KITTI dataset uses 160x576 images
DATA_DIR = './data'
RUNS_DIR = './runs'


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'


    tf.saved_model.loader.load(sess,[vgg_tag],vgg_path);

    #print the tensor names:
    for tensor in tf.get_default_graph().as_graph_def().node:
        print (tensor.name) 

    image_input = tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """    
    # Scale the pooling layers (as reccomended by Pierluigi Ferrari in his
    # project tips document):
    pool3_out_scaled = tf.multiply(vgg_layer3_out, 0.0001, name= "pool3_out_scaled")
    pool4_out_scaled = tf.multiply(vgg_layer4_out, 0.01, name="pool4_out_scaled")
                                              
    #1x1 Convolution, replaces "fully connected layer", reduces # of filters to num_classes
    conv1x1 = tf.layers.conv2d(inputs = vgg_layer7_out, \
                             filters = num_classes, \
                             kernel_size = 1, \
                             padding = 'same', \
                             kernel_regularizer = tf.contrib.layers.l2_regularizer(L2_REG), \
                             kernel_initializer = tf.random_normal_initializer(stddev = STD_DEV));       
    
    # Implement Transpose Convolutional Layers to  increase the height and width dimensions of
    # the 4D input tensor from VGG.

    # Transpose Convolution Layer 1, T1:
    conv_T1= tf.layers.conv2d_transpose(inputs = conv1x1, \
                                        filters = num_classes, \
                                        kernel_size = 4 , \
                                        strides = (2,2), \
                                        padding = 'same', \
                                        kernel_regularizer = tf.contrib.layers.l2_regularizer(L2_REG), \
                                        kernel_initializer = tf.random_normal_initializer(stddev = STD_DEV));
    
    # Adjust size of the skip connection layer using 1x1 convolutions:
    pool4_out_scaled_adj = tf.layers.conv2d(inputs = pool4_out_scaled, \
                                            filters = num_classes, \
                                            kernel_size = 1, \
                                            padding = 'same', \
                                            kernel_regularizer = tf.contrib.layers.l2_regularizer(L2_REG), \
                                            kernel_initializer = tf.random_normal_initializer(stddev = STD_DEV)); 
    #Add-in Skip Connection - conv_T1 with vgg layer 3
    T1 = tf.add(conv_T1,pool4_out_scaled_adj)
    
    # Transpose Convolution Layer 2, conv_T2:
    conv_T2 = tf.layers.conv2d_transpose(inputs = T1, \
                                         filters = num_classes, \
                                         kernel_size =  4 , \
                                         strides = (2,2), padding = 'same', \
                                         kernel_regularizer = tf.contrib.layers.l2_regularizer(L2_REG), \
                                         kernel_initializer = tf.random_normal_initializer(stddev = STD_DEV)); 
    

    # Adjust size of the skip connection layer using 1x1 convolutions:
    pool3_out_scaled_adj = tf.layers.conv2d(inputs = pool3_out_scaled, \
                                            filters = num_classes, \
                                            kernel_size = 1, \
                                            padding = 'same', \
                                            kernel_regularizer = tf.contrib.layers.l2_regularizer(L2_REG), \
                                            kernel_initializer = tf.random_normal_initializer(stddev = STD_DEV));   
    # Add-in Skip Connection - conv_T2 with vgg layer 4
    T2 = tf.add(conv_T2,pool3_out_scaled_adj)
    
    # Transpose Convolution Layer 2, conv_T2:
    output= tf.layers.conv2d_transpose(inputs = T2, \
                                       filters = num_classes, \
                                       kernel_size =  16, \
                                       strides = (8,8), padding = 'same', \
                                       kernel_regularizer = tf.contrib.layers.l2_regularizer(L2_REG), \
                                       kernel_initializer = tf.random_normal_initializer(stddev = STD_DEV)); 
    return output
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    
    #Re-shape output and label tensors to 2-D
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1,num_classes))
        
    #Loss Function (include regularization losses here)
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( \
                                         logits= logits, labels= correct_label))    
    cross_entropy_loss = cross_entropy_loss + tf.losses.get_regularization_loss()
    
    #Training using AdamOptimizer
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)
    
    return logits,train_op,cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    
    #init global variables
    sess.run(tf.global_variables_initializer())
    
    print("Training...\n")
    for i in range(epochs):
        print("EPOCH {} ...".format(i+1))
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss], \
                               feed_dict={input_image: image, correct_label: label, keep_prob: KEEP_PROB, learning_rate: LEARNING_RATE})            
            print("\tLoss: = {:.4f}".format(loss))
    pass
tests.test_train_nn(train_nn)


def run():

    
    # Ensure data-set is present
    tests.test_for_kitti_dataset(DATA_DIR)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(DATA_DIR)
    tf.reset_default_graph()
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(DATA_DIR, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(DATA_DIR, 'data_road/training'), IMAGE_SHAPE_KITI)        
        
        #tf placeholders:
        correct_label = tf.placeholder(tf.float32, [None, None, None, NUM_CLASSES], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        # Load vgg16 network:
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        #Build upsample layers
        final_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, NUM_CLASSES)
    
        # Define the loss and optimization functions
        logits, train_op, cross_entropy_loss = optimize(final_layer, correct_label, learning_rate, NUM_CLASSES)
        
        # Train NN using the train_nn function
        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, input_image,
                correct_label, keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples        
        helper.save_inference_samples(RUNS_DIR, DATA_DIR, sess, IMAGE_SHAPE_KITI, logits, keep_prob, input_image)



if __name__ == '__main__':
    run()
