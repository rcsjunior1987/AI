# -*- coding: utf-8 -*-
""" testess
Created on Thu Oct  1 19:23:07 2019

@author: Roberto Carlos da Silva Junior 10374647
@author: Roberto Carlos da Silva Junior n10374647
"""

# Imports modules to complete the Assignment
import numpy as np
import random

# Imports required modules from the Keras Functional API
import keras
import keras.backend as K
from keras.layers import Input
from keras.models import Model

'''Defining ranges for testing and training
    top(0), trouser(1), pullover(2), coat(4), sandal(5), ankle boot(9)
'''
set_testing_and_training = [0, 1, 2, 4, 5, 9]

'''Defining ranges for only testing
    dress(3), sneaker(7), bag(8), shirt(6)
'''

set_only_testing = [3,7,8,6]
'''Defining ranges for (testing and training) U (only testing)
    
    top(0), trouser(1), dress(3), coat(4), sandal(5)
    shirt(6),  sneaker(7), bag(8), ankle boot(9)
'''
set_testing_and_training_union_only_testing = [0,1,2,3,4,5,6,7,8,9]

'''How many epochs that should be run for the siamese model
epochs is nothing but the number of times to loop through the entire dataset
'''
epochs_model = 10

'''Verbose is a general programming term for produce lots of logging output,
in this case Verbose = 1 means that an animated bar will show the training progress
'''
verbose=1

#------------------------------------------------------------------------------

def siamese_network():
    '''Generates data, based on the three specified ways of evaluation.
    A dataset of images from the library Fashion-MNIST dataset is loaded,
    and pairs of these images are created to be used for training and testing
    whether they are from the same group by a a siamese network model that uses a Sequential CNN model as a base network
    
    The model is trained on data images that are from the group: ["top", "trouser", "pullover", "coat",
                                                                  "sandal", "ankle boot"]
    THIS set is tested against 3 dataset, that are:
        
        Set 1 = set of images with labels ["top", "trouser", "pullover", "coat",
                                                    "sandal", "ankle boot"]
        
        
        Set 2 = set of images with labels ["top", "trouser", "pullover", "coat",
                                                   "sandal", "ankle boot"]
                                             union ["dress", "sneaker", "bag", "shirt"]
                                             
                                             
        Set 3 = set of images with labels ["dress", "sneaker", "bag", "shirt"]
        
    '''    
    
# ----------------- Load the datas set    
    (train_images
   , train_target
   , test_images
   , test_target) = load_datasets()
    
    
    '''
        Prepocesses for the datasets data_set_training, data_set_testing1
                                     data_set_testing2, data_set_testing3
                                     
        as defined above.                                     
    '''           
    ((data_set_training, target_training)
   , (data_set_testing1, target_testing1) 
   , (data_set_testing2, target_testing2) 
   , (data_set_testing3, target_testing3)) = split_datasets(train_images
                                                          , train_target
                                                          , test_images
                                                          , test_target)
# ----------------- Reshape the datas set
    
    (data_set_training
   , data_set_testing1
   , data_set_testing2
   , data_set_testing3
   , datas_shape) = reshape_data_sets(data_set_training
                                    , data_set_testing1
                                    , data_set_testing2
                                    , data_set_testing3)   
    
# ----------------- Create pairs of images that will be used to train the model
    
    (training_pairs_set,
     training_target_set,
     
     test_pairs_set1,
     test_target_set1,
     
     test_pairs_set2,
     test_target_set2,
     
     test_pairs_set3,
     test_target_set3) = create_pairs(data_set_training
                                         , target_training               
               
                                         , data_set_testing1
                                         , target_testing1
             
                                         , data_set_testing2
                                         , target_testing2
               
                                         , data_set_testing3
                                         , target_testing3)

# ----------------- Train and avaluates the models
    
    evalutate_models(training_pairs_set
                  , training_target_set
                
                  , test_pairs_set1
                  , test_target_set1
            
                  , test_pairs_set2
                  , test_target_set2
            
                  , test_pairs_set3
                  , test_target_set3
                  , datas_shape)  
    
#------------------------------------------------------------------------------    
    
    
def reshape_data_sets(data_set_training, data_set_testing1, data_set_testing2, data_set_testing3):
    '''Calls the internal function responsible to reshape the datas (reshape_dataset)
    for each one of the input parameters
    and returns these inputs reshaped and also
    how the datas shape looks like after be reshaped

    Parameters:
        data_set_training: Dataset training  to be reshaped
        data_set_testing1: Dataset testing 1 to be reshaped
        data_set_testing2: Dataset testing 2 to be reshaped
        data_set_testing3: Dataset testing 3 to be reshaped
    Returns:
        data_set_training_reshaped: Dataset training  reshaped
        data_set_testing1_reshaped: Dataset testing 1 reshaped
        data_set_testing2_reshaped: Dataset testing 2 reshaped
        data_set_testing3_reshaped: Dataset testing 3 reshaped
    '''
    
    # Reshape the dataset training
    data_set_training_reshaped = reshape_dataset(data_set_training)    
        
    # Reshape the dataset testing 1
    data_set_testing1_reshaped = reshape_dataset(data_set_testing1)
    
    # Reshape the dataset testing 2
    data_set_testing2_reshaped = reshape_dataset(data_set_testing2)
    
    # Reshape the dataset testing 3
    data_set_testing3_reshaped = reshape_dataset(data_set_testing3)
    
    #input_shape = (img_rows, img_cols, 1)
    datas_shape = data_set_training_reshaped.shape[1:]
    
    # return de datasets reshaped
    return (data_set_training_reshaped
          , data_set_testing1_reshaped
          , data_set_testing2_reshaped
          , data_set_testing3_reshaped
          , datas_shape)
    
#------------------------------------------------------------------------------    

def create_pairs(data_set_training
               , target_training
               , data_set_testing1
               , target_testing1
               , data_set_testing2
               , target_testing2
               , data_set_testing3
               , target_testing3):
    '''Calls the internal function responsible to create pairs
    for each one of the input parameters
    and returns the pairs of these sets

    Parameters:
        data_set_training: Array with images of the set training and testing,
        target_training:  Array with with the labels of the set training and testing,
        data_set_testing1: Array with images of the set testing1,
        target_testing1:  Array with with the labels of the set testing1,
        data_set_testing2: Array with images of the set testing2,
        target_testing2:  Array with with the labels of the set testing2,
        data_set_testing3: Array with images of the set testing3,
        target_testing3:  Array with with the labels of the set testing3,
    Returns:
        data_set_training_reshaped: Dataset training  reshaped
        data_set_testing1_reshaped: Dataset testing 1 reshaped
        data_set_testing2_reshaped: Dataset testing 2 reshaped
        data_set_testing3_reshaped: Dataset testing 3 reshaped
    '''
            
    #Creates pairs of images that will be used to train the model with digits in set_testing_and_training
    (training_pairs_set
   , training_target) = create_pairs_sets(data_set_training
                                        , target_training
                                        , set_testing_and_training
                                        , 1)  
    
    #Creates pairs of images that will be used to test the model with digits in set_testing_and_training
    (test_pairs_set1
   , test_target_set1) = create_pairs_sets(data_set_testing1
                                        , target_testing1
                                        , set_testing_and_training
                                        , 1)

    #Creates pairs of images that will be used to test the model with digits in set_only_testing
    (test_pairs_set2
   , test_target_set2) = create_pairs_sets(data_set_testing2
                                         , target_testing2
                                         , set_only_testing
                                         , 2)

    #Creates pairs of images that will be used to test the model with digits in set_testing_and_training_union_only_testing
    (test_pairs_set3
   , test_target_set3) = create_pairs_sets(data_set_testing3
                                         , target_testing3
                                         , set_testing_and_training_union_only_testing
                                          , 3)
    
    return (training_pairs_set
          , training_target
          
          , test_pairs_set1
          , test_target_set1
          
          , test_pairs_set2
          , test_target_set2
          
          , test_pairs_set3
          , test_target_set3)
    
#------------------------------------------------------------------------------    
    
def evalutate_models(training_pairs
                   , training_target
                   , test_pairs_set1
                   , test_target_set1
                   , test_pairs_set2
                   , test_target_set2
                   , test_pairs_set3
                   , test_target_set3
                   , input_shape):
    '''Create, train and evaluate the siamese network, besides
    prit the Accuracy on training e test of the datasets at its final.
    
    The first two parameter of this function are the set used
    for training and testing, this set will be tested against three
    pairs of sets.
        pairs of set 1 = set of images with labels ["top", "trouser", "pullover", "coat",
                                                    "sandal", "ankle boot"]
        
        
        pairs of set 2 = set of images with labels ["top", "trouser", "pullover", "coat",
                                                   "sandal", "ankle boot"]
                                             union ["dress", "sneaker", "bag", "shirt"]
                                             
                                             
        pairs of set 3 = set of images with labels ["dress", "sneaker", "bag", "shirt"]                                             
    Parameters:
        training_pairs:    Array with the pairs of set training and testing,
        training_target:   Array with the labels of the pairs of images of the set training and testing,
        test_pairs_set1:   Array with the pairs of set 1,
        test_target_set1:  Array with the labels of the pairs of images of the set 1,
        test_pairs_set2:   Array with the pairs of set 2,
        test_target_set2:  Array with the labels of the pairs of images of the set 2,
        test_pairs_set3:   Array with the pairs of set 3,
        test_target_set3: Array with the labels of the pairs of images of the set 3,
        input_shape:       Parameter with the shape of the datas,
    Returns:
    '''

    '''Loop for the creation, training and evalutation of the siamese network model 3 times..
    which each interation represents a data set such as
      interaction 1 = set_testing_and_training
    , set_only_testing
    , set_testing_and_training_union_only_testing
    In order to generate data about validation accuracy and validation loss after each epoch that is run.
    '''
    for i in range(3):

        # Depending on what loop index is running different test data is used for validation.
        if (i == 0):
            test_pairs = test_pairs_set1
            test_target = test_target_set1
            
        if (i == 1):
            test_pairs = test_pairs_set2
            test_target = test_target_set2
            
        if (i == 2):
            test_pairs = test_pairs_set3
            test_target = test_target_set3    
            
        # Calls the function responsible to train and validate the models.
        model = train_model_with_validation(input_shape
                                           , training_pairs
                                           , training_target
                                           , test_pairs
                                           , test_target
                                           , epochs_model
                                           , verbose)
        
    print('-------------------------------------------------------------------------------')
    print('Final accuracies for the different datasets ', (i+1), " after a total of ", epochs_model, " epoch(s).")
        
    # Compute and print final accuracy, as percentage with 2 decimals, on training and test sets.
    y_pred = model.predict([training_pairs[:, 0], training_pairs[:, 1]])
    print('* Accuracy on training set: %0.2f%%' % (100 * compute_accuracy(training_target, y_pred)))    
        
    y_pred = model.predict([test_pairs_set1[:, 0], test_pairs_set1[:, 1]])
    print('* Accuracy on test set 1: %0.2f%%' % (100 * compute_accuracy(test_target_set1, y_pred)))
        
    y_pred = model.predict([test_pairs_set2[:, 0], test_pairs_set2[:, 1]])
    print('* Accuracy on test set 2: %0.2f%%' % (100 * compute_accuracy(test_target_set2, y_pred)))
        
    y_pred = model.predict([test_pairs_set3[:, 0], test_pairs_set3[:, 1]])
    print('* Accuracy on test set 3: %0.2f%%' % (100 * compute_accuracy(test_target_set3, y_pred)))
    print('-------------------------------------------------------------------------------')
        
#------------------------------------------------------------------------------        

def load_datasets():
    '''Load the Fashion-MNIST dataset
    And returns four NumPy arrays that are train_images
                                         , train_target
                                         , test_images
                                         and test_target
    
    train_images and train_target are arrays for the training set(the data the model uses to learn).
    whereas test_images and test_target which are arrays to be tested against the model.
    
    
        For the arrays train_target and test_target the index for the
            group top  = 0
          , trouser    = 1
          , pullover   = 2
          , dress      = 3
          , coat       = 4
          , sandal     = 5
          , shirt      = 6
          , sneaker    = 7
          , bag        = 8
          , ankle boot = 9
    
    Parameters:
    Returns:
        train_images: Array with the images to be trained
        train_labels: Array with the labels of the images to be trained
        test_images:  Array with the images to be tested
        test_labels:  Array with the labels of the images to be tested
    '''
    
    # Loads the FASHION MNIST-dataset from the keras library.
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_target), (test_images, test_target) = fashion_mnist.load_data()    
            
    # Returns the arrays train_images, train_labels, test_images and test_labels
    return (train_images
          , train_target
          , test_images
          , test_target)
    
#------------------------------------------------------------------------------    
    
def split_datasets(train_images, train_target, test_images, test_target):   
    '''Split the dataset such as
    
    1. The images with labels in ["top(index's group = 0)"
                                , "trouser(1)"
                                , "pullover(2)"
                                , "coat(4)"
                                , "sandal(5)"
                                , "ankle boot(9)"]
    which are used for training and testing;
        
    2. The images with labels in ["dress(3)"
                                , "sneaker(7)"
                                , "bag(8)"
                                , "shirt(6)"]
    which are used only for testing.
        
    And returns a total of 4 different datasets: one for training and 3 for testing.
    Parameters:
        train_images:  Array with the images to be trained
        train_target: Array with the labels of the images to be trained
        test_images:   Array with the images to be tested
        test_target:  Array with the labels of the images to be tested
    Returns:
        (data_train, target_train):            Dataset for training the model
        (data_test, target_test):              Dataset 1 for testing
        (only_test_dataset, only_test_target): Dataset 2 for testing
        (final_test_data, final_test_target):  Dataset 3 for testing
    '''     
    
    '''Concatenate train and test data into their own datasets.
    train_(training and testing)
    test_(only test)
    '''
    image_dataset = np.concatenate([train_images, test_images])
    target_dataset = np.concatenate([train_target, test_target])
        
    '''Creates a mask with all images that should be used only for testing.
    I.e. dress(3), "sneaker(7)", "bag(8)", "shirt(6)"
    '''
    only_test_mask = np.where(target_dataset>=6, target_dataset<=8, target_dataset==3)
    
    '''Initiate two new arrays with data and target based on only_test_mask.       
    These are ONLY used for testing and are dress(3), sneaker(7), bag(8), shirt(6).        
    '''
    only_test_dataset = image_dataset[only_test_mask,:,:]
    only_test_target = target_dataset[only_test_mask]
       
    '''Initiate two new arrays with data and target based on train_and_test_mask.
    These are used for both testing and traning.
        
    I.e. top(0), trouser(1), pullover(2), coat(4), sandal(5), ankle boot(9).
    '''
    test_and_train_dataset = image_dataset[~only_test_mask,:,:]
    test_and_train_target = target_dataset[~only_test_mask]
    
    # Import module for splitting datasets.
    from sklearn.model_selection import train_test_split
    '''Splits the dataset that are used for both train and test into respective sets. 
        
    80% for the set train and test = top(0), trouser(1), pullover(2), coat(4), sandal(5), ankle boot(9) 
    and 
    20% for set ONLY test = dress(3), sneaker(7), bag(8), shirt(6)."
    '''
    data_train, data_test, target_train, target_test = train_test_split(test_and_train_dataset,
                                                                          test_and_train_target,
                                                                          test_size=0.20)  
    
    # Concatenates the data that should be used for testing.
    final_test_data = np.concatenate([data_test, only_test_dataset])
    final_test_target = np.concatenate([target_test, only_test_target])
    
    #Returns one for training and 3 for testing.
    return ((data_train, target_train)
          , (data_test, target_test)
          , (only_test_dataset, only_test_target)
          , (final_test_data, final_test_target))
    
#------------------------------------------------------------------------------    

def reshape_dataset(dataset):
    '''Reshape the datassets in order to create a pattern
    of all datasets of images before training the network
    Parameters:
        dataset: data set to be reshaped
    Returns:
        dataset_reshaped: dataset reshaped
    '''
    
    # Gets the dimensions of the input_data
    img_rows, img_cols = dataset.shape[1:3]

    # Reshape dataset into 4D array (amount of images, rows, columns, channels)
    dataset_reshaped = dataset.reshape(dataset.shape[0], img_rows, img_cols, 1)
    
    # Convert to float32
    dataset_reshaped = dataset_reshaped.astype('float32')
    
    '''The datas are RGB number between 0 and 255]
    Reason why the vectors are divided by 255.
    '''
    dataset_reshaped /= 255

    # Returns the dataset reshaped
    return dataset_reshaped

#------------------------------------------------------------------------------

def create_pairs_sets(dataset, target, ranges, test_target):
    '''Create pairs of images separating by their dataset 
        
    And returns two arrays, one with the images and the other
    with their labels
    Parameters:
        dataset:    Array with the data set of the pairs to be created
        target:     Array with labels of the image of the pairs to be created
        ranges:      Range of the set of the pairs
                     to be created (set_testing_and_training
                                  / set_only_testing)
                                  / set_testing_and_training_union_only_testing
        test_target:  Labels of the images of the pairs
                     to be created (1 = set of testing and training
                                  , 2 = only testing
                                  , 3 = union of testing and training
                                    and only testing)
    Returns:
        np.array(pairs): Array with the pair of images
        np.array(target): Array with the group's target of images
    '''
    
    digit_indices = [np.where(target == i)[0] for i in ranges]
    
    # Creates arrays named pairs and labels
    pairs = []
    target = []
    
    '''Defines the range of digits that are in the current dataset from where the pairs are to be created.
        
    if the pars to be created are set of testing and training
    '''
    if (test_target == 1):
        digits = [0, 1, 2, 4, 5, 9]
        
    # if the pars to be created are set of only testing
    if (test_target == 2):
        digits = [3,7,8,6]
                
    # if the pars to be created are set of (testing and training) U (only testing)
    if (test_target == 3):
        digits = [0,1,2,3,4,5,6,7,8,9]
    
    # Defines the minimum sample as the length of the digit_indices variable
    min_sample = [len(digit_indices[d]) for d in range(len(digits))]
    
    # The smallest number in the min_sample minus 1 is assigned in n
    n = min(min_sample) -1
       
    # Iterates through the range of digits
    for d in range(len(digits)):
        for i in range(n):

            # Assigns values z1 and z2
            z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
            
            # adds the z1 and z2 coordinates to the pairs array
            pairs += [[dataset[z1], dataset[z2]]]

            '''A random number between 1 and the length of the digits array is got
            and assigned in the variable random_Number
            '''
            random_Number = random.randrange(1, len(digits))
            
            '''The values of (d + random_Number) 
             is divided by the length of the digits array
             and assigns it to the variable dn
            ''' 
            dn = (d + random_Number) % len(digits)

            # Assigns the values of z1 and z2 and adds them to the pairs array, using the dn variable
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            #z1, z2 = digit_indices[d], digit_indices[dn]
            pairs += [[dataset[z1], dataset[z2]]]

            # Adds the coordinates 1,0 to the labels array
            target += [1, 0]
            
    # Returns the 2 arrays
    return (np.array(pairs)
          , np.array(target))
    
#------------------------------------------------------------------------------    
    
def train_model_with_validation(input_shape,
                                training_pairs,
                                training_target,
                                test_pairs,
                                test_target,
                                epochs,
                                verbose
                               ):
    '''Train/validate the model
        
    And returns a Siamese model to be avaluated
    Parameters:
        input_shape:      Parameter with the shape of the datasets,
        training_pairs:   Set of pairs to be trained,
        training_target:  Set of the labels of the pairs to be trained,
        test_pairs:       Set of pairs to be tested,
        test_target :     Set of the labels of the pairs the pairs to be tested,
        epochs:           Value of the parameter epochs, which is defined above as a static variable,
        verbose:          Value of the parameter verbose, which is defined above as a static variable
    Returns:
        model: Siamese model to be avaluated 
    '''

    # Use a CNN network as the shared network.
    cnn_network_model = build_CNN(input_shape)

    '''Initiates inputs with the same amount of slots to keep the image arrays
    sequences to be used as input data when processing the inputs.
    '''
    image_vector_shape_1 = Input(shape=input_shape)
    image_vector_shape_2 = Input(shape=input_shape)

    # The CNN network model will be shared including weights
    output_cnn_1 = cnn_network_model(image_vector_shape_1)
    output_cnn_2 = cnn_network_model(image_vector_shape_2)

    # Concatenates the two output vectors into one.
    distance = keras.layers.Lambda(euclidean_distance,
                                   output_shape=eucl_dist_output_shape)([output_cnn_1, output_cnn_2])

    '''It is define a trainable model, which it is linked the two different image inputs to the distance
    between the processed input by the cnn network.
    '''
    model = Model([image_vector_shape_1, image_vector_shape_2],
                  distance
                 )
    # Specifying the optimizer for the netwrok model
    rms = keras.optimizers.RMSprop()

    # Compiles the model with the contrastive loss function.
    model.compile(loss=contrastive_loss_function,
                  optimizer=rms,
                  metrics=[accuracy])

    # Validating and printing data using the test data with index i.
    model.fit([training_pairs[:, 0], training_pairs[:, 1]], training_target,
              batch_size=128,
              epochs=epochs,
              verbose=verbose,
              validation_data=([test_pairs[:, 0], test_pairs[:, 1]], test_target)
             )

    return model

#------------------------------------------------------------------------------

def build_CNN(input_shape):
    '''Build a CNN model to be used as a shared network in the siamese network model.    
    Parameters:
        input_shape: The dimenstions of the dataset to be used
    Returns:
        cnn_model:   A keras Sequential model
    '''

    # Initiates a sequential model
    cnn_model = keras.models.Sequential()

    # Adds layers to the sequential model
    cnn_model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    cnn_model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    cnn_model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(keras.layers.Dropout(0.25))
    cnn_model.add(keras.layers.Flatten())
    cnn_model.add(keras.layers.Dense(128, activation='relu'))
    cnn_model.add(keras.layers.Dropout(0.5))
    cnn_model.add(keras.layers.Dense(10, activation='softmax'))

    # Retunrs the specified sequential model
    return cnn_model

#------------------------------------------------------------------------------

def euclidean_distance(vects):
    '''Function used to calculate Euclidean distance, which is the straight-line distance between two points
    Taken from https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py
    '''

    x, y = vects

    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)

    return K.sqrt(K.maximum(sum_square, K.epsilon()))

#------------------------------------------------------------------------------

def eucl_dist_output_shape(shapes):
    '''Function used to return the Euclidean shape
    Taken from https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py
    '''

    shape1, shape2 = shapes

    return (shape1[0], 1)

#------------------------------------------------------------------------------

def contrastive_loss_function(y_true, y_pred):
    '''Contrastive loss function
    Taken from https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py
    '''

    # The margin m > 0 determines how far the embeddings of a negative pair should be pushed apart.
    m = 2 # margin # Might need to be changed and evaluated for what value the siamese network performs best.

    sqaure_pred = K.square(y_pred)

    margin_square = K.square(K.maximum(m - y_pred, 0))

    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

#------------------------------------------------------------------------------

def accuracy(y_true, y_pred):
    '''Computes classification accuracy with a fixed threshold on distances.
    Taken from https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py
    '''

    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

#------------------------------------------------------------------------------

def compute_accuracy(y_true, y_pred):
    '''For evaluating the prediction accuracy of the model.
    Taken from https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py
    '''

    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)
        
#------------------------------------------------------------------------------ 
       
siamese_network()
