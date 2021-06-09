
# -*- coding: utf-8 -*-
"""
@author: Lowie Cloesen

Coursework C: Neuron Spike Classifier

Objective: 
Prrocess a time recording from an electrode inserted into the brain, identifying where in the recording 
the spikes corresponding to neuron activity are located and assigning one of 4 types of neurons to each
spike. 
Two different machine learning algorithms should be tested for this and the best one will be used to
process the submission recording. The location and type of each detected neuron spike should be saved
into a MATLAB-style .mat file and will be compared to the results of a state-of-the art classifier to
evaluate performance.

MAKE SURE TO CHANGE THE root_directory IN THE "USER INPUT VARIABLES" SECTION BEFORE RUNNING THIS CODE 
ON ANOTHER COMPUTER.
"""

import os

# Numpy for useful maths
import numpy as np
import math

# Matplotlib for plotting
import matplotlib.pyplot as plt

# Scipy for peak finding and manipulating matlab data files
from scipy import signal, io, special

# Sklearn for useful CI tools
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

"""SUPPORTING FUNCTIONS & CLASSES"""
def get_metrics(actual_classes, pred_classes):
    """
    Function to calculate performance metrics for the classifier
    For each class, the following is calculated
    TP: True positives = samples that were correctly put into the class
    TN: True negatives = samples that were correctly not put into the class
    FP: False positive = samples that were incorectly put into the class
    FN: False negatives = samples that should be in the class but were put into
                            another class
                        

    Parameters
    ----------
    pred_classes : neuron types predicted by the classifier
    actual_classes : known neuron types

    Returns
    -------
    conf_mat:   Confusion matrix = a visual representation of the algorithm's performance
    acc:        Accuracy =  the fraction of correctly classified samples
    MK:         Markedness = a measure of how trustworthy a classification is,
                accounting for both positive and negative classifications. 
                Value close to 1 means the classifier makes mostly correct predictions, value
                close to -1 means the classifier makes mostly wrong predictions.
    """
    
    conf_mat = metrics.confusion_matrix(actual_classes, pred_classes)
    acc = metrics.balanced_accuracy_score(actual_classes, pred_classes)
    
    """
    the next portion of code is copied from:
    https://towardsdatascience.com/multi-class-classification-extracting-performance-metrics-from-the-confusion-matrix-b379b427a872
    """
    FP = conf_mat.sum(axis=0) - np.diag(conf_mat) 
    FN = conf_mat.sum(axis=1) - np.diag(conf_mat)
    TP = np.diag(conf_mat)
    TN = conf_mat.sum() - (FP + FN + TP)    
    
    FP = np.sum(FP)
    FN = np.sum(FN)
    TP = np.sum(TP)
    TN = np.sum(TN)
    """
    end of copied code
    """
    MK = (TP/(TP+FP)) + (TN/(TN+FN)) - 1
    return conf_mat, acc, MK

"""The simulated annealing class is adapted from a generic simulated annealing code provided in a lab exercise"""
class Annealer:
    def __init__(self, initial_n, train_data, train_class, test_data, test_class):
        """
        Parameters
        ----------
        initial_n : Number of neighbours to use in the unoptimised KNN algorithm
        train_data : spike datapoints used to train the KNN
        train_class : neuron-type for each training instance
        test_data : spike datapoints used to test the KNN
        test_class : neuron_type for each testing instance

        Returns
        -------
        None   
        """
        self.n = initial_n
        self.train_data = train_data
        self.train_class = train_class
        self.test_data = test_data
        self.test_class = test_class
        self.knn = KNeighborsClassifier(n_neighbors=self.n, p=2, weights='distance')
        self.knn.fit(self.train_data, self.train_class)
        self.pred = self.knn.predict(self.test_data)
        
    def acceptance_probability(self, old_cost, new_cost, T):
        """
        If the new cost is lower than the old cost, the acceptance probability
        is greater than 1 and the new classifier is always accepted as the new
        solution. 
        If the new cost is higher, the acceptance probability is lower than 1, 
        thus less likely to get accepted. 
        If the difference in the two costs is small or the temperature is high, 
        the acceptance probability is close to 1 and thenew solution is likely 
        to be accepted. 

        Parameters
        ----------
        old_cost : cost calculated for the old solution
        new_cost : cost calculated for the new solution
        T : current temperature of the 'annealer'

        Returns
        -------
        a : probability that the new solution will be accepted as the best solution
        """
        a = math.exp((old_cost-new_cost)/T)
        return a
    
    def cost(self, predicted_classes, actual_classes):
        """
        Combines the accuracy and markedness values to return a cost that 
        reflects the performance in both. Because accuracy and markedness are
        small values (between 0 and 1 or -1 and 1), the cost is exaggerated so 
        as to have an actual impact on the optimiser

        Parameters
        ----------
        predicted_classes : list of neuron types predicted by the KNN
        actual_classes : list of known neuron types

        Returns
        -------
        cost : a measure of how far the current solution is from being perfect
        """
        _, accuracy, MK = get_metrics(actual_classes, predicted_classes)
        cost = 2-(accuracy + MK)
        cost = cost*10
        return cost
    
    def neighbour(self):
        """
        Changes the number of neighbours used by the classifier and creates
        a new set of predictions. The number of neighbours is only varied by one
        or two neighbours each time
        """
        m = 0
        while(m == 0):
            m = np.random.randint(-2,3)
            
        new_nneighbours = np.add(self.n, m)
        # Make sure the number of neighbours is not 0 or negative
        if new_nneighbours < 1:
            new_nneighbours = np.add(self.n, 1)
        # Change the number of neighbours used for predictions
        self.knn.set_params(n_neighbors = new_nneighbours)
        new_pred = self.knn.predict(self.test_data)
        return new_nneighbours, new_pred
    
    def anneal(self, alpha, iterations):
        """
        Calculates the cost associated with the performance of the current
        KNN classifier, creates a new KNN classifier using a different number 
        of neigbours and calculates the new cost. 
        Decides whether the new classifier should become the new best solution 
        (more likely to do so if the temperature is high and the new classifier 
        has a lower cost). 
        Annealing continues until the temperature has reached its minimum value.

        Parameters
        ----------
        alpha : a multiplier (< 1) to reduce the temperature after changing the 
                parameter to optimise a certain number of times.
        iterations : how many times the parameter to optimise is changed before
                lowering the temperature 

        Returns
        -------
        pred : the last neuron type predictions (using the optimal number of neighbours)
        cost_values : a history of the cost calculated with each parameter change
        n_values : a history of the number of neighbours used in the KNN
        """
        pred = self.pred
        test_class = self.test_class
        old_cost = self.cost(pred, test_class)
        cost_values = list()
        n_values = list()
        cost_values.append(old_cost)
        T = 1.0
        T_min = 0.000001
        while T > T_min:
            i = 1
            while i <= iterations:
                # Create a new KNN classifier
                new_nneighbours, new_pred = self.neighbour()
                # Calculate the cost
                new_cost = self.cost(new_pred, test_class)
                # Decide if the new classifier should become the current best solution
                ap = self.acceptance_probability(old_cost, new_cost, T)
                if ap > np.random.random():
                    pred = new_pred
                    self.n = new_nneighbours
                    old_cost = new_cost
                i += 1
                cost_values.append(old_cost)
                n_values.append(self.n)
            # Decrease the temperature
            T = T*alpha
        return pred, cost_values, n_values

"""The NeuralNetwork class is copied from the code provided for a previous piece of coursework"""
class NeuralNetwork:
    # Init the network, this gets run whenever we make a new instance of this class
    def __init__ (self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set the number of nodes in each input, hidden and output layer
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes
        # Weight matrices, wih (input -> hidden) and who (hidden -> output)
        self.wih = np.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.who = np.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))
        # Set the learning rate
        self.lr = learning_rate
        # Set the activation function, the logistic sigmoid
        self.activation_function = lambda x: special.expit(x)

    # Train the network using back-propagation of errors
    def train(self, inputs_list, targets_list):
        # Convert inputs into 2D arrays
        inputs_array = np.array(inputs_list, ndmin=2).T
        targets_array = np.array(targets_list, ndmin=2).T
        # Calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs_array)
        # Calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # Calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # Calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        # Current error is (target - actual)
        output_errors = targets_array - final_outputs
        # Hidden layer errors are the output errors, split by the weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)
        # Update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
        np.transpose(hidden_outputs))
        # Update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
        np.transpose(inputs_array))

    # Query the network
    def query(self, inputs_list):
        # Convert the inputs list into a 2D array
        inputs_array = np.array(inputs_list, ndmin=2).T
        # Calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs_array)
        # Calculate output from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # Calculate signals into final layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # Calculate outputs from the final layer
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
"""end of copied code""" 

"""START OF MAIN CODE"""
plt.close('all')

"""USER INPUT VARIABLES"""
# Folder containing training and submission data. 
### MAKE SURE TO CHANGE THE ROOT_DIRECTORY TO THE FOLDER WHERE YOU PUT THE TRAINING AND SUBMISSION .MAT FILES###
root_directory = r"/Users/RootDirName"
training_file_name = "training.mat"
submission_file_name = "submission.mat"

# Do you want to run the algorithm optimiser (see STEP 6)? 
# This will take a few minutes to run. 
# If the optimisation is not used, the KNN classifier used for the submission
# data will use 4 neighbours by default as this should give the best results.
run_optimisation = 1 # no = 0, yes = 1
"""END OF USER INPUT VARIABLES"""

# Location of the training data
if os.path.exists(root_directory):
    trainpath = [os.path.join(root, name)
                for root, dirs, files in os.walk(root_directory)
                for name in files
                if name.endswith((training_file_name))]

    # Location of the submision data
    testpath = [os.path.join(root, name)
                for root, dirs, files in os.walk(root_directory)
                for name in files
                if name.endswith((submission_file_name))]
else:
    raise Exception("No {0} folder found. Please check that the root_directory variable is correct".format(root_directory))

# Load the recording data
if len(trainpath) > 0:
    mat = io.loadmat(trainpath[0], squeeze_me=True)
else:
    raise Exception("No training data file found in the chosen folder. Please check that the root_directory variable is correct and that you have downloaded the training.mat file")

if len(testpath) > 0:
    mat_s = io.loadmat(testpath[0], squeeze_me=True)
else:
    raise Exception("No testing data file found in the chosen folder. Please check that the root_directory variable is correct and that you have downloaded the submission.mat file")

d = mat['d']            # Complete training recording data
Index = mat['Index']    # The location of the starting point of each neuron spike inside the recording
Class = mat['Class']    # The type of neuron to which each spike corresponds

# Sort the Class and Index arrays based on the values in Index to have each spike location and its class in
# chronological order.
idx   = np.argsort(Index)
Index = np.array(Index)[idx]
Class = np.array(Class)[idx]

"""STEP 1: REMOVE NOISE FROM TRAINING DATA"""
# Remove high frequency noise using a Butterworth filter.
# Signal components with a frequency above 2500 Hz are attenuated.
sos = signal.butter(3, 2500, 'lowpass', fs=25000, output='sos')
filtered = signal.sosfiltfilt(sos, d)

# Find the peaks in the recording based on the amplitude of the signal.
peaks, properties = signal.find_peaks(filtered, height=2)


"""STEP 2: EXTRACT DATA AROUND EACH PEAK (I.E THE SPIKES) AND ITS CORRESPONDING NEURON CLASS"""
spike_idx = []
spike_class = []
spike_data = []

# For each peak, identify which spike it corresponds to, what its neuron type is and extract the spike
# waveform to obtain all spike shapes in a common format (51 data samples)
for index_peak in peaks:
    # Find the spike index nearest to the peak index. 
    # As the spike index corresponds to the starting point of a spike we have spike_index < peak_index
    idx = np.searchsorted(Index, index_peak, side='right')
    true_index = Index[idx-1]
    spike_idx.append(true_index)

    # Find the class that the spike corresponds to
    c = Class[idx-1]
    spike_class.append(c)

    # Extract the spike (20 samples before and 31 after the peak [including the peak])
    # Spikes are extracted from the original recording to preserve their shape
    spike = d[index_peak-20:index_peak+31]
    spike_data.append(spike)


"""STEP 3: SEPARATE DATA INTO TRAINING AND TEST SETS"""
test_ratio = 0.20 # proportion of spikes that will be used for testing 
random_state = 5 # random_state value ensures the same split is obtained on each run
spike_train, spike_test, class_train, class_test = train_test_split(spike_data, spike_class, test_size=test_ratio, random_state=random_state)


"""STEP 4: TRAIN AND TEST A MULTILAYER PERCEPTRON"""
"""
1st classifier = Multilayer Perceptron (MLP)
The MLP maps input data (51 data samples that make up a spike) to one of the possible outputs (the 4 neuron types).
The MLP is trained by feeding in inputs with know outputs. With each set of training data, the weights of the perceptrons
inside the network are updated to better map inputs to outputs. 
When test data is given to the trained MLP, it returns a probability of correspondence for each of the outputs with the
highest probability indicating the class it thinks the input corresponds to.
For a comprehensive introduction to MLPs see: https://machinelearningmastery.com/neural-networks-crash-course/ 
"""
print("Testing MLP classifier...")
input_nodes = 51        # corresponds to the number of data points per spike
hidden_nodes = 200      # number of perceptrons in the hidden layer of the MLP
output_nodes = 4        # corresponds to the number of possible classes (neuron types)
learning_rate = 0.02

iterations = 10
metrics_mlp = np.zeros([iterations,5])

# Create, train and test an MLP multiple times
for itr in range(iterations):
    nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    
    # Loop through all of the records in the training data set
    for s_data, s_class in zip(spike_train, class_train):
        inputs = s_data
        # Create the target output values (all 0.01, except the desired label which is 0.99)
        targets = np.zeros(output_nodes) + 0.01
        # s_class is the target label for this spike
        targets[s_class-1] = 0.99
        # Train the network
        nn.train(inputs, targets)
    
    
    class_pred = []
    # Loop through all of the records in the test data set
    for s_data, s_class in zip(spike_test, class_test):
        correct_label = s_class
        inputs = s_data
        # Query the network (i.e. make a predictions for the neuron type)
        outputs = nn.query(inputs)
        # The index of the highest output value corresponds to the label (i.e. type)
        label = np.argmax(outputs) + 1
        class_pred.append(label)
        
    # Performance metrics
    cm, acc, mk = get_metrics(class_test, class_pred)
    metrics_mlp[itr,0:2] = acc, mk

# Average MLP performance 
acc_avg = np.mean(metrics_mlp[:,0])
mk_avg = np.mean(metrics_mlp[:,1])
print("MLP - Tested over {0} iterations with {1} % of the data used for testing".format(iterations, test_ratio*100))
print("Hidden neurons: {0}\tLearning rate: {1}".format(hidden_nodes, learning_rate))
print("Accuracy\t{0:0.4f}".format(acc_avg))
print("Markedness\t{0:0.4f}".format(mk_avg))


"""STEP 5: TRAIN AND TEST A K-NEAREST NEIGHBOURS ALGORITHM"""
"""
2nd classifier = K-nearest neigbours (KNN) algorithm
Instead of creating a model for each type of spike, the KNN algorithm compares test inputs to training data and 
looks for the instance in the training data that most resembles the test data. The output is simply the class or type
of that most similar training instance (i.e. of the nearest neighbour). When multiple nearest neighbours are used, the
output is the average or the weighted average of the classes of those neighbours.
For a comprehensive introduction to KNN see: https://towardsdatascience.com/a-simple-introduction-to-k-nearest-neighbors-algorithm-b3519ed98e
"""
print("\nTesting KNN classifier...")
n = 4       # number of neighbours
dist = 2    # code for the Euclidian measure of the 'distance' between data points.  

# Create the KNN classifier
knn = KNeighborsClassifier(n_neighbors=n, p=dist, weights = 'distance') # closer neigbours will have a higher weight 
knn.fit(spike_train, class_train)

# Feed the test data in the classifier to get the predictions
knn_pred = knn.predict(spike_test)
# Performance metrics
cm_knn, acc_knn, mk_knn = get_metrics(class_test, knn_pred)

# KNN performance 
# (NB: The KNN algorithm is repeatable so there is no need to test it multiple times)
print("KNN - Tested with {0} % of the data used for testing".format(test_ratio*100))
print("Neighbours: {0}\tDistance type: {1}".format(n, dist))
print("Accuracy\t{0:0.4f}".format(acc_knn))
print("Markedness\t{0:0.4f}".format(mk_knn))


"""STEP 6: OPTIMISE KNN USING SIMULATED ANNEALING"""
"""
KNN selected for optimisation as it always outputs the same results as long as the parameters 
don't change (MLP outputs change every time it is re-trained).
Simulated Annealing Optimisation: The cost of an initial solution is calculated -> A parameter is
changed to give a new solution -> the cost of the new solution is calculated -> The new solution is
accepter or rejected as the latest optimal solution depending on the current 'temperature' of the
annealer and the new cost (see Annealer class).
For a comprehensive introduction to Simulated Annealing see: 
https://towardsdatascience.com/optimization-techniques-simulated-annealing-d6a4785a1de7
"""

# Run the optimisation if the user asked for it
if run_optimisation == 1:
    print("\nOptimising KNN classifier...")
    annealer = Annealer(1, spike_train, class_train, spike_test, class_test)
    best_knn_pred, cost_values, n_values = annealer.anneal(alpha = 0.9, iterations = 5)
    
    # Find best n
    best_n_idx = cost_values.index(min(cost_values))
    best_n = n_values[best_n_idx]
    
    fig, axs = plt.subplots(2, 1, sharex = True)
    axs[0].plot(cost_values)
    axs[0].set_ylabel("Cost")
    axs[0].set_title("Optimisation")
    axs[1].plot(n_values)
    axs[1].set_ylabel("Number of neighbours")
 

    print("End of optimisation...")
else:
    best_n = 4
    print("\nSkipped optimisation...")

"""STEP 7: TEST THE ALGORITHM ON THE SUMBISSION RECORDING"""
"""
The performance of the chosen algorithm is evaluated by testing it on data from a real-life recording.
This data is much noisier than the training data and contains additional low-frequency noise
because the patient was moving during the recording.
The indices of the peaks found in the submission recording and their predicted neuron type
are saved so that Dr. B. Metcalfe can compare them to the predictions of a state-of-the-art classifier.
"""

# Load the submission data
d_s = mat_s['d']

# Remove low frequency noise to align the signal's baseline with the x-axis
sos_flat = signal.butter(3, 30, 'lowpass', fs=25000, output='sos')
low_s = signal.sosfiltfilt(sos_flat, d_s)
flat_s = d_s - low_s

# Remove high frequency noise 
sos2 = signal.butter(3, 2000, 'lowpass', fs=25000, output='sos')
filt_s = signal.sosfiltfilt(sos2, flat_s)

# Find peaks
peaks_s, _ = signal.find_peaks(filt_s, prominence=0.1, height=2)

spike_s_data = []
# Extract spikes (from the flat_s signal to obtain spikes that most resemble those from the training data)
for index_peak in peaks_s:
    spike = flat_s[index_peak-20:index_peak+31]
    spike_s_data.append(spike)

best_knn = KNeighborsClassifier(n_neighbors=best_n, p=2, weights = 'distance')
best_knn.fit(spike_data, spike_class)

# Predict the neuron types
final_pred = best_knn.predict(spike_s_data)
unique, counts = np.unique(final_pred, return_counts=True)

# Save the results
mat_out = {'Index': peaks_s, 'Class':final_pred}
submission_file_name = "LowieCloesen.mat"
submission_file_path = os.path.join(root_directory, submission_file_name)
io.savemat(submission_file_path, mat_out)
print("\nResults for submission saved as {0} in {1}".format(submission_file_name, root_directory))

if run_optimisation == 1:
   plt.show()