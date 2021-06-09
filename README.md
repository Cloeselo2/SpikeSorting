# SpikeSorting

## Context
This project was part of a university module on Computational Intelligence which provided an introduction to machine learning and its applications. The application here was the use of machine learning algorithms to process recordings of brain activity measured from electrodes inserted in a subject’s brain. These pick up the activity of a few neurons located near the tip of the electrode with the aim of tracking the response of neighbouring neurons as these can fire in response to different things. Accurate tracking of single-neuron activity is important to further our understanding of brain mechanisms and disorders such as Alzheimer’s.
Because the electrodes record the combined activity of neighbouring neurons, the activity of an individual neuron needs to be isolated from the rest of the signal. This is done by looking for spikes in the recording which correspond to a neuron firing and differentiating them based on their morphology as multiple spikes produced by the same neuron should have the same shape. A visual explanation of this process is shown below.

![Spike Sorting Workflow](images/Spike sorting workflow.png)

## Objectives

Two sets of data were provided: a training recording and a submission recording. The training recording was a simulated signal containing each of the four spike types. Along with the recording data, arrays containing the location of each spike within the recording and their corresponding neuron type were provided. 
After training, the final algorithm was used on the data from the submission recording (real measured data). The location and type of each spike in the submission recording had to be saved to a .mat file (MATLAB) to be compared to the results of a state-of-the-art classifier.

## How to run the program

1. To run the program you'll need to have Python 3 installed on your computer as well as the following libraries:
- numpy
- math
- matplotlib
- scipy
- sklearn

2. Then, download the LC_NeuronClassifier.py, submission.mat and training.mat files all into the same folder on your computer.
3. Open the LC_NeuronClassifier.py file with a python IDE (e.g. Spyder, VS Code or Eclipse) and change the root_directory variable to the path of the folder where you put the files.
4. Click run and wait for the program to finish.
