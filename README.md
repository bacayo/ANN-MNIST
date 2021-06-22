### Building Artificial Neural Network on MNIST dataset

- The ANN structure build with 3 layer with one output layer within the layers
- The first input and hidden layer has 128 neuron, second layer has 256 neurons, the output layer has 10 neuron according to label column in dataset
- The learning rate is calculated 0.001 in terms of the adam algoritm, which is used as optimizer when compiling the ann

* if **sigmoid** function used as activation, the f1 score calculated 0.978, test accuracy 0.978 and total loss 0.134
* if **relu** function used as activation, the f1 score calculated 0.098, test accuracy 0.098 and total loss 2.303
* if **tanh** function used as activation, the f1 score calculated 0.223, test accuracy 0.223 and total loss 2.303
* The highest classification performance is provided by sigmoid function, the lowest perfomance is tanh function

# Additional notes
- if **relu** function is used only hidden layers and **sigmoid** function is used output layer, the ann structure performance is increased significantly according to f1 score: 0.98
