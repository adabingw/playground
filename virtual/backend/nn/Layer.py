import numpy as np 

class Dense: 
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0, 
                 bias_regularizer_l1=0, bias_regularizer_l2=0): 
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) 
        self.bias = np.zeros((1, n_neurons)) 
        
        self.weight_regularizer_l1 = weight_regularizer_l1 
        self.weight_regularizer_l2 = weight_regularizer_l2 
        self.bias_regularizer_l1 = bias_regularizer_l1 
        self.bias_regularizer_l2 = bias_regularizer_l2
        
    def get_parameters(self): 
        return self.weights, self.biases

    def set_parameters(self, weights, biases): 
        self.weights = weights 
        self.biases = biases 
        
    def forward(self, inputs):
        self.inputs = inputs 
        self.output = np.dot(inputs, self.weights) + self.biases
        
    # backward pass 
    def backward(self, dvalues): 
        # gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues) 
        self.dbias = np.sum(dvalues, axis=0, keepdims=True) 
        
        # gradients on regularization 
        # l1 on weights 
        if self.weight_regularizer_l1 > 0: 
            dL1 = np.ones_like(self.weights) 
            dL1[self.weights < 0] = -1 
            self.dweights += self.weight_regularizer_l1 * dL1 
            
        # l2 on weights 
        if self.weight_regularizer_l2 > 0: 
            self.dweights += 2 * self.weight_regularizer_l2 * \
                    self.weights 

        # l1 on bias 
        if self.bias_regularizer_l1 > 0: 
            dL1 = np.ones_like(self.bias) 
            dL1[self.bias < 0] = -1 
            self.dbias += self.bias_regularizer_l1 * dL1 
            
        # l2 on bias 
        if self.bias_regularizer_l2 > 0: 
            self.dbias += 2 * self.bias_regularizer_l2 * \
                    self.bias 

        
        # gradients on values
        self.dinputs = np.dot(dvalues, self.weights.T)
        
class Dropout: 
    def __init__(self, rate): 
        self.rate = 1 - rate
    
    def forward(self, inputs, training): 
        self.inputs = inputs 
        
        if not training: 
            self.output = inputs.copy()
            return 
        
        self.binary_mask = np.random.binomial(1, self.rate, 
                        size=inputs.shape) / self.rate 
        self.output = inputs * self.binary_mask 
    
    def backward(self, dvalues): 
        self.dinputs = dvalues * self.binary_mask 
    
class Input: 
    def forward(self, inputs): 
        self.output = inputs 