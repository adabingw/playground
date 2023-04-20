import numpy as np 

class ReLU: 
    def predictions(self, outputs): 
        return outputs 
    
    def forward(self, input): 
        self.inputs = input 
        self.output = np.maximum(0, input)
    def backward(self, dvalues): 
        self.dinputs = dvalues.copy() 
        
        # 0 gradient when input values are negative
        self.dinputs[self.inputs <= 0] = 0
    
class Softmax: 
    def predictions(self, outputs): 
        return np.argmax(outputs, axis=1)
        
    def forward(self, input): 
        self.inputs = input 
        exp = np.exp(input - np.max(input, axis=1, keepdims=True)) 
        probabilities = exp / np.sum(exp, axis=1, keepdims=True) 
        self.ouput = probabilities 
        
    def backward(self, dvalues): 
        # create uninitialized array 
        self.dinputs = np.empty_like(dvalues) 

        # enumerate outputs and gradients 
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)): 
            # flatten output array 
            single_output = single_output.reshape(-1, 1) 

            # calculate jacobian matrix of output and 
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T) 
        
            # calcualte sample-wise gradient 
            # add it to the array of sample gradients 
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues) 

class Sigmoid: 
    def predictions(self, outputs): 
        return (outputs > 0.5) * 1 
        
    def forward(self, inputs): 
        self.inputs = inputs 
        self.output = 1 / (1 + np.exp(-inputs)) 
        
    def backward(self, dvalues):  
        self.dinputs = dvalues * (1 - self.output) * self.output 
        

class Linear: 
    def predictions(self, outputs): 
        return outputs
    
    def forward(self, inputs): 
        self.inputs = inputs 
        self.output = inputs 
        
    def backward(self, dvalues): 
        self.dinputs = dvalues.copy()