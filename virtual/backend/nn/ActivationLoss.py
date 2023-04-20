# softmax classifier - combined softmax activation 
# and cross-entropy loss for faster backward step 
import numpy as np 

from ActivationFunction import Softmax 
from Loss import CategoricalCrossEntropy

class Softmax_CategoricalCrossEntropy():     
    def backward(self, dvalues, target): 
        samples = len(dvalues) 
        
        if len(target.shape) == 2: 
            target = np.argmax(target, axis=1) 
        
        self.dinputs = dvalues.copy() 
        
        # calcualte gradient 
        self.dinputs[range(samples), target] -= 1 
        
        # normalize gradient 
        self.dinputs = self.dinputs / samples 