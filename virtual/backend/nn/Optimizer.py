import numpy as np 

class SGD: 
    def __init__(self, learning_rate=1.0, decay=0., momentum=0.): 
        self.learning_rate = learning_rate 
        self.current_learning_rate = learning_rate 
        self.decay = decay 
        self.iterations = 0 
        self.momentum = momentum 

    def pre_update_params(self): 
        if self.decay: 
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
    
    def update_params(self, layer): 
        if self.momentum: 
            # if layer doesn't contain momentum arrays, create them 
            if not hasattr(layer, 'weight_momentums'): 
                layer.weight_momentums = np.zeroes_like(layer.weights) 
                layer.bias_momentum = np.zeroes_like(layer.biases)
        
            weight_updates = self.momentum * layer.weight_momentums - \
                            self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates 

            bias_updates = self.momentum * layer.bias_momentums - \
                            self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates 
        
        # vanilla
        else:            
            weight_updates = -self.current_learning_rate * \
                                layer.dweights 
            bias_updates = -self.current_learning_rate * \
                                layer.dbiases    
            
        layer.weights += -self.learning_rate * layer.dweights 
        layer.biases += -self.learning_rate * layer.dbiases 
        
    def post_update_params(self): 
        self.iterations += 1 
        

class Adagrad: 
    def __init__(self, learning_rate=1., decay = 0., epsilon=13-7): 
        self.learning_rate = learning_rate 
        self.current_learning_rate = learning_rate 
        self.decay = decay 
        self.iterations = 0 
        self.epsilon = epsilon 
    
    def pre_update_params(self): 
        if self.decay: 
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations)) 
        
    def update_params(self, layer): 
        if not hasattr(layer, 'weight_cache'): 
            layer.weight_cache = np.zeroes_like(layer.weights) 
            layer.bias_cache = np.zeroes_like(layer.biases) 
        
        layer.weight_cache += layer.dweights ** 2 
        layer.bias_cache += layer.dbiases ** 2 
        
        layer.weights += -self.current_learning_rate *  \
                        layer.dweights / \
                        (np.sqrt(layer.weight_cache) + self.epsilon)
                        
    def post_update_params(self): 
        self.iterations += 1 
        

class RMSprop: 
    
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate 
        self.current_learning_rate = learning_rate 
        self.decay = decay 
        self.iterations = 0 
        self.epsilon = epsilon 
        self.rho = rho 
    
    def pre_update_params(self): 
        if self.decay: 
            self.current_learning_rate = self.learning_rate *  \
                    (1. / (1. + self.decay * self.iterations))
        
    def update_params(self, layer): 
        if not hasattr(layer, 'weight_cache'): 
            layer.weight_cache = np.zeroes_like(layer.weights) 
            layer.bias_cache = np.zeroes_like(layer.biases) 
            
        layer.weight_cache = self.rho * layer.weight_cache + \
            (1 - self.rho) * layer.dweights ** 2 
        layer.bias_cache = self.rho * layer.bias_cache + \
            (1 - self.rho) * layer.dbiases ** 2
            
        layer.weights += -self.current_learning_rate * \
                    layer.dweights / \
                    (np.sqrt(layer.weight_cache) + self.epsilon) 
        layer.biases =+ -self.current_learning_rate *  \
                    layer.biases / \
                    (np.sqrt(layer.boas_cache) + self.epsilon)
                    
class Adam: 
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, 
                 beta_1=0.9, beta_2=0.999): 
        self.learning_rate = learning_rate 
        self.current_learning_rate = learning_rate 
        self.decay = decay 
        self.iterations = 0 
        self.epsilon = epsilon 
        self.beta_1 = beta_1 
        self.beta_2 = beta_2 
        
    def pre_update_params(self): 
        if self.decay: 
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))
    
    def update_params(self, layer): 
        if not hasattr(layer, 'weight_cache'): 
            layer.weight_momentums = np.zeroes_like(layer.weights) 
            layer.weight_cache = np.zeroes_like(layer.weights) 
            layer.bias_momentums = np.zeroes_like(layer.biases) 
            layer.bias_cache = np.zeroes_like(layer.biases) 
            
        layer.weight_momentums = self.beta_1 * \
                        layer.weight_momentums + \
                        (1 - self.beta_1) * layer.dweights 
        layer.bias_momentums = self.beta_1 * \
                        layer.bias_momentums + \
                        (1 - self.beta_1) * layer.dbiases 

        weight_momentums_corrected = layer.weight_momentums / \
                (1 - self.beta_1 ** (self.iterations + 1)) 
        bias_momentums_corrected = layer.bias_momentums / \
                (1 - self.beta_1 ** (self.iterations + 1)) 
        
        weight_cache_corrected = layer.weight_cache / \
                (1 - self.beta_2 ** (self.iterations + 1)) 
        bias_cache_corrected = layer.bias_cache / \
                (1 - self.beta_2 ** (self.iterations + 1)) 
                
        layer.weights += -self.current_learning_rate * \
                weight_momentums_corrected / \
                (np.sqrt(weight_cache_corrected) + self.epsilon) 
        layer.biases += -self.current_learning_rate * \
                bias_momentums_corrected / \
                (np.sqrt(bias_cache_corrected) + self.epsilon) 
        
    def post_update_params(self): 
        self.iterations += 1 