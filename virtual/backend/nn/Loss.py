import numpy as np 

class Loss: 
    def calculate(self, output, y, *, include_regularization=False): 
        sample_loss = self.forward(output, y) 
        data_loss = np.mean(sample_loss) 
        
        self.accumulated_sum += np.sum(sample_loss) 
        self.accumulated_count += len(sample_loss) 
        
        if not include_regularization: 
            return data_loss 
        
        return data_loss, self.regularization_loss()
    
    def calculate_accumulated(self, *, include_regularization=False): 
        data_loss = self.accumulated_sum / self.accumulated_count 
        
        if not include_regularization: 
            return data_loss 
        
        return data_loss, self.regularization_loss()
    
    # reset variables for accumulated loss 
    def new_pass(self): 
        self.accumulated_sum = 0 
        self.accumulated_count = 0 
        
    
    def remember_trainable_layers(self, trainable_layers): 
        self.trainable_layers = trainable_layers 

    def regularization_loss(self, layer): 
        # 0 by default 
        regularization_loss = 0 
        
        for layer in self.trainable_layers:        
            # l1 regularization - weights 
            if layer.weight_regularizer_l1 > 0: 
                regularization_loss += layer.weight_regularizer_l1 * \
                    np.sum(np.abs(layer.weights))
                    
            # l2 regularization - weights 
            if layer.weight_regularizer_l2 > 0: 
                regularization_loss += layer.weight_regularizer_l2 * \
                    np.sum(layer.weights * layer.weights)
                    
            # l1 regularization - biases 
            if layer.bias_regularizer_l1 > 0: 
                regularization_loss += layer.bias_regularizer_l1 * \
                    np.sum(np.abs(layer.bias))
                    
            # l2 regularization - biases 
            if layer.bias_regularizer_l2 > 0: 
                regularization_loss += layer.bias_regularizer_l2 * \
                    np.sum(layer.bias * layer.bias)
        
        return regularization_loss 


    
class CategoricalCrossEntropy(Loss): 
    def forward(self, output, target): 
        # samples in batch
        samples = len(output) 
        
        # clip data to prevent division by 0
        output_clip = np.clip(output, 1e-7, 1-1e-7) 
        
        # probabilities for target values
        
        # for categorical values 
        if len(target.shape) == 1: 
            confidences = output_clip[
                range(samples), 
                target
            ]
            
        # masked values - one hot
        elif len(target.shape) == 2: 
            confidences = np.sum(
                output_clip * target, 
                axis = 1
            )
            
        # Losses
        negative_log_likelihoods = -np.log(confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, target): 
        # number of samples
        samples = len(dvalues) 

        # number of labels in every sample
        labels = len(dvalues[0]) 

        # sparse -> turn into one hot 
        if len(target.shape) == 1: 
            target = np.eye(labels)[target] 

        # calculate gradient 
        self.dinputs = -target / dvalues 

        # normalize gradient
        self.dinputs = self.dinputs / samples 
        

class BinaryCrossentropy(Loss): 
    def forward(self, output, target): 
        output_clipped = np.clip(output, 1e-7, 1-1e-7) 
        sample_losses = -(target * np.log(output_clipped)) + \
                        (1 - target) * np.log(1 - output_clipped)
        return sample_losses 
    
    def backward(self, dvalues, target): 
        samples = len(dvalues) 
        outputs = len(dvalues[0])
        
        clipped_dvalues = np.clip(dvalues, 1e-7, 1-1e-7) 
        
        self.dinputs = -(target / clipped_dvalues - 
                         (1 - target) / (1 - clipped_dvalues)) / outputs
        
        self.dinputs = self.dinputs / samples 
        
class MeanSquaredError(Loss): 
    def forward(self, output, target): 
        sample_losses = np.mean((output - target) ** 2, axis=-1) 
        return sample_losses 
    
    def backward(self, dvalues, target): 
        samples = len(dvalues) 
        outputs = len(dvalues[0]) 
        
        self.dinputs = -2 * (target - dvalues) / outputs 
        self.dinputs = self.dinputs / samples
        
class MeanAbsoluteError(Loss): 
    def forward(self, output, target): 
        sample_losses = np.mean(np.abs(target - output), axis=-1) 
        return sample_losses
    
    def backward(self, dvalues, target): 
        samples = len(dvalues) 
        outputs = len(dvalues[0]) 
        
        self.dinputs = np.sign(target - dvalues) / outputs 
        self.dinputs = self.dinputs / samples 