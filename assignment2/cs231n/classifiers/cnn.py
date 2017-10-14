from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        # conv - relu - 2x2 max pool - affine - relu - affine - softmax
        # W1 b1 for convolutional layer
        self.params['W1'] = np.random.randn(num_filters,input_dim[0],filter_size,filter_size) * weight_scale
        self.params['b1'] = np.zeros(num_filters)
        # W2 b2 for hidden affine layer
        self.params['W2'] = np.random.randn(int(num_filters*input_dim[1]*input_dim[2]/4),hidden_dim) * weight_scale
        # input is N*F*H/2*W/2,output should be N*hidden_dim, see affine_forward document for W size
        self.params['b2'] = np.zeros(hidden_dim)
        # W3 b3 for output affine layer
        self.params['W3'] = np.random.randn(hidden_dim,num_classes) * weight_scale
        self.params['b3'] = np.zeros(num_classes)

        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        # conv - relu - 2x2 max pool - affine - relu - affine - softmax
        out,crp_cache = conv_relu_pool_forward(X,W1,b1,conv_param,pool_param)
        out,al_cache = affine_relu_forward(out,W2,b2)
        out,affine_cache = affine_forward(out,W3,b3)
        scores = out
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        # conv - relu - 2x2 max pool - affine - relu - affine - softmax
        loss,dsoft = softmax_loss(scores,y) 
        loss += 0.5*self.reg*(np.sum(self.params['W1']*self.params['W1'])+np.sum(self.params['W2']*self.params['W2'])+np.sum(self.params['W3']*self.params['W3']))
        dx,dw3,db3 = affine_backward(dsoft,affine_cache)
        grads['W3'] = dw3 + self.reg * self.params['W3']
        grads['b3'] = db3
        dx,dw2,db2 = affine_relu_backward(dx,al_cache)
        grads['W2'] = dw2 + self.reg * self.params['W2']
        grads['b2'] = db2
        _,dw1,db1 = conv_relu_pool_backward(dx,crp_cache)
        grads['W1'] = dw1 + self.reg * self.params['W1']
        grads['b1'] = db1
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
