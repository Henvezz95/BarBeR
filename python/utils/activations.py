from torch import nn

act_dict = {
    'sigmoid': nn.Sigmoid(),                # For binary segmentation (for each channel)
    'softmax': nn.Softmax(dim=1),           # For multi-class segmentation
    'linear': nn.Identity(),                # No activation
    'tanh': nn.Tanh(),                      # Useful in some cases (e.g. if output must be between -1 and 1)
    'relu': nn.ReLU(),                      # Rectified linear unit
    'leaky_relu': nn.LeakyReLU(),           # Similar to ReLU, prevents dead neurons
    'softplus': nn.Softplus(),              # Smooth version of ReLU
    'hard_sigmoid': nn.Hardsigmoid(),       # Approximation of Sigmoid, faster
    'hard_tanh': nn.Hardtanh(),             # Approximation of Tanh, avoids saturation
    'log_softmax': nn.LogSoftmax(dim=1),    # Sometimes useful for numerical stability
}