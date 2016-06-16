# min-char-rnn
Minimal character-level language model with a Vanilla Recurrent Neural Network, in Python/numpy

## RNN/LSTM
Actually this model use the simple RNN, not using LSTM.
This model use the characters as input, then we use a one-hot vector as input X, the dimension of X is the size of characters in input file.
Let assume that the char vocab size is V, and hidden size is H, then output layer size is V.
This simple RNN model contain 3 matrices:
* Whh: H * H, hidden layer to hidden layer
* Wxh: V * H, input layer to hidden layer
* Why: V * H, hidden layer to output layer

In the output layer, softmax is used to compute the character probability distribution, then we could sample the next character according previous input.

