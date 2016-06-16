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

## Training data
This model use the characters in input.txt file, use current character and next character as a training data pair. Each character is represented by one hot vector, target value means which character we expected given current character.

For example:
in input file, we read in "hello", then 'h' and 'e' will be used as a training pair, and will be encoded into vectors. 
Let's assume that we only have 4 characters in our vocab, ('h','e','l','o'), then 'h' will be encoded to [1,0,0,0]<sup>T</sup>, 'e' will be encoded to [0,1,0,0]<sup>T</sup>

## Input Layer
Input layer size is V, then input value is a V * 1 one hot vector.

## Hidden Layer
Hidden layer size is H, we also need to record the hidden state(value of hidden layer).

## Output Layer
Output layer size is V, we get a character probability distribution in output layer, then we could sample a character in this probability distribution given a sequence of input.



