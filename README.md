# min-char-rnn
Minimal character-level language model with a Vanilla Recurrent Neural Network, in Python/numpy

Reference page [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

## RNN/LSTM
Actually this model use the simple RNN, not using LSTM.
This model use the characters as input, then we use a one-hot vector as input X, the dimension of X is the size of characters in input file.

Let assume that the char vocab size is V, and hidden size is H, then output layer size is V.

This simple RNN model contain 3 matrices:
* Whh: H * H, hidden layer to hidden layer
* Wxh: V * H, input layer to hidden layer
* Why: V * H, hidden layer to output layer

In the output layer, softmax is used to compute the character probability distribution, then we could sample the next character according previous input.

## RNN Equation
**update hidden state**

h<sup>t</sup> = tanh(Whh * h<sup>t-1</sup> + Wxh*X)

**compute output vector**

y = Why * h

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

## Cost Function
This RNN model use cross entropy as cost function (error), cross entropy for one training data:
$$H(t, y) = -\sum_{i=1}^{V}t_{i}logy_{i}$$

Because in t, most of the value is 0, so, we could rewrite the above equation as:
$$H(t, y) = -t_{i}logy_{i}$$

Here i is the indice of 1 in vector t.

# Train RNN Model

* Install numpy
```
pip install numpy
```
* Run this code
```
python min-char-rnn.py
```

## Output of this model
The output of this model is sampled characters given current input characters.
Examples out output:
```
iter 92400, loss: 48.542305
---- sample -----
----
 cet for dons of he wast oune tofus shee loolf the was hering thity,
Youpres
To make his it you gain fell must out you yie t.
Wert; your geang'p his you cageiingeal my madm; -hat fould the conquall to  
----

```
