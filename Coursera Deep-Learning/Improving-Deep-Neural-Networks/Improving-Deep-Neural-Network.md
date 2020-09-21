# **Improving Deep Neural Network**

Week 1:

1. Train/dev/test sets:
  1. Make sure that your test and dev sets come from the same distribution
  2. Not having a test set might be okay
  3. Dev set for cross validation
2. Bias/Variance:
  1. Problem with high bias – underfitting model
  2. Problem with high variance – overfitting model
3. Basic recipe for ML:
  1. High bias:
    1. Make bigger network (More hidden units or layer)
    2. Train longer
  2. High variance:
    1. Need more data
    2. Regularization
    3. Find more appropriate neural model
4. Regularizing NN:
  1. L2 regularization:
    1. We wish to minimize cost function, for L2 we add component that penalize large weights, where lambda is regularization parameter Now, _lambda_ is a parameter than can be tuned. Larger weight values will be more penalized if the value of _lambda_ is large. Similarly, for a smaller value of _lambda_, the regularization effect is smaller.
  2. Dropout regularization:
    1. More technically, At each training stage, individual nodes are either dropped out of the net with probability _1-p_ or kept with probability _p_, so that a reduced network is left; incoming and outgoing edges to a dropped-out node are also removed.
5. Normalizing inputs:
  1. The first is to subtract out or to zero out the mean
  2. The second step is to normalize the variances
6. Vanishing/Exploding Gradients:
  1. In a network of _n_ hidden layers, _n _derivatives will be multiplied together. If the derivatives are large then the gradient will increase exponentially as we propagate down the model until they eventually explode, and this is what we call the problem of _ **exploding gradient** _.
  2. Alternatively, if the derivatives are small then the gradient will decrease exponentially as we propagate through the model until it eventually vanishes, and this is the _ **vanishing gradient** _ problem.
7. Gradient checking:
  1. Don&#39;t use in training only to debug
  2. If algorithms fails grad check look at components to try to identify bug
  3. Does not work with dropout

Week 2:

1. Gradient descent:
  1. **In Batch Gradient Descent** , all the training data is taken into consideration to take a single step. We take the average of the gradients of all the training examples and then use that mean gradient to update our parameters. So that&#39;s just one step of gradient descent in one epoch.
  2. **Stochastic Gradient Descent -** Suppose our dataset has 5 million examples, then just to take one step the model will have to calculate the gradients of all the 5 million examples. This does not seem an efficient way. To tackle this problem we have Stochastic Gradient Descent. In Stochastic Gradient Descent (SGD), we consider just one example at a time to take a single step. We do the following steps in  **one epoch**  for SGD:
    1. Take an example
    2. Feed it to Neural Network
    3. Calculate it&#39;s gradient
    4. Use the gradient we calculated in step 3 to update the weights
    5. Repeat steps 1–4 for all the examples in training dataset
  3. **Mini batch -** Neither we use all the dataset all at once nor we use the single example at a time. We use a batch of a fixed number of training examples which is less than the actual dataset and call it a mini-batch. Doing this helps us achieve the advantages of both the former variants we saw. (The same steps like in SGD)
2. Choosing your mini-batch size:
  1. If small train set – use Batch Gradient Descent (\&lt;= 2000)
  2. Typical mini-batch size: 64, 128,256, 512, 1024
  3. Make sure your mini-batch fit in CPU/GPU memory
3. Gradient descent with momentum:
  1. Compute dW, db on current mini-batch
  2. Use Exponentially weighting averages:
    1. Vdw = Bvdw + (1-B)dW
    2. Vdb = Bvdb + (1-B)db
    3. W = W – avdw, b = b-avdb
    4. B = 0.9
4. Adam optimization algorithm:
  1. Vdw = 0, Vdb = 0, Sdw = 0, Sdb = 0
  2. Compute dW and db using current mini batch
  3. Vdw = BVdw + (1-B)dW, Vdb = BVdb +(1-B)db
  4. Sdw = B2 Vdw + (1-B)dW\*\*2, Sdb = B2 Vdw + (1-B)db\*\*2
  5. V(correction) = Vdw/(1-B\*\*t), Vdb = Vd b/(1-B\*\*t)
  6. Sd = Sdw/(1-B2\*\*t), Sdb = Sdb(1-B2\*\*t)
  7. W = W - alpha\*(Vdw/ sqrt(Swd)+ e), b = W – alpha(Vdb/sqrt(Sdb)+e)
5. Learning rate decay:
  1. 1 epoch = 1 pass through the date
  2. Learning rate = 1/ (1+ rate-decay \* epoch-num) \* learning-rate[0])

Week 3:

1. Tuning process:
  1. &quot;Coarse to Fine&quot; usually refers to the hyperparameter optimization of a neural network during which you would like to try out different combinations of the hyperparameters and evaluate the performance of the network.However, due to the large number of parameters AND the big range of their values, it is almost impossible to check all the available combinations. For that reason, you usually discretize the available value range of each parameter into a &quot;coarse&quot; grid of values (i.e. val = 5,6,7,8,9) to estimate the effect of increasing or decreasing the value of that parameter. After selecting the value that seems most promising/meaningful (i.e. val = 6), you perform a &quot;finer&quot; search around it (i.e. val = 5.8, 5.9, 6.0, 6.1, 6.2) to optimize even further.
2. Normalizing activations in a network:
  1. Batch normalization allows each layer of a network to learn by itself a little bit more independently of other layers.
  2. It reduces overfitting because it has a slight regularization effects. Similar to dropout, it adds some noise to each hidden layer&#39;s activations.
  3. To increase the stability of a neural network, batch normalization normalizes the output of a previous activation layer by subtracting the batch mean and dividing by the batch standard deviation.
3. Softmax regression - The Softmax regression is a form of logistic regression that normalizes an input value into a vector of values that follows a probability distribution whose total sums up to 1. The output values are between the range [0,1] which is nice because we are able to avoid binary classification and accommodate as many classes or dimensions in our neural network model. This is why softmax is sometimes referred to as a multinomial logistic regression.
4. Writing and running programs in TensorFlow has the following steps:

1. Create Tensors (variables) that are not yet executed/evaluated. Tf.Variable()
2. Write operations between those Tensors. Tf.add()/ Tf.multiply
3. Initialize your Tensors. Tf.global\_variable\_initializers()
4. Create a Session. With tf.session() as session
5. Run the Session. This will run the operations you&#39;d written above. Session.run()

1. Tf.place\_holder(tf.float32, name=) - A Tensor that may be used as a handle for feeding a value, but not evaluated directly.
2. tf.one\_hot(labels, C, axis=0)
3. tf.nn.relu(Z) – relu function
4. tf.nn.sigmoid(Z) – sigmoid
5. tf.train.GradientDescentOptimizer(learning\_rate = learning\_rate).minimize(cost)
6. tf.train.AdamOptimizer(learning\_rate = learning\_rate).minimize(cost)