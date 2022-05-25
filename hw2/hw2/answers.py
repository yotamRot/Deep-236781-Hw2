r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**
1. 

A.

Y shape = [N,out_feature] = $[64,512]$  

X shape = [N,in_feature]  = $[64,1024]$ 
    
Since the jacobian is constructed by the derivative of every element in the output according to every element in 
the input the dimension that we will get is $[64\cdot512 , 64\cdot1024]$

B.

Since the layer is fully connected, every element in Y is connected to every element in X.
Therefore since element i,j in jacobian represents derivative of Yi by Xj we will get relevant weight according 
to the relation between X and Y. 
So the matrix would **not be sparse** because we have a connection between every two elements in X and Y(the layer is fully connected).
Then we will get a weight parameter (non zero value) in each element in jacobian.

C. 
No we don't have to materialize this jacobian.
According to matrix multiplation and chain rules $\delta\mat{X} =\pderiv{L}{\mat{Y}}W^T$
Therefore instead of materialization the entire jacobian matrix we can just use use expression written above because we 
know both $\pderiv{L}{\mat{Y}}$ and $W^T$



2.

 A.

Y shape = [N,out_feature] = $[64,512]$  

W shape = [out_feature,in_feature]  = $[512,1024]$ 
    
Since the jacobian is constructed by the derivative of every element in the output according to every element in 
the weight matrix the dimension that we will get is $[64\cdot512 , 512\cdot1024]$

B.

Since the layer is fully connected, every element in Y is connected to every element in X.

Therefore since element i,j in jacobian represents derivative of Yi by Wij we will get relevant X according 
to the relation between Yi and Wij (will give as Xj). 
But for $Yk \neq Yi$  the derivative of Yk by Wij will give as 0 because this weight is not part of the connection between 
Yk and Xj.
Therefore the matrix **would be sparse** because only derivatives between some Yi element and weights that
represents connection between same Yi and some X will not be zero. 
So since many elements do not hold this connection we will get a lot of zeroes.

C. 
No we don't have to materialize this jacobian.
According to matrix multiplication and chain rules $\delta\mat{W} = X^T\pderiv{L}{\mat{Y}}$
Therefore instead of materialization the entire jacobian matrix we can just use use expression written above because we 
know both $\pderiv{L}{\mat{Y}}$ and $X^T$
"""

part1_q2 = r"""
**Your answer:**

As we have learned the purpose of the gradient decent is to find the minimum of the loss function. 

As we saw at the tutorial "Back-propagation is an efficient way to calculate these gradients using the chain rule".

Therefore it is just a method and another methods can be used so it is **not required**.

Alternatively the gradient can be simply computed invoking the chain rule, but it wouldn't be efficient.

Another way that we found is in the following article: https://arxiv.org/abs/2202.08587 .

"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.01
    lr = 0.01
    reg = 0
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.2
    lr_vanilla = 0.02
    lr_momentum = 0.002
    lr_rmsprop = 0.0002
    reg = 0.002
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.2
    lr = 0.002
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**

1. The purpose of dropout is to prevent over-fitting to our train data.

In the graphs we can see that:

not using dropout:

-give us high accuracy and low loss on the train set but it gives us bad results at the test set since we over-fit.

using dropout:

In general we expect to see an improvement on test set, but to get worst result on train set.

As we can see:

-low dropout give us good result on test set - meaning we improved the over-fitting issue on the train set.
  
-high dropout decrease the over-fit as we can see in the graph, but since high dropout result that are worse then low
 dropout (both on train and test) we can understand that we under-fit our data in the training process because we 
 ignored too many neurons.
 
 **this is what we expected to see**, using dropout can improve our test results by decreasing the over-fitting 
 on the train set, But the choice of parameter dropout should be made carefully.
 
2. as explained at 1. both dropout decrease the over-fitting on the train set, but low dropout gives better results both
on train and test. 

That's because using high dropout might cause under-fitting to our data meaning we cannot generalize new data.    


"""

part2_q2 = r"""
**Your answer:**

**Yes it is possible.**

Cross-entropy is evaluated loss while considering scores for all labels and the ground true. 
on the other hand, accuracy depends only on the highest score label and the ground true.

For example: Suppose we have 2 classes triangle and circle, and the data is circle, circle.
for first input:
So we might get the output scores triangle: 0.1, circle: 0.9, than the cross-entropy loss is ~0.152 and the prediction 
is correct.
for second input:
we might get the output scores triangle: 0.51, circle: 0.49, than the cross-entropy loss is ~1.029 and the prediction 
isn't correct

We will get accuracy of 50% and average loss of ~0.5905.

In second epoch:
And for both inputs the output scores: triangle: 0.4, circle: 0.6, than the cross-entropy loss is ~0.736 and the 
prediction is correct.

We will get accuracy of 100% and average loss of ~0.736.

As we can see both prediction are correct and both accuracy and loss result increased. 


"""

part2_q3 = r"""
**Your answer:**

1. Gradient decent is an optimization method used to minimized function by repeatedly moving to the direction opposite
to the gradient.
On the other hand backpropagation is an efficient way of computing gradients.
Therefore the main difference is that gradient decent is a method to find the minimum and backpropagation is an 
efficient way that might be used by gradient decent to find the direction to the minimum efficiently from a given 
location.

2. Both gradient descent and stochastic gradient descent are optimization algorithms to find the minimum of function.

The main difference is that **gradient descent** take in to account **all samples** in the training set for updating 
parameter in some iteration, while in **stochastic gradient descent** we are using **one or subset of training samples**
from the training set for updating a parameter in some iteration.
The Advantage of SGD is that it can escape more easily from local minima.

Furthermore SGD will be more efficient because less calculation is made but it will be more noisy because we are not 
taking into account all samples

3. SGD is used more often because it has the following benefits:

- Faster because its less calculation is done in each iteration.

- Can be used for large training sets because we dont need to save all data in memory in each iteration.

- As we have learn the nosiness of SGD helps to escape local minima.


4. 1.Yes both options GD and given suggestion will produce equal gradient. This will happen because of the way forward 
and backpropagation stages depend on each other. First we do forward pass calculation and save all values. Afterwards 
we start running backpropagation to calculate the gradient. Gradient values depends on forward stage calculation but 
since we already made those calculations all information needed to calculate gradient already exists. New suggestion 
will give same gradient because although forward pass it done in multiple steps it is finished before starting 
backpropagation. So when we start backpropagation all values required already exists and network is in same as if we 
were using Normal GD. 

2. As explained above since backpropagation stage calculation depends on forward pass calculation related to all data 
after each forward batch we still need to remember it's calculation in the network. Therefore we will need to remember 
a lot of data related to hidden layers variables. So although we did not save all dataset in memory we needed to save 
a lot of data related to our model since backpropagation needed to use him because it depends on him.


"""


# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    n_layers = 3
    hidden_dims = 10
    activation = "relu"
    out_activation = "none"
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    lr = 0.001
    weight_decay = 0.005
    momentum = 0.9
    loss_fn = torch.nn.CrossEntropyLoss()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**

1. Optimization error is a measure of error resulting from not finding the exact minimizer for training data loss.
As we can see from plots we reached pretty good results in manner of lost and acc. 
Results are not perfect so we might reached local minima but still global minima is not much deeper the one optimizer found.
So our optimization error is not high.

2. Generalization error is a measure of error resulting from the fact that we use sample as a substitute for the true 
distribution and our inability to find the optimal parametric model.
As we can see from a plots obviously we did not predict perfectly the validation set, but the result over validation set
(unseen data) in general are not so different from train results until some epoch. 
From some epoch we can see that our model started to over fit the train data because validation result decrease 
comparing train result.
So from that point generalization error got bigger because of the fact that we train the model over sample and not using
real distribution.
To conclude our model was able to generalize result over new data quite good and we did not over fitted too much over 
the train set so our generalization error is not high.
    
3. Approximation error is a measure of error resulting from the fact that we limited our self to some family of models,
where the optimal model may not be found.
As we can see from plot we got pretty good result meaning that our model was not so far from optimal one.
But when we look at decision boundary plot, we can see that there are some areas in which we are mistaken
(mistaken for inputs that have similar features) therefore approximation error that we do have is probably caused 
by the fact that our model could not identify this "area" of true labels.
"""

part3_q2 = r"""
**Your answer:**

According to the train confusion matrix we can see that the FNR is the common kind of mistake.
Therefore when we apply the model on the validation data which is noisier we expect to get more FNR then FPR.
That because we knew that the FNR was bigger then the FPR in the training phase so it will increase using noisier unseen data. 

"""

part3_q3 = r"""
**Your answer:**

When we choose the optimal threshold above it was important for us to decrease the sum of the FNR and FPR because both 
were equally important.

A. In this scenario we know that the disease is not lethal therefore a patient can be with the disease and it wont be so bad.
We have to remember that our goal is to reduce the cost of the tests Therefore we prefer to have more sick patient that
was classified 
as not sick (FNR) then Patient that will predict as sick and after the expensive test will be found not sick (FPR).
All of this is because that the disease is not lethal so we would like to **decrease the FPR** the this is how we will
choose the threshold.


B. Obviously in this scenario since we want to keep people alive because life as no price, we would prefer to classify a
healthy people as sick. Also an important fact that we know is that the expansive tests are involve **high risk to the patient**
so sending healthy person to this test is something we would like to avoid (because we care about life).
To conclude in this case both FPR and FNR are important with the same weight so we would use the same method that we 
choose above, **same optimal point** on the ROC curve. 

"""


part3_q4 = r"""
**Your answer:**

1. As we can see at the results while the depth is fixed we are improving in general our test accuracy while increasing 
the width of the model. Furthermore we can see that the decision boundary is getting more curvy.
In depth 1 we can see that the result is getting better all the way as we are increasing the width of the model, because 
as we have learned the model become more expressive and can generalize complex data in a better way.
While in depth 2, 4 we have reach the peak in low width value and we dont get improvement afterwards meaning that our
model was expressive enough in early stage.

2. As we can when the model is becoming more complex until a certain point we get better result because the model is 
more expressive and can represent a complex function but after that point we might start to over-fit to our train set 
therefore the results are getting worse. In width 2 we can see the improvement in a clear way (as the depth increase the
result is better), while in the other width the increasing of the depth cause the model to be more complex which case 
to over-fit and the decrease in accuracy. As we can see in row 2,3,4 the best result is not the most complex model 
(in row 3 and 4 we are getting to the best result in the first depth because its expressive enough).

3.

depth=1, width=32 and depth=4, width=8: The better test result is for depth=1, width=32

depth=1, width=128 and depth=4, width=32: The better test result is for depth=1, width=128

for both of them we can explain that the result was better in the wider networks the following way. 
Wider networks are very good at memorization and deeper are better in generalization. So in our model in the conflict between memorization and 
generalization the wider model achieve the best result because although the fact that it generalize worse, it was less
affective.


4. Yes as we can see in all the tests we got better results using the optimal threshold. This happened because we choose 
this threshold in a way to minimize both FNR and FPR over the validation set.
Since the sets was selected randomly we got the improvement in the test set thanks to the optimal threshold.
"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    lr = 0.001
    weight_decay = 0.005
    momentum = 0.9
    loss_fn = torch.nn.CrossEntropyLoss()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**
1. Number of parameters after each convolution is:

 $K \cdot (C_in \cdot F^2 + 1) = (channels_out \cdot ((channels_in \cdot width \cdot height) + 1)$

Example 1 - Regular  block
    First Conv layer:
    
   $ num of parameters = 256 \cdot ((256 \cdot 3 \cdot 3) + 1) = 590,080 $
   
    Second Conv layer:
    
   $ num of parameters = 256 \cdot ((256 \cdot 3 \cdot 3) + 1) = 590,080 $
   
   So in total $ num of parameters = 590,080 \cdot 2 = 1,180,160$
   
Example 2 - BottleNeck block

    First Conv layer:
    
   $ num of parameters = 64 \cdot ((256 \cdot 1 \cdot 1) + 1) = 16,448 $
   
    Second Conv layer:
    
   $ num of parameters = 64 \cdot ((64 \cdot 3 \cdot 3) + 1) = 36,928 $
   
   Third Conv layer:
   
   $ num of parameters = 256 \cdot ((64 \cdot 1 \cdot 1) + 1) = 16,640 $
   
   So in total $ num of parameters = 16,448 + 36,928 + 16,640 = 70,016 $
   
   
   To conclude we can see that in the **bottleneck there are fewer parameters**
    
    
    2. We know that FLOPs are basic math operations and RELU activation function.
    To calculate one convolution layer between ${C_in,H_in,W_in}$ to $C_out,H_out,W_out$ we have  
    $ 2\cdot c_in \codt k^2 \cdot c_out \cdot W_out \cdot H_out$ floating point operations,
    the 2 is for both sum and mul operation witch are require in convolution.
    Also activation function (RELU) of each layer require $H_out \cdot W_out$ FLOPs.
    We can see that for the regular block there are alot more FLOPs then the bottleneck block.
    Therefore the **bottleneck is much lighter in FLOPs**.
    
    3. 
    **Spatial**
    In the **regular block** we have two convolution layers of 3x3 therefore the respective field is 5x5.
       In the **bottleneck block** we have two convolution layers of 1x1 and one convolution layer of 3x3 therefore the
       respective field is 3x3. We can conclude that the **regular block combine the input better in terms of spatial**.
       
       
       **Across feature map**
       Since we project in **bottleneck block** the first layer to smaller dimension, not all inputs has the same 
       influence across feature map, on the other hand in the **regular block** since we don't project the input have 
       the same influence across feature map. 
"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**
1. As we can see from the graph increasing the depth (L) doesn't necessarily improve for both values of K.
with L=2 we get the best result therefore increasing the depth doesn't help to improve the results.
with L=8,16 the model is not trainable -> will explain in the next section.

2. **YES** for L=8,16 we get un trainable model and we believe that the cause of this was "vanishing gradient problem".
This might happened in deep networks were the gradient is getting very close to 0 so the gradient information failed to
reach the beginning of the network in back propagation process

Suggestion for solving the problem:

A. Residual block - skipping layers will help overcome this issue

B. Batch normalization - will help the derivative to stay in good range

"""

part5_q2 = r"""
**Your answer:**

We can see from the graphs that while we increase K we get better results for L=2,4.
For L=8 we are the same issue that we had in the previous experiment when the model was un trainable.
And compare to the results in exp1_1 for the trainable model here we got a little improvement.

"""

part5_q3 = r"""
**Your answer:**

As we can see from the graph using **L=1** is the best result, the explanation is similar as was in experiment 1.1.
After L=1 we get decrease in results and in L=3,4 the model getting too complex (many features) until its 
un trainable (Probably vanishing gradient issue again).
This is fit to the results that we get until now from the previous experiments.

"""

part5_q4 = r"""
**Your answer:**

First of all we can see that we don't have the issue of un trainable model. we believe that this happened because resnet
is skipping connection as we suggested in question 1 part 2.
Therefore we can use complex models and deeper networks and achieve better results compare to the previous experiments.

Secondly, we got much better results in compare to previous experiments probably because what we mentioned above.
 
"""

part5_q5 = r"""
**Your answer:**
1. In YourCnn we decided to use the following:
    a. Dropout - To prevent over-fit over our train set, we played with it's value until we found an optimal one.
    b. Batch normalization - to make our learning process faster and more stable.
    c. Residual Block - to prevent issues like had in previous experiments where the model was un trainable.
    
2. As we can see in graphs the modifications gave us much better results then Exp1 result.
    Best results was given for L=12 about 82% accuracy.

"""
# ==============
