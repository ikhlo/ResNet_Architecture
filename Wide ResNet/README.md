# Wide Residual Neural Network

Here is the research paper explaining dense neural network and from which this code has been implemented : [Wide Residual Networks](https://arxiv.org/pdf/1605.07146v4.pdf)

Wide Residual Network is one of the first residual networks variant. It keeps the same idea, working with skip connections, however instead of increasing the deepness of the model, wide residual networks increase the network's width. So here we are not going to use a lot of layers, compare to resnet, but they become wider as we uses more filters. The advantage is that a wide network training is faster than a deep network because GPU is more efficient working in parallel computations with wider layers. An other advantage is the using of drop out method which reduce performance for basic resnet but not for wide resnet.

<br>

 ![](https://debuggercafe.com/wp-content/uploads/2021/03/wrn_vs_resnet.png)

<br>

## DenseNet Model

Here is a quick descritpion about the arguments of the **wide_resnet_model** function.

<ul>
    <li> <b>input_shape</b> : The shape of <b>an</b> individual in the dataset. It should be a 3-length tuple <i>(height, width, channels)</i> or <i>(channels, height, width)</i></li>
    <li> <b>nb_of_classes</b> : The possible number of outputs/classes/target. It should be an integer.</li>
    <li> <b>depth</b> : The number of layers inside the neural network. It should be an integer.</li>
    <li> <b>k</b> : The width ratio of the network. It should be an integer.</li>
    <li> <b>dropout_rate</b> : The rate of neurons inhibited by Dropout layers. It should be a float between 0.0 and 1.0.</li>
</ul>