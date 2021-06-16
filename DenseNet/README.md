# Dense Neural Network

Here is the research paper explaining dense neural network and from which this code has been implemented : [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)

Dense Neural Network are a variant of Residual Neural Network with a simple idea : as add shortcut between layer seems to perform well, we just have to connect each layer with all the previous ones. Those connections between layers, called **bottleneck layer**, are made through what we call a **dense block** which are themselves connected by **transition layers** inside the network. 

<br>

 ![](https://paperswithcode.com/media/methods/Screen_Shot_2020-06-20_at_11.35.53_PM_KroVKVL.png)

<br>

While resnets' layers are connected by adding the values of previous layer and the current one (so two inputs layer with shape (n, n, k) are added to give a final layer with output shape (n, n, k) with *n -> height, width and k -> depth*), inside **dense block** the layers are concatenated : two input layers with shape (n, n, k1) and (n, n, k2) are concatenated to give a final layer with output shape (n, n, k1+k2).

## DenseNet Model

Here is a quick descritpion about the arguments of the **densenet_model** function.

<ul>
    <li> <b>input_shape</b> : The shape of <b>an</b> individual in the dataset. It should be a 3-length tuple <i>(height, width, channels)</i> or <i>(channels, height, width)</i></li>
    <li> <b>nb_of_classes</b> : The possible number of outputs/classes/target. It should be an integer.</li>
    <li> <b>nb_dense_block</b> : The number of dense block inside the neural network. It should be an integer.</li>
    <li> <b>dropout_rate</b> : The rate of neurons inhibited by Dropout layers. It should be a float between 0.0 and 1.0.</li>
    <li> <b>nb_filters_list</b> : The number of filters used for each dense block. It should be an iterable of integer and this interable's size must equal with nb_dense_block value.</li>
</ul>