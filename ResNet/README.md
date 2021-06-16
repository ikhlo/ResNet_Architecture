# Residual Neural Network

Here is the research paper explaining dense neural network and from which this code has been implemented : [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)

Residual Neural Network apparition was an unexpected breakthrough into AI domain, especially image recognition : it solves a known bottleneck in deep learning which is degradation problem. As neural networks became deeper and deeper with time, we figured out that at a specific moment, the accuracy get saturated and finally drop. To fix it, some researchers find a solution by adding skip connections which link shallow layers to deeper ones. The simple idea behind this is the following : as we give a correct previous state to the network, even with more training time and if it becomes deeper, the accuracy should not be worse than during previous steps.

<br>

 ![](https://images4.programmersought.com/767/4e/4ebe0ae4bc9aa6dba1a2c2d17895aa4f.png)


## DenseNet Model

Here is a quick descritpion about the arguments of the **resnet_model** function.

<ul>
    <li> <b>input_shape</b> : The shape of <b>an</b> individual in the dataset. It should be a 3-length tuple <i>(height, width, channels)</i> or <i>(channels, height, width)</i></li>
    <li> <b>nb_of_classes</b> : The possible number of outputs/classes/target. It should be an integer.</li>
</ul>