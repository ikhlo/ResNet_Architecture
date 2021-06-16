# Dense Neural Network

Dense Neural Network are a variant of Residual Neural Network with a simple idea : as add shortcut between layer seems to perform well, we just have to connect each layer with all the previous ones. Those connections between layers, called **bottleneck layer**, are made through what we call a **dense block** which are themselves connected by **transition layers** inside the network. 

<br><br>

 ![](https://paperswithcode.com/media/methods/Screen_Shot_2020-06-20_at_11.35.53_PM_KroVKVL.png)

<br><br>

While resnets' layers are connected by adding the values of previous layer and the current one (so two inputs layer with shape (n, n, k) are added to give a final layer with output shape (n, n, k) with *n -> height, width and k -> depth*), inside **dense block** the layers are concatenated : two input layers with shape (n, n, k1) and (n, n, k2) are concatenated to give a final layer with output shape (n, n, k1+k2).