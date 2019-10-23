# A Tensorflow Optimizer for Deep Neural Networks 

Stochastic Optimization by Simulated Annealing and Langevin Dynamics


### Requirements
Python (3 or higher)
Tensorflow
  
### How to use
```python
opt=SALDOptimizer(learning_rate=0.1, T0=1e-6, fric=100.0, beta=0.99995)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
```

T0 is the temperature. Default T0=1e-6\ 
beta is a Simulated Annealing parameter. In the range from 0.9999 to 1. If beta=1 the temperature is constant and Simulated Annealing is not applied\ 
fric is a parameter of the Langevin Dynamics. Default fric=100. 
If fric=0 the algorithm is equivalent to the Velocity-Verlet algorithm

### Comparision with Adam optimizer 
The comparision was done on CIFAR10 with ResNet, using  
[this](https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py) keras code

![Training results](https://github.com/borbysh/SALDOptimizer/blob/master/Figure_1.png)



Kirkpatrick, Scott, C. Daniel Gelatt, and Mario P. Vecchi. "Optimization by simulated annealing." Science 220.4598 (1983): 671-680.

Vanden-Eijnden, Eric, and Giovanni Ciccotti. "Second-order integrators for Langevin equations with holonomic constraints." Chemical physics letters 429.1-3 (2006): 310-316.
