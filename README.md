# Physics-infromed transformers for air pollution assessmnet

## Abstract
Physics-Informed neural networks (PINNs) are widely used for numerical solving of partial differential equation in huge variety of physics spheres
as these approaches are less expensive in terms of computational costs. With the rapid development of artificial intelligence in recent years new advanced techniques are also popular in area of proxy models of physics domains and partial differential equation (PDE) solvers. There are a number of algorithms such as Fourier neural operator (FNO), CoDa-NO (GNO), DeepONet, PINNs with dynamic weights strategy and Transolver. This thesis aimed to experimentally demonstrate how to solve Navier- Stokes equations applied for fluid dynamics via Physic-Informed Transformers like Transolver. This study endeavors to show how Transolver can handle specific examples of Navier-Stokes equations on real-datasets. The result
of the work is an application of an existing deep learning model to the new dataset, which was not tested on this example of Navier-Stokes equation before.

## Results
In this work, we proposed solutions to the problem of determining the coordinates of air pollution sources using cutting-edge approaches for approximating and predicting the behavior of physical systems. The main goal was to demonstrate that Transolver can accurately determine the state of a physical system at the initial moment in time, thereby enabling successful resolution of the inverse problem of locating pollution sources. Based on this model, both classifiers and regressors were developed that successfully identify the coordinates of the source using synthetic data on air pollution in the city of Novosibirsk.

The solutions examined in this study show how useful this transformer-based model can be not only for approximating a system at a specific moment in time, but also for identifying other critical characteristics directly related to various physical systems — for example, the detection of air pollution sources.

Future work may focus on adapting Transolver to handle more complex data, where external influences affect the spread of particulate matter in the air — for instance, wind. Another planned step is the transition from a two-dimensional to a three-dimensional problem. This will require more complex models; however, Transolver is capable of approximating physical systems in three-dimensional space as well, making it a viable core method for solving this task.

The current and future results of this research will be integrated into the operations of the CityAir company, which conducts real-time environmental monitoring to detect sources of pollution.

