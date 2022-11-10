# EE604A Assignment 3

> **Gurbaaz Singh Nandra (190349)**

## sw

**IP Software**: **MIScnn**,  *a framework for 2D/3D Medical Image Segmentation*

**Github URL**: https://github.com/frankkramer-lab/MIScnn 

**Paper**: https://doi.org/10.1186/s12880-020-00543-7

```latex
Article{miscnn21,
  title={MIScnn: a framework for medical image segmentation with convolutional neural networks and deep learning},
  author={Dominik Müller and Frank Kramer},
  year={2021},
  journal={BMC Medical Imaging},
  volume={21},
  url={https://doi.org/10.1186/s12880-020-00543-7},
  doi={10.1186/s12880-020-00543-7},
  eprint={1910.09308},
  archivePrefix={arXiv},
  primaryClass={eess.IV}
}
```

The open-source Python library **MIScnn** is an intuitive API allowing fast setup of medical image segmentation pipelines with state-of-the-art convolutional neural network and deep learning models in just a few lines of code. Medical image segmentation is highly useful for segmentation of CT scans pertaining to deadly diseases, resulting in a better diagnosis and analysis. e.g. Kidney Tumor Segmentation challenge 2019 (KITS19) involved computing a semantic segmentation of arterial phase abdominal CT scans from 300 kidney cancer patients. 

![image-20221110035207162](/home/gurbaaz/.config/Typora/typora-user-images/image-20221110035207162.png)

**Image segmentation** is a method in which a digital image is broken down into various subgroups called image segments, which helps in reducing the complexity of the image to make further processing or analysis of the image simpler. 

MIScnn undergoes this task in a pipelined fashion as follows:-

- Data I/O
- Data Augmentation (optional)
- Preprocessor
- Neural Network Model

![MIScnn workflow](https://github.com/frankkramer-lab/MIScnn/raw/master/docs/MIScnn.pipeline.png)

Once the preprocessing steps involving intensity normalisation, clipping etc. are completed, a one hot encoding method is performed by creating a single binary variable for each segmentation class. For the segmentation prediction, an already fitted CNN model is used. The model predicts for every pixel a sigmoid value for each class. The sigmoid value represents a probability estimation of this pixel for the associated label. Consequently, the argmax of the one hot encoded class are identifed for multi-class segmentation problems and then converted back to a single result variable containing the class with the highest sigmoid value.

![image-20221110042404208](/home/gurbaaz/.config/Typora/typora-user-images/image-20221110042404208.png)

![](/home/gurbaaz/Pictures/Screenshots/Screenshot from 2022-11-10 03-51-36.png)

## q4q

### q4q-analytical

Bob, your friend, is an avid wildlife photographer, and captures images of endangered animals like Tiger for posters on awareness on Save Wildlife. Bob always takes his shot in CMYK domain, whereas the printing service requires an HVS-domain image to print posters. He is usually able to accomplish this task using his favorite tool:- Photoshop. But one day, due to some bug in photoshop across all systems, the CMYK to HVS conversion tool was not working. Bob was utterly disappointed, but then remembered you have recently completed a course in Image Processing, and is hence seeking your help. ***Don't let Bob down!*** 

As a test for you, Bob provides you a random pixel from the image, with **CMYK** values as **(0, 44, 85, 16)** so that you don't waste his ink with some faulty calculation. Calculate the value of pixel in HVS space.

![image-20221110055358557](/home/gurbaaz/.config/Typora/typora-user-images/image-20221110055358557.png)

Answer:- **HSV = (29, 85, 84)**

The easiest way to go from CMYK to HSV space is via RGB domain. Use the following formulae to get RGB value of the pixel,
$$
K = 1 - max(R, G, B) \\
C = \frac{1-r-K}{1-K} \\
M = \frac{1-g-K}{1-K} \\
Y = \frac{1-b-K}{1-K}
$$
The RGB comes out to be **(214, 120, 32)**. Now finally, we can move to HSV space with the following set of rules,
$$
I = \frac{r+g+b}{3} \\
S = 1 - \frac{3}{r+g+b}min(r, g, b) \\
H = \theta \text{ if } B \leq G \text{ else } 360-\theta \text{ , where}\\
cos(\theta) =  \frac{1/2((r-g) + (r-b))}{{{((r-g)}^2 + (r-g)(g-b))}^{1/2}}
$$


### q4q-mcq

1. Which of the following image's fourier transform resembles the given fourier transform plot?

   <img src="/home/gurbaaz/sem7/ee604/EE604A-Assignments/6.png" style="zoom: 33%;" />

(a) <img src="/home/gurbaaz/sem7/ee604/EE604A-Assignments/download.jpeg" style="zoom:90%;" />

(b) <img src="/home/gurbaaz/sem7/ee604/EE604A-Assignments/download (1).jpeg" style="zoom:67%;" />

**(c)** <img src="/home/gurbaaz/sem7/ee604/EE604A-Assignments/51jRoO1CRXS.jpg" style="zoom:20%;" />

(d) <img src="/home/gurbaaz/sem7/ee604/EE604A-Assignments/51RnQNx3+bL.jpg" style="zoom:33%;" />

Answer:- **(c)**. Count the number of lines at different angles. As there are 5 evident lines in the fourier transform at different angles, there must 5 edges in different directions in the image. Pyramix in option (c) most closely resemles our required constraint.

2. Consider the given unsupervised clustering plots in figure below and mark the correct option(s)
   ![image-20221110052120714](/home/gurbaaz/.config/Typora/typora-user-images/image-20221110052120714.png)

   ​          Figure 1                                          Figure 2

(a) Figure 1  is a result of Spectral Clustering based on kNN graph while Figure 2 is a result of k-means clustering

**(b)** Figure 1  is a result of k-means clustering while Figure 2 is a result of Spectral Clustering based on kNN graph

(c) k-means perform better (as shown in Figure 2) as the algorithm is guaranteed to converge in a finite number of iterations

**(d)** k-means perform worse (as shown in Figure 1) due to the algorithm's assumption of uniform distribution of data and circular clusters 

Answer:- (a, c) The reasoning is well-covered in the options itself. 𝑘-means assume similar size spherical clusters and uniform istribution of data. Most often graph-based clustering algorithms utilize internode distance and aim to construct the nearest neighbor (NN) graph.  They assume no underlying structure of the dataset.
