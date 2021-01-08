## Computer Vision

Week 1: Working with images & CNN Building Blocks

Overview Working with images: This week we start dealing with unstructured data, specifically images. We start with an intro to what an image means to a machine, various formats of an image, ways of displaying an image in python and resolution of an image. We also learn how to read a histogram of an image, understand the process of correlation and convolution of images and finally the concept of filtering.

Overview CNN Building Blocks: This week we start with a neural network architecture which works best for images. We address all the areas a CNN is better than a plain neural network. We dive deep into the forward prop and backward prop for a CNN later we see how it compares to a fully connected neural network and finally, we discuss the need for pooling and ways of doing pooling in CNN.

Working with Images_Introduction 
 Looking at an image as a bunch of pixel values 

 Types of images – Greyscale, Binary, colour 
 Image formats – RAW, JPEG, PNG etc. 
 Displaying images in python environment using PIL and Matplotlib 

Working with Images - Digitization, Sampling, and Quantization (13 min)
 Sampling and quantization of 1D and 2D continuous signals ~5.5min
 Approximation of an arc in the digital space 
 Definition of a Picture 
 Resolution of a picture 

Working with images - Filtering 
 Recap on what an image is 
 Domain and Range in the context of an image 
 Understanding a Histogram 
 Correlation and convolution of a dummy image grid with a kernel ~8min
 Concept of Filtering with examples ~8min

Hands-on Python Demo: Working with images 
 Getting familiar with using an image in python 
 Building a simple Averaging Filter from scratch 

Introduction to Convolutions (20 min)
 Convolutional Neural Network vs plain Neural Network 
 Need for Convolutional as an alternative to a plain neural network ~8min
 Translational invariance 
 A simple demonstration of convolution using a 1D array 

2D convolutions for Images (19 min)
 Demonstration of convolution on a simple 2D image, using manual calculations Convolution - Forward (6 min)
 Demonstration of convolution using a dummy array, in tensorflow 
 Demonstration of convolution using an image in tensorflow 

Convolution - Backward (15 min)
 Recap of convolution during forwarding path 
 Demo of convolution during backward prop using a simple example 

Transposed Convolution and Fully Connected Layer as a Convolution 
 Dilated convolution ~1min
 Deconvolution/Transposed convolution 
 Implementation of convolution to mimic a fully connected layer ~4min
 Key points to remember when converting from linear to convolution 

Pooling: Max Pooling and Other pooling options (11 min)
 Explanation of pooling and specifically max pooling and average pooling? ~4min
 Demo of max pooling with tensorflow ~1min
 Max and average pooling during back prop 
 Other pooling options
Hands-on Keras Demo: MNIST CNN Building Blocks code walk-through (13 min)
 Implementation of CNN in Keras framework, using MNIST dataset

 

Week - 2: CNN Architectures and Transfer Learning (~2.5 hours)

Overview: This week we go through all the popular neural nets, which were the state of the art in their time, and compare their performances with each other. We get to see how GPU is a much better alternative over CPU to train neural networks. And finally, we address the importance of transfer learning and how to achieve transfer learning and finally ways of visualizing a neural network for the purpose of debugging or fine-tuning.

CNN Architectures and LeNet Case Study 
 Two major challenge-datasets – MNIST and IMAGENET 
 Discussion on LeNet for MNIST 
 Discussion on various CNNs built for IMAGENET challenge ~6min
 The network architecture of LeNet-5 ~6min

Case Study: AlexNet (16 min)
 Architecture o AlexNet in detail ~16min

Case Study: ZFNet and VGGNet (15 min)
 Performance and architecture of ZFNet 
 Performance and architecture of VGGNet in detail ~12.5m

Case Study: GoogleNet (13 min)
 Architecture and performance of GoogleNet in detail 

Case Study: ResNet (29 min)
 Performance comparison of all the architectures until now ~9min
 Architecture comparison of VGGNet and ResNet 
 ResNet vs Plain NN ~6min
 Key points about ResNet 
 The idea behind Squeeze and Excitation Network 
 Why use ConvNets? And a brief summary of the evolution of CNNs 

GPU vs CPU 
 How GPU is different from CPU? 
 Comparison of the speed of execution of CPU and GPU ~6min

Transfer Learning Principles and Practice 
 Transfer learning as a smart workaround to enable the use of CNNs on

smaller datasets 
 Some standard practices when implementing transfer learning ~9min
 A brief discussion on transfer learning for image captioning 

Hands-on Keras Demo: SVHN Transfer learning from MNIST dataset 
 Using a network trained on MNIST to classify SVHN digits to demonstrate

Transfer learning Visualization (run package, occlusion experiment)

 The idea behind visualizing CNNs ~1min
 Visualize patches that maximally activate neurons ~4min
 Visualize the filters/kernels (weights) ~1min
 Visualize the representation space ~1min
 Occlusion space 
 Addressing the Universal approximation theorem and the need for deep neural networks 

Hands-on demo -T-SNE (7min)
 Creating T-SNE embedding of each MNIST image to visualize in 2D 

 

Week-3: CNN's at Work - Semantic Segmentation(20 min)

Overview: Going beyond applying CNNs for regular classification problems, we discuss how to
achieve semantic segmentation using CNNs and specifically, we look at the U-Net architecture. 

CNNs at Work - Semantic Segmentation (20 min)
 Meaning of Semantic segmentation ~4min
 Recap of CNN architecture 
 What is padding and why do we need it? ~4min
 At a high level, how the architecture uses surrounding pixels to get the spatial context of a certain pixel, to classify/label that pixel ~6min
 Addressing the issue of Double-convolution or redundant convolutions ~1min
 Clarification on what is the input and what we expect as an output 

Semantic Segmentation process (20 min)
 Pooling, as a way to decrease the size of individual feature map as well as increase the receptive field of each pixel of each feature map ~4.5min
 Need for and ways to up-sample/un-pool ~8min
 Hour-glass structure of the CNN used for semantic segmentation 
 Challenge with the hour-glass structure and a workaround for that 

U-Net Architecture for Semantic Segmentation 
 The workflow of U-Net architecture ~8min
 Sample output for nucleus segmentation in pathology 

Hands-on demo - Semantic Segmentation using U-Net 
 Demonstration of Semantic Segmentation using U-Net architecture 

Other variants of Convolutions 
 1x1 convolution – concept, advantage and applications 
 Atrous (dilated) convolution – concept, advantage and applications ~4min

Inception and MobileNet models (15 min)
 Concept of Inception module and Separable convolutions used in
Inception style convolution ~8min
 Concept of layer-wise convolution implemented in MobileNet 

 

Week-4: Object Detection

Overview: Going even beyond semantic segmentation, we learn how to define boundary boxes after object detection. We go through architectures like R-CNN, YOLO and SSD for the same purpose.

CNN's at Work - Object Detection with region proposals 
 What is localization? 
 How localization is achieved ~9min
 Faster R-CNN - architecture and workflow ~9min

CNN's at Work - Object Detection with Yolo and SSD 
 YOLO approach to detect multiple objects 

 SSD framework – concept and code ~6min

Hands-on demo- Bounding box regressor (28 min)
 Extensive hands-on, on detecting bounding box

 

Week-5: Project Semantic Segmentation 

 

Week-6: CNN's at work: Siamese Network for Metric Learning

Overview: This final week will be all about defining metric learning, appreciating the need for metric learning and specifically we discuss the Siamese Network architecture to implement metric learning. And finally a brief discussion on Joint layers

Metric Learning 
 CNN recap 
 The usefulness of features obtained in the 2nd last layer of a trained network ~6.5min
 Properties of RBF kernel 
 Metric learning and properties of a metric 
 Need for metric learning 

Siamese Network as metric learning 
 Siamese network architecture 
 Computation of final metric value when inputs are given 
 Two ways of obtaining the metric ~4min
 Examples of distance and similarity measures ~6min

How to train a Neural Network in Siamese way 
 Forming triplets to compute the loss 
 The loss function for a Siamese network 
 Advantage of using a pre-trained neural network to build a Siamese-style network 
 Discussion on joint layers ~6min

Hands-on demo - Siamese Network 
 Demo on building and optimizing a Siamese network on Arcadian language dataset

 

 Week-7: Project Face Recognition
