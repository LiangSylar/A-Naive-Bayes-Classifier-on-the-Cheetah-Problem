# A-Naive-Bayes-Classifier-on-the-Cheetah-Problem
This is a mini-classification task on studying the Maximum a Posteriori decision rule.

## Motivation
Given the cheetagh image, we write a program that can segment if into object(foreground) and backgorund. In this repository, this goal is achived by constructing a simple Bayes Classifier.
Specifically, due to our choice of loss function is 0-1 loss, we are equivalently applying the Maximum a-posteriori decision rule. We will learn the posterior and prior distribution from the train image, and then apply the MAP rule on an test image to get the classification result. This repository includes two ways of constructing the classifier, mainly dependong on the different ways of estimating the class-conditional distributions on input data. The first method extracts only oe feature from the training data, and uses a non-parametric method to estimate conditioanl distributions. The second method fits a 64-dim Multivariate Gaussian distribution for the 64-dim training data, and selects a group of 8 features to finish the final computation of posterior probabilities. 

## Image Representation
The image is preprocessed using discrete cosine transform (DCT) which takes in an matrix block and then outputs extracted infomation in matrix form. A 8*8 sliding window is used. We first break the training images into 8*8 blocks. For each block, we compute its DCT, and then flatten the matrix with order according to the zig-zag pattern (go to zig-zag.txt). Next we perform feature extraction by picking the position(i.e., index) of second largest magnitude as the feature value. The collection of all such positions will serve as the training set. The DCT results for the train image has been provided in TrainingSamplesDCT_8.mat. 

## Estimation of Prior Probabilities
The prior probabilities are estimated from the training image. For the training image, a corresponding classification mask is given. We take the fraction of cheetah on the whole image as the probability of foreground, and take the rest fraction (i.e. the fraction taken by the grass and so on) as the probability of background. We have only one train image, which is definitely not an accurate estimate of the prior probabilities. 

## Non-parametric estimation of Conditional Distribution
The core step in MAP decision rule is to estimate the conditional distirbution of X given its class Y. Here, we adopt the simple non-parametric method, which is to use index histogram. The number of bins used by the histogram serves as a hyper-parameter that might further improve the classification result. 

### Result Preview (Non-parametric method)
The following is the result obtained by the code. This simple bayes classifier is obviously a poor one as 1) we used only 1 extracted feature out of the 64D extracted vector, and that 2) we estimated the conditional distributions by indexed-histograms. Comparing with the ground true mask, this classifier fails to capture details such as the head, the tail and the legs 
very clearly. Also, this classifier is still very noisy, as there are a lot of points in the background classified as part of the cheetah. Overall, this simple and raw classifier still 
managed to capture the main body of cheetah, which is very surprising upon the fact this is indeed a simple classifier. The estimated error of probability is 17.53%.

![image](https://user-images.githubusercontent.com/64362092/196350720-1a8c3098-2d62-458e-8281-e6946847f00f.png)

#### Figure 1. True mask of the test image.

![image](https://user-images.githubusercontent.com/64362092/196350857-67838c26-999d-432d-82a6-109df9a439d1.png)

#### Figure 2. Estimated result for the test image.

## Parametric estimation of Conditioanl Distribution: Gaussian Classifier 
The mean and covariance matrix are estimated by MLE. A hint here is to the log posterior formula rather computing posterior by likelihood*prior directly. Otherwise the probability is too small and the classification fails. For the detailed formula, go to http://www.svcl.ucsd.edu/courses/ece271A/handouts/GC.pdf. 


