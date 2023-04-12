# Apple-Disease-Detection-Using-CNN

This project aims to develop a machine learning model that can accurately detect and classify diseases in apple trees using Convolutional Neural Networks (CNNs). The model is trained on a dataset of images of healthy and diseased apple leaves, and is able to classify the images into one of several categories of diseases, including apple scab, blotch apple and rotten apple.

# Introduction
Apple trees are susceptible to various diseases that can significantly reduce crop yields and quality. Early detection and accurate diagnosis of these diseases are crucial for effective disease management and prevention. In recent years, machine learning techniques, particularly Convolutional Neural Networks (CNNs), have shown promising results in image classification tasks, including plant disease detection. In this project, we aim to develop a CNN model that can accurately detect and classify diseases in apple trees using images of apple leaves.

# Dataset
The dataset used for training and testing the model is the taken from a randomized dataset consisting of variooous images over the internet , which contains 512 RGB images of healthy and diseased apple leaves, captured under different lighting conditions and angles. The dataset includes four classes of diseases: apple scab, blotch apple , rotten apple , and healthy apples. The images are labeled according to their disease category, with approximately equal numbers of images in each class.


![rotten-apple-isolated-on-black-260nw-1936327780](https://user-images.githubusercontent.com/99320225/231492290-888c2bd5-c6d1-46b8-b179-4bd27e61a599.jpg)
Rotten Apple

![DSC_3770-1defcwh](https://user-images.githubusercontent.com/99320225/231491114-1b6b2864-3f76-4255-ba25-f74793f261be.jpg) Scab Appple


![AnyConv com__images (4)](https://user-images.githubusercontent.com/99320225/231491861-f5677af9-81be-4802-8b03-3ec0d8050f04.jpg)
Normal Apple

# Methodology
We used the Keras library in Python to build and train the CNN model. The model architecture consists of several convolutional layers followed by max pooling layers and dropout layers to prevent overfitting. The output from the convolutional layers is flattened and fed into a fully connected layer, which outputs the probability distribution over the four disease classes. We used the categorical cross-entropy loss function and the Adam optimizer to train the model for 30 epochs.

We split the dataset into training (60%), validation (40%), and test sets. We augmented the training data by randomly rotating, shifting, and zooming the images to increase the diversity of the training set and reduce overfitting.

# Results

The trained model achieved an accuracy of 91.5% on the training set and 67% on validation and test sets. The confusion matrix shows that the model was able to correctly classify the majority of the images, with the highest accuracy for healthy apples and the lowest accuracy for blotch and rotten apples. We also visualized the activations of the convolutional layers to gain insights into the features learned by the model.


