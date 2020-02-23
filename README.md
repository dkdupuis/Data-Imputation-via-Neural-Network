# Generative Autoencoder for Data Imputation

## Methodology

### Setup

* Begin with dataset that includes missing elements
* Fill NAs with random values

### Neural Network Architecture & Training

The architecture is the same as autoencoders. Traditionally autoencoders just attempt to learn a lower dimensional feature space of the data. While this methodology also generates a lower dimensional space for the data, the purpose is to learn about the variance in the input space in order predict missing values on the basis of known values.

## Todo

* Refactor loss function to only evaluate elements that weren't originally NA
* After each training loop, update NA elements with the values predicted by the autoencoder
* Clean up code
