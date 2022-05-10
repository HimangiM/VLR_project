# Improving Contrastive Learning by increasing positive samples

Name: Aditya Ghuge, Ayush Pandey, Himangi Mittal

Group: 13

## Convert image to embeddings

To ensure that our network learns representations that are more invariant to view, deformation, and illumination changes than BYOL we try to get include a positive image that belongs to the same category in our training.  To do this we first try to learn a latent space that is semantically rich and continuous. We hypothesize that images belonging to the same category lie closer to each other in this latent space and images belonging to different categories lie further apart. We try 3 different learning paradigms to learn this latent space:


**1. Auto Decoder:** Our work in training the Auto Decoder module is inspired by the DeepSDF paper. In training an Auto Decoder we randomly initialize latent vectors for all the images in the training data and jointly then optimize a decoder network and the latent vectors to reconstruct the original images. We assume a prior multi-variate Gaussian distribution over the latent codes with 0 mean and Identity as spherical covariance which translates to an additional L2 regularization on the latent codes apart from the mean squared error loss used for reconstruction. The final optimized latent vectors corresponding to the images are used as their representation.

To train the Auto Decoder run:

```python
python train_decoder.py
```
**2. Auto Encoder:** An Auto Encoder is a bottleneck architecture that learns to compress the data. It comprises an encoder that encodes an image to a latent vector and a decoder that reconstructs the original image using this latent vector. Reconstruction loss between the original image and the generated image is used to train the network. To represent the encoding of the image we use the output of the encoder network.

To train the Auto Encoder run:

```python
python train_encoder.py --mode ae
```

**3. Variational Auto Encoder:**  Variational Auto Encoder is similar to Auto Encoder except that it learns to encode an image to distribution rather than a latent vector. Instead of compressing the image into one vector, the encoder outputs two latent vectors i.e a vector of means and a vector of standard deviation ( we model it as a vector of log variances for numerical stability ). These vectors can be used to define a Gaussian distribution from which the final latent vector which is used by the decoder is sampled.  We minimize the KL divergence between the generated distribution and target distribution of mean 0 and standard deviation 1 along with the reconstruction loss to train the network. To represent the encodings for an image we use the latent vectors that represent the mean vector. 

To train Variational Auto Encoder run:

```python
python train_encoder.py --mode vae
```
## Nearest neighbor

In the previous step, we learned a latent space to represent an image. As a result of running the scripts, there would be a pickle file generated that stores a latent vector corresponding to each image in the dataset. We then compute the nearest neighbor for each image in the dataset by computing its distance from all the other images in the dataset. We use Euclidean or Cosine distance to compute the nearest neighbors.

To compute the nearest neighbors:

```python
python nearest_neighbor.py input_pickle_file out_file_name mode
```

Where the mode can be euclidean or cosine.

## Training BYOL


We have used Bring Your Own Latent (BYOL) as a self-supervised learning framework to implement our idea. We have investigated 2 approaches to augment the traditional positive images in the framework. 

**1. First approach:** We randomly sample a positive pair image from the traditional image augmentation image or the nearest neighbor pair with some probability. We hypothesize that by doing so we force our model to learn only from the semantically relevant part of the images and get better results.

To train BYOL with a probabilistic sampling run (we sample with a probability of 0.5):

```python
python train_byol_probablistic.py
```

**2. Second approach:** We run both traditional augmented image and nearest neighbor image through a momentum encoder model to get a latent vector from both images. We then resultant target vector using a weighted average of the two vectors and use that vector for training BYOL. 

To train BYOL with a weighted average of the two vectors (default weight for the vector corresponding to the augmented image is 0.5):

```python
python byol_pytorch_weighted.py  --pos_weight 0.5
```

Additionally, we run a baseline BYOL network to compare the efficacy of our approaches on the downstream task of image classification.

To train baseline BYOL:

```python
python train_byol.py 
```
