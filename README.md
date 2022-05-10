# Improving Contrastive Learning by increasing positive samples

Name: Aditya Ghuge, Ayush Pandey, Himangi Mittal

Group: 13

## Convert image to embeddings

To ensure that our network learns representations that are more invariant to view, deformation, and illumination changes than BYOL we try to get include a positive image that belongs to the same category in our training.  To do this we first try to learn a latent space that is semantically rich and continuous. We hypothesize that images belonging to the same category lie closer to each other in this latent space and images belonging to different categories lie further apart. We try 3 different learning paradigms to learn this latent space:


*1. Auto Decoder:* Our work in training the Auto Decoder module is inspired by the DeepSDF paper. In training an Auto Decoder we randomly initialize latent vectors for all the images in the training data and jointly then optimize a decoder network and the latent vectors to reconstruct the original images. We assume a prior multi-variate Gaussian distribution over the latent codes with 0 mean and Identity as spherical covariance which translates to an additional L2 regularization on the latent codes apart from the mean squared error loss used for reconstruction. The final optimized latent vectors corresponding to the images are used as their representation.

```python
To train the Auto Decoder run:python train_decoder.py
```
