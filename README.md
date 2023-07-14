# Seeing Beyond the Frames!
It's a fascinating idea that we can reconstruct an entire scene given a set of views, and in this project we explore two such ways: one is the classical Structure from Motion (SfM) and the other is the learning-based Neural Radiance Fields (NeRF).

## Phase I: Structure from Motion (SfM)
Structure from motion helps reconstruct a scene from two or more different views using epipolar geometry. A few years ago, Agarwal et. al published Building Rome in a Day in which they reconstructed the entire city just by using a large collection of photos from the internet. In this phase, we take a detailed step-by-step approach from feature matching to obtaining camera locations to reconstructing the scene.

## Phase II: Neural Radiance Fields (NeRF)
Neural Radiance Fields (NeRF) is an innovative approach to view synthesis that pushed the boundaries of computer vision and graphics. NeRF's underlying neural network architecture models the radiance and geometry of a scene taking sparse image set and camera poses as inputs, enabling it to generate novel views. In this phase, we build a vanilla NeRF model from the original paper by Mildenhall et al.


### References:
1. https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-generating-camera-rays/generating-camera-rays.html