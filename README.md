# Hands-On Coding SEMLA: Sunday 30/10/2022
## Getting started
### 0. Set up Python

you need to have/install python 3. 

### 1. Install requirements

Run the following:

```
pip install -r requirements.txt
```

### 2. Run the tests

Run the following:

```
python tests/random_testing.py
python tests/search_based_testing.py
```
### 3. Practice the test development

The objective is to implement a metamorphic unit testing for Deep Neural Networks (DNNs) applied in computer vision. The task is to add unit tests to validate the robustness of the EffNet model against image transformations

The module `helpers/transformations` contains different mutations from two categories of image-based transformations:

1. Pixel value transformations: change image contrast, image brightness, image blur, and image sharpness.
2. Affine transformations: image translation, image scaling, image shearing and image rotation.

You can pick any transformation and code its corresponding unit tests in both `RandomTest` and `SearchBasedTest` classes by following the example of robustness against white noise.
