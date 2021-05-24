**<font color='red'> Warning:</font> Make sure this file is named ClassifyingImages.ipynb on Coursera or the submit button will not work.**

*If you plan to run the assignment locally:*
You can download the assignments and run them locally, but please be aware that as much as we would like our code to be universal, computer platform differences may lead to incorrectly reported errors even on correct solutions. Therefore, we encourage you to validate your solution in Coursera whenever this may be happening. If you decide to run the assignment locally, please: 
   1. Try to download the necessary data files from your home directory one at a time,
   2. Don't update anything other than this Jupyter notebook back to Coursera's servers, and 
   3. Make sure this notebook maintains its original name after you upload it back to Coursera.
   
Note: You need to submit the assignment to be graded, and passing the validation button's test does not grade the assignment. The validation button's functionality is exactly the same as running all cells.


```python

```


```python
%matplotlib inline
%load_ext autoreload
%autoreload 2

import matplotlib.pyplot as plt

import numpy as np
from skimage.transform import resize
from sklearn.naive_bayes import GaussianNB, BernoulliNB
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os

from aml_utils import show_test_cases, test_case_checker, perform_computation
```

Libraries such as `math` are neither as accurate nor as efficient as `numpy`.

**Note**: Do not import or use any other libraries other than what is already imported above. 

# *Assignment Summary

You may find it useful to read Chapter 1 and 2 of the textbook about classification.

The MNIST dataset is a dataset of 60,000 training and 10,000 test examples of handwritten digits, originally constructed by Yann Lecun, Corinna Cortes, and Christopher J.C. Burges. It is very widely used to check simple methods. There are 10 classes in total ("0" to "9"). This dataset has been extensively studied, and there is a history of methods and feature constructions at https://en.wikipedia.org/wiki/MNIST_database and at the original site, http://yann.lecun.com/exdb/mnist/ . You should notice that the best methods perform extremely well.

There is also a version of the data that was used for a Kaggle competition. I used it for convenience so I wouldn't have to decompress Lecun's original format. I found it at http://www.kaggle.com/c/digit-recognizer .

Regardless of which format you find the dataset stored in, the dataset consists of 28 x 28 images. These were originally binary images, but appear to be grey level images as a result of some anti-aliasing. We will ignore mid grey pixels (there aren't many of them) and call dark pixels "ink pixels", and light pixels "paper pixels"; you can modify the data values with a threshold to specify the distinction, as described here https://en.wikipedia.org/wiki/Thresholding_(image_processing) . The digit has been centered in the image by centering the center of gravity of the image pixels, but as mentioned on the original site, this is probably not ideal. Here are some options for re-centering the digits that I will refer to in the exercises.

  * **Untouched**: Do not re-center the digits, but use the images as is.
  
  * **Bounding box**: Construct a 20 x 20 bounding box so that the horizontal (resp. vertical) range of ink pixels is centered in the box.

  * **Stretched bounding box**: Construct a 20 x 20 bounding box so that the horizontal (resp. vertical) range of ink pixels runs the full horizontal (resp. vertical) range of the box. Obtaining this representation will involve rescaling image pixels: you find the horizontal and vertical ink range, cut that out of the original image, then resize the result to 20 x 20. Once the image has been re-centered, you can compute features.
  
Here are some pictures, which may help.

![alt text](../ClassifyingImages-lib/bounding_v2.png "Illustrations of the bounding box options described in text")

**Part 2A** Investigate classifying MNIST using naive Bayes. Fill in the accuracy values for the four combinations of Gaussian v. Bernoulli distributions and untouched images v. stretched bounding boxes in a table like this. Please use 20 x 20 for your bounding box dimensions.

<table width="400" border="1">
  <tbody>
    <tr>
      <th scope="col"> Accuracy</th>
      <th scope="col">Gaussian</th>
      <th scope="col">Bernoulli</th>
    </tr>
    <tr>
      <th scope="row">Untouched images</th>
      <td>&nbsp;</td>
      <td>&nbsp;</td>
    </tr>
    <tr>
      <th scope="row">Stretched bounding box</th>
      <td>&nbsp;</td>
      <td>&nbsp;</td>
    </tr>
  </tbody>
</table>
Which distribution (Gaussian or Bernoulli) is better for untouched pixels? Which is better for stretched bounding box images?

**Part 2B** Investigate classifying MNIST using a decision forest. For this you should use a library. For your forest construction, try out and compare the combinations of parameters shown in the table (i.e. depth of tree, number of trees, etc.) by listing the accuracy for each of the following cases: untouched raw pixels; stretched bounding box. Please use 20 x 20 for your bounding box dimensions. In each case, fill in a table like those shown below.

<table width="400" border="1">
  <tbody>
    <tr>
      <th scope="col">Accuracy</th>
      <th scope="col">depth = 4</th>
      <th scope="col">depth = 8</th>
      <th scope="col">depth = 16</th>
    </tr>
    <tr>
      <th scope="row">#trees = 10</th>
      <td>&nbsp;</td>
      <td>&nbsp;</td>
      <td>&nbsp;</td>
    </tr>
    <tr>
      <th scope="row">#trees = 20</th>
      <td>&nbsp;</td>
      <td>&nbsp;</td>
      <td>&nbsp;</td>
    </tr>
    <tr>
      <th scope="row">#trees = 30</th>
      <td>&nbsp;</td>
      <td>&nbsp;</td>
      <td>&nbsp;</td>
    </tr>
  </tbody>
</table> 	 	 

# 0. Data

Since the MNIST data (http://yann.lecun.com/exdb/mnist/) is stored in a binary format, we would rather have an API handle the loading for us. 

Pytorch (https://pytorch.org/) is an Automatic Differentiation library that we may see and use later in the course. 

Torchvision (https://pytorch.org/docs/stable/torchvision/index.html?highlight=torchvision#module-torchvision) is an extension library for pytorch that can load many of the famous data sets painlessly. 

We already used Torchvision for downloading the MNIST data. It is stored in a numpy array file that we will load easily.


```python
if os.path.exists('../ClassifyingImages-lib/mnist.npz'):
    npzfile = np.load('../ClassifyingImages-lib/mnist.npz')
    train_images_raw = npzfile['train_images_raw']
    train_labels = npzfile['train_labels']
    eval_images_raw = npzfile['eval_images_raw']
    eval_labels = npzfile['eval_labels']
else:
    import torchvision
    download_ = not os.path.exists('../ClassifyingImages-lib/mnist.npz')
    data_train = torchvision.datasets.MNIST('mnist', train=True, transform=None, target_transform=None, download=download_)
    data_eval = torchvision.datasets.MNIST('mnist', train=False, transform=None, target_transform=None, download=download_)

    train_images_raw = data_train.data.numpy()
    train_labels = data_train.targets.numpy()
    eval_images_raw = data_eval.data.numpy()
    eval_labels = data_eval.targets.numpy()

    np.savez('../ClassifyingImages-lib/mnist.npz', train_images_raw=train_images_raw, train_labels=train_labels, 
             eval_images_raw=eval_images_raw, eval_labels=eval_labels) 
```

## 0.1 Getting to Know The Imported Data

Let's visibly check and see what we have imported.
1. What is the order of dimensions? (We need to check whether the number of samples is the first dimension or the last dimension)
2. What is the training data type?
3. What is the labels shape and data type? Is it one-hot encoded?
4. What are the pixel value ranges? Is it between 0 and 1? Or is it between 0 and 255? or something else?
5. How is an ink (resp. background) pixel represented? Is it represented with 0? Is it represented with 1? Is it represented with 255? or something else? 

The following cells should help you answer these questions in order.


```python
train_images_raw.shape
```




    (60000, 28, 28)




```python
train_images_raw.dtype
```




    dtype('uint8')




```python
train_labels.shape, train_labels.dtype
```




    ((60000,), dtype('int64'))




```python
train_labels[:10]
```




    array([5, 0, 4, 1, 9, 2, 1, 3, 1, 4])




```python
train_images_raw[0].min(), train_images_raw[0].max()
```




    (0, 255)




```python
for row_im in train_images_raw[0]:
    print(row_im.tolist())
plt.imshow(train_images_raw[0], cmap='Greys')
```

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 18, 18, 18, 126, 136, 175, 26, 166, 255, 247, 127, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 30, 36, 94, 154, 170, 253, 253, 253, 253, 253, 225, 172, 253, 242, 195, 64, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 49, 238, 253, 253, 253, 253, 253, 253, 253, 253, 251, 93, 82, 82, 56, 39, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 18, 219, 253, 253, 253, 253, 253, 198, 182, 247, 241, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 80, 156, 107, 253, 253, 205, 11, 0, 43, 154, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 1, 154, 253, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 139, 253, 190, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 190, 253, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 241, 225, 160, 108, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 81, 240, 253, 253, 119, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 186, 253, 253, 150, 27, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 93, 252, 253, 187, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 249, 253, 249, 64, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 130, 183, 253, 253, 207, 2, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 39, 148, 229, 253, 253, 253, 250, 182, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 114, 221, 253, 253, 253, 253, 201, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 23, 66, 213, 253, 253, 253, 253, 198, 81, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 18, 171, 219, 253, 253, 253, 253, 195, 80, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 55, 172, 226, 253, 253, 253, 253, 244, 133, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 136, 253, 253, 253, 212, 135, 132, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]





    <matplotlib.image.AxesImage at 0x7f50a7a55ac0>




![png](output_16_2.png)


## 0.2 Thresholding

# <span style="color:blue">Task 1</span>

Write the function `get_thresholded` that does image thresholding and takes following the arguments:

1. `images_raw`: A numpy array. Do not assume anything about its shape, dtype or range of values. Your function should be careless about these attributes.
2. `threshold`: A scalar value.

and returns the following:

* `threshed_image`: A numpy array with the same shape as `images_raw`, and the `bool` dtype. This array should indicate whether each elemelent of `images_raw` is **greater than or equal to**  `threshold`.


```python
def get_thresholded(images_raw, threshold):
    """
    Perform image thresholding.

        Parameters:
                images_raw (np,array): Do not assume anything about its shape, dtype or range of values. 
                Your function should be careless about these attributes.
                threshold (int): A scalar value.

        Returns:
                threshed_image (np.array): A numpy array with the same shape as images_raw, and the bool dtype. 
                This array should indicate whether each elemelent of images_raw is greater than or equal to 
                threshold.
    """
    
    # your code here
    sh = images_raw.shape
    threshed_image= np.copy(images_raw)
    for idx,x in np.ndenumerate(images_raw):
        if x>=threshold:
            threshed_image[idx] = 1
        elif x<threshold:
            threshed_image[idx]=0
    
    threshed_image=threshed_image.astype(bool)
    
    return threshed_image 
```


```python
# The following are hints to make your life easier duing debugging if you failed the pre-computed tests.
#
#   When an error is raised in checking against the pre-computed test database:
#
#     0. test_results will be a python dictionary, with the bug information stored in it. Don't be afraid to look into it!
#
#     1. You can access the failed test arguments by reading test_results['test_kwargs']. test_results['test_kwargs'] will be
#        another python dictionary with its keys being the argument names and the values being the argument values.
#
#     2. test_results['correct_sol'] will contain the correct solution.
#
#     3. test_results['stu_sol'] will contain your implementation's returned solution.


```


```python
(orig_image, ref_image, test_im, success_thr) = show_test_cases(lambda x: get_thresholded(x, 20), task_id='1_V')

assert success_thr
```

    The reference and solution images are the same to a T! Well done on this test case.



![png](output_22_1.png)


        Enter nothing to go to the next image
    or
        Enter "s" when you are done to recieve the three images. 
            **Don't forget to do this before continuing to the next step.**
    s


### **Warning**: 
Do not leave the previous cell hanging; unless you enter "s" to stop it, you cannot evaluate other cells.


```python
# The following are hints to make your life easier duing debugging if you failed the pre-computed tests.
#
#   When an error is raised in checking against the pre-computed test database:
#
#     0. test_results will be a python dictionary, with the bug information stored in it. Don't be afraid to look into it!
#
#     1. You can access the failed test arguments by reading test_results['test_kwargs']. test_results['test_kwargs'] will be
#        another python dictionary with its keys being the argument names and the values being the argument values.
#
#     2. test_results['correct_sol'] will contain the correct solution.
#
#     3. test_results['stu_sol'] will contain your implementation's returned solution.


```


```python
# Checking against the pre-computed test database
test_results = test_case_checker(get_thresholded, task_id=1)
assert test_results['passed'], test_results['message']
```


```python
train_images_threshed = get_thresholded(train_images_raw, threshold=20)
eval_images_threshed = get_thresholded(eval_images_raw, threshold=20)
```

## 0.3 Creating "Bounding Box" Images

### 0.3.1 Finding Inky Rows

# <span style="color:blue">Task 2</span>

Write the function `get_is_row_inky` that finds the rows with ink pixels and takes following the arguments:

* `images`: A numpy array with the shape `(N,height,width)`, where 
    * `N` is the number of samples and could be anything,
    * `height` is each individual image's height in pixels (i.e., number of rows in each image),
    * and `width` is each individual image's width in pixels (i.e., number of columns in each image).
   
      * Do not assume anything about `images`'s dtype or the number of samples or the `height` or the `width`.

and returns the following:

* `is_row_inky`: A numpy array with the shape `(N, height)`, and the `bool` dtype. 
    * `is_row_inky[i,j]` should be True if **any** of the pixels in the `j`th row of the `i`th image was an ink pixel, and False otherwise.


```python
def get_is_row_inky(images):
    """
    Finds the rows with ink pixels.

        Parameters:
                images (np,array): A numpy array with the shape (N, height, width)

        Returns:
                is_row_inky (np.array): A numpy array with the shape (N, height), and the bool dtype. 
    """
    
    # your code here
    n,h,w = images.shape
    is_row_inky=np.zeros((n, h), dtype=bool)
    
    for i in range(0,n):
        for j in range(0,h):
            for k in range (0,w):
                if images[i][j][k]>0:
                    is_row_inky[i][j]=True
                    
    
    return is_row_inky
```


```python
# The following are hints to make your life easier duing debugging if you failed the pre-computed tests.
#
#   When an error is raised in checking against the pre-computed test database:
#
#     0. test_results will be a python dictionary, with the bug information stored in it. Don't be afraid to look into it!
#
#     1. You can access the failed test arguments by reading test_results['test_kwargs']. test_results['test_kwargs'] will be
#        another python dictionary with its keys being the argument names and the values being the argument values.
#
#     2. test_results['correct_sol'] will contain the correct solution.
#
#     3. test_results['stu_sol'] will contain your implementation's returned solution.


```


```python
(orig_image, ref_image, test_im, success_is_row_inky) = show_test_cases(lambda x: np.expand_dims(get_is_row_inky(x), axis=2), 
                                                                        task_id='2_V')

assert success_is_row_inky
```

    The reference and solution images are the same to a T! Well done on this test case.



![png](output_33_1.png)


        Enter nothing to go to the next image
    or
        Enter "s" when you are done to recieve the three images. 
            **Don't forget to do this before continuing to the next step.**
    s


### **Warning**: 
Do not leave the previous cell hanging; unless you enter "s" to stop it, you cannot evaluate other cells.


```python
# The following are hints to make your life easier duing debugging if you failed the pre-computed tests.
#
#   When an error is raised in checking against the pre-computed test database:
#
#     0. test_results will be a python dictionary, with the bug information stored in it. Don't be afraid to look into it!
#
#     1. You can access the failed test arguments by reading test_results['test_kwargs']. test_results['test_kwargs'] will be
#        another python dictionary with its keys being the argument names and the values being the argument values.
#
#     2. test_results['correct_sol'] will contain the correct solution.
#
#     3. test_results['stu_sol'] will contain your implementation's returned solution.


```


```python
# Checking against the pre-computed test database
test_results = test_case_checker(get_is_row_inky, task_id=2)
assert test_results['passed'], test_results['message']
```

### 0.3.2 Finding Inky Columns

# <span style="color:blue">Task 3</span>

Similar to `get_is_row_inky`, Write the function `get_is_col_inky` that finds the columns with ink pixels and takes following the arguments:

* `images`: A numpy array with the shape `(N,height,width)`, where 
    * `N` is the number of samples and could be anything,
    * `height` is each individual image's height in pixels (i.e., number of rows in each image),
    * and `width` is each individual image's width in pixels (i.e., number of columns in each image).
   
      * **Note**: Do not assume anything about `images`'s dtype or the number of samples or the `height` or the `width`.

and returns the following:

* `is_col_inky`: A numpy array with the shape `(N, width)`, and the `bool` dtype. 
    * `is_col_inky[i,j]` should be True if **any** of the pixels in the `j`th column of the `i`th image was an ink pixel, and False otherwise.


```python
def get_is_col_inky(images):
    """
    Finds the columns with ink pixels.

        Parameters:
                images (np.array): A numpy array with the shape (N,height,width).
                
        Returns:
                is_col_inky (np.array): A numpy array with the shape (N, width), and the bool dtype. 
    """
    
    # your code here
    n,h,w = images.shape
    is_col_inky=np.zeros((n, w), dtype=bool)
    
    for i in range(0,n):
        for j in range(0,h):
            for k in range (0,w):
                if images[i][j][k]>0:
                    is_col_inky[i][k]=True
    
    return is_col_inky
```


```python
# The following are hints to make your life easier duing debugging if you failed the pre-computed tests.
#
#   When an error is raised in checking against the pre-computed test database:
#
#     0. test_results will be a python dictionary, with the bug information stored in it. Don't be afraid to look into it!
#
#     1. You can access the failed test arguments by reading test_results['test_kwargs']. test_results['test_kwargs'] will be
#        another python dictionary with its keys being the argument names and the values being the argument values.
#
#     2. test_results['correct_sol'] will contain the correct solution.
#
#     3. test_results['stu_sol'] will contain your implementation's returned solution.


```


```python
(orig_image, ref_image, test_im, success_is_col_inky) = show_test_cases(lambda x: np.expand_dims(get_is_col_inky(x), axis=1), 
                                                                        task_id='3_V')

assert success_is_col_inky
```

    The reference and solution images are the same to a T! Well done on this test case.



![png](output_42_1.png)


        Enter nothing to go to the next image
    or
        Enter "s" when you are done to recieve the three images. 
            **Don't forget to do this before continuing to the next step.**
    s


### **Warning**: 
Do not leave the previous cell hanging; unless you enter "s" to stop it, you cannot evaluate other cells.


```python
# The following are hints to make your life easier duing debugging if you failed the pre-computed tests.
#
#   When an error is raised in checking against the pre-computed test database:
#
#     0. test_results will be a python dictionary, with the bug information stored in it. Don't be afraid to look into it!
#
#     1. You can access the failed test arguments by reading test_results['test_kwargs']. test_results['test_kwargs'] will be
#        another python dictionary with its keys being the argument names and the values being the argument values.
#
#     2. test_results['correct_sol'] will contain the correct solution.
#
#     3. test_results['stu_sol'] will contain your implementation's returned solution.


```


```python
# Checking against the pre-computed test database
test_results = test_case_checker(get_is_col_inky, task_id=3)
assert test_results['passed'], test_results['message']
```

### 0.3.3.1 Getting the First Inky Rows

# <span style="color:blue">Task 4</span>

Write the function `get_first_ink_row_index` that finds the first row containing ink pixels and takes following the arguments:

* `is_row_inky`: A numpy array with the shape `(N, height)`, and the `bool` dtype. This is the output of the `get_is_row_inky` function that you implemented before.
    
and returns the following:

* `first_ink_rows`: A numpy array with the shape `(N,)`, and the `int64` dtype. 
    * `first_ink_rows[i]` is the index of the first row containing any ink pixel in the `i`th image. The indices should be **zero-based**.


```python
def get_first_ink_row_index(is_row_inky):
    """
     Finds the first row containing ink pixels

        Parameters:
                is_row_inky (np.array): A numpy array with the shape (N, height), and the bool dtype. 
                This is the output of the get_is_row_inky function that you implemented before.
                
        Returns:
                first_ink_rows (np.array): A numpy array with the shape (N,), and the int64 dtype. 
    """
    
    # your code here
    N,h = is_row_inky.shape
    first_ink_rows=[]
    for i in range(0,N):
        first_ink_rows.append(np.min(np.nonzero(is_row_inky[i])))
    first_ink_rows=np.array(first_ink_rows)    
            
    
    return first_ink_rows
```


```python
# Performing sanity checks on your implementation
assert np.array_equal(get_first_ink_row_index(get_is_row_inky(train_images_threshed[:10,:,:])), 
                      np.array([5, 4, 5, 5, 7, 5, 4, 5, 5, 4]))

```


```python
# The following are hints to make your life easier duing debugging if you failed the pre-computed tests.
#
#   When an error is raised in checking against the pre-computed test database:
#
#     0. test_results will be a python dictionary, with the bug information stored in it. Don't be afraid to look into it!
#
#     1. You can access the failed test arguments by reading test_results['test_kwargs']. test_results['test_kwargs'] will be
#        another python dictionary with its keys being the argument names and the values being the argument values.
#
#     2. test_results['correct_sol'] will contain the correct solution.
#
#     3. test_results['stu_sol'] will contain your implementation's returned solution.


```


```python
# Checking against the pre-computed test database
test_results = test_case_checker(get_first_ink_row_index, task_id=4)
assert test_results['passed'], test_results['message']
```

### 0.3.3.2 Getting the First Inky Columns

Since `is_row_inky` and `is_col_inky` have the same data structure, we could re-use the `get_first_ink_row_index` to define a corresponding function `get_first_ink_col_index` for columns; both of them have the same functionality and are essentially the same.


```python
def get_first_ink_col_index(is_col_inky):
    return get_first_ink_row_index(is_col_inky)
```

### 0.3.4.1 Getting the Last Inky Rows

# <span style="color:blue">Task 5</span>

Write the function `get_last_ink_row_index` that finds the last row containing ink pixels and takes following the arguments:

* `is_row_inky`: A numpy array with the shape `(N, height)`, and the `bool` dtype. This is the output of the `get_is_row_inky` function that you implemented before.
    
and returns the following:

* `last_ink_rows`: A numpy array with the shape `(N,)`, and the `int64` dtype. 
    * `last_ink_rows[i]` is the index of the last row containing any ink pixel in the `i`th image. The indices should be **zero-based**.


```python
def get_last_ink_row_index(is_row_inky):
    """
    Finds the last row containing ink pixels.

        Parameters:
                is_row_inky (np.array): A numpy array with the shape (N, height), and the bool dtype. 
                This is the output of the get_is_row_inky function that you implemented before.
                
        Returns:
                last_ink_rows (np.array): A numpy array with the shape (N,), and the int64 dtype. 
    """
    
    # your code here
    N,h = is_row_inky.shape
    last_ink_rows=[]
    for i in range(0,N):
        last_ink_rows.append(np.max(np.nonzero(is_row_inky[i])))
    last_ink_rows=np.array(last_ink_rows)
    
    return last_ink_rows
```


```python
# Performing sanity checks on your implementation
assert (get_last_ink_row_index(get_is_row_inky(train_images_threshed[:10,:,:])) == 
        np.array([24, 23, 24, 24, 26, 22, 23, 24, 24, 23])).all()

```


```python
# The following are hints to make your life easier duing debugging if you failed the pre-computed tests.
#
#   When an error is raised in checking against the pre-computed test database:
#
#     0. test_results will be a python dictionary, with the bug information stored in it. Don't be afraid to look into it!
#
#     1. You can access the failed test arguments by reading test_results['test_kwargs']. test_results['test_kwargs'] will be
#        another python dictionary with its keys being the argument names and the values being the argument values.
#
#     2. test_results['correct_sol'] will contain the correct solution.
#
#     3. test_results['stu_sol'] will contain your implementation's returned solution.


```


```python
# Checking against the pre-computed test database
test_results = test_case_checker(get_last_ink_row_index, task_id=5)
assert test_results['passed'], test_results['message']
```

### 0.3.4.2 Getting the Last Inky Columns

Since `is_row_inky` and `is_col_inky` have the same data structure, we could re-use the `get_last_ink_row_index` to define a corresponding function `get_last_ink_col_index` for columns; both of them have the same functionality and are essentially the same.


```python
def get_last_ink_col_index(is_col_inky):
    return get_last_ink_row_index(is_col_inky)
```

### 0.3.5 The Final "Bounding Box" Pre-processor

# <span style="color:blue">Task 6</span>

Write the function `get_images_bb` that applies the "Bounding Box" pre-processing step and takes the following arguments:

* `images`: A numpy array with the shape `(N,height,width)`, where 
    * `N` is the number of samples and could be anything,
    * `height` is each individual image's height in pixels (i.e., number of rows in each image),
    * and `width` is each individual image's width in pixels (i.e., number of columns in each image).
   
    Do not assume anything about `images`'s dtype or number of samples.
    
* `bb_size`: A scalar with the default value of 20, and represents the desired bounding box size.

and returns the following:

* `images_bb`: A numpy array with the shape `(N,bb_size,bb_size)`, and the same dtype as `images`. 

We have provided a template function that uses the previous functions and only requires you to fill in the missing parts. It also handles the input shapes in an agnostic way.

**Important Note**: Make sure that you use the `np.roll` function for this implementation.

**Clarification with Example**: Here we will describe in exact details how to produce the output image.

Let's focus on a single raw image, and let's call it `X`. We wish to produce the output image `Y`. Let's assume `X` has a `height` and a `width` of 28, and `bb_size` is 20. This means that `X` has a shape of `(28,28)` and `Y` has a shape of `(20,20)`. As a visual example, we'll assume that `X` is the left image (i.e., the raw image) shown below, and `Y` is the right image (i.e., the solution image). 

![alt text](../ClassifyingImages-lib/bb_example.png "Bounding Box Definitions")

Let's define the first/last inky rows with an example:
 * The **first inky row** of `X` has an index of $r_1$. In the picture example, we have $r_1=6$. This means that `X[5,:]` has no ink in it, and `X[6,:]` has some ink elements in it.
 * The **last inky row** of `X` has an index of $r_2$. In the picture example, we have $r_2=25$. This means that `X[25,:]` has some ink in it, and `X[26,:]` has no ink in it.
 
Let's define the first/last inky columns in a similar manner:
 * The **first inky column** of `X` has an index of $c_1$. In the picture example, we have $c_1=5$. This means that `X[:,4]` has no ink in it, and `X[:,5]` has some ink elements in it.
 * The **last inky column** of `X` has an index of $c_2$. In the picture example, we have $c_2=20$. This means that `X[:,20]` has some ink in it, and `X[:,21]` has no ink in it.
 
 
Now let's define the **inky middle row/column** of the raw image.

 * The **inky middle row** of the raw image is $r_m = \lfloor \frac{r_1 + r_2 + 1}{2} \rfloor$. In this example, we have $r_m=16$, which is also shown in the picture.
 * The **inky middle column** of the raw image is $c_m = \lfloor \frac{c_1 + c_2 + 1}{2} \rfloor$. In this example, we have $c_m=13$, which is also shown in the picture.


The middle row index of the output image is $r_{out} = \lfloor \frac{\text{bb_size}}{2} \rfloor$. Similarly, we have the middle column index of the output image $c_{out} = \lfloor \frac{\text{bb_size}}{2} \rfloor$. In this example, we have $r_{out}=c_{out}=10$, which are marked with blue boxes in the solution image.

The **middle inky pixel of the raw image** is `X[r_m, c_m]`. This middle inky pixel is colored red in the raw image for clarification.

The **middle inky pixel of the solution image** is `Y[r_out, c_out]`. This middle inky pixel is colored red in the solution image for clarification

You must shift the raw image in a way that the **middle inky pixel of the raw image** gets placed on the **middle inky pixel of the solution image**. In other words, the red pixels should be placed on top of each other.

You will also have to cut some of rows/columns of the solution image properly to make sure it would have a shape of `(bb_size, bb_size)`. Furthermore, this whole discussion was for a single image, but you will have to take care of a batch of images as input, and produce a batch of bounded-box images as an output, and make sure all the dimensions/shapes work out properly.


```python
def get_images_bb(images, bb_size=20):
    """
    Applies the "Bounding Box" pre-processing step to images.

        Parameters:
                images (np.array): A numpy array with the shape (N,height,width)
                
        Returns:
                images_bb (np.array): A numpy array with the shape (N,bb_size,bb_size), 
                and the same dtype as images. 
    """
    
    if len(images.shape)==2:
        # In case a 2d image was given as input, we'll add a dummy dimension to be consistent
        images_ = images.reshape(1,*images.shape)
    else:
        # Otherwise, we'll just work with what's given
        images_ = images
        
    is_row_inky = get_is_row_inky(images_)
    is_col_inky = get_is_col_inky(images_)
    
    first_ink_rows = get_first_ink_row_index(is_row_inky)
    last_ink_rows = get_last_ink_row_index(is_row_inky)
    first_ink_cols = get_first_ink_col_index(is_col_inky)
    last_ink_cols = get_last_ink_col_index(is_col_inky)
    N,height,width = images_.shape
    images_bb=[]
    for i in range (0,N):
        midpointy = int((first_ink_rows[i]+ last_ink_rows[i]+1)/2)
        midpointx = int ((first_ink_cols[i]+last_ink_cols[i]+1)/2)
        dx = midpointx - int(bb_size/2)
        dy = midpointy - int(bb_size/2)
        img_roll = images_[i].copy()
        img_roll = np.roll(img_roll, -dy, axis = 0)    # Positive y rolls up
        img_roll = np.roll(img_roll, -dx, axis = 1) #positive x rolls right
        final_img = img_roll[0:bb_size,0:bb_size]
        images_bb.append(final_img)
        
    
        
    images_bb = np.array(images_bb)
    if len(images.shape)==2:
        # In case a 2d image was given as input, we'll get rid of the dummy dimension
        return images_bb[0]
    else:
        # Otherwise, we'll just work with what's given
        return images_bb
```


```python
# The following are hints to make your life easier duing debugging if you failed the pre-computed tests.
#
#   When an error is raised in checking against the pre-computed test database:
#
#     0. test_results will be a python dictionary, with the bug information stored in it. Don't be afraid to look into it!
#
#     1. You can access the failed test arguments by reading test_results['test_kwargs']. test_results['test_kwargs'] will be
#        another python dictionary with its keys being the argument names and the values being the argument values.
#
#     2. test_results['correct_sol'] will contain the correct solution.
#
#     3. test_results['stu_sol'] will contain your implementation's returned solution.


```


```python
(orig_image, ref_image, test_im, success_bb) = show_test_cases(get_images_bb, task_id='6_V')

assert success_bb
```

    The reference and solution images are the same to a T! Well done on this test case.



![png](output_72_1.png)


        Enter nothing to go to the next image
    or
        Enter "s" when you are done to recieve the three images. 
            **Don't forget to do this before continuing to the next step.**
    s


### **Warning**: 
Do not leave the previous cell hanging; unless you enter "s" to stop it, you cannot evaluate other cells.


```python
# The following are hints to make your life easier duing debugging if you failed the pre-computed tests.
#
#   When an error is raised in checking against the pre-computed test database:
#
#     0. test_results will be a python dictionary, with the bug information stored in it. Don't be afraid to look into it!
#
#     1. You can access the failed test arguments by reading test_results['test_kwargs']. test_results['test_kwargs'] will be
#        another python dictionary with its keys being the argument names and the values being the argument values.
#
#     2. test_results['correct_sol'] will contain the correct solution.
#
#     3. test_results['stu_sol'] will contain your implementation's returned solution.


```


```python
# Checking against the pre-computed test database
test_results = test_case_checker(get_images_bb, task_id=6)
assert test_results['passed'], test_results['message']
```


```python
train_images_bb = get_images_bb(train_images_threshed)
eval_images_bb = get_images_bb(eval_images_threshed)
```

### 0.3.6 The Final "Stretched Bounding Box" Pre-processor

# <span style="color:blue">Task 7</span>

Similarly, write the function `get_images_sbb` that applies the "Stretched Bounding Box" pre-processing step and takes following the arguments:

* `images`: A numpy array with the shape `(N,height,width)`, where 
    * `N` is the number of samples and could be anything,
    * `height` is each individual image's height in pixels (i.e., number of rows in each image),
    * and `width` is each individual image's width in pixels (i.e., number of columns in each image).
   
    Do not assume anything about `images`'s dtype or number of samples.
    
* `bb_size`: A scalar with the default value of 20, and represents the desired bounding box size.

and returns the following:

* `images_sbb`: A numpy array with the shape `(N,bb_size,bb_size)`, and the same dtype and the range of values as `images`. 


The `get_images_sbb` should find a tight-canvas of the inky area in each input image, and stretch it out to fill the full height and width of the output bounding-box. Please see the visual example in the **Assignment Summary** section; the right image should supposedly be the `get_images_sbb` function's output.

We have provided a template function that uses the previous functions and only requires you to fill in the missing parts. It also handles the input shapes in an agnostic way.

**Hint**: Make sure that you use the `skimage.transform.resize` function from the skimage library. Read about it at https://scikit-image.org/docs/dev/api/skimage.transform.html?highlight=resize#skimage.transform.resize. You may need to pay attention to the `preserve_range` argument.


```python
def get_images_sbb(images, bb_size=20):
    """
    Applies the "Stretched Bounding Box" pre-processing step to images.

        Parameters:
                images (np.array): A numpy array with the shape (N,height,width)
                
        Returns:
                images_sbb (np.array): A numpy array with the shape (N,bb_size,bb_size), 
                and the same dtype and the range of values as images. 
    """
    
    if len(images.shape)==2:
        # In case a 2d image was given as input, we'll add a dummy dimension to be consistent
        images_ = images.reshape(1,*images.shape)
    else:
        # Otherwise, we'll just work with what's given
        images_ = images
        
    is_row_inky = get_is_row_inky(images)
    is_col_inky = get_is_col_inky(images)
    
    first_ink_rows = get_first_ink_row_index(is_row_inky)
    last_ink_rows = get_last_ink_row_index(is_row_inky)
    first_ink_cols = get_first_ink_col_index(is_col_inky)
    last_ink_cols = get_last_ink_col_index(is_col_inky)
    images_sbb=[]
    N,height,width = images_.shape
    for i in range (0,N):
        precropped_img = images_[i].copy()
        cropped_img = precropped_img[(first_ink_rows[i]):(last_ink_rows[i]+1),(first_ink_cols[i]):(last_ink_cols[i]+1)]
        resized_img  = resize(cropped_img, [bb_size,bb_size], order=None, mode='reflect', cval=0, clip=True, preserve_range=True, anti_aliasing=None, anti_aliasing_sigma=None)
        resized_img = resized_img.astype(cropped_img.dtype)
        images_sbb.append(resized_img)
    
        
    images_sbb = np.array(images_sbb)
    
    if len(images.shape)==2:
        # In case a 2d image was given as input, we'll get rid of the dummy dimension
        return images_sbb[0]
    else:
        # Otherwise, we'll just work with what's given
        return images_sbb
```


```python
# The following are hints to make your life easier duing debugging if you failed the pre-computed tests.
#
#   When an error is raised in checking against the pre-computed test database:
#
#     0. test_results will be a python dictionary, with the bug information stored in it. Don't be afraid to look into it!
#
#     1. You can access the failed test arguments by reading test_results['test_kwargs']. test_results['test_kwargs'] will be
#        another python dictionary with its keys being the argument names and the values being the argument values.
#
#     2. test_results['correct_sol'] will contain the correct solution.
#
#     3. test_results['stu_sol'] will contain your implementation's returned solution.


```


```python
(orig_image, ref_image, test_im, success_sbb) = show_test_cases(get_images_sbb, task_id='7_V')

assert success_sbb
```

    The reference and solution images are the same to a T! Well done on this test case.



![png](output_82_1.png)


        Enter nothing to go to the next image
    or
        Enter "s" when you are done to recieve the three images. 
            **Don't forget to do this before continuing to the next step.**
    s


### **Warning**: 
Do not leave the previous cell hanging; unless you enter "s" to stop it, you cannot evaluate other cells.


```python
# The following are hints to make your life easier duing debugging if you failed the pre-computed tests.
#
#   When an error is raised in checking against the pre-computed test database:
#
#     0. test_results will be a python dictionary, with the bug information stored in it. Don't be afraid to look into it!
#
#     1. You can access the failed test arguments by reading test_results['test_kwargs']. test_results['test_kwargs'] will be
#        another python dictionary with its keys being the argument names and the values being the argument values.
#
#     2. test_results['correct_sol'] will contain the correct solution.
#
#     3. test_results['stu_sol'] will contain your implementation's returned solution.


```


```python
# Checking against the pre-computed test database
test_results = test_case_checker(get_images_sbb, task_id=7)
assert test_results['passed'], test_results['message']
```


```python
if perform_computation:
    print('This is gonna take a while to finish...')
    %time train_images_sbb = get_images_sbb(train_images_threshed)
    %time eval_images_sbb = get_images_sbb(eval_images_threshed)
```

    This is gonna take a while to finish...
    CPU times: user 2min 46s, sys: 18.9 s, total: 3min 5s
    Wall time: 2min 34s
    CPU times: user 27.2 s, sys: 3.23 s, total: 30.4 s
    Wall time: 25.2 s


# 1. Naive Bayes Performances

# <span style="color:blue">Task 8</span>

Similarly, write the function `train_nb_eval_acc` that trains Naive Bayes models and takes following the arguments:

* `train_images`: A numpy array with the shape `(N,height,width)`, where 
    * `N` is the number of samples and could be anything,
    * `height` is each individual image's height in pixels (i.e., number of rows in each image),
    * and `width` is each individual image's width in pixels (i.e., number of columns in each image).
 
    Do not assume anything about `images`'s dtype or number of samples.

* `train_labels`: A numpy array with the shape `(N,)`, where `N` is the number of samples and has the `int64` dtype.

* `eval_images`: The evaluation images with similar characteristics to `train_images`.

* `eval_labels`: The evaluation labels with similar characteristics to `train_labels`.
    
* `density_model`: A string that is either `'Gaussian'` or `'Bernoulli'`. In the former (resp. latter) case, you should train a Naive Bayes with the Gaussian (resp. Bernoulli) density model.

and returns the following:

* `eval_acc`: a floating number scalar between 0 and 1 that represents the accuracy of the trained model on the evaluation data.

We have provided a template function that uses the previous functions and only requires you to fill in the missing parts. It also handles the input shapes in an agnostic way.

**Note**: You do not need to implement the Naive Bayes classifier from scratch in this assignment; Make sure you use `scikit-learn`'s Naive Bayes module for training and prediction in this task. We have already imported these two functions in the first code cell:

  * `from sklearn.naive_bayes import GaussianNB, BernoulliNB`


```python
def train_nb_eval_acc(train_images, train_labels, eval_images, eval_labels, density_model='Gaussian'):
    """
    Trains Naive Bayes models, apply the model, and return an accuracy.

        Parameters:
                train_images (np.array): A numpy array with the shape (N,height,width)
                train_labels (np.array): A numpy array with the shape (N,), where N is the number of samples and 
                has the int64 dtype.
                eval_images (np.array): The evaluation images with similar characteristics to train_images.
                eval_labels (np.array): The evaluation labels with similar characteristics to train_labels.
                density_model (string): A string that is either 'Gaussian' or 'Bernoulli'. 
                
        Returns:
                eval_acc (np.float): a floating number scalar between 0 and 1 that 
                represents the accuracy of the trained model on the evaluation data.
    """
    
    assert density_model in ('Gaussian', 'Bernoulli')
    Nt,ht,wt = train_images.shape
    Ne,he,we = eval_images.shape
    train_img = train_images.reshape((Nt, ht*wt))
    eval_img = eval_images.reshape((Ne, he*we))
    if density_model.lower() == 'gaussian':
        gnb = GaussianNB()
        gnb = gnb.fit(train_img, train_labels)
        
        eval_acc = gnb.score(eval_img,eval_labels)
    else :
        bnb = BernoulliNB()
        bnb = bnb.fit(train_img, train_labels)
        
        eval_acc = bnb.score(eval_img,eval_labels)
        
    
    
    # your code here
    
    
    return eval_acc

# Don't mind the following lines and do not change them
train_nb_eval_acc_gauss = lambda *args, **kwargs: train_nb_eval_acc(*args, density_model='Gaussian', **kwargs)
train_nb_eval_acc_bern = lambda *args, **kwargs: train_nb_eval_acc(*args, density_model='Bernoulli', **kwargs)
```


```python

```


```python
# The following are hints to make your life easier duing debugging if you failed the pre-computed tests.
#
#   When an error is raised in checking against the pre-computed test database:
#
#     0. test_results will be a python dictionary, with the bug information stored in it. Don't be afraid to look into it!
#
#     1. You can access the failed test arguments by reading test_results['test_kwargs']. test_results['test_kwargs'] will be
#        another python dictionary with its keys being the argument names and the values being the argument values.
#
#     2. test_results['correct_sol'] will contain the correct solution.
#
#     3. test_results['stu_sol'] will contain your implementation's returned solution.


```


```python
# Checking against the pre-computed test database
test_results = test_case_checker(train_nb_eval_acc_gauss, task_id='8_G')
assert test_results['passed'], test_results['message'] # Gaussian Model Test Results

test_results = test_case_checker(train_nb_eval_acc_bern, task_id='8_B')
assert test_results['passed'], test_results['message'] # Bernoulli Model Test Results
```


```python

```


```python
df = None
if perform_computation:
    acc_nbg_thr = train_nb_eval_acc(train_images_threshed, train_labels, 
                                    eval_images_threshed, eval_labels, density_model='Gaussian')
    acc_nbb_thr = train_nb_eval_acc(train_images_threshed, train_labels, 
                                    eval_images_threshed, eval_labels, density_model='Bernoulli')
    acc_nbg_sbb = train_nb_eval_acc(train_images_sbb, train_labels, 
                                    eval_images_sbb, eval_labels, density_model='Gaussian')
    acc_nbb_sbb = train_nb_eval_acc(train_images_sbb, train_labels, 
                                    eval_images_sbb, eval_labels, density_model='Bernoulli')

    df = pd.DataFrame([('Untouched images', acc_nbg_thr, acc_nbb_thr),
                       ('Stretched bounding box', acc_nbg_sbb, acc_nbb_sbb)
                      ], columns = ['Accuracy' , 'Gaussian', 'Bernoulli'])

df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>Gaussian</th>
      <th>Bernoulli</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Untouched images</td>
      <td>0.5491</td>
      <td>0.8430</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Stretched bounding box</td>
      <td>0.8253</td>
      <td>0.8098</td>
    </tr>
  </tbody>
</table>
</div>



# 2. Decision Forests Performances

# <span style="color:blue">Task 9</span>

Similarly, write the function `train_tree_eval_acc` that trains Decision Forest models and takes following the arguments:

* `train_images`: A numpy array with the shape `(N,height,width)`, where 
    * `N` is the number of samples and could be anything,
    * `height` is each individual image's height in pixels (i.e., number of rows in each image),
    * and `width` is each individual image's width in pixels (i.e., number of columns in each image).
 
    Do not assume anything about `images`'s dtype or number of samples.

* `train_labels`: A numpy array with the shape `(N,)`, where `N` is the number of samples and has the `int64` dtype.

* `eval_images`: The evaluation images with similar characteristics to `train_images`.

* `eval_labels`: The evaluation labels with similar characteristics to `eval_labels`.
    
* `tree_num`: An integer number representing the number of trees in the dicision forest.

* `tree_depth`: An integer number representing the maximum tree depth in the dicision forest.

* `random_state`: An integer with a default value of 12345 that should be passed to the scikit-learn's classifer constructor for reproducibility and auto-grading (**Do not assume** that it is always 12345).

and returns the following:

* `eval_acc`: A floating number scalar between 0 and 1 that represents the accuracy of the trained model on the evaluation data.

We have provided a template function that uses the previous functions and only requires you to fill in the missing parts. It also handles the input shapes in an agnostic way.

**Note**: You do not need to implement the Random Forest classifier from scratch in this assignment; Make sure you use `scikit-learn`'s Random Forest module for training and prediction in this task. We have already imported this function in the first code cell:

  * `from sklearn.ensemble import RandomForestClassifier`
  * You may need to set "shuffle = True" due to a known sklearn issue.


```python

def train_tree_eval_acc(train_images, train_labels, eval_images, eval_labels, tree_num=10, tree_depth=4, random_state=12345):
    """
    Trains Naive Bayes models, apply the model, and return an accuracy.

        Parameters:
                train_images (np.array): A numpy array with the shape (N,height,width)
                train_labels (np.array): A numpy array with the shape (N,), where N is the number of samples and 
                has the int64 dtype.
                eval_images (np.array): The evaluation images with similar characteristics to train_images.
                eval_labels (np.array): The evaluation labels with similar characteristics to train_labels.
                tree_num (int): An integer number representing the number of trees in the decision forest.
                tree_depth (int): An integer number representing the maximum tree depth in the decision forest.
                random_state (int): An integer with a default value of 12345 that should be passed to 
                the scikit-learn's classifer constructor for reproducibility and auto-grading
                
        Returns:
                eval_acc (np.float): a floating number scalar between 0 and 1 that 
                represents the accuracy of the trained model on the evaluation data.
    """
    
    tree_num = int(tree_num)
    tree_depth = int(tree_depth)
    random_state = int(random_state)

    Nt,ht,wt = train_images.shape
    Ne,he,we = eval_images.shape
    train_img = train_images.reshape((Nt, ht*wt))
    eval_img = eval_images.reshape((Ne, he*we))
    model = RandomForestClassifier(tree_num,"gini", tree_depth,2,1,0.0,"auto",None,0.0,None,True,False,None,random_state,0,False,None,0.0,None)
    model = model.fit(train_img, train_labels)
    eval_label_pred = model.predict(eval_img)
    
    eval_acc = model.score(eval_img,eval_labels) 
    

        
    # your code here
    
    
    return eval_acc
```


```python
# The following are hints to make your life easier duing debugging if you failed the pre-computed tests.
#
#   When an error is raised in checking against the pre-computed test database:
#
#     0. test_results will be a python dictionary, with the bug information stored in it. Don't be afraid to look into it!
#
#     1. You can access the failed test arguments by reading test_results['test_kwargs']. test_results['test_kwargs'] will be
#        another python dictionary with its keys being the argument names and the values being the argument values.
#
#     2. test_results['correct_sol'] will contain the correct solution.
#
#     3. test_results['stu_sol'] will contain your implementation's returned solution.


```


```python
# Checking against the pre-computed test database
test_results = test_case_checker(train_tree_eval_acc, task_id=9)
assert test_results['passed'], test_results['message']
```


```python

```

## 2.1 Accuracy on the Untouched Images


```python
df = None
if perform_computation:
    tree_nums = [10, 20, 30]
    tree_depths = [4, 8, 16]

    train_images = train_images_threshed
    eval_images = eval_images_threshed
    acc_arr_unt = np.zeros((len(tree_nums), len(tree_depths)))
    for row, tree_num in enumerate(tree_nums):
        for col, tree_depth in enumerate(tree_depths):
            acc_arr_unt[row, col] = train_tree_eval_acc(train_images, train_labels, eval_images, eval_labels, 
                                                        tree_num=tree_num, tree_depth=tree_depth, random_state=12345)

    df = pd.DataFrame([(f'#trees = {tree_num}', *tuple(acc_arr_unt[row])) for row, tree_num in enumerate(tree_nums)],
                      columns = ['Accuracy'] + [f'depth={tree_depth}'for col, tree_depth in enumerate(tree_depths)])

    print('Untouched Images:')
df
```

    Untouched Images:





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>depth=4</th>
      <th>depth=8</th>
      <th>depth=16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>#trees = 10</td>
      <td>0.7496</td>
      <td>0.8923</td>
      <td>0.9489</td>
    </tr>
    <tr>
      <th>1</th>
      <td>#trees = 20</td>
      <td>0.7707</td>
      <td>0.9127</td>
      <td>0.9585</td>
    </tr>
    <tr>
      <th>2</th>
      <td>#trees = 30</td>
      <td>0.7883</td>
      <td>0.9169</td>
      <td>0.9630</td>
    </tr>
  </tbody>
</table>
</div>



## 2.2 Accuracy on the "Stretched Bounding Box" Images


```python
df = None
if perform_computation:
    tree_nums = [10, 20, 30]
    tree_depths = [4, 8, 16]

    train_images = train_images_sbb
    eval_images = eval_images_sbb
    acc_arr_sbb = np.zeros((len(tree_nums), len(tree_depths)))
    for row, tree_num in enumerate(tree_nums):
        for col, tree_depth in enumerate(tree_depths):
            acc_arr_sbb[row, col] = train_tree_eval_acc(train_images, train_labels, eval_images, eval_labels, 
                                                        tree_num=tree_num, tree_depth=tree_depth, random_state=12345)

    df = pd.DataFrame([(f'#trees = {tree_num}', *tuple(acc_arr_sbb[row])) for row, tree_num in enumerate(tree_nums)],
                      columns = ['Accuracy'] + [f'depth = {tree_depth}'for col, tree_depth in enumerate(tree_depths)])

    print('Stretched Bounding Box Images:')
df
```

    Stretched Bounding Box Images:





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>depth = 4</th>
      <th>depth = 8</th>
      <th>depth = 16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>#trees = 10</td>
      <td>0.7419</td>
      <td>0.9043</td>
      <td>0.9527</td>
    </tr>
    <tr>
      <th>1</th>
      <td>#trees = 20</td>
      <td>0.7715</td>
      <td>0.9162</td>
      <td>0.9639</td>
    </tr>
    <tr>
      <th>2</th>
      <td>#trees = 30</td>
      <td>0.7879</td>
      <td>0.9248</td>
      <td>0.9671</td>
    </tr>
  </tbody>
</table>
</div>



## 2.3 Accuracy on the "Bounding Box" Images


```python
df = None
if perform_computation:
    tree_nums = [10, 20, 30]
    tree_depths = [4, 8, 16]

    train_images = train_images_bb
    eval_images = eval_images_bb
    acc_arr_bb = np.zeros((len(tree_nums), len(tree_depths)))
    for row, tree_num in enumerate(tree_nums):
        for col, tree_depth in enumerate(tree_depths):
            acc_arr_bb[row, col] = train_tree_eval_acc(train_images, train_labels, eval_images, eval_labels, 
                                                       tree_num=tree_num, tree_depth=tree_depth, random_state=12345)

    df = pd.DataFrame([(f'#trees = {tree_num}', *tuple(acc_arr_bb[row])) for row, tree_num in enumerate(tree_nums)],
                      columns = ['Accuracy'] + [f'depth = {tree_depth}'for col, tree_depth in enumerate(tree_depths)])

    print('Bounding Box Images:')
df
```

    Bounding Box Images:





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Accuracy</th>
      <th>depth = 4</th>
      <th>depth = 8</th>
      <th>depth = 16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>#trees = 10</td>
      <td>0.7406</td>
      <td>0.8865</td>
      <td>0.9476</td>
    </tr>
    <tr>
      <th>1</th>
      <td>#trees = 20</td>
      <td>0.7716</td>
      <td>0.9050</td>
      <td>0.9576</td>
    </tr>
    <tr>
      <th>2</th>
      <td>#trees = 30</td>
      <td>0.7801</td>
      <td>0.9089</td>
      <td>0.9608</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
