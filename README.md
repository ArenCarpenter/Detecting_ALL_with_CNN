# Detecting Acute Lymphoblastic Leukemia (ALL) with a Convolutional Neural Network

Aren Carpenter 

DS Cohort 062220

### Repository Navigation

- **001_Load_and_Clean_Images.ipynb** // From downloading images, split into folder hierarchy of normal and all subdirectories in train/test/validation superdirectories.
- **002_Exploratory_Data_Analysis.ipynb** // Creating visualizations of representative images, mean images, and class imbalance, in addition to model visuals.
- **003_Modeling.ipynb** // Script for utilizing AWS SageMaker training instances and accessing AWS S3 buckets for storing images.
- **Model_Scripts** // Directory for defining Keras model architectures as python scripts to be called in the Modeling.ipynb. 

The slide deck for this project can be found [here](https://docs.google.com/presentation/d/1lgJ2BSfsK7DATqfkMKr0cNbLKv_MyEpwrUyLg41oASc/edit?usp=sharing).

## Introduction

Acute Lymphoblastic Leukemia (ALL) is the most common pediatric cancer and the most frequent cause of death from cancer before 20 years of age. In the 1960s ALL had a survival rate of only 10%, but advancements in diagnostic testing and refinements to chemotherapies have have increased survival rates to 90% (in developed countries, that is). ([1](https://www.nejm.org/doi/full/10.1056/NEJMra1400972)) Researchers are attempting a variety of personalized approaches, mainly using epigenetic screenings and genome-wide association studies (GWAS) to identify potential targets for inhibition, to push survival rates even higher. ([2](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4567699/), [3](https://www.nature.com/articles/bcj201753)) About 80% of ALL cases are children, but, as Terwilliger and Abdul-Hay note, there is another peak of ALL incidence at 50 years of age and long-term remission rates in the older subset of patients is lower than children, about 30-40%. ([3](https://www.nature.com/articles/bcj201753))

ALL is described as the proliferation and differentiation of lymphoid cells in the bone marrow. Important cellular processes, such as the regulation of lymphoid differentiation, cell cycle regulation, growth factor and tumor-suppressor receptor signaling, and epigenetic modification, are perturbed. Additionally, chromosomal translocations are present in about a third of ALL cases. This can cause the overexpression of  oncogenes by relocating them to actively transcribed regions or underexpression of tumor-suppressing genes by relocating them to non-transcribed regions of the genome. ([1](https://www.nejm.org/doi/full/10.1056/NEJMra1400972), [3](https://www.nature.com/articles/bcj201753)) ALL is commonly polyclonal which further complicates treatment because a number of sub-populations will likely be resistent to any one treatment. ([1](https://www.nejm.org/doi/full/10.1056/NEJMra1400972))

### ALL Cell Morphology

ALL can be split into 3 distinct subtypes that makes identification difficult, even for experienced practitioners. L1 are small and homogeneous, round with no clefting, and with no obvious nucleoli or vacuoles. These are the most likely to pass as normal lymphoblasts. L2 are larger and heterogeneous, irregular shape and often clefted, and have defined nucleoli and vacuoles. L3 have the shape of L1 but have prominent nucleoli and vacuoles. ([4](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4335145/), [5](http://piurilabs.di.unimi.it/Papers/cimsa_2005.pdf))

![](Images/FAB.jpeg)

## Data Collection

The data consists of 10,000+ images of single-cell microscopy acute lymphoblastic leukemia and normal lymphoblasts with a class imbalance of about 2:1 ALL to normal. Having enough images and computing resources without using all images, I decided to downsample the positive ALL class to manage class imbalance. 

Images are 450x450 RGB images stored as .bmp files, a raster graphics bitmap which stores images as 2D matrices.

Data can be found [here](https://app.box.com/s/xeclwwd2xep9ntljtgyptmt4k5wone9n).

## Exploratory Data Analysis

Here are some normal cells from our set. We see spherical, non-clefted cells with homogeneous chromatin and few vacuoles. 

<p float="left">
  <img src="Images/normal_1.bmp" width="150" />
  <img src="Images/normal_2.bmp" width="150" /> 
  <img src="Images/normal_3.bmp" width="150" />
  <img src="Images/normal_4.bmp" width="150" />
</p>

Here are some ALL cells from our set. We see irregularly shaped, clefted cells with heterogeneous chromatin and multiple nucleoli and vacuoles. 

<p float="left">
  <img src="Images/all_1.bmp" width="150" />
  <img src="Images/all_2.bmp" width="150" /> 
  <img src="Images/all_3.bmp" width="150" />
  <img src="Images/all_4.bmp" width="150" />
</p>

### Average Images

Looking at the average image for each class we see that the interior of the cells have too much variation to identify meaningful differences, but we see clearly that ALL cells are much larger on average than normal cells. This should not be surprising as cancerous cells have unregulated growth. 

<p float="left">
  <img src="Images/Average_Normal.png" width="300" />
  <img src="Images/Average_ALL.png" width="300" /> 
</p>

## Modeling

> *I used this [post](https://blog.betomorrow.com/keras-in-the-cloud-with-amazon-sagemaker-67cf11fb536) from Paul Breton and the corresponding GitHub [repo](https://github.com/Pravez/KerasSageMaker) for guidance on utilizing Keras with Sagemaker.*  

I utilized the Keras framework in AWS Sagemaker by specifying neural network architecture and compilation hyperparameters in a separate Python script located in the Model_Scripts directory. Training was accomplished in a ml.m4.xlarge notebook instance allowing for hundreds of epochs in a tractable training time. 

I adopted an iterative approach to modeling based on the CRISP-DM process. A dummy classifier predicting the majority class would have an accuracy of 57%. I created a Baseline model with a single Conv2D layer and a single Dense layer which had an accuracy of 68%, already better than the dummy. I then created successively larger and more complex architectures by adding additional Conv2D layers and blocks of layers separated by MaxPooling layers. 

The most complex had 9 convolutions in 3 blocks of 3 layers, but this was not the most successful model. It became clear that deep, but narrow blocks were achieving higher metrics than wider blocks. The best model was a 2x2x1 architecture with 5 total convolutions. Dropout layers of 25% were added after MaxPooling layers to combat overfitting. 

Final network architecture:

![]()

#### Model Compilation Hyperparameters

I used binary crossentropy for the loss function as this is a binary classification problem, and RMSprop for optimization. The learning rate was set to 0.001 with a decay of 0.0001.

## Insights and Recommendations

The model achieved high accuracy of xx, allowing it to be a useful tool for identifying ALL in novel cases. As blood sample microscopy is already the default diagnostic test for ALL, this model could easily be used to verify a human physician or to flag cases that the model is not confident in for further review. 

## Next Steps

### Model Improvements

Ways to improve the model via feature engineering or hyperparameter tuning.

### Product Improvements

Model interpretability is often as or more important than model accuracy, especially for medical diagnostic needs. It is very important in real-world applications that a doctor can see why the model has reached a certain decision. To that end, building an image segmentation model that identifies and marks important features, such as presence and number of vaculoes, non-spherical cells, or clefted edges.