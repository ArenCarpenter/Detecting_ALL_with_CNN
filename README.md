# Detecting Acute Lymphoblastic Leukemia (ALL) with a Convolutional Neural Network

**Aren Carpenter** 

DS Cohort 062220

## Introduction

Acute Lymphoblastic Leukemia (ALL) is the most common pediatric cancer and the most frequent cause of death from cancer before 20 years of age. In the 1960s ALL had a survival rate of only 10%, but advancements in diagnostic testing and refinements to chemotherapies have have increased survival rates to 90% in developed countries. ([1](https://www.nejm.org/doi/full/10.1056/NEJMra1400972)) Researchers are attempting a variety of personalized approaches, mainly using epigenetic screenings and genome-wide association studies (GWAS) to identify potential targets for inhibition, to push survival rates even higher. ([2](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4567699/), [3](https://www.nature.com/articles/bcj201753)) About 80% of ALL cases are children, but, as Terwilliger and Abdul-Hay note, there is another peak of ALL incidence at 50 years of age and long-term remission rates in the older subset of patients is lower than children, about 30-40%. ([3](https://www.nature.com/articles/bcj201753))

ALL is described as the proliferation and differentiation of lymphoid cells in the bone marrow. Important cellular processes, such as the regulation of lymphoid differentiation, cell cycle regulation, growth factor and tumor-suppressor receptor signaling, and epigenetic modification, are perturbed. Additionally, chromosomal translocations are present in about a third of ALL cases. This can cause the overexpression of  oncogenes by relocating them to actively transcribed regions or underexpression of tumor-suppressing genes by relocating them to non-transcribed regions of the genome. ([1](https://www.nejm.org/doi/full/10.1056/NEJMra1400972), [3](https://www.nature.com/articles/bcj201753)) ALL is commonly polyclonal which further complicates treatment because a number of sub-populations will likely be resistant to any one treatment. ([1](https://www.nejm.org/doi/full/10.1056/NEJMra1400972))

### Repository Navigation

- **[001_Load_and_Clean_Images.ipynb](001_Load_and_Clean_Images.ipynb)**: From downloading images, split into folder hierarchy of normal and all subdirectories in train/test/validation superdirectories.
- **[002_Exploratory_Data_Analysis.ipynb](002_Exploratory_Data_Analysis.ipynb)**: Creating visualizations of representative images, mean images, and class imbalance, in addition to model visuals.
- **[003_Modeling_AWS.ipynb](003_Modeling_AWS.ipynb)**: Script for utilizing AWS SageMaker training instances and accessing AWS S3 buckets for storing images.
- **[004_Modeling_Local.ipynb](004_Modeling_Local.ipynb)**: CPU-based local modeling with Keras framework
- **[005_Transfer_Learning](005_Transfer_Learning.ipynb)**: Local modeling for transfer learning with Keras framework, including Xception, VGG16, and ResNet
- **Model_Scripts**: Directory for defining Keras model architectures as python scripts to be called in the 003_Modeling_AWS.ipynb. 

One should run 001_Load_and_Clean_Images and 002_EDA in a local notebook to create directories locally for use in EDA, but one must use AWS Sagemaker for the 003_Modeling_AWS file to work. Individual models are run by calling a script from the Modeling_Scripts folder in Sagemaker. Local models can be run via the Keras framework using 004_Modeling_Local. 

The slide deck for this project can be found [here](https://docs.google.com/presentation/d/1Ghsj7xBMBZUB8QsEuey53VU2La74BctSShgnL4ZN4eo/edit?usp=sharing). And this project is explored further via a Medium article [here](https://medium.com/@arencarpenter).

### ALL Cell Morphology

ALL can be split into 3 distinct subtypes that makes identification difficult, even for experienced practitioners. L1 are small and homogeneous, round with no clefting, and with no obvious nucleoli or vacuoles. These are the most likely to pass as normal lymphoblasts. L2 are larger and heterogeneous, irregular shape and often clefted, and have defined nucleoli and vacuoles. L3 have the shape of L1 but have prominent nucleoli and vacuoles. ([4](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4335145/), [5](http://piurilabs.di.unimi.it/Papers/cimsa_2005.pdf))

![](Images/FAB.jpeg)

## Data Collection

The data consists of 10,000+ images of single-cell microscopy acute lymphoblastic leukemia and normal lymphoblasts with a class imbalance of about 1:2 ALL to normal. Having enough images and computing resources without using all images, I decided to downsample the positive ALL class to manage class imbalance. Thus, training was completed with 4,000 images with a 55:45 class imbalance. The testing set remained imbalanced (3:10 ALL to normal) to more accurately evaluate the model in a real-world setting.

Images are 450x450 RGB images stored as .bmp files, a raster graphics bitmap which stores images as 2D matrices.

Data can be found [here](https://app.box.com/s/xeclwwd2xep9ntljtgyptmt4k5wone9n). It was sourced from the University of Arkansas for Medical Sciences (UAMS) study on ALL microscopy. 

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

I adopted an iterative approach to modeling based on the CRISP-DM process. A dummy classifier predicting the majority class had an accuracy of 31%. I created a Vanilla model with a single Conv2D layer and a single Dense layer which had an accuracy of 52% and recall of 33%, already better than the dummy. I then created successively larger and more complex architectures by adding additional Conv2D layers and blocks of layers separated by MaxPooling layers. 

I used Recall as my main metric because, like in most medical imaging, the outcome of a false negative is much worse for the patient than a false positive. It is better to over-predict ALL and have followup procedures to confirm or not, rather than predicting no ALL and being wrong. 

The most complex had 9 convolutions in 3 blocks of 3 layers, but this was not the most successful model as it appeared to overfit our training data. It became clear that deep, but narrow blocks were achieving higher metrics than wider blocks. The best model was a 2x2 architecture with 4 total convolutions. Dropout layers (of .25) improved the model's testing performance as it was not able to rely on a few specific nodes when predicting. 

Transfer learning using a selection of pre-trained models on ImageNet were used to some success. Recall was on average higher than custom models, however no transfer learning model was able to beat the best performing custom model. 

#### Final Network Architecture

![](Images/2x2C1Dsum.png)

#### Model Compilation Hyperparameters

I used binary crossentropy for the loss function as this is a binary classification problem, and RMSprop and Adam for optimization. The learning rate was set to 0.001 with a decay of 0.0001. 

#### Model Evaluation

A selection of my iterative modeling process with accuracy and loss metrics. I kept the class imbalance in the testing set to ensure my model can handle an imbalance more representative of real world applications.  

Model | Accuracy | Recall | Loss
---------- | ----------- | ---------- | ----------
1C1D                 | 0.5164 | 0.3272 | 1.2477
2C1D                 | 0.5744 | 0.4676 | 0.8072
2x2x1C2D             | 0.5259 | 0.3688 | 0.8977
2x2x1x1C1D Adam      | 0.5248 | 0.3981 | 0.8501
2C3D                 | 0.5438 | 0.4738 | 0.9085
**2x2C1D Dropout(0.25)** | 0.5767 | **0.9592** | 0.7373

#### Model Deployment

The best performing model was deployed using AWS Endpoints in order to load in not-yet-seen images from the testing set in order to generate predictions. The deployment is located in the 003_Modeling_AWS notebook under model training.

#### Misclassified Images

Here we see an image the model misclassified as Normal when it was actually ALL. The model is apparently responding to the lack of interior vacuoles and clefting as well as a dense chromatin to classify as normal. The irregular shape should have shown that it was ALL. 

<p float="left">
  <img src="Images/Misclass_ALL_as_Neg.png" width="300" />
</p>

Here we see an image the model misclassified as ALL when it was actually Normal. The model is likely responding to the irregular cell outline as well as the lighter, heterogeneous areas in the interior that suggest vacuoles or unpacked chromatin to classify it as ALL. This is definitely a challenging cell to correctly sort.

<p float="left">
  <img src="Images/Misclass_Norm_as_Pos.png" width="300" />
</p>

## Insights and Recommendations

The model achieved a recall of 96% on the imbalanced test set, allowing it to be a useful tool for identifying ALL in novel cases. As blood sample microscopy is already the default diagnostic test for ALL, this model could easily be used to verify a human physician or to flag cases that the model is not confident in for further review. As diagnosing ALL is difficult even for humans, having a robust, accurate verification model could improve the speed and rigor of diagnosis. Due to ALL being an acute leukemia, it is especially vital that it is consistently identified early, left untreated it can kill within a few weeks or months. 

## Next Steps

### Model Improvements

There are several potential avenues for improvement for this model. I attempted to use the Adam optimizer, which adds a sense of momentum and bias-correction to the gradient calculated by RMSprop, and Batch Normalization to improve model performance, though it did not help modeling thus far. I could also implement Early Stopping and Model Checkpoints to combat overfitting by allowing the model to stop training once a threshold of overfitting has been reached. I experimented with several levels of Dropout, settling on 25%, but further investigation could yield better results. 

### Product Improvements

Model interpretability is often as or more important than model accuracy, especially for medical diagnostic needs. It is very important in real-world applications that a doctor can see why the model has reached a certain decision. To that end, building an image segmentation model that identifies and marks important features, such as presence and number of vaculoes, non-spherical cells, or clefted edges within an image could greatly improve the model's usability. Additionally, deploying the model and allowing for live-integration of new imaging would keep the model up-to-date. 
