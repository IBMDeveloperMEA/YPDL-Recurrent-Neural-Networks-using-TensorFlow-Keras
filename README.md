# Recurrent Neural Networks using TensorFlow Keras

## Workshop Resources
Login/Sign Up for IBM Cloud: https://ibm.biz/YourPathToDeepLearning

Hands-On Guide: https://ibm.biz/rnn-tensorflow-NNLab

Slides: https://ibmdevelopermea.github.io/YPDL-Recurrent-Neural-Networks-using-TensorFlow-Keras/#/

Workshop Replay:  https://www.crowdcast.io/e/ypdl-3

Survey - https://ibm.biz/YPDL-Survey

## Tutorial

Language Modeling is the task of assigning probabilities to sequences of words. Given a context of one or a sequence of words in the language that the language model was trained on, the model should provide the next most probable words or sequence of words that follows from the given sequence of words in the sentence. Language Modeling is one of the most important tasks in Natural Language Processing.

Recurrent neural networks (RNN) are a class of neural networks that is powerful for modeling sequence data such as time series or natural language. Basically an RNN uses a *for* loop and performs multiple iterations over the timesteps of a sequence while maintaining an internal state that encodes information about the timesteps it has seen so far. RNNs can easily be constructed using the Keras RNN API available within [TensorFlow](https://www.tensorflow.org) - an end-to-end open source machine learning platform that makes it easier to build and deploy machine learning models.

[IBM's Watson&reg; Studio](https://www.ibm.com/cloud/watson-studio) is a data science platform that provides all the tools necessary to develop a data-centric solution on the cloud. It makes use of Apache Spark clusters to provide the computational power needed to develop complex machine learning models. You can choose to create assets in Python, Scala, and R, and leverage open source frameworks (such as TensorFlow) that are already installed on Watson&reg; Studio.

In this tutorial, you will perform Language Modeling on the Penn Treebank dataset by creating a Recurrent Neural Network (RNN) using the Long Short-Term Memory unit and deploying it on IBM Watson&reg; Studio on IBM Cloud&reg; Pak for Data as a Service.

## Learning objectives

The goal of this tutorial is to import a Jupyter notebook written in Python into IBM Watson&reg; Studio on IBM Cloud&reg; Pak for Data as a Service and run through it. The notebook creates a Recurrent Neural Network model based on the Long Short-Term Memory unit to train and benchmark on the Penn Treebank dataset. By the end of this notebook, you should be able to understand how TensorFlow builds and executes a RNN model for Language Modeling. You'll learn how to:

* Run a Jupyter Notebook using IBM Watson&reg; Studio on IBM Cloud&reg; Pak for Data as a Service.
* Build a Recurrent Neural Network model using the Long Short-Term Memory unit for Language Modeling.
* Train the model and evaluate the model by performing validation and testing.

## Prerequisites

The following prerequisites are required to follow the tutorial:

* An [IBM Cloud Account](https://ibm.biz/YourPathToDeepLearning)
* [IBM Cloud Pak for Data](https://www.ibm.com/products/cloud-pak-for-data)

## Architecture Diagram
![Architecture Diagram](images/architecture-flow.png)

## Estimated time

This tutorial will take approximately four hours to complete. The bulk of this time will be spent training/evaluating the LSTM model. You can refer the [reducing the notebook execution time](#-reducing-the-notebook-execution-time) section for methods to reduce the time required for execution.

## Steps

1. [Set up IBM Cloud Pak for Data as a Service](#set-up-ibm-cloud-pak-for-data-as-a-service)
1. [Create a new project and import the notebook](#create-a-new-project-and-import-the-notebook)
1. [Read through the notebook](#read-through-the-notebook)
1. [Run the notebook](#run-the-notebook)

### Set up IBM Cloud Pak for Data as a Service

1. Open a browser, and log in to [IBM Cloud](https://ibm.biz/YourPathToDeepLearning) with your IBM Cloud credentials.

    ![Log into IBM Cloud](images/log-into-ibm-cloud.png)

1. Type `Watson Studio` in the search bar at the top. If you already have an instance of Watson Studio, it should be visible. If so, click it. If not, click **Watson Studio** under Catalog Results to create a new service instance.

    ![Select Watson Studio Service](images/select-watson-studio-service.png)

1. Select the type of plan to create if you are creating a new service instance. A Lite (free) plan should suffice for this tutorial). Click **Create**.

    ![Watson Studio Lite plan](images/watson-studio-lite-plan.png)

1. Click **Get Started** on the landing page for the service instance.

    ![Get Started - Watson Studio](images/get-started-watson-studio.png)

    This should take you to the landing page for IBM Cloud Pak for Data as a Service.

1.  Click your avatar in the upper-right corner, then click **Profile and settings** under your name.

    ![CPDaaS - profile and settings](images/cpdaas-profile-and-settings.png)

1. Switch to the Services tab. You should see the Watson Studio service instance listed under Your Cloud Pak for Data services.

    You can also associate other services such as Watson Knowledge Catalog and Watson Machine Learning with your IBM Cloud Pak for Data as a Service account. These are listed under Try our available services.

    In the example shown here, a Watson Knowledge Catalog service instance already exists in the IBM Cloud account, so it's automatically associated with the IBM Cloud Pak for Data as a Service account. To add any other service (Watson Machine Learning in this example), click **Add** within the tile for the service under Try our available services.

    ![CPDaaS - associated services](images/cpdaas-associated-services.png)

1. Select the type of plan to create (a Lite plan should suffice), and click **Create**.

    ![Machine Learning Lite Plan](images/machine-learning-lite-plan.png)

After the service instance is created, you are returned to the IBM Cloud Pak for Data as a Service instance. You should see that the service is now associated with your IBM Cloud Pak for Data as a Service account.

![CPDaas - all services associated](images/cpdaas-all-services-associated.png)

### Create a new project and import the notebook

1. Navigate to the hamburger menu (☰) on the left, and choose **View all projects**. After the screen loads, click **New +** or **New project +** to create a new project.

    ![CPDaas - new project](images/cpdaas-new-project.png)

1. Select **Create an empty project**.

    ![CPDaaS - empty project](images/cpdaas-empty-project.png)

1. Provide a name for the project. You must associate an IBM Cloud Object Storage instance with your project. If you already have an IBM Cloud Object Storage service instance in your IBM Cloud account, it should automatically be populated here. Otherwise, click **Add**.

    ![CPDaaS - project name](images/cpdaas-project-name.png)

1. Select the type of plan to create (a Lite plan should suffice for this tutorial), and click **Create**.

    ![COS Lite plan](images/cos-lite-plan.png)

1. Click **Refresh** on the project creation page.

    ![CPDaaS - refresh COS](images/cpdaas-refresh-cos.png)

1. Click **Create** after you see the IBM Cloud Object Storage instance you created displayed under Storage.

    ![CPDaaS - create project](images/cpdaas-create-project.png)

1. After the project is created, you can add the notebook to the project. Click **Add to project +**, and select **Notebook**.

    ![CPDaaS - add notebook to project](images/cpdaas-add-notebook-to-project.png)

1. Switch to the From URL tab. Provide the name of the notebook as `RecurrentNeuralNetworkUsingTensorFlow` and the Notebook URL as `https://raw.githubusercontent.com/IBM/dl-learning-path-assets/main/fundamentals-of-deeplearning/notebooks/RecurrentNeuralNetworkUsingTensorFlow.ipynb`.

1. Under the Select runtime drop-down menu, select **Default Python 3.7 S (4 vCPU 16 GB RAM)**. Click **Create**.

    ![CPDaaS - create notebook](images/cpdaas-create-notebook.png)

1. After the Jupyter Notebook is loaded and the kernel is ready, you can start executing the cells in the notebook.

    ![CPDaaS - notebook loaded](images/cpdaas-notebook-loaded.png)

> **Important**: *Make sure that you stop the kernel of your notebooks when you are done to conserve memory resources.*

![Stop kernel](images/JupyterStopKernel.png)

> **Note**: The Jupyter Notebook included in the project has been cleared of output. If you would like to see the notebook that has already been completed with output, refer to the [example notebook](https://github.com/IBM/dl-learning-path-assets/tree/main/fundamentals-of-deeplearning/examples/RecurrentNeuralNetworkUsingTensorFlow_with_output.ipynb).

### Read through the notebook

Spend some time looking through the sections of the notebook to get an overview. A notebook is composed of text (markdown or heading) cells and code cells. The markdown cells provide comments on what the code is designed to do.

You run cells individually by highlighting each cell, then either clicking **Run** at the top of the notebook or using the keyboard shortcut to run the cell (**Shift + Enter **, but this can vary based on the platform). While the cell is running, an asterisk (`[*]`) shows up to the left of the cell. When that cell has finished running, a sequential number appears (for example, `[17]`).

> **Note**: Some of the comments in the notebook are directions for you to modify specific sections of the code. Perform any changes as indicated before running the cell.

The notebook is divided into multiple sections.

* Section 1 gives an introduction to Language Modeling.
* Section 2 provides information about the Penn Treebank dataset which is being used to train and validate the model being built in this tutorial.
* Section 3 gives an introduction to Word Embeddings.
* Section 4 contains the code to build the LSTM model for Language Modeling.
* Section 5 contains the code to train, validate and test the model.

### Run the notebook

1. Run the code cells in the notebook starting with the ones in section 4. The first few cells bring in the required modules such as tensorflow, numpy, reader and the dataset.

> **Note**: The second code cell checks for the version of TensorFlow. The notebook only works with TensorFlow version 2.2.0-rc0, therefore, if an error is thrown here, you will need to ensure that you have installed TensorFlow version 2.2.0-rc0 in the first code cell. 

![TensorFlow Version error](images/tensorflow-version-error.png)

> **Note**: If you get the error in spite of installing TensorFlow version 2.2.0-rc0, your changes are not being picked up and you will need to restart the kernel by clicking on "Kernel"->"Restart and Clear Output". Wait until all the outputs disappear and then your changes should be picked up.

![Restart kernel](images/restart-kernel.png)

1. The training, validation and testing of the model does not happen until the last code cell. Due to the number of epochs and the sheer size of the dataset, running this cell can take about 3 hours.

![Train, test, validate](images/train-test-validate.png)

## Reducing the notebook execution time

There are a number of ways in which you can reduce the time required for executing the notebook, i.e., the time needed to train and validate the model. While all of these methds will affect the performance of the model, some will cause a drastic change in performance.

1. Fewer training epochs

The number of epochs set for training the model in the notebook is 15. You can reduce the number of epochs by changing the value of `max_epochs`.

![Change the number of epochs](images/change-number-of-epochs.png)

A lower number of epochs may not bring down the model perplexity, resulting in poor performance of the model. On the other hand, an extremely higher number of epochs can cause overfitting, which will also result in poor model performance.

1. Smaller dataset

Reducing the size of the dataset is another method to reduce the amount of time required for training. However, one should note that the this can negatively impact model performance. The model will be better trained when it is trained on a large amount of varied data.

1. Use GPUs to increase processing power

You can make use of the GPU (Graphics Processing Unit) environments available within Watson Studio in order to accelerate model training. With GPU environments, you can reduce the training time needed for compute-intensive machine learning models you create in a notebook. With more compute power, you can run more training iterations while fine-tuning your machine learning models.

**Note**: GPU environments are only available with paid plans and not with the Lite (Free) Watson Studio plan. See the [Watson Studio pricing plans](https://cloud.ibm.com/catalog/services/watson-studio)

1. Early stopping

You can also make use of EarlyStopping in order to improve training time. EarlyStopping stops training when a monitored metric has stopped improving. So, for example, you can specify that the training needs to stop if there is no improvement in perplexity for 3 consecutive epochs.

Refer the notebook used in the [Demand forecasting using deep learning](https://developer.ibm.com/patterns/demand-forecasting-for-cash-vending-machines-using-deep-learning/) code pattern for an example of EarlyStopping.

![Early Stopping Example](images/early-stopping-example.png)

## Summary

In this tutorial, you learned about Language Modeling. You learned how to run a Jupyter Notebook using Watson Studio on IBM Cloud Pak for Data as a Service and how to create a Recurrent Neural Network model using TensorFlow based on the Long Short-Term Memory unit. Finally, you used this model to train and benchmark on the Penn Treebank dataset.

## Workshop Resources
Login/Sign Up for IBM Cloud: https://ibm.biz/YourPathToDeepLearning

Hands-On Guide: https://ibm.biz/rnn-tensorflow-NNLab

Slides: https://ibmdevelopermea.github.io/YPDL-Recurrent-Neural-Networks-using-TensorFlow-Keras/#/

Workshop Replay:  https://www.crowdcast.io/e/ypdl-3

Survey - https://ibm.biz/YPDL-Survey
