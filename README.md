# Martian CNN Image Classifier

## Project Overview
This deep learning project focuses on the automated classification of geological landmarks on the Martian surface. Using the Mars Reconnaissance Orbiter dataset, this custom Convolutional Neural Network identifies eight distinct categories of Martian terrain with high precision. The ultimate goal is to establish a reliable baseline for automated terrain mapping to assist in autonomous planetary exploration.

## Performance and Results
The model achieved a 96.1% weighted average accuracy. By implementing regularization techniques like a 0.5 Dropout and Batch Normalization, the architecture demonstrates excellent generalization with a very minimal gap between training and validation performance.

* **Weighted Accuracy:** 96.1%
* **Macro Average F1 Score:** 0.86
* **Training Fit:** Robust convergence showing 99.7% Training Accuracy and 95.8% Validation Accuracy.

## Dataset Information
This project utilizes the Mars orbital image HiRISE labeled data set version 3. 

**Dataset Authors:** Kiri L Wagstaff, Steven Lu, Gary Doran, Lukas Mandrake
**Original Attribution:** If you use this data set in your own work, please cite this DOI 10.5281 zenodo 2538136.

https://data.nasa.gov/dataset/mars-orbital-image-hirise-labeled-data-set-version-3/resource/c93bf426-1eae-4d3b-8afd-548add5e24ce

The dataset contains a total of 73031 landmarks. Initially, 10433 landmarks were detected and extracted from 180 HiRISE browse images. For each original landmark, a square bounding box was cropped to include the full extent of the feature plus a 30 pixel margin on all sides. Each cropped landmark was resized to 227 by 227 pixels and then augmented to generate 62598 additional samples using the following methods:
1. 90 degrees clockwise rotation
2. 180 degrees clockwise rotation
3. 270 degrees clockwise rotation
4. Horizontal flip
5. Vertical flip
6. Random brightness adjustment

The network classifies these grayscale images into eight geological categories:
1. Other Background
2. Crater
3. Dark Dune
4. Slope Streak
5. Bright Dune
6. Impact Ejecta
7. Swiss Cheese
8. Spider

## Technical Insights and Core Concepts Learned
Throughout the development of this Martian surface classifier, several key deep learning methodologies and visualization strategies were explored and implemented.

### Neural Network Layers and Their Functions
Building a model to process low contrast grayscale imagery required a highly specific architectural approach:
* **Conv2D 3 by 3 Kernels:** Used as the foundational feature extractors. Because Martian geological features lack distinct color boundaries, these convolutional layers were critical for learning complex textural and spatial hierarchies.
* **Batch Normalization:** Applied immediately after convolutions to stabilize the learning process. This mitigated internal covariate shift, ensuring the network did not get derailed by large weight updates during early epochs.
* **ReLU Activation:** Chosen to map nonlinearities while actively preventing the vanishing gradient problem in the deeper blocks of the network.
* **MaxPooling2D 2 by 2:** Used for spatial dimension reduction. This progressively compressed the feature maps, extracting only the most prominent abstract features while significantly reducing computational overhead.
* **Dropout and L2 Regularization:** Crucial for preventing model overspecialization. By randomly dropping 50 percent of the neurons during training and penalizing overly large weights, the network was forced to learn robust and generalized patterns.
* **Dense and Softmax:** The final classification head. The Softmax function was mathematically essential to squash the raw logit outputs into a normalized probability distribution across the 8 classes.

### Dynamic Optimization Strategies
* **Adam Optimizer:** Selected for its adaptive learning rate capabilities, which handled the sparse gradients of our image data exceptionally well.
* **Callbacks:** The ReduceLROnPlateau callback dynamically reduced the learning rate when validation loss stagnated, unlocking further convergence. The ModelCheckpoint callback ensured we only saved the optimal epoch weights.

### Best Suited Evaluation Plots
* **Convergence Curves:** Comparing the training and validation trajectories allowed us to visually confirm that the model was converging optimally.
* **Confusion Matrix Heatmap:** Essential for diagnosing multi class decision boundaries, highlighting systematic overlaps between texturally similar classes.
* **Category Specific Accuracy Bar Chart:** By isolating the true positive rate per class, this plot proved the network relied on true topographical geometry.
* **20 Sample Qualitative Grid:** Plotting unseen test images directly alongside their predicted and true labels provided immediate visual proof of the model capabilities.

## Technology Stack
* **Core Frameworks:** Python, TensorFlow, Keras
* **Data and Math:** NumPy, Pandas, Scikit learn
* **Visualization:** Matplotlib, Seaborn
* **Preprocessing Pipeline:** 1 over 255 Pixel Rescaling and automated data batching


## Local Setup and Installation Guide

Follow these exact steps to run this deep learning project on your local machine.

### Step 1: Clone the Repository
Open your terminal and clone the project files to your local machine.

    git clone https://github.com/nainesh-builds/martian-cnn-image-classifier.git
    cd martian-cnn-image-classifier

### Step 2: Set Up a Virtual Environment
It is highly recommended to use a virtual environment to keep the dependencies isolated.

    python -m venv venv
    source venv/bin/activate
    # Note: On Windows use
    venv\Scripts\activate

### Step 3: Create the Requirements File
Create a new text file named `requirements.txt` in your project root folder and paste the following list of dependencies into it:

    tensorflow
    pandas
    numpy
    scikit-learn
    matplotlib
    seaborn
    jupyter

### Step 4: Install Required Dependencies
Ensure you have Python 3.8 or higher installed. Install all the necessary libraries using the requirements file you just created.

    pip install -r requirements.txt

### Step 5: Create Project Directories
Create the required folder structure to store your data and outputs.

    mkdir data models notebooks plots report

### Step 6: Execute the Code
Launch Jupyter Notebook to interact with the project. Run the notebook sequentially from top to bottom.

    jupyter notebook notebooks/main.ipynb

