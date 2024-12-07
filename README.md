# COVID-19-Chest- Xray-Analyzer
## Introduction  
COVID-19, caused by the SARS-CoV-2 virus, emerged as a global pandemic, significantly affecting public health worldwide. The disease primarily impacts the respiratory system, with common symptoms including fever, dry cough, fatigue, and in severe cases, difficulty breathing. Early detection and diagnosis play a critical role in managing the disease and preventing its spread.  

This project focuses on detecting COVID-19 using chest X-ray images by leveraging machine learning techniques to classify X-rays as either **COVID-19 affected** or **Normal**.  

## Dataset Description  
The dataset used in this project is publicly available on Kaggle and contains chest X-ray images for both COVID-19 affected individuals and normal individuals.  
You can access the dataset here: [COVID-19 Chest X-Ray Dataset](https://www.kaggle.com/datasets/alifrahman/covid19-chest-xray-image-dataset/code).  

- Contains labeled chest X-ray images.
- Used for binary classification: **COVID-19** vs. **Normal**.  

## Preprocessing  
To prepare the dataset for training and ensure robust model performance, the following preprocessing steps were undertaken:  

- **Data Augmentation:** Augmentation techniques were applied to increase the dataset's diversity and improve generalization. The augmentation pipeline included:  
   - **Rotation:** Up to 20 degrees.  
   - **Width and Height Shifting:** Up to 20% of the total image dimensions.  
   - **Shearing:** Up to 20%.  
   - **Zooming:** Up to 20%.  
   - **Horizontal Flipping:** Randomly flipping the images horizontally.  
   - **Fill Mode:** Filling any gaps created during augmentation with the nearest pixel values.  

    The augmentation process was implemented using the `ImageDataGenerator` class from TensorFlow/Keras.
   ```python
   datagen = ImageDataGenerator(
       rotation_range=20,
       width_shift_range=0.2,
       height_shift_range=0.2,
       shear_range=0.2,
       zoom_range=0.2,
       horizontal_flip=True,
       fill_mode='nearest'
   )
<p align="center">
  <img src="https://github.com/mahita2104/COVID-Xray-Analyzer/blob/main/Images/Covid_image_augmentation.png" />
</p> 
<p align="center">
  <img src="https://github.com/mahita2104/COVID-Xray-Analyzer/blob/main/Images/Normal_image_augmentation.png" />
</p> 
   
 - **Feature Extraction with HOG (Histogram of Oriented Gradients):**
  For each augmented image, features were extracted as follows:
      - The augmented color images were converted to grayscale.
      - HOG (Histogram of Oriented Gradients) was applied to extract important spatial features.
      - Features were normalized using the L2-Hys block normalization method, and the cell size was set to (16, 16) pixels.
 - **Feature Vector Compilation:**
    The extracted HOG features from all augmented images were stored as feature vectors for downstream classification tasks.

## Methodology
The methodology followed in this project includes the following steps:

1. **Preprocessing:**
   - Augmented the dataset using various transformations (detailed above).
   - Extracted HOG features to capture critical spatial information and reduce the computational complexity of the model.
2. **Dataset Preparation:**
   - Collected the feature vectors into a dataset.
   - Shuffled the dataset to ensure unbiased training and testing.
3. **Classification with SVM:**
   - Implemented Support Vector Machines (SVM) with the RBF kernel for classification.
   - Tuned hyperparameters using Grid Search to optimize performance.
   - Trained and tested the classifier on the feature vectors to distinguish between COVID-19 and normal X-ray images.
   ### Hyperparameter Grid  

      | Hyperparameter | Values                      | Description                       |
      |----------------|-----------------------------|-----------------------------------|
      | `C`            | 0.1, 1, 10                  | Regularization parameter          |
      | `kernel`       | `linear`, `rbf`             | Kernel type                       |
      | `gamma`        | `scale`, `auto`             | Kernel coefficient                |

## Classification Report
The classification performance was evaluated using the following metrics and visualizations:
 - **ROC Curve**: Demonstrates the tradeoff between sensitivity and specificity for the classifier.
<p align="center">
  <img src="https://github.com/mahita2104/COVID-Xray-Analyzer/blob/main/Images/Roc_curve.png" />
</p> 

 - **Evaluation Metrics Table**: Displays the model's performance across metrics commonly used in classification model evaluation.
 <p align="center">
  <img src="https://github.com/mahita2104/COVID-Xray-Analyzer/blob/main/Images/Evaluation_metrics.png" />
</p> 

 - **3D Scatter Plot**: Visualizes the decision boundaries and classification results of the SVM classifier.
 <p align="center">
  <img src="https://github.com/mahita2104/COVID-Xray-Analyzer/blob/main/Images/3d_plot.png" />
</p> 


