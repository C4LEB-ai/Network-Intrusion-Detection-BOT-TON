# Deep Learning-Based Network Intrusion Detection System Using BOT and TON Datasets

## Project Overview

This project focuses on developing a robust Network Intrusion Detection System (NIDS) using deep learning techniques. The system leverages three models—Artificial Neural Network (ANN), Convolutional Neural Network (CNN), and Recurrent Neural Network (RNN)—to detect and classify potential intrusions in network traffic. The BOT and TON datasets were used, which are well-suited for identifying various types of network attacks. This project is particularly relevant to the communication industry, where the risk of system glitches or hacks is increasing.

## Project Objectives

- **Dataset Preparation:** Preprocess the BOT and TON datasets, which includes handling missing data, normalizing the features, and encoding categorical variables.
  
- **Model Development:** Build three deep learning models—ANN, CNN, and RNN—to detect and classify network intrusions. Each model is tailored to capture different aspects of the network traffic data.
  
- **Training and Optimization:** Train the models on the prepared datasets, optimize hyperparameters such as learning rate, number of epochs, and batch size, and apply techniques like dropout and early stopping to prevent overfitting.
  
- **Model Evaluation:** Evaluate the performance of each model using metrics such as accuracy, precision, recall, F1 score, and confusion matrix. Identify the best-performing model for deployment.
  
- **Application:** Apply the trained models to real-time network data in the communication industry to monitor and detect possible intrusions, reducing the risk of system hacks or glitches.

## Datasets

- **BOT Dataset:** This dataset contains labeled network traffic data that includes various types of network attacks.
- **TON Dataset:** This dataset provides extensive network traffic data, focusing on network anomalies and intrusions.

## Methodology

1. **Data Preprocessing:**
   - Clean and preprocess the datasets by addressing missing values and normalizing the features.
   - Split the data into training, validation, and test sets to ensure robust model performance.

2. **Model Development:**
   - **ANN:** A multi-layer perceptron model designed to capture the general patterns in network traffic data.
   - **CNN:** A convolutional neural network model applied to detect spatial patterns in the network traffic.
   - **RNN:** A recurrent neural network model employed to capture temporal patterns and dependencies in sequential data.

3. **Training and Optimization:**
   - Train each model using the processed data, optimizing for parameters like learning rate, epochs, and batch size.
   - Apply techniques such as early stopping and dropout to improve model performance and avoid overfitting.

4. **Evaluation:**
   - Assess the models' performance on the test set using evaluation metrics like accuracy, precision, recall, F1 score, and confusion matrix.
   - Compare the results to determine the most effective model for intrusion detection.

## Tools and Technologies

- **Programming Language:** Python
- **Deep Learning Frameworks:** TensorFlow, Keras
- **Data Manipulation:** NumPy, Pandas
- **Model Evaluation:** scikit-learn

## Results and Conclusion

The project successfully developed and evaluated three deep learning models for network intrusion detection. Among the models, ANN achieved the highest accuracy and robustness, making it suitable for deployment in the communication industry. This system provides an effective solution for mitigating the risks of network attacks, ensuring the security and integrity of communication networks.

## Future Work

- **Model Refinement:** Further optimize the models by exploring advanced architectures and hyperparameter tuning.
- **Real-Time Deployment:** Implement the best-performing model in a real-time environment to monitor network traffic and detect intrusions.
- **Expansion to Other Datasets:** Apply the developed models to other datasets to test their generalizability and effectiveness across different network environments.

## How to Use

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/c4leb-ai/Network-Intrusion-Detection-BOT-TON.git
