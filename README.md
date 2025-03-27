# **Project 5- README**

## **Project Information**
**Project Title:** Recognition using Deep Networks

**Group Members:** Adit Shah & Jheel Kamdar

**Submission Date:** 27 March 2025

## **Project Overview**
This project focuses on developing and analyzing deep learning models for image recognition tasks using the MNIST and Fashion MNIST datasets. The project is divided into several tasks that demonstrate the ability of convolutional neural networks (CNNs) to learn from digit images, transfer learned features to new domains like Greek letter classification, and explore the impact of architectural changes on performance.

Core tasks include training a CNN from scratch on MNIST, analyzing learned filters, testing the model on new handwritten samples, and extending the model to classify Greek letters using transfer learning. Additional extensions explore expanded Greek alphabets, fixed feature extractors (like Gabor filters), and systematic experimentation with network depth, dropout, and convolutional filters using the Fashion MNIST dataset.


---

## **Video Demonstration**
N/A

---

## **Development Environment**
- **Operating System:** macOS
- **IDE Used:** Visual Studio Code
- **Python Version:** 3.11
- Key Libraries Used:
  - PyTorch
  - Torchvision
  - NumPy
  - Matplotlib
  - OpenCV (for Gabor filtering)
  - Scikit-learn (for evaluation metrics)
  - Seaborn (for plotting confusion matrices)
  - Pandas

---

## **Instructions to Run the Code**
### **Step 1: Clone or Download the Repository**  
```bash
git clone git@github.com:shah9678/MNIST_Torch.NN.git
cd MNIST_TORCH.NN
```
### **Step 2:Ensure your directory contains the following structure**
   ```
  MNIST_TORCH.NN/
├── Task1/
│   ├── MNIST.py
│   ├── classify_handwriting.py
│   ├── Test.py
│   ├── trained_mnist_model.pth
├── Task2/
│   ├── analyze_network.py
├── Task3/
│   ├── greek.py
├── Task4/
│   ├── personal_model.py
│   ├── experiment_results.csv
├── Extension gabor filter/
│   ├── gabor.py
│   ├── trained_mnist_gabor.pth

```
---
### **Step 3: Task 1 – Train & Evaluate MNIST Digit Classifier**  
```bash
cd Task1
python MNIST.py                  # Train model on MNIST
python Test.py                   # Test saved model
python classify_handwriting.py   # Test on personal handwriting samples
```
### **Step 4: Task 2 – Analyze Network Filters**  
```bash
cd ../Task2
python analyze_network.py        # Visualize learned filters and their effects
```
### **Step 5: Task 3 – Transfer Learning: Greek Letter Classification**  
```bash
cd ../Task3
python greek.py
```
### **Step 5: Task 4 – Evaluate Model Variations on Fashion MNIST**  
```bash
cd ../Task4
python personal_model.py         # Trains 27 variations and logs results
```
### **Step 6: Greek Letters (Alpha, Beta, Gamma) + Extensions**  
```bash
cd 'Extension Task3 + 2 extensions'
python extension.py              # Runs both: Task3 and 5-letter extension
```
### **Step 7: Greek Letters (Alpha, Beta, Gamma) + Extensions**  
```bash
cd ../Extension\ gabor\ filter
python gabor.py                  # Train model with frozen Gabor filters as conv1
```

---

**Time Travel Days Used:** 0  
We have completed and are submitting this project within the original deadline, without the use of any additional time travel days.

---

## **Acknowledgements
We referred to the official PyTorch documentation and OpenCV docs.
Inspiration for the Gabor filter implementation was drawn from OpenCV’s getGaborKernel.

---
