# 🚀 MNIST Machine Learning Portfolio

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20App-red?style=for-the-badge&logo=streamlit&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github&logoColor=white)

**🔥 Advanced MNIST Digit Recognition with CNN, Neural Networks, and Interactive Demo**

[![Star](https://img.shields.io/github/stars/ASHU00ASHU/extream-learnig?style=social)](https://github.com/ASHU00ASHU/extream-learnig)
[![Fork](https://img.shields.io/github/forks/ASHU00ASHU/extream-learnig?style=social)](https://github.com/ASHU00ASHU/extream-learnig/fork)

</div>

---

## 📋 **Project Description**

This comprehensive machine learning portfolio demonstrates advanced neural network techniques for **handwritten digit recognition** using the famous MNIST dataset. The project showcases three different approaches to solving the same problem, providing valuable insights into the trade-offs between model complexity, accuracy, and computational efficiency.

### **🎯 Key Highlights:**
- **98.85% accuracy** achieved with Convolutional Neural Networks
- **Multiple model architectures** for comparison and learning
- **Interactive web application** for hands-on experimentation
- **Production-ready code** with comprehensive documentation
- **Real-world applications** in document digitization and computer vision

### **💡 Why This Project Matters:**
MNIST digit recognition is a fundamental problem in computer vision that serves as a benchmark for machine learning algorithms. This project demonstrates practical implementation of deep learning concepts that are directly applicable to real-world scenarios like:
- **Banking**: Check processing and digit recognition
- **Postal Services**: Automated mail sorting
- **Document Processing**: OCR and digitization systems
- **Education**: Learning platform for ML concepts

---

## 🌐 **Live Demo**

**🚀 [Interactive Streamlit App](https://d8guvpkpouoydzbv88cghw.streamlit.app/)**

Experience the project hands-on with our interactive web application featuring:
- **Real-time digit generation** and visualization
- **Model performance comparisons** with interactive charts
- **Project statistics** and technical details
- **Direct links** to notebooks and documentation

---

## 🎯 **What's Inside?**

This repository contains **3 machine learning projects** focused on **MNIST digit recognition** with different approaches:

| 🧠 **MNIST CNN** | 🔢 **MNIST Neural Net** | 📈 **Bonus: Stock Prediction** |
|:---:|:---:|:---:|
| **98.85% Accuracy** | **High Performance** | **ELM Example** |
| Convolutional Neural Network | Simple Feedforward Network | Extreme Learning Machine |
| Image Recognition | Digit Classification | Financial Forecasting |

---

## 🚀 **Quick Start**

```bash
# Clone the repository
git clone https://github.com/ASHU00ASHU/extream-learnig.git
cd extream-learnig

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

**Or try the live demo:** [🌐 Streamlit App](https://d8guvpkpouoydzbv88cghw.streamlit.app/)

---

## 📊 **Project Showcase**

### 🧠 **1. MNIST CNN - Convolutional Neural Network**
- **🎯 Goal**: Handwritten digit recognition using CNN
- **📈 Performance**: **98.85% test accuracy**
- **🏗️ Architecture**: 3 Conv2D layers + 2 Dense layers
- **💡 Key Features**:
  - Advanced image preprocessing
  - Multi-layer CNN architecture
  - Real-time training visualization
  - Comprehensive model evaluation

```python
# Model Architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

### 🔢 **2. MNIST Simple Neural Network**
- **🎯 Goal**: Alternative approach to MNIST digit classification
- **🏗️ Architecture**: Flatten + 3 Dense layers
- **💡 Key Features**:
  - Lightweight model (~105K parameters)
  - Fast training and inference
  - Comparison with CNN approach
  - Individual prediction testing

### 📈 **3. Bonus: Stock Price Prediction with ELM**
- **🎯 Goal**: Demonstrate Extreme Learning Machine on financial data
- **📊 Data**: Apple (AAPL) stock data from Yahoo Finance
- **🤖 Model**: ELM with sigmoid activation
- **💡 Key Features**:
  - Real-time data fetching
  - Advanced preprocessing
  - Time series prediction
  - Interactive visualizations

---

## 🛠️ **Tech Stack**

<div align="center">

| **Category** | **Technologies** |
|:---:|:---|
| **🤖 Deep Learning** | TensorFlow, Keras |
| **📊 Data Science** | NumPy, Pandas, Scikit-learn |
| **📈 Visualization** | Matplotlib |
| **💰 Finance** | yfinance API |
| **🧠 ML Algorithms** | CNN, ELM, Neural Networks |
| **💻 Development** | Jupyter Notebook, Python |
| **🌐 Web App** | Streamlit |

</div>

---

## 📈 **Performance Metrics**

<div align="center">

| **Project** | **Model Type** | **Accuracy** | **Parameters** | **Training Time** |
|:---:|:---:|:---:|:---:|:---:|
| 🧠 MNIST CNN | Convolutional Neural Network | **98.85%** | ~1.2M | ~5 minutes |
| 🔢 MNIST Simple | Feedforward Network | **97%+** | ~105K | ~2 minutes |
| 📈 Stock Prediction | Extreme Learning Machine | **High** | 10 neurons | ~30 seconds |

</div>

---

## 🎨 **Visual Results**

### MNIST CNN Training Progress
```
Epoch 1/5: accuracy: 0.8946 - val_accuracy: 0.9866
Epoch 2/5: accuracy: 0.9844 - val_accuracy: 0.9906
Epoch 3/5: accuracy: 0.9910 - val_accuracy: 0.9846
Epoch 4/5: accuracy: 0.9915 - val_accuracy: 0.9865
Epoch 5/5: accuracy: 0.9941 - val_accuracy: 0.9885
```

### Interactive Demo Features
- **Real-time digit generation** (0-9)
- **Model performance visualization**
- **Interactive charts and graphs**
- **Project statistics dashboard**

---

## 🚀 **Getting Started**

### **Option 1: Try the Live Demo**
**🌐 [Interactive Streamlit App](https://d8guvpkpouoydzbv88cghw.streamlit.app/)**

### **Option 2: Run Locally**
```bash
# Install all dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook

# Or run the Streamlit app locally
streamlit run streamlit_app.py
```

### **Option 3: Run Individual Projects**
```bash
# MNIST CNN
jupyter notebook notebooks/mnist_cnn.ipynb

# MNIST Simple NN
jupyter notebook notebooks/mnist_classification.ipynb

# Stock Prediction (Bonus)
jupyter notebook notebooks/stock_prediction.ipynb
```

---

## 📁 **Repository Structure**

```
extream-learnig/
├── 📁 notebooks/
│   ├── 🧠 mnist_cnn.ipynb          # CNN for MNIST digit recognition
│   ├── 🔢 mnist_classification.ipynb  # Simple NN for MNIST
│   └── 📈 stock_prediction.ipynb   # Bonus: ELM for stock forecasting
├── 📁 data/                        # Data storage
├── 📁 models/                      # Saved models
├── 🌐 streamlit_app.py            # Interactive web application
├── 🌐 index.html                  # Static portfolio website
├── 📄 requirements.txt             # Dependencies
├── 📄 PROJECT_SUMMARY.md          # Recruiter summary
└── 📄 README.md                   # This file
```

---

## 🎯 **Key Learning Outcomes**

### **MNIST Focus**
- **Computer Vision**: Image classification techniques
- **CNN Architecture**: Convolutional layers, pooling, dense layers
- **Model Comparison**: CNN vs Simple Neural Networks
- **Performance Optimization**: Achieving 98.85% accuracy

### **Technical Skills**
- **TensorFlow/Keras**: Deep learning framework mastery
- **Data Preprocessing**: Image normalization and reshaping
- **Model Evaluation**: Accuracy metrics and validation
- **Visualization**: Training progress and results
- **Web Development**: Streamlit app deployment

### **Real-World Applications**
- **Document Processing**: OCR and digitization
- **Banking Systems**: Check and form processing
- **Educational Tools**: Interactive learning platforms
- **Research**: Benchmarking ML algorithms

---

## 🤝 **Contributing**

We welcome contributions! Here's how you can help:

1. **🍴 Fork** the repository
2. **🌿 Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **💾 Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **📤 Push** to the branch (`git push origin feature/amazing-feature`)
5. **🔀 Open** a Pull Request

---

## 📝 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 **Author**

**ASHU00ASHU** - *Machine Learning Enthusiast*

- 🌐 **GitHub**: [@ASHU00ASHU](https://github.com/ASHU00ASHU)
- 🚀 **Live Demo**: [Streamlit App](https://d8guvpkpouoydzbv88cghw.streamlit.app/)

---

## 🙏 **Acknowledgments**

- [TensorFlow Team](https://www.tensorflow.org/) for the amazing deep learning framework
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) creators
- [Streamlit](https://streamlit.io/) for the interactive web app platform
- The open-source community for continuous inspiration

---

<div align="center">

### ⭐ **If you found this helpful, please give it a star!** ⭐

**Made with ❤️ by ASHU00ASHU**

[![GitHub](https://img.shields.io/badge/GitHub-ASHU00ASHU-black?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ASHU00ASHU)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-red?style=for-the-badge&logo=streamlit&logoColor=white)](https://d8guvpkpouoydzbv88cghw.streamlit.app/)

</div>
