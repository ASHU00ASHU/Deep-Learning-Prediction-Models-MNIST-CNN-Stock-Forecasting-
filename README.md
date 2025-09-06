# Machine Learning Projects Collection

This repository contains three different machine learning projects demonstrating various approaches to classification and prediction tasks.

## 📁 Project Structure

```
MNIST-ML-Project/
├── notebooks/
│   ├── mnist_cnn.ipynb          # CNN approach for MNIST digit classification
│   ├── mnist_classification.ipynb  # Simple neural network for MNIST
│   └── stock_prediction.ipynb   # ELM for stock price prediction
├── data/                        # Data storage directory
├── models/                      # Saved model files
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## 🚀 Projects Overview

### 1. MNIST CNN Classification (`mnist_cnn.ipynb`)
- **Objective**: Handwritten digit recognition using Convolutional Neural Networks
- **Dataset**: MNIST (28x28 grayscale images of digits 0-9)
- **Model**: CNN with Conv2D, MaxPooling2D, and Dense layers
- **Performance**: Achieved 98.85% test accuracy
- **Key Features**:
  - Data normalization and reshaping
  - Multi-layer CNN architecture
  - Training visualization
  - Model evaluation and testing

### 2. MNIST Simple Neural Network (`mnist_classification.ipynb`)
- **Objective**: Alternative approach to MNIST using a simple feedforward neural network
- **Dataset**: MNIST dataset
- **Model**: Sequential model with Flatten and Dense layers
- **Performance**: Achieved high accuracy with fewer parameters
- **Key Features**:
  - Simple architecture comparison
  - Training history visualization
  - Individual prediction testing
  - Model summary and analysis

### 3. Stock Price Prediction (`stock_prediction.ipynb`)
- **Objective**: Stock price prediction using Extreme Learning Machine (ELM)
- **Dataset**: Apple (AAPL) stock data from Yahoo Finance
- **Model**: ELM with sigmoid activation
- **Key Features**:
  - Real-time data fetching using yfinance
  - Data preprocessing and normalization
  - ELM implementation for time series prediction
  - Visualization of actual vs predicted prices

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MNIST-ML-Project.git
cd MNIST-ML-Project
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

## 📊 Key Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Matplotlib**: Data visualization
- **Scikit-learn**: Machine learning utilities
- **yfinance**: Financial data API
- **HPELM**: Extreme Learning Machine implementation

## 🎯 Results Summary

| Project | Model Type | Accuracy/Performance | Key Insight |
|---------|------------|---------------------|-------------|
| MNIST CNN | Convolutional Neural Network | 98.85% | CNN excels at image recognition |
| MNIST Simple NN | Feedforward Neural Network | High accuracy | Simpler models can be effective |
| Stock Prediction | Extreme Learning Machine | Time series analysis | ELM suitable for financial forecasting |

## 🔧 Usage

### Running MNIST Projects
1. Open `mnist_cnn.ipynb` or `mnist_classification.ipynb`
2. Run all cells to train and evaluate the models
3. Observe the training progress and final accuracy

### Running Stock Prediction
1. Open `stock_prediction.ipynb`
2. The notebook will automatically download AAPL stock data
3. Run all cells to see the prediction results

## 📈 Model Performance

### MNIST CNN
- **Training Accuracy**: 99.41%
- **Validation Accuracy**: 98.85%
- **Architecture**: 3 Conv2D layers + 2 Dense layers
- **Parameters**: ~1.2M trainable parameters

### MNIST Simple NN
- **Architecture**: Flatten + 3 Dense layers
- **Parameters**: ~105K trainable parameters
- **Training**: 25 epochs with validation split

### Stock Prediction
- **Method**: Extreme Learning Machine
- **Input**: 5-day rolling window of returns
- **Output**: Next day return prediction
- **Data**: 3 years of AAPL stock data

## 🤝 Contributing

Feel free to contribute to this project by:
1. Forking the repository
2. Creating a feature branch
3. Making your changes
4. Submitting a pull request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

Created by [Your Name] - showcasing different machine learning approaches for classification and prediction tasks.

## 🔗 Related Resources

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Yahoo Finance API](https://pypi.org/project/yfinance/)
- [Extreme Learning Machine Paper](https://ieeexplore.ieee.org/document/1380068)

---

**Note**: This project is for educational purposes and demonstrates various machine learning techniques. For production use, additional validation and optimization would be recommended.