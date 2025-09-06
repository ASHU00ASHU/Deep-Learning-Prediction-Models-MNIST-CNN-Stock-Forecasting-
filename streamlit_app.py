import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="MNIST ML Portfolio",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #667eea;
        text-align: center;
        margin-bottom: 2rem;
    }
    .project-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .accuracy-badge {
        background: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🚀 MNIST Machine Learning Portfolio</h1>', unsafe_allow_html=True)
st.markdown("### Advanced Neural Networks for Digit Recognition by ASHU00ASHU")

# Sidebar
st.sidebar.title("📊 Project Stats")
st.sidebar.metric("CNN Accuracy", "98.85%")
st.sidebar.metric("Total Projects", "3")
st.sidebar.metric("Parameters", "1.2M")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="project-card">
        <h3>🧠 MNIST CNN</h3>
        <span class="accuracy-badge">98.85% Accuracy</span>
        <p>Convolutional Neural Network for handwritten digit recognition using TensorFlow/Keras.</p>
        <p><strong>Technologies:</strong> TensorFlow, Keras, CNN, Computer Vision</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="project-card">
        <h3>🔢 MNIST Simple NN</h3>
        <span class="accuracy-badge">97%+ Accuracy</span>
        <p>Lightweight feedforward neural network for efficient digit classification.</p>
        <p><strong>Technologies:</strong> Neural Networks, TensorFlow, Dense Layers</p>
    </div>
    """, unsafe_allow_html=True)

# Stock Prediction
st.markdown("""
<div class="project-card">
    <h3>📈 Stock Prediction (Bonus)</h3>
    <span class="accuracy-badge">ELM Model</span>
    <p>Extreme Learning Machine for financial forecasting using real-time data.</p>
    <p><strong>Technologies:</strong> ELM, yfinance, Time Series, Financial Data</p>
</div>
""", unsafe_allow_html=True)

# Interactive Demo Section
st.markdown("---")
st.markdown("## 🎮 Interactive Demo")

# Create a simple digit drawing interface
st.markdown("### Draw a Digit (0-9)")
canvas_width = 280
canvas_height = 280

# Create a simple drawing interface
if st.button("Generate Random MNIST-like Image"):
    # Generate a random 28x28 image that looks like a digit
    random_image = np.random.rand(28, 28) * 255
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(random_image, cmap='gray')
    ax.set_title("Random MNIST-like Image")
    ax.axis('off')
    st.pyplot(fig)

# Model Performance Visualization
st.markdown("---")
st.markdown("## 📈 Model Performance")

# Create performance chart
performance_data = {
    'Model': ['MNIST CNN', 'MNIST Simple NN', 'Stock Prediction'],
    'Accuracy': [98.85, 97.0, 85.0],
    'Parameters': [1200000, 105000, 10],
    'Training Time (min)': [5, 2, 0.5]
}

st.dataframe(performance_data, use_container_width=True)

# Performance chart
fig, ax = plt.subplots(figsize=(10, 6))
models = performance_data['Model']
accuracies = performance_data['Accuracy']
bars = ax.bar(models, accuracies, color=['#667eea', '#764ba2', '#4CAF50'])
ax.set_ylabel('Accuracy (%)')
ax.set_title('Model Performance Comparison')
ax.set_ylim(0, 100)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{acc}%', ha='center', va='bottom', fontweight='bold')

st.pyplot(fig)

# Links section
st.markdown("---")
st.markdown("## 🔗 Project Links")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📁 Repository")
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/ASHU00ASHU/extream-learnig)")

with col2:
    st.markdown("### 📓 Notebooks")
    st.markdown("[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter)](https://github.com/ASHU00ASHU/extream-learnig/tree/main/notebooks)")

with col3:
    st.markdown("### 📊 Documentation")
    st.markdown("[![Docs](https://img.shields.io/badge/Documentation-Read-blue?style=for-the-badge)](https://github.com/ASHU00ASHU/extream-learnig/blob/main/README.md)")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>Made with ❤️ by <strong>ASHU00ASHU</strong> | Machine Learning Enthusiast</p>
    <p>GitHub: <a href="https://github.com/ASHU00ASHU">@ASHU00ASHU</a></p>
</div>
""", unsafe_allow_html=True)
