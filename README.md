# 🎬 IMDB Movie Review Sentiment Analysis using SimpleRNN

Welcome to my deep learning project where I implement a **SimpleRNN-based sentiment classifier** using the IMDB dataset and deploy it via `Streamlit`. This project demonstrates end-to-end pipeline — from model training to web app deployment.

---

## 📊 **Dataset Details**

The project uses the IMDB Movie Reviews Dataset, which contains 50,000 highly polarized movie reviews labeled as positive or negative. This dataset is widely used for binary sentiment classification tasks.
- Source: Provided by TensorFlow/Keras datasets
- Data split: 25,000 reviews for training and 25,000 for testing
- Preprocessing: Reviews are tokenized into word indices with a vocabulary size limited to 10,000 most frequent words. Sequences are padded/truncated to a fixed length of 500 words for consistent input shape.

---

## Tools and Libraries
[![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?style=for-the-badge\&logo=tensorflow\&logoColor=white)](https://www.tensorflow.org/) 
[![Keras](https://img.shields.io/badge/-Keras-D00000?style=for-the-badge\&logo=keras\&logoColor=white)](https://keras.io/) 
[![Scikit-Learn](https://img.shields.io/badge/-Scikit--Learn-F7931E?style=for-the-badge\&logo=scikit-learn\&logoColor=white)](https://scikit-learn.org/) 
[![Scikeras](https://img.shields.io/badge/-Scikeras-333333?style=for-the-badge)](https://scikeras.readthedocs.io/en/stable/) [![Pandas](https://img.shields.io/badge/-Pandas-150458?style=for-the-badge\&logo=pandas\&logoColor=white)](https://pandas.pydata.org/) 
[![NumPy](https://img.shields.io/badge/-NumPy-013243?style=for-the-badge\&logo=numpy\&logoColor=white)](https://numpy.org/)
[![Pickle](https://img.shields.io/badge/-Pickle-000000?style=for-the-badge)](https://docs.python.org/3/library/pickle.html)
[![TensorBoard](https://img.shields.io/badge/-TensorBoard-FF6F00?style=for-the-badge\&logo=tensorflow\&logoColor=white)](https://www.tensorflow.org/tensorboard) 
[![Matplotlib](https://img.shields.io/badge/-Matplotlib-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](https://matplotlib.org/)                                                                                                    [![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)](https://streamlit.io/) 
[![Google Colab](https://img.shields.io/badge/-Google%20Colab-F9AB00?style=for-the-badge\&logo=googlecolab\&logoColor=white)](https://colab.research.google.com/)
[![Git](https://img.shields.io/badge/-Git-F05032?style=for-the-badge\&logo=git\&logoColor=white)](https://git-scm.com/) 
[![GitHub](https://img.shields.io/badge/-GitHub-181717?style=for-the-badge\&logo=github\&logoColor=white)](https://github.com/)

## 📽️ Demo Preview
![Demo](streamlit_app/streamlit-demo-gif.gif)

## 🚀 **Project Highlights**

- [x] **Data Loading:** Loaded the IMDB dataset with a vocabulary size limited to ***10,000*** most frequent words.  
- [x] **Preprocessing:** Converted raw text reviews into sequences of integer word indices using the IMDB `word_index`, then *padded* or *truncated* all sequences to a fixed length of **500 tokens** to standardize input size.  
- [x] **Model Architecture:** Built a **SimpleRNN** model with:  
  - An **Embedding layer** (dimension = `128`)  
  - A **SimpleRNN layer** with `128` units and **ReLU** activation  
  - A **Dense output layer** with **sigmoid** activation for binary classification  
- [x] **Training:** Compiled the model with the **Adam optimizer** and **binary cross-entropy loss**, trained for **10 epochs** with batch size **32**, using **20% validation split**.  
- [x] **Model Saving:** Saved the trained model as an `.h5` file for easy reuse.  
- [x] **Prediction:** Created a separate notebook to load the saved model, preprocess new reviews, and generate sentiment predictions.  
- [x] **Deployment:** Developed an interactive **Streamlit** app to input movie reviews, preprocess text, run sentiment predictions in real time, and display results with a user-friendly interface.

