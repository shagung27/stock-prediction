# Stock Price Analysis Dashboard

This project is a **stock visualization and prediction dashboard** built using **Plotly Dash**. It includes:

* LSTM model prediction for NSE-TATAGLOBAL
* Visualization of stock trends and volume for Facebook, Tesla, Apple, and Microsoft
* Interactive dashboard with dropdown selection and multiple charts
* A supporting Jupyter Notebook for model training steps


---

## 🛠️ Setup Instructions

### 1. Clone the Repository (if applicable)

```bash
git clone https://github.com/shagung27/stock-prediction
cd stock-prediction
```

### 2. Create a Virtual Environment (Windows)

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

* Command Prompt:

```bash
venv\Scripts\activate
```

* PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install Required Packages

```bash
pip install -r requirements.txt
```

If you face an issue with TensorFlow installation, ensure you're using Python 3.12 or lower (TensorFlow may not support 3.13+).

---

## 🚀 Running the Dashboard

```bash
python stock_app.py
```

Then open the browser at:

```
http://127.0.0.1:8050/
```

---

## 📓 LSTM Model Training

You can view and run the training process of the LSTM model using the Jupyter Notebook:

```bash
jupyter notebook lstm_training.ipynb
```

This will walk you through:

* Loading and preprocessing data
* Creating input sequences
* Training an LSTM model
* Saving the trained model

---

## 📊 Dashboard Features

### Tab 1: NSE-TATAGLOBAL LSTM Prediction

* **Actual vs Predicted Closing Prices** using LSTM

**🔽 Graphs representation:*

#### Real data:

![real](https://github.com/user-attachments/assets/155164d4-f863-4d13-9498-631b6acfe8ec)

#### Predicted data:
![lstm_predicted](https://github.com/user-attachments/assets/821a959b-b8cd-46b3-a2c8-dcff86ad2522)


### Tab 2: Stock High/Low and Volume Analysis

* Stocks supported: Facebook, Tesla, Apple, Microsoft
* Interactive dropdowns to select one or more companies
* **High vs Low prices** over time
* **Market volume** trends

---

## 🎥 Demo Video
[Demo](https://drive.google.com/file/d/1_hOw8p_7C3-jKj2gwQxPjPMGTLvBbiCI/view?usp=drive_link)



---

## 📌 Notes

* The application assumes `saved_model.h5` exists in the root folder.
* You must have CSV files in the correct format.

---

## 📧 Contact

For any issues or enhancements, feel free to contact me: shagungupta2702@gmail.com

---

Happy Forecasting! 📈📉
