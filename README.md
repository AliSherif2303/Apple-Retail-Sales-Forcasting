# 🍏 Apple Retail Sales Forecasting

This project analyzes and forecasts **Apple’s retail sales performance** to uncover insights into **regional trends**, **store performance**, and **product demand patterns**.  

It leverages **Python**, **Pandas**, **Scikit-Learn**, and **Streamlit** for data analysis, predictive modeling, and interactive dashboard visualization.

---

## 📁 Project Structure

Apple-Retail-Sales-Forecasting/
│
├── app/ # Streamlit app for interactive visualization
├── data/ # Raw and processed datasets
│ ├── files.txt
│ └── ...
├── notebooks/ # Jupyter notebooks for EDA and model development
├── scripts/ # Python scripts for preprocessing and training
├── requirements.txt # Python dependencies
├── .gitignore # Ignored files and folders (e.g., .venv, data cache)
└── README.md # Project documentation


---

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/AliSherif2303/Apple-Retail-Sales-Forecasting.git
   cd Apple-Retail-Sales-Forecasting

Create and activate a virtual environment

python -m venv .venv
source .venv/bin/activate     # On macOS/Linux
.venv\Scripts\activate        # On Windows

pip install -r requirements.txt

Run the Streamlit app

streamlit run app/main.py

## 📊 Project Workflow

   Exploratory Data Analysis (EDA)
   
   Cleaned and explored Apple retail sales data.
   
   Identified seasonal and regional sales patterns.
   
   Visualized trends in product category performance.
   
   Feature Engineering & Preprocessing
   
   Encoded categorical features using Label and Target Encoding.
   
   Handled missing values and outliers.
   
   Aggregated data by region, store, and time period.
   
   Modeling & Forecasting
   
   Applied XGBoost Regressor for time-based forecasting.
   
   Tuned hyperparameters using cross-validation.
   
   Evaluated performance with RMSE and R² metrics.
   
   Interactive Dashboard
   
   Built with Streamlit to visualize predictions and KPIs.
   
   Allows users to explore forecasts by region, product, and time period.
   
## 📈 Example Insights

   📅 Seasonal peaks in Q4 due to holiday sales.
   
   🏬 Highest-performing regions identified by total revenue.
   
   📱 Product categories showing strongest year-over-year growth.

## 🧠 Tech Stack

   Languages: Python
   
   Libraries: Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn
   
   Visualization & App: Streamlit
   
   Version Control: Git, GitHub
   
## 🚀 Future Enhancements

   Integrate real-time sales data via API.
   
   Implement deep learning forecasting models (e.g., LSTM).
   
   Add customer segmentation and product recommendation modules.
   
## 👨‍💻 Team

   Ali Sherif Salaheldin
   Abdelrahman Mohamed Mahmoud

---

## 🧠 Datasets Overview

   These CSV files come from the [Apple Retail Sales Dataset](https://www.kaggle.com/datasets/amangarg08/apple-retail-sales-dataset).

   | File | Description | Shape |
   |------|--------------|--------|
   | `Products.csv` | Product info (ID, category, price, etc.) | (89, 5) |
   | `Sales.csv` | Sales transactions | (1,040,200, 5) |
   | `Warranty.csv` | Warranty claims and repair statuses | (30,000, 4) |
   | `Stores.csv` | Apple store locations | (75, 4) |
   | `Category.csv` | Product category mapping | (10, 2) |

   ---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/AliSherif2303/Apple-Retail-Sales-Forcasting.git
cd Apple-Retail-Sales-Forcasting
