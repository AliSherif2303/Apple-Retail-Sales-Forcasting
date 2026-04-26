import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# -----------------
# 1. Intro Markdown
# -----------------
cell_md_1 = """# 📊 The Ultimate Demand Classification Showdown (XGBoost vs CatBoost)

**Goal:** Shift from predicting exact sales figures (Regression) to classifying the **Risk/Surge Category** (Classification). 

Instead of relying on a single algorithm, this notebook pits the two most powerful tabular algorithms in the world against each other: **XGBoost** and **CatBoost**. We have applied deep hyperparameter boosting (`max_depth=10`, `n_estimators=300`) to guarantee high-accuracy results."""

# -----------------
# 2. Imports & Load
# -----------------
cell_code_1 = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')

print("Loading Realistic Enriched Dataset...")
df = pd.read_csv('../data/processed/cleaned_apple_sales_enriched_realistic.csv')
print(f"Dataset shape: {df.shape}")"""

# -----------------
# 3. Target Binning
# -----------------
cell_md_2 = """### Phase 1: Data Binning (Per-Product Mathematical Thresholds)
We calculate the 25th and 75th percentiles **strictly per product** to guarantee fairness across cheap and expensive items."""

cell_code_2 = """# Calculate the thresholds PER PRODUCT mathematically
df['p25'] = df.groupby('product_name')['quantity_realistic'].transform(lambda x: x.quantile(0.25))
df['p75'] = df.groupby('product_name')['quantity_realistic'].transform(lambda x: x.quantile(0.75))

# Vectorized class assignment
df['demand_class'] = 1 # Normal
df.loc[df['quantity_realistic'] <= df['p25'], 'demand_class'] = 0 # Low Demand
df.loc[df['quantity_realistic'] >= df['p75'], 'demand_class'] = 2 # High Demand

print("Target Classes Successfully Built.")"""

# -----------------
# 4. Feature Eng
# -----------------
cell_md_3 = """### Phase 2: Feature Engineering & Academic Splitting
We prepare the columns and use an Academic 80/20 Random Split to allow the models to learn global economic patterns safely."""

cell_code_3 = """features = [
    'price_realistic', 'promo_flag', 'product_age_days', 
    'inflation_rate', 'gdp_per_capita', 'internet_usage_pct',
    'month', 'season_factor', 'trend_factor',
    'category_name', 'product_name', 'country_norm_mapped', 'store_name'
]

ml_df = df.copy()

# Native Categorical Encoding for both XGBoost and CatBoost
cat_cols = ['category_name', 'product_name', 'country_norm_mapped', 'store_name']
for col in cat_cols:
    ml_df[col] = ml_df[col].astype('category')
    
# Academic 80/20 Random Split
X = ml_df[features]
y = ml_df['demand_class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} rows.")
print(f"Testing on {len(X_test)} rows.")"""

# -----------------
# 5. XGBoost
# -----------------
cell_md_4 = """### Phase 3: XGBoost Training
We initialize XGBoost with extreme `max_depth=10` and Native Categorical Support."""

cell_code_4 = """xgb_model = XGBClassifier(
    n_estimators=300, 
    max_depth=10, 
    learning_rate=0.05, 
    random_state=42,
    tree_method='hist',
    enable_categorical=True 
)

print("Training XGBoost Classifier... (This will take ~2 minutes)")
xgb_model.fit(X_train, y_train)
print("✅ XGBoost Training Complete!")"""

# -----------------
# 6. CatBoost
# -----------------
cell_md_5 = """### Phase 4: CatBoost Training
We initialize CatBoost with identical deep parameters to keep the fight fair."""

cell_code_5 = """cat_model = CatBoostClassifier(
    iterations=300,
    depth=10,
    learning_rate=0.05,
    cat_features=cat_cols,
    random_state=42,
    verbose=50 # Print progress every 50 trees
)

print("\\nTraining CatBoost Classifier... (This may take ~3-4 minutes depending on CPU)")
cat_model.fit(X_train, y_train)
print("✅ CatBoost Training Complete!")"""

# -----------------
# 7. Evaluation
# -----------------
cell_md_6 = """### Phase 5: The Grand Showdown (Evaluation)
We predict blindly on the 20% test set using both models and plot their Confusion Matrices side-by-side to declare a winner!"""

cell_code_6 = """# Predictions
y_pred_xgb = xgb_model.predict(X_test)
y_pred_cat = cat_model.predict(X_test)

# ROC AUC
xgb_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test), multi_class='ovr')
cat_auc = roc_auc_score(y_test, cat_model.predict_proba(X_test), multi_class='ovr')

print("\\n" + "="*60)
print(f"🏆 OVERALL ACCURACY RACE (ROC-AUC SCORES) 🏆")
print("="*60)
print(f"XGBoost ROC-AUC : {xgb_auc:.4f}")
print(f"CatBoost ROC-AUC: {cat_auc:.4f}")
print("="*60 + "\\n")

# Confusion Matrices
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
cm_cat = confusion_matrix(y_test, y_pred_cat)

plt.figure(figsize=(16, 6))

# XGBoost Plot
plt.subplot(1, 2, 1)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low (0)', 'Normal (1)', 'High (2)'], 
            yticklabels=['Low (0)', 'Normal (1)', 'High (2)'])
plt.title(f'XGBoost Confusion Matrix', fontweight='bold')
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')

# CatBoost Plot
plt.subplot(1, 2, 2)
sns.heatmap(cm_cat, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=['Low (0)', 'Normal (1)', 'High (2)'], 
            yticklabels=['Low (0)', 'Normal (1)', 'High (2)'])
plt.title(f'CatBoost Confusion Matrix', fontweight='bold')
plt.xlabel('Predicted Class')
plt.ylabel('')
plt.tight_layout()
plt.show()"""

# Assemble
nb['cells'] = [
    nbf.v4.new_markdown_cell(cell_md_1),
    nbf.v4.new_code_cell(cell_code_1),
    nbf.v4.new_markdown_cell(cell_md_2),
    nbf.v4.new_code_cell(cell_code_2),
    nbf.v4.new_markdown_cell(cell_md_3),
    nbf.v4.new_code_cell(cell_code_3),
    nbf.v4.new_markdown_cell(cell_md_4),
    nbf.v4.new_code_cell(cell_code_4),
    nbf.v4.new_markdown_cell(cell_md_5),
    nbf.v4.new_code_cell(cell_code_5),
    nbf.v4.new_markdown_cell(cell_md_6),
    nbf.v4.new_code_cell(cell_code_6),
]

os.makedirs('notebooks', exist_ok=True)
with open('notebooks/Demand_Classification_XGBoost.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook generated successfully at notebooks/Demand_Classification_XGBoost.ipynb")
