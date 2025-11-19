# -----------------------------
# Data Cleaning, Encoding, and SMOTE
# -----------------------------
import pandas as pd
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import numpy as np
import random

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Load dataset
data = pd.read_csv("Dataset/employee_promotion.csv")  
#print(data.describe())
#print(data.info())
#print(data.isnull().sum())
print(data['is_promoted'].value_counts())

'''# Drop employee_id if exists
if "employee_id" in data.columns:
    data = data.drop(columns=["employee_id"])

# Identify numeric and categorical columns
num_cols = data.select_dtypes(include=["int64","float64"]).columns
cat_cols = data.select_dtypes(include=["object"]).columns

# Impute missing values
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

data[num_cols] = num_imputer.fit_transform(data[num_cols])
if len(cat_cols) > 0:
    data[cat_cols] = cat_imputer.fit_transform(data[cat_cols])


outlier_cols = ["no_of_trainings", "length_of_service"]

for col in outlier_cols:
    upper_limit = data[col].quantile(0.99)  # 99th percentile
    lower_limit = data[col].quantile(0.01)  # 1st percentile (optional)
    
    # Cap values above 99th percentile
    data[col] = np.where(data[col] > upper_limit, upper_limit, data[col])
    
    # If you also want to cap very low values (like extreme negatives):
    data[col] = np.where(data[col] < lower_limit, lower_limit, data[col])

print("✅ Outliers treated with percentile capping")

# One-hot encode categorical variables
data = pd.get_dummies(data, drop_first=True)

# -----------------------------
# Save cleaned dataset
# -----------------------------
data.to_csv("Dataset/employee_promotion_cleaned.csv", index=False)
print("✅ Cleaned dataset saved as 'employee_promotion_cleaned.csv'")

# Split features and target
X = data.drop("is_promoted", axis=1)
y = data["is_promoted"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# Apply SMOTE to balance classes in training set
smote = SMOTE(random_state=SEED)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("After SMOTE:", y_train.value_counts())
'''