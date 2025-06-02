import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import ast
import json

# 1. Load and Preprocess Data
def load_and_preprocess(filename):
    df = pd.read_csv(filename)
    
    # Fix incorrectly split rows (Labels and System Artifacts)
    def fix_row(row):
        if 'Labels' in row['category'] or 'System Artifacts' in row['category']:
            parts = row['category'].split()
            row['category'] = ' '.join(parts[:-1])
            row['feature'] = parts[-1] + (' ' + row['feature'] if pd.notna(row['feature']) else '')
        return row
    
    df = df.apply(fix_row, axis=1)
    
    # Pivot to get one row per sample
    df_pivot = df.pivot_table(index='sample_id', 
                             columns=['category', 'feature'], 
                             values='value',
                             aggfunc='first')
    
    # Flatten multi-index columns
    df_pivot.columns = ['_'.join(col).strip() for col in df_pivot.columns.values]
    df_pivot = df_pivot.reset_index()
    
    # Convert string representations to Python objects
    def parse_value(x):
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except (ValueError, SyntaxError):
                return x
        return x
    
    for col in df_pivot.columns:
        df_pivot[col] = df_pivot[col].apply(parse_value)
    
    return df_pivot

# 2. Feature Engineering
def engineer_features(df):
    # Convert boolean strings to actual booleans
    bool_cols = [col for col in df.columns if df[col].astype(str).str.upper().isin(['TRUE', 'FALSE']).any()]
    for col in bool_cols:
        df[col] = df[col].astype(str).str.upper() == 'TRUE'
    
    # Create features for list lengths
    list_cols = [col for col in df.columns if isinstance(df[col].iloc[0], list)]
    for col in list_cols:
        df[f'{col}_count'] = df[col].apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    # Extract target variables
    targets = {
        'is_NotPetya': df['Labels_Is Notpetya'],
        'damage_severity': df['Labels_Damage Severity'],
        'recovery_possibility': df['Labels_Recovery Possibility']
    }
    
    # Drop label columns from features
    df = df.drop(columns=[col for col in df.columns if 'Labels_' in col])
    
    return df, targets

# 3. Load Data
df = load_and_preprocess('notpetya_dataset_corrected.csv')
features, targets = engineer_features(df)

# 4. Prepare Data for Modeling
# For binary classification (is_NotPetya)
X = features.select_dtypes(include=['bool', 'number'])
y = targets['is_NotPetya']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
print("Classification Report for is_NotPetya:")
print(classification_report(y_test, y_pred))

# 7. Feature Importance
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Important Features:")
print(importances.head(10))

# Save model
import joblib
joblib.dump(model, 'notpetya_classifier.joblib')
