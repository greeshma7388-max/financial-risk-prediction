import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

def main():
    print(" Prediction of Financial Lifestyle Risk Score Using Supervised ML")

    
    print("Data Preprocessing")
    
    
    try:
        genz_df = pd.read_csv('genz_money_spends.csv')
        indian_df = pd.read_csv('data (1).csv')
        print(f"Loaded Gen Z dataset with {genz_df.shape[0]} records and {genz_df.shape[1]} variables.")
        print(f"Loaded Indian Finance dataset with {indian_df.shape[0]} records and {indian_df.shape[1]} variables.")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    
    df = genz_df.copy()
    
   
    df.dropna(inplace=True)
    
    
    df.drop_duplicates(inplace=True)
    
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if col != 'ID': # Skip ID column
            upper_limit = df[col].quantile(0.99)
            df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])
            
    
    
    print("Data Preprocessing Completed.\n")

    
    print("Feature Engineering (Risk Score Creation)")
    
    
    expense_cols = ['Rent (USD)', 'Groceries (USD)', 'Eating Out (USD)', 'Entertainment (USD)', 
                    'Subscription Services (USD)', 'Education (USD)', 'Online Shopping (USD)', 
                    'Travel (USD)', 'Fitness (USD)', 'Miscellaneous (USD)']
    
    df['Total Spend'] = df[expense_cols].sum(axis=1)
    
    
    discretionary_cols = ['Eating Out (USD)', 'Entertainment (USD)', 'Online Shopping (USD)', 
                          'Travel (USD)', 'Fitness (USD)']
    df['Discretionary Spend'] = df[discretionary_cols].sum(axis=1)
    
    
    df['Income (USD)'] = df['Income (USD)'].replace(0, 1)
    df['Total Spend'] = df['Total Spend'].replace(0, 1)
    
    
    df['Total Savings'] = df['Savings (USD)'] + df['Investments (USD)']
    df['Savings Rate'] = df['Total Savings'] / df['Income (USD)']
    
    
    df['Discretionary Spending Ratio'] = df['Discretionary Spend'] / df['Total Spend']
    
    
    df['Disposable Income Ratio'] = (df['Income (USD)'] - df['Total Spend']) / df['Income (USD)']
    
    
    df['Subscription Burden'] = df['Subscription Services (USD)'] / df['Income (USD)']
    
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[['Savings Rate', 'Discretionary Spending Ratio', 
                                               'Disposable Income Ratio', 'Subscription Burden']])
    
    
    df['Composite Risk Score'] = (
        -scaled_features[:, 0] * 1.5 +  # Savings Rate (High is good)
         scaled_features[:, 1] * 1.0 +  # Discretionary Ratio (High is bad)
        -scaled_features[:, 2] * 1.5 +  # Disposable Income Ratio (High is good)
         scaled_features[:, 3] * 0.5    # Subscription Burden (High is bad)
    )
    
    
    df['Risk Level'] = pd.qcut(df['Composite Risk Score'], q=3, labels=[0, 1, 2])
    
    print("Risk Level Class Distribution:")
    print(df['Risk Level'].value_counts().sort_index())
    print("Feature Engineering Completed.\n")

    
    print("Train/Test Split")
    
    
    X = df.drop(columns=['ID', 'Composite Risk Score', 'Risk Level', 'Total Spend', 
                         'Discretionary Spend', 'Total Savings', 'Savings Rate', 
                         'Discretionary Spending Ratio', 'Disposable Income Ratio', 
                         'Subscription Burden'])
    y = df['Risk Level']
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set size: {X_train_scaled.shape}")
    print(f"Test set size: {X_test_scaled.shape}")
    print("Data Scaling Completed.\n")

    
    print(" Model Training ")
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model

    print("Model Training Completed.\n")


    print(" Model Evaluation ")
    
    plt.figure(figsize=(15, 10))
    
    for i, (name, model) in enumerate(trained_models.items(), 1):
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        
        
        acc = accuracy_score(y_test, y_pred)
        print(f"\\n[{name}] Performance:")
        print(f"Accuracy: {acc:.4f}")
        
       
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        print(f"ROC-AUC Score (OVR): {auc:.4f}")
        
        
        plt.subplot(2, 2, i)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
        plt.title(f"{name} - Confusion Matrix")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig("genz_confusion_matrices.png", dpi=150)
    print("\nConfusion matrices saved as 'genz_confusion_matrices.png'.")
    
    
    print("\n STEP 6: Feature Importance ")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    
    rf_importances = trained_models['Random Forest'].feature_importances_
    rf_indices = np.argsort(rf_importances)[::-1]
    
    axes[0].bar(range(X.shape[1]), rf_importances[rf_indices], align='center')
    axes[0].set_xticks(range(X.shape[1]))
    axes[0].set_xticklabels(X.columns[rf_indices], rotation=90)
    axes[0].set_title('Random Forest - Feature Importances')
    
    
    xgb_importances = trained_models['XGBoost'].feature_importances_
    xgb_indices = np.argsort(xgb_importances)[::-1]
    
    axes[1].bar(range(X.shape[1]), xgb_importances[xgb_indices], align='center')
    axes[1].set_xticks(range(X.shape[1]))
    axes[1].set_xticklabels(X.columns[xgb_indices], rotation=90)
    axes[1].set_title('XGBoost - Feature Importances')
    
    plt.tight_layout()
    plt.savefig("genz_feature_importances.png", dpi=150)
    print("Feature importance plots saved as 'genz_feature_importances.png'.")
    
    
    print("\nTop 5 Financial Behaviours predicting Risk (Random Forest):")
    for i in range(5):
        print(f"{i+1}. {X.columns[rf_indices[i]]}")

    
    print("\n What-If Analysis ")
    
    # WHAT-IF ANALYSIS:
    # In real-world financial context, high-risk individuals often have deeply entrenched habits.
    # Therefore, they may not shift entirely to a 'Low Risk' or 'Medium Risk' class immediately 
    # just by altering one variable (like saving slightly more), because the model evaluates their 
    # holistic composite financial profile.
    # However, even if the absolute predicted class does not change, any reduction in the 
    # PROBABILITY of being High Risk is still a highly meaningful and actionable improvement.
    # This illustrates a real-world financial insight: mitigating risk is a gradual progression.
    
    high_risk_idx = y_test[y_test == 2].index[0]
    sample_person = df.loc[high_risk_idx].copy()
    
    print("\nOriginal Profile (High Risk Person):")
    print(f"Income: ${sample_person['Income (USD)']}")
    print(f"Savings: ${sample_person['Savings (USD)']}")
    print(f"Eating Out: ${sample_person['Eating Out (USD)']}")
    
    
    sample_features = sample_person[X.columns].values.reshape(1, -1)
    sample_scaled = scaler.transform(sample_features)
    
    rf_model = trained_models['Random Forest']
    orig_pred = rf_model.predict(sample_scaled)[0]
    print(f"Original Predicted Risk Level: {['Low', 'Medium', 'High'][orig_pred]}")
    
    
    print("\nSimulating Behavioral Improvement...")
    improved_person = sample_person.copy()
    
   
    improved_person['Savings (USD)'] = improved_person['Savings (USD)'] * 2.0
    improved_person['Eating Out (USD)'] = improved_person['Eating Out (USD)'] * 0.5
    
    print(f"New Savings: ${improved_person['Savings (USD)']}")
    print(f"New Eating Out: ${improved_person['Eating Out (USD)']}")
    
    improved_features = improved_person[X.columns].values.reshape(1, -1)
    improved_scaled = scaler.transform(improved_features)
    
    new_pred = rf_model.predict(improved_scaled)[0]
    print(f"New Predicted Risk Level: {['Low', 'Medium', 'High'][new_pred]}")
    print("\nPipeline Execution Finished Successfully!")

if __name__ == "__main__":
    main()