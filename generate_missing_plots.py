import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.table import table
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

import warnings
warnings.filterwarnings('ignore')

def main():
    print("Loading datasets")
    
    genz_df = pd.read_csv('genz_money_spends.csv')
    genz_df.dropna(inplace=True)
    genz_df.drop_duplicates(inplace=True)
    
    
    num_cols = genz_df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if col != 'ID':
            upper_limit = genz_df[col].quantile(0.99)
            genz_df[col] = np.where(genz_df[col] > upper_limit, upper_limit, genz_df[col])
            
    expense_cols = ['Rent (USD)', 'Groceries (USD)', 'Eating Out (USD)', 'Entertainment (USD)', 
                    'Subscription Services (USD)', 'Education (USD)', 'Online Shopping (USD)', 
                    'Travel (USD)', 'Fitness (USD)', 'Miscellaneous (USD)']
    genz_df['Total Spend'] = genz_df[expense_cols].sum(axis=1)
    
    discretionary_cols = ['Eating Out (USD)', 'Entertainment (USD)', 'Online Shopping (USD)', 
                          'Travel (USD)', 'Fitness (USD)']
    genz_df['Discretionary Spend'] = genz_df[discretionary_cols].sum(axis=1)
    
    genz_df['Income (USD)'] = genz_df['Income (USD)'].replace(0, 1)
    genz_df['Total Spend'] = genz_df['Total Spend'].replace(0, 1)
    genz_df['Total Savings'] = genz_df['Savings (USD)'] + genz_df['Investments (USD)']
    genz_df['Savings Rate'] = genz_df['Total Savings'] / genz_df['Income (USD)']
    genz_df['Discretionary Spending Ratio'] = genz_df['Discretionary Spend'] / genz_df['Total Spend']
    genz_df['Disposable Income Ratio'] = (genz_df['Income (USD)'] - genz_df['Total Spend']) / genz_df['Income (USD)']
    genz_df['Subscription Burden'] = genz_df['Subscription Services (USD)'] / genz_df['Income (USD)']
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(genz_df[['Savings Rate', 'Discretionary Spending Ratio', 
                                               'Disposable Income Ratio', 'Subscription Burden']])
    genz_df['Composite Risk Score'] = (
        -scaled_features[:, 0] * 1.5 +  
         scaled_features[:, 1] * 1.0 +  
        -scaled_features[:, 2] * 1.5 +  
         scaled_features[:, 3] * 0.5    
    )
    genz_df['Risk Level'] = pd.qcut(genz_df['Composite Risk Score'], q=3, labels=[0, 1, 2])
    
    X = genz_df.drop(columns=['ID', 'Composite Risk Score', 'Risk Level', 'Total Spend', 
                         'Discretionary Spend', 'Total Savings', 'Savings Rate', 
                         'Discretionary Spending Ratio', 'Disposable Income Ratio', 
                         'Subscription Burden'])
    y = genz_df['Risk Level']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42)
    }
    
    base_acc = {}
    base_auc = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        base_acc[name] = accuracy_score(y_test, model.predict(X_test_scaled))
        base_auc[name] = roc_auc_score(y_test, model.predict_proba(X_test_scaled), multi_class='ovr')
    
    # CROSS-VALIDATION FIX: Update StratifiedKFold from 3 splits to 5 splits
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10]}
    xgb_grid = {'n_estimators': [100, 200], 'max_depth': [3, 5]}
    
    rf_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_grid, cv=cv, scoring='accuracy', n_jobs=1)
    rf_search.fit(X_train_scaled, y_train)
    best_rf = rf_search.best_estimator_
    
    xgb_search = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42), xgb_grid, cv=cv, scoring='accuracy', n_jobs=1)
    xgb_search.fit(X_train_scaled, y_train)
    best_xgb = xgb_search.best_estimator_
    
    tuned_acc = {
        "Random Forest": accuracy_score(y_test, best_rf.predict(X_test_scaled)),
        "XGBoost": accuracy_score(y_test, best_xgb.predict(X_test_scaled))
    }
    
    
    labels = ['Random Forest', 'XGBoost']
    before = [base_acc['Random Forest'], base_acc['XGBoost']]
    after = [tuned_acc['Random Forest'], tuned_acc['XGBoost']]
    
    x_pos = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x_pos - width/2, before, width, label='Before Tuning', color='skyblue')
    ax.bar(x_pos + width/2, after, width, label='After Tuning', color='salmon')
    ax.set_ylabel('Accuracy')
    ax.set_title('Gen Z Model Performance Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, 1.1])
    ax.legend()
    plt.tight_layout()
    plt.savefig('genz_model_comparison.png', dpi=150)
    
    
    plt.figure(figsize=(10, 8))
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    
    # ROC CURVE IMPROVEMENT: Setup for per-class One-vs-Rest ROC curves
    class_names = ['Low Risk', 'Medium Risk', 'High Risk']
    class_colors = ['green', 'orange', 'red']
    
    for name, model in [('Random Forest', best_rf), ('XGBoost', best_xgb)]:
        y_proba = model.predict_proba(X_test_scaled)
        
        # 1. Micro-Averaged ROC Curve (Keeping existing as baseline)
        fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
        auc_micro = roc_auc_score(y_test, y_proba, multi_class='ovr')
        plt.plot(fpr_micro, tpr_micro, lw=2, linestyle=':', 
                 label=f"{name} Micro-avg (AUC = {auc_micro:.4f})")
        
        # 2. Per-Class ROC Curves (One-vs-Rest)
        for i, color in enumerate(class_colors):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            auc_score = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
            plt.plot(fpr, tpr, color=color, lw=1.5, 
                     label=f"{name} - {class_names[i]} vs Rest (AUC = {auc_score:.4f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Gen Z Per-Class and Micro-Averaged ROC Curves (Tuned Models)')
    plt.legend(loc="lower right", fontsize='small')
    plt.tight_layout()
    plt.savefig("genz_roc_curves.png", dpi=150)
    
    
    # WHAT-IF ANALYSIS (INTERPRETATION ADDITION):
    # High-risk individuals may not shift entirely to a lower risk class immediately 
    # just by changing a single habit because the ML model assesses a complex, composite profile.
    # However, observing a reduction in the probability of being 'High Risk' is still a 
    # highly meaningful and actionable real-world financial insight, demonstrating gradual progress.
    
    high_risk_idx = y_test[y_test == 2].index[0]
    sample = genz_df.loc[high_risk_idx].copy()
    
    def predict_scenario(s):
        features = s[X.columns].values.reshape(1, -1)
        return best_rf.predict_proba(scaler_x.transform(features))[0][2]
    
    probs = []
    names = ['Original', 'Double Savings', 'Reduce Expenses 30%', 'Savings x2 & Exp -30%', 'Reduce Subs 50%']
    
    probs.append(predict_scenario(sample))
    
    s1 = sample.copy()
    s1['Savings (USD)'] *= 2.0
    probs.append(predict_scenario(s1))
    
    s2 = sample.copy()
    for c in expense_cols:
        s2[c] *= 0.7
    probs.append(predict_scenario(s2))
    
    s3 = sample.copy()
    s3['Savings (USD)'] *= 2.0
    for c in expense_cols:
        s3[c] *= 0.7
    probs.append(predict_scenario(s3))
    
    s4 = sample.copy()
    s4['Subscription Services (USD)'] *= 0.5
    probs.append(predict_scenario(s4))
    
    plt.figure(figsize=(10, 6))
    plt.bar(names, probs, color=['red', 'orange', 'gold', 'lightgreen', 'green'])
    plt.ylabel('Probability of HIGH RISK')
    plt.title('Gen Z What-If Analysis: 5 Scenarios')
    plt.xticks(rotation=15)
    plt.ylim(0, 1.0)
    for i, v in enumerate(probs):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig("genz_whatif_analysis.png", dpi=150)
    
    
    indian_accs = {
        "Logistic Regression": 0.9587,
        "Decision Tree": 0.8685,
        "Random Forest": 0.9240, # tuned 0.9270, but we'll use base to compare apples to apples
        "XGBoost": 0.9533 # tuned 0.9527
    }
    
    
    rf_imp = best_rf.feature_importances_
    genz_top_3 = [X.columns[i] for i in np.argsort(rf_imp)[::-1][:3]]
    
    indian_top_3 = ["Loan_Repayment", "Disposable_Income", "City_Tier"]
    
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(['Gen Z (Random Forest)', 'Indian (XGBoost)'], [tuned_acc['Random Forest'], indian_accs['XGBoost']], color=['#ff9999','#66b3ff'])
    ax1.set_ylim([0, 1.1])
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Best Model Accuracy Comparison')
    for i, v in enumerate([tuned_acc['Random Forest'], indian_accs['XGBoost']]):
        ax1.text(i, v + 0.02, f"{v:.4f}", ha='center')
        
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    table_data = [
        ['Rank', 'Gen Z Dataset', 'Indian Finance Dataset'],
        ['1', genz_top_3[0].replace(' (USD)',''), indian_top_3[0]],
        ['2', genz_top_3[1].replace(' (USD)',''), indian_top_3[1]],
        ['3', genz_top_3[2].replace(' (USD)',''), indian_top_3[2]]
    ]
    tbl = table(ax2, cellText=table_data, loc='center', cellLoc='center', colWidths=[0.1, 0.45, 0.45])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 2)
    ax2.set_title('Top 3 Risk Predictors', pad=20)
    
    
    ax3 = fig.add_subplot(gs[1, :])
    labels = list(models.keys())
    genz_vals = [base_acc[m] for m in labels]
    indian_vals = [indian_accs[m] for m in labels]
    
    x = np.arange(len(labels))
    width = 0.35
    ax3.bar(x - width/2, genz_vals, width, label='Gen Z', color='#ff9999')
    ax3.bar(x + width/2, indian_vals, width, label='Indian Finance', color='#66b3ff')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Base Model Accuracies: Gen Z vs Indian Finance')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_ylim(0, 1.1)
    ax3.legend()
    for i, v in enumerate(genz_vals):
        ax3.text(i - width/2, v + 0.01, f"{v:.2f}", ha='center', fontsize=9)
    for i, v in enumerate(indian_vals):
        ax3.text(i + width/2, v + 0.01, f"{v:.2f}", ha='center', fontsize=9)
        
    plt.tight_layout()
    plt.savefig('combined_comparison.png', dpi=150)
    print("Done generating all plots.")

if __name__ == "__main__":
    main()
