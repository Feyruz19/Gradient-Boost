import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import shap

df = pd.read_excel("health_risk_dataset.xlsx")

le = LabelEncoder()
df['ExerciseLevel'] = le.fit_transform(df['ExerciseLevel'])

feature_names = ['Age', 'BMI', 'GlucoseLevel', 'BloodPressure', 'FamilyHistory', 'ExerciseLevel']
X = df[feature_names]
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(max_depth=5, min_samples_split=4, random_state=42)
rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=4, n_jobs=-1, random_state=42)
xgb = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42, eval_metric='logloss')

models = {
    'Decision Tree': dt,
    'Random Forest': rf,
    'XGBoost': xgb
}

results = {}
plt.figure(figsize=(18, 12))
plot_idx = 1

plt.subplot(2, 3, 1)
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': auc,
        'model': model
    }

plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(2, 3, 2)
for i, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.subplot(2, 3, 2 + i)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{name}\nAcc={results[name]["accuracy"]:.3f} AUC={results[name]["auc"]:.3f}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

plt.tight_layout()
plt.savefig('static/images/roc_cm.png', dpi=150, bbox_inches='tight')
plt.close()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='RdBu_r', center=0, fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('static/images/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 10))
for i, (name, model) in enumerate(models.items()):
    if name == 'Decision Tree':
        imp = model.feature_importances_
    elif name == 'Random Forest':
        imp = model.feature_importances_
    else:
        imp = model.feature_importances_
    plt.subplot(2, 2, i + 1)
    sorted_idx = np.argsort(imp)
    plt.barh(range(len(imp)), imp[sorted_idx], color='teal')
    plt.yticks(range(len(imp)), [feature_names[j] for j in sorted_idx])
    plt.xlabel('Importance')
    plt.title(f'{name} Feature Importance')
plt.tight_layout()
plt.savefig('static/images/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# Model Comparison Bar Chart
plt.figure(figsize=(10, 6))
names = list(results.keys())
accs = [results[n]['accuracy'] for n in names]
aucs = [results[n]['auc'] for n in names]
x = np.arange(len(names))
width = 0.35
plt.bar(x - width/2, accs, width, label='Accuracy', color='#2c7bb6')
plt.bar(x + width/2, aucs, width, label='AUC', color='#fdae61')
plt.xticks(x, names)
plt.ylabel('Score')
plt.title('Model Comparison')
plt.legend()
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('static/images/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# SHAP Explanation
xgb_model = results['XGBoost']['model']
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(12, 6))
shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig('static/images/shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()

# Population Risk Pie Chart
risk_counts = df['Outcome'].value_counts()
plt.figure(figsize=(8, 8))
colors = ['#66c2a5', '#fc8d62']
plt.pie(risk_counts.values, labels=['Healthy (0)', 'Disease (1)'], autopct='%1.1f%%', startangle=90, colors=colors, explode=(0, 0.05))
plt.title('Population Risk Distribution')
plt.tight_layout()
plt.savefig('static/images/risk_pie.png', dpi=150, bbox_inches='tight')
plt.close()

# Risk Trend (by Age groups)
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 100], labels=['18-30', '31-40', '41-50', '51-60', '60+'])
age_risk = df.groupby('AgeGroup')['Outcome'].mean() * 100
plt.figure(figsize=(10, 6))
age_risk.plot(kind='bar', color='#d7191c', edgecolor='black')
plt.ylabel('Risk (%)')
plt.title('Risk Trend by Age Group')
plt.xticks(rotation=0)
plt.grid(alpha=0.3, axis='y')
for i, v in enumerate(age_risk.values):
    plt.text(i, v + 1, f'{v:.1f}%', ha='center')
plt.tight_layout()
plt.savefig('static/images/risk_trend.png', dpi=150, bbox_inches='tight')
plt.close()

# Save models
joblib.dump(dt, 'models/dt_model.pkl')
joblib.dump(rf, 'models/rf_model.pkl')
joblib.dump(xgb, 'models/xgb_model.pkl')
joblib.dump(X_train.columns.tolist(), 'models/feature_names.pkl')
joblib.dump(le, 'models/label_encoder.pkl')

print("Training complete!")
print()
for name in names:
    r = results[name]
    print(f"{name}: Acc={r['accuracy']:.4f}, AUC={r['auc']:.4f}")
print(f"\nBest model: XGBoost (AUC={results['XGBoost']['auc']:.4f})")
