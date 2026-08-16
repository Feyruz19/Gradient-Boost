from django.shortcuts import render
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from .forms import DiagnosisForm

dt = joblib.load('models/dt_model.pkl')
rf = joblib.load('models/rf_model.pkl')
xgb = joblib.load('models/xgb_model.pkl')
feature_names = joblib.load('models/feature_names.pkl')

def home(request):
    form = DiagnosisForm()
    return render(request, 'home.html', {'form': form})

def result(request):
    if request.method != 'POST':
        return render(request, 'home.html', {'form': DiagnosisForm()})
    form = DiagnosisForm(request.POST)

    if not form.is_valid():
        return render(request, 'home.html', {'form': form})

    data = form.cleaned_data
    input_data = np.array([[
        data['Age'], data['BMI'], data['GlucoseLevel'],
        data['BloodPressure'], int(data['FamilyHistory']), int(data['ExerciseLevel'])
    ]])

    dt_proba = dt.predict_proba(input_data)[0][1]
    rf_proba = rf.predict_proba(input_data)[0][1]
    xgb_proba = xgb.predict_proba(input_data)[0][1]

    dt_pred = dt.predict(input_data)[0]
    rf_pred = rf.predict(input_data)[0]
    xgb_pred = xgb.predict(input_data)[0]

    final_risk = round(xgb_proba * 100, 1)

    # SHAP Explanation
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(input_data)
    input_df = pd.DataFrame(input_data, columns=feature_names)
    shap_fig = shap.force_plot(explainer.expected_value, shap_values[0], input_df.iloc[0], matplotlib=True, show=False)
    buf = BytesIO()
    shap_fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    shap_plot = base64.b64encode(buf.getvalue()).decode()

    # Risk gauge - simple bar
    plt.figure(figsize=(6, 2))
    colors = ['#2ecc71' if final_risk < 40 else '#f39c12' if final_risk < 70 else '#e74c3c']
    plt.barh(0, final_risk, color=colors[0], height=0.5)
    plt.barh(0, 100, color='#ecf0f1', height=0.5, alpha=0.3)
    plt.xlim(0, 100)
    plt.xticks(range(0, 101, 10))
    plt.yticks([])
    plt.text(final_risk + 1, 0, f'{final_risk}%', va='center', fontsize=14, fontweight='bold')
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    gauge_plot = base64.b64encode(buf.getvalue()).decode()

    context = {
        'form': form,
        'age': data['Age'],
        'bmi': data['BMI'],
        'glucose': data['GlucoseLevel'],
        'bp': data['BloodPressure'],
        'family_history': 'Yes' if data['FamilyHistory'] == '1' else 'No',
        'exercise': ['Low', 'Medium', 'High'][int(data['ExerciseLevel'])],
        'final_risk': final_risk,
        'dt_proba': round(dt_proba * 100, 1),
        'rf_proba': round(rf_proba * 100, 1),
        'xgb_proba': round(xgb_proba * 100, 1),
        'dt_pred': 'Disease' if dt_pred == 1 else 'Healthy',
        'rf_pred': 'Disease' if rf_pred == 1 else 'Healthy',
        'rf_acc': 0.746,
        'rf_auc': 0.811,
        'xgb_acc': 0.744,
        'xgb_auc': 0.805,
        'dt_acc': 0.696,
        'dt_auc': 0.764,
        'shap_plot': shap_plot,
        'gauge_plot': gauge_plot,
    }
    return render(request, 'result.html', context)
