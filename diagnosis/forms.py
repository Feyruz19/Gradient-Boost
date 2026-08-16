from django import forms

class DiagnosisForm(forms.Form):
    Age = forms.IntegerField(label='Age', min_value=1, max_value=120, initial=45)
    BMI = forms.FloatField(label='BMI', min_value=10.0, max_value=60.0, initial=28.0)
    GlucoseLevel = forms.IntegerField(label='Glucose Level', min_value=40, max_value=300, initial=120)
    BloodPressure = forms.IntegerField(label='Blood Pressure', min_value=60, max_value=220, initial=120)
    FamilyHistory = forms.ChoiceField(label='Family History', choices=[(0, 'No'), (1, 'Yes')], initial=0)
    ExerciseLevel = forms.ChoiceField(label='Exercise Level', choices=[(0, 'Low'), (1, 'Medium'), (2, 'High')], initial=1)
