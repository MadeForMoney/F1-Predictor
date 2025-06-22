from django import forms

class PredictionForm(forms.Form):
    driver = forms.CharField(label="Driver", max_length=100)
    track = forms.CharField(label="Track", max_length=100)
    starting_grid = forms.IntegerField(label="Starting Grid Position")
