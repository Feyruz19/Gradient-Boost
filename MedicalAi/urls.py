from django.urls import include, path

urlpatterns = [
    path('', include('diagnosis.urls')),
]
