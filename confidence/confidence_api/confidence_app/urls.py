from django.urls import path
from .views import AudioUploadView

urlpatterns = [
    path('predict/', AudioUploadView.as_view(), name='predict_confidence'),
]
