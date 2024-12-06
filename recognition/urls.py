from django.urls import path
from .views import FacialRecognitionAPI, AddFaceAPI

urlpatterns = [
    path('add-face/', AddFaceAPI.as_view(), name='add_face'),
    path('facial-recognition/', FacialRecognitionAPI.as_view(), name='facial_recognition'),
]
