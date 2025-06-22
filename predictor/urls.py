from django.urls import path
from .views import predict_position,about_us,home_page

urlpatterns = [
    path('predict', predict_position, name='predict_position'),
    path('about',about_us,name='about'),
    path('',home_page,name='home_page')
]
