
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name = 'home'),
    path('send_image', views.send_image, name = 'send_image'),
]
