from django.urls import path

from . import views

urlpatterns = [
    path('down', views.some_streaming_csv_view, name='some_streaming_csv_view'),
    path('test', views.test, name='test'),
]