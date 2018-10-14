from django.urls import path

from . import views

urlpatterns = [
    path('predict_cnn', views.predict_cnn, name='predict'),
    path('predict_svm', views.predict_svm, name='predict'),
    path('cnn', views.index_cnn, name='index_cnn'),
    path('svm', views.index_svm, name='index_svm')
]