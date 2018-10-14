from django.db import models

# Create your models here.


class User(models.Model):
    sdt = models.CharField(max_length=200)
    face = models.CharField(max_length=200)
