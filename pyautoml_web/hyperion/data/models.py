from django.db import models


# Create your models here.
class Source(models.Model):

    data_name = models.CharField(max_length=30, unique=True, null=False, blank=False)
    path = models.CharField(max_length=200, unique=True, null=False, blank=False)

    def __str__(self):
        return self.data_name
