from django.db import models

class UserProfile(models.Model):
    name = models.CharField(max_length=100)
    image = models.ImageField(upload_to='profile_images/')
    created_at = models.DateTimeField(auto_now_add=True)
