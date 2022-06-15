from django.db import models
from django.utils import timezone

class AudioFile(models.Model):
    file_name = models.FileField()
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        app_label = "Audio"