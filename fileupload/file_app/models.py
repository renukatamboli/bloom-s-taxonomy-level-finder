from django.db import models

class File(models.Model):

  file = models.FileField(blank=False, null=False)
  remark = models.CharField(max_length=20)
  timestamp = models.DateTimeField(auto_now_add=True)

  def __str__(self):
      return str(self.file) + '-' + self.remark