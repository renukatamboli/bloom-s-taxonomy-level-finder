from django.conf.urls import url

from .views import FileView,File1

urlpatterns = [
  url(r'^upload/$', FileView.as_view(), name='file-upload'),
  url(r'^File1/$',  File1.as_view(), name='file-show'),
]