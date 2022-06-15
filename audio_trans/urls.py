from django.contrib import admin
from django.urls import path
from .views import record
urlpatterns = [
    path('admin/', admin.site.urls),
    path("record/", record, name = "record")
]
