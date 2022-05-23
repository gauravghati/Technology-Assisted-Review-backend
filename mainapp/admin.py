from django.contrib import admin
from numpy import var
from .models import Document, Variable, Datasets

# Register your models here.
admin.site.register(Document)
admin.site.register(Variable)
admin.site.register(Datasets)
