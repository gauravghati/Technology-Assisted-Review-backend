from django.urls import path
from . import views

urlpatterns = [
    path('createdocument/', views.createDocument, name='createDocument'),
    path('getspecificdoc/', views.getSpecificDoc, name='getSpecofocDocument'),
    path('getmostuncertaindoc/', views.getMostUncertainDoc, name='getMostUncertainDoc'),
    path('updatedoc/', views.updateDocument, name='updateDocument'),
    path('documentlist/', views.documentList, name='documentList'),
    path('createpdf/', views.createPDF, name='createPDF'),
]