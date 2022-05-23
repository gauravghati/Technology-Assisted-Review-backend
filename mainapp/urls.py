from django.urls import path
from . import views

urlpatterns = [
    path('getspecificdoc/', views.getSpecificDoc, name='getSpecofocDocument'),
    path('updatedoc/', views.updateDocument, name='updateDocument'),
    path('documentlist/', views.documentList, name='documentList'),
    path('createpdf/', views.createPDF, name='createPDF'),
    path('predictdocs/', views.predictDocs, name='predictDocs'),
    path('calacc/', views.calAcc, name='calAccuracy'),
    path('getvariables/', views.getVariables, name='getVariables'),
    path('createdataset/', views.createDataset, name='createDataset'),
    path('updatevars/', views.updateVars, name='updateVars'),
    path('alldatasets/', views.alldatasets, name='alldatset')
]
