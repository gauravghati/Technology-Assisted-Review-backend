from django.urls import path
from . import views

urlpatterns = [
    # path('createdocument/', views.createDocument, name='createDocument'),
    path('getspecificdoc/', views.getSpecificDoc, name='getSpecofocDocument'),
    # path('getmostuncertaindoc/', views.getMostUncertainDoc, name='getMostUncertainDoc'),
    path('updatedoc/', views.updateDocument, name='updateDocument'),
    path('documentlist/', views.documentList, name='documentList'),
    path('createpdf/', views.createPDF, name='createPDF'),
    path('inittrain/', views.initTrain, name='initTrain'),
    # path('retrain/', views.reTrain, name='reTrain'),
    path('predictdocs/', views.predictDocs, name='predictDocs'),
    path('calacc/', views.calAcc, name='calAccuracy'),
]