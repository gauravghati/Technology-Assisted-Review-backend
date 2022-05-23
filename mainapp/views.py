from django.http import JsonResponse
from mainapp.models import Document, Datasets, Variable
from mainapp.serializers import DocumentSerializer, VariableSerializer, DatasetSerializer
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from ml.scripts.ml_functions import retrain_single_doc, predict_docs, initial_training
from ml.scripts.utilities import create_pdf_file
from ml.scripts.variables import TRAIN_CHOICES

from ml.scripts.variables import *


@api_view(['GET'])
def documentList(request):
    """
    # GET: Return All document List
    """
    variable = Variable.objects.first()
    documents = Document.objects.filter( dataset_name = variable.curr_dataset )
    serialized_state = DocumentSerializer(documents, many=True)
    return Response(serialized_state.data, status=status.HTTP_200_OK)


@api_view(['POST'])
def getSpecificDoc(request):
    """
    # POST: Return specific document
    #     body:
    #       document_id : pk
    """
    document_id = request.data['document_id']
    document = Document.objects.get(pk=document_id)
    serialized_docs = DocumentSerializer( document )
    return Response(serialized_docs.data, status=status.HTTP_200_OK)


@api_view(['POST'])
def updateDocument(request):
    """
    # POST: Update specific document
    #     body:
    #       document_id : pk
    #       reviewed_label_name : string
    """
    variable = Variable.objects.first()
    curr_dataset = Datasets.objects.get( dataset_name = variable.curr_dataset )

    document_id = request.data['document_id']
    reviewed_label_name = request.data['reviewed_label_name']

    document = Document.objects.get(auto_id=document_id)
    document.is_reviewed = True

    document.reviewed_label_name = reviewed_label_name
    document.used_for_training = TRAIN_CHOICES.INQUE
    document.uncertainity_score = 0
    document.save()

    queue_length = Document.objects.filter( dataset_name = variable.curr_dataset, used_for_training = TRAIN_CHOICES.INQUE ).count()
    print( "Queue Length : ", queue_length )

    if curr_dataset.initial_trained :
        print( " Retraing single doc " )
        retrain_single_doc( document )
    elif queue_length > curr_dataset.initial_train_docs and not curr_dataset.initial_trained:
        curr_dataset.initial_trained = True
        curr_dataset.save()
        predict_docs()
        initial_training()

    serialized_docs = DocumentSerializer( document )
    return Response(serialized_docs.data, status=status.HTTP_200_OK)


@api_view(['POST'])
def createPDF(request):
    start_point = int( request.data['startIdx'] )
    end_point = int( request.data['endIdx'] )
    create_pdf_file(start_point, end_point)
    return JsonResponse({"test" : "PDFs created"}, status=status.HTTP_200_OK)


@api_view(['GET'])
def predictDocs(request):
    predict_docs()
    return JsonResponse({"test" : "model prediction done"}, status=status.HTTP_200_OK)


@api_view(['GET'])
def calAcc(request):
    variable = Variable.objects.first()
    curr_dataset = Datasets.objects.get( dataset_name = variable.curr_dataset )
    
    arr1 = [ curr_dataset.label1_accuracy, curr_dataset.label2_accuracy, curr_dataset.label3_accuracy, curr_dataset.label4_accuracy ]
    totalAcc = curr_dataset.total_accuracy

    json_response = {
        "accArr" : arr1,
        "acc" : totalAcc
    }
    return JsonResponse(json_response, status=status.HTTP_200_OK)


@api_view(['GET'])
def getVariables(request):
    variable = Variable.objects.first()
    serialized_variable = VariableSerializer( variable )
    return Response(serialized_variable.data, status=status.HTTP_200_OK)


@api_view(['POST'])
def createDataset(request):
    dataset_name = request.data['datasetName']
    label_1_name = request.data['label1Name']
    label_2_name = request.data['label2Name']
    label_3_name = request.data['label3Name']
    label_4_name = request.data['label4Name']
    token_size = request.data['tokenSize']
    initial_train_docs = request.data['initialDoc']

    new_dataset = Datasets( 
        dataset_name = dataset_name, 
        label_1_name = label_1_name, 
        label_2_name = label_2_name, 
        label_3_name = label_3_name,
        label_4_name = label_4_name,
        token_size = token_size,
        initial_train_docs = initial_train_docs
    )
    new_dataset.save()
    return JsonResponse({"test" : "Done Creation"}, status=status.HTTP_201_CREATED)


@api_view(['POST'])
def updateVars(request):
    main_project_location = request.data['main_project_location']
    initial_epochs = request.data['initial_epochs']
    increment_epochs = request.data['increment_epochs']
    inque_maxlen = request.data['inque_maxlen']
    batch_size = request.data['batch_size']
    curr_dataset_name = request.data['curr_dataset_name']

    variable = Variable.objects.first()

    variable.curr_dataset = curr_dataset_name
    variable.main_project_location = main_project_location
    variable.initial_epochs = initial_epochs
    variable.increment_epochs = increment_epochs
    variable.inque_maxlen = inque_maxlen
    variable.batch_size = batch_size
    variable.save()

    return JsonResponse({"test" : "Done Updation"}, status=status.HTTP_200_OK)


@api_view(['GET']) 
def alldatasets(request):
    all_datasets = Datasets.objects.all()
    serialized_state = DatasetSerializer(all_datasets, many=True)
    return Response(serialized_state.data, status=status.HTTP_200_OK)
