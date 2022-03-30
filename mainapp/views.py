from email.policy import HTTP
from django.http import JsonResponse
from mainapp.models import Document, TYPE_CHOICES, LABEL_CHOICES, TRAIN_CHOICES
from mainapp.serializers import DocumentSerializer
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from ml.scripts.token_file_to_pdf import create_pdf_file, init_train, retrain_single_doc

@api_view(['POST'])
def createDocument(request):
    """
    # POST: To create document
    #     body:
    #       document_name : string
    #       document_file : string
    """
    document_name = request.data.get('document_name')
    document_file = request.data.get('document_file')

    document = Document( document_name=document_name, document_file=document_file )
    document.save()
    return Response(DocumentSerializer(document).data, status=status.HTTP_201_CREATED)


@api_view(['GET'])
def documentList(request):
    """
    # GET: Return All document List
    """
    documents = Document.objects.all()
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
    document_id = request.data['document_id']
    reviewed_label_name = request.data['reviewed_label_name']

    if document_id == "NULL":
        document = Document.objects.order_by('-uncertainity_score')[0]
    else:
        document = Document.objects.get(auto_id=document_id)

    document.is_reviewed = True
    document.reviewed_label_name = reviewed_label_name
    document.uncertainity_score = 0
    document.save()

    serialized_docs = DocumentSerializer( document )
    return Response(serialized_docs.data, status=status.HTTP_200_OK)

@api_view(['GET'])
def getMostUncertainDoc(request):
    """
    # GET: Return the most uncertain document
    """
    document = Document.objects.order_by('-uncertainity_score')[0]
    serialized_docs = DocumentSerializer( document )
    return Response(serialized_docs.data, status=status.HTTP_200_OK)


@api_view(['GET'])
def createPDF(request):
    create_pdf_file(50)
    return JsonResponse({"test" : "PDFs created"}, status=status.HTTP_200_OK)


@api_view(['GET'])
def initTrain(request):
    init_train()
    return JsonResponse({"test" : "Initial model trained"}, status=status.HTTP_200_OK)


@api_view(['GET'])
def reTrain(request):
    doc = Document.objects.get( auto_id = 287 )
    retrain_single_doc( doc )
    return JsonResponse({"test" : "single model trained"}, status=status.HTTP_200_OK)
