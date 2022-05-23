from django.db import models
from django.core.files.storage import FileSystemStorage

UPLOAD_ROOT = "../frontend-veritas/public/"
upload_storage = FileSystemStorage(location=UPLOAD_ROOT)

TYPE_CHOICES = (
    ("pdf", "pdf"),
    ("text", "text"),
    ("pict", "pict")
)

LABEL_CHOICES = (
    ("null", "null"),
    ("Label 0", "Label 0"),
    ("Label 1", "Label 1"),
    ("Label 2", "Label 2"),
    ("Label 3", "Label 3")
)

TRAIN_CHOICES = (
    ("not_used", "not_used"),
    ("in_queue", "in_queue"),
    ("used", "used")
)


class Document(models.Model):
    auto_id = models.AutoField(primary_key=True)
    created_on = models.DateTimeField(('Created date Time'), auto_now_add=True )
    updated_on = models.DateTimeField(('Update date Time'), auto_now=True )
    dataset_name = models.CharField(('Dataset Name'), default="null", max_length=500 )
    document_name = models.CharField(('Document Name'), max_length=500 )
    document_type = models.CharField(('Document Type'), choices=TYPE_CHOICES, max_length=20, default=0 )
    document_text = models.TextField(('Document Text'), null=True, blank=True )
    document_token = models.TextField(('Document Token'), null=True, blank=True )
    document_size = models.FloatField(('Document Size'), null=True, blank=True )
    document_file = models.FileField( upload_to="dataset/", storage=upload_storage, null=True, blank=True )
    uncertainity_score = models.FloatField( default=100 )
    is_reviewed = models.BooleanField(('Reviewed'), default=False )
    word_count = models.IntegerField(('Word Count'), null=True, blank=True )
    page_count = models.IntegerField(('Page Count'), null=True, blank=True )
    class_a_predit_percentage = models.FloatField(('class a predict'), null=True, blank=False )
    class_b_predit_percentage = models.FloatField(('class b predict'), null=True, blank=False )
    class_c_predit_percentage = models.FloatField(('class c predict'), null=True, blank=False )
    class_d_predit_percentage = models.FloatField(('class d predict'), null=True, blank=False )
    predicted_label_name = models.CharField(('Predicted Label Name'), choices=LABEL_CHOICES, max_length=50, default="null" )
    reviewed_label_name = models.CharField(('Reviewed Label Name'), choices=LABEL_CHOICES, max_length=50, default="null" )
    used_for_training = models.CharField(('Used For Training'), choices=TRAIN_CHOICES, max_length=50, default="not_used" )

    class Meta:
        verbose_name = ('Document')
        verbose_name_plural = ('Documents')

    def __str__(self):
        return str(self.document_name)


class Datasets( models.Model ):
    dataset_name = models.CharField(('Dataset Name'), max_length=500, default="", unique=True, primary_key=True)
    label_1_name = models.CharField(('Label 1 Name'), max_length=500, default="")
    label_2_name = models.CharField(('Label 2 Name'), max_length=500, default="")
    label_3_name = models.CharField(('Label 3 Name'), max_length=500, default="")
    label_4_name = models.CharField(('Label 4 Name'), max_length=500, default="")
    total_accuracy = models.FloatField(('Total Accuracy'), default=0 )
    label1_accuracy = models.FloatField(('Label 1 Accuracy'), default=0 )
    label2_accuracy = models.FloatField(('Label 2 Accuracy'), default=0 )
    label3_accuracy = models.FloatField(('Label 3 Accuracy'), default=0 )
    label4_accuracy = models.FloatField(('Label 4 Accuracy'), default=0 )
    token_size = models.IntegerField(('Token Size Length'), default=0 )
    initial_trained = models.BooleanField('Intially Trained', default=False)
    initial_train_docs = models.IntegerField(('Initial Training Document Count'), default=0)

    class Meta :
        verbose_name = ('Dataset')
        verbose_name_plural = ('Datasets')

    def __str__(self):
        return str(self.dataset_name)



class Variable( models.Model ):
    main_project_location = models.CharField(('Main Project Path'), max_length=1000 )
    initial_epochs = models.IntegerField(('Intial Epochs'), default=0 )
    increment_epochs = models.IntegerField(('Incremental Epochs'), default=0 )
    inque_maxlen = models.IntegerField(('Inque Maxlen'), default=0 )
    batch_size = models.IntegerField(('Batch Size'), default=0 )
    curr_dataset = models.CharField('Current Dataset', max_length=500, default="")

    class Meta:
        verbose_name = ('Variable')
        verbose_name_plural = ('Variables')
