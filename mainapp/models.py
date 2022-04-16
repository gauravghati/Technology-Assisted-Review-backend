from django.db import models
from django.core.files.storage import FileSystemStorage
from model_utils import Choices

UPLOAD_ROOT = "../frontend-veritas/public/"
upload_storage = FileSystemStorage(location=UPLOAD_ROOT)

TYPE_CHOICES = Choices(
    ("pdf", "pdf"),
    ("text", "text"),
    ("pict", "pict")
)

LABEL_CHOICES = Choices(
    ("null", "null"),
    ("label_0", "label_0"),
    ("label_1", "label_1"),
    ("label_2", "label_2"),
    ("label_3", "label_3")
)

TRAIN_CHOICES = Choices(
    ("not_used", "not_used"),
    ("in_queue", "in_queue"),
    ("used", "used")
)

class Document(models.Model):
    auto_id = models.AutoField(primary_key=True)
    created_on = models.DateTimeField(('Created date Time'), auto_now_add=True )
    updated_on = models.DateTimeField(('Update date Time'), auto_now=True )
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
    predicted_label_name = models.CharField(('Predicted Label Name'), choices=LABEL_CHOICES, max_length=50, default="-" )
    reviewed_label_name = models.CharField(('Reviewed Label Name'), choices=LABEL_CHOICES, max_length=50, default="-" )
    used_for_training = models.CharField(('Used For Training'), choices=TRAIN_CHOICES, max_length=50, default=0 )

    class Meta:
        verbose_name = ('Document')
        verbose_name_plural = ('Documents')

    def __str__(self):
        return str(self.document_name)
