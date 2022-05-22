from ml.scripts.variables import *

from mainapp.models import Variable
import pandas as pd
from fpdf import FPDF
from django.core.files.base import File
from PyPDF2 import PdfFileReader
import os
from mainapp.models import Document

# will take the data from "tokenized_data_final.csv" and 
# create the objects of document in database and also pdf files.
def create_pdf_file(start_point, end_point):
    variable = Variable.objects.first()
    data = pd.read_csv( variable.main_project_location + BACKEND_FOLDER + COMPONENT_FOLDER + variable.curr_dataset + '/' + COMPLETE_DATAFILE )
    data = data[ start_point : end_point ]

    for index, row in data.iterrows():
        title = row[COLUMNS.TITLE]
        text = row[COLUMNS.TEXT]
        token = row[COLUMNS.TOKEN]

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", size = 20)
        pdf.multi_cell(200, 10, txt = title, align = 'C')

        pdf.set_font("Arial", size = 15)
        pdf.multi_cell(200, 10, txt = text )

        document = Document()
        document.save()

        pdf_path = variable.curr_dataset + '_' + str(document.pk) + ".pdf"
        pdf.output(name= pdf_path, dest='F')

        f = open(pdf_path, 'rb')
        myfile = File(f)

        pdf = PdfFileReader(open(pdf_path,'rb'))

        document.dataset_name = variable.curr_dataset
        document.document_size = os.path.getsize(pdf_path)
        document.document_file = myfile
        document.document_name = title
        document.document_text = text
        document.document_token = token
        document.word_count = len(text.split()) + len(title.split())
        document.page_count = pdf.getNumPages()

        document.save()
        os.remove(pdf_path)
