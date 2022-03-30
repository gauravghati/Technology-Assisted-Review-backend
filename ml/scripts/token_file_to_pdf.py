MAIN_PROJECT_LOCATION = "~/Desktop/Final-Year-Project/"

BACKEND_FOLDER = "backend-veritas/"
FRONTEND_FOLDER = "frontend-veritas/"
PDF_FOLDER_LOCATION = "public/dataset/"
COMPONENT_FOLDER = "ml/components/"
SCRIPT_FOLDER = "ml/scripts/"
COMPLETE_DATAFILE = "complete_datafile.csv"
REMAINING_DATAFILE = "remaining_datafile.csv"
TRAINED_MODEL = 'trained_model.h5'
BLANK_MODEL = "blank_model_4classes.h5"

BERT_MODEL = 'bert-base-uncased'
CASED = 'uncased' in BERT_MODEL
MAXLEN = 600
EPOCHS = 3

from typing import Collection
import pandas as pd
from fpdf import FPDF
from django.core.files.base import File
from PyPDF2 import PdfFileReader
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from tensorflow import keras
from mainapp.models import TRAIN_CHOICES, Document
from model_utils import Choices

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

COLUMNS = dotdict( {
    "ID": "id",
    "CLASS": "Class Index",
    "TITLE": "title",
    "TEXT": "text",
    "TOKEN": "Tokenized"
})

LABELS = dotdict( {
    "NULL": "null",
    "LABEL0": "label_0",
    "LABEL1": "label_1",
    "LABEL2": "label_2",
    "LABEL3": "label_3"
})

TRAIN_CHOICES = dotdict({
    "NOTUSED": "not_used",
    "INQUE": "in_queue",
    "USED": "used"
})

# will take the data from "tokenized_data_final.csv" and 
# create the objects of document in database and also pdf files.
def create_pdf_file( NUM_DOC ):
    data = pd.read_csv( MAIN_PROJECT_LOCATION + BACKEND_FOLDER + COMPONENT_FOLDER + COMPLETE_DATAFILE )
    data = data[0 : NUM_DOC]

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

        pdf_path = str(document.pk) + ".pdf"
        pdf.output(name= pdf_path, dest='F')

        f = open(pdf_path, 'rb')
        myfile = File(f)

        pdf = PdfFileReader(open(pdf_path,'rb'))

        document.document_size = os.path.getsize(pdf_path)
        document.document_file = myfile
        document.document_name = title
        document.document_text = text
        document.document_token = token
        document.word_count = len(text.split()) + len(title.split())
        document.page_count = pdf.getNumPages()

        document.save()
        os.remove(pdf_path)


def addToDatafram( df, main_doc_label, curr_label ):
    if main_doc_label == curr_label:
        return df
    doc = Document.objects.filter( predicted_label_name = curr_label, used_for_training = TRAIN_CHOICES.NOTUSED )
    single_doc = doc.order_by('uncertainity_score')[0]
    val = {
        COLUMNS.ID : single_doc.auto_id, 
        COLUMNS.CLASS : curr_label,
        COLUMNS.TITLE : single_doc.document_name,
        COLUMNS.TEXT : single_doc.document_text,
        COLUMNS.TOKEN : single_doc.document_token
    }
    df = df.append( val, ignore_index = True )
    return df


# retraining => retraing model on single doc.
def retrain_single_doc( main_doc ):
    df = pd.DataFrame(columns = [COLUMNS.ID, COLUMNS.CLASS, COLUMNS.TITLE, COLUMNS.TEXT, COLUMNS.TOKEN])
    main_label = main_doc.reviewed_label_name

    val = {
        COLUMNS.ID : main_doc.auto_id, 
        COLUMNS.CLASS : main_label,
        COLUMNS.TITLE : main_doc.document_name,
        COLUMNS.TEXT : main_doc.document_text,
        COLUMNS.TOKEN : main_doc.document_token
    }
    df = df.append( val, ignore_index = True )

    # adding labels of other 4 class 
    df = addToDatafram( df, main_label, LABELS.LABEL0 )
    df = addToDatafram( df, main_label, LABELS.LABEL1 )
    df = addToDatafram( df, main_label, LABELS.LABEL2 )
    df = addToDatafram( df, main_label, LABELS.LABEL3 )

    remain_df, train = slice_docs_from_dataset( df, 1 )
    model = keras.models.load_model( COMPONENT_FOLDER + TRAINED_MODEL )
    train_model( model, train )
    model.save( COMPONENT_FOLDER + TRAINED_MODEL )
    remain_df.to_csv( MAIN_PROJECT_LOCATION + BACKEND_FOLDER + COMPONENT_FOLDER + REMAINING_DATAFILE )


# initial training => fetching data from csv file, by default 50 Documents per class and intially training the model
def init_train( NUM_TRAIN = 50 ):
    df = pd.read_csv( MAIN_PROJECT_LOCATION + BACKEND_FOLDER + COMPONENT_FOLDER + COMPLETE_DATAFILE )
    remain_df, train = slice_docs_from_dataset( df, NUM_TRAIN )
    model = keras.models.load_model( COMPONENT_FOLDER + BLANK_MODEL )
    train_model( model, train )
    model.save( COMPONENT_FOLDER + TRAINED_MODEL )
    remain_df.to_csv( MAIN_PROJECT_LOCATION + BACKEND_FOLDER + COMPONENT_FOLDER + REMAINING_DATAFILE )


# passing the df model gets trained on it.
def train_model(model, train_data, validation_split = 0):
  X_train, y_train = deep_learning_prep(train_data)
  history = model.fit(X_train, y_train, epochs = EPOCHS, batch_size = 4, verbose = 1)
  return model, history


def slice_docs_from_dataset(df, size):
    class1 = df.loc[df[COLUMNS.CLASS] == 1].head(size)
    class2 = df.loc[df[COLUMNS.CLASS] == 2].head(size)
    class3 = df.loc[df[COLUMNS.CLASS] == 3].head(size)
    class4 = df.loc[df[COLUMNS.CLASS] == 4].head(size)

    df = df.drop(df.index[list(class1.index.values) + list(class2.index.values) + list(class3.index.values) + list(class4.index.values)])

    train = pd.concat([class1, class2, class3, class4])
    train.reset_index(drop=True, inplace=True)
    train = train.sample(frac=1).reset_index(drop=True)
    print("train set class counts:\n", train[COLUMNS.CLASS].value_counts())
    print("\n\ntrain set:\n", train.head())

    df.reset_index(drop=True, inplace=True)
    print("\n\nremaining dataset class counts:\n", df[COLUMNS.CLASS].value_counts())
    return df, train


def deep_learning_prep(train, test=-1):
    X = np.zeros((train.shape[0],MAXLEN),dtype=np.int)

    for i,ids in tqdm(enumerate(list(train[COLUMNS.TOKEN]))):
        input_ids = [int(i) for i in ids.split()[:MAXLEN]]
        inp_len = len(input_ids)
        X[i,:inp_len] = np.array(input_ids)

    Y = pd.get_dummies(train[COLUMNS.CLASS]).values
    
    if(type(test) is not int):
        X_test = np.zeros((test.shape[0],MAXLEN),dtype=np.int)

        for i,ids in tqdm(enumerate(list(test[COLUMNS.TOKEN]))):
            input_ids = [int(i) for i in ids.split()[:MAXLEN]]
            inp_len = len(input_ids)
            X_test[i,:inp_len] = np.array(input_ids)

        Y_test = pd.get_dummies(test[COLUMNS.CLASS]).values
        return X, Y, X_test, Y_test
    else:
        return X,Y
