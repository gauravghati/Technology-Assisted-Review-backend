MAIN_PROJECT_LOCATION = "~/Desktop/Final-Year-Project/"

BACKEND_FOLDER = "backend-veritas/"
FRONTEND_FOLDER = "frontend-veritas/"
PDF_FOLDER_LOCATION = "public/dataset/"
COMPONENT_FOLDER = "ml/components/"
SCRIPT_FOLDER = "ml/scripts/"
TOKENIZED_FILE = "tokenized_data_final.csv"
TRAINED_MODEL = 'trained_model.h5'

BERT_MODEL = 'bert-base-uncased'
CASED = 'uncased' in BERT_MODEL
MAXLEN = 600

import pandas as pd
from fpdf import FPDF
from django.core.files.base import File
from mainapp.models import Document
from PyPDF2 import PdfFileReader
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from tensorflow import keras

ID = "id"
TITLE = "title"
TEXT = "text"
TOKEN = "Tokenized"
CLASS = "Class Index"
COLUMNS = (
    (ID, "id"),
    (CLASS, "Class Index"),
    (TITLE, "title"),
    (TEXT, "text"),
    (TOKEN, "Tokenized"),
)

def create_pdf_file( NUM_DOC ):
    data = pd.read_csv( MAIN_PROJECT_LOCATION + BACKEND_FOLDER + COMPONENT_FOLDER + TOKENIZED_FILE )
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


def trainDocs( NUM_TRAIN ):
    df = pd.read_csv( MAIN_PROJECT_LOCATION + BACKEND_FOLDER + COMPONENT_FOLDER + TOKENIZED_FILE )
    remain_df, train = sliceDocsFromDataset( df, NUM_TRAIN )
    model = keras.models.load_model( COMPONENT_FOLDER + TRAINED_MODEL )
    trainModel( model, train )
    model.save( COMPONENT_FOLDER + TRAINED_MODEL )


def trainModel(model, train_data, validation_split = 0):
  X_train, y_train = deepLearningPrep(train_data)
  history = model.fit(X_train, y_train, epochs = 20,batch_size = 4, verbose = 1)
  return model, history


def sliceDocsFromDataset(df, size):
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


def deepLearningPrep(train, test=-1):
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
