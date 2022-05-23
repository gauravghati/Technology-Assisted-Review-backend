from sklearn.metrics import classification_report, accuracy_score
from ml.scripts.variables import *
import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from mainapp.models import Datasets, Document, Variable

tf.compat.v1.disable_eager_execution()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)
tf.compat.v1.experimental.output_all_intermediates(True)
warnings.filterwarnings('ignore')


# calculate accuracy per label and return the array of 4 labels
def calAccuracy( model ):
    variable = Variable.objects.first()
    df = pd.read_csv( variable.main_project_location + BACKEND_FOLDER + COMPONENT_FOLDER + variable.curr_dataset + '/' + INIT_TEST_FILE )
    X_df, Y_df = deep_learning_prep(df)

    predict_arr = model.predict(X_df)
    df_Y_predict = np.array([])
    df_Y_true = np.array([])

    for predicts in predict_arr:
        idx = np.where( predicts == max( predicts ) )
        idx[0][0] += 1
        df_Y_predict = np.append( df_Y_predict, idx )

    for doc_class in df[COLUMNS.CLASS]:
        df_Y_true = np.append( df_Y_true, doc_class )

    accuracy = accuracy_score( df_Y_predict, df_Y_true )

    cr = classification_report(df_Y_true, df_Y_predict, digits = 4, output_dict=True)
    df_cr = pd.DataFrame(cr).transpose()
    label1acc = df_cr['recall'].iloc[0]
    label2acc = df_cr['recall'].iloc[1]
    label3acc = df_cr['recall'].iloc[2]
    label4acc = df_cr['recall'].iloc[3]

    return [label1acc, label2acc, label3acc, label4acc], accuracy


# adding documents of every class to the dataframe and in queue also
def adding_single_doc_every_class( df, main_doc_label, curr_label ):
    variable = Variable.objects.first()
    if main_doc_label == curr_label:
        return df
    
    single_doc = Document.objects.filter (
        dataset_name = variable.curr_dataset,
        predicted_label_name = curr_label,
        used_for_training = TRAIN_CHOICES.NOTUSED
    ).order_by('uncertainity_score').first()

    if not single_doc :
        single_doc = Document.objects.filter(
            dataset_name = variable.curr_dataset,
            used_for_training = TRAIN_CHOICES.NOTUSED
        ).order_by('uncertainity_score').first()

    # pushing the most confident document of every label into the training queue
    single_doc.used_for_training = TRAIN_CHOICES.INQUE
    single_doc.save()
    val = {
        COLUMNS.ID : single_doc.auto_id, 
        COLUMNS.CLASS : label_to_class( curr_label ),
        COLUMNS.TITLE : single_doc.document_name,
        COLUMNS.TEXT : single_doc.document_text,
        COLUMNS.TOKEN : single_doc.document_token
    }
    df = df.append( val, ignore_index = True )
    return df


# Adding all documents of training queue to the dataframe
def add_in_queue_docs( df ):
    variable = Variable.objects.first()
    docs = Document.objects.filter( dataset_name = variable.curr_dataset, used_for_training = TRAIN_CHOICES.INQUE )

    for single_doc in docs:
        curr_label = ( single_doc.reviewed_label_name ) if ( single_doc.is_reviewed ) else ( single_doc.predicted_label_name )
        val = {
            COLUMNS.ID : single_doc.auto_id, 
            COLUMNS.CLASS : label_to_class( curr_label ),
            COLUMNS.TITLE : single_doc.document_name,
            COLUMNS.TEXT : single_doc.document_text,
            COLUMNS.TOKEN : single_doc.document_token
        }
        df = df.append( val, ignore_index = True )
    return df


# when the accuracy increases this function will be called, it'll remove all the documents from the training Queue
def empty_training_queue():
    variable = Variable.objects.first()
    docs = Document.objects.filter( dataset_name = variable.curr_dataset, used_for_training = TRAIN_CHOICES.INQUE )
    for doc in docs:
        doc.used_for_training = TRAIN_CHOICES.USED
        doc.save()


# returning the class number from the label
def label_to_class( main_label ):
    if main_label == LABELS.LABEL0 :
        return 1
    elif main_label == LABELS.LABEL1 :
        return 2
    elif main_label == LABELS.LABEL2 :
        return 3
    return 4


#removing docs predicted by model and saving only human reviewed docs for training 
def remove_predicted_inque_doc():
    variable = Variable.objects.first()
    docs = Document.objects.filter( dataset_name = variable.curr_dataset, is_reviewed = False, used_for_training = TRAIN_CHOICES.INQUE )
    for doc in docs:
        doc.used_for_training = TRAIN_CHOICES.NOTUSED
        doc.save()


# retraining => retraing model on single doc.
def retrain_single_doc( main_doc ):
    variable = Variable.objects.first()
    df = pd.DataFrame(columns = [COLUMNS.ID, COLUMNS.CLASS, COLUMNS.TITLE, COLUMNS.TEXT, COLUMNS.TOKEN])
    main_label = main_doc.reviewed_label_name

    val = {
        COLUMNS.ID : main_doc.auto_id, 
        COLUMNS.CLASS : label_to_class( main_label ),
        COLUMNS.TITLE : main_doc.document_name,
        COLUMNS.TEXT : main_doc.document_text,
        COLUMNS.TOKEN : main_doc.document_token
    }
    df = df.append( val, ignore_index = True )

    confident_queue_len = Document.objects.filter( dataset_name = variable.curr_dataset, is_reviewed = False, used_for_training = TRAIN_CHOICES.INQUE ).count()

    print( "InQue Length: ", confident_queue_len )
    if confident_queue_len > variable.inque_maxlen:
        remove_predicted_inque_doc()

    df = add_in_queue_docs( df )

    # adding labels of other 3 class
    df = adding_single_doc_every_class( df, main_label, LABELS.LABEL0 )
    df = adding_single_doc_every_class( df, main_label, LABELS.LABEL1 )
    df = adding_single_doc_every_class( df, main_label, LABELS.LABEL2 )
    df = adding_single_doc_every_class( df, main_label, LABELS.LABEL3 )

    model = keras.models.load_model( COMPONENT_FOLDER + variable.curr_dataset + '/' + TRAINED_MODEL )

    arr1, old_accuracy = calAccuracy( model )
    increment_train_model( model, df )
    arr2, new_accuracy = calAccuracy( model )

    print( "New Accuracy: ", new_accuracy )
    print( "Old Accuracy: ", old_accuracy )

    # saving model only if new accuracy is greater than old accuracy
    if new_accuracy > old_accuracy :
        empty_training_queue()
        model.save( COMPONENT_FOLDER + variable.curr_dataset + '/' + TRAINED_MODEL )

        curr_dataset = Datasets.objects.get( dataset_name = variable.curr_dataset )
        curr_dataset.total_accuracy = new_accuracy
        curr_dataset.label1_accuracy = arr2[0]
        curr_dataset.label2_accuracy = arr2[1]
        curr_dataset.label3_accuracy = arr2[2]
        curr_dataset.label4_accuracy = arr2[3]
        curr_dataset.save()


# initial training => fetching data from csv file, by default 50 Documents per class and intially training the model
def initial_training():
    variable = Variable.objects.first()
    curr_dataset = Datasets.objects.get( dataset_name = variable.curr_dataset )
    documents_inque = Document.objects.filter( dataset_name = curr_dataset.dataset_name, used_for_training = TRAIN_CHOICES.INQUE )

    df = pd.DataFrame(columns = [COLUMNS.ID, COLUMNS.CLASS, COLUMNS.TITLE, COLUMNS.TEXT, COLUMNS.TOKEN])

    for document in documents_inque :
        df = add_single_doc_in_df(df, document, document.reviewed_label_name)

    model = keras.models.load_model( COMPONENT_FOLDER + variable.curr_dataset + '/' + BLANK_MODEL )

    X_train, y_train = deep_learning_prep(df)
    model.fit(X_train, y_train, epochs = variable.initial_epochs, batch_size = variable.batch_size, verbose = 1)

    model.save( COMPONENT_FOLDER + variable.curr_dataset + '/' + TRAINED_MODEL )

    for document in documents_inque :
        document.used_for_training = TRAIN_CHOICES.USED
        document.save()

    predict_docs()

    # df = pd.read_csv( variable.main_project_location + BACKEND_FOLDER + COMPONENT_FOLDER + variable.curr_dataset + '/' + COMPLETE_DATAFILE )
    # remain_df, train = slice_docs_from_dataset( df, NUM_TRAIN )
    # model = keras.models.load_model( COMPONENT_FOLDER + variable.curr_dataset + '/' + BLANK_MODEL )
    # train_model( model, train )
    # model.save( COMPONENT_FOLDER + variable.curr_dataset + '/' + TRAINED_MODEL )
    # remain_df.to_csv( variable.main_project_location + BACKEND_FOLDER + COMPONENT_FOLDER + variable.curr_dataset + '/' + REMAINING_DATAFILE )


def add_single_doc_in_df(df, document, label = ""):
    val = {
        COLUMNS.ID : document.auto_id,
        COLUMNS.TITLE : document.document_name,
        COLUMNS.TEXT : document.document_text,
        COLUMNS.TOKEN : document.document_token,
        COLUMNS.CLASS : label,
    }
    return df.append( val, ignore_index = True )

def predict_docs():
    variable = Variable.objects.first()
    docs = Document.objects.filter(dataset_name = variable.curr_dataset)
    df = pd.DataFrame(columns = [COLUMNS.ID, COLUMNS.CLASS, COLUMNS.TITLE, COLUMNS.TEXT, COLUMNS.TOKEN])
    model = keras.models.load_model( COMPONENT_FOLDER + variable.curr_dataset + '/' + TRAINED_MODEL )

    for doc in docs :
        df = add_single_doc_in_df( df, doc )
    
    X_df, Y_df = deep_learning_prep(df)
    predict_arr = model.predict(X_df)

    j = 0
    for doc in docs:
        predict_array = predict_arr[j]
        j = j + 1
        doc.class_a_predit_percentage = predict_array[0] * 100
        doc.class_b_predit_percentage = predict_array[1] * 100
        doc.class_c_predit_percentage = predict_array[2] * 100
        doc.class_d_predit_percentage = predict_array[3] * 100
        
        max_val = predict_array[0]
        sum = 0
        doc_label = 0

        for i in range(0, 4):
            sum += predict_array[i]
            if max_val < predict_array[i]:
                max_val = predict_array[i]
                doc_label = i

        if doc_label == 0 :
            doc.predicted_label_name = LABELS.LABEL0
        elif doc_label == 1:
            doc.predicted_label_name = LABELS.LABEL1
        elif doc_label == 2:
            doc.predicted_label_name = LABELS.LABEL2
        elif doc_label == 3:
            doc.predicted_label_name = LABELS.LABEL3

        print( doc.predicted_label_name )
        doc.uncertainity_score = ( 1 - (max_val / sum) ) * 100
        doc.save()

    acc_arr, tot_acc = calAccuracy( model )

    curr_dataset = Datasets.objects.get( dataset_name = variable.curr_dataset )

    curr_dataset.total_accuracy = tot_acc
    curr_dataset.label1_accuracy = acc_arr[0]
    curr_dataset.label2_accuracy = acc_arr[1]
    curr_dataset.label3_accuracy = acc_arr[2]
    curr_dataset.label4_accuracy = acc_arr[3]
    curr_dataset.save()


# passing the df model gets trained on it.
def increment_train_model(model, train_data):
    variable = Variable.objects.first()
    X_train, y_train = deep_learning_prep(train_data)
    history = model.fit(X_train, y_train, epochs = variable.increment_epochs, batch_size = variable.batch_size, verbose = 1)
    return model, history


# take out all documents from df of every class and return the remaing df.
def slice_docs_from_dataset(df, size):
    class1 = df.loc[df[COLUMNS.CLASS] == CLASS_LABEL_NAME.LABEL0 ].head(size)
    class2 = df.loc[df[COLUMNS.CLASS] == CLASS_LABEL_NAME.LABEL1 ].head(size)
    class3 = df.loc[df[COLUMNS.CLASS] == CLASS_LABEL_NAME.LABEL2 ].head(size)
    class4 = df.loc[df[COLUMNS.CLASS] == CLASS_LABEL_NAME.LABEL3 ].head(size)

    df = df.drop(df.index[list(class1.index.values) + list(class2.index.values) + list(class3.index.values) + list(class4.index.values)])

    train = pd.concat([class1, class2, class3, class4])
    train.reset_index(drop=True, inplace=True)
    train = train.sample(frac=1).reset_index(drop=True)
    print("train set class counts:\n", train[COLUMNS.CLASS].value_counts())
    print("\n\ntrain set:\n", train.head())

    df.reset_index(drop=True, inplace=True)
    print("\n\nremaining dataset class counts:\n", df[COLUMNS.CLASS].value_counts())
    return df, train


# doing padding and dividing the dataset into X and Y colums
def deep_learning_prep(train, test=-1):
    variable = Variable.objects.first()
    curr_dataset = Datasets.objects.get( dataset_name = variable.curr_dataset )

    X = np.zeros((train.shape[0], curr_dataset.token_size ),dtype=np.int)
    for i,ids in tqdm(enumerate(list(train[COLUMNS.TOKEN]))):
        input_ids = [int(i) for i in ids.split()[: curr_dataset.token_size ]]
        inp_len = len(input_ids)
        X[i,:inp_len] = np.array(input_ids)

    Y = pd.get_dummies(train[COLUMNS.CLASS]).values
    
    if(type(test) is not int):
        X_test = np.zeros((test.shape[0], curr_dataset.token_size ),dtype=np.int)

        for i,ids in tqdm(enumerate(list(test[COLUMNS.TOKEN]))):
            input_ids = [int(i) for i in ids.split()[: curr_dataset.token_size ]]
            inp_len = len(input_ids)
            X_test[i,:inp_len] = np.array(input_ids)

        Y_test = pd.get_dummies(test[COLUMNS.CLASS]).values
        return X, Y, X_test, Y_test
    else:
        return X, Y
