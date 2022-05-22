BACKEND_FOLDER = "backend-veritas/"
FRONTEND_FOLDER = "frontend-veritas/"
PDF_FOLDER_LOCATION = "public/dataset/"
COMPONENT_FOLDER = "ml/components/"
SCRIPT_FOLDER = "ml/scripts/"
COMPLETE_DATAFILE = "complete_datafile.csv"
REMAINING_DATAFILE = "remaining_datafile.csv"
TRAINED_MODEL = 'trained_model.h5'
BLANK_MODEL = "blank_model_4classes.h5"
INIT_TEST_FILE = 'init_test.csv'

BERT_MODEL = 'bert-base-uncased'
CASED = 'uncased' in BERT_MODEL


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


CLASS_LABEL_NAME = dotdict( {
    "LABEL0": "0",
    "LABEL1": "1",
    "LABEL2": "2",
    "LABEL3": "3",
} )

COLUMNS = dotdict( {
    "ID": "id",
    "CLASS": "Class Index",
    "TITLE": "title",
    "TEXT": "text",
    "TOKEN": "Tokenized"
})

LABELS = dotdict( {
    "NULL": "null",
    "LABEL0": "Label 0",
    "LABEL1": "Label 1",
    "LABEL2": "Label 2",
    "LABEL3": "Label 3"
})

TRAIN_CHOICES = dotdict({
    "NOTUSED": "not_used",
    "INQUE": "in_queue",
    "USED": "used"
})
