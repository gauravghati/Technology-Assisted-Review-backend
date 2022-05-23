import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "veritas.settings")

import django
django.setup()

from django.contrib.auth.models import User
from mainapp.models import Variable, Datasets

# saving sample datafile
AG_news_dataset = Datasets( 
    dataset_name = "AG_News",
    label_1_name = 0, label_2_name = 1, label_3_name = 2, label_4_name = 3,
    token_size = 600,
    initial_train_docs = 60
)
AG_news_dataset.save()

curr_dir = os.getcwd()
prev_dir = os.path.abspath(os.path.join(curr_dir, os.pardir)) + "/"

# Adding Default 
global_variable = Variable(
    curr_dataset = "AG_News",
    main_project_location = prev_dir,
    initial_epochs = 20,
    increment_epochs = 1,
    inque_maxlen = 24,
    batch_size = 4
)
global_variable.save()

superuser = User(username='admin')
superuser.set_password('1234')
superuser.is_superuser = True
superuser.is_staff = True
superuser.save()
