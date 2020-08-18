import os
PATH_PROJECT= os.path.abspath(os.path.dirname(__file__))
PATH_FILE_DATA=os.path.join(PATH_PROJECT,'data')

class Config :
    DATASET_1_DIR= os.path.join(PATH_FILE_DATA,'input_data','google_trace','1_job','5_mins.csv')
    DATASET_2_DIR=os.path.join(PATH_FILE_DATA,'azure','5_mins.csv')
    FEATURE=[3]
    RATIO_TRAIN_TEST=0.8
    RATIO_TRAIN_VAL=0.8

    LSTM={'batch_size':49,
          'neurol':4,
          'epochs':100,
          'lock_back':2,
          'loss': 'mse',
          'optimizer':'adam'
          }
    GRU={'batch_size':49,
          'neurol':4,
          'epochs':100,
          'lock_back':2,
          'loss': 'mse',
          'optimizer':'adam'
          }








