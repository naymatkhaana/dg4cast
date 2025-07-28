import numpy as np
import pandas as pd
from db import get_connection
import glob
#from embedding import generate_embeddings, read_pdf_file
from psycopg2.extensions import register_adapter, AsIs
from sklearn.preprocessing import StandardScaler

def addapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)

def addapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)

def addapt_numpy_float32(numpy_float32):
    return AsIs(numpy_float32)

def addapt_numpy_int32(numpy_int32):
    return AsIs(numpy_int32)

def addapt_numpy_array(numpy_array):
    return AsIs(tuple(numpy_array))

register_adapter(np.float64, addapt_numpy_float64)
register_adapter(np.int64, addapt_numpy_int64)
register_adapter(np.float32, addapt_numpy_float32)
register_adapter(np.int32, addapt_numpy_int32)
register_adapter(np.ndarray, addapt_numpy_array)


def import_data(data_source):

    scaler = StandardScaler()
    filenames = glob.glob(data_source+"/*.csv")
    window_size=36
    
    windows_list = []; idx_ts_windows_list = []; col_ts_windows_list = []; parent_ts_windows_list = []; dates_list = [];
    
    for fname in filenames:
        df = pd.read_csv(fname)
        for col in df.columns:
            if col != 'date':
                windows = np.lib.stride_tricks.sliding_window_view(df[col].to_numpy(), window_size)
                windows_list.append( windows )
                idx_ts_windows_list.extend(  list(range(len(windows))) )
                col_ts_windows_list.extend(  [col]*len(windows) )
                parent_ts_windows_list.extend(  [fname]*len(windows) )
                dates_list.extend(  list(df['date'][-len(windows):]) )
                
    windows_list = np.vstack(windows_list)

        




    conn = get_connection()
    cursor = conn.cursor()

    # Store each embedding in the database
    for i in range(len(idx_ts_windows_list)): # (parent_ts, embedding) in enumerate(embeddings):
    
        windows_list[i,:] = scaler.fit_transform(windows_list[i,:].reshape(-1,1)).reshape(-1,)   
        cursor.execute(
            "INSERT INTO embeddings (id, idx_ts, col_ts, parent_ts, enddate, embeddings) VALUES (%s, %s, %s, %s, %s, %s)",
            (i, idx_ts_windows_list[i], col_ts_windows_list[i], parent_ts_windows_list[i],dates_list[i], list(windows_list[i,:])),
        )
    conn.commit()

    print( "import-data command executed. Data source: {}".format(data_source ) )
    
import_data('/scratch/fs47816/workdir/sample_scripts/time_series_dl/time-series-v5/Time-Series-Library/dataset/rag_data/data')
