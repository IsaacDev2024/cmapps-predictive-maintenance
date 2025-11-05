import os
import pickle
import numpy as np
import pandas as pd 
import streamlit as st 

@st.cache_data
def load_data(path_or_url, sep = ',', header = 'infer') -> pd.DataFrame:
    df = pd.read_csv(path_or_url, sep = sep, header = header)
    return df

@st.cache_resource
def load_stationary_pickle(path) -> dict|None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except:
        return None

