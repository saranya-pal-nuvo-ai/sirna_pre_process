import pandas as pd
from src.Pre_Process.pre_processing import extract_accessibility_df
from src.Pre_Process.pre_processing import Filters
from src.AttSioff.inference import perform_inference



if __name__ == '__main__':

    f = Filters()

    df = perform_inference()

    print(df.shape)