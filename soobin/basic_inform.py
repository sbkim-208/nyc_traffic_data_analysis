
import pandas as pd




def basic_info(df:pd.DataFrame) -> pd.DataFrame:

    

    return df.describe(), df.info(), df.shape
