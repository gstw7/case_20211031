import pandas as pd

def aux(df):
    
    '''
    in: DataFrame;
    out: DataFrame auxiliar
    '''
    
    
    df_aux = pd.DataFrame(
        {
        'columns': df.columns,
        'dtype': df.dtypes,
        'missing' : df.isna().sum(),
        'size' : df.shape[0],
        'nunique': df.nunique()
        }
        )

    df_aux['percentage'] = round(df_aux['missing'] / df_aux['size'], 3)*100
    
    return df_aux

def discount(s1_calc, s2):
    dsc = 100 - ((s2 * 100)/s1_calc)
    dsc = round(dsc, 0)
    if dsc > 0:
        dsc = 1
    elif dsc == 0:
        dsc = 0
    elif dsc < 0:
        dsc = -1
    return dsc

