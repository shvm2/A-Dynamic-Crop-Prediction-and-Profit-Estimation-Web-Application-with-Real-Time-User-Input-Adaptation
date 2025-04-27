import pandas as pd

def get_rainfall(state, district, month):
    df = pd.read_csv('data/district wise rainfall normal.csv', index_col=False)
    row = df[(df['STATE_UT_NAME'] == state) & (df['DISTRICT'] == district)]
    vals = row[month].values
    if vals.shape[0] == 0:
        raise Exception(f"Unable to match month:{month} with state:{state}, district:{district}")
    return float(vals[0])
