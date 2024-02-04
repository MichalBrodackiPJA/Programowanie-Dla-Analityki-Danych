# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import statsmodels.formula.api as smf

import plotly.graph_objects as go
from dash import Dash, dcc, html
import plotly.express as px


app = Dash(__name__)

##########################################################################
########### Kod analizy  #################################################

df = pd.read_csv('projektowa_praca_domowa/messy_data.csv', sep=', ')
df.rename(columns={'x dimension': "x_dim"}, inplace=True)
df.rename(columns={'y dimension': "y_dim"}, inplace=True)
df.rename(columns={'z dimension': "z_dim"}, inplace=True)
print("tu jeszcze ok")

df.table.replace(',', np.nan, inplace=True)
df.table.replace('58,', 58, inplace=True)
df.table.replace('57,', 57, inplace=True)
df.table.unique()
df.table = df.table.astype(float)

df.color = df.color.astype(str).str.upper()
df.cut = df.cut.astype(str).str.upper()
df.clarity = df.clarity.astype(str).str.upper()

#Oddzielę w kopii na chwilę gdzie zmienna depth ma wartości NAN a xyz nie mają
df_copy = df[df['depth'].isna() & ~df[['x_dim', 'y_dim', 'z_dim']].isna().any(axis=1)]
df_copy['depth'] = (2*df_copy.z_dim/(df_copy.y_dim + df_copy.x_dim))*100 # razy 100 żeby skala byłą jak w głównych danych
df.loc[df_copy.index] = df_copy # powrót do starego df
# Analog dla x,y,z 
#z
df_copy = df[df['z_dim'].isna() & ~df[['x_dim', 'y_dim', 'depth']].isna().any(axis=1)]
df_copy['z_dim'] = (df_copy.depth*(df_copy.x_dim + df_copy.y_dim))/200 # razy 100 żeby skala byłą jak w głównych danych
df.loc[df_copy.index] = df_copy # powrót do starego df
#x
df_copy = df[df['x_dim'].isna() & ~df[['z_dim', 'y_dim', 'depth']].isna().any(axis=1)]
df_copy['x_dim'] = ((df_copy.z_dim*200)/df_copy.depth) - df_copy.y_dim# razy 100 żeby skala byłą jak w głównych danych
df.loc[df_copy.index] = df_copy # powrót do starego df
#y
df_copy = df[df['y_dim'].isna() & ~df[['z_dim', 'x_dim', 'depth']].isna().any(axis=1)]
df_copy['y_dim'] = ((df_copy.z_dim*200)/df_copy.depth) - df_copy.x_dim# razy 100 żeby skala byłą jak w głównych danych
df.loc[df_copy.index] = df_copy # powrót do starego df

#zastąpmy NA
df['carat'].fillna(df.carat.median(), inplace=True)
df['table'].fillna(df.table.median(), inplace=True)
df = df.dropna(subset=['price'])

#Pierwsze podejscie
df_test_yz_mean = df.copy()

df_test_yz_mean['z_dim'].fillna(df_test_yz_mean.z_dim.mean(), inplace=True)
df_test_yz_mean['y_dim'].fillna(df_test_yz_mean.y_dim.mean(), inplace=True)

df_copy = df_test_yz_mean[df_test_yz_mean['depth'].isna() & ~df_test_yz_mean[['x_dim', 'y_dim', 'z_dim']].isna().any(axis=1)]
df_copy['depth'] = (2*df_copy.z_dim/(df_copy.y_dim + df_copy.x_dim))*100 # razy 100 żeby skala byłą jak w głównych danych
df_test_yz_mean.loc[df_copy.index] = df_copy # powrót do starego df

#Drugie podejscie
df_test_depth_mean = df.copy()
df_test_depth_mean['depth'].fillna(df_test_depth_mean.depth.mean(), inplace=True)
#z
df_copy = df_test_depth_mean[df_test_depth_mean['z_dim'].isna() & ~df_test_depth_mean[['x_dim', 'y_dim', 'depth']].isna().any(axis=1)]
df_copy['z_dim'] = (df_copy.depth*(df_copy.x_dim + df_copy.y_dim))/200 # razy 100 żeby skala byłą jak w głównych danych

df_test_depth_mean.loc[df_copy.index] = df_copy # powrót do starego df
#y
df_copy = df_test_depth_mean[df_test_depth_mean['y_dim'].isna() & ~df_test_depth_mean[['z_dim', 'x_dim', 'depth']].isna().any(axis=1)]
df_copy['y_dim'] = ((df_copy.z_dim*200)/df_copy.depth) - df_copy.x_dim# razy 100 żeby skala byłą jak w głównych danych
df_test_depth_mean.loc[df_copy.index] = df_copy # powrót do starego df

df_test_delete = df.dropna().copy()
######## Tu były boxploty #########
df = df_test_depth_mean

######### Tu też i to więcej ######

IQR = df.depth.quantile(0.75) - df.depth.quantile(0.25)
lower_outliers = df[df.depth < df.depth.quantile(0.25) - 1.5*IQR]
df = df[-df['depth'].isin(lower_outliers['depth'])]

le_cut = LabelEncoder()
le_clarity = LabelEncoder()
ohe = OneHotEncoder()

kolejnosc_cut = ["FAIR", "GOOD", "VERY GOOD", "IDEAL", "PREMIUM"]
kolejnosc_clarity = ['IF','VVS2', 'VVS1', 'SI2', 'SI1', 'I1']
kolejnosc_clarity = kolejnosc_clarity[::-1] #napisałem od tyłu

encoded_cut = le_cut.fit_transform(kolejnosc_cut)
encoded_clarity = le_clarity.fit_transform(kolejnosc_clarity)

mapping_cut = dict(zip(le_cut.classes_, le_cut.transform(le_cut.classes_)))
mapping_clarity = dict(zip(le_clarity.classes_, le_clarity.transform(le_clarity.classes_)))

df['cut'] = df['cut'].map(mapping_cut)
df['clarity'] = df['clarity'].map(mapping_clarity)

df_test = pd.get_dummies(df["color"],prefix='color')
df = pd.concat([df, df_test], axis=1)

df_mini = df[df.price < 100000]

formulabest = 'price ~ z_dim + clarity + table + carat + cut'
best_model = smf.ols(formula=formulabest, data=df_mini).fit()
##########################################################################
########### Funkcje do layoutu Dashbouardu #############################
def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

##########################################################################
########### Kod layoutu Dashboardu #######################################
app.layout = html.Div([
    html.H4(children='Wygląd danych po obróbce'),
    generate_table(df)
])


##########################################################################
if __name__ == '__main__':
    app.run_server(debug=True)
