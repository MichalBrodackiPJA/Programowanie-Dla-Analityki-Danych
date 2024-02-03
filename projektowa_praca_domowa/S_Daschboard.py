import streamlit as st 
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.graph_objects as go   

df1 = pd.read_csv('messy_data.csv', sep=',')
df2 = pd.read_csv('messy_data.csv', sep=', ')

df_test_yz_mean = pd.read_csv('df_test_yz_mean.csv')
df_test_depth_mean = pd.read_csv('df_test_depth_mean.csv')
df_test_delete = pd.read_csv('df_test_delete.csv')

df_before_Standarization = pd.read_csv('df_before_Standarization.csv')

df = pd.read_csv('df.csv')
df_mini = pd.read_csv('df_mini.csv')



selected = option_menu(
    menu_title='Menu główne',
    options = ['Dane', "Obróbka Danych", 'Modelowanie', 'Ewaluacja modelu'], 
    icons= ['upload', 'key', 'heart','flag'],
    menu_icon = 'cast',
    default_index=0,
    orientation='horizontal'
)

################################ Dane #########################################



if selected == 'Dane':
    st.title(f'Podgląd danych')
    with st.sidebar:
        Selected2 = option_menu(menu_title='Wybierz Moment obróbki danych:',
                                options = ['Wczytane domyślnie', 'Wczytane poprawnie', 
                                        "Po obróbce","Po standaryzacji", "Po usunięciu wartości odstających - końcowe" ])
        st.write("Jakie zmienne cię interesują?")
        Selected3 = st.checkbox(label="Dane numeryczne", value=True)
        Selected4 = st.checkbox(label="Dane kategoryczne", value=True)
    
    
    if Selected2 == 'Wczytane domyślnie' and Selected3==True and Selected4==True:
        st.write("Dane wczytane domyślnie - źle postawiony przecinek rozjeżdża wszystko :")
        st.write(df1)
        st.write("Podstawowe informacje o danych")
        st.write(df1.describe(include='all'))
    elif Selected2 == 'Wczytane domyślnie' and Selected3==False and Selected4==True:
        categorical_data = df1.select_dtypes(include='object')
        st.write("Dane wczytane domyślnie - źle postawiony przecinek rozjeżdża wszystko :")
        st.write(categorical_data)
        st.write("Podstawowe informacje o danych")
        st.write(categorical_data.describe(include='all'))
    elif Selected2 == 'Wczytane domyślnie' and Selected3==True and Selected4==False:
        numerical_data = df1.select_dtypes(include='number')
        st.write("Dane wczytane domyślnie - źle postawiony przecinek rozjeżdża wszystko :")
        st.write(numerical_data)
        st.write("Podstawowe informacje o danych")
        st.write(numerical_data.describe(include='all'))
        
    elif Selected2 == 'Wczytane poprawnie' and Selected3==True and Selected4==True:
        st.write("Dane Wczytane poprawnie - widać poprawę :")
        st.write(df2)
        st.write("Podstawowe informacje o danych")
        st.write(df2.describe(include='all'))
    elif Selected2 == 'Wczytane poprawnie' and Selected3==False and Selected4==True:
        categorical_data = df2.select_dtypes(include='object')
        st.write("Dane Wczytane poprawnie - widać poprawę :")
        st.write(categorical_data)
        st.write("Podstawowe informacje o danych")
        st.write(categorical_data.describe(include='all'))
    elif Selected2 == 'Wczytane poprawnie' and Selected3==True and Selected4==False:
        numerical_data = df2.select_dtypes(include='number')
        st.write("Dane Wczytane poprawnie - widać poprawę :")
        st.write(numerical_data)
        st.write("Podstawowe informacje o danych")
        st.write(numerical_data.describe(include='all'))
    
    elif Selected2 == 'Po usunięciu wartości odstających - końcowe' and Selected3==True and Selected4==True:
        st.write("Dane Wczytane poprawnie - widać poprawę :")
        st.write(df)
        st.write("Podstawowe informacje o danych")
        st.write(df.describe(include='all'))
    elif Selected2 == 'Po usunięciu wartości odstających - końcowe' and Selected3==False and Selected4==True:
        categorical_data = df.select_dtypes(include='object')
        st.write("Dane Wczytane poprawnie - widać poprawę :")
        st.write(categorical_data)
        st.write("Podstawowe informacje o danych")
        st.write(categorical_data.describe(include='all'))
    elif Selected2 == 'Po usunięciu wartości odstających - końcowe' and Selected3==True and Selected4==False:
        numerical_data = df.select_dtypes(include='number')
        st.write("Dane Wczytane poprawnie - widać poprawę :")
        st.write(numerical_data)
        st.write("Podstawowe informacje o danych")
        st.write(numerical_data.describe(include='all'))

elif selected == 'Obróbka Danych':
    st.title(f'Podstawowe wykresy pomocne przy obróbce danych')
    with st.sidebar:
        Selected2 = option_menu(menu_title='Wybierz zagadnienie z tematyki obróbki danych:',
                        options = ['Usuwanie NANów', "Badanie rozkładów", "Skalowanie", "Wartości odstające"])
    
    if Selected2 == "Usuwanie NANów":
        st.write("W tej części testowałem różne podejścia do zastąpienia NANów, w wyborze najelpszej pomogły wykresy:")
        st.image("output1.png", caption="Boxploty dla różnych podejśc do dodawania zmiennych")
        
    elif Selected2 == "Badanie rozkładów":
        st.write("W tej części dla lepszego zrozumienia zmiennych numerycznych i kategorycznych zwizualizowane zostały ich rozkłady")
        with st.sidebar:
            Selected3 = st.selectbox('Wybierz rodzaj zmiennych:', ['Numeryczne', 'Kategoryczne'])
        if Selected3 == "Numeryczne":
            st.image("output2.png", caption="Rozkłady zmiennych numerycznych")
        else:
            with st.sidebar:
                Selected4 = st.selectbox('Wybierz zmienną kategoryczną:', ['Clarity', 'Color', 'cut'])
            if Selected4 == 'cut':
                st.image("output34.png", caption="Rozkład cut")
            elif Selected4 == 'Clarity':
                st.image("output33.png", caption="Rozkład clarity")
            elif Selected4 == 'Color':
                st.image("output32.png", caption="Rozkład color")
    elif Selected2 == "Skalowanie":
        with st.sidebar:
            Selected3 = st.selectbox('Skalowanie?', ['Przed', "Po"])
        st.write("Tutaj próbowaliśmy sobie odpowiedzieć, czy jest potrzebne skalowanie")
        if Selected3 == "Przed":
            st.image("output3.png", caption="Przed skalowanie")    
        else:
            st.image("output4.png", caption="Po skalowaniu")
    elif Selected2 == "Wartości odstające":
        st.write("Sprawdźmy wpływ wartości odstających na rozkłady")
        with st.sidebar:
            Selected3 = st.selectbox('Usuwanie?', ['Przed', "Po"])
        if Selected3 == "Przed":
            with st.sidebar:
                Selected4 = st.selectbox('Numeryczne?', ['Tak', "Nie"])
            if Selected4 == "Tak":
                st.image("output2.png", caption="Przed usnięciem")
            else:
                st.image("output35.png", caption="Przed usnięciem")    
        else:
            with st.sidebar:
                Selected4 = st.selectbox('Numeryczne?', ['Tak', "Nie"])
            if Selected4 == "Tak":
                st.image("output22.png", caption="Po usunięciu")
            else:
                st.image("output33.png", caption="Po usunięciu")             

elif selected == 'Modelowanie':
    st.title(f'Wykresy wykorzystywane przy Modelowaniu')
    st.write("Korzystając już z ostatecznych danych, spróbujmy na podstawie korelacji znaleźć te z największą zależnością")
    with st.sidebar:
        st.title("Wybierz zmienne, żeby sprawdzić ich korelacje ze zmienną objaśnianą price")
        clarity = st.checkbox(label="clarity", value=True)
        carat = st.checkbox(label="carat", value=True)
        x_dim  = st.checkbox(label="x dimension", value=True)
        y_dim  = st.checkbox(label="y dimension", value=True)
        z_dim  = st.checkbox(label="z dimension", value=True)
        depth  = st.checkbox(label="depth", value=True)
        table  = st.checkbox(label="table", value=True)
    selected_columns = ['price']
    if clarity == True:
        selected_columns.append('clarity')
    if carat == True:
        selected_columns.append('carat')
    if x_dim == True:
        selected_columns.append('x_dim')
    if z_dim == True:
        selected_columns.append('z_dim')
    if y_dim == True:
        selected_columns.append('y_dim')
    if depth == True:
        selected_columns.append('depth')
    if table == True:
        selected_columns.append("table")
    
    # Create a heatmap using Seaborn
    df_selected = df_mini[selected_columns]

    correlation_matrix = df_selected.corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

    plt.title('Heatmapa Korelacji')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Display the Seaborn heatmap in Streamlit
    st.pyplot()
    
elif selected == 'Ewaluacja modelu':
    st.title(f'Oceńmy jakośc predykcji na podstawie wykresów')
    with st.sidebar:
        Selected2 = option_menu(menu_title="Wybierz rodzaj wykresu jaki chcesz zobaczyć",
                        options = ['Wykres Rezyduów - dowolna liczba zmiennych objaśniających',
                                   "Wykres dopasowania lini regresji - maksymalnie jedna zmienna objaśniająca"])
    if Selected2 == 'Wykres Rezyduów - dowolna liczba zmiennych objaśniających':
        st.write('TODO')
        
    elif Selected2 == "Wykres dopasowania lini regresji - maksymalnie jedna zmienna objaśniająca":
        st.write("TODO")
    
    
    
    
    