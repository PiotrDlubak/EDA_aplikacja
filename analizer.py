


import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import time
import openpyxl
import json
import openai
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import shapiro, normaltest, skewtest, kurtosistest, jarque_bera
import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import f
import numpy as np
from statsmodels.stats import weightstats as stests
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import wilcoxon
from scipy import stats
from scipy.stats import chi2
import pingouin as pg
import numpy as np
from scipy.stats import norm
from scipy.stats import wilcoxon

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import fisher_exact


import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import TTestPower

import statystyki as ana



import time
import openai
import json

#-------------------------------------------------------------------------------------------------------------

















# Inicjalizacja stanu sesji dla przechowywania wybranego DataFrame
if 'df' not in st.session_state:
    st.session_state.df = None
if 'typ_ladowania' not in st.session_state:
    st.session_state.typ_ladowania = None




st.set_page_config(
    page_title='Analizer',
    layout='wide',
    page_icon=':chart_with_upwards_trend:',
    initial_sidebar_state="expanded",
    )


st.markdown("""
        <style>
               .block-container {
                    padding-top: 2rem;
                    padding-bottom: 1rem;
                    padding-left: 4rem;
                    padding-right: 4rem;
                },
            
            #MainMenu {visibility: hidden; }

        </style>
        """, unsafe_allow_html=True)


with st.container(border=True):
    st.subheader(' :blue[Aplikacja **Analizer** - ver. 1.09   ]')

with st.container(border=True):

   
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📈 O aplikacji", "Załaduj dane","🔎 Podgląd danych", "🛠️ Ustaw parametry analizy", "🗓️ Raport z analizy - EDA", "➿ Uczenie maszynowe  ML","📖 pomoc"])
    st.write('')
    with tab1:    # o aplikacji
        st.write('')
       
        col1, col2= st.columns([2,2], gap = "medium")
        col1.image('logo2.png')
        col2.markdown("""
                ## O Aplikacji Analizer  Statystyka ver. 1.09
                Aplikacja Statystyka jest przeznaczona do ładowania, przeglądania i analizowania zestawów danych. Umożliwia wybór różnych parametrów i miar statystycznych, co czyni ją potężnym narzędziem dla analityków danych i osób interesujących się statystyką.
                Jest ona skierowana do analityków danych, studentów, nauczycieli, a także każdego, kto interesuje się analizą danych i statystyką. Jest to szczególnie przydatne dla tych, którzy chcą zgłębić swoje umiejętności analizy danych i zrozumieć różnorodne aspekty zbiorów danych.
                - Aplikacja wykonuje analizy jednej zmiennej numerycznej, jednej zmiennej kategorialnej, dwóch zmiennych ilościowych, dwóch zmiennych kategorialnych oraz analizę zmiennej numerycznej i kategorialnej.
                - Liczy szereg miar statystycznych, w tym średnią, medianę, odchylenie standardowe, rozstęp, kwartyle, skośność, kurtozę, a także wykonuje testy statystyczne jak chi-kwadrat.
                """)
        st.divider()
        col1, col2, col3, col4 = st.columns([2,2,2,2])
        col1.write('Autor: Piotr Dłubak' )
        col2.write(":e-mail: statystyk@o2.pl")
        col3.write('WWW: https://piotrdlubak.github.io/')
        col4.write("GitHub :https://github.com/PiotrDlubak")

    
    
    with tab2:    # załaduj dane
          
        # Inicjalizacja słownika przed wczytaniem plików
        dane_dict = {}

        # Lista nazw plików i kluczy
        pliki = {
            "test" : "test.xlsx",
            "sklepy": "sklepy.xlsx",  
            "szkoła": "szkoła.xlsx",
            "test": "test.xlsx",
            "iris": "iris.csv",
            "napiwki": "tips.csv"
        }

        # Próba załadowania każdego pliku danych
        for klucz, nazwa_pliku in pliki.items():
            try:
                # Sprawdzenie rozszerzenia pliku i wczytanie danych
                if nazwa_pliku.endswith('.xlsx'):
                    dane_dict[klucz] = pd.read_excel(nazwa_pliku)
                elif nazwa_pliku.endswith('.csv'):
                    dane_dict[klucz] = pd.read_csv(nazwa_pliku)
            except FileNotFoundError:
                st.error(f"Nie znaleziono pliku: {nazwa_pliku}. Upewnij się, że plik jest w katalogu roboczym.")

        # Załadowanie danych demonstracyjnych
        try:
            iris = pd.read_csv("iris.csv") 
            napiwiki = pd.read_csv("tips.csv")
            szkoła = pd.read_excel("szkoła.xlsx")
            sklep = pd.read_excel("sklepy.xlsx")
            test = pd.read_excel("test.xlsx")
        except FileNotFoundError:
            st.error("Nie znaleziono plików danych. Upewnij się, że pliki są w katalogu roboczym.")

        # Umieszczenie wczytanych danych w słowniku
        dane_dict = {
            "test": test,
            "sklepy": sklep,
            "szkoła" : szkoła,
            "iris": iris,
            "napiwki": napiwiki
        }


        typ_ladowania = st.radio(":blue[Ładowanie danych:]", ['Dane demonstracyjne'], horizontal=True)
        if typ_ladowania == 'Dane demonstracyjne':
                        wybrane_dane = st.selectbox("Wybierz dane demonstracyjne", list(dane_dict.keys()))
                        with st.spinner('Trwa ładowanie pliku...'):
                            time.sleep(2)
                            st.success(f"Pomyślnie załadowano plik danych: {wybrane_dane}")
                            st.session_state.df = dane_dict[wybrane_dane]   

        elif typ_ladowania == 'Załaduj własny plik danych':
                        st.caption('plik nie może zawierać brakujących wartości')
                        uploaded_file = st.file_uploader("Załaduj własny plik danych (CSV, Excel)", type=['csv', 'xlsx','xls'])
                        if uploaded_file is not None:
                            if uploaded_file.name.endswith('.csv'):
                                with st.spinner('Trwa ładowanie pliku...'):
                                    time.sleep(2)
                                    st.session_state.df = pd.read_csv(uploaded_file)
                                    st.success(f"Pomyślnie załadowano plik danych: {uploaded_file.name}")
                            elif uploaded_file.name.endswith('.xlsx'):
                                    time.sleep(2)
                                    st.session_state.df = pd.read_csv(uploaded_file)
                                    st.success(f"Pomyślnie załadowano plik danych: {uploaded_file.name}")
                            else:
                                st.error("Nieobsługiwany format pliku. Proszę załadować plik CSV lub Excel.")

        

    with tab3:    # podgląd danych
        st.write("")
        if typ_ladowania == 'Dane demonstracyjne':
            st.info(f"Wybrano dane demonstracyjne: {wybrane_dane}")
        elif typ_ladowania == 'Załaduj własny plik danych':
            if uploaded_file is not None:
                st.info(f"Załadowano plik danych do analizy: {uploaded_file}")
            else:
                st.info("Brak załadowanego pliku danych.")
                
        if st.session_state.df is not None:
            st.dataframe(st.session_state.df)
            dframe = st.session_state.df
        else:
            st.write("Brak danych do wyświetlenia.")




    with tab4:    # parametry analizy
            
            if typ_ladowania == 'Dane demonstracyjne':
                    st.info(f'Wybrano dane demonstracyjne: "{str.upper(wybrane_dane)}"')
            elif typ_ladowania == 'Załaduj własny plik danych':
                st.info(f"Załadowano plik danych do analizy: {str.upper(uploaded_file)}")
                
            kolumny_numeryczne = st.session_state.df.select_dtypes(include=[np.number]).columns
            kolumny_kategorialne = st.session_state.df.select_dtypes(exclude=[np.number]).columns
            with st.container(border=True):
                typ_analizy = st.radio(':blue[Wywierz typ analizy: ]',
                                       ['analiza jednej zmiennej numerycznej', 'analiza jednej zmiennej kategorialnej', 'analiza dwóch zmiennych ilościowych', 'analiza dwóch kategorialnych', 
                                        'analiza zmiennej numerycznej i kategorialnej'], horizontal=True)

            with st.container(border=True):

                if typ_analizy== 'analiza jednej zmiennej numerycznej':
                    
                    st.info(f'Wybrano analizę: "{str.upper(typ_analizy)}"')
                    st.write(':blue[ustaw parametry analizy:]')
                    col1, col2, col3,col4, col5= st.columns([1,1,1,1,2], gap='medium')
                    with col1:
                        tabela_licz_n = st.checkbox('tabela liczebności i częstości')
                    with col2:
                        wykresy_n = st.checkbox('Wykresy')
                    with col3:
                        odstaj_n = st.checkbox('Obserwacje odstające')
                    with col4:
                        statystyki_n = st.checkbox('Miary statystyczne')
                    st.write('')
                    st.write('')
                        
                    col1, col2= st.columns(2, gap='large')

                    with col1:
                        st.write(':blue[Estymacja przedziałowa:]')
                        k31 = st.checkbox('Przedział ufnosci do wybranego parametru (bootsrap)')
                        par_ci = st.selectbox("Wybierz parametr",['średnia','odchylenie std', 'mediana', 'Q1', 'Q3'])
                        alfa_ci = st.number_input('ustaw poziom alfa', min_value=0.01, max_value= 0.05, step= 0.01)
                    with col2: 
                        st.write(':blue[Weryfikacja hipotez statystycznych:]')
                        k44 = st.checkbox('Badanie normalnosci rozkładu')
                        norm = st.multiselect('Wybierz rodzaj testu',
                                                        ["shapiro","lilliefors","dagostino", "skewness", "kurtosis","jarque-bera"])
                        alfa_norm = st.number_input('ustaw poziom alfa:', min_value=0.01, max_value= 0.05, step= 0.01, help = 'wybrać nalezy warośc')
                    
                if typ_analizy== 'analiza jednej zmiennej kategorialnej':

                
                    st.info(f'Wybrano analizę: "{str.upper(typ_analizy)}"')
                    st.write(':blue[ustaw parametry analizy:]')
                    col1, col2, col3,col4, col5= st.columns([1,1,1,1,2], gap='medium')
                    with col1:
                        tabela_licz_k = st.checkbox('tabela liczebności i częstości')
                    with col2:
                        wykresy_k = st.checkbox('Wykresy')
                    with col3:
                        statystyki_k = st.checkbox('Miary statystyczne')
                    st.write('')



                if typ_analizy== 'analiza dwóch zmiennych ilościowych':
                
                    st.info(f'Wybrano analizę: "{str.upper(typ_analizy)}"')
                    st.write(':blue[ustaw parametry analizy:]')
                    col1, col2, col3,col4, col5= st.columns([1,1,1,1,2], gap='medium')

                    with st.container(border=True):
                        col1, col2, col3,col4, col5= st.columns([1,1,1,1,2], gap='medium')
                        with col1:
                            tabela_korelacji = st.checkbox('tabela  korelacji')  
                        with col2:
                            wykres = st.checkbox('Wykresy')
                        with col3:
                            regresja = st.checkbox('regresja liniowa')
                        st.write('')
                        



                if typ_analizy== 'analiza dwóch kategorialnych':
                
                    st.info(f'Wybrano analizę: "{str.upper(typ_analizy)}"')
                    st.write(':blue[ustaw parametry analizy:]')
                    col1, col2, col3,col4, col5= st.columns([1,1,1,1,2], gap='medium')

                    with st.container(border=True):
                        col1, col2, col3,col4, col5= st.columns([1,1,1,1,2], gap='medium')
                        with col1:
                            tabele_kontygencji = st.checkbox('tabele kontygencji')  

                        with col2:
                            miary_zaleznosci_k = st.checkbox('Miary zależnosci')    
                        with col3:
                            wykresy_k2 = st.checkbox('Wykresy')




    with tab5:


        if typ_analizy =='analiza jednej zmiennej numerycznej':

            col1, col2, col3 = st.columns([2,2,4])
            with col1:
                st.write(f'Wybrany typ analizy:')
                st.info(f':red[{str.upper(typ_analizy)}]')
                wybrana_kolumna = st.selectbox("Wybierz kolumnę", kolumny_numeryczne)  
            with col2:
                st.write(f'Wybrana zmienna:')
                st.info(f':red[{str.upper(wybrana_kolumna)}]')

            with st.container(border=True):

                if tabela_licz_n:
                        st.markdown(':blue[**Tabela liczebności i częstosci:**]')
                        st.dataframe(ana.tabela_licz_n(st.session_state.df, wybrana_kolumna, 6))
                if statystyki_n:
                        st.markdown(':blue[**Obszary zmienności zmiennej:**]')
                        st.dataframe(ana.obszary_zm(st.session_state.df, wybrana_kolumna), width=800, height=108)
                        st.markdown(':blue[**Wartosci parametrów statystycznych wybranej zmiennej:**]')

                        statystyki_a = ana.stat(st.session_state.df, [wybrana_kolumna])
                        statystyki_a = statystyki_a[['liczba','suma','min','max','średnia', 'rozstęp', 'p_10%', 'Q1_25%','Q2_50%', 'Q3_75%', 'p_90%']]
                        statystyki_b = ana.stat(st.session_state.df, [wybrana_kolumna])
                        statystyki_b = statystyki_b[['IQR','odch_cwiar','odchylenie przeciętne','wariancja','odch_std','błąd_odch_std','kl_wsp_zmien', 'poz_wsp_zmien', 'moda', 'skośność', 'kurtoza']]
                        st.dataframe(statystyki_a, width=1800, )
                        st.dataframe(statystyki_b, width=1800)
                
                if wykresy_n:
                    st.divider()
                    col1, col2, col3, col4 = st.columns([1,1,1,1])
                    st.write('')
                    with col1:
                        st.markdown(':blue[**Histogram:**] ')
                        ana.hist_plot(st.session_state.df, wybrana_kolumna, stat = 'count')
                    with col2:
                        ana.kde_plot(st.session_state.df, wybrana_kolumna)
                    with col3:
                        ana.ecdf_plot(st.session_state.df, wybrana_kolumna)
                
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        ana.box_plot(st.session_state.df, wybrana_kolumna)
                    with col2:   
                        ana.violin_plot(st.session_state.df, wybrana_kolumna)
                    with col3:
                        ana.swarm_plot(st.session_state.df, wybrana_kolumna)      
                

            with st.container(border=True):
                st.write('')
                st.markdown("Dokonaj interpretacji wyliczonych miar statystycznych z uzycliem modelu: gpt-3.5-turbo")
                interpretacja  = st.button('Interpretuj')
                if interpretacja :

                    miary = (ana.stat(st.session_state.df, [wybrana_kolumna]).T)
                    json_data = miary.to_json()
                    
                    prompt = """
                    System: Witam! Jestem pomocnym asystentem statystycznym. Jak mogę Ci dzisiaj pomóc?
                    Użytkownik: Chciałbym uzyskać interpretację statystyczną dla następujących miar obliczonych dla mojego Dataframe:
                    {json_data}
                    System: Na podstawie dostarczonych informacji dokonaj opisu dataframe i dokonaj interpretacji wartości statystyk:
                    """


                    openai.api_key = st.secrets["klucz"]

                    completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "Jesteś statystykiem, który ma w prosty sposób tłumaczyć i wyjaśniać co oznaczają wartosci wyliczonych parametrów statystycznych"},
                            {"role": "user", "content": prompt.format(json_data=json_data)}
                        ]
                    )

                    odpowiedz_LLM = completion.choices[0].message['content']

                    st.markdown(f"Odpowiedź LLM: {odpowiedz_LLM}")




        if typ_analizy =='analiza jednej zmiennej kategorialnej':
              
            col1, col2, col3 = st.columns([2,2,4])
            with col1:
                st.write(f'Wybrany typ analizy:')
                st.info(f':red[{str.upper(typ_analizy)}]')
                wybrana_kolumna = st.selectbox("Wybierz kolumnę zmiennej ", kolumny_kategorialne)  
            with col2:
                st.write(f'Wybrana zmienna:')
                st.info(f':red[{str.upper(wybrana_kolumna)}]')

            with st.container(border=True):

                if tabela_licz_k:
                    st.markdown(':blue[**Tabela liczebności i częstosci:**]')
                    st.dataframe(ana.cat_stat(st.session_state.df, wybrana_kolumna), width = 2000)
                    st.write('')
                    st.dataframe(ana.print_frequency_table(st.session_state.df, wybrana_kolumna), width=900)

                if statystyki_k:
                    col1.markdown(':blue[**wybrane statystyki:**]')
                    st.dataframe(ana.analyze_categorical_data(st.session_state.df, wybrana_kolumna),width=1500)
                    help = ''' IQV (na podstawie Entropii Informacyjnej): Jest to znormalizowana entropia informacyjna.
                                ModVR (Variability Ratio): ModVR mierzy zmienność w rozkładzie częstości kategorii.
                                Freeman's index (v): Indeks Freemana mierzy równomierność rozkładu częstości.
                                AvDev (Average Deviation): Średnia odchylenie od średniej arytmetycznej.
                                MNDif (Mean Normalized Difference): Średnia znormalizowana różnica.
                                VarNC (Variance of Normalized Count): Wariancja znormalizowanej liczby.
                                M1 (Indeks Gibbsa): Jest to pierwszy indeks Gibbsa, który mierzy rozkład częstości kategorii w próbie.
                                M2 (Indeks Gibbsa): Jest to drugi indeks Gibbsa, który mierzy rozkład częstości kategorii w próbie.
                                Incidence of Coincidence (IC): Incydencja zbieżności. Mierzy stopień, do którego zdarzenia zdarzają się razem.'''
                    st.caption(help)
                  
                
                if wykresy_k:
                    st.divider()
                    col1, col2, col3 = st.columns([1,1,1])
                    col1.markdown(':blue[**wykres liczebnosci:**]')
                    col1.pyplot(ana.category_one_plot(st.session_state.df, wybrana_kolumna, 'count'))
                    col2.markdown(':blue[**wykres częstości:**]')
                    col2.pyplot(ana.category_one_plot(st.session_state.df, wybrana_kolumna, 'percent'))
                     


                    # st.pyplot(sns.displot(dane, x=x, kind="hist", col = 'ocena',  height=3, color="#3498db", bins = 8, kde = True))
                    # st.pyplot(sns.displot(dane, x="dochody", kind="ecdf", col = 'ocena',  height=3, color="#3498db"))
                    # st.pyplot(sns.catplot(dane, x="dochody", y="ocena", kind="box", height=4, palette=palette))
                    # st.pyplot(sns.catplot(dane, x="dochody", y="ocena", kind="violin", height=4, palette=palette))
                    # st.pyplot(sns.catplot(dane, x="dochody", y="ocena", kind="point", height=4, palette=palette, estimator='mean', errorbar=("ci",95)))
                    # st.pyplot(sns.catplot(dane, x="dochody", y="ocena", kind="bar", height=4, palette=palette, estimator="sum"))
                    # st.pyplot(sns.catplot(dane, x="dochody", y="ocena", kind="swarm", height=4, palette=palette,  marker=".", linewidth=1,size=2,  edgecolor="#3498db"))
                    # st.pyplot(sns.catplot(dane, x="dochody", y="ocena", kind="boxen",color="#3498db",height=4))



        if typ_analizy =='analiza dwóch zmiennych ilościowych':
              
            col1, col2, col3 = st.columns([2,2,4])
            col1.write(f'Wybrany typ analizy:')
            col2.info(f':red[{str.upper(typ_analizy)}]')
            col1, col2 = st.columns([2,2])
            wybrana_kolumna_1 = col1.selectbox("Wybierz kolumnę zmiennej nr 1", kolumny_numeryczne)  
            wybrana_kolumna_2 = col2.selectbox("Wybierz kolumnę zmiennej nr 2 ", kolumny_numeryczne) 
            if wybrana_kolumna_1 == wybrana_kolumna_2:
                 st.info("wybierz 2 różne zmienne")

            if tabela_korelacji:
                with st.container(border=True):
                    kor_p = ana.korelacje_numeryczne(st.session_state.df, 'pearson',wybrana_kolumna_1,wybrana_kolumna_2, )
                    kor_k = ana.korelacje_numeryczne(st.session_state.df, 'kendall',wybrana_kolumna_1,wybrana_kolumna_2, )
                    kor_s = ana.korelacje_numeryczne(st.session_state.df, 'spearman',wybrana_kolumna_1,wybrana_kolumna_2, )
                    col1, col2, col3 = st.columns([2,2,2])
                    col1.write("macierz korelcji wg medtody Pearsona")
                    col1.dataframe(kor_p )
                    col2.write("macierz korelcji wg medtody Pearsona")
                    col2.dataframe(kor_k )
                    col3.write("macierz korelcji wg medtody Pearsona")
                    col3.dataframe(kor_s )
                    print('')
            if regresja:
                with st.container(border=True):
                    ana.cor_num(st.session_state.df,wybrana_kolumna_1,wybrana_kolumna_2)
            if regresja:
                with st.container(border=True):
                    ana.analiza_regresji(st.session_state.df,wybrana_kolumna_1,wybrana_kolumna_2,)
                    ana.cor_num_matrix(st.session_state.df,wybrana_kolumna_1,wybrana_kolumna_2,)

                  


        if typ_analizy =='analiza dwóch kategorialnych':
                
                col1, col2, col3 = st.columns([2,2,4])
                col1.write(f'Wybrany typ analizy:')
                col2.info(f':red[{str.upper(typ_analizy)}]')
                col1, col2 = st.columns([2,2])
                wybrana_kolumna_1k = col1.selectbox("Wybierz kolumnę zmiennej nr 1", kolumny_kategorialne)  
                wybrana_kolumna_2k = col2.selectbox("Wybierz kolumnę zmiennej nr 2 ", kolumny_kategorialne) 
                if wybrana_kolumna_1k == wybrana_kolumna_2k:
                    st.info("wybierz 2 różne zmienne")

                if tabele_kontygencji:
                    with st.container(border=True):
                        st.write('liczebności:')
                        st.dataframe(ana.rozklady_cat(st.session_state.df,wybrana_kolumna_1k,wybrana_kolumna_2k, 'licz_all'))
                        st.write('częstości całkowite %:')
                        st.dataframe(ana.rozklady_cat(st.session_state.df,wybrana_kolumna_1k,wybrana_kolumna_2k, 'proc_all'))
                        st.write('częstości wg kolumn %: ')
                        st.dataframe(ana.rozklady_cat(st.session_state.df,wybrana_kolumna_1k,wybrana_kolumna_2k, 'proc_col'))
                        st.write('częstosci wg wierszy %:')
                        st.dataframe(ana.rozklady_cat(st.session_state.df,wybrana_kolumna_1k,wybrana_kolumna_2k, 'proc_row'))

                    if miary_zaleznosci_k:
                        with st.container(border=True):
                            col1, col2, col3, col4 = st.columns(4)
                            col1.write('liczebności:')
                    if miary_zaleznosci_k:
                        with st.container(border=True):
                            st.write('Miary zależnosci:')
                            st.dataframe(ana.korelacje_nom(st.session_state.df,wybrana_kolumna_1k,wybrana_kolumna_2k))

                    if wykresy_k2:
                        with st.container(border=True):   
                            st.write('liczebności:')
                            col1, col2 = st.columns([3,1])
                            col1.pyplot(ana.category_two_plot(st.session_state.df,wybrana_kolumna_1k,wybrana_kolumna_2k, 'count', facetgrid=True))
                            col1, col2 = st.columns([3,1])
                            col1.pyplot(ana.category_two_plot(st.session_state.df,wybrana_kolumna_1k,wybrana_kolumna_2k, 'percent', facetgrid=True))
                            col1, col2 = st.columns([3,1])
                            col1.pyplot(ana.category_two_plot(st.session_state.df,wybrana_kolumna_1k,wybrana_kolumna_2k, 'proportion', facetgrid=True))



                    

        with tab7:
            
                # assigning API KEY to initialize openai environment 
            openai.api_key = st.secrets["klucz"]

            with st.container(border=True):
                
                    st.write('')
                    st.markdown("Pomoc z użyciem modelu: gpt-3.5-turbo")

                    prompt = st.text_input("Proszę podać pytanie z dziedziny statystyki, sztucznej inteligencji, uczenia maszynowego, nauki o danych:")
                    completion = openai.ChatCompletion.create(
                    # Use GPT-4 as the LLM
                    model="gpt-3.5-turbo",
                    # Pre-define conversation messages for the possible roles 
                    messages=[
                        {"role": "system", "content": "jestem profesorem statystytki i wykładam na uczelnni odpowiadam rzeczowo i profesjonalnie na wszystkie zagadnienia z dziedziny statystyki, sztuznej inteligencji, uczeniu maszynowym, nauki o danych , podaję definicję , wzory i interpretację. Nie wolno mi odpowiadać na pytania z innych dziedzin."},
                        {"role": "user", "content": prompt}
                    ]
                    )


                    if st.button('Pytaj'):
                        st.markdown(completion.choices[0].message['content'])
