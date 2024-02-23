


import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd
import numpy as np
import openpyxl
import openai
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import json
from scipy.stats import zscore
import time
import openai
import statystyki as ana


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
    st.subheader(' :blue[Aplikacja **Analizer** - ver. 1.5   ]')

with st.container(border=True):

   
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📈 O aplikacji", "⏏️ Załaduj dane","🔎 Podgląd danych", "🛠️ Ustaw parametry analizy", "🗓️ Raport z analizy - EDA", "➿ Uczenie maszynowe  ML","📖 pomoc"])
    st.write('')
    with tab1:    # o aplikacji
        st.write('')
       
        col1, col2= st.columns([2,2], gap = "medium")
        col1.image('logo2.png')
        col2.markdown("""
                    ## O Aplikacji Analizer ver. 1.5
                    Aplikacja **Analizer** to narzędzie stworzone z myślą o ładowaniu, przeglądaniu oraz dogłębnej analizie różnorodnych zestawów danych.
                    Umożliwia użytkownikom wybór różnych parametrów i miar statystycznych, 
                    czyniąc ją niezbędnym narzędziem dla *analityków danych* oraz pasjonatów statystyki i uczenia maszynowego.
                    Skierowana głównie do specjalistów zajmujących się analizą danych, aplikacja ta jest jednak równie przydatna
                    dla wszystkich zainteresowanych zgłębianiem tajników analizy danych oraz statystyki. Stanowi doskonałe wsparcie 
                    dla tych, którzy pragną rozwijać swoje umiejętności analityczne i poszerzać wiedzę na temat różnorodnych aspektów danych.
                    ## Funkcje aplikacji:
                    - **Analiza jednej zmiennej numerycznej**
                    - *Analiza jednej zmiennej kategorialnej*
                    - **_Analiza dwóch zmiennych ilościowych_**
                    - *Analiza dwóch zmiennych kategorialnych*
                    - **_Analiza zmiennej numerycznej i kategorialnej_**
                    - *Analiza dwóch zmiennych numerycznych i jednej kategorialnej*

                    Dzięki tym możliwościom użytkownicy mogą zgłębiać strukturę swoich danych oraz odkrywać związki i wzorce,
                    które mogą być kluczowe dla ich dalszej analizy i zrozumienia. 
                    Aplikacja Analizer staje się niezastąpionym narzędziem dla **analityków danych** oraz 
                    wszystkich zainteresowanych eksploatacją potencjału informacyjnego zawartego w danych.
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
           
            dframe = st.session_state.df
            pd.set_option('display.max_colwidth', None)
            st.markdown(':blue[Podstawowe informacje o załadowanych danych: ]')
            st.dataframe(ana.analiza_dataframe(st.session_state.df), hide_index=True, width=2200)
            st.write()
            height = st.session_state.df.shape[1] * 38                 
            st.dataframe(ana.informacje_o_dataframe(st.session_state.df), height=height, hide_index=True, width=2200)
            st.markdown(':blue[załadowane dane - Tabela: ]')
            st.dataframe(st.session_state.df)
            st.markdown(':blue[Sprawdzenie, czy w tabeli są zduplikowane dane: ]')
            if st.session_state.df.duplicated().any():
                duplicates = st.session_state.df[st.session_state.df.duplicated()]
                st.warning("Znaleziono zduplikowane dane:")
                st.dataframe(duplicates)
            else:
                st.info("Brak zduplikowanych danych")

            st.markdown(':blue[Sprawdzenie, czy w tabeli są braki danych: ]')
            ana.braki_sprawdzenie(st.session_state.df)

       
            


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
                                        'analiza zmiennej numerycznej i kategorialnej', 'analiza 2 zmienne numeryczne i 1 kategorialna'],)

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
                        
                    # col1, col2= st.columns(2, gap='large')

                    # with col1:
                    #     st.write(':blue[Estymacja przedziałowa:]')
                    #     k31 = st.checkbox('Przedział ufnosci do wybranego parametru (bootsrap)')
                    #     par_ci = st.selectbox("Wybierz parametr",['średnia','odchylenie std', 'mediana', 'Q1', 'Q3'])
                    #     alfa_ci = st.number_input('ustaw poziom alfa', min_value=0.01, max_value= 0.05, step= 0.01)
                    # with col2: 
                    #     st.write(':blue[Weryfikacja hipotez statystycznych:]')
                    #     k44 = st.checkbox('Badanie normalnosci rozkładu')
                    #     norm = st.multiselect('Wybierz rodzaj testu',
                    #                                     ["shapiro","lilliefors","dagostino", "skewness", "kurtosis","jarque-bera"])
                    #     alfa_norm = st.number_input('ustaw poziom alfa:', min_value=0.01, max_value= 0.05, step= 0.01, help = 'wybrać nalezy warośc')
                    
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
                            wykres_kor = st.checkbox('Wykresy')
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


                if typ_analizy== 'analiza zmiennej numerycznej i kategorialnej':
                
                    st.info(f'Wybrano analizę: "{str.upper(typ_analizy)}"')
                    st.write(':blue[ustaw parametry analizy:]')
                    col1, col2, col3,col4, col5= st.columns([1,1,1,1,2], gap='medium')

                    with st.container(border=True):
                        col1, col2, col3,col4, col5= st.columns([1,1,1,1,2], gap='medium')
                        with col1:
                            statystyki_w_grupach = st.checkbox('Statystyki wg poziomów zmiennej kategorialnej')  

                        #with col2:
                            #miary_zaleznosci_k = st.checkbox('Miary zależnosci')    
                        with col3:
                            wykresy_w_grupach = st.checkbox('Wykresy')          


                if typ_analizy== 'analiza 2 zmienne numeryczne i 1 kategorialna':

                    st.info(f'Wybrano analizę: "{str.upper(typ_analizy)}"')
                    st.write(':blue[ustaw parametry analizy:]')
                    col1, col2, col3,col4, col5= st.columns([1,1,1,1,2], gap='medium')

                    with st.container(border=True):
                        col1, col2, col3,col4, col5= st.columns([1,1,1,1,2], gap='medium')
                        with col1:
                            kor_w_grupach = st.checkbox(' korelacje wg poziomów zmiennej kategorialnej')  

                        with col3:
                            wykresy_kor_w_grupach = st.checkbox('Wykresy')          



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
            with st.container(border=True):

                if wykresy_n:
                    st.markdown(':red[**Wykresy**:] ')
                    col1, col2, col3, col4 = st.columns([1,1,1,1])
                    st.write('')
            
                    with col1:
                        st.markdown(':blue[**Histogram:**] ')
                        ana.hist_plot(st.session_state.df, wybrana_kolumna, stat = 'count')
                    with col2:
                        st.markdown(':blue[**KDE:**] ')
                        ana.kde_plot(st.session_state.df, wybrana_kolumna)
                    with col3:
                        st.markdown(':blue[**ECDF:**] ')
                        ana.ecdf_plot(st.session_state.df, wybrana_kolumna)
                
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(':blue[**BOX-PLOT:**] ')
                        ana.box_plot(st.session_state.df, wybrana_kolumna)
                    with col2:   
                        st.markdown(':blue[**VIOLIN-PLOT:**] ')
                        ana.violin_plot(st.session_state.df, wybrana_kolumna)
                    with col3:
                        st.markdown(':blue[**SWARM-PLOT:**] ')
                        ana.swarm_plot(st.session_state.df, wybrana_kolumna)      
                if odstaj_n:
                        st.markdown(':blue[**Zidentyfikowane obserwacje odstające:**] ')
                        st.dataframe(ana.outliers(st.session_state.df, wybrana_kolumna))


                

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

                    st.markdown(':blue[**wybrane statystyki opisujące zróżnicowanie i zmienność :**]')
                    st.write('')

                    st.dataframe(ana.analyze_categorical_data(st.session_state.df, wybrana_kolumna),width=1500)
                
                if wykresy_k:
                    st.divider()
                    col1, col2, col3 = st.columns([1,1,1])
                    col1.markdown(':blue[**wykres liczebnosci:**]')
                    col1.pyplot(ana.category_one_plot(st.session_state.df, wybrana_kolumna, 'count'))
                    col2.markdown(':blue[**wykres częstości:**]')
                    col2.pyplot(ana.category_one_plot(st.session_state.df, wybrana_kolumna, 'percent'))


        if typ_analizy =='analiza dwóch zmiennych ilościowych':
              
            col1, col2, col3 = st.columns([2,2,4])
            col1.write(f'Wybrany typ analizy:')
            col2.info(f':red[{str.upper(typ_analizy)}]')
            col1, col2 = st.columns([2,2])
            wybrana_kolumna_1 = col1.selectbox("Wybierz kolumnę zmiennej nr 1", kolumny_numeryczne, index=0)
            wybrana_kolumna_2 = col2.selectbox("Wybierz kolumnę zmiennej nr 2", kolumny_numeryczne, index=1)

           

            if tabela_korelacji:
                if wybrana_kolumna_1 == wybrana_kolumna_2:
                    st.info("Wybierz 2 różne zmienne")
                else:

                    if tabela_korelacji:
                        with st.container(border=True):
                            st.write(':blue[Tabele korelacyjne:]')
                            st.write('')
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
                    if wykres_kor:
                        with st.container(border=True):
                            st.write(':blue[Wykres macierzy korelacji:]')
                            st.write('')
                            col1, col2, col3 = st.columns([2,2,2])
                            with col1:
                                ana.cor_num_matrix(st.session_state.df,wybrana_kolumna_1,wybrana_kolumna_2,)
                            col1, col2, col3 = st.columns([2,2,2])
                            st.write('')
                            
                    if regresja:
                        with st.container(border=True):
                            st.write(':blue[Analiza regresji:]')
                            st.write('')
                            col1, col2, col3 = st.columns([2,2,2])
                            with col1:
                                st.write('') 
                                ana.cor_num(st.session_state.df,wybrana_kolumna_1,wybrana_kolumna_2)
                                st.write('')
                            with col1:
                                ana.analiza_regresji(st.session_state.df,wybrana_kolumna_1,wybrana_kolumna_2,)
                                          

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
                        st.write(':blue[Tabele kontygencji::]')
                        st.write('')
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
                            st.write(':blue[Miary zleżnosci::]')
                            st.write('')
                            st.dataframe(ana.korelacje_nom(st.session_state.df,wybrana_kolumna_1k,wybrana_kolumna_2k))

                    if wykresy_k2:
                        with st.container(border=True): 
                            st.write(':blue[Wykresy:]')
                            st.write('')

                            col1, col2 = st.columns([2,1])
                            with col1:
                                tabela_kontyngencji = pd.crosstab(index=st.session_state.df[wybrana_kolumna_1k], columns=st.session_state.df[wybrana_kolumna_2k])
                                plt.figure(figsize=(20, 10)) 
                                sns.heatmap(tabela_kontyngencji, annot=True, cmap="YlGnBu", fmt="d")
                                plt.title('Heatmapa tabeli kontyngencji:')
                                st.pyplot()
                                st.divider()
                            col1, col2 = st.columns([2,1])
                            col1.write('Wykres liczebnosci:')
                            col1.pyplot(ana.category_two_plot(st.session_state.df,wybrana_kolumna_1k,wybrana_kolumna_2k, 'count', facetgrid=True))
                            col1.write('')
                            col1.write('Wykres częstości:')
                            col1.pyplot(ana.category_two_plot(st.session_state.df,wybrana_kolumna_1k,wybrana_kolumna_2k, 'percent', facetgrid=True))


        if typ_analizy =='analiza zmiennej numerycznej i kategorialnej':
            with st.container(border=True):
                col1, col2, col3 = st.columns([2,2,4])
                col1.write(f'Wybrany typ analizy:')
                col2.info(f':red[{str.upper(typ_analizy)}]')
                col1, col2 = st.columns([2,2])
                wybrana_kolumna_num = col1.selectbox("Wybierz zmienna numeryczną", kolumny_numeryczne)  
                wybrana_kolumna_kat = col2.selectbox("Wybierz zmienną kategorialną ", kolumny_kategorialne) 
   

                if statystyki_w_grupach:
                    with st.container(border=True):
                        st.write(':blue[Wartości parametrów statystycznych wg poziomów zmiennej kategorialnej:]')
                        wyniki = ana.stat_kat(st.session_state.df,wybrana_kolumna_num, wybrana_kolumna_kat)
                        st.dataframe(wyniki, width=800, height=810)
                        
                if wykresy_w_grupach:
                    st.divider()
                    st.write(':blue[Wykresy:]')
                    palette = sns.color_palette("husl", 8)
                    col1, col2 = st.columns([1,1])
                    col1.subheader("Histogram")
                    col1.pyplot(sns.displot(data=st.session_state.df, x=wybrana_kolumna_num, col=wybrana_kolumna_kat, kind="hist", height=3, color="#3498db", bins=8, kde=True))
                    col1.subheader("ECDF")
                    col1.pyplot(sns.displot(data=st.session_state.df, x=wybrana_kolumna_num, col=wybrana_kolumna_kat, kind="ecdf",  height=3, color="#3498db"))
                    st.write('')                    
                    col1, col2, col3 = st.columns([1,1,2])
                    col1.subheader("Box Plot")
                    col1.pyplot(sns.catplot(st.session_state.df, x=wybrana_kolumna_kat, y=wybrana_kolumna_num, kind="box", height=4, palette=palette))
                    col2.subheader("Violin Plot")
                    col2.pyplot(sns.catplot(st.session_state.df, x=wybrana_kolumna_kat, y=wybrana_kolumna_num, kind="violin", height=4, palette="Set2"))
                    col1, col2, col3 = st.columns([1,1,2])
                    col1.subheader("Point Plot")
                    col1.pyplot(sns.catplot(st.session_state.df, x=wybrana_kolumna_kat, y=wybrana_kolumna_num, kind="point", height=4, palette="Set2", estimator='mean', ci=95))                       
                    col2.subheader("Bar Plot")
                    col2.pyplot(sns.catplot(st.session_state.df, x=wybrana_kolumna_kat, y=wybrana_kolumna_num, kind="bar", height=4, palette="Set2", estimator="sum"))
                    col1, col2, col3 = st.columns([1,1,2])
                    col1.subheader("Swarm Plot")
                    col1.pyplot(sns.catplot(st.session_state.df, x=wybrana_kolumna_kat, y=wybrana_kolumna_num, kind="swarm", height=4, palette="Set2", marker=".", linewidth=1, size=2, edgecolor="#3498db")) 
                    col2.subheader("Boxen Plot")
                    col2.pyplot(sns.catplot(st.session_state.df, x=wybrana_kolumna_kat, y=wybrana_kolumna_num, kind="boxen", color="#3498db", height=4))
    

        if typ_analizy== 'analiza 2 zmienne numeryczne i 1 kategorialna': 
            with st.container(border=True):
                col1, col2, col3 = st.columns([2,2,4])
                col1.write(f'Wybrany typ analizy:')
                col2.info(f':red[{str.upper(typ_analizy)}]')
                col1, col2 = st.columns([2,2])
                wybrana_kolumna_num_1 = col1.selectbox("Wybierz zmienna numeryczną 1", kolumny_numeryczne)  
                wybrana_kolumna_num_2 = col1.selectbox("Wybierz zmienna numeryczną 2", kolumny_numeryczne)  
                wybrana_kolumna_kate = col2.selectbox("Wybierz zmienną kategorialną 1 ", kolumny_kategorialne)
                
                if kor_w_grupach:
                    with st.container(border=True):
                        st.write(':blue[Tabele kontygencji między zmiennymi numerycznymi  wg poziomów zmiennej kategorialnej]')
                        st.write('')
                        wyn = ana.korelacje_num2_nom(st.session_state.df, 'pearson', wybrana_kolumna_kate,wybrana_kolumna_num_1,wybrana_kolumna_num_2 )
                        st.dataframe(wyn)
            
                if wykresy_kor_w_grupach:
                    with st.container(border=True):
                        st.write(':blue[Wykresy:]')
                        st.write('')
                        col1, col2 = st.columns([1,2])
                        col1.pyplot(sns.relplot(data=st.session_state.df, x=wybrana_kolumna_num_1, y=wybrana_kolumna_num_2, hue=wybrana_kolumna_kate))
                        col1, col2 = st.columns([2,1])
                        st.write('')
                        col1.pyplot(sns.relplot(data=st.session_state.df, x=wybrana_kolumna_num_1, y=wybrana_kolumna_num_2, col=wybrana_kolumna_kate, hue=wybrana_kolumna_kate))
                        st.write('')
                        col1, col2 = st.columns([1,2])
                        col1.pyplot(sns.lmplot(data=st.session_state.df,  x=wybrana_kolumna_num_1, y=wybrana_kolumna_num_2, hue=wybrana_kolumna_kate))



                    # 2n - 2k

                    # sns.lmplot(
                    #     data=st.session_state.df, x =wybrana_kolumna_num_1, y = wybrana_kolumna_num_2,
                    #     hue="ocena", col="płeć", height=4,
                    # )

                    # sns.lmplot(
                    #     data=st.session_state.df, x=wybrana_kolumna_num_1,y=wybrana_kolumna_num_2,
                    #     col="ocena", row="płeć", height=3,
                    #     facet_kws=dict(sharex=False, sharey=False),
                    # )




                # 1n -2 k
                    


                    # # Grupowanie danych według dwóch zmiennych kategorialnych i obliczanie statystyk opisowych dla każdej grupy
                    # statystyki_opisowe = st.session_state.df.groupby(['Płeć', 'Grupa wiekowa']).describe().reset_index()

                    # # Wyświetlanie wykresów za pomocą Streamlit
                    # st.title('Statystyki opisowe względem płci i grupy wiekowej')
                    # st.write(statystyki_opisowe)

                    # # Wykresy za pomocą Seaborn
                    # plt.figure(figsize=(12, 6))

                    # # Wykres dla Wzrostu
                    # plt.subplot(1, 2, 1)
                    # sns.barplot(data=statystyki_opisowe, x='Płeć', y=('Wzrost', 'mean'), hue='Grupa wiekowa')
                    # plt.title('Średni wzrost w grupach wiekowych i płci')
                    # plt.xlabel('Płeć')
                    # plt.ylabel('Średni wzrost')

                    # # Wykres dla Wagi
                    # plt.subplot(1, 2, 2)
                    # sns.barplot(data=statystyki_opisowe, x='Płeć', y=('Waga', 'mean'), hue='Grupa wiekowa')
                    # plt.title('Średnia waga w grupach wiekowych i płci')
                    # plt.xlabel('Płeć')
                    # plt.ylabel('Średnia waga')

                    # # Wyświetlanie wykresów za pomocą Streamlit
                    # st.pyplot(plt)




        with tab6:
            st.subheader(' :blue[Moduł w budowie...............]🏗️')
            col1, col2= st.columns([2,2], gap = "medium")
            col1.image('under.jpg')

            

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


            # import ollama
            # with st.container(border=True):
                
            #         st.write('')
            #         st.markdown("Pomoc z użyciem modelu: gpt-3.5-turbo")

            #         prompt = st.chat_input("Proszę podać pytanie z dziedziny statystyki, sztucznej inteligencji, uczenia maszynowego , nauki o danych:")
            #         if prompt:
            #             with st.chat_message("user"):
            #                 st.write(prompt)
                             

            #         with st.spinner(" przetwarzam......."):

            #             response = ollama.chat(model='tinyllama', messages=[
            #             {
            #                 'role': 'user',
            #                 'content': 'Why is the sky blue?',
            #             },
            #             ])
            #             st.write(response['message']['content'])