


import warnings
from sklearn.metrics import ConfusionMatrixDisplay
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
import pingouin as pg
import openai
import scipy.stats as stats

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
            #"test": test,
            "sklepy": sklep,
            "szkoła" : szkoła,
            #"iris": iris,
            #"napiwki": napiwiki
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
            
            col1, col2= st.columns([1,2])
            with col1:
                with st.container(border=True):
   
                    st.write('')


                    typ_analizy = st.radio(':blue[Wybierz typ analizy danych : ]',
                                       ['analiza jednej zmiennej numerycznej', 'analiza jednej zmiennej kategorialnej', 'analiza dwóch zmiennych ilościowych', 'analiza dwóch kategorialnych', 
                                        'analiza zmiennej numerycznej i kategorialnej', 'analiza 2 zmienne numeryczne i 1 kategorialna'])
                    st.write('')
                                        #'Przedziały ufnosci do wybranych parametrów', 
                                        #'weryfikacja założeń testów statystycznych', 'Testy parametryczne  -  porównania 1 grupa', 
                                        #'Testy parametryczne  - porównania 2 grupy','Testy parametryczne  - porównania >2 grupy'],)
            
            with col2:
                with st.container(border=True):
                    st.write('')
                    if typ_analizy== 'analiza jednej zmiennej numerycznej':
                            st.write(f':blue[ustaw parametry analizy:  { (typ_analizy)}] ')

                            col1, col2, col3= st.columns([1,1,1], gap='medium')
                            with col1:
                                tabela_licz_n = st.checkbox('tabela liczebności i częstości')
                            with col1:
                                wykresy_n = st.checkbox('Wykresy')
                            with col1:
                                odstaj_n = st.checkbox('Obserwacje odstające')
                            with col1:
                                statystyki_n = st.checkbox('Miary statystyczne')
                            with col2:
                                ci_srednia = st.checkbox('przedział ufnosci do średniej') 
                                ci_mediana = st.checkbox('przedział ufnosci do mediany')
                                ci_std = st.checkbox('przedział ufnoci do odchylenia standardowego')
                                ci_q1 = st.checkbox('przedział ufnoci do Q1 i Q3')
                            with col3:    
                                wer_norm = st.checkbox('testowanie normalnosci rozkładu')
                    st.write('')

                        
                    st.set_option('deprecation.showPyplotGlobalUse', False)   
                    if typ_analizy== 'analiza jednej zmiennej kategorialnej':

                        st.write(f':blue[ustaw parametry analizy:  { (typ_analizy)}] ')
                        col1, col2, col3= st.columns([1,1,1], gap='medium')
                        with col1:
                            tabela_licz_k = st.checkbox('tabela liczebności i częstości')
                        with col2:
                            wykresy_k = st.checkbox('Wykresy')
                        with col3:
                            statystyki_k = st.checkbox('Miary statystyczne')
                        st.write('')


                    if typ_analizy== 'analiza dwóch zmiennych ilościowych':
                    
                       
                        st.write(f':blue[ustaw parametry analizy:  { (typ_analizy)}] ')

                        col1, col2, col3,col4, col5= st.columns([1,1,1,1,2], gap='medium')
                        with col1:
                                tabela_korelacji = st.checkbox('tabela  korelacji')  
                        with col2:
                                wykres_kor = st.checkbox('Wykresy')
                        with col3:
                                regresja = st.checkbox('regresja liniowa')
                        st.write('')
                            
                            

                    if typ_analizy== 'analiza dwóch kategorialnych':
                
                        st.write(f':blue[ustaw parametry analizy:  { (typ_analizy)}] ')
                        col1, col2, col3= st.columns([1,1,1], gap='medium')
                        with col1:
                            tabele_kontygencji = st.checkbox('tabele kontygencji')  
                        with col2:
                            miary_zaleznosci_k = st.checkbox('Miary zależnosci')    
                        with col3:
                            wykresy_k2 = st.checkbox('Wykresy')



                    if typ_analizy== 'analiza zmiennej numerycznej i kategorialnej':
                    
                       
                        st.write(f':blue[ustaw parametry analizy:  { (typ_analizy)}] ')
                        col1, col2, col3= st.columns([1,1,1], gap='medium')

                        with col1:
                            statystyki_w_grupach = st.checkbox('Statystyki wg poziomów zmiennej kategorialnej')  
  
                        with col3:
                            wykresy_w_grupach = st.checkbox('Wykresy')          




                    if typ_analizy== 'analiza 2 zmienne numeryczne i 1 kategorialna':

                        st.write(f':blue[ustaw parametry analizy:  { (typ_analizy)}] ')
                        col1, col2, col3= st.columns([1,1,1], gap='medium')
                        with col1:
                            kor_w_grupach = st.checkbox(' korelacje wg poziomów zmiennej kategorialnej')  
                        with col3:
                            wykresy_kor_w_grupach = st.checkbox('Wykresy')          
                
             
                            
                        
                    if typ_analizy =='weryfikacja założeń testów statystycznych':
                    
                        st.write(f':blue[ustaw parametry analizy:  { (typ_analizy)}] ')
                        col1, col2 =  st.columns([1,1],)
                        with col1:
                            #wer_norm = st.checkbox('testowanie normalnosci rozkładu') 
                            wer_jed_warian = st.checkbox('testowanie jednorodnosci wariancji -   2 grupy')
                        with col2:
                            wer_jed_warian2 = st.checkbox('testowanie jednorodnosci wariancji - > 2 grupy')
                            wer_rowne_grupy = st.checkbox('równoliczność grup')

                        
                    if typ_analizy == 'Testy parametryczne  -  porównania 1 grupa':
                    
                        st.write(f':blue[ustaw parametry analizy:  { (typ_analizy)}] ')
                        col1, col2 =  st.columns([1,1],)
                        with col1:
                            testy_jeden_t = st.checkbox('Test t-Studenta dla pojedynczej próby')
                            testy_jeden_wilk = st.checkbox('Test Wilcoxona (rangowanych znaków')
                        with col2:
                            testy_jeden_chi = st.checkbox('Test chi-kwadrat wariancji pojedynczej próby')
                            testy_jeden_prop = st.checkbox('Testy dla jednej proporcji')
                            
                    if typ_analizy == 'Testy parametryczne  - porównania 2 grupy':
                    
                        st.write(f':blue[ustaw parametry analizy:  { (typ_analizy)}] ')
                        col1, col2 =  st.columns([1,1],)
                        with col1:
                            testy_dwa_t = st.checkbox('Test t-Studenta dla  2 grup niezależnych', help = 'wariancje badanych zmiennych w obu populacjach są równe')
                            testy_dwa_t_coch= st.checkbox('Test t-Studenta  dla  2 grup niezależnych z korektą Cochrana-Coxa', help = 'wariancje badanych zmiennych w obu populacjach są różne')

                        with col2:
                            testy_jeden_U = st.checkbox('Test U Manna-Whitneya', help = 'brak normalnosci rozkładu')
                            testy_dwa_prop = st.checkbox('Testy dla dwóch proporcji')
                            testy_dwa_chi = st.checkbox('Testy chi2' ,help = 'testy 2x2, RxC')

                    if typ_analizy == 'Testy parametryczne  - porównania >2 grupy':
                    
                        st.write(f':blue[ustaw parametry analizy:  { (typ_analizy)}] ')
                        col1, col2 =  st.columns([1,1],)
                        with col1:
                            anowa = st.checkbox('ANOVA', help = 'normalność rozkładu + równość wariancji')
                            anowa_f =  st.checkbox('ANOVA dla grup niezależnych z korektą F* i F', help = 'normalność rozkładu + różne wariancje')

                        with col2:
                            testy_jeden_U = st.checkbox('ANOVA Kruskala-Wallisa', help = 'brak normalnosci rozkładu')



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
                        st.dataframe(ana.stat(st.session_state.df, [wybrana_kolumna]).T, height=808, width=350)

                        # statystyki_a = ana.stat(st.session_state.df, [wybrana_kolumna])
                        # statystyki_a = statystyki_a[['liczba','suma','min','max','średnia', 'rozstęp', 'p_10%', 'Q1_25%','Q2_50%', 'Q3_75%', 'p_90%']]
                        # statystyki_b = ana.stat(st.session_state.df, [wybrana_kolumna])
                        # statystyki_b = statystyki_b[['IQR','odch_cwiar','odchylenie przeciętne','wariancja','odch_std','błąd_odch_std','kl_wsp_zmien', 'poz_wsp_zmien', 'moda', 'skośność', 'kurtoza']]
                        # st.dataframe(statystyki_a.T, width=1800, )
                        # st.dataframe(statystyki_b.T, width=1800)
            
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
                    with st.container(border=True):
                        
                        st.markdown(':blue[**Zidentyfikowane obserwacje odstające:**] ')
                        st.dataframe(ana.outliers(st.session_state.df, wybrana_kolumna))
                        
                        
                if wer_norm :
                    with st.container(border=True):
                        st.write(':blue[**Weryfikacja hipotez statystycznych:**]')
                        #wybrana_kolumna2 = st.selectbox("Wybierz kolumnę  ", kolumny_numeryczne)
                        col1, col2 , col3= st.columns([1,2,3], gap = 'large')
                        with col1:
                            alpha = st.slider('Określ poziom alpha dla testu:',0.9,0.99, step = 0.01, value = 0.95)
                        st. write('')
                        with col2:
                            norm = st.multiselect('Wybierz rodzaj testu',
                                                                ["shapiro","lilliefors","dagostino", "skewness", "kurtosis","jarque-bera"])
                        st.dataframe(ana.testy_normalnosci_jeden(st.session_state.df, wybrana_kolumna,  wybrane_testy=norm ,alpha=alpha))
                    
                  
                    
                with st.container(border=True):
                        st.write(':blue[**Estymacja przedziałowa:**]')
                        col1, col2 , col3= st.columns([1,1,3], gap = 'large')
                        with col1:
                            alpha = st.slider('Określ poziom alpha dla testu:  ',0.9,0.99, step = 0.01, value = 0.95)
                        with col2:   
                            boot  = st.number_input('Określ liczbę n_bootstraps: ', value =100)                             
                        if ci_srednia:
                            wyn = ana.bootstrap_ci(st.session_state.df, wybrana_kolumna , 'średnia', alpha=alpha, n_bootstraps=boot)
                            st.write(':red[przedział ufnosci do średniej arytmetycznej:]')
                            st.dataframe(wyn)
                        if ci_mediana:
                            wyn = ana.bootstrap_ci(st.session_state.df, wybrana_kolumna , 'mediana', alpha=alpha, n_bootstraps=boot)
                            st.write(':red[przedział ufnosci do mediany:]')
                            st.dataframe(wyn)
                        if ci_std:
                            wyn = ana.bootstrap_ci(st.session_state.df, wybrana_kolumna , 'odchylenie', alpha=alpha, n_bootstraps=boot)
                            st.write(':red[przedział ufnosci do odchylenia standardowego:]')
                            st.dataframe(wyn)
                        if ci_q1:
                            wyn = ana.bootstrap_ci(st.session_state.df, wybrana_kolumna , 'q25', alpha=alpha, n_bootstraps=boot)
                            st.write(':red[przedział ufnosci do Q1 [.025]:]')
                            st.dataframe(wyn)   
                            wyn3 = ana.bootstrap_ci(st.session_state.df, wybrana_kolumna , 'q75', alpha=alpha, n_bootstraps=boot)
                            st.write(':red[przedział ufnosci do Q3 [.075]:]')
                            st.dataframe(wyn)   
                     
         
                          

                

            with st.container(border=True):pass
                #st.write('')
                #st.markdown("Dokonaj interpretacji wyliczonych miar statystycznych z uzycliem modelu: gpt-3.5-turbo")
                # interpretacja  = st.button('Interpretuj')
                # if interpretacja :

                #     miary = (ana.stat(st.session_state.df, [wybrana_kolumna]).T)
                #     json_data = miary.to_json()
                    
                #     prompt = """
                #     System: Witam! Jestem pomocnym asystentem statystycznym. Jak mogę Ci dzisiaj pomóc?
                #     Użytkownik: Chciałbym uzyskać interpretację statystyczną dla następujących miar obliczonych dla mojego Dataframe:
                #     {json_data}
                #     System: Na podstawie dostarczonych informacji dokonaj opisu dataframe i dokonaj interpretacji wartości statystyk:
                #     """
                #     from openai import OpenAI
                #     api_key = st.secrets["klucz"]
    
                #     client = OpenAI(api_key=api_key)

    
                #     # Definicja komunikatów systemowego i użytkownika
                #     komunikat_systemowy = {"role": "system", "content": "Jesteś statystykiem, który ma w prosty sposób tłumaczyć i wyjaśniać co oznaczają wartości wyliczonych parametrów statystycznych"}
                #     komunikat_uzytkownika = {"role": "user", "content": prompt.format(json_data=json_data)}
                    
                #     # Tworzenie zapytania do modelu
                #     completion = openai.ChatCompletion.create(
                #         model="gpt-3.5-turbo",
                #         messages=[komunikat_systemowy, komunikat_uzytkownika]
                #     )
                    
                #     # Pobieranie odpowiedzi
                #     odpowiedz_LLM = completion.choices[0].message['content']



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
                            col1, col2= st.columns([1,1], gap = 'large')
                            col1.markdown(':blue[**wykres liczebnosci:**]')
                            col1.pyplot(ana.category_one_plot(st.session_state.df, wybrana_kolumna, 'count'))
                            col2.markdown(':blue[**wykres częstości % :**]')
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



        if typ_analizy =='Przedziały ufnosci do wybranych parametrów':
            with st.container(border=True):
                col1, col2, col3 = st.columns([2,2,4])
                col1.write(f'Wybrany typ analizy:')
                col2.info(f':red[{str.upper(typ_analizy)}]')
                col1, col2 = st.columns([2,2])
                
            with st.container(border=True):  
                st.write(':blue[przedział ufnosci do wyliczonych parametrów statystycznych]')
                st.write('')
                col1, col2, col3 = st.columns([2,1,1], gap = 'large')
                
                with col1: wybrana_kolumna = st.selectbox("Wybierz kolumnę", kolumny_numeryczne) 
                with col2: alpha = st.slider('Określ poziom alpha:',0.9,0.99, step = 0.01, value = 0.95)
                with col3: boot  = st.number_input('Określ liczbę n_bootstraps: ', value =100)
                st. write('')
                
                if ci_srednia:
                    wyn = ana.bootstrap_ci(st.session_state.df, wybrana_kolumna , 'średnia', alpha=alpha, n_bootstraps=boot)
                    st.write(':red[przedział ufnosci do średniej arytmetycznej:]')
                    st.dataframe(wyn)
                if ci_mediana:
                    wyn = ana.bootstrap_ci(st.session_state.df, wybrana_kolumna , 'mediana', alpha=alpha, n_bootstraps=boot)
                    st.write(':red[przedział ufnosci do mediany:]')
                    st.dataframe(wyn)
                if ci_std:
                    wyn = ana.bootstrap_ci(st.session_state.df, wybrana_kolumna , 'odchylenie', alpha=alpha, n_bootstraps=boot)
                    st.write(':red[przedział ufnosci do odchylenia standardowego:]')
                    st.dataframe(wyn)
                if ci_q1:
                    wyn = ana.bootstrap_ci(st.session_state.df, wybrana_kolumna , 'q25', alpha=alpha, n_bootstraps=boot)
                    st.write(':red[przedział ufnosci do Q1 [.025]:]')
                    st.dataframe(wyn)
             

        if typ_analizy =='weryfikacja założeń testów statystycznych':
            with st.container(border=True):   
                if wer_norm :
                    col1, col2, col3 = st.columns([2,2,4])
                    col1.write(f'Wybrany typ analizy:')
                    col2.info(f':red[{str.upper(typ_analizy)}]')
                    col1, col2 = st.columns([2,2])   

                    wybrana_kolumna = st.selectbox("Wybierz kolumnę numeryczną", kolumny_numeryczne) 
                    alpha = st.slider('Określ poziom alpha dla testu:',0.9,0.99, step = 0.01, value = 0.95)
                    st. write('')
                    st.dataframe(ana.testy_normalnosci_jeden(st.session_state.df, wybrana_kolumna,  wybrane_testy=None, alpha=alpha))

                        
                if wer_jed_warian :
                    wybrana_kolumna_n = st.selectbox("Wybierz kolumnę numeryczną   ", kolumny_numeryczne) 
                    wybrana_kolumna_k = st.selectbox("Wybierz kolumnę kategorialną    ", kolumny_kategorialne) 
                    st.text(ana.fisher_snedecor_test(st.session_state.df, wybrana_kolumna_k, wybrana_kolumna_n, alpha=0.05))

                if wer_jed_warian2 :
                    col1, col2, col3 = st.columns([2,2,4])
                    col1.write(f'Wybrany typ analizy:')
                    col2.info(f':red[{str.upper(typ_analizy)}]')
                    col1, col2 = st.columns([2,2])   
                    with col1:
                        st. write('')
                        wybrana_kolumna_n = st.selectbox("Wybierz kolumnę numeryczną        ", kolumny_numeryczne) 
                        wybrana_kolumna_k = st.selectbox("Wybierz kolumnę kategorialną         ", kolumny_kategorialne)
                        st.write(ana.test_jednorodnosci_wariancji(st.session_state.df, wybrana_kolumna_k, wybrana_kolumna_n, alpha=0.05))
                                                     
                    
                
                if wer_rowne_grupy :
                    col1, col2, col3 = st.columns([2,2,4])
                    col1.write(f'Wybrany typ analizy:')
                    col2.info(f':red[{str.upper(typ_analizy)}]')
                    col1, col2 = st.columns([2,2])   
                    with col1:
                        wybrana_kolumna = st.selectbox("Wybierz kolumnę numeryczną ", kolumny_numeryczne) 
                    with col2: 
                        alpha = st.slider('Określ poziom alpha testu:',0.9,0.99, step = 0.01, value = 0.95)
                    st. write('')
                    st.write(ana.test_rownowaznosci_kategorii(st.session_state.df, wybrana_kolumna, alfa= alpha))

        with tab6:
            st.subheader(' :blue[Moduł w budowie...............]🏗️')
            col1, col2= st.columns([2,2], gap = "medium")
            col1.image('under.jpg',width=100)



            import warnings
            warnings.filterwarnings("ignore")
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            from sklearn.preprocessing import KBinsDiscretizer, Binarizer, StandardScaler, OrdinalEncoder, MinMaxScaler, Normalizer, OneHotEncoder, StandardScaler, FunctionTransformer, RobustScaler
            from sklearn.pipeline import make_pipeline, Pipeline
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_validate, cross_val_predict, learning_curve, validation_curve
            from sklearn.metrics import roc_curve, roc_auc_score, cohen_kappa_score, matthews_corrcoef, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
            from sklearn.impute import SimpleImputer
            from sklearn.compose import make_column_transformer, ColumnTransformer
            from sklearn.base import BaseEstimator, TransformerMixin
            from sklearn.svm import SVC
            from sklearn.tree import DecisionTreeClassifier, plot_tree
            from sklearn.metrics import precision_recall_curve
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.naive_bayes import GaussianNB
            from sklearn.preprocessing import LabelEncoder
            from sklearn.linear_model import LogisticRegression
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
            from sklearn.preprocessing import PowerTransformer
            from sklearn.neural_network import MLPClassifier
            from sklearn.gaussian_process import GaussianProcessClassifier
            from sklearn.gaussian_process.kernels import RBF
            from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
            from sklearn.linear_model import SGDClassifier
            from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
            from sklearn.pipeline import make_pipeline, FeatureUnion
            from sklearn.impute import SimpleImputer, KNNImputer
            from sklearn.model_selection import GridSearchCV
            from sklearn.metrics import make_scorer, mean_squared_error
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import OneHotEncoder
            from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,ExtraTreesClassifier
            from sklearn.model_selection import cross_validate, cross_val_score
            from sklearn.ensemble import BaggingClassifier
            from sklearn.ensemble import VotingClassifier
            from sklearn import set_config
            set_config(transform_output='pandas')
            set_config(display='diagram')
            random_state=42


            st.write("")
            st.subheader('W tej wersji aplikacji do analizy mozna wybrac jedynie dataset : "szkoła"')
            if wybrane_dane =='szkoła':
                st.info(f"Wybrano dane demonstracyjne: {wybrane_dane}")
                df = st.session_state.df
             
                # Tworzenie DataFrame'a dla zmiennych objaśniających (X) i zmiennej docelowej (y)
                X = df.drop('czy zdał egzamin', axis=1)
                y = df['czy zdał egzamin'] 

                
                # Pobieranie nazw zmiennych X i nazwy zmiennej y
                nazwy_zmiennych_X = X.columns.tolist()
                nazwa_zmiennej_y = 'czy zdał egzamin'
                
                tabela_x = pd.DataFrame({'x': [nazwy_zmiennych_X]})
                tabela_y = pd.DataFrame({'y': [[nazwa_zmiennej_y]]})                                                       
                st.write(tabela_x, width=2000)
                st.write(tabela_y, width=2000)

                st.dataframe(ana.informacje_o_dataframe(st.session_state.df), height=height, hide_index=True, width=2200)


                numeric_variables = X.select_dtypes(include=['int64', 'float64'])
                numeric_variable_names = numeric_variables.columns.tolist()
                non_numeric_variables = X.select_dtypes(exclude=['int64', 'float64'])
                non_numeric_variable_names = non_numeric_variables.columns.tolist()
            
                wyniki = pd.DataFrame({
                    'Zmienne liczbowe': [numeric_variable_names],
                    'Zmienne tekstowe': [non_numeric_variable_names]}).T
                
                st.dataframe(wyniki,  width=2000)
                
                
                prop = { 'Struktura zmiennej celu': y.value_counts(normalize=True) * 100,}
                st.dataframe(pd.DataFrame(prop))
                
        

                st.info(f' UWAGA !:    Typ zmiennej celu: {y.dtype}, wartości zmiennej celu:  {y.unique()}')
                st.write('kodowanie zmiennej celu')
                

                # Utwórz instancję LabelEncoder
                label_encoder = LabelEncoder()

                # Przekształć zmienną y
                y = label_encoder.fit_transform(y)
                st.info(f' UWAGA !:    Typ zmiennej celu po kodowaniu: {y.dtype}')
                y_df = pd.DataFrame(y)
                st.write(y_df.value_counts(normalize=True) * 100,)
                
                
                
    
        

                # Podział danych na zbiór treningowy i testowy
                X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)


                shape_dict = {  "Dane": ["X", "y", "X_train", "y_train", "X_test", "y_test"],
                                "Rozmiar": [X.shape,y.shape,X_train.shape,y_train.shape,X_test.shape,y_test.shape,],}

                shape_df = pd.DataFrame(shape_dict)
                st.dataframe(shape_df)

                st.write(f'Struktura danych X_train: {(X_train.shape[0]/X.shape[0])*100}%')
                st.write(f'Struktura danych X_test: {(X_test.shape[0]/X.shape[0])*100}%')
                
            
                
                
                binary_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(sparse_output=False, handle_unknown='ignore',drop='if_binary',dtype='int'))
                ordinal_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'),OrdinalEncoder(categories=[['podstawowe', 'zawodowe', 'średnie', 'wyższe'],
                                                                                                                ['wieś','małe miasteczko','miasto średnie', 
                                                                                                                    'miasto duże' , 'miasto pow. 500 tys.']],dtype='int')) 
                ohe_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(sparse_output=False,handle_unknown='ignore'),)    
                ohe_rare_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(sparse_output=False,handle_unknown='infrequent_if_exist', max_categories=5, dtype='int'))    
                numeric_pipeline = make_pipeline(StandardScaler())  
                binarizer_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'),Binarizer(threshold=20)) #>20min
                kbins_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform'))




                transformers = [
                    ('binary', binary_pipeline, ['płeć', 'pali', 'problemy z rówieśnikami', 'typ szkoły', 'nadużywanie alkoholu', 'korzystanie z korepetycji']),
                    ('ordinal', ordinal_pipeline, ['wykształcenie', 'zamieszkanie']),
                    ('ohe', ohe_pipeline, ['tryb nauki']),
                    ('ohe_rare', ohe_rare_pipeline, ['ulubione social media']),
                    ('numerical', numeric_pipeline, ['srednia ocen sem'])]


                preprocessor = ColumnTransformer(
                    transformers=transformers,
                    verbose_feature_names_out=False,
                    remainder='passthrough')


                X_transformed = preprocessor.fit_transform(X_train)
                X_transformed_rounded = pd.DataFrame(X_transformed).round(2)
                st.dataframe(X_transformed_rounded.head())
                
                
                
                #Definicja modelu drzewa decyzyjnego
                
                from sklearn.tree import DecisionTreeClassifier

                # Definicja modelu z dodatkowymi parametrami
                model = DecisionTreeClassifier(
                    random_state=42, 
                    max_depth=5, 
                    min_samples_split=2, 
                    min_samples_leaf=1, 
                    max_features=None
                    )

                
                    # # Definiuj siatkę hiperparametrów do przeszukania dla Decision Tree
                    # tree_param_grid = {
                    #     'model__criterion': ['gini', 'entropy'],       # Kryterium podziału
                    #     'model__max_depth': [2,3,4,5,6,7],        # Maksymalna głębokość drzewa
                    #     'model__min_samples_split': [1,2,3,4,5,6],       # Minimalna liczba próbek do podziału węzła
                    #     'model__min_samples_leaf': [1,2,3,4,5]          # Minimalna liczba próbek w liściu
                    # }
                
                            
                pipe_DecisionTreeClassifier = Pipeline([
                ('preprocessor', preprocessor),
                ('model', DecisionTreeClassifier())])
               
                
                cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    

                # Ocena modelu za pomocą walidacji krzyżowej
                cv_results = cross_validate(pipe_DecisionTreeClassifier, X_train, y_train, cv=cv, scoring = {'accuracy': 'accuracy','f1': 'f1'})

                score_train = pd.DataFrame(cv_results)
                st.write('Wyniki skuteczności modelu wg CV:')
                st.dataframe(score_train)
                st.write(f'* średnia miara test_accuracy: {score_train["test_accuracy"].mean()}')
                st.write(f'* odchylenie standardowe : {score_train["test_accuracy"].std()}')

                score_train["test_accuracy"].plot(kind = 'line',  marker='o',linestyle='-', color='b', title='Skuteczność modelu danych testowych', xlabel='Fold Number', ylabel='Test Accuracy')
                st.pyplot()



                wyniki = cross_validate(pipe_DecisionTreeClassifier, X_train, y_train, cv=cv, scoring='accuracy', return_train_score=True)

                # Obliczasz średnie i odchylenia standardowe
                srednia_test = round(wyniki['test_score'].mean(), 3)
                std_test = round(wyniki['test_score'].std(), 3)
                srednia_train = round(wyniki['train_score'].mean(), 3)
                std_train = round(wyniki['train_score'].std(), 3)
                srednia_score_time = round(wyniki['score_time'].mean(), 3)
                std_score_time = round(wyniki['score_time'].std(), 3)

                # Tworzysz ramkę danych wyniki_df
                wyniki_df = pd.DataFrame({
                    'mean score_time': [srednia_score_time],
                    'std score_time': [std_score_time],
                    'mean train_score': [srednia_train],
                    'mean test_score': [srednia_test],
                    'std test_score': [std_test],
                    'std train_score': [std_train]
                })

                # Wyświetlasz ramkę danych
                st.dataframe(wyniki)
                st.dataframe(wyniki_df)

                train_scores = wyniki['train_score']
                test_scores = wyniki['test_score']

                # Utwórz zakres kolumn
                fold_indices = np.arange(1, len(train_scores) + 1)

                # Stwórz wykres
                plt.figure(figsize=(10, 6))
                plt.plot(fold_indices, train_scores, marker='o', label='Train Accuracy', color='blue')
                plt.plot(fold_indices, test_scores, marker='o', label='Test Accuracy', color='orange')
                plt.xlabel('Fold Index')
                plt.ylabel('Accuracy')
                plt.title('Train and Test Accuracy Across Folds')
                plt.legend()
                plt.grid(True)
        
                st.pyplot()
                
             
                # Tworzymy krzywą uczenia
                train_sizes, train_scores, test_scores = learning_curve(pipe_DecisionTreeClassifier, X_train, y_train, cv=cv, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))

                # Obliczamy średnie i odchylenia standardowe dla wyników treningowych i testowych
                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                test_mean = np.mean(test_scores, axis=1)
                test_std = np.std(test_scores, axis=1)

                # Tworzymy wykres krzywej uczenia
                plt.figure(figsize=(10, 6))
                plt.plot(train_sizes, train_mean, marker='o', color='blue', label='Training accuracy')
                plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
                plt.plot(train_sizes, test_mean, marker='o', color='orange', label='Test accuracy')
                plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='orange')

                # Dodajemy opisy osi i tytuł
                plt.title('Learning Curve')
                plt.xlabel('Number of training examples')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.grid(True)
                st.pyplot()
        
        
                # Trenowanie modelu na całym zbiorze treningowym (opcjonalnie)
                pipe_DecisionTreeClassifier.fit(X_train, y_train)

                # Testowanie modelu na zbiorze testowym
                y_pred = pipe_DecisionTreeClassifier.predict(X_test)

                st.text(' ================================= OCENA MODELU K-NN PREDYKCJA NA ZBIORZE TRENINGOWYM ============================')


                y_train_pred = cross_val_predict(pipe_DecisionTreeClassifier, X_train, y_train, cv=cv)


                def score_train():
                    cm_test = confusion_matrix(y_train, y_train_pred)
                    TP, FP, FN, TN = cm_test.ravel()
                    accuracy = (TP + TN) / (TP + TN + FP + FN)
                    error_ratio = (FP + FN) / (TP + TN + FP + FN)
                    precision_pos = TP / (TP + FP)
                    precision_neg = TN / (TN + FN)
                    recall_pos = TP / (TP + FN)
                    recall_neg = TN / (TN + FP)
                    f1_score = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos)
                    data = {'miara': ['TP', 'FP', 'FN', 'TN','accuracy','error_ratio', 'precision_pos','precision_neg','recall_pos', 'recall_neg','f1_score'],
                            'wartość': [TP, FP, FN, TN,accuracy,error_ratio, precision_pos,precision_neg,recall_pos, recall_neg,f1_score]}
                    df = pd.DataFrame(data).set_index('miara')
                    return df.T


                st.dataframe(score_train())

                # Tworzymy wykres macierzy pomyłek
                cm_train = confusion_matrix(y_train, y_train_pred)
                classes = ['Klasa Negatywna', 'Klasa Pozytywna']  # Zdefiniuj nazwy klas
                disp = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=classes)
                disp.plot(cmap=plt.cm.Blues, values_format=".2f")
                plt.title("Macierz Pomyłek")
                plt.xlabel("Przewidziane etykiety")
                plt.ylabel("Rzeczywiste etykiety")
                st.pyplot()





                # Predykcja na danych testowych
                y_test_pred = pipe_DecisionTreeClassifier.predict(X_test)

                # Prawdopodobieństwo przynależności do klasy pozytywnej (klasa 1) dla danych testowych
                y_test_proba = pipe_DecisionTreeClassifier.predict_proba(X_test)[:, 1]


                def score_test():
                    cm_test = confusion_matrix(y_test, y_test_pred)
                    TP, FP, FN, TN = cm_test.ravel()
                    accuracy = (TP + TN) / (TP + TN + FP + FN)
                    error_ratio = (FP + FN) / (TP + TN + FP + FN)
                    precision_pos = TP / (TP + FP)
                    precision_neg = TN / (TN + FN)
                    recall_pos = TP / (TP + FN)
                    recall_neg = TN / (TN + FP)
                    f1_score = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos)
                    roc_auc_test = roc_auc_score(y_test, y_test_proba)
                    cohen_kappa = cohen_kappa_score(y_test, y_test_pred)
                    matthews_corrcoef_score = matthews_corrcoef(y_test, y_test_pred)
                    data = {'miara': ['TP', 'FP', 'FN', 'TN','accuracy','error_ratio', 'precision_pos','precision_neg','recall_pos', 'recall_neg','f1_score',
                                    'roc_auc_test','cohen_kappa','matthews_corrcoef_score'],
                            'wartość': [TP, FP, FN, TN,accuracy,error_ratio, precision_pos,precision_neg,recall_pos, recall_neg,f1_score,
                                        roc_auc_test,cohen_kappa,matthews_corrcoef_score]}
                    df = pd.DataFrame(data).set_index('miara')
                    return df

                st.text('-'*80)
                st.text('OCENA MODELU K-NN PREDYKCJA NA ZBIORZE TESTOWYM ')
            
                st.dataframe(score_test().T)

                cm_test = confusion_matrix(y_test, y_test_pred)
                classes = ['nie zdał', 'zdał']  
                disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=classes)
                disp.plot(cmap=plt.cm.Blues, values_format=".2f")
                plt.title("Macierz Pomyłek")
                plt.xlabel("Przewidziane etykiety")
                plt.ylabel("Rzeczywiste etykiety")
                st.pyplot()










                # # Dane o ważności cech i ich nazwach
                # importances = pipe_DecisionTreeClassifier.feature_importances_
                # feature_names = ['płeć', 'pali', 'wykształcenie', 'liczba osób', 'typ szkoły',
                #     'dochód roczny', 'srednia ocen sem', 'tryb nauki', 'zamieszkanie',
                #     'problemy z rówieśnikami', 'czas do szkoły min',
                #     'godzin nauki przed egzaminem', 'nadużywanie alkoholu', 'poziom stresu',
                #     'korzystanie z korepetycji',
                #     'czas spedzany tygodniu na social mediach w godz',
                #     'ulubione social media']

                # # Tworzenie ramki danych z ważnością cech
                # feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

                # # Sortowanie ramki danych według ważności w kolejności malejącej
                # feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

                # # Tworzenie wykresu słupkowego
                # plt.figure(figsize=(8, 4))
                # plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
                # plt.xlabel('Ważność cech')
                # plt.ylabel('Cecha')
                # plt.title('Ważność cech w modelu drzewa decyzyjnego')
                # plt.gca().invert_yaxis()  # Odwrócenie osi Y, aby najważniejsze cechy były na górze
                # st.pyplot()



                # Oblicz krzywą ROC i pole pod krzywą ROC (AUC-ROC)
                fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
                roc_auc = roc_auc_score(y_test, y_test_proba)
                plt.figure(figsize=(6, 4))
                plt.plot(fpr, tpr, lw=1, color='red', linestyle='--', label='Krzywa ROC (AUC = %0.2f)' % roc_auc)
                plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                plt.xlabel('False Positive Rate (FPR)')
                plt.ylabel('True Positive Rate (TPR)')
                plt.title('Krzywa ROC')
                plt.legend(loc='lower right')
                plt.yticks(np.arange(0, 1.1, 0.1))
                plt.xticks(np.arange(0, 1.1, 0.1))
                plt.grid(True)
                thresh_points = np.linspace(0, 1, num=10)
                for thresh_point in thresh_points:
                    index = np.argmin(np.abs(thresholds - thresh_point))
                    plt.scatter(fpr[index], tpr[index], c='blue', s=20)
                    #plt.annotate(f'{thresh_point:.2f}', (fpr[index], tpr[index]), textcoords="offset points", xytext=(20, -10), ha='center', fontsize=9)

                st.pyplot()














        # client = OpenAI(api_key = 'API_KEY')    

        with tab7:
            from openai import OpenAI
            import streamlit as st
            
            st.markdown("Pomoc z użyciem modelu: gpt-3.5-turbo")
            st.write("w trakcie testów....")
            
            #api_key = st.secrets["klucz"]
            # client = OpenAI(api_key=api_key)
            # prompt1 = st.text_input("Proszę podać pytanie z dziedziny statystyki, sztucznej inteligencji, uczenia maszynowego, nauki o danych:")
            # completion = client.chat.completions.create(
            #     model="gpt-3.5-turbo",
            #     messages=[
            #         {"role": "system", "content": "Jestem profesorem statystytki i wykładam na uczelnni odpowiadam rzeczowo i profesjonalnie na wszystkie zagadnienia z dziedziny statystyki, sztuznej inteligencji, uczeniu maszynowym, nauki o danych , podaję definicję , wzory i interpretację. Nie wolno mi odpowiadać na pytania z innych dziedzin."},
            #         {"role": "user", "content": prompt1}
            #     ]
            # )
            # prompt = st.chat_input(prompt)
            # if prompt:
            #     st.write(f"User has sent the following prompt: {prompt}")
       
            #     tresc_odpowiedzi = completion.choices[0].message.content

            #     with st.chat_message("user"):
            #         st.write("Moja odpowiedź na Twoje pytanie: ")
            #         st.write(tresc_odpowiedzi)
        
        
            
            #client = OpenAI(api_key=st.secrets["klucz"])
            
            import streamlit as st
            from openai import OpenAI

            
            #openai_api_key = st.text_input("Wprowadź OpenAI API Key", key="chatbot_api_key", type="password")
            openai_api_key = api_key = st.secrets["klucz"]

            st.title("💬 Jak zapytasz to odpowiem ....")

            if "messages" not in st.session_state:
                st.session_state["messages"] = [{"role": "assistant", "content": "Jak mogę Ci pomóc?"}]

            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg["content"])

            if prompt := st.chat_input():
                if not openai_api_key:
                    st.info("Please add your OpenAI API key to continue.")
                    st.stop()

                client = OpenAI(api_key=openai_api_key)
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.chat_message("user").write(prompt)
                response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
                msg = response.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": msg})
                
                st.chat_message("assistant").write(msg)

