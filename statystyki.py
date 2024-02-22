import pandas as pd
import warnings
warnings.filterwarnings('ignore')
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


import random
from scipy.stats import zscore
import re
import math


# #-------------------------------------------------------------------------------------------------
# #                                 PRZEDZIAŁY UFNOSCI
# #-------------------------------------------------------------------------------------------------



def przedzial_ufnosci_srednia(df, zmienna, alpha=0.05, duza_proba=True):
    """
    Oblicza przedział ufności dla średniej.

    :param dane: lista lub seria danych
    :param alpha: poziom istotności
    :param duza_proba: czy próba jest duża (True) czy mała (False)
    """
    dane = df[zmienna]
    srednia = sum(dane) / len(dane)
    if duza_proba:
        std_err = stats.sem(dane)
        przedzial = stats.norm.interval(1-alpha, loc=srednia, scale=std_err)
    else:
        przedzial = stats.t.interval(1-alpha, len(dane)-1, loc=srednia, scale=stats.sem(dane))
    przedzial = (round(przedzial[0], 2), round(przedzial[1], 2))
    srednia = round(srednia, 2)
    
    print(f'Przedział ufności dla średniej na poziomie {1-alpha} dla zmiennej: "{zmienna}": ')
    print('-'*80)
    print(f'średnia : {srednia:.2f}    przedział ufnosci:  {przedzial}')

    return 




def przedzial_ufnosci_proporcja(df, zmienna, confidence=0.95):
    """
    Oblicza przedziały ufności dla każdej kategorii w kolumnie kategorycznej w DataFrame
    używając biblioteki statsmodels.

    :param df: DataFrame zawierający dane
    :param zmienna: nazwa kolumny kategorycznej w DataFrame
    :param confidence: poziom ufności
    """
    wyniki = {}
    print(f'Przedział ufności do wskaźnika struktury zmiennej: {zmienna}  (na poziomie {confidence}):')
    print('-'*80)
    for kategoria in df[zmienna].unique():
        successes = sum(df[zmienna] == kategoria)
        n = len(df[zmienna])
        confidence_interval = sm.stats.proportion_confint(successes, n, alpha=(1 - confidence))
        confidence_interval_rounded = (round(confidence_interval[0], 2), round(confidence_interval[1], 2))
        wyniki[kategoria] = confidence_interval_rounded

    for kategoria, przedzial in wyniki.items():
        print(f'Kategoria "{kategoria}": Przedział ufności (na poziomie {confidence}):  {przedzial}')
    
    return 




def przedzial_ufnosci_roznicy_srednich(df, zmienna_num, zmienna_kat, confidence=0.95):
    """
    Oblicza przedziały ufności dla różnic między średnimi różnych grup określonych przez zmienną kategoryczną,
    pokazując również zaokrąglone wartości średnich w poszczególnych grupach oraz ich różnice.

    :param df: DataFrame zawierający dane
    :param zmienna_num: nazwa zmiennej numerycznej w DataFrame
    :param zmienna_kat: nazwa zmiennej kategorycznej w DataFrame, która definiuje grupy
    :param confidence: poziom ufności
    """
    wyniki = {}
    unikalne_kategorie = df[zmienna_kat].unique()

    for i in range(len(unikalne_kategorie)):
        for j in range(i + 1, len(unikalne_kategorie)):
            grupa1, grupa2 = unikalne_kategorie[i], unikalne_kategorie[j]
            dane1 = df[df[zmienna_kat] == grupa1][zmienna_num]
            dane2 = df[df[zmienna_kat] == grupa2][zmienna_num]

            mean1, mean2 = np.mean(dane1), np.mean(dane2)
            roznica_srednich = mean1 - mean2
            se1, se2 = stats.sem(dane1), stats.sem(dane2)
            sed = np.sqrt(se1**2 + se2**2)  # standard error of the difference

            z = stats.norm.ppf((1 + confidence) / 2)
            margin_error = z * sed
            confidence_interval = (roznica_srednich - margin_error, roznica_srednich + margin_error)

            wyniki[(grupa1, grupa2)] = {
                'średnia_1': round(mean1, 2),
                'średnia_2': round(mean2, 2),
                'różnica_średnich': round(roznica_srednich, 2),
                'przedział_ufności': (round(confidence_interval[0], 2), round(confidence_interval[1], 2))
            }
    print(f'Przedzial_ufnosci_roznicy_srednich zmiennej {zmienna_num} wg poziomów zmiennej {zmienna_kat} (na poziomie {confidence}):')
    print('-'*80)
    for grupy, info in wyniki.items():
        print(f"Różnica między grupami {zmienna_kat} - '{grupy[0]}' i '{grupy[1]}':")
        print(f" Średnia ({zmienna_num}) w grupie '{grupy[0]}' : {info['średnia_1']}")
        print(f" Średnia ({zmienna_num}) w grupie '{grupy[1]}' : {info['średnia_2']}")
        print(f" Różnica średnich: {info['różnica_średnich']}")
        print(f" Przedział ufności: {info['przedział_ufności']}\n")
    
    return 





def przedzial_ufnosci_mediany(df, zmienna, alpha=0.05, n_bootstraps=1000):
    """
    Oblicza przedział ufności dla mediany za pomocą metody bootstrap.

    :param df: DataFrame zawierający dane
    :param zmienna: nazwa zmiennej, dla której obliczany jest przedział ufności
    :param alpha: poziom istotności (domyślnie 0.05)
    :param n_bootstraps: liczba próbek bootstrapowych
    """
    data = df[zmienna].values
    bootstrapped_medians = np.array([np.median(np.random.choice(data, replace=True, size=len(data))) for _ in range(n_bootstraps)])
    ci_lower = np.percentile(bootstrapped_medians, 100*alpha/2)
    ci_upper = np.percentile(bootstrapped_medians, 100*(1-alpha/2))
    
    print(f'Przedział ufności dla mediany na poziomie {1-alpha} dla zmiennej: "{zmienna}": ')
    print('-'*80)
    print(f"Mediana: {np.median(data):.2f}")
    print(f"Przedział ufności: ({ci_lower:.2f}, {ci_upper:.2f})")

    return 


def przedzial_ufnosci_odchylenia_standardowego(df, zmienna, alpha=0.05):
    """
    Oblicza przedział ufności dla odchylenia standardowego i wyświetla wynik wraz z nazwą zmiennej.

    :param df: DataFrame zawierający dane
    :param zmienna: nazwa zmiennej
    :param alpha: poziom istotności
    """
    n = len(df[zmienna])
    odchylenie_std = np.std(df[zmienna], ddof=1)
    var = np.var(df[zmienna], ddof=1)
    dolny = np.sqrt((n - 1) * var / stats.chi2.ppf(1 - alpha / 2, n - 1))
    gorny = np.sqrt((n - 1) * var / stats.chi2.ppf(alpha / 2, n - 1))
    print(f'Przedzial_ufnosci_odchylenia_standardowego {1-alpha} dla zmiennej: "{zmienna}": ')
    print('-'*80)
    print(f"Zmienna: {zmienna}")
    print(f"Odchylenie standardowe: {odchylenie_std:.2f}")
    print(f"Przedział ufności: ({dolny:.2f}, {gorny:.2f})")
    return 




def bootstrap_ci(df, zmienna, stat_func_name, alpha=0.05, n_bootstraps=10000):
    """
    Oblicza przedział ufności bootstrap dla dowolnej funkcji statystycznej.

    :param df: DataFrame zawierający dane.
    :param zmienna: Nazwa zmiennej, dla której obliczany jest przedział ufności.
    :param stat_func_name: Nazwa funkcji statystycznej do obliczenia na danych, np. 'średnia', 'mediania''odchylenie','wariancja''q25','q75'
    :param alpha: Poziom istotności dla przedziału ufności.
    :param n_bootstraps: Liczba próbek bootstrapowych do wygenerowania.
    :return: Krotka z dolnym i górnym ograniczeniem przedziału ufności.
    """
    # Rozszerzony słownik mapujący nazwy funkcji na funkcje numpy
    stat_funcs = {
        'średnia': np.mean,
        'mediana': np.median,
        'odchylenie':  lambda x: np.std(x, ddof=1),
        'wariancja':  lambda x: np.var(x, ddof=1),
        'q25': lambda x: np.quantile(x, 0.25), 
        'q75': lambda x: np.quantile(x, 0.75) 
    }
    stat_func = stat_funcs.get(stat_func_name)
    
    if stat_func is None:
        raise ValueError(f"{stat_func_name} nie jest obsługiwaną nazwą funkcji. Wybierz jedną z: {list(stat_funcs.keys())}.")
    
    data = df[zmienna].values
    original_stat = stat_func(data)
    bootstrapped_stats = np.array([stat_func(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstraps)])
    ci_lower = np.percentile(bootstrapped_stats, 100*alpha/2)
    ci_upper = np.percentile(bootstrapped_stats, 100*(1-alpha/2))
    
    print(f'Przedział ufności bootstrap {stat_func_name} na poziomie alfa = {1-alpha} dla zmiennej: "{zmienna}": ')
    print('-'*80)
    print(f"Zmienna: {zmienna}")
    print(f"{stat_func_name}: {original_stat:.2f}")
    print(f"Przedział ufności: ({ci_lower:.2f}, {ci_upper:.2f})")

    return 



def dwumianowy_przedzial_ufnosci(df, zmienna, sukces, alpha=0.05, method='normal'):
    """
    Oblicza i drukuje dwumianowy przedział ufności oraz wyliczoną proporcję sukcesów dla określonej kategorii.

    :param df: DataFrame zawierający dane.
    :param zmienna: Nazwa kolumny, dla której obliczany jest przedział ufności.
    :param sukces: Wartość reprezentująca 'sukces' w analizowanej kategorii.
    :param alpha: Poziom istotności dla przedziału ufności.
    :param method: Metoda obliczania przedziału ('normal' lub 'exact').
    :return: Wyliczona proporcja sukcesów oraz przedział ufności jako krotka.
    """
    successes = df[zmienna].apply(lambda x: 1 if x == sukces else 0).sum()
    trials = len(df[zmienna])
    p_hat = successes / trials  # Wyliczona proporcja sukcesów

    if method == 'normal':
        z = stats.norm.ppf(1 - alpha / 2)
        margin_error = z * np.sqrt((p_hat * (1 - p_hat)) / trials)
        confidence_interval = (p_hat - margin_error, p_hat + margin_error)
    elif method == 'exact':
        confidence_interval = sm.stats.proportion_confint(successes, trials, alpha=alpha, method='binom_test')
    else:
        raise ValueError("Method should be 'normal' or 'exact'.")
    
    print(f'dwumianowy_przedzial_ufnosci na poziomie alfa = {1-alpha} dla zmiennej: "{zmienna}": ')
    print('-'*80)
    print(f"Wyliczona proporcja sukcesów dla '{zmienna}' = {p_hat:.2f}")
    print(f"Przedział ufności: ({confidence_interval[0]:.2f}, {confidence_interval[1]:.2f})")

    return




# #-------------------------------------------------------------------------------------------------------------------------------
# #                                         TESTY STATYSTYCZNE
# #--------------------------------------------------------------------------------------------------------------------------------





    # Porównanie - 1 grupa
    #     Testy parametryczne
    #             Test t-Studenta dla pojedynczej próby
    #     Testy nieparametryczne
    #             Test Wilcoxona (rangowanych znaków)
    #             Test chi-kwadrat zgodności
    #             Testy dla jednej proporcji







#-----------------
# TEST NORMALNOSCI
#-----------------




def testy_normalnosci_jeden(df, zmienna, wybrane_testy=None, alpha=0.05):
    """
    Przeprowadza wybrane testy na normalność rozkładu danych.

    :param x: dane do testowania
    :param wybrane_testy: lista testów do wykonania (domyślnie wszystkie)
    :param alpha: poziom istotności dla testów (domyślnie 0.05)
    """

    def wykonaj_test(statystyka, p_value, nazwa_testu):
        print(f'{nazwa_testu}: statystyka-t: {statystyka:.4f}, p-value {p_value:.4f}')
        if p_value > alpha:
            print(f' Brak dowodów na odrzucenie hipotezy o normalności rozkładu danych.Próbka ma rozkład normalny (na poziomie istotności {alpha})\n')
        else:
            print(f'Istnieją dowody na to, że rozkład danych odbiega od normalności.Próbka nie ma rozkładu normalnego (na poziomie istotności {alpha})\n')

    dostepne_testy = {
        "shapiro": lambda: stats.shapiro(df[zmienna]),
        "lilliefors": lambda: sm.stats.diagnostic.lilliefors(df[zmienna], 'norm'),
        "dagostino": lambda: stats.normaltest(df[zmienna]),
        "skewness": lambda: stats.skewtest(df[zmienna]),
        "kurtosis": lambda: stats.kurtosistest(df[zmienna]),
        "jarque-bera": lambda: stats.jarque_bera(df[zmienna])
    }

    if wybrane_testy is None:
        wybrane_testy = dostepne_testy.keys()
    print(f'Testowanie normalnosci rozkładu zmiennej: {zmienna}')
    print('-'*150)
    try:
        for test in wybrane_testy:
            if test in dostepne_testy:
                statystyka, p_value = dostepne_testy[test]()
                wykonaj_test(statystyka, p_value, f"Test {test.capitalize()}")
            else:
                print(f"Nieznany test: {test}")
    except Exception as e:
        print(f"Wystąpił błąd podczas przeprowadzania testów: {e}")








# #----------------------------------
# #  ZMIENNA NUMERYCZNA - ŚREDNIA        TESTY  T , Z
# #----------------------------------

from scipy import stats
import numpy as np
from statsmodels.stats.power import TTestPower

def test_t_jednej_probki(df, zmienna, srednia_populacji, alpha=0.05):
    # Definicje hipotez
    hipoteza_zerowa = f"Średnia {zmienna} w populacji jest równa {srednia_populacji}"
    hipoteza_alternatywna = f"Średnia {zmienna} w populacji nie jest równa {srednia_populacji}"
    poziom_istotnosci = alpha

    # Wykonanie testu T
    statystyka, p_wartosc = stats.ttest_1samp(df[zmienna], srednia_populacji)
    
    # Średnia próbki i odchylenie standardowe
    srednia_probki = df[zmienna].mean()
    odchylenie_std = df[zmienna].std()

    # Wielkość efektu d Cohena
    d_cohena = (srednia_probki - srednia_populacji) / odchylenie_std

    # Obliczanie mocy testu
    analiza_mocy = TTestPower()
    rozmiar_probki = len(df[zmienna])
    moc = analiza_mocy.solve_power(effect_size=d_cohena, nobs=rozmiar_probki, alpha=alpha, alternative='two-sided')

    # Interpretacja wyników
    interpretacja = "statystycznie istotne różnice" if p_wartosc < alpha else "brak statystycznie istotnych różnic"
    print('test_t_jednej_probki:')
    print('--------------------------------------------------------------------')
    print(f"Hipoteza zerowa: {hipoteza_zerowa}")
    print(f"Hipoteza alternatywna: {hipoteza_alternatywna}")
    print(f"Poziom istotności: {poziom_istotnosci}")
    print(f"Średnia {zmienna} w próbce: {srednia_probki:.2f}")
    print(f"Hipotetyczna średnia populacji: {srednia_populacji}")
    print(f"Wynik testu T: {statystyka:.4f}, P-wartość: {p_wartosc:.4f}")
    print(f"Wielkość efektu (d Cohena): {d_cohena:.4f}")
    print(f"Moc testu: {moc:.4f}")
    print(f"Interpretacja: {interpretacja}")




from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import zt_ind_solve_power

def test_z_jednej_probki(df, zmienna, population_mean, alpha=0.05, population_std=None):
    # Hipotezy
    hipoteza_zerowa = f"Średnia {zmienna} w populacji jest równa {population_mean}"
    hipoteza_alternatywna = f"Średnia {zmienna} w populacji nie jest równa {population_mean}"

    # Test Z
    z_statistic, p_value = stests.ztest(x1=df[zmienna], value=population_mean)
    sample_mean = df[zmienna].mean()
    sample_std = df[zmienna].std() if population_std is None else population_std
    n = len(df[zmienna])

    # Wielkość efektu d Cohena
    d_cohena = (sample_mean - population_mean) / sample_std

    interpretacja = "statystycznie istotne różnice" if p_value < alpha else "brak statystycznie istotnych różnic"
    print('test_Z_jednej_probki:')
    print('---------------------------------------------------------------------------')
    print(f"Hipoteza zerowa: {hipoteza_zerowa}")
    print(f"Hipoteza alternatywna: {hipoteza_alternatywna}")
    print(f"Poziom istotności: {alpha}")
    print(f"Średnia {zmienna} w próbce: {sample_mean}")
    print(f"Hipotetyczna średnia populacji: {population_mean}")
    print(f"Wynik testu Z: {z_statistic:.4f}, P-wartość: {p_value:.4f}")
    print(f"Wielkość efektu (d Cohena): {d_cohena:.4f}")

    print(f"Interpretacja: {interpretacja}")




def test_t_jednej_probki2(df, zmienna, srednia_populacji, alpha=0.05, typ_testu='obustronny'):
    hipoteza_zerowa = f"Średnia {zmienna} w populacji jest równa {srednia_populacji}"
    if typ_testu == 'obustronny':
        hipoteza_alternatywna = f"Średnia {zmienna} w populacji nie jest równa {srednia_populacji}"
    elif typ_testu == 'lewostronny':
        hipoteza_alternatywna = f"Średnia {zmienna} w populacji jest mniejsza niż {srednia_populacji}"
    else:  # prawostronny
        hipoteza_alternatywna = f"Średnia {zmienna} w populacji jest większa niż {srednia_populacji}"
    
    poziom_istotnosci = alpha
    statystyka, p_wartosc = stats.ttest_1samp(df[zmienna], srednia_populacji)
    
    # Dostosowanie P-wartości do typu testu
    if typ_testu == 'lewostronny':
        p_wartosc = p_wartosc / 2 if statystyka < 0 else 1 - (p_wartosc / 2)
    elif typ_testu == 'prawostronny':
        p_wartosc = p_wartosc / 2 if statystyka > 0 else 1 - (p_wartosc / 2)
    
    srednia_probki = df[zmienna].mean()
    interpretacja = "statystycznie istotne różnice" if p_wartosc < alpha else "brak statystycznie istotnych różnic"
    print('test_t_jednej_probki:')
    print('--------------------------------------------------------------------')
    print(f"Hipoteza zerowa: {hipoteza_zerowa}")
    print(f"Hipoteza alternatywna: {hipoteza_alternatywna}")
    print(f"Poziom istotności: {poziom_istotnosci}")
    print(f"Średnia {zmienna} w próbce: {srednia_probki}")
    print(f"Hipotetyczna średnia populacji: {srednia_populacji}")
    print(f"Wynik testu T: {statystyka:.4f}, P-wartość: {p_wartosc:.4f}")
    print(f"Interpretacja: {interpretacja}")




# #--------------------------------
# #  ZMIENNA NUMERYCZNA -  WARIANCJA
# #---------------------------------



import numpy as np
from scipy.stats import chi2



def chi_square_variance_test(df, zmienna, population_variance, alpha=0.05):
    n = len(df[zmienna])
    sample_variance = np.var(df[zmienna], ddof=1)  # Oblicz wariancję próbki
    chi_square_statistic = (n - 1) * sample_variance / population_variance
    degrees_of_freedom = n - 1
    p_value = 1 - chi2.cdf(chi_square_statistic, degrees_of_freedom)
    
    H0 = f"Hipoteza zerowa: Wariancja '{zmienna}' w populacji jest równa {population_variance:.2f}"
    H1 = f"Hipoteza alternatywna: Wariancja '{zmienna}' w populacji nie jest równa {population_variance:.2f}"
    
    interpretation = "istnieją statystycznie istotne różnice (odrzucamy H0)" if p_value < alpha else "brak statystycznie istotnych różnic (nie odrzucamy H0)"
    
    print('chi_square_variance_test:')
    print('---------------------------------------------------------------------------')
    print(H0)
    print(H1)
    print(f"Poziom istotności (alpha): {alpha}")
    print(f"Wariancja '{zmienna}' w próbce: {sample_variance:.4f}")
    print(f"Hipotetyczna wariancja populacji: {population_variance:.2f}")
    print(f"Stopnie swobody: {degrees_of_freedom}")
    print(f"Wynik testu Chi-kwadrat: {chi_square_statistic:.4f}")
    print(f"P-wartość: {p_value:.4f}")
    print(f"Interpretacja: {interpretation}")




#----------------------------------
# ZMIENNA KATEGORIALNA - PROPORCJE
#---------------------------------------

def test_z_proporcji(df, zmienna, target_poziom, population_proportion, alpha=0.05):

    H0 = f"H0: Proporcja sukcesów ('{target_poziom}') w próbce jest równa {population_proportion}."
    H1 = f"H1: Proporcja sukcesów ('{target_poziom}') w próbce nie jest równa {population_proportion}."
    df['binarna'] = df[zmienna].apply(lambda x: 1 if x == target_poziom else 0)
    successes = df['binarna'].sum()
    n_obs = len(df['binarna'])
    z_statistic, p_value = proportions_ztest(count=successes, nobs=n_obs, value=population_proportion)
    interpretacja = "statystycznie istotne" if p_value < alpha else "brak statystycznie istotnych"
    proporcje = df[zmienna].value_counts(normalize=True).to_dict()
    print('test_z_proporcji:')
    print('---------------------------------------------------------------------------')
    wynik = f"{H0}\n{H1}\nWynik testu Z: {z_statistic:.4f}\nP-wartość: {p_value:.4f}\nInterpretacja: {interpretacja} różnice"
    proporcje_str = "\n".join([f"{klucz}: {wartosc:.2f}" for klucz, wartosc in proporcje.items()])
    return wynik + "\nProporcje poziomów zmiennej:\n" + proporcje_str




# ROZKLAAD INNY NIZ NORMALNY

#---------------------------------
# ZMIENNA NUMERYCZNA - MEDIANY
#---------------------------------



def wilcoxon_signed_rank_test(df, zmienna, population_median, alpha=0.05):
    n = len(df[zmienna])
    differences = np.array(df[zmienna]) - population_median
    abs_differences = np.abs(differences)
    ranked_differences = abs_differences.argsort().argsort() + 1
    T = sum(np.sign(differences) * ranked_differences)
    p_value = wilcoxon(differences, alternative='two-sided')[1]
    H0 = f"Hipoteza zerowa: Mediana {zmienna} w populacji jest równa {population_median}"
    H1 = f"Hipoteza alternatywna: Mediana {zmienna} w populacji nie jest równa {population_median}"
    interpretation = "statystycznie istotne różnice" if p_value < alpha else "brak statystycznie istotnych różnic"
    print('wilcoxon_signed_rank_test:')
    print('---------------------------------------------------------------------------')
    wynik = f"{H0}\n{H1}\nPoziom istotności: {alpha}\nMediana {zmienna} w próbce: {np.median(df[zmienna]):.2f}\nHipotetyczna mediana populacji:{population_median:.2f}\nWynik testu Wilcoxona: {T:.4f}, P-wartość: {p_value:.2f}\nInterpretacja: {interpretation}"
    return wynik



#-------------------------------------------
# ZMIENNA KATEGORIALNA -  TEST DOPASOWANIA  - PROPORCJE
#-------------------------------------------


import scipy.stats as stats

def chi_square_goodness_of_fit(df, zmienna, alpha=0.05):
    observed_values = df[zmienna].value_counts().sort_index()
    n_categories = len(observed_values)
    expected_counts = np.full(shape=n_categories, fill_value=observed_values.sum() / n_categories)
    chi_squared_statistic, p_value = stats.chisquare(f_obs=observed_values, f_exp=expected_counts)
    
    H0 = "Hipoteza zerowa: Obserwowane częstości pasują do równomiernego rozkładu."
    H1 = "Hipoteza alternatywna: Obserwowane częstości nie pasują do równomiernego rozkładu."
    interpretation = "statystycznie istotne różnice" if p_value < alpha else "brak statystycznie istotnych różnic"
    
    print('chi_square_goodness_of_fit:')
    print('---------------------------------------------------------------------------')
    print(f"{H0}\n{H1}\nPoziom istotności: {alpha}\n"
          f"Wynik testu Chi-kwadrat: {chi_squared_statistic:.4f}, P-wartość: {p_value:.4f}\n"
          f"Interpretacja: {interpretation}")
    
    df_wynik = pd.DataFrame({'Obserwowane': observed_values, 'Oczekiwane': expected_counts})
    df_wynik['Proporcje'] = df_wynik['Obserwowane'] / df_wynik['Obserwowane'].sum() * 100
    
    return df_wynik


import pandas as pd
import scipy.stats as stats

def test_dwumianowy(df, zmienna, nazwa_poziom, oczekiwana_proporcja=0.5, alpha=0.05):
    """
    Przeprowadza test dwumianowy dla określonej zmiennej kategorialnej w DataFrame.

    :param df: DataFrame zawierający dane.
    :param zmienna: Nazwa kolumny, dla której przeprowadzany jest test.
    :param nazwa_poziom: Nazwa kategorii traktowanej jako 'sukces'.
    :param oczekiwana_proporcja: Oczekiwana proporcja sukcesów w populacji.
    :param alpha: Poziom istotności testu.
    :return: Wynik testu jako ciąg tekstowy.
    """
    # Przygotowanie danych
    df['binarna'] = df[zmienna].apply(lambda x: 1 if x == nazwa_poziom else 0)
    n = len(df)  # Rozmiar próbki
    x = df['binarna'].sum()  # Liczba sukcesów
    
    # Obliczanie p-wartości dla testu dwumianowego
    result = stats.binomtest(x, n, oczekiwana_proporcja, alternative='two-sided')
    p_value = result.pvalue

    print('test_dwumianowy:')
    print('---------------------------------------------------------------------------')
    print(f'Hipoteza zerowa: Obserwowane częstości pasują do oczekiwanej proporcji "{nazwa_poziom}" wynoszącej {oczekiwana_proporcja}.')
    print("Hipoteza alternatywna: Obserwowane częstości nie pasują do oczekiwanej proporcji.")
    
    # Interpretacja wyniku
    if p_value < alpha:
        wynik = (f'P-wartość dla testu dwumianowego: {p_value:.4f}\nOdrzucamy hipotezę zerową, proporcja "{nazwa_poziom}" jest różna od {oczekiwana_proporcja}.')
    else:
        wynik = (f'P-wartość dla testu dwumianowego: {p_value:.4f}\nNie ma podstaw do odrzucenia hipotezy zerowej, proporcja "{nazwa_poziom}" jest równa {oczekiwana_proporcja}.')

    # Usunięcie kolumny 'binarna' utworzonej tylko do celów analizy
    del df['binarna']
    
    return wynik



#------------------------------------
#             2 GRUPY  NIEZALEŻNE
#----------------------------------


    # Porównanie - 2 grupy
    #     Testy parametryczne
    #             Test Fishera-Snedecora
    #             Test t-Studenta dla grup niezależnych
    #             Test t-Studenta z korektą Cochrana-Coxa
    #             Test t-Studenta dla grup zależnych
    #     Testy nieparametryczne
    #             Test U Manna-Whitneya
    #             Test Wilcoxona (kolejności par)
    #             Testy chi-kwadrat
    #             Test chi-kwadrat dla dużych tabel
    #             Test chi-kwadrat dla małych tabel
    #             Test Fishera dla tabel dużych tabel
    #             Poprawki testu chi-kwadrat dla małych tabel
    #             Test chi-kwadrat dla trendu
    #             Test Z dla dwóch niezależnych proporcji
    #             Test Z dla dwóch zależnych proporcji
    #             Test McNemara, test wewnętrznej symetrii Bowkera





#--------------------------
#  TEST RÓWNILICZNOŚC GRUP
#--------------------------



def test_rownowaznosci_kategorii(df, zmienna, alfa= 0.05):
    """
    Przeprowadza test chi-kwadrat na równoliczność kategorii w ramach jednej zmiennej kategorialnej.

    :param kategoria: pd.Series, zmienna kategorialna do analizy
    """
    # Liczenie obserwacji w każdej kategorii
    liczebnosc_zmiennej = df[zmienna].value_counts()

    # Oczekiwana liczebność dla każdej kategorii przy założeniu równomiernego rozkładu
    oczekiwane = [len(df[zmienna]) / len(liczebnosc_zmiennej)] * len(liczebnosc_zmiennej)

    # Przeprowadzenie testu chi-kwadrat
    chi2, p_value = stats.chisquare(liczebnosc_zmiennej, f_exp=oczekiwane)


    print(f'Test chi-kwadrat na równoliczność kategorii w zmiennej: ')
    print('-'*100)
    print(f'  Chi2 statystyka: {chi2:.4f}, p-value: {p_value:.4f}')
    print(f'  Poziom istotnosci alfa: {alfa}')

    # Interpretacja wyniku
    if p_value > alfa:
        print('Brak dowodów na istotne statystycznie różnice w liczebności kategorii.Różnica między liczebnością otrzymaną a oczekiwaną nie jest istotna')
    else:
        print('Różnica między liczebnością otrzymaną a oczekiwaną jest istotna. Istnieją dowody na istotne statystycznie różnice w liczebności kategorii.')



#-------------------------------------
# TEST NA NORMALNOSĆ W WIELU GRUPACH
#-------------------------------------


def test_normalnosc_wiele_grup(df, zmienna_kat, zmienna_num, alpha=0.05):
    kategorie = df[zmienna_kat].unique()
    if len(kategorie) < 2:
        print("Za mało kategorii do przeprowadzenia testu.")
        return

    testy = {
        "shapiro": stats.shapiro,
        "lilliefors": lambda x: sm.stats.diagnostic.lilliefors(x, 'norm'),
        "dagostino": stats.normaltest,
        #"skewness": stats.skewtest,
        #"kurtosis": stats.kurtosistest,
        "jarque-bera": stats.jarque_bera
    }

    print(f'Testowanie rozkładu zmiennej "{zmienna_num}" w poszczególnych grupach zmiennej "{zmienna_kat}"')
    print('-'*80)

    # Przygotowanie DataFrame do wyników
    kolumny = ['Test', 'Grupa', 'Statystyka', 'p-value', 'Wnioski']
    wyniki = pd.DataFrame(columns=kolumny)

    for nazwa_testu, funkcja_testu in testy.items():
        for kategoria in kategorie:
            grupa = df[df[zmienna_kat] == kategoria][zmienna_num]
            statystyka, p_value = funkcja_testu(grupa)
            wnioski = 'rozkład normalny' if p_value > alpha else 'rozkład inny niż normalny'
            wiersz_wynikow = pd.DataFrame([{'Test': nazwa_testu, 'Grupa': f'Grupa {kategoria}', 
                                            'Statystyka': statystyka, 'p-value': p_value, 'Wnioski': wnioski}])
            wyniki = pd.concat([wyniki, wiersz_wynikow], ignore_index=True)

    return wyniki



#---------------------------------------------
# TEST NA równośc WARIancji 2 PRÓBY
#----------------------------------------------


def fisher_snedecor_test(df, zmienna_kat,zmienna_num, alpha=0.05):
    from scipy.stats import f
    kategoria1 = df[zmienna_kat].unique()[0]
    kategoria2 = df[zmienna_kat].unique()[1]
    grupa1 = df[df[zmienna_kat] == kategoria1][zmienna_num]
    grupa2 = df[df[zmienna_kat] == kategoria2][zmienna_num]
    variance1 = np.var(grupa1, ddof=1)  
    variance2 = np.var(grupa2, ddof=1)
    f_statistic = variance1 / variance2
    df1 = len(grupa1) - 1
    df2 = len(grupa2) - 1
    p_value = 1 - f.cdf(f_statistic, dfn=df1, dfd=df2)
    interpretacja = "statystycznie istotne" if p_value < alpha else "brak statystycznie istotnych"
    hipoteza_zerowa = f"Wariancja  {kategoria1} w próbie jest równa wariancji {kategoria2} "
    hipoteza_alternatywna = f"Wariancja  {kategoria1} w próbie nie jest równa wariancji {kategoria2} "
    print('test rowności dwoch wariancji   -  fisher_snedecor_test:')
    print('--------------------------------------------------------')
    print(f"Hipoteza zerowa: {hipoteza_zerowa}")
    print(f"Hipoteza alternatywna: {hipoteza_alternatywna}")
    print(f"Poziom istotności: {alpha}")
    print(f'Wariancja"{kategoria1}" = {variance1:.2f}')
    print(f'Wariancja "{kategoria2}" = {variance2:.2f}')
    print(f"Wynik testu F: {f_statistic:.4f}, p-value: {p_value:.4f}")
    print(f"Interpretacja: {interpretacja}")
    return 



# #----------------------------------------------------------------
# #  ŚREDNIA 2X - ROZKŁAD NORMALNY + RÓWNE WARIANCJE
# #-----------------------------------------------------------



def test_t_dwoch_probek(df, zmienna_kat,zmienna_num, alpha=0.05):
    kategoria1 = df[zmienna_kat].unique()[0]
    kategoria2 = df[zmienna_kat].unique()[1]
    grupa1 = df[df[zmienna_kat] == kategoria1][zmienna_num]
    grupa2 = df[df[zmienna_kat] == kategoria2][zmienna_num]
    grupa1_mean= grupa1.mean()
    grupa2_mean= grupa2.mean()
    statystyka, p_wartosc = stats.ttest_ind(grupa1,grupa2)
    interpretacja = "statystycznie istotne różnice średnich" if p_wartosc < alpha else "brak statystycznie istotnych różnic średnich"
    hipoteza_zerowa = f"Średnia {zmienna_num} w grupie {kategoria1} jest równa Średniej {zmienna_num} w grupie {kategoria2}"
    hipoteza_alternatywna = f"Średnia {zmienna_num} w grupie {kategoria2}  nie jest równa Średniej {zmienna_num} w grupie {kategoria2}"
    print('test_t_dwoch_probek:')
    print('--------------------------------------------------------')
    print(f"Hipoteza zerowa: {hipoteza_zerowa}")
    print(f"Hipoteza alternatywna: {hipoteza_alternatywna}")
    print(f"Poziom istotności: {alpha}")
    print(f"Średnia {zmienna_num} w grupie: {kategoria1} = {grupa1_mean:.2f}")
    print(f"Średnia {zmienna_num} w grupie: {kategoria2} = {grupa2_mean:.2f}") 
    print(f"Wynik testu t: {statystyka:.4f}, p-value: {p_wartosc:.4f}")
    print(f"Interpretacja: {interpretacja}")
    return


def test_z_dwoch_probek(df, zmienna_kat,zmienna_num, alpha=0.05):
    kategoria1 = df[zmienna_kat].unique()[0]
    kategoria2 = df[zmienna_kat].unique()[1]
    grupa1 = df[df[zmienna_kat] == kategoria1][zmienna_num]
    grupa2 = df[df[zmienna_kat] == kategoria2][zmienna_num]
    grupa1_mean= grupa1.mean()
    grupa2_mean= grupa2.mean()
    z_statistic, p_value = stests.ztest(x1=grupa1, x2=grupa2)
    interpretacja = "statystycznie istotne różnice średnich" if p_value < alpha else "brak statystycznie istotnych różnic średnich"
    hipoteza_zerowa = f"Średnia {zmienna_num} w grupie {kategoria1} jest równa Średniej {zmienna_num} w grupie {kategoria2}"
    hipoteza_alternatywna = f"Średnia {zmienna_num} w grupie {kategoria2}  nie jest równa Średniej {zmienna_num} w grupie {kategoria2}"
    print('test_z_dwoch_probek:')
    print('--------------------------------------------------------')
    print(f"Hipoteza zerowa: {hipoteza_zerowa}")
    print(f"Hipoteza alternatywna: {hipoteza_alternatywna}")
    print(f"Poziom istotności: {alpha}")
    print(f"Średnia {zmienna_num} w grupie: {kategoria1} = {grupa1_mean:.2f}")
    print(f"Średnia {zmienna_num} w grupie: {kategoria2} = {grupa2_mean:.2f}") 
    print(f"Wynik testu z: {z_statistic:.4f}, p-value: {p_value:.4f}")
    print(f"Interpretacja: {interpretacja}")
    return




# #----------------------------------------------------------------
# #  ŚREDNIA 2X - ROZKŁAD NORMALNY  I  INNEWARIANCJE
# #-----------------------------------------------------------


import numpy as np
from scipy.stats import t

def student_t_cochran_cox(df, zmienna_kat, zmienna_num, alpha=0.05):
    """
    Przeprowadza Test t-Studenta z korektą Cochrana-Coxa w celu porównania średnich dwóch próbek z różnymi wariancjami."""
    # Oblicz średnie i wariancje dla obu próbek
    kategoria1 = df[zmienna_kat].unique()[0]
    kategoria2 = df[zmienna_kat].unique()[1]
    grupa1 = df[df[zmienna_kat] == kategoria1][zmienna_num]
    grupa2 = df[df[zmienna_kat] == kategoria2][zmienna_num]
    mean1 = np.mean(grupa1)
    mean2 = np.mean(grupa2)
    variance1 = np.var(grupa1, ddof=1)  # ddof=1 oznacza, że używamy estymatora obciążonego dla wariancji
    variance2 = np.var(grupa2, ddof=1)
    # Oblicz liczność próbek
    n1 = len(grupa1)
    n2 = len(grupa2)
    # Oblicz statystykę t
    t_statistic = (mean1 - mean2) / np.sqrt((variance1 / n1) + (variance2 / n2))
    # Oblicz stopnie swobody dla korekty Cochrana-Coxa
    df_numerator = ((variance1 / n1) + (variance2 / n2))**2
    df_denominator = ((variance1**2 / ((n1**2) * (n1 - 1))) + (variance2**2 / ((n2**2) * (n2 - 1))))
    df = df_numerator / df_denominator

    # Oblicz p-wartość
    p_value = 2 * (1 - t.cdf(np.abs(t_statistic), df))
    hipoteza_zerowa = f"Średnia {zmienna_num} w grupie {kategoria1} jest równa Średniej {zmienna_num} w grupie {kategoria2}"
    hipoteza_alternatywna = f"Średnia {zmienna_num} w grupie {kategoria2}  nie jest równa Średniej {zmienna_num} w grupie {kategoria2}"
    interpretacja = "statystycznie istotne różnice średnich" if p_value < alpha else "brak statystycznie istotnych różnic średnich"
    print('Test t-Studenta z korektą Cochrana-Coxa')
    print('-'*100)
    print(f"Hipoteza zerowa: {hipoteza_zerowa}")
    print(f"Hipoteza alternatywna: {hipoteza_alternatywna}")
    print(f"Poziom istotności: {alpha}")
    print(f"Średnia {zmienna_num} w grupie: {kategoria1} = {mean1:.2f}")
    print(f"Średnia {zmienna_num} w grupie: {kategoria2} = {mean2:.2f}") 
    print(f"Wynik testu t: {t_statistic:.4f}, p-value: {p_value:.4f}")
    print(f"Interpretacja: {interpretacja}")
    return 





# #----------------------------------------------------------------
# #  MEDIANA 2X - ROZKŁAD  INNY NIŻ NORMALNY
# #-----------------------------------------------------------


import numpy as np
from scipy.stats import mannwhitneyu

def mann_whitney_u_test(df, zmienna_kat, zmienna_num, alpha=0.05):
    """
    Przeprowadza test U Manna-Whitneya (test sumy rang Wilcoxona) dla dwóch niezależnych grup.

    :param df: DataFrame zawierający dane.
    :param zmienna_kat: Nazwa kolumny kategorycznej w df, która definiuje grupy.
    :param zmienna_num: Nazwa kolumny numerycznej w df, dla której analizowane są różnice median.
    :param alpha: Poziom istotności (domyślnie 0.05).
    :return: Wynik testu U, p-wartość i interpretacja wyniku.
    """

    kategoria1 = df[zmienna_kat].unique()[0]
    kategoria2 = df[zmienna_kat].unique()[1]
    grupa1 = df[df[zmienna_kat] == kategoria1][zmienna_num]
    grupa2 = df[df[zmienna_kat] == kategoria2][zmienna_num]
    mediana1 = np.median(grupa1)
    mediana2 = np.median(grupa2)
    statistic, p_value = mannwhitneyu(grupa1, grupa2, alternative='two-sided')
    hipoteza_zerowa = f"Mediana {zmienna_num} w grupie {kategoria1} jest równa Mediana {zmienna_num} w grupie {kategoria2}"
    hipoteza_alternatywna = f"Mediana {zmienna_num} w grupie {kategoria1} nie jest równa Mediana {zmienna_num} w grupie {kategoria2}"
    interpretacja = "statystycznie istotne różnice median" if p_value < alpha else "brak statystycznie istotnych różnic median"
    print('Test U Manna-Whitneya (test sumy rang Wilcoxona) dla dwóch niezależnych grup')
    print('-'*100)
    print(f"Hipoteza zerowa: {hipoteza_zerowa}")
    print(f"Hipoteza alternatywna: {hipoteza_alternatywna}")
    print(f"Poziom istotności: {alpha}")
    print(f"Mediana {zmienna_num} w grupie: {kategoria1} = {mediana1:.2f}")
    print(f"Mediana {zmienna_num} w grupie: {kategoria2} = {mediana2:.2f}") 
    print(f"Wynik testu U: {statistic:.4f}, p-value: {p_value:.4f}")
    print(f"Interpretacja: {interpretacja}")
    return 





#----------------------------------------------------------------
#Test Z dla dwóch niezależnych proporcji
#----------------------------------------------------------------



def test_z_dwoch_proporcji(df,zmienna_kat, alpha=0.05):
    n = df[zmienna_kat].count()
    kategoria1 = df[zmienna_kat].unique()[0]
    kategoria2 = df[zmienna_kat].unique()[1]
    grup1 = df[zmienna_kat].value_counts()[0]
    grup2 = df[zmienna_kat].value_counts()[1]
    z_statistic, p_value = proportions_ztest(count=[grup1, grup2], nobs=[n, n])
    interpretacja = "statystycznie istotne" if p_value < alpha else "brak statystycznie istotnych"
    hipoteza_zerowa = f"Proprorcja  {kategoria1} w próbie jest równa proporcji {kategoria2} "
    hipoteza_alternatywna = f"Proprorcja  {kategoria1} w próbie nie jest równa proporcji {kategoria2} "

    print('test_z_dwoch_proporcji:')
    print('--------------------------------------------------------')
    print(f"Hipoteza zerowa: {hipoteza_zerowa}")
    print(f"Hipoteza alternatywna: {hipoteza_alternatywna}")
    print(f"Poziom istotności: {alpha}")
    print(f'Proporcja "{kategoria1}" = {grup1/n:.2%}')
    print(f'Proporcja "{kategoria2}" = {grup2/n:.2%}')
    print(f"Wynik testu z: {z_statistic:.4f}, p-value: {p_value:.4f}")
    print(f"Interpretacja: {interpretacja}")
    return 





#------------------------
# TESTY CHI 2
#---------------------

#  TABELE (R  x  C)

def chi_squared_test(df, col_x, col_y, alpha=0.05):
    data = pd.crosstab(df[col_x], df[col_y])
    statistic, p_value, dof, expected = chi2_contingency(data)
    hipoteza_zerowa = f"Brak zależności między {col_x} a {col_y}"
    hipoteza_alternatywna = f"Istnieje zależność między {col_x} a {col_y}"
    interpretation = "Odrzucamy hipotezę zerową - Istnieje zależność między zmiennymi"  if p_value < alpha else "Brak podstaw do odrzucenia hipotezy zerowej - nie Istnieje zależność między {col_x} a {col_y}"
    wartosci_oczekiwane = pd.DataFrame(expected, columns=data.columns, index=data.index)
    wartosci_obserwowane = pd.DataFrame(data.values, columns=data.columns, index=data.index)
    print(f"Test Chi-Kwadrat dla zależności między {col_x} a {col_y}:")
    print('-'*70)
    print(f"Hipoteza zerowa:         {hipoteza_zerowa}")
    print(f"Hipoteza alternatywna:   {hipoteza_alternatywna}")
    print()
    print("Wartości Obserwowane:\n", wartosci_obserwowane)
    print()
    print("Wartości Oczekiwane:\n", wartosci_oczekiwane)
    print()
    print(f"Statystyka: {statistic:.2f},  p-value: {p_value:.4},   Stopnie swobody: {dof} ")
    print(f"Interpretacja: {interpretation}")
    return 


 # TABELE (2  x  2)

def fisher_test_2x2(df, col_x, col_y, alpha=0.05):
    """
    Przeprowadza test Fishera dla tabel 2x2 utworzonych na podstawie dwóch kolumn DataFrame.

    :param df: DataFrame zawierający dane.
    :param col_x: Nazwa pierwszej kolumny kategorycznej.
    :param col_y: Nazwa drugiej kolumny kategorycznej.
    :param alpha: Poziom istotności (domyślnie 0.05).
    :return: Nic, ale drukuje wyniki testu lub komunikat o błędzie.
    """
    # Sprawdzenie, czy zmienne mają dokładnie 2 poziomy
    if len(pd.unique(df[col_x])) != 2 or len(pd.unique(df[col_y])) != 2:
        print(f"Błąd: Każda z kolumn {col_x} i {col_y} musi mieć dokładnie 2 unikalne wartości.")
        return  # Zakończenie funkcji w przypadku błędu

    print(f"Test Fishera dla tabel 2x2 {col_x} a {col_y}:")
    print('-'*70)
    data = pd.crosstab(df[col_x], df[col_y])
    odds_ratio, p_value = fisher_exact(data)
    hipoteza_zerowa = "Nie ma różnicy w rozkładzie między grupami."
    hipoteza_alternatywna = "Istnieje różnica w rozkładzie między grupami."
    interpretation = "statystycznie istotne" if p_value < alpha else "brak statystycznie istotnych"
    print(f"Test Fishera dla tabeli kontyngencji 2x2:")
    print(f"Iloraz szans (Odds Ratio): {odds_ratio:.4f}, P-wartość: {p_value:.4f}")
    print(f"{hipoteza_zerowa} | {hipoteza_alternatywna}")
    print(f"Interpretacja: {interpretation} różnice między grupami.")


def chi2_yates_2x2(df, col_x, col_y, alpha=0.05):
    """
    Przeprowadza test chi-kwadrat z poprawką Yatesa dla tabel 2x2 utworzonych na podstawie dwóch kolumn DataFrame.
    Poprawka na ciągłość ma zapewnić możliwość przyjmowania przez statystykę 
    testową wszystkich wartości liczb rzeczywistych zgodnie z założeniem rozkładu chi-kwadra
    :param df: DataFrame zawierający dane.
    :param col_x: Nazwa pierwszej kolumny kategorycznej.
    :param col_y: Nazwa drugiej kolumny kategorycznej.
    :param alpha: Poziom istotności (domyślnie 0.05).
    :return: Nic, ale drukuje wyniki testu.
    """
    print(f"test chi-kwadrat z poprawką Yatesa dla tabel 2x2  {col_x} a {col_y}:")
    print('-'*70)
    data = pd.crosstab(df[col_x], df[col_y])
    chi2, p_value, dof, expected = chi2_contingency(data, correction=True)
    hipoteza_zerowa = "Nie ma różnicy w rozkładzie między grupami."
    hipoteza_alternatywna = "Istnieje różnica w rozkładzie między grupami."
    interpretation = "statystycznie istotne" if p_value < alpha else "brak statystycznie istotnych"
    print(data)
    print(f"Chi2: {chi2:.4f}, P-wartość: {p_value:.4f}, Stopnie swobody: {dof}")
    print(f"Oczekiwane wartości: {expected}")
    print(f"{hipoteza_zerowa} | {hipoteza_alternatywna}")
    print(f"Interpretacja: {interpretation} różnice między grupami.")



def midp_2x2(df, col_x, col_y):
    """
    Przeprowadza test mid-p dla tabel 2x2 utworzonych na podstawie dwóch kolumn DataFrame.

    :param df: DataFrame zawierający dane.
    :param col_x: Nazwa pierwszej kolumny kategorycznej.
    :param col_y: Nazwa drugiej kolumny kategorycznej.
    :return: Nic, ale drukuje wyniki testu.
    """
    # Tworzenie tabeli kontyngencji
    table = pd.crosstab(df[col_x], df[col_y]).to_numpy()

    # Sprawdzenie, czy tabela jest 2x2
    if table.shape != (2, 2):
        print("Funkcja wymaga tabeli kontyngencji 2x2.")
        return

    # Przeprowadzenie testu McNemara z opcją mid-p
    result = mcnemar(table, exact='midp')
    print(f"Test mid-p dla tabeli 2x2:")
    print(f"Statystyka testowa: {result.statistic:.4f}, P-wartość: {result.pvalue:.4f}")


def calculate_odds_ratio(df, col_x, col_y):
    """
    Oblicza iloraz szans na podstawie danych z dwóch kolumn DataFrame.

    :param df: DataFrame zawierający dane.
    :param col_x: Nazwa pierwszej kolumny kategorycznej.
    :param col_y: Nazwa drugiej kolumny kategorycznej.
    :return: Iloraz szans (odds ratio).
    """
    table = pd.crosstab(df[col_x], df[col_y]).to_numpy()
    a, b, c, d = table.ravel()
    odds_ratio = (a * d) / (b * c)
    print(f"Iloraz szans (Odds Ratio): {odds_ratio:.4f}")



def calculate_relative_risk(df, col_x, col_y):
    """
    Oblicza ryzyko względne na podstawie danych z dwóch kolumn DataFrame.

    :param df: DataFrame zawierający dane.
    :param col_x: Nazwa pierwszej kolumny kategorycznej.
    :param col_y: Nazwa drugiej kolumny kategorycznej.
    :return: Ryzyko względne (relative risk).
    """
    table = pd.crosstab(df[col_x], df[col_y]).to_numpy()
    a, b, c, d = table.ravel()
    risk_exposed = a / (a + b)
    risk_unexposed = c / (c + d)
    relative_risk = risk_exposed / risk_unexposed
    print(f"Ryzyko względne (Relative Risk): {relative_risk:.4f}")





#------------------------------------
#            < 2 GRUPY  NIEZALEŻNE
#----------------------------------




    # Porównanie - więcej niż 2 grupy
    #     Testy parametryczne
    #             ANOVA dla grup niezależnych
    #             Kontrasty i testy POST-HOC
    #             ANOVA dla grup niezależnych z korektą F* i F''
    #             Test Browna-Forsythea i Levenea
    #             ANOVA powtarzanych pomiarów
    #             ANOVA powtarzanych pomiarów z korektą Epsilon i MANOVA
    #             Sferyczność Mauchly’a
    #     Testy nieparametryczne
    #             ANOVA Kruskala-Wallisa
    #             Test Jonckheere-Terpstra dla trendu
    #             Test wariancji rang Conover
    #             ANOVA Friedmana
    #             Test Page dla trendu
    #             ANOVA Durbina (brakujących danych)
    #             ANOVA Skillings-Mack (brakujących danych)
    #             Test chi-kwadrat dla wielowymiarowych tabel kontyngencji
    #             ANOVA Q-Cochrana




#----------------------------------
    # RÓWNOŚĆ WARIANCJI DLA K > 2
#------------------------------------

def test_jednorodnosci_wariancji(df, zmienna_kat, zmienna_num, wybrane_testy=None, alpha=0.05):
    if wybrane_testy is None:
        wybrane_testy = ['levene', 'bartlett', 'brown-forsythe']
    grupy = [group[zmienna_num].dropna() for _, group in df.groupby(zmienna_kat)]
    wariancje = [grupa.var() for grupa in grupy]
    for i, kategoria in enumerate(df[zmienna_kat].unique()):
        print(f'Wariancja dla grupy {kategoria}: {wariancje[i]:.4f}')
    test_functions = {
        'levene': lambda g: stats.levene(*g),
        'bartlett': lambda g: stats.bartlett(*g),
        'brown-forsythe': lambda g: stats.levene(*g, center='median')
    }
    for test in wybrane_testy:
        if test in test_functions:
            stat, p_value = test_functions[test](grupy)
            interpretacja = "statystycznie istotne" if p_value < alpha else "brak statystycznie istotnych różnic"
            hipoteza_zerowa = f"Wariancje w grupach dla '{zmienna_num}' są równe"
            hipoteza_alternatywna = f"Wariancje w grupach dla '{zmienna_num}' nie są równe"
            print(f"\nTest {test.capitalize()} na jednorodność wariancji:")
            print('--------------------------------------------------------')
            print(f"Hipoteza zerowa: {hipoteza_zerowa}")
            print(f"Hipoteza alternatywna: {hipoteza_alternatywna}")
            print(f"Poziom istotności: {alpha}")
            print(f"Statystyka: {stat:.4f}, p-value: {p_value:.4f}")
            print(f"Interpretacja: {interpretacja}")
        else:
            print(f'Nieznany test: {test}')



#-----------------------------------------------------------------------
#       ANALIZA WARIANCJI ANOVA  - 1 czynnikowa  DLA K > 2  i RÓWNE WARIANCJE 
#---------------------------------------------------------------------------



def wykonaj_anova(df, zmienna_kat, zmienna_num):
    formula = f'{zmienna_num} ~ C({zmienna_kat})'
    model = ols(formula, data=df).fit()
    anova_results = sm.stats.anova_lm(model, typ=2)
    pd.options.display.float_format = '{:.4f}'.format 
    # Objasnienia wyników
    print(f"Jednoczynnikowa analiza wariancji Anova między zmiennymi:  {formula}:")
    print('-'*80)
    print(anova_results)
    print()
    print(anova_lm(ols(formula, df).fit()))
    print()
    print(model.summary())
    # Interpretacja p-value
    p_value = anova_results['PR(>F)'][0]
    if p_value < 0.05:
        print(f'\nRóżnice między grupami są statystycznie istotne (p = {p_value:.6f}).')
        print(f'Oznacza to, że analiza wykazała, że średnie zmiennej: {zmienna_num} różnią się między co najmniej dwoma grupami zmiennej {zmienna_kat}. ')
        print(f'Ta istotność statystyczna sugeruje, że zmienna {zmienna_kat} ma wpływ na wartości zmiennej {zmienna_num}')

    else:
        print(f'\nBrak statystycznie istotnych różnic między grupami (p = {p_value:.6f}).')
        print("To wskazuje, że nie ma wystarczających dowodów, by stwierdzić, iż ocena wpływa na dochody w populacji. Możliwe, że inne zmienne mają większy wpływ na dochody lub że potrzebna jest większa próba danych do analizy.")

        print("\nObjaśnienia wyników:")
    print()
    print("sum_sq - suma kwadratów dla danego źródła zmienności, pokazuje całkowitą zmienność wyjaśnianą przez model.")
    print("df - stopnie swobody skojarzone z danym źródłem zmienności.")
    print("F - statystyka F, mierzy stosunek wariancji między grupami do wariancji wewnątrz grup.")
    print("PR(>F) - p-wartość dla testu F, mówi o prawdopodobieństwie obserwacji takich lub bardziej ekstremalnych wyników, jeśli hipoteza zerowa byłaby prawdziwa.")
    print("Residual - rezydualna suma kwadratów, pokazuje zmienność nie wyjaśnioną przez model.")
    print("C(zmienna_kat) - oznacza kategoryzację zmiennej 'zmienna_kat', która jest używana do podziału danych na grupy dla analizy.")



#-----------------------------------------------------------------------
#       ANALIZA WARIANCJI ANOVA  - 2 czynnikowa  DLA K > 2  i RÓWNE WARIANCJE 
#---------------------------------------------------------------------------



def wykonaj_anova_2_czynnikowa(df, zmienna_kat1, zmienna_kat2, zmienna_num,alpha = 0.95):
    formula = f'{zmienna_num} ~ C({zmienna_kat1}) + C({zmienna_kat2}) + C({zmienna_kat1}):C({zmienna_kat2})'
    model = ols(formula, data=df).fit()
    anova_results = sm.stats.anova_lm(model, typ=2)
    pd.options.display.float_format = '{:.4f}'.format
    # Objasnienia wyników
    print(f"Dwuczynnikowa analiza wariancji Anova między zmiennymi: {formula}:")
    print('-'*80)
    print(anova_results)
    print(model.summary())
    # Interpretacja p-value
    p_value_main_effect_1 = anova_results['PR(>F)'][0]
    p_value_main_effect_2 = anova_results['PR(>F)'][1]
    p_value_interaction = anova_results['PR(>F)'][2]
    
    interpretacja = ""
    if p_value_main_effect_1 < alpha:
        interpretacja += f'\nEfekt główny zmiennej {zmienna_kat1} jest statystycznie istotny (p = {p_value_main_effect_1:.6f}).'
    if p_value_main_effect_2 < alpha:
        interpretacja += f'\nEfekt główny zmiennej {zmienna_kat2} jest statystycznie istotny (p = {p_value_main_effect_2:.6f}).'
    if p_value_interaction < alpha:
        interpretacja += f'\nInterakcja między zmiennymi {zmienna_kat1} i {zmienna_kat2} jest statystycznie istotna (p = {p_value_interaction:.6f}).'
    
    if interpretacja == "":
        interpretacja = "Brak statystycznie istotnych efektów głównych ani interakcji."
    
    print(interpretacja)




#--------------------------------------------------------------------------------
#     ANALIZA WARIANCJI ANOVA  1 czynnikowa DLA K > 2  Z KOREKTĄ DLA RÓZNYCH WARIANCJI
#-----------------------------------------------------------------------------

def wykonaj_anova_z_korekta(df, zmienna_kat, zmienna_num):
    """
    Wykonuje jednoczynnikową analizę wariancji ANOVA z korektą dla nierównych wariancji (np. korekta Welch'a).
    
    :param df: DataFrame zawierający dane.
    :param zmienna_kat: Nazwa kolumny kategorycznej w DataFrame, która definiuje grupy.
    :param zmienna_num: Nazwa kolumny numerycznej w DataFrame, której różnice średnich są testowane.
    """
    # Przeprowadzenie ANOVA z korektą dla nierównych wariancji
    results = anova_oneway(df[zmienna_num], df[zmienna_kat], use_var='unequal', welch_correction=True)
    
    print("Wyniki ANOVA z korektą dla nierównych wariancji:")
    print(results)
    
    # Interpretacja p-value
    if results.pvalue < 0.05:
        print(f"\nRóżnice między grupami są statystycznie istotne (p = {results.pvalue:.6f}).")
        print(f"Oznacza to, że analiza wykazała, że średnie zmiennej: {zmienna_num} różnią się między co najmniej dwoma grupami zmiennej {zmienna_kat}.")
        print(f"Ta istotność statystyczna sugeruje, że zmienna {zmienna_kat} ma wpływ na wartości zmiennej {zmienna_num}.")
    else:
        print(f"\nBrak statystycznie istotnych różnic między grupami (p = {results.pvalue:.6f}).")
        print("To wskazuje, że nie ma wystarczających dowodów, by stwierdzić, iż zmienna kategoryczna wpływa na wartości zmiennej numerycznej.")





#-------------------------------------------------
#    KONTRASTY I TESTY   POST-HOC
#-----------------------------------------------


def wykonaj_test_post_hoc(df, zmienna_num, zmienna_kat):
    # Przeprowadzenie testu Tukey's HSD
    tukey = pairwise_tukeyhsd(endog=df[zmienna_num], groups=df[zmienna_kat], alpha=0.05)
    dane = df
    print("Wyniki testu Tukey's HSD:")
    print(tukey)
    print(dane.pairwise_tukey(dv='dochody', between='ocena').round(3))





#--------------------------------------------------------------------------
#       ANALIZA WARIANCJI ANOVA  DLA K > 2   + INNA ROZKŁAD NIŻ NORMALNY
#------------------------------------------------------------------------


def kruskal_wallis_test(df, col_group, col_value):
    """
    Przeprowadza jednoczynnikową analizę wariancji dla rang Kruskala-Walisa.

    :param df: DataFrame zawierający dane.
    :param col_group: Nazwa kolumny w DataFrame, która zawiera identyfikatory grup.
    :param col_value: Nazwa kolumny w DataFrame, która zawiera zmierzone wartości.
    :return: Statystyka testowa, p-wartość i stopnie swobody.
    """
    # Utworzenie listy z wartościami dla każdej z grup
    groups_values = df.groupby(col_group)[col_value].apply(list).values.tolist()
    
    # Przeprowadzenie testu Kruskala-Wallisa
    statistic, p_value = stats.kruskal(*groups_values)
    
    # Obliczenie stopni swobody
    degrees_of_freedom = len(groups_values) - 1
    
    # Wyświetlenie wyników
    print(f'Statystyka H = {statistic}   p-value = {p_value}, df = {degrees_of_freedom}')
    if p_value < 0.05:
        print("Istnieją statystycznie istotne różnice między grupami.")
    else:
        print("Brak statystycznie istotnych różnic między grupami.")

    
    return 



#---------------------------------------------------------------
#                              STATYSTYKA OPISOWA 
#--------------------------------------------------------------------


import pandas as pd

def informacje_o_dataframe(df):
    informacje = []

    for kolumna in df.columns:
        typ_kolumny = df[kolumna].dtype
        unikalne_wartosci = df[kolumna].nunique()
        puste_wartosci = df[kolumna].isnull().sum()
        niepuste_wartosci = df[kolumna].count()

        if typ_kolumny == 'object':
            wartosci_unikalne = df[kolumna].unique()
            liczba_zer = None
            rodzaj_kolumny = 'Tekstowa'
        else:
            wartosci_unikalne = None
            liczba_zer = df[kolumna].tolist().count(0)
            rodzaj_kolumny = 'Numeryczna'

        informacje.append([kolumna, typ_kolumny, unikalne_wartosci, puste_wartosci, niepuste_wartosci, wartosci_unikalne, liczba_zer, rodzaj_kolumny])

    informacje_df = pd.DataFrame(informacje, columns=['Nazwa kolumny', 'Typ kolumny', 'Liczba unikalnych wartości', 'Liczba wartości pustych', 'Liczba wartości niepustych', 'Wartości unikalne', 'Liczba zer', 'Rodzaj kolumny'])
    return informacje_df





import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def braki_sprawdzenie(dane):
    liczba = dane.isnull().sum().sum()
    proc = (liczba / (dane.shape[0]*dane.shape[1])*100).round(2)
    
    if liczba == 0:
        st.write('Analiza brakujących danych:')
        st.write('='*45)
        st.write('W tabeli nie stwierdzono brakujących danych!')
    else:
        st.write('Analiza brakujących danych:')
        st.write('='*45)
        st.write(f'Liczba brakujących danych w tabeli: {liczba}')
        st.write(f'Procent brakujących danych w tabeli: {proc}%')
        st.write('='*45)
        
        rows_with_missing_data = dane[dane.isnull().any(axis=1)]
        brakujace_dane = rows_with_missing_data.isnull().sum(axis=0)
        udzial_brakujacych_danych = ((rows_with_missing_data.isnull().sum(axis=0) / dane.shape[0])*100).round(1)
        wyniki = pd.DataFrame({'liczba': brakujace_dane, 'proc': udzial_brakujacych_danych})
        
        st.write('Brakujące dane w zmiennych (kolumny):')
        st.dataframe(wyniki)

        rows_with_missing_data = dane[dane.isnull().any(axis=1)]
        brakujace_dane = rows_with_missing_data.isnull().sum(axis=1)
        udzial_brakujacych_danych = (rows_with_missing_data.isnull().sum(axis=1) / dane.shape[1]*100).round(1)
        wyniki = pd.DataFrame({'liczba': brakujace_dane, 'proc': udzial_brakujacych_danych})
        
        st.write('='*45)
        st.write('Brakujące dane w obserwacjach (wiersze):')
        st.dataframe(wyniki)

        fig, ax = plt.subplots(figsize=(9.5, 4))
        sns.heatmap(dane.isnull(), cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        rows_with_missing_data = dane[dane.isnull().any(axis=1)]
        st.write('Tabela z brakującymi danymi:')
        st.dataframe(rows_with_missing_data)


def outliers(df, var):
    data = df[var]
 
    def detect_outliers_iqr(data):
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        outliers_IQR = [i for i, x in enumerate(data) if x < lower_bound or x > upper_bound]
        return outliers_IQR

    def detect_outliers_mean_std(data):
        mean = np.mean(data)
        std = np.std(data)
        lower_bound = mean - (3 * std)
        upper_bound = mean + (3 * std)
        outliers_mean_std = [i for i, x in enumerate(data) if x < lower_bound or x > upper_bound]
        return outliers_mean_std

    def detect_outliers_zscore(data):
        threshold = 2
        z_scores = zscore(data)
        outliers_zscore = [i for i, z in enumerate(z_scores) if abs(z) > threshold]
        return outliers_zscore

    def detect_outliers_winsorizing(data):
        lower_bound, upper_bound = np.percentile(data, [5, 95])
        data = np.clip(data, lower_bound, upper_bound)
        outliers_winsorizing = [i for i, x in enumerate(data) if x < lower_bound or x > upper_bound]
        return outliers_winsorizing

    # Reshape the data to a 2D array
    data = np.array(data).reshape(-1, 1)

    outliers_IQR = detect_outliers_iqr(data)
    outliers_mean_std = detect_outliers_mean_std(data)
    outliers_zscore = detect_outliers_zscore(data)
    outliers_winsorizing = detect_outliers_winsorizing(data)

    df = pd.DataFrame(data)
    df['Odstające_IQR'] = np.where(df.index.isin(outliers_IQR), -1, 1)
    df['Odstające_mean_std'] = np.where(df.index.isin(outliers_mean_std), -1, 1)
    df['Odstające_Zscore'] = np.where(df.index.isin(outliers_zscore), -1, 1)
    df['Odstające_Winsorizing'] = np.where(df.index.isin(outliers_winsorizing), -1, 1)
   
    # Dodatkowa kolumna: Czy_Odstające
    df['Czy_Odstające'] = np.where((df['Odstające_IQR'] == -1) |
                                (df['Odstające_mean_std'] == -1) |
                                (df['Odstające_Zscore'] == -1) |
                                (df['Odstające_Winsorizing'] == -1) , True, False)
    df = df.loc[df['Czy_Odstające'] == True]
    df = df.rename(columns={0: var})  # Zmiana nazwy kolumny 0 na Nazwa_Zmiennej
    
    # Dodaj kolumnę "ile_ident"
    df['ile_ident'] = df[['Odstające_IQR', 'Odstające_mean_std', 'Odstające_Zscore', 'Odstające_Winsorizing']].apply(lambda row: row.value_counts().get(-1, 0), axis=1)
    
    return df




# Tabele

#     Tabele liczności i rozkłady empiryczne
       #Statystyka opisowa





#---------------------------
#  ( 1 ) ZMIENNA NUMERYCZNA
#--------------------------



def tabela_licz_n(df, zmienna, bin):
    # Utworzenie przedziałów z dokładnością do dwóch miejsc po przecinku
    # i obliczenie częstości występowania wartości w tych przedziałach
    tab = pd.DataFrame(df[zmienna].value_counts(bins=bin, normalize=True).sort_index().round(4) * 100)
    tab.columns = ['częstość']  # Zmiana nazwy kolumny na 'częstość'
    
    # Dodanie kolumny z liczbą wystąpień (liczba n)
    tab['liczba n'] = df[zmienna].value_counts(bins=bin).sort_index()
    
    # Obliczenie liczby skumulowanej
    tab['licz skum'] = tab['liczba n'].cumsum()
    
    # Obliczenie częstości skumulowanej
    tab['częstość skum'] = tab['częstość'].cumsum()
    
    # Formatowanie przedziałów z dokładnością do dwóch miejsc po przecinku
    tab.index = tab.index.map(lambda x: f"({x.left:.2f}, {x.right:.2f}]")
    tab.index.name = 'Przedziały'
    
    return tab



import numpy as np

import numpy as np
import pandas as pd

def obszary_zm(df, zmienna):
    x = df[zmienna].dropna()  # Usuwanie braków danych
    odch_std = np.std(x)
    srednia = np.mean(x)
    Me = np.median(x)
    Q = (np.quantile(x, .75) - np.quantile(x, .25)) / 2
    kl_obszar_zm_L = srednia - odch_std
    kl_obszar_zm_P = srednia + odch_std
    poz_obszar_zm_L = Me - Q
    poz_obszar_zm_P = Me + Q

    # Tworzenie DataFrame z wynikami, zaokrąglone do dwóch miejsc po przecinku
    wyniki_df = pd.DataFrame({
        'Metoda': ['klasyczny_obszar_zm', 'pozycyjny_obszar_zm'],
        'Lewa granica': [round(kl_obszar_zm_L, 2), round(poz_obszar_zm_L, 2)],
        'Prawa granica': [round(kl_obszar_zm_P, 2), round(poz_obszar_zm_P, 2)]
    })

    return wyniki_df




def stat(data, zmienne):
                    wyniki_all = {}
                    for zmienna in zmienne:
                        df = data[zmienna]
                        średnia = df.mean()
                        odch_std = df.std(ddof=1)
                        rozstęp = df.max() - df.min()
                        Q1 = df.quantile(.25)
                        Q2 = df.quantile(.5)
                        Q3 = df.quantile(.75)
                        IQR = Q3 - Q1
                        moda = df.mode().iloc[0] if not df.mode().empty else np.nan
                        std = df.std(ddof=1)
                        wyniki = {
                            'liczba': df.count(),
                            'suma': df.sum(),
                            'min': df.min(),
                            'max': df.max(),
                            'średnia': średnia,
                            'rozstęp': rozstęp,
                            'p_10%': df.quantile(.1),
                            'Q1_25%': Q1,
                            'Q2_50%': Q2,
                            'Q3_75%': Q3,
                            'p_90%': df.quantile(.9),
                            'IQR': IQR,
                            'odch_cwiar': IQR / 2,
                            'odchylenie przeciętne': np.mean(np.abs(df - średnia)) / średnia * 100,
                            'wariancja': df.var(ddof=1),
                            'odch_std': odch_std,
                            'błąd_odch_std': odch_std / np.sqrt(df.count()),
                            'kl_wsp_zmien': odch_std / średnia,
                            'poz_wsp_zmien': IQR / Q2,
                            'moda': moda,
                            'skośność': df.skew(),
                            'kurtoza': df.kurtosis()
                        }
                        wyniki_all[zmienna] = wyniki
                    wyniki_df = pd.DataFrame(wyniki_all).round(2)
                    pd.options.display.float_format = '{:.2f}'.format
                    return wyniki_df.T




def stat_kat(data, zmienna_numeryczna, zmienna_kategoryczna):
    wyniki_all = {}
    grouped_data = data.groupby(zmienna_kategoryczna)[zmienna_numeryczna]
    for group_name, group_data in grouped_data:
        średnia = group_data.mean()
        odch_std = group_data.std(ddof=1)
        rozstęp = group_data.max() - group_data.min()
        Q1 = group_data.quantile(.25)
        Q2 = group_data.quantile(.5)
        Q3 = group_data.quantile(.75)
        IQR = Q3 - Q1
        moda = group_data.mode().iloc[0] if not group_data.mode().empty else np.nan
        wyniki = {
            'liczba': group_data.count(),
            'suma': group_data.sum(),
            'min': group_data.min(),
            'max': group_data.max(),
            'średnia': średnia,
            'rozstęp': rozstęp,
            'p_10%': group_data.quantile(.1),
            'Q1_25%': Q1,
            'Q2_50%': Q2,
            'Q3_75%': Q3,
            'p_90%': group_data.quantile(.9),
            'IQR': IQR,
            'odch_cwiar': IQR / 2,
            'odchylenie przeciętne': np.mean(np.abs(group_data - średnia)) / średnia * 100,
            'wariancja': group_data.var(ddof=1),
            'odch_std': odch_std,
            'błąd_odch_std': odch_std / np.sqrt(group_data.count()),
            'kl_wsp_zmien': odch_std / średnia,
            'poz_wsp_zmien': IQR / Q2,
            'moda': moda,
            'skośność': group_data.skew(),
            'kurtoza': group_data.kurtosis()
        }
        wyniki_all[group_name] = wyniki
    return wyniki_all



#---------------------------
#  ( 1 ) ZMIENNA KATEGORIALNA
#--------------------------



def cat_stat(df, zm):
    statystyki = {
        'Podstawowe statystyki zmiennej': [zm],
        'liczba obs': [df[zm].count()],
        'liczba poziomów': [df[zm].nunique()],
        'poziomy': [list(df[zm].unique())],
        'top': [df[zm].describe()[2]]
    }
    wyniki_df = pd.DataFrame(statystyki)
    return wyniki_df


def print_frequency_table(df, column_name):
    print('='*50)
    frequency_table = pd.DataFrame(df[column_name].value_counts())
    frequency_table.columns = ['Liczba']
    frequency_table['f %'] = (frequency_table['Liczba'] / frequency_table['Liczba'].sum() * 100).round(2)
    frequency_table['Liczba skum.'] = frequency_table['Liczba'].cumsum()
    frequency_table['f % skum.'] = frequency_table['f %'].cumsum().round(2)
    return frequency_table


import pandas as pd
import numpy as np

def analyze_categorical_data(data, category_column):
    frequencies = data[category_column].value_counts()
    total_cases = frequencies.sum()
    proportions = frequencies / total_cases
    entropy = -np.sum(proportions * np.log2(proportions + np.finfo(float).eps))
    normalized_entropy = entropy / np.log2(len(frequencies))
    
    N = total_cases
    K = len(frequencies)
    fm = frequencies.max()
    ModVR = K * (N - fm) / (N * (K - 1))
    v = 1 - fm / N
    AvDev = 1 - (1 / (2 * N)) * (K / (K - 1)) * sum(abs(fi - N / K) for fi in frequencies)
    MNDif = 1 - (1 / (N * (K - 1))) * sum(abs(fi - fj) for i, fi in enumerate(frequencies) for fj in list(frequencies)[i+1:])
    VarNC = 1 - (1 / (N**2)) * (K / (K - 1)) * sum((fi - N / K)**2 for fi in frequencies)
    
    p = proportions.values
    M1 = 1 - np.sum(p**2)
    M2 = (K / (K - 1)) * (1 - np.sum(p**2))
    IC = np.sum(p * (p - 1)) / (N * (N - 1))
    
    data_dict = {
        'IQV (na podstawie Entropii Informacyjnej)': [normalized_entropy],
        'ModVR': [ModVR],
        "Freeman's index (v)": [v],
        'AvDev': [AvDev],
        'MNDif': [MNDif],
        'VarNC': [VarNC],
        'M1 (Indeks Gibbsa)': [M1],
        'M2 (Indeks Gibbsa)': [M2],
        'Incidence of Coincidence (IC)': [IC]
    }
    
    results_df = pd.DataFrame(data_dict)
    return results_df





#------------------------------------------------------------
#  ( 1 ) ZMIENNA NUMERYCZNA  x   (N) ZMIENNA KATEGORIALNA
#----------------------------------------------------------


def korelacje_nom_num(df, var1, var2):
    """ df - tabela danych  
    var1 - zmienna nominalna 2 poziomowa   
    var2 - zmienna numeryczna wynik ->(-1,1) """
    groupby_cat = df.groupby(var1)[var2].mean()
    y0 = groupby_cat.iloc[0]
    y1 = groupby_cat.iloc[1]
    p = df[var1].value_counts() / df.shape[0]
    std = df[var2].std()
    Point_Biserial= np.round((y1 - y0) * np.sqrt(p[1] * (1 - p[1])) / std,3)
    print(f'współczynnik korelacji pomiędzy [{var1}] a [{var2}]  --> Point_Biserial = {Point_Biserial}')






def korelacje_num2_nom(df, method, var_nom, *vars_num):
    """
    pearson : standard correlation coefficient
    kendall : Kendall Tau correlation coefficient
    spearman : Spearman rank correlation
    """
    print(f'Macierz Korelacji  [metoda: "{method}"]')
    
    # Sprawdzamy, czy podano co najmniej dwie zmienne numeryczne do analizy
    if len(vars_num) < 2:
        print("Podaj co najmniej dwie zmienne numeryczne.")
        return None
    
    # Tworzymy pustą listę na wyniki
    results = []
    
    # Iterujemy przez każdą grupę określoną przez zmienną nominalną
    for name, group in df.groupby(var_nom):
        corr_val = group[vars_num[0]].corr(group[vars_num[1]], method=method).round(2)
        results.append((name, corr_val))
    
    # Tworzenie DataFrame z wynikami
    result_df = pd.DataFrame(results, columns=[var_nom, 'Korelacja'])
    
    return result_df




#------------------------------------------------------------
#  ( 1 ) ZMIENNA NUMERYCZNA  x    1) ZMIENNA NUMERYCZNA
#----------------------------------------------------------





def korelacje_numeryczne(df, method='pearson', *vars):
    """
    Oblicza korelację pomiędzy wybranymi zmiennymi numerycznymi w DataFrame.

    :param df: DataFrame zawierający dane.
    :param vars: Lista zmiennych, między którymi obliczana będzie korelacja.
    :param method: Metoda korelacji ('pearson', 'kendall', 'spearman').
    :return: DataFrame z macierzą korelacji.
    """
    # Sprawdzanie, czy podano zmienne
    if not vars:
        print("Nie podano zmiennych.")
        return None
    
    # Wybór kolumn
    df_selected = df[list(vars)]
    
    # Obliczenie macierzy korelacji
    corr_matrix = df_selected.corr(method=method).round(2)
    
    return corr_matrix

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def analiza_regresji(df, zmienna1, zmienna2):
    # Przygotowanie danych
    X = df[zmienna1]
    y = df[zmienna2]
    X = sm.add_constant(X)  # dodanie stałej do modelu
    
    # Budowanie modelu regresji liniowej
    model = sm.OLS(y, X).fit()
    
    # Wyświetlanie podsumowania modelu
    st.subheader("Podsumowanie modelu regresji")
    st.text(model.summary())
    
    # Wizualizacja zależności liniowej
    st.subheader("Zależność liniowa")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x=df[zmienna1], y=df[zmienna2], ci=None, line_kws={'color': 'red'}, ax=ax)
    st.pyplot(fig)
    
    # Testowanie normalności reszt za pomocą testu Shapiro-Wilka
    _, p_value = stats.shapiro(model.resid)
    st.write(f"Test Shapiro-Wilka p-wartość: {p_value} (Normalność reszt, p > 0.05 oznacza normalność)")
    
    # Sprawdzenie homoscedastyczności reszt
    _, p_value = stats.levene(model.fittedvalues, model.resid)
    st.write(f"Test Levene'a p-wartość: {p_value} (Homoscedastyczność, p > 0.05 oznacza homoscedastyczność)")
    
    # Wizualizacja reszt
    st.subheader("Reszty vs. Dopasowane wartości")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(model.fittedvalues, model.resid)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_title('Reszty vs. Dopasowane wartości')
    ax.set_xlabel('Dopasowane wartości')
    ax.set_ylabel('Reszty')
    st.pyplot(fig)
    
    # Sprawdzenie autokorelacji reszt za pomocą testu Durbin-Watson
    dw = sm.stats.durbin_watson(model.resid)
    st.write(f"Test Durbin-Watson: {dw} (2 oznacza brak autokorelacji, wartości <1 lub >3 wskazują na autokorelację)")





#------------------------------------------------------------
#  ( 1 ) ZMIENNA KATEGORIALNA  x    1) KATEGORIALNA
#----------------------------------------------------------




def rozklady_cat(df, cat1, cat2, widok):

    if widok == 'licz_all':
                    print('liczebności:')
                    t = pd.crosstab(df[cat1], df[cat2], margins=True, margins_name="Razem")
    elif widok == 'proc_all': 
                    print('częstości całkowite %:')
                    t = (pd.crosstab(df[cat1], df[cat2], dropna=False, normalize='all', margins=True, margins_name='suma')*100).round(2)
    elif widok == 'proc_col':
                    print('częstości wg kolumn %: ')
                    t = (pd.crosstab(df[cat1], df[cat2], dropna=False, normalize='columns')*100).round(2)
    elif widok == 'proc_row':
                    print('częstosci wg wierszy %:')
                    t = (pd.crosstab(df[cat1], df[cat2], dropna=False, normalize='index', margins_name='suma')*100).round(2)
    return t


import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def korelacje_nom(df, var1, var2):
    """ 'phi' - phi Yule'a, 'cp' - C-Pearsona, 'v' - V-Cramera,'t' - T-Czuprowa, 'c' - Cohena """
    table = pd.crosstab(df[var1], df[var2])
    chi2, p, dof, expected = chi2_contingency(table)
    N = df.shape[0]  # liczba elementów/obserwacji
    r = table.shape[0]  # liczba wierszy w tabeli kontyngencji
    k = table.shape[1]  # liczba kolumn w tabeli kontyngencji
    phi = np.round(np.sqrt(chi2 / N), 3)
    C_pearson = np.round(np.sqrt(chi2 / (chi2 + N)), 3)
    V_cramer = np.round(np.sqrt(chi2 / (N * min(k - 1, r - 1))), 3)
    T_czuprow = np.round(np.sqrt(chi2 / (N * np.sqrt((r - 1) * (k - 1)))), 3)
    Cohen = np.round(V_cramer * np.sqrt(min(k - 1, r - 1) - 1), 3)
    
    # Tworzenie listy wyników
    results = [{'Measure': 'Phi Yule\'a', 'Value': phi},
               {'Measure': 'C-Pearsona', 'Value': C_pearson},
               {'Measure': 'V-Cramera', 'Value': V_cramer},
               {'Measure': 'T-Czuprowa', 'Value': T_czuprow},
               {'Measure': 'Cohena', 'Value': Cohen}]
    
    # Tworzenie ramki danych z wynikami
    result_df = pd.DataFrame(results)
    
    return result_df












import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white", font='Arial', font_scale=4.1, color_codes=False)
sns.set_context("paper")
palette = sns.husl_palette(9)
sns.set_palette(palette)
sns.set_color_codes("deep")


# def hist_plot(df, zmienna_num, cumulative = False):
#     plt.figure(figsize=(6, 3)) 
#     # count: the number of observations
#     # density: normalize so that the total area of the histogram equals 1
#     # percent: normalize so that bar heights sum to 100
#     # probability or proportion: normalize so that bar heights sum to 1
#     # frequency: divide the number of observations by the bin width
#     if cumulative == False:
#         sns.histplot(df[zmienna_num], bins=8, kde=False, stat = 'count', color = "#3498db", fill=True)
#         #sns.rugplot(df[zmienna])
#     else:   
#         sns.histplot(df[zmienna_num], bins=8, kde=False, stat = 'count', cumulative = True,color = "#3498db", fill=True)
#     sns.despine()
#     plt.show()


import streamlit as st
def hist_plot(df, zmienna_num, stat,cumulative=False):
    ''''
    count: the number of observations
    density: normalize so that the total area of the histogram equals 1
    percent: normalize so that bar heights sum to 100
    probability or proportion: normalize so that bar heights sum to 1
    frequency: divide the number of observations by the bin width
    '''

    fig, ax = plt.subplots()
    if cumulative == False:
        sns.histplot(df[zmienna_num], bins=8, kde=False, stat=stat, color="#3498db", fill=True)
    else:
        sns.histplot(df[zmienna_num], bins=8, kde=False, stat=stat, cumulative=True, color="#3498db", fill=True)
    sns.despine()
    st.pyplot(fig)
    








def kde_plot(df, zmienna, cumulative = False):
    fig, ax = plt.subplots()
    # count: the number of observations
    # density: normalize so that the total area of the histogram equals 1
    # percent: normalize so that bar heights sum to 100
    # probability or proportion: normalize so that bar heights sum to 1
    # frequency: divide the number of observations by the bin width
    if cumulative == False:
        sns.kdeplot(df[zmienna],  color = "#3498db", fill=True)
    else:   
        sns.kdeplot(df[zmienna], cumulative = True,color = "#3498db", fill=True)
    sns.despine()
    # plt.title('Tytuł wykresu') 
    # plt.xlabel('Oś X')  
    # plt.ylabel('Oś Y') 
    st.pyplot(fig)


def ecdf_plot(df, zmienna):
    fig, ax = plt.subplots() 
    # stat{{“proportion”, “percent”, “count”}}
    sns.ecdfplot(df[zmienna],color = "#3498db", stat = 'proportion')
    sns.despine()
    st.pyplot(fig)


def box_plot(df, zmienna):
    fig, ax = plt.subplots() 
    sns.boxplot(df[zmienna], orient='h',  color="#3498db", linewidth=1, fill=True, saturation = 10)
    sns.despine()
    st.pyplot(fig)

def violin_plot(df, zmienna):
    fig, ax = plt.subplots() 
    sns.violinplot(df[zmienna], orient='h',fill=False, inner_kws=dict(box_width=15, whis_width=2, color=".8"),  color="#3498db" )
    sns.despine()
    st.pyplot(fig)

def swarm_plot(df, zmienna):
    fig, ax = plt.subplots() 
    sns.swarmplot(data=df, x=zmienna, marker=".", linewidth=1,size=2,  edgecolor="#3498db")
    sns.despine()
    st.pyplot(fig)










# def ci_plot(df, zmienna, estymator):
#     plt.figure(figsize=(6, 3)) 
#     sns.pointplot( df, zmienna,estimator=estymator, errorbar=('ci', 95), capsize=.4, color=".5")
#     sns.despine()
#     return

# ci_plot(dane, 'dochody', 'mean')





# palette = sns.color_palette("husl", 8)
# sns.displot(dane, x="dochody", kind="hist", col = 'ocena',  height=3, color="#3498db", bins = 8, kde = True)
# sns.displot(dane, x="dochody", kind="ecdf", col = 'ocena',  height=3, color="#3498db")
# sns.catplot(dane, x="dochody", y="ocena", kind="box", height=4, palette=palette)
# sns.catplot(dane, x="dochody", y="ocena", kind="violin", height=4, palette=palette)
# sns.catplot(dane, x="dochody", y="ocena", kind="point", height=4, palette=palette, estimator='mean', errorbar=("ci",95))
# sns.catplot(dane, x="dochody", y="ocena", kind="bar", height=4, palette=palette, estimator="sum")
# sns.catplot(dane, x="dochody", y="ocena", kind="swarm", height=4, palette=palette,  marker=".", linewidth=1,size=2,  edgecolor="#3498db")
# sns.catplot(dane, x="dochody", y="ocena", kind="boxen",color="#3498db",height=4)

# # import seaborn as sns
# # import matplotlib.pyplot as plt



def category_one_plot(df, zmienna, stat):
    '''
    stat{'count', 'percent', 'proportion', 'probability'}
    '''
    plt.figure(figsize=(6, 3)) 
    palette = sns.color_palette("husl", df[zmienna].nunique())
    ax = sns.countplot(data=df, x=zmienna, palette=palette,stat = stat)
    for container in ax.containers:
        ax.bar_label(container, fontsize=8, color='gray')
    sns.despine()

import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def category_two_plot(df, zmienna1, zmienna2, stat, facetgrid=False):
    # stat{'count', 'percent', 'proportion', 'probability'}
    if facetgrid:
        g = sns.FacetGrid(data=df, col=zmienna1)
        
        def countplot_with_labels(data, **kwargs):
            ax = sns.countplot(x=zmienna2, data=data, palette="husl", stat=stat, **kwargs)
            for container in ax.containers:
                ax.bar_label(container, fontsize=8, color='gray')
                
        g.map_dataframe(countplot_with_labels)
        g.add_legend()
        plt.tight_layout()
        st.pyplot(plt)  # Wyświetlenie wykresu w Streamlit
    else:
        plt.figure(figsize=(6, 3)) 
        palette = sns.color_palette("husl", df[zmienna2].nunique())
        ax = sns.countplot(data=df, x=zmienna1, stat=stat, hue=zmienna2, palette=palette)
        for container in ax.containers:
            ax.bar_label(container, fontsize=8, color='gray')
        sns.despine()
        st.pyplot(plt)  # Wyświetlenie wykresu w Streamlit



 
# def category_three_plot(df, zmienna1, zmienna2, zmienna3, stat):
#     #stat{‘count’, ‘percent’, ‘proportion’, ‘probability’}
    
#         g = sns.FacetGrid(data=df, col=zmienna1)
#         def countplot_with_labels(data, **kwargs):
#             # Dodanie zmiennej3 jako hue w countplot
#             ax = sns.countplot(x=zmienna2, hue=zmienna3, data=data, palette="husl", stat=stat, **kwargs)
#             for container in ax.containers:
#                 ax.bar_label(container, fontsize=8, color='gray')
#         g.map_dataframe(countplot_with_labels)
#         g.add_legend()
#         plt.tight_layout()
#         sns.despine()
#         plt.show()

    


# category_one_plot(dane, 'ocena', 'percent')
# category_two_plot(dane, 'płeć','ocena', 'percent', facetgrid=False)
# category_three_plot(dane, 'zdał','ocena', 'płeć','percent')




# tabela_kontyngencji = pd.crosstab(index=dane['płeć'], columns=dane['ocena'])


# # Tworzenie heatmapy
# plt.figure(figsize=(6, 3))
# sns.heatmap(tabela_kontyngencji.pivot("Płeć", "Ocena", "Liczba"), annot=True, cmap="YlGnBu", fmt="d")
# plt.title('Heatmapa tabeli kontyngencji: Płeć vs Ocena')
# plt.xlabel('Ocena')
# plt.ylabel('Płeć')
# # Wyświetlanie heatmapy za pomocą Streamlit
# st.pyplot()








# tabela_kontyngencji_3d = pd.crosstab(index=[dane['płeć'], dane['zdał']], columns=dane['ocena'])
# tabela_long = tabela_kontyngencji_3d.reset_index().melt(id_vars=['płeć', 'zdał'], var_name='ocena', value_name='liczba')
# unique_zdał = tabela_long['zdał'].unique()
# fig, axes = plt.subplots(1, len(unique_zdał), figsize=(6 * len(unique_zdał), 4))
# for i, zdal in enumerate(unique_zdał):
#     filtr = tabela_long[tabela_long['zdał'] == zdal]
#     heatmap_data = filtr.pivot(index='płeć', columns='ocena', values='liczba')
#     sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt="d", ax=axes[i])
#     axes[i].set_title(f'Zdał: {zdal}')
#     axes[i].set_xlabel('Ocena')
#     axes[i].set_ylabel('Płeć')

# plt.tight_layout()
# plt.show()



# import seaborn as sns
# import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



def cor_num(df, zmienna1, zmienna2):
    # Tworzenie wykresu zależności między zmiennymi
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.regplot(x=zmienna1, y=zmienna2, data=df, color="#3498db", marker='.', scatter_kws={'s': 10}, line_kws={'color': 'red'}, ax=ax)
    
    # Obliczanie współczynników regresji
    x = df[zmienna1].values
    y = df[zmienna2].values
    slope, intercept = np.polyfit(x, y, 1)
    formula = f"y = {slope:.2f}x + {intercept:.2f}"
    ax.text(0.05, 0.95, formula, fontsize=8, ha="left", va="top", transform=ax.transAxes)
    
    # Konfiguracja wykresu
    ax.set_title(f'Regresja między {zmienna1} a {zmienna2}')
    ax.set_xlabel(zmienna1.capitalize())
    ax.set_ylabel(zmienna2.capitalize())
    ax.grid(True, which='both', linestyle='--', linewidth=0.1, color='gray')
    
    # Wyświetlenie wykresu w interfejsie Streamlit
    st.pyplot(fig)

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




def cor_num_matrix(df, zmienna1, zmienna2):
    # Tworzenie macierzy korelacji i wyświetlenie jej za pomocą heatmap
    df = df[[zmienna1, zmienna2]]
    corr = df.corr()
    plt.figure(figsize=(6, 5))
    plt.title("Wykres korelacji", color='gray', fontsize=10)
    sns.heatmap(corr, annot=True, vmin=-1, vmax=1, center=0, cmap="Blues", linewidths=1, linecolor='black')
    sns.despine()
    plt.grid(True, which='both', linestyle='--', linewidth=0.1, color='gray')
    st.pyplot(plt)









# sns.relplot(data=dane, x="dochody", y="wydatki", hue="płeć", height=4, palette=palette)
# sns.relplot(data=dane, x="dochody", y="wydatki", col="płeć", height=4,color="#3498db")
# sns.relplot(data=dane, x="dochody", y="wydatki", hue="płeć", col="zdał", height=4,palette=["b", "r"], sizes=(10, 100))



# df = dane[['dochody', 'wydatki']]
# plt.figure(figsize=(6, 5))
# sns.pairplot(df, diag_kind="hist")



# plt.show()

# sns.lmplot(data=dane, x="dochody", y="wydatki", hue="zdał")

# sns.lmplot(
#     data=dane, x="dochody", y="wydatki",
#     hue="ocena", col="płeć", height=4,
# )

# sns.lmplot(
#     data=dane, x="dochody", y="wydatki",
#     col="ocena", row="płeć", height=3,
#     facet_kws=dict(sharex=False, sharey=False),
# )




# def analiza_regresji():
#     # Przygotowanie danych
#     X = dane['dochody']
#     y = dane['wydatki']
#     X = sm.add_constant(X)  # dodanie stałej do modelu
    
#     # Budowanie modelu regresji liniowej
#     model = sm.OLS(y, X).fit()
    
#     print(model.summary())
    
#     # Wizualizacja zależności liniowej
#     plt.figure(figsize=(10, 6))
#     sns.regplot(x=dane['dochody'], y=dane['wydatki'], ci=None, line_kws={'color': 'red'})
#     plt.title('Regresja liniowa dochodów na wydatki')
#     plt.xlabel('Dochody')
#     plt.ylabel('Wydatki')
#     plt.show()
    
#     # Testowanie normalności reszt za pomocą testu Shapiro-Wilka
#     _, p_value = stats.shapiro(model.resid)
#     print(f"Test Shapiro-Wilka p-wartość: {p_value} (Normalność reszt, p > 0.05 oznacza normalność)")
    
#     # Sprawdzenie homoscedastyczności reszt
#     _, p_value = stats.levene(model.fittedvalues, model.resid)
#     print(f"Test Levene'a p-wartość: {p_value} (Homoscedastyczność, p > 0.05 oznacza homoscedastyczność)")
    
#     # Wizualizacja reszt
#     plt.figure(figsize=(10, 6))
#     plt.scatter(model.fittedvalues, model.resid)
#     plt.axhline(y=0, color='r', linestyle='--')
#     plt.title('Reszty vs. Dopasowane wartości')
#     plt.xlabel('Dopasowane wartości')
#     plt.ylabel('Reszty')
#     plt.show()
    
#     # Sprawdzenie autokorelacji reszt za pomocą testu Durbin-Watson
#     dw = sm.stats.durbin_watson(model.resid)
#     print(f"Test Durbin-Watson: {dw} (2 oznacza brak autokorelacji, wartości <1 lub >3 wskazują na autokorelację)")


# analiza_regresji()




#---------------------------------------------------------------------------------------------------------------------------------------------------

# import pandas as pd

# # Przykładowe dane
# data = {
#     'Płeć': ['Mężczyzna', 'Kobieta', 'Kobieta', 'Mężczyzna', 'Kobieta'],
#     'Grupa wiekowa': ['18-25', '26-35', '26-35', '36-45', '36-45'],
#     'Wzrost': [170, 165, 155, 180, 175],
#     'Waga': [70, 60, 55, 85, 70]
# }

# # Tworzenie ramki danych
# df = pd.DataFrame(data)

# # Grupowanie danych według dwóch zmiennych kategorialnych i obliczanie statystyk opisowych dla każdej grupy
# statystyki_opisowe = df.groupby(['Płeć', 'Grupa wiekowa']).describe().reset_index()

# # Wyświetlenie ramki danych z wynikami
# print(statystyki_opisowe)

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# # # Przykładowe dane
# # data = {
# #     'Płeć': ['Mężczyzna', 'Kobieta', 'Kobieta', 'Mężczyzna', 'Kobieta'],
# #     'Grupa wiekowa': ['18-25', '26-35', '26-35', '36-45', '36-45'],
# #     'Wzrost': [170, 165, 155, 180, 175],
# #     'Waga': [70, 60, 55, 85, 70]
# # }

# # # Tworzenie ramki danych
# # df = pd.DataFrame(data)

# # Grupowanie danych według dwóch zmiennych kategorialnych i obliczanie statystyk opisowych dla każdej grupy
# statystyki_opisowe = df.groupby(['Płeć', 'Grupa wiekowa']).describe().reset_index()

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
