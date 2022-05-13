# Ladataan kaikki tarvittavat kirjastot
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import seaborn as sns
import sklearn
from sklearn import ensemble
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
sklearn.metrics.SCORERS.keys()
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pickle
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(page_title="Airbnb-asuntojen hinnoista", page_icon=None, layout="wide", 
        initial_sidebar_state="expanded", menu_items=None)

#Ladataan käytettävä data ja jalostetaan se
@st.cache
def get_data_a():
    df_a ='http://data.insideairbnb.com/the-netherlands/north-holland/amsterdam/2021-12-05/data/listings.csv.gz'
    df_a = pd.read_csv(df_a, compression='gzip')
    df_a = df_a[['price', 'review_scores_rating', 'accommodates', 'bedrooms', 'number_of_reviews','review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'maximum_nights','minimum_nights','neighbourhood_cleansed','property_type']]
    df_a= df_a[(df_a['property_type'].isin(['Entire condominium (condo)','Entire rental unit', 'Entire townhouse', 'Entire loft', 'Entire serviced apartment']))]
    df_a= df_a[df_a['review_scores_rating'].notnull()]
    df_a = df_a[df_a['bedrooms'].notnull()]
    df_a = df_a[df_a['review_scores_accuracy'].notnull()]
    df_a = df_a[df_a['review_scores_cleanliness'].notnull()]
    df_a['city'] = 'Amsterdam'
    df_a['price'] = df_a['price'].replace('[\$,)]','',  \
        regex=True).replace('[(]','-', regex=True).astype(float)
    return df_a

@st.cache
def get_data_b():
    df_b ='http://data.insideairbnb.com/spain/catalonia/barcelona/2021-12-07/data/listings.csv.gz'
    df_b = pd.read_csv(df_b, compression='gzip')
    df_b = df_b[['price', 'review_scores_rating', 'accommodates', 'bedrooms', 'number_of_reviews','review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'maximum_nights','minimum_nights', 'neighbourhood_cleansed','property_type']]
    df_b= df_b[(df_b['property_type'].isin(['Entire condominium (condo)','Entire rental unit', 'Entire townhouse', 'Entire loft', 'Entire serviced apartment']))]
    df_b = df_b[df_b['review_scores_rating'].notnull()]
    df_b = df_b[df_b['bedrooms'].notnull()]
    df_b = df_b[df_b['review_scores_accuracy'].notnull()]
    df_b = df_b[df_b['review_scores_location'].notnull()]
    df_b = df_b[df_b['review_scores_communication'].notnull()]
    df_b['city'] = 'Barcelona'
    df_b['price'] = df_b ['price'].replace('[\$,)]','',  \
        regex=True).replace('[(]','-', regex=True).astype(float)
    return df_b

@st.cache
def get_data_ab():
    df_a = get_data_a()
    df_b = get_data_b()
    df_ab= pd.concat([df_a, df_b])
    df_ab = df_ab[(df_ab['price'] < 201)]
    df_ab = df_ab[(df_ab['bedrooms'] < 4)]
    df_ab = df_ab[(df_ab['minimum_nights'] < 31)]
    df_ab = df_ab.loc[~df_ab.index.duplicated(), :]
    return df_ab

df_a = get_data_a()
df_b = get_data_b()
df_ab = get_data_ab()

def main() -> None:
    
    #Koko sivun otsikko
    st.title("Airbnb-asuntojen hintaan vaikuttavia tekijöitä ja hinnan ennustin :bar_chart:")

    #Lisätietoja lyhyesti
    st.write("""Tälle sivulle on koottuna tietoa Airbnb-asunnoista Amsterdamissa ja Barcelonassa sekä Airbnb-asunnon hinnan ennustin. 
    Tarkempia tietoja saat näkyviin avaamalla alla olevan Lisätietoja-laatikon.""")

    #Sivun piilotettu lisäingressiteksti
    with st.expander("Lisätietoja"):
        url = "http://insideairbnb.com/get-the-data"
        st.write("""Tälle sivulle on koottuna tietoa Airbnb-asunnoista Amsterdamissa ja Barcelonassa sekä Airbnb-asunnon hinnan ennustin.
        Sivun kohderyhmää ovat kokonaisen asunnon lyhytaikaista Airbnb-vuokrausta pohtivat henkilöt, jotka haluavat lisätietoja hintaan 
        vaikuttavista tekijöistä. Ennustimen kohderyhmää ovat kokonaisen asunnon lyhytaikaista Airbnb-vuokrausta Amsterdamissa tai Barcelonassa 
        pohtivat.""")

        st.write ("""Ennustettu hinta ei ole tae vaan lisätyöväline vuorokauden mittaisen majoituksen hinnan määrittämiseen. 
        Käytetty data on kerätty [Inside Airbnb -palvelusta](%s).""" % url)
        
        st.write(""" Data on siivottu ja jalostettu Pythonia-kirjastoja hyödyntäen muotoon, jossa sitä voi esittää interakttivisesti 
        tällä sivulla. Tarkempi dokumentointi löytyy osana harjoitustyötä palautetusta Jupytern Notebookista.""" ) 
    
    #Tyhjä rivi
    st.write("""""")

    #Käyttäjä valitsee näkymän (Koko data - Amsterdam - Barcelona)
    st.subheader("""Valitse ensin haluatko katsoa kaikkea dataa vai vain Amsterdamin tai Barcelonan""")
    
    city = df_ab["city"]
    city_choice = st.selectbox("Valitse näkymä",["Kaikki","Amsterdam", "Barcelona"] , help = "Valitse mistä datakokonaisuudesta näkymä koostetaan")
    if city_choice in ["Amsterdam"]:
        df = df_ab[(df_ab['city'] == "Amsterdam")]
    elif city_choice in ["Barcelona"]:
        df = df_ab[(df_ab['city'] == "Barcelona")]
    else:
        df = df_ab
    st.subheader("""Alla olevasta taulukosta voit tutustua dataan tarkemmin""")
    st.write(df)
    st.write("""""")

    #Näytetään muutama keskeinen kuvaaja
    st.subheader("""Valitsemasi datasetin keskeinen sisältö""")

    st.write("Kokonaiarviointien jakautuminen")
    arviot = pd.DataFrame(df["review_scores_rating"].value_counts())
    st.bar_chart(arviot)

    st.write("""""")

    st.write("Nukkumapaikkojen lukumäärät")
    nukkumapaikat = pd.DataFrame(df["accommodates"].value_counts())
    st.bar_chart(nukkumapaikat)

    st.write("""""")

    st.write("Majoituksen vähimmäiskesto")
    vah_yot = pd.DataFrame(df["minimum_nights"].value_counts())
    st.bar_chart(vah_yot)

    st.write("""""")

    st.write("Kohteiden vuorokausihinnat")
    hinta = pd.DataFrame(df["price"].value_counts())
    st.bar_chart(hinta)
    
    #Datasetin suodatus
    st.subheader("""Datasetin suodatus""")
    st.write(""" Alla olevaan taulukkoon voit halutessasi suodattaa näkyä datasetistä. Suodatukset voi tehdä vasemmassa navigaatiossa""")

    accommodates_choice = st.number_input('Nukkumapaikkojen lukumäärä', min_value=1,max_value=10,step=1)
    review_choice = st.number_input('Yleisarvio', min_value=1.0,max_value=5.0,step=0.1)
    min_nights_choice = st.number_input('Yöiden minimimäärä', min_value=1,max_value=60,step=1)


    suodatettu_kuva = df.loc[(df['accommodates']==accommodates_choice) & (df['review_scores_rating']== review_choice) & (df['minimum_nights']== min_nights_choice)]
    st.write(suodatettu_kuva)

    st.write("Suodatettujen kohteiden vuorokausihinnat")
    hinta = pd.DataFrame(suodatettu_kuva["price"].value_counts())
    st.line_chart(hinta)

    st.write("""""")

    st.subheader("Hintaan vaikuttavia tekijöitä")
    with st.expander("Huom!!"):
        st.write("Vain alussa tekemäsi datasetin valinta vaikuttaa alla näkyviin tuloksiin. Muuten otoskoot jäisivät liian pieniksi.")

    st.write("""""")

    korrelaatiot = (df.corr()['price'])
    st.write("Muuttujien korrelaatiot suhteessa hintaan",city_choice, korrelaatiot)

    st.write("""""")
    st.write("""""")

    #Lineaarisen regression toteuttaminen
    feature_names = df.columns
    target = df["price"]

    FeaturesName = ["review_scores_rating", 'accommodates','bedrooms', 'number_of_reviews', 'review_scores_accuracy', \
            'review_scores_cleanliness', 'review_scores_communication', 'review_scores_location', 'review_scores_value',\
            'maximum_nights', 'minimum_nights', 'neighbourhood_cleansed','property_type']

    #Rakennetaan interaktiivinen toiminto lineaarisen regression esitättämiseksi
    st.write("Valitse alla olevasta alasvetovalikosta muuttuja, niin alle piirtyy hinnan ja valitsevasi muuttujan suhdetta kuvaava \
        lineaarisen regression kuvaaja")
    checked_variable = st.selectbox('', FeaturesName)

    fig = sns.lmplot(x =checked_variable, y ='price', data = df, height=3, aspect=1.5)
    plt.xlabel(checked_variable)
    plt.ylabel("Hinta")
    st.pyplot(fig)

    #Rakennetaan ennustin
    st.subheader("""Hinnan ennustin""")

    st.write("""Anna ennustimelle seuraavat tiedot""")
    accommodates_choice_e = st.slider('Nukkumapaikkojen lukumäärä asunnossasi', min_value=1,max_value=10,step=1)
    review_choice_e = st.slider('Asunnon odotettu yleisarvio', min_value=1.0,max_value=5.0,step=0.1)
    location_choice_e = st.slider('Asunnon odotettu sijaintiarvio', min_value=1.0,max_value=5.0,step=0.1)

    X = df[['review_scores_location', 'accommodates', 'review_scores_rating']]
    y = df[['price']]

    #jaetaan opetus- ja testiaineistoon
    X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=0)  

    regr = linear_model.LinearRegression()
    regr.fit(X, y)

    regr_trained=LinearRegression()  
    regr_trained.fit(X_train,Y_train)

    # Ennustetaan majoituksen yhden yön hinta, kun käyttäjä antaa yleisarvion ja makuupaikkojen lukumäärän.
    predicted_price_trained = regr_trained.predict([[location_choice_e, accommodates_choice_e, review_choice_e]])

    st.write ("Hinta-arvio kaupungissa", city_choice, " sijaitsevalle asunnolle, jossa on" ,accommodates_choice_e, \
        "nukkumapaikkaa, jonka sijainnin arvioksi annat",location_choice_e, "ja yleisarvioksi", review_choice_e, "on:",\
        predicted_price_trained)
    
main()


