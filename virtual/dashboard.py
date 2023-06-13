# For running of streamlit, please use the command --   streamlit run "C:\Hochschule Aalen\Visual Analytics\Visual_Analytics\virtual\dashboard.py"
#Wichtig mit "" um den Verzeichnisort !!!

import streamlit as st
from PIL import Image

def main():
    st.set_page_config(layout="wide")
    # Accordion-Widget für Seitenauswahl
    with st.sidebar:
        st.markdown("<h1 style='text-align: center; text-decoration: underline;'>Virtual Analytics</h1>", unsafe_allow_html=True)
        st.sidebar.title("Seiten")
        page = st.sidebar.radio("Seiten auswählen:", ["Beispiel 1", "Beispiel 2", "Beispiel 3"])

    # Spaltenaufteilung des Dashboards
    col1, col2, col3 = st.columns(3)

    if page == "Beispiel 1":
        # Titel des Dashboards
        st.title("Covid-19 Dashboard 1")

        # Spaltenaufteilung des Dashboards
        col1, col2, col3 = st.columns([3, 3, 3])

        # Bilder in der oberen Zeile
        with col1:
            st.header("Grad-CAM")
            image_top_left_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/Normal/images/Normal-10004.png"
            image_top_left = Image.open(image_top_left_path)
            st.image(image_top_left, width=150)
            st.text("Beschreibung für Bild links oben")

        with col2:
            st.header("Grad-CAM")
            image_top_middle_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/Normal/images/Normal-10004.png"
            image_top_middle = Image.open(image_top_middle_path)
            st.image(image_top_middle, width=150)
            st.text("Beschreibung für Bild in der Mitte oben")

        with col3:
            st.header("Grad-CAM")
            image_top_right_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/Normal/images/Normal-10004.png"
            image_top_right = Image.open(image_top_right_path)
            st.image(image_top_right, width=150)
            st.text("Beschreibung für Bild rechts oben")
        
        # Bilder in der mittleren Zeile
        with col1:
            st.header("Originalbild")
            image_middle_left_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/Viral Pneumonia/images/Viral Pneumonia-1003.png"
            image_middle_left = Image.open(image_middle_left_path)
            st.image(image_middle_left, width=150)
            st.text("Beschreibung für Bild links in der Mitte")

        with col2:
            st.header("Originalbild")
            image_middle_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/Viral Pneumonia/images/Viral Pneumonia-1003.png"
            image_middle = Image.open(image_middle_path)
            st.image(image_middle, width=150)
            st.text("Beschreibung für Bild in der Mitte")

        with col3:
            st.header("Originalbild")
            image_middle_right_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/Viral Pneumonia/images/Viral Pneumonia-1003.png"
            image_middle_right = Image.open(image_middle_right_path)
            st.image(image_middle_right, width=150)
            st.text("Beschreibung für Bild rechts in der Mitte")

        # Bilder in der unteren Zeile
        with col1:
            st.header("CEM pertinent positive")
            image_bottom_left_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/COVID/images/COVID-1008.png"
            image_bottom_left = Image.open(image_bottom_left_path)
            st.image(image_bottom_left, width=150)
            st.text("Beschreibung für Bild links unten")

        with col2:
            st.header("CEM pertinent positive")
            image_bottom_middle_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/COVID/images/COVID-1008.png"
            image_bottom_middle = Image.open(image_bottom_middle_path)
            st.image(image_bottom_middle, width=150)
            st.text("Beschreibung für Bild unten in der Mitte")

        with col3:
            st.header("CEM pertinent negative")
            image_bottom_right_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/COVID/images/COVID-1008.png"
            image_bottom_right = Image.open(image_bottom_right_path)
            st.image(image_bottom_right, width=150)
            st.text("Beschreibung für Bild rechts unten")

    elif page == "Beispiel 2":
        
        # Titel des Dashboards
        st.title("Covid-19 Dashboard 2")

        # Spaltenaufteilung des Dashboards
        col1, col2, col3 = st.columns([3, 3, 3])

        # Bilder in der oberen Zeile
        with col1:
            st.header("Grad-CAM")
            image_top_left_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/Normal/images/Normal-10004.png"
            image_top_left = Image.open(image_top_left_path)
            st.image(image_top_left, width=150)
            st.text("Beschreibung für Bild links oben")

        with col2:
            st.header("Grad-CAM")
            image_top_middle_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/Normal/images/Normal-10004.png"
            image_top_middle = Image.open(image_top_middle_path)
            st.image(image_top_middle, width=150)
            st.text("Beschreibung für Bild in der Mitte oben")

        with col3:
            st.header("Grad-CAM")
            image_top_right_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/Normal/images/Normal-10004.png"
            image_top_right = Image.open(image_top_right_path)
            st.image(image_top_right, width=150)
            st.text("Beschreibung für Bild rechts oben")

        # Bilder in der mittleren Zeile
        with col1:
            st.header("Originalbild")
            image_middle_left_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/Viral Pneumonia/images/Viral Pneumonia-1003.png"
            image_middle_left = Image.open(image_middle_left_path)
            st.image(image_middle_left, width=150)
            st.text("Beschreibung für Bild links in der Mitte")

        with col2:
            st.header("Originalbild 2")
            image_middle_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/Viral Pneumonia/images/Viral Pneumonia-1003.png"
            image_middle = Image.open(image_middle_path)
            st.image(image_middle, width=150)
            st.text("Beschreibung für Bild in der Mitte")

        with col3:
            st.header("Originalbild")
            image_middle_right_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/Viral Pneumonia/images/Viral Pneumonia-1003.png"
            image_middle_right = Image.open(image_middle_right_path)
            st.image(image_middle_right, width=150)
            st.text("Beschreibung für Bild rechts in der Mitte")

        # Bilder in der unteren Zeile
        with col1:
            st.header("CEM pertinent positive")
            image_bottom_left_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/COVID/images/COVID-1008.png"
            image_bottom_left = Image.open(image_bottom_left_path)
            st.image(image_bottom_left, width=150)
            st.text("Beschreibung für Bild links unten")

        with col2:
            st.header("CEM pertinent positive")
            image_bottom_middle_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/COVID/images/COVID-1008.png"
            image_bottom_middle = Image.open(image_bottom_middle_path)
            st.image(image_bottom_middle, width=150)
            st.text("Beschreibung für Bild unten in der Mitte")

        with col3:
            st.header("CEM pertinent negative")
            image_bottom_right_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/COVID/images/COVID-1008.png"
            image_bottom_right = Image.open(image_bottom_right_path)
            st.image(image_bottom_right, width=150)
            st.text("Beschreibung für Bild rechts unten")

    elif page == "Beispiel 3":
        
        # Titel des Dashboards
        st.title("Covid-19 Dashboard 3")

        # Spaltenaufteilung des Dashboards
        col1, col2, col3 = st.columns([3, 3, 3])

        # Bilder in der oberen Zeile
        with col1:
            st.header("Grad-CAM")
            image_top_left_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/Normal/images/Normal-10004.png"
            image_top_left = Image.open(image_top_left_path)
            st.image(image_top_left, width=150)
            st.text("Beschreibung für Bild links oben")

        with col2:
            st.header("Grad-CAM")
            image_top_middle_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/Normal/images/Normal-10004.png"
            image_top_middle = Image.open(image_top_middle_path)
            st.image(image_top_middle, width=150)
            st.text("Beschreibung für Bild in der Mitte oben")

        with col3:
            st.header("Grad-CAM")
            image_top_right_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/Normal/images/Normal-10004.png"
            image_top_right = Image.open(image_top_right_path)
            st.image(image_top_right, width=150)
            st.text("Beschreibung für Bild rechts oben")

        # Bilder in der mittleren Zeile
        with col1:
            st.header("Originalbild")
            image_middle_left_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/Viral Pneumonia/images/Viral Pneumonia-1003.png"
            image_middle_left = Image.open(image_middle_left_path)
            st.image(image_middle_left, width=150)
            st.text("Beschreibung für Bild links in der Mitte")

        with col2:
            st.header("Originalbild 3")
            image_middle_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/Viral Pneumonia/images/Viral Pneumonia-1003.png"
            image_middle = Image.open(image_middle_path)
            st.image(image_middle, width=150)
            st.text("Beschreibung für Bild in der Mitte")

        with col3:
            st.header("Originalbild")
            image_middle_right_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/Viral Pneumonia/images/Viral Pneumonia-1003.png"
            image_middle_right = Image.open(image_middle_right_path)
            st.image(image_middle_right, width=150)
            st.text("Beschreibung für Bild rechts in der Mitte")

        # Bilder in der unteren Zeile
        with col1:
            st.header("CEM pertinent positive")
            image_bottom_left_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/COVID/images/COVID-1008.png"
            image_bottom_left = Image.open(image_bottom_left_path)
            st.image(image_bottom_left, width=150)
            st.text("Beschreibung für Bild links unten")

        with col2:
            st.header("CEM pertinent positive")
            image_bottom_middle_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/COVID/images/COVID-1008.png"
            image_bottom_middle = Image.open(image_bottom_middle_path)
            st.image(image_bottom_middle, width=150)
            st.text("Beschreibung für Bild unten in der Mitte")

        with col3:
            st.header("CEM pertinent negative")
            image_bottom_right_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/COVID/images/COVID-1008.png"
            image_bottom_right = Image.open(image_bottom_right_path)
            st.image(image_bottom_right, width=150)
            st.text("Beschreibung für Bild rechts unten")

if __name__ == '__main__':
    main()