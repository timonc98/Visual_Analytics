# For running of streamlit, please use the command --   streamlit run "D:Daten-Marcel/2.Fachsemester/01_Visual Analytics/Projekt/Visual_Analytics/virtual/dashboard.py"
# Wichtig mit "" um den Verzeichnisort !!!
# For running of streamlit, please use the command --   streamlit run "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/dashboard.py"
# Wichtig mit "" um den Verzeichnisort !!!
##
##
##python -m streamlit run "D:/Daten-Marcel/2.Fachsemester/01_Visual Analytics/Projekt/Visual_Analytics/virtual/dashboard.py"

import streamlit as st
from PIL import Image

def main():
    st.set_page_config(layout="wide")
    image = Image.open("C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/covid-19.png")

    st.sidebar.image(image, use_column_width=True)
    st.sidebar.markdown("<h1 style='text-align: center; text-decoration: underline;'>Visual Analytics</h1>", unsafe_allow_html=True)
    
    st.sidebar.markdown("<h3 style='text-align: center;'>Seiten auswählen:</h3>", unsafe_allow_html=True)

    pages = {
        "Allgemein": "Allgemein",
        "Beispiele": "Beispiele"
    }
    selected_page = st.sidebar.radio("", list(pages.keys()))

    if selected_page == "Allgemein":
        st.markdown("<h1 style='text-align: center; text-decoration: underline; margin-top: 0; padding_top: 0;'>Visual Analytics</h1>", unsafe_allow_html=True)
        st.header("Allgemeine Daten :microbe:")
        st.subheader("Source Dataset:")

        # Dataset link
        url = "https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database"
        st.write("Dataset: [https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database](%s)" % url)

        # Counterplot Dataset
        st.subheader("Counterplot Data:")
        image = Image.open('C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dashboard_Images/plot.png')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(' ')

        with col2:
            st.image(image, caption='Counter Plot Covid-19 Database')

        with col3:
            st.write(' ')

        # Expander für Data Augmentation
        with st.expander("Data Augmentation"):
            image_augmentation = Image.open('C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dashboard_Images/Augmentations.png')
            st.image(image_augmentation, caption='Data Augmentation')

    elif selected_page == "Beispiele":
        selected_example = st.selectbox("Beispiel auswählen:", ["Beispiel 1", "Beispiel 2", "Beispiel 3"])
        if selected_example == "Beispiel 1":
            st.header("Covid-19 Beispiel 1 :microbe:")
            # Spaltenaufteilung des Dashboards
            col1, col2, col3 = st.columns([3, 3, 3])
            
            # Bilder in der oberen Zeile
            with col1:
                st.subheader("Original")
                image_top_left_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/COVID/images/COVID-828.png"
                image_top_left = Image.open(image_top_left_path)
                st.image(image_top_left, width=250)
                st.text("Labeled as: Normal")

            with col2:
                st.subheader("Grad-CAM")
                image_top_middle_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/COVID/images/COVID-828.png"
                image_top_middle = Image.open(image_top_middle_path)
                st.image(image_top_middle, width=250)
                st.text("Predicted as: Normal")

            with col3:
                st.subheader("Pertinent Positives")
                image_top_right_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/COVID/images/COVID-828.png"
                image_top_right = Image.open(image_top_right_path)
                st.image(image_top_right, width=250)
                st.text("Beschreibung für Bild rechts oben")
            
            # Bilder in der mittleren Zeile
            with col1:
                st.subheader("Grad-CAM")
                image_middle_left_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dashboard_Images/COVID_828_Grad_Cam.jpg"
                image_middle_left = Image.open(image_middle_left_path)
                st.image(image_middle_left, width=250)
                st.text("Beschreibung für Bild links in der Mitte")

            with col2:
                st.subheader("Pertinent Negatives")
                image_middle_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dashboard_Images/COVID-828_PN.png"
                image_middle = Image.open(image_middle_path)
                st.image(image_middle, width=250)
                st.text("Beschreibung für Bild in der Mitte")

            with col3:
                st.subheader("Pertinent Positives")
                image_middle_right_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dashboard_Images/COVID-828_PP.png"
                image_middle_right = Image.open(image_middle_right_path)
                st.image(image_middle_right, width=250)
                st.text("Beschreibung für Bild rechts in der Mitte")

        elif selected_example == "Beispiel 2":
            st.header("Covid-19 Beispiel 2 :microbe:")
            # Spaltenaufteilung des Dashboards
            col1, col2, col3 = st.columns([3, 3, 3])
            
            # Bilder in der oberen Zeile
            with col1:
                st.subheader("Original")
                image_top_left_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/COVID/images/COVID-828.png"
                image_top_left = Image.open(image_top_left_path)
                st.image(image_top_left, width=250)
                st.text("Labeled as: Normal")

            with col2:
                st.subheader("Grad-CAM")
                image_top_middle_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/COVID/images/COVID-828.png"
                image_top_middle = Image.open(image_top_middle_path)
                st.image(image_top_middle, width=150)
                st.text("Predicted as: Normal")

            with col3:
                st.subheader("Pertinent Positives")
                image_top_right_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/COVID/images/COVID-828.png"
                image_top_right = Image.open(image_top_right_path)
                st.image(image_top_right, width=250)
                st.text("Beschreibung für Bild rechts oben")
            
            # Bilder in der mittleren Zeile
            with col1:
                st.subheader("Grad-CAM")
                image_middle_left_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dashboard_Images/COVID_828_Grad_Cam.jpg"
                image_middle_left = Image.open(image_middle_left_path)
                st.image(image_middle_left, width=250)
                st.text("Beschreibung für Bild links in der Mitte")

            with col2:
                st.subheader("Pertinent Negatives")
                image_middle_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dashboard_Images/COVID-828_PN.png"
                image_middle = Image.open(image_middle_path)
                st.image(image_middle, width=250)
                st.text("Beschreibung für Bild in der Mitte")

            with col3:
                st.subheader("Pertinent Positives")
                image_middle_right_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dashboard_Images/COVID-828_PP.png"
                image_middle_right = Image.open(image_middle_right_path)
                st.image(image_middle_right, width=250)
                st.text("Beschreibung für Bild rechts in der Mitte")

        elif selected_example == "Beispiel 3":
            st.header("Covid-19 Beispiel 3 :microbe:")
            # Spaltenaufteilung des Dashboards
            col1, col2, col3 = st.columns([3, 3, 3])
            
            # Bilder in der oberen Zeile
            with col1:
                st.subheader("Original")
                image_top_left_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/COVID/images/COVID-828.png"
                image_top_left = Image.open(image_top_left_path)
                st.image(image_top_left, width=250)
                st.text("Labeled as: Normal")

            with col2:
                st.subheader("Grad-CAM")
                image_top_middle_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/COVID/images/COVID-828.png"
                image_top_middle = Image.open(image_top_middle_path)
                st.image(image_top_middle, width=250)
                st.text("Predicted as: Normal")

            with col3:
                st.subheader("Pertinent Positives")
                image_top_right_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/COVID/images/COVID-828.png"
                image_top_right = Image.open(image_top_right_path)
                st.image(image_top_right, width=250)
                st.text("Beschreibung für Bild rechts oben")
            
            # Bilder in der mittleren Zeile
            with col1:
                st.subheader("Grad-CAM")
                image_middle_left_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dashboard_Images/COVID_828_Grad_Cam.jpg"
                image_middle_left = Image.open(image_middle_left_path)
                st.image(image_middle_left, width=250)
                st.text("Beschreibung für Bild links in der Mitte")

            with col2:
                st.subheader("Pertinent Negatives")
                image_middle_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dashboard_Images/COVID-828_PN.png"
                image_middle = Image.open(image_middle_path)
                st.image(image_middle, width=250)
                st.text("Beschreibung für Bild in der Mitte")

            with col3:
                st.subheader("Pertinent Positives")
                image_middle_right_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dashboard_Images/COVID-828_PP.png"
                image_middle_right = Image.open(image_middle_right_path)
                st.image(image_middle_right, width=250)
                st.text("Beschreibung für Bild rechts in der Mitte")

if __name__ == '__main__':
    main()
