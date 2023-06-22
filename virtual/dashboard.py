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
        "Interface": "Interface"
    }
    selected_page = st.sidebar.radio("", list(pages.keys()))

    if selected_page == "Allgemein":
        
        st.markdown("<h1 style='text-align: center; text-decoration: underline; margin-top: 0; padding_top: 0;'>Visual Analytics</h1>", unsafe_allow_html=True)
        st.header("Allgemeine Daten :microbe:")
        st.subheader ("Teammitglieder:")
        st.write ("Timon Clauß (76635) & Marcel Dittrich (77777)")
        st.subheader("Source Dataset:")

        # Dataset link
        url = "https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database"
        st.write("Dataset: [https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database](%s)" % url)

        # Expander für Data Augmentation
        with st.expander("Counterplot Data:"):
        # Counterplot Dataset
            st.subheader("Counterplot Data:")
            image = Image.open('C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dashboard_Images/plot.png')
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(' ')

            with col2:
                st.image(image, width=300)

            with col3:
                st.write(' ')

        # Expander für Data Augmentation
        with st.expander("Data Augmentation"):
            image_augmentation = Image.open('C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dashboard_Images/Augmentations.png')
            st.image(image_augmentation, caption='Data Augmentation')

    elif selected_page == "Interface":
        selected_example = st.selectbox("", ["Interface 1", "Interface 2", "Interface 3", "Interface 4"])
        if selected_example == "Interface 1":
            st.header("Covid-19 Interface 1 COVID-86 :microbe:")

            with st.expander("Data Augmentation"):
                image_augmentation = Image.open('C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dashboard_Images/COVID-828_Augmentations.png')
                st.image(image_augmentation, caption='Data Augmentation')

            # Spaltenaufteilung der oberen Zeile
            col1_top, col2_top, col3_top = st.columns([3, 3, 3])
            

            # Textboxen in der oberen Zeile
            label_left = "COVID"
            label_right = "COVID"
            
            # Bilder in der oberen Zeile
            with col1_top:
                st.subheader("Labeled as:")
                st.subheader("")
                st.subheader("")
                st.subheader("")
                st.markdown(f"<div style='text-align: center; border: 3px solid black; padding: 10px; width: 50%;'>{label_left}</div>", unsafe_allow_html=True)

            with col2_top:
                st.subheader("Original")
                image_top_middle_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/COVID/images/COVID-828.png"
                image_top_middle = Image.open(image_top_middle_path)
                st.image(image_top_middle, width=230)

            with col3_top:
                st.subheader("Predicted as:")
                st.subheader("")
                st.subheader("")
                st.subheader("")
                st.markdown(f"<div style='text-align: center; border: 5px solid green; padding: 10px; width: 50%;'>{label_right}</div>", unsafe_allow_html=True)
            
            # Spaltenaufteilung der mittleren Zeile
            col1_middle, col2_middle, col3_middle = st.columns([3, 3, 3])
            
            # Bilder in der mittleren Zeile
            with col1_middle:
                st.subheader("Grad-CAM")
                image_bottom_left_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dashboard_Images/COVID_828_Grad_Cam.jpg"
                image_bottom_left = Image.open(image_bottom_left_path)
                st.image(image_bottom_left, width=250)

            with col2_middle:
                st.subheader("Pertinent Negatives")
                image_bottom_middle_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dashboard_Images/COVID-828_PN_neu2.png"
                image_bottom_middle = Image.open(image_bottom_middle_path)
                st.image(image_bottom_middle, width=700)

            with col3_middle:
                st.subheader("Pertinent Positives")
                image_bottom_right_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dashboard_Images/COVID-828_PP.png"
                image_bottom_right = Image.open(image_bottom_right_path)
                st.image(image_bottom_right, width=350)
            

        elif selected_example == "Interface 2":
            st.header("Covid-19 Interface 2 Normal-210 :microbe:")
            
            with st.expander("Data Augmentation"):
                image_augmentation = Image.open('C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dashboard_Images/Normal-210_Augmentations.png')
                st.image(image_augmentation, caption='Data Augmentation')

            # Spaltenaufteilung der oberen Zeile
            col1_top, col2_top, col3_top = st.columns([3, 3, 3])
            
            # Textboxen in der oberen Zeile
            label_left = "Normal"
            label_right = "Normal"
            
            # Bilder in der oberen Zeile
            with col1_top:
                st.subheader("Labeled as:")
                st.subheader("")
                st.subheader("")
                st.subheader("")
                st.markdown(f"<div style='text-align: center; border: 3px solid black; padding: 10px; width: 50%;'>{label_left}</div>", unsafe_allow_html=True)

            with col2_top:
                st.subheader("Original")
                image_top_middle_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/Normal/images/Normal-210.png"
                image_top_middle = Image.open(image_top_middle_path)
                st.image(image_top_middle, width=250)

            with col3_top:
                st.subheader("Predicted as:")
                st.subheader("")
                st.subheader("")
                st.subheader("")
                st.markdown(f"<div style='text-align: center; border: 5px solid green; padding: 10px; width: 50%;'>{label_right}</div>", unsafe_allow_html=True)
            
            # Spaltenaufteilung der mittleren Zeile
            col1_middle, col2_middle, col3_middle = st.columns([3, 3, 3])
            
            # Bilder in der mittleren Zeile
            with col1_middle:
                st.subheader("Grad-CAM")
                image_bottom_left_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dashboard_Images/Normal-210_Grad_Cam.jpg"
                image_bottom_left = Image.open(image_bottom_left_path)
                st.image(image_bottom_left, width=250)

            with col2_middle:
                st.subheader("Pertinent Negatives")
                image_bottom_middle_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dashboard_Images/Normal-210_PN.png"
                image_bottom_middle = Image.open(image_bottom_middle_path)
                st.image(image_bottom_middle, width=250)

            with col3_middle:
                st.subheader("Pertinent Positives")
                image_bottom_right_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dashboard_Images/Normal-210_PP.png"
                image_bottom_right = Image.open(image_bottom_right_path)
                st.image(image_bottom_right, width=250)

        elif selected_example == "Interface 3":
            st.header("Covid-19 Interface 3 Normal-219 :microbe:")
            
            with st.expander("Data Augmentation"):
                image_augmentation = Image.open('C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dashboard_Images/Normal-219_Augmentations.png')
                st.image(image_augmentation, caption='Data Augmentation')

            # Spaltenaufteilung der oberen Zeile
            col1_top, col2_top, col3_top = st.columns([3, 3, 3])
            
            # Textboxen in der oberen Zeile
            label_left = "Normal"
            label_right = "Normal"
            
            # Bilder in der oberen Zeile
            with col1_top:
                st.subheader("Labeled as:")
                st.subheader("")
                st.subheader("")
                st.subheader("")
                st.markdown(f"<div style='text-align: center; border: 3px solid black; padding: 10px; width: 50%;'>{label_left}</div>", unsafe_allow_html=True)

            with col2_top:
                st.subheader("Original")
                image_top_middle_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/Normal/images/Normal-219.png"
                image_top_middle = Image.open(image_top_middle_path)
                st.image(image_top_middle, width=250)

            with col3_top:
                st.subheader("Predicted as:")
                st.subheader("")
                st.subheader("")
                st.subheader("")
                st.markdown(f"<div style='text-align: center; border: 5px solid green; padding: 10px; width: 50%;'>{label_right}</div>", unsafe_allow_html=True)
            
            # Spaltenaufteilung der mittleren Zeile
            col1_middle, col2_middle, col3_middle = st.columns([3, 3, 3])
            
            # Bilder in der mittleren Zeile
            with col1_middle:
                st.subheader("Grad-CAM")
                image_bottom_left_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dashboard_Images/Normal-219_Grad_Cam.jpg"
                image_bottom_left = Image.open(image_bottom_left_path)
                st.image(image_bottom_left, width=250)

            with col2_middle:
                st.subheader("Pertinent Negatives")
                image_bottom_middle_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dashboard_Images/Normal-219_PN.png"
                image_bottom_middle = Image.open(image_bottom_middle_path)
                st.image(image_bottom_middle, width=250)

            with col3_middle:
                st.subheader("Pertinent Positives")
                image_bottom_right_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dashboard_Images/Normal-219_PP.png"
                image_bottom_right = Image.open(image_bottom_right_path)
                st.image(image_bottom_right, width=250)


        if selected_example == "Interface 4":
            st.header("Covid-19 Interface 4 Missclasification :microbe:")

            with st.expander("Data Augmentation"):
                image_augmentation = Image.open('C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dashboard_Images/COVID-86_Augmentations.png')
                st.image(image_augmentation, caption='Data Augmentation')

            # Spaltenaufteilung der oberen Zeile
            col1_top, col2_top, col3_top = st.columns([3, 3, 3])


            # Textboxen in der oberen Zeile
            label_left = "COVID"
            label_right = "Normal"
            
            # Bilder in der oberen Zeile
            with col1_top:
                st.subheader("Labeled as:")
                st.subheader("")
                st.subheader("")
                st.subheader("")
                st.markdown(f"<div style='text-align: center; border: 3px solid black; padding: 10px; width: 50%;'>{label_left}</div>", unsafe_allow_html=True)

            with col2_top:
                st.subheader("Original")
                image_top_middle_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dataset/COVID/images/COVID-86.png"
                image_top_middle = Image.open(image_top_middle_path)
                st.image(image_top_middle, width=230)

            with col3_top:
                st.subheader("Predicted as:")
                st.subheader("")
                st.subheader("")
                st.subheader("")
                st.markdown(f"<div style='text-align: center; border: 5px solid red; padding: 10px; width: 50%;'>{label_right}</div>", unsafe_allow_html=True)
            
            # Spaltenaufteilung der mittleren Zeile
            col1_middle, col2_middle, col3_middle = st.columns([3, 3, 3])
            
            # Bilder in der mittleren Zeile
            with col1_middle:
                st.subheader("Grad-CAM")
                image_bottom_left_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dashboard_Images/COVID-86_Grad_Cam.jpg"
                image_bottom_left = Image.open(image_bottom_left_path)
                st.image(image_bottom_left, width=250)

            with col2_middle:
                st.subheader("Pertinent Negatives")
                image_bottom_middle_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dashboard_Images/COVID-86_PN.png"
                image_bottom_middle = Image.open(image_bottom_middle_path)
                st.image(image_bottom_middle, width=250)

            with col3_middle:
                st.subheader("Pertinent Positives")
                image_bottom_right_path = "C:/Hochschule Aalen/Visual Analytics/Visual_Analytics/virtual/Dashboard_Images/COVID-86_PP.png"
                image_bottom_right = Image.open(image_bottom_right_path)
                st.image(image_bottom_right, width=250)
if __name__ == '__main__':
    main()
