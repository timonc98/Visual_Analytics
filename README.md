# Visual_Analytics
Projektbeschreibung: 
Dieses Projekt befasst sich mit der Erkennung von COVID-positiven Fällen mithilfe von Deep Learning. Das Ziel ist es, ein Modell zu entwickeln, das auf Röntgenbildern von Patienten trainiert wird und in der Lage ist, zwischen COVID-positiven und normalen (negativen) Fällen zu unterscheiden.

Datensatz: 
Der verwendete Datensatz enthält Röntgenbilder von Patienten mit vier Klassen: "Normal", "Pneumonia", "Lung Opacity" und "COVID". Es wurden allerdings nur die zwei Klassen "Normal" und "COVID" für diese Arbeit verwendet. Der Datensatz wurde bereits vorverarbeitet und enthält Bilder sowie die entsprechenden Labels.
https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

Vorgehensweise:
Importieren der benötigten Bibliotheken und Einstellen von Warnungen.
Laden des Datensatzes und Aufbereiten der Daten.
Datenvisualisierung: Anzeigen einiger Beispiele für Bilder aus dem Datensatz.
Data Augmentation: Durchführen verschiedener Bildtransformationen, um den Datensatz zu erweitern und die Modellleistung zu verbessern.
Aufteilen der Daten in Trainings-, Test- und Validierungssets.
Definition des CNN-Modells: Ein Convolutional Neural Network (CNN) wird erstellt, das aus mehreren Convolutional-, Pooling-, Dropout- und Dense-Schichten besteht.
Training des Modells: Das Modell wird auf den Trainingsdaten trainiert.
Evaluierung des Modells: Das Modell wird auf den Test- und Validierungsdaten evaluiert, um die Leistung zu bewerten.
Speichern des trainierten Modells für zukünftige Verwendung.
XAI Methoden Grad-CAM und CEM werden angewendet auf das trainierte Modell.

Installationsguide: 
Projekt-Repository klonen: https://github.com/timonc98/Visual_Analytics.git
In das Projektverzeichnis wechseln
Projektumgebung erstellen (optional)
Erforderliche Bibliotheken installieren (siehe unten Bibliotheken)
Datensatz herunterladen ("https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database"), entzippen und in den gleichen Speicherort speichern, wie das Projekt (gleiches Repository). Hier dann einfach den Ordner "dataset" erstellen und alle Ordner reinkopieren:
![image](https://github.com/timonc98/Visual_Analytics/assets/92914593/36881f3f-e34a-44ce-912f-da7435b433cf)
Dashboard mit folgendem Befehl starten: streamlit run "C:/"Speicherort"/Visual_Analytics/virtual/dashboard.py".
Zur Nutzung von CEM oder Grad-Cam Speicherorte der Bilder und Dateistrukturen abändern und das Python Skript ausführen.

Dateistruktur: 
Archiv: Altdateien und alte Ansätze
Beispiel-Bilder: Alte Testungen von Dataset-fremden Bilder
Dashboard_Images: Alle Bilder für das Dashboard sind hier abgespeichert
Dataset: muss erst importiert werden 
XAI_Methoden: Grad Cam und CEM mit CNN und Autoencoder enthalten
dashboard.py: Streamlit Dashboard 
Trainierte Modelle werden auch weiter unten gespeichert.
gitignore: Eine File, welche das Pushen von dem zu großen Datensatz und damit dem Ordner "dataset" verhindert.

Bibliotheken:
PIL (Python Imaging Library): 9.5.0
streamlit: 1.23.1
numpy: 1.23.5
pandas: 2.0.1
cv2 (OpenCV): 4.7.0
albumentations: 1.3.0
plotly: 5.14.1
matplotlib: 3.7.1
seaborn: 0.12.2
sklearn: 1.2.2
keras: 2.12.0
tensorflow: 2.12.0


Quellen:

https://www.kaggle.com/code/sana306/detection-of-covid-positive-cases-using-dl
https://www.kaggle.com/code/danushkumarv/covid-19-cnn-grad-cam-viz#3-%7C-Exploratory-Data-Analysis
https://www.kaggle.com/code/smitisinghal/cem-on-mnist
https://www.kaggle.com/code/smitisinghal/cem-on-fashion-mnist
https://github.com/IBM/Contrastive-Explanation-Method
https://github.com/david-knigge/contrastive-explanation-method
https://keras.io/examples/vision/grad_cam/


Beitragende: Timon Clauß (76635), Marcel Dittrich (76435), Domenik Raab (Dozent)
