Analyse amplitude sonore & flux optique

Auteur : Stéphane Meurisse
Site : www.codeandcortex.fr
Version : v2
Date : 07-05-2025

Description

Cette application Streamlit permet d’analyser simultanément l’amplitude sonore et le flux optique d’une vidéo. Elle offre :

Import depuis YouTube (via yt-dlp) ou fichier MP4 local.

Extraction audio en WAV mono 16 kHz et calcul de l’enveloppe sonore.

Détection automatique des observations audio atypiques basées sur un seuil μ ± kσ.

Transcription des extraits audio autour des pics sonores avec Whisper (fr).

Calcul du flux optique Farneback, affichage en heatmap JET, superpositions et vecteurs.

Génération d’un rapport texte téléchargeable listant les observations et transcriptions.

Librairies principales

Installez les librairies requises :

pip install \
  streamlit \
  opencv-python \
  soundfile \
  plotly \
  openai-whisper \
  yt-dlp \
  numpy

Le calcul du flux optique est assuré par le script opticalflow.py (fourni dans ce dépôt).
