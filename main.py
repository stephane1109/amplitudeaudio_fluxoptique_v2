############
# Analyse de l'amplitude sonore et du flux optique
# import fichier local < 200 Mo et/url - Fenêtre textuell : 1-5-10-20-30-40-50-60 sec
# Stéphane Meurisse
# www.codeandcortex.fr
# Date : 08-05-2025
############
# python -m streamlit run main.py

# pip install opencv-python soundfile plotly openai-whisper yt-dlp

import os
import subprocess
import streamlit as st
import numpy as np
import soundfile as sf
import plotly.graph_objects as go
import cv2
import whisper
from yt_dlp import YoutubeDL
from opticalflow import compute_optical_flow_metrics, _get_frame_at_time

# --- Initialisation du rapport dans le session_state ---
if 'rapport_observations' not in st.session_state:
    st.session_state['rapport_observations'] = ''

# --- Fonctions utilitaires ---

def convertir_en_min_sec(seconds: float) -> str:
    """Convertit des secondes en format mm:ss."""
    m, s = divmod(max(0, int(seconds)), 60)
    return f"{m:02d}:{s:02d}"


def telecharger_video_et_extraire_audio(video_url: str, cookiefile: str=None, rep: str="downloads") -> tuple[str, str]:
    """Télécharge une vidéo YouTube et extrait l'audio en WAV mono 16 kHz."""
    os.makedirs(rep, exist_ok=True)
    opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
        "outtmpl": os.path.join(rep, "%(id)s.%(ext)s"),
        "quiet": True
    }
    if cookiefile:
        opts["cookiefile"] = cookiefile
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
    vid = info.get("id")
    video_path = os.path.join(rep, f"{vid}.mp4")
    wav_path = os.path.join(rep, f"{vid}.wav")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return video_path, wav_path


def extraire_audio_video_locale(video_path: str, rep: str="downloads") -> tuple[str, str]:
    """Extrait l'audio d'un fichier vidéo local MP4 en WAV mono 16 kHz."""
    os.makedirs(rep, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    wav_path = os.path.join(rep, f"{base}.wav")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return video_path, wav_path


def extraire_clip_audio(wav_path: str, centre: float, demi: int, rep: str="downloads") -> str:
    """Extrait un clip audio de durée 2*demi secondes centré sur centre."""
    os.makedirs(rep, exist_ok=True)
    start = max(0, centre - demi)
    clip_path = os.path.join(rep, f"clip_{start:.2f}_{2*demi:.2f}.wav")
    cmd = [
        "ffmpeg", "-y", "-i", wav_path,
        "-ss", str(start), "-t", str(2*demi), clip_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return clip_path

####
def extraire_clip_video(video_path: str, centre: float, demi: int, rep: str="downloads") -> str:
    """
    Extrait un clip vidéo MP4 (durée 2*demi s) centré sur centre,
    sans réencodage (mode copy), et retourne son chemin.
    """
    os.makedirs(rep, exist_ok=True)
    start = max(0, centre - demi)
    dur = 2 * demi
    base = os.path.splitext(os.path.basename(video_path))[0]
    clip_mp4 = os.path.join(rep, f"{base}_clip_{start:.2f}s_{dur:.2f}s.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ss", str(start), "-t", str(dur),
        "-c", "copy", clip_mp4
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return clip_mp4
####

def transcrire_clip_whisper(wav_clip: str, highlight: float) -> str:
    """Transcrit un clip audio et surligne en rouge le segment autour de highlight."""
    model = whisper.load_model("small")
    res = model.transcribe(wav_clip, language="fr")
    segs = res.get('segments', [])
    texte = " ".join(s['text'].strip() for s in segs)
    cible = next((s['text'].strip() for s in segs if s['start'] <= highlight <= s['end']), None)
    if cible:
        texte = texte.replace(cible, f"<span style='color:red'>{cible}</span>", 1)
    return texte


def downsample_by_second(data: np.ndarray, times: np.ndarray, sr: int):
    """Calcule l'enveloppe min/max et temps moyens par seconde."""
    step = sr
    cnt = len(data) // step
    t_int, mn, mx, env = [], [], [], []
    for i in range(cnt):
        seg = data[i*step:(i+1)*step]
        t_int.append(times[i*step:(i+1)*step].mean())
        mn.append(float(seg.min())); mx.append(float(seg.max()))
        env.append((float(seg.min()) + float(seg.max())) / 2)
    return np.array(t_int), np.array(mn), np.array(mx), np.array(env)


def chercher_pic(data: np.ndarray, sr: int, centre: float) -> float:
    """Retourne le timestamp du pic absolu autour de centre ±0.5s."""
    demi = sr // 2
    idx = int(centre * sr)
    start, end = max(idx-demi, 0), min(idx+demi, len(data))
    wnd = data[start:end]
    if wnd.size == 0:
        return centre
    rel = np.argmax(np.abs(wnd))
    return (start + rel) / sr

# --- Fonctions flux optique ---

def faire_carte_flux(flow_map: np.ndarray) -> np.ndarray:
    """Génère une heatmap JET d'une carte de flux optique."""
    mag = np.linalg.norm(flow_map, axis=2) if flow_map.ndim == 3 else flow_map
    norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_JET)


def superposer_vecteurs(frame: np.ndarray, flow_map: np.ndarray, step: int=16) -> np.ndarray:
    """Superpose des vecteurs de flux optique sur une image."""
    img = frame.copy(); h, w = img.shape[:2]
    fx, fy = flow_map[...,0], flow_map[...,1]
    for y in range(0, h, step):
        for x in range(0, w, step):
            dx, dy = int(fx[y,x]), int(fy[y,x])
            cv2.arrowedLine(img, (x,y), (x+dx,y+dy), (0,255,0), 1, tipLength=0.3)
    return img

# --- Interface Streamlit ---
st.title("Analyse amplitude sonore & mouvements - version 2")
st.markdown("www.codeandcortex.fr - version 2")

# Aide :
with st.expander("Aide"):
    # 4) Explications et interprétations
    st.subheader("Interprétation des résultats")

    st.markdown("**Qu'est-ce que la magnitude optique ?**")
    st.markdown("La **magnitude optique** correspond à la moyenne des normes des vecteurs de déplacement calculés"
                "entre deux images consécutives par l'algorithme Farneback. Elle quantifie l'intensité du mouvement visuel :")

    st.markdown(
        "- *Valeurs élevées* : mouvements rapides ou importants\n"
        "- *Valeurs faibles* : mouvements lents ou quasi-statiques")

    st.markdown("**Calcul de l'observation atypique audio :**")
    st.markdown("Une observation audio atypique est détectée lorsque l'amplitude moyenne de l'enveloppe audio dépasse le seuil défini par μ ± kσ,"
                "où μ est la moyenne des amplitudes sur la vidéo et σ leur écart-type. Cela permet d'identifier des pics sonores significatifs.")

    st.markdown("**Flux optique :**")
    st.markdown("Le flux optique (Farneback) mesure les déplacements de pixels entre deux images consécutives."
                "Une heatmap JET traduit ces déplacements en intensité de mouvement, du bleu (faible) au rouge (fort).")

    st.markdown("**Superposition :**")
    st.markdown("La superposition de la heatmap sur l'image d'origine met en évidence les zones de mouvement significatif,"
                "conservant la perception visuelle du contenu tout en signalant le mouvement.")

    st.markdown("**Vecteurs de flux :**")
    st.markdown("Les flèches tracées représentent les vecteurs de déplacement (dx, dy) de blocs de pixels."
                "Leur densité et leur orientation illustrent la direction et l'amplitude du mouvement.")

video_url = st.text_input("URL YouTube")
video_local = st.file_uploader("Importer vidéo locale (MP4)", type=["mp4"])
cookie_file = st.file_uploader("Importer cookies (Netscape)", type=["txt","cookies"])
k_value = st.slider("k (intervalle [μ ± kσ])", 1.0, 5.0, 2.0, 0.1)
window = st.selectbox("Intervalle transcription autour du pic (s)", [1,5,10,20,30,40,50,60], index=0)

if st.button("Lancer l’analyse"):
    # Réinitialise rapport
    st.session_state['rapport_observations'] = ''
    # Vérification source
    if not video_url and not video_local:
        st.error("URL YouTube ou fichier local requis.")
        st.stop()
    # Gestion cookies
    cp = None
    if cookie_file:
        os.makedirs("downloads", exist_ok=True)
        cp = os.path.join("downloads", cookie_file.name)
        with open(cp, 'wb') as f:
            f.write(cookie_file.read())
    # Chargement vidéo
    if video_local:
        st.info("Chargement vidéo locale…")
        os.makedirs("downloads", exist_ok=True)
        chemin = os.path.join("downloads", video_local.name)
        with open(chemin, 'wb') as f:
            f.write(video_local.read())
        video_path, audio_path = extraire_audio_video_locale(chemin)
    else:
        st.info("Téléchargement YouTube…")
        video_path, audio_path = telecharger_video_et_extraire_audio(video_url, cp)

    # Affichage vidéo
    st.video(video_path)
    # Lecture audio
    data, sr = sf.read(audio_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    dur = len(data) / sr
    st.write(f"Durée : {dur:.1f}s — {len(data)} échantillons à {sr}Hz")

    # Calcul enveloppe
    times = np.linspace(0, dur, len(data))
    t_int, mn, mx, env = downsample_by_second(data, times, sr)

    # Transcription complète avec Whisper modèle small
    st.info("Transcription complète…")
    whisper.load_model("small").transcribe(audio_path, language="fr")

    # Détection anomalies
    mu, si = env.mean(), env.std()
    mu = 0 if abs(mu) < 1e-6 else mu
    si = 0 if abs(si) < 1e-6 else si
    lb, ub = mu - k_value * si, mu + k_value * si
    idx = np.where((env < lb) | (env > ub))[0]
    t_out, env_out = t_int[idx], env[idx]
    st.info(f"{len(idx)} observations atypiques détectées")

    # Graphique amplitude
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.r_[t_int, t_int[::-1]], y=np.r_[mn, mx[::-1]], fill='toself', fillcolor='rgba(255,255,255,0.2)', line=dict(width=0), name='Enveloppe'))
    fig.add_trace(go.Scatter(x=t_int, y=env, mode='lines', name='Enveloppe moyenne'))
    fig.add_trace(go.Scatter(x=t_out, y=env_out, mode='markers', marker=dict(color='red', size=8), name='Anomalies'))
    fig.update_layout(xaxis_title='Temps (s)', yaxis_title='Amplitude audio')
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "displaylogo": False})

    # Analyse détaillée et constitution du rapport
    st.subheader(f"Analyse anomalies et transcription (±{window}s autour du pic audio)")
    rapport = []
    for i, t0 in enumerate(t_out):
        # 1) Calcul du pic précis
        t_pic = chercher_pic(data, sr, t0)

        # 2) Extraction et affichage du clip vidéo centré ± window
        # video_clip = extraire_clip_video(video_path, t_pic, window)
        # st.video(video_clip)

        # 3) Calcul des bornes et du flux optique
        s0, s1 = t_pic - window, t_pic + window
        m0, m1 = convertir_en_min_sec(s0), convertir_en_min_sec(min(s1, dur))
        evt = compute_optical_flow_metrics(video_path, [t_pic], dt=1.0)[0]

        # Affichage observation
        st.markdown(f"---\n**Observation #{i+1} [{s0:.1f}s→{s1:.1f}s] ({m0}→{m1})**")
        st.markdown(f"- mag_t-1: {evt['mag_prev']:.2f} | mag_t: {evt['mag_next']:.2f}")

        # Transcription clip
        clip = extraire_clip_audio(audio_path, t_pic, window)
        txt = transcrire_clip_whisper(clip, window)
        st.markdown(f"> **pic audio** Transcript : {txt}", unsafe_allow_html=True)

        # Images brutes et heatmaps/superpositions
        cols = st.columns(3)
        cap = cv2.VideoCapture(video_path)
        for off, col in zip([-1, 0, 1], cols):
            frame = _get_frame_at_time(cap, t_pic + off)
            col.image(frame, channels='BGR', caption=f"t_pic+{off}s")
        cap.release()

        h1, h2 = st.columns(2)
        heat_prev = faire_carte_flux(evt['flow_map_prev'])
        heat_next = faire_carte_flux(evt['flow_map_next'])
        h1.image(heat_prev, channels='BGR', caption='Flux t-1→t')
        h2.image(heat_next, channels='BGR', caption='Flux t→t+1')

        sp1, sp2 = st.columns(2)
        sup_prev = cv2.addWeighted(evt['frame_prev'], 0.7, heat_prev, 0.3, 0)
        sup_next = cv2.addWeighted(evt['frame'], 0.7, heat_next, 0.3, 0)
        sp1.image(sup_prev, channels='BGR', caption='Superposition t-1 → t')
        sp2.image(sup_next, channels='BGR', caption='Superposition t → t+1')

        v1, v2 = st.columns(2)
        vec_prev = superposer_vecteurs(evt['frame_prev'], evt['flow_map_prev'])
        vec_next = superposer_vecteurs(evt['frame'], evt['flow_map_next'])
        v1.image(vec_prev, channels='BGR', caption='Vecteurs t-1 → t')
        v2.image(vec_next, channels='BGR', caption='Vecteurs t → t+1')

        # Ajout au rapport
        rapport.append(f"Observation {i+1} [{s0:.1f}s→{s1:.1f}s] ({m0}→{m1}) - mag_t-1: {evt['mag_prev']:.2f}, mag_t: {evt['mag_next']:.2f}")
        rapport.append(f"Transcript: {txt}")

    # Stockage dans session_state
    st.session_state['rapport_observations'] = "\n".join(rapport)

# Bouton de téléchargement après analyse détaillée
if st.session_state['rapport_observations']:
    # Le téléchargement ne réinitialise pas la session
    st.download_button(
        label="Télécharger rapport",
        data=st.session_state['rapport_observations'],
        file_name="rapport_observations.txt",
        mime="text/plain",
        key="download_btn",
        on_click=lambda: None
    )




