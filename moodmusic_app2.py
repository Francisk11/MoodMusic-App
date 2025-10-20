# 🎧 MoodMusic - Web App (RECOMENDACIÓN BASADA EN RELEVANCIA ILIMITADA)
# Autor: Fran Apaza
# App que recomienda canciones según cómo te sentís

import streamlit as st
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuración de la página ---
st.set_page_config(page_title="MoodMusic", page_icon="🎵", layout="centered")

# --- Inicialización del Estado de Sesión ---
# Streamlit usa esto para recordar valores entre clics de botón
if 'recomendaciones' not in st.session_state:
    st.session_state.recomendaciones = []
if 'indice_actual' not in st.session_state:
    st.session_state.indice_actual = 0
if 'emocion_actual' not in st.session_state:
    st.session_state.emocion_actual = ""


# --- Título y descripción ---
st.title("🎧 MoodMusic")
st.markdown("Esta app usa *Machine Learning* para recomendarte canciones según tu estado de ánimo 💭")

# --- Base de datos de canciones (TU LISTA EXTENDIDA) ---
canciones = {
    "M.A.I - Milo J": "amor ternura deseo",
    "GANAS - YSY A": "motivacion energia alegria",
    "OKUPA - WOS": "amor deseo rebeldia",
    "ALMA DINAMITA - WOS": "amor energia",
    "Tu Locura - Gustavo Cerati": "amor ternura deseo",
    "Te llevo para que me lleves - Gustavo Cerati": "amor ternura nostalgia",
    "Luna de miel en la mano - Virus": "alegria diversion energia",
    "Me gustas mucho - Viejas Locas": "amor deseo ternura",
    "Cicatrices - Airbag": "nostalgia melancolia tristeza",
    "Corazon delator - Soda Stereo": "nostalgia melancolia desconfianza",
    "Ella uso mi cabeza como un revolver - Soda Stereo": "desamor melancolia enojo",
    "Un Misil en Mi Placard - Soda Stereo": "desamor melancolia vulnerabilidad",
    "Adiós - Gustavo Cerati": "desamor nostalgia melancolia",
    "Nada Personal - Soda Stereo": "deseo alegria energia",
    "Te para tres - Soda Stereo": "melancolia tristeza soledad",
    "Flechazo en el centro - Ysy a, Bhavi, Milo J": "amor alegria deseo",
    "Puente - Gustavo Cerati": "nostalgia melancolia reflexion",
    "No soy eterno - Milo J": "melancolia tristeza",
    "Mira mamá - WOS": "melancolia nostalgia reflexion",
    "Creep - Radiohead": "tristeza desamor soledad",
    "Feel Good INC - Gorillaz": "melancolia ironia energia",
    "Historia de un Taxi - Ricardo Arjona": "amor tristeza nostalgia",
    "Porque te Demoras? - Plan B": "alegria energia deseo",
    "When Me up When Septembers Ends - Green Day": "melancolia desamor reflexion",
    "Canguro - WOS": "enojo energia rebeldia",
    "Luz Delito - WOS": "enojo energia frustracion",
    "NO HAGO TRAP - Milo J": "enojo energia actitud",
    "Niño Gordo Flaco - WOS": "enojo energia rebeldia",
    "Sugar on My Tongue - Tyler, The Creator": "alegria energia diversion",
    "See You Again - Tyler, The Creator": "amor alegria deseo",
    "NEW MAGIC WAND - Tyler, The Creator": "enojo frustracion rabia",
    "ARE WE STILL FRIENDS? - Tyler, The Creator": "melancolia desamor confusion",
    "Stronger - Kanye West": "energia alegria motivacion confianza",
    "Bound 2 - Kanye West": "amor deseo ternura",
    "I Wonder - Kanye West": "reflexion melancolia motivacion",
    "Flashing Lights - Kanye West ft. Dwele": "amor deseo nostalgia",
    "Good Life - Kanye West ft. T-Pain": "alegria celebracion confianza",
    "Champion - Kanye West": "motivacion confianza orgullo alegria",
    "Good Morning - Kanye West": "motivacion energia esperanza",
    "Ghost Town - Kanye West ft. PARTYNEXTDOOR, 070 Shake & Kid Cudi": "esperanza melancolia liberacion",
    "Thinkin Bout You - Frank Ocean": "amor nostalgia deseo ternura",
    "Ivy - Frank Ocean": "amor desamor melancolia nostalgia",
    "Pink + White - Frank Ocean": "nostalgia amor reflexion ternura",
    "Tumbando el Club Remix - Neo Pistea": "energia fiesta diversion",
    "FULL ICE - YSY A": "confianza energia ego",
    "De Música Ligera - Soda Stereo": "alegria energia nostalgia amor",
    "Persiana Americana - Soda Stereo": "amor deseo energia ternura",
    "En la Ciudad de la Furia - Soda Stereo": "melancolia nostalgia soledad reflexion",
    "Mil Horas - Los Abuelos de la Nada": "amor nostalgia melancolia ternura",
    "Flaca - Andrés Calamaro": "desamor nostalgia tristeza",
    "Ji Ji Ji - Patricio Rey y sus Redonditos de Ricota": "energia rebeldia diversion",
    "Cerca de la Revolución - Charly García": "energia rebeldia motivacion",
    "Rezo por Vos - Charly García y Luis Alberto Spinetta": "esperanza amor nostalgia ternura",
    "Seguir Viviendo Sin Tu Amor - Luis Alberto Spinetta": "desamor melancolia tristeza soledad",
    "Muchacha (Ojos de Papel) - Almendra": "amor deseo nostalgia ternura",
    "Demoliendo Hoteles - Charly Garcia": "energia enojo rebeldia",
    "Fanky - Charly Garcia": "energia alegria diversion",
    "Quizas, Porque - Sui Generis": "amor nostalgia esperanza ternura",
    "11 y 6 - Fito Páez": "amor melancolia ternura esperanza",
    "La Bestia Pop - Patricio Rey y sus Redonditos de Ricota": "energia rebeldia enojo diversion",
    "Hablando a Tu Corazón - Charly García": "amor ternura reflexion nostalgia",
    "Lago en el Cielo - Gustavo Cerati": "amor ternura melancolia",
    "Something - The Beatles": "amor deseo ternura",
    "And I Love Her - The Beatles": "amor melancolia ternura",
    "Michelle - The Beatles": "amor melancolia ternura",
    "If I Fell - The Beatles": "amor deseo ternura",
    "Bohemian Rhapsody - Queen": "tristeza melancolia drama intensidad",
    "We Will Rock You - Queen": "energia motivacion rebeldia",
    "We Are The Champions - Queen": "orgullo motivacion alegria",
    "Somebody to Love - Queen": "amor deseo tristeza melancolia",
    "Don't Stop Me Now - Queen": "energia alegria diversion"
}

# --- Entrada del usuario ---
emocion = st.text_input("🎙️ ¿Cómo te sentís hoy? (ej: melancolia, alegria, amor, nostalgia)", 
                        st.session_state.emocion_actual)

# --- Función para generar las recomendaciones (ILIMITADO PERO FILTRADO) ---
def generar_recomendaciones(emocion_input, cancion_db):
    # 1. Preparar los datos
    lista_canciones = list(cancion_db.keys())
    descripciones = list(cancion_db.values())
    descripciones.append(emocion_input)
    
    # 2. Vectorización y Similitud
    vectorizer = CountVectorizer()
    matriz = vectorizer.fit_transform(descripciones)
    similitudes = cosine_similarity(matriz[-1], matriz[:-1])[0]
    
    # 3. Emparejar similitudes con índices y filtrar solo si similitud > 0
    # Creamos una lista de (similitud, indice)
    matches_relevantes = []
    for i, similitud in enumerate(similitudes):
        # SOLO incluimos si la similitud es mayor a cero (hay al menos una palabra clave compartida)
        if similitud > 0:
            matches_relevantes.append((similitud, i))
            
    # 4. Ordenar de mayor a menor similitud
    # Ordenamos por el primer elemento de la tupla (la similitud)
    matches_relevantes.sort(key=lambda x: x[0], reverse=True)
    
    # 5. Obtener las canciones ordenadas
    recomendaciones = [lista_canciones[indice] for similitud, indice in matches_relevantes]
    
    return recomendaciones # Devolvemos solo las que coinciden (> 0)

# --- Botón principal: GENERAR RECOMENDACIÓN (Reinicia la búsqueda) ---
if st.button("🎵 Recomendar canción"):
    if emocion.strip() == "":
        st.warning("Por favor, escribí cómo te sentís.")
    else:
        # Generar TODAS las canciones ordenadas y filtradas por relevancia (> 0)
        recomendaciones_generadas = generar_recomendaciones(emocion, canciones)
        
        if not recomendaciones_generadas:
             st.error("No encontramos ninguna canción que coincida con esa emoción.")
             st.session_state.recomendaciones = [] # Asegurar que la lista esté vacía si no hay coincidencias
        else:
            # Almacenar en el estado de sesión
            st.session_state.recomendaciones = recomendaciones_generadas
            st.session_state.indice_actual = 0 # Siempre empezamos con la primera (la mejor)
            st.session_state.emocion_actual = emocion # Guardamos la emoción
            
            # Mostramos la primera (la más relevante)
            st.success(f"🎶 Según tu emoción, te recomiendo escuchar: **{st.session_state.recomendaciones[0]}**")

# --- Lógica del Botón Secundario (Rotación de canciones) ---
if st.session_state.recomendaciones:
    
    total_canciones = len(st.session_state.recomendaciones)
    
    # Usamos st.columns para alinear el botón y el resultado
    col1, col2 = st.columns([1, 3])
    
    # 1. Definición y Lógica del botón de rotación
    with col1:
        # El botón solo se muestra si NO estamos en la última canción
        if st.session_state.indice_actual < total_canciones - 1:
            # Quitamos el contador. Ahora solo dice "Ver la siguiente sugerencia"
            mensaje_boton = "▶️ Ver la siguiente sugerencia"
            
            if st.button(mensaje_boton):
                # Aumentamos el índice
                st.session_state.indice_actual += 1
                st.rerun() 
        else:
            # Mensaje genérico al terminar de rotar
            st.info("Fin de las sugerencias.")
            
    # 2. Visualización de la Recomendación Actual
    with col2:
        cancion_actual = st.session_state.recomendaciones[st.session_state.indice_actual]
        
        if st.session_state.indice_actual == 0 and st.session_state.emocion_actual:
             # Primera recomendación (el mejor match)
             st.success(f"🎶 La mejor recomendación es: **{cancion_actual}**")
        elif st.session_state.indice_actual > 0:
             # Opciones secundarias (quitamos el contador de match)
             st.info(f"O prueba con esta opción: **{cancion_actual}**")


# --- Pie de página ---
st.markdown("---")