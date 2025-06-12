"""
Para este laboratorio se utilizo un audio de 46m de duracion al cual se le realizaron distintas tecnicas de preprocesamiento de texto a  la transcripcion generada. Entre las cuales estan:

- Normalizacion de texto: Consiste en pasar el texto a minusculas, corregir la ortografia y eliminar espacios extra. Es importante ya que si se posee la palabra "Hola" y la palbra "hola",
                          la maquina lo identifica como una palabra distinta. Es por este motivo que se realiza la normalizacion del texto, para evitar que palabras iguales se traten como diferentes.

- Lematizacion: Consiste en extraer la forma lematizada de cada token (palabra) del texto original. Parecido a la normalizacion, la lematizacion agrupa variantes de una palabra bajo su forma base o lema.
                Por ejemplo, las palabras "corría", "corriendo" o "corre" se agrupan bajo el lema "correr". Esto mejora la generalizacion, reduce la dispersion del vocabulario

- Eliminacion de stop words: Consiste en filtrar las palabras que no aportan significado al texto, como preposiciones, articulos, etc. Esto permite que los modelos de PLN sean mas eficientes 
                             y se enfoquen en las palabras que realmente aportan significado al texto.

"""


from huggingsound import SpeechRecognitionModel
import spacy
import os
from pydub import AudioSegment
from spellchecker import SpellChecker

# Configuración de carpetas y archivos
CHUNK_FOLDER = "./audio_chunks"
TRANSCRIPT_ORIGINAL_FILE = "./transcription_original.txt"
TRANSCRIPT_NORMALIZED_FILE = "./transcription_normalized.txt"
TRANSCRIPT_LEMMATIZED_FILE = "./transcription_lemmatized.txt"
TRANSCRIPT_STOPWORDS_FILE = "./transcription_stopwords_removed.txt"
TRANSCRIPT_ALL_FILTERS_FILE = "./transcription_all_filters.txt"
CHUNK_LENGTH_MS = 5 * 60 * 1000  # Duracion de cada fragmento  en milisegundos (5 minutos)

# Se inicializa el corrector ortografico en español
spell = SpellChecker(language='es')

def corregir_ortografia(texto):
    palabras = texto.split()
    palabras_corregidas = [spell.correction(palabra) or palabra for palabra in palabras]
    return " ".join(palabras_corregidas)

# se crea un carpeta para almacenar los chunks de audio
os.makedirs(CHUNK_FOLDER, exist_ok=True)

audio_file = "./audio.wav"  # Reemplaza con la ruta real

# Con AudioSegment puedo manipular el audio
audio = AudioSegment.from_file(audio_file)

# Se separa el audio en fragmentos de 5 minutos, ya que si se intenta procesar todo el audio a la vez, puede llegar a ocupar mas de 40GB de RAM
chunks = [audio[i:i + CHUNK_LENGTH_MS] for i in range(0, len(audio), CHUNK_LENGTH_MS)]
chunk_paths = []

# Se guardan los fragmentos de audio en la carpeta creada "./audio_chunks" en formato .wav
for i, chunk in enumerate(chunks):
    chunk_path = os.path.join(CHUNK_FOLDER, f"chunk_{i}.wav")
    chunk.export(chunk_path, format="wav")
    chunk_paths.append(chunk_path)  # Guardamos la ruta para procesarla después

print("\n✅ Todos los fragmentos de audio han sido creados y almacenados.")

# Se crea un modelo de reconocimiento de voz para el idioma español. Adicionalmente, el modelo se carga en la GPU para acortar los tiempos de procesado
model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-spanish", device="cuda")

# Se crea un modelo de procesamiento de lenguaje natural para el idioma español. Entre otras cosas, este modelo permite lematizar y eliminar stop words
nlp = spacy.load("es_core_news_sm")

# Se crean los archivos para guardar las transcripciones
with open(TRANSCRIPT_ORIGINAL_FILE, "w", encoding="utf-8") as original_f, \
     open(TRANSCRIPT_NORMALIZED_FILE, "w", encoding="utf-8") as normalized_f, \
     open(TRANSCRIPT_LEMMATIZED_FILE, "w", encoding="utf-8") as lemmatized_f, \
     open(TRANSCRIPT_STOPWORDS_FILE, "w", encoding="utf-8") as stopwords_f, \
     open(TRANSCRIPT_ALL_FILTERS_FILE, "w", encoding="utf-8") as all_filters_f:

    for i, chunk_path in enumerate(chunk_paths):
        print(f"\n🟡 Procesando chunk {i + 1}/{len(chunk_paths)}: {chunk_path}")

        # El modelo se encarga de transcribir los fragmentos de audio
        transcriptions = model.transcribe([chunk_path])

        for transcription in transcriptions:
            
            # Se extrae el texto de la transcripcion
            texto = transcription["transcription"]

            # Se guarda la transcripción original en su archivo txt correspondiente
            original_f.write(f"Chunk {i + 1}:\n{texto}\n\n")

            # Se realiza la normalización del texto: pasar a minusculas, corregir ortografia y eliminar espacios extra
            texto_normalizado = texto.lower()
            texto_normalizado = corregir_ortografia(texto_normalizado).replace("  ", " ").strip()

            # Se guarda la transcripción normalizada en su archivo txt correspondiente
            normalized_f.write(f"Chunk {i + 1}:\n{texto_normalizado}\n\n")

            # Se procesa el texto original con spacy para la lematizacion y eliminacion de stop words
            # Procesar el texto con nlp permite acceder a los tokens (palabras) del texto y poder hacer cambios con mayor facilidad
            doc_original = nlp(texto)
            
            # Se extrae la forma lematizada de cada token (palabra) del texto original
            texto_lemmatizado = [token.lemma_ for token in doc_original]

            # Se guarda la transcripción lematizada en su archivo txt correspondiente
            lemmatized_f.write(f"Chunk {i + 1}:\n{' '.join(texto_lemmatizado)}\n\n")

            # Se filtran los stop words del texto original
            texto_sin_stopwords = [token.text for token in doc_original if not token.is_stop]

            # Se guarda la transcripción sin stop words en su archivo txt correspondiente
            stopwords_f.write(f"Chunk {i + 1}:\n{' '.join(texto_sin_stopwords)}\n\n")

            # Se extrae la forma lematizada de cada token (palabra) del texto normalizado
            doc = nlp(texto_normalizado)

            # Se extrae la forma lematizada de cada token (palabra) del texto normalizado y se filtran los stop words
            # De esta forma se esta realizando un filtrado completo aplicando lematizacion, eliminando stop words y normalizacion del texto
            texto_final = [token.lemma_ for token in doc if not token.is_stop]

            # Se guarda la transcripción final con todos los filtros aplicados en su archivo txt correspondiente
            all_filters_f.write(f"Chunk {i + 1}:\n{' '.join(texto_final)}\n\n")

print("\n✅ Transcripciones guardadas.")