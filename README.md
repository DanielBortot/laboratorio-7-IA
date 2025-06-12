# Laboratorio 7
Este es el repositorio del laboratorio 7 donde se usa un modelo de Machine Learning para transcribir un audio

## Desarrolladores
<table align="center">
    <tbody>
        <tr>
            <td align="center"><a href="https://github.com/DanielBortot" rel="nofollow"><img src="https://avatars.githubusercontent.com/u/103535845?v=4" width="150px;" alt="" style="max-width:100%;"><br><sub><b>Daniel Bortot</b></sub></a><br><a href="" title="Commits"><g-emoji class="g-emoji" alias="book" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/1f4d6.png">📖</g-emoji></a></td>
        </tr>
    </tbody>
</table>

## Creacion del Ambiente Virtual e instalación de dependencias

```bash
# Crear entorno virtual
$ python -m venv venv

# Activar entorno virtual (Windows)
$ .venv\Scripts\activate

# Activar entorno virtual (Linux)
$ .venv\bin\activate

# Desactivar entorno virtual
$ deactivate

#Instalar dependencias
$ pip install huggingsound --no-deps
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
$ pip install datasets transformers jiwer librosa spacy pydub pyspellchecker
$ python -m spacy download es_core_news_sm
```
