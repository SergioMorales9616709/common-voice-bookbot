# Design: MLS Spanish Dataset Pipeline para Piper TTS

**Fecha:** 2026-06-24  
**Objetivo:** Extraer un dataset de voz femenina en español latinoamericano del corpus MLS, listo para entrenar un modelo ONNX con piper-tts en la nube.

## Atribución de licencia

El dataset de origen es **Multilingual LibriSpeech (MLS)** de Facebook/Meta, derivado de grabaciones de LibriVox.  
**Licencia:** CC BY 4.0 — uso comercial permitido con atribución.  
Atribución requerida: *"Audio data sourced from Multilingual LibriSpeech (Pratap et al., 2020), derived from LibriVox recordings. Licensed under CC BY 4.0."*

---

## Contexto

- **Dataset fuente:** `facebook/multilingual_librispeech`, subconjunto `spanish` (~220 horas, ~1,000 hablantes)
- **Plataforma de entrenamiento:** Cloud (Kaggle / Google Colab / VPS con GPU) — fuera del scope de este proyecto
- **OS de preparación:** Windows 10 con Python 3.13 + uv
- **Uso final:** Agentes conversacionales sobre LiveKit usando piper-tts como servicio HTTP

---

## Arquitectura

```
HuggingFace (facebook/multilingual_librispeech)
        │
        ▼
  analizar_mls.py
  ─ descarga metadatos de hablantes
  ─ filtra por género femenino
  ─ muestra ranking por minutos de audio
  ─ usuario elige speaker_id
        │
        ▼
  exportar_dataset.py  <── argumento: --speaker <speaker_id>
  ─ descarga clips de la hablante elegida
  ─ resample a 22050 Hz mono  (librosa)
  ─ loudness normalization EBU R128  (pyloudnorm)
  ─ escribe WAV a ./data/dataset/wavs/
  ─ genera ./data/dataset/metadata.csv
        │
        ▼
  ./data/dataset/
    wavs/
      <speaker_id>_0001.wav
      <speaker_id>_0002.wav
      ...
    metadata.csv   (LJSpeech: filename|transcripción)
```

---

## Componentes

### `analizar_mls.py`

**Responsabilidad:** Explorar hablantes del corpus MLS Spanish y ayudar al usuario a elegir una hablante femenina con suficiente audio.

**Entradas:** Ninguna (descarga automática vía HuggingFace Hub).

**Salida:** Tabla en consola con top 10 hablantes femeninas por minutos totales.

**Comportamiento:**
1. Usa streaming de HuggingFace para no descargar el audio completo (~220 GB)
2. Acumula `speaker_id`, `gender`, y suma de `duration` por hablante
3. Filtra `gender == "female"`
4. Imprime ranking con `speaker_id | minutos | clips`
5. Al final imprime el comando exacto a ejecutar: `uv run exportar_dataset.py --speaker <speaker_id>`

**Consideración:** MLS no expone `gender` directamente en el dataset de HuggingFace. Se descarga el archivo `metainfo.txt` del repositorio MLS en HuggingFace Hub para cruzar `speaker_id → gender`.

---

### `exportar_dataset.py`

**Responsabilidad:** Descargar todos los clips de una hablante, procesarlos y generar el dataset en formato LJSpeech.

**Entradas:** `--speaker <speaker_id>` (argumento CLI).

**Salida:** `./data/dataset/wavs/*.wav` + `./data/dataset/metadata.csv`

**Pipeline de audio por clip:**
1. Leer bytes de audio desde el campo `audio` del dataset
2. Resample a **22050 Hz mono** con `librosa.resample`
3. Normalizar loudness a **-23 LUFS (EBU R128)** con `pyloudnorm`
4. Escribir como WAV 16-bit con `soundfile`

**Formato `metadata.csv`:**
```
filename|text
wavs/speaker_0001.wav|La casa está al lado del río.
wavs/speaker_0002.wav|El tren llega a las ocho de la mañana.
```
Sin encabezado, separador `|`, sin normalización de texto adicional (piper lo maneja internamente).

**Manejo de errores:**
- Clips corruptos o con duración < 0.5 s: omitir y loggear
- Si el `speaker_id` no existe en el dataset: error claro con mensaje al usuario

---

## Dependencias nuevas

| Librería | Uso | Agregar a pyproject.toml |
|---|---|---|
| `pyloudnorm` | Normalización EBU R128 | Sí |
| `datasets` | Streaming de HuggingFace | Sí |

Las demás (`librosa`, `soundfile`, `huggingface-hub`, `pyarrow`) ya están en el proyecto.

---

## Fuera de scope

- Entrenamiento del modelo piper (se hace en la nube)
- Normalización de texto / limpieza de transcripciones
- Separación de fuentes con Demucs (MLS ya es audio limpio)
- Empaquetado/compresión del dataset (el usuario lo hace manualmente)
- Evaluación de calidad del modelo resultante

---

## Criterio de éxito

El pipeline es exitoso cuando:
1. `analizar_mls.py` muestra un ranking de hablantes femeninas con duración total
2. `exportar_dataset.py` genera un `dataset/` con WAVs a 22050 Hz normalizados y un `metadata.csv` válido
3. El dataset puede subirse directamente a Kaggle/Colab y ser consumido por los scripts de entrenamiento de piper sin modificaciones adicionales
