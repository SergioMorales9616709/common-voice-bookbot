# Design: Soporte para Mozilla Common Voice (Opción A — Scripts Paralelos)

**Fecha:** 2026-06-25
**Estado:** Aprobado

## Contexto

El proyecto ya extrae audio de Multilingual LibriSpeech (MLS) de Facebook para entrenar modelos piper-tts. Se agrega soporte para Mozilla Common Voice como segunda fuente de datos, manteniendo los scripts MLS intactos. Ambas fuentes producen datasets en formato LJSpeech idéntico.

## Objetivo

Permitir exportar hablantes femeninas de español desde Mozilla Common Voice con la misma interfaz y formato de salida que MLS, para usarlos indistintamente en experimentos de entrenamiento.

## Archivos sin cambios

- `audio_utils.py` — reutilizado sin modificaciones
- `mls_metadata.py` — sin cambios
- `analizar_mls.py` — sin cambios
- `exportar_dataset.py` — sin cambios

## Archivos nuevos

### `cv_metadata.py`

Módulo de metadatos para Common Voice. Descarga el split `train` del dataset `mozilla-foundation/common_voice_17_0` (subset `es`) en modo streaming, filtra por `gender == "female"`, y acumula duración por `client_id`.

Interfaz pública:

```python
def get_top_female_speakers(n: int = 20) -> pd.DataFrame:
    """Retorna DataFrame con columnas: client_id, minutes, clips."""
```

No descarga audio durante el análisis — solo lee los campos de metadatos del stream.

### `analizar_cv.py`

Script de análisis equivalente a `analizar_mls.py`. Imprime el top 10 de hablantes femeninas de CV español por minutos de audio y sugiere el comando de exportación.

Uso:
```bash
uv run analizar_cv.py
```

### `exportar_cv.py`

Script de exportación equivalente a `exportar_dataset.py`. Itera el dataset CV en streaming filtrando por `client_id`, decodifica MP3 con `av`, procesa con `process_clip` / `save_wav` de `audio_utils.py`, y escribe `metadata.csv` en formato LJSpeech.

Interfaz:
```bash
uv run exportar_cv.py --speaker <client_id> [--output ./data/<client_id>]
```

Diferencias de esquema respecto a MLS:

| Campo | MLS | Common Voice |
|---|---|---|
| Identificador speaker | `speaker_id` (int) | `client_id` (hash) |
| Texto | `transcript` | `sentence` |
| Formato audio | Opus/FLAC | MP3 |

El pipeline de audio (`process_clip`) es idéntico en ambas fuentes.

### `LICENSES.md`

Nota de licencia en la raíz del proyecto:

- **MLS (Multilingual LibriSpeech):** CC BY 4.0 — requiere atribución a Facebook/LibriVox
- **Mozilla Common Voice:** CC0 — dominio público, sin restricciones

## Flujo de datos (Common Voice)

```
HuggingFace (mozilla-foundation/common_voice_17_0, es, streaming)
    → filtrar client_id
    → campo "audio" → bytes MP3
    → decode_audio_bytes() [av]
    → process_clip() [librosa + pyloudnorm]
    → save_wav() [soundfile]
    → wavs/<client_id>_NNNN.wav
    → metadata.csv (LJSpeech: filename|text)
```

## Estructura de archivos resultante

```
common-voice-bookbot/
├── audio_utils.py
├── mls_metadata.py
├── analizar_mls.py
├── exportar_dataset.py
├── cv_metadata.py          # NUEVO
├── analizar_cv.py          # NUEVO
├── exportar_cv.py          # NUEVO
└── LICENSES.md             # NUEVO
```

## Tasks de implementación

- [ ] Crear `cv_metadata.py` con `get_top_female_speakers()`
- [ ] Crear `analizar_cv.py` (top 10 hablantes femeninas CV español)
- [ ] Crear `exportar_cv.py` con la misma interfaz CLI que `exportar_dataset.py`
- [ ] Crear `LICENSES.md` con notas de licencia para MLS y Common Voice

## Criterios de éxito

- `uv run analizar_cv.py` lista hablantes femeninas de CV español sin errores
- `uv run exportar_cv.py --speaker <client_id>` produce un directorio con `wavs/` y `metadata.csv` compatible con piper-tts
- Los scripts MLS existentes funcionan sin cambios
- `LICENSES.md` documenta ambas licencias correctamente
