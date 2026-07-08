# Design: Mozilla Data Collective — Descarga y Procesamiento Local de Common Voice

**Fecha:** 2026-06-25
**Estado:** Aprobado

## Contexto

Mozilla retiró los datos de Common Voice de HuggingFace en octubre 2025. Los scripts `cv_metadata.py`, `analizar_cv.py` y `exportar_cv.py` fallan porque dependen de streaming HuggingFace. El corpus ahora se distribuye exclusivamente a través de Mozilla Data Collective con una API REST autenticada.

El dataset objetivo es **Common Voice Scripted Speech 26.0 - Spanish**:
- ID: `cmqim2spa00synr071fcp7av0`
- Tamaño: 48.30 GB
- Formato: `tar.gz` con archivos MP3 + TSV de metadatos
- Licencia: CC0-1.0
- Campos TSV: `client_id`, `path`, `sentence`, `gender`, `age`, `up_votes`, `down_votes`, etc.

## Objetivo

Reemplazar el pipeline de streaming HuggingFace con un pipeline local: descarga única del corpus completo, seguido de análisis y exportación desde disco.

## Archivos afectados

### Nuevos
- `descargar_cv.py` — descarga el corpus via API de Mozilla Data Collective con soporte de reanudación

### Reescritos
- `cv_metadata.py` — lee el TSV local en lugar de hacer streaming desde HuggingFace
- `exportar_cv.py` — lee MP3s locales en lugar de decodificar desde HuggingFace datasets

### Pequeños cambios
- `analizar_cv.py` — agrega check de existencia de `./data/cv-raw/` con mensaje de ayuda
- `tests/test_cv_metadata.py` — actualizar mocks para archivos TSV locales
- `tests/test_exportar_cv.py` — actualizar mocks para MP3s locales
- `LICENSES.md` — actualizar con dataset correcto (Scripted Speech 26.0)
- `Taskfile.yml` — agregar tarea `descargar-cv`
- `.env` — agregar `MDC_API_KEY`

### Sin cambios
`audio_utils.py`, `mls_metadata.py`, `analizar_mls.py`, `exportar_dataset.py`, `conftest.py`

## Componente 1: `descargar_cv.py`

**Responsabilidad:** Descargar y extraer el corpus de Mozilla Data Collective.

**Variables de entorno requeridas:**
- `MDC_API_KEY` — API key de Mozilla Data Collective (en `.env`, gitignored)

**Dataset fijo:** `cmqim2spa00synr071fcp7av0` (Common Voice Scripted Speech 26.0 - Spanish)

**Flujo:**

1. Verificar si `./data/cv-raw/` ya existe → si existe y no hay `--force`, salir con mensaje.
2. Llamar `POST <MDC_API_BASE>/datasets/cmqim2spa00synr071fcp7av0/download` con header `Authorization: Bearer <MDC_API_KEY>` → obtiene `downloadUrl` (válida 12h), `sizeBytes`, `checksum` (SHA-256), `filename`.
3. Verificar si existe `./data/cv-raw.tar.gz.part`:
   - Si existe: obtener su tamaño actual → enviar `Range: bytes=<offset>-` al `downloadUrl`
   - Si no: descargar desde el inicio
4. Descargar en chunks de 8 MB con `tqdm` mostrando velocidad + progreso.
5. Al completar la descarga: verificar SHA-256. Si falla → borrar `.part` y salir con error.
6. Extraer el `.tar.gz` a `./data/cv-raw/` con `tarfile`.
7. Eliminar el `.tar.gz` tras extracción exitosa.

**Archivos temporales:**
- `./data/cv-raw.tar.gz.part` — descarga parcial (se elimina tras extracción exitosa)

**CLI:**
```bash
uv run descargar_cv.py           # descarga a ./data/cv-raw/
uv run descargar_cv.py --force   # re-descarga aunque exista ./data/cv-raw/
```

**Nota sobre la API base URL:** El endpoint raíz debe verificarse en `https://mozilladatacollective.com/api-reference/docs`. La constante `MDC_API_BASE` en el código debe ser confirmada durante implementación.

## Componente 2: `cv_metadata.py` (reescritura)

**Responsabilidad:** Descubrir el TSV local y agregar speakers femeninas.

**Interfaz pública:**
```python
def find_clips_tsv(data_dir: Path) -> Path:
    """Busca el archivo TSV de clips en el directorio extraído.
    Busca el primer .tsv con columnas client_id, path, gender, sentence.
    Lanza FileNotFoundError si no encuentra ninguno."""

def get_top_female_speakers(data_dir: Path, n: int = 20) -> pd.DataFrame:
    """Lee el TSV local, filtra gender=='female', agrupa por client_id.
    Retorna DataFrame con columnas: client_id (str), clips (int), minutes (float).
    minutes = clips * 5.0 / 60 si no hay columna de duración en el TSV."""
```

**Cambio de firma:** `get_top_female_speakers` ahora recibe `data_dir: Path` como primer argumento (antes no recibía argumentos).

## Componente 3: `analizar_cv.py` (pequeño cambio)

Agrega verificación de existencia de `./data/cv-raw/` antes de llamar a `get_top_female_speakers`:

```python
DATA_DIR = Path("./data/cv-raw")
if not DATA_DIR.exists():
    print("Error: corpus no descargado. Ejecuta: task descargar-cv")
    sys.exit(1)
df = get_top_female_speakers(DATA_DIR, n=10)
```

## Componente 4: `exportar_cv.py` (reescritura)

**Responsabilidad:** Exportar clips de un speaker a formato LJSpeech leyendo desde disco.

**CLI:**
```bash
uv run exportar_cv.py --speaker <client_id> \
    [--data-dir ./data/cv-raw] \
    [--output ./data/<client_id[:8]>]
```

**Flujo:**
1. Leer TSV via `cv_metadata.find_clips_tsv(data_dir)`
2. Filtrar filas donde `client_id == speaker`
3. Por cada fila: leer `<data_dir>/clips/<path>` como bytes → `decode_audio_bytes()` de `exportar_dataset.py`
4. Procesar con `audio_utils.process_clip` + `save_wav`
5. Escribir `metadata.csv` en formato LJSpeech (`wavs/<prefix>_NNNN.wav|sentence`)

**Nota:** Importa `decode_audio_bytes` y `write_metadata_csv` desde `exportar_dataset.py` (misma lógica, sin duplicar). `build_wav_filename` permanece local en `exportar_cv.py` porque usa `client_id[:8]` como prefijo, no el `speaker_id` completo de MLS.

## Flujo de datos (post-cambio)

```
Mozilla Data Collective API
    → presigned URL (12h)
    → ./data/cv-raw.tar.gz.part   (descarga reanudable)
    → ./data/cv-raw/              (extracción)
        ├── clips/*.mp3
        └── *.tsv                 (clips metadata)

analizar_cv.py
    → cv_metadata.find_clips_tsv() → lee TSV → top speakers femeninas

exportar_cv.py
    → cv_metadata.find_clips_tsv() → filtra por client_id
    → lee clips/*.mp3 → decode_audio_bytes() → process_clip() → save_wav()
    → ./data/<client_id[:8]>/wavs/*.wav + metadata.csv
```

## Testing

- `test_descargar_cv.py`: mock del endpoint HTTP (no red real), mock de `tarfile.open`
- `test_cv_metadata.py`: reescribir usando `tmp_path` con TSV de prueba en disco
- `test_exportar_cv.py`: reescribir usando `tmp_path` con MP3 sintético y TSV de prueba

Todos los tests deben pasar sin red ni `MDC_API_KEY`.

## Taskfile

Agregar:
```yaml
descargar-cv:
  desc: "Descarga Common Voice Scripted Speech 26.0 Spanish a ./data/cv-raw/"
  cmds:
    - uv run descargar_cv.py
```

## Criterios de éxito

- `task descargar-cv` descarga, verifica checksum y extrae el corpus
- Si se interrumpe y se vuelve a ejecutar, reanuda desde donde quedó
- `task analizar-cv` lista hablantes femeninas leyendo desde `./data/cv-raw/`
- `task exportar-cv SPEAKER=<id>` produce LJSpeech compatible con piper-tts
- Todos los tests unit pasan sin red

## Tasks de implementación

- [ ] Crear `descargar_cv.py` + `tests/test_descargar_cv.py`
- [ ] Reescribir `cv_metadata.py` + actualizar `tests/test_cv_metadata.py`
- [ ] Actualizar `analizar_cv.py`
- [ ] Reescribir `exportar_cv.py` + actualizar `tests/test_exportar_cv.py`
- [ ] Actualizar `LICENSES.md` y `Taskfile.yml`
