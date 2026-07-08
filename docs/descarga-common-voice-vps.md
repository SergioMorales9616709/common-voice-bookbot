# Descargar y descomprimir Common Voice en un VPS Linux

Guía para correr `descargar_cv.py` en un VPS en lugar de localmente. Pensada para el caso en que la descarga/extracción local (Windows) lleva más de 12 horas y nunca termina.

## Por qué se cuelga localmente en Windows

El corpus objetivo es **Common Voice Scripted Speech 26.0 - Spanish** (dataset `cmqim2spa00synr071fcp7av0` en Mozilla Data Collective):

- **48.30 GB** comprimidos en un único `.tar.gz`
- Contiene **cientos de miles de archivos MP3** individuales (uno por clip) más un TSV de metadatos

El script extrae con `tarfile.extractall()`, que escribe archivo por archivo en un solo hilo. En Windows esto es particularmente lento porque:

- **Windows Defender** (u otro antivirus con protección en tiempo real) escanea cada archivo nuevo a medida que se crea. Con cientos de miles de MP3 pequeños, esto multiplica el tiempo de extracción muchas veces — es la causa más común de "la extracción nunca termina" en Windows.
- **NTFS** tiene más overhead por archivo que los sistemas de archivos típicos de Linux (ext4/xfs) cuando hay que crear una cantidad masiva de archivos pequeños.
- El indexador de búsqueda de Windows puede sumarse a escanear la carpeta mientras se llena.
- Si el disco es HDD en vez de SSD, todo lo anterior se agrava.

Ninguno de estos problemas está en el código del proyecto — es un problema de plataforma. Un VPS Linux con disco SSD/NVMe y sin antivirus interceptando cada `write()` resuelve esto directamente.

## Requisitos del VPS

- **SO:** Ubuntu 22.04/24.04 o Debian 12 (cualquier distro con `apt` sirve; ajustar el gestor de paquetes si usás otra)
- **Disco:** al menos **150 GB libres**. Desglose:
  - `~48 GB` para `cv-raw.tar.gz.part` durante la descarga
  - `~48–55 GB` para `./data/cv-raw/` una vez extraído (el MP3 ya viene comprimido, así que el tar.gz no gana mucho más al descomprimir)
  - El script no borra el `.tar.gz.part` hasta que la extracción termina con éxito → durante buena parte del proceso **ambos coexisten en disco**, de ahí el margen
  - Espacio adicional para el dataset LJSpeech final de la hablante exportada (normalmente unos cientos de MB a pocos GB, mucho más chico que el corpus completo)
- **RAM:** 2 GB alcanza (la descarga es streaming por chunks de 8 MB, la extracción no carga todo en memoria)
- **CPU:** 2 vCPUs es más que suficiente (el cuello de botella es I/O, no CPU)
- Acceso SSH con un usuario que pueda instalar paquetes (sudo)

## Paso a paso

### 1. Provisionar el VPS

Cualquier proveedor (Hetzner, DigitalOcean, OVH, Contabo, etc.) con una instancia Ubuntu 24.04 y al menos 150 GB de disco. Si el plan base trae menos, agregar un volumen extra.

### 2. Conectarse por SSH

```bash
ssh usuario@ip-del-vps
```

### 3. Instalar dependencias del sistema

```bash
sudo apt update && sudo apt install -y git tmux build-essential curl
```

`tmux` es clave: permite que la descarga/extracción siga corriendo aunque se corte la conexión SSH.

### 4. Instalar `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

`uv` se encarga de instalar Python 3.13 automáticamente al hacer `uv sync`.

### 5. Clonar el repositorio

```bash
git clone <url-de-tu-repo> common-voice-bookbot
cd common-voice-bookbot
```

(Si no recordás la URL, correlo localmente: `git remote get-url origin`.)

### 6. Instalar dependencias del proyecto

```bash
uv sync
```

### 7. Configurar `.env`

El archivo `.env` está gitignoreado — no viaja con `git clone`, hay que crearlo de nuevo en el VPS:

```bash
nano .env
```

Contenido mínimo:

```
MDC_API_KEY=tu_api_key_de_mozilla_data_collective
```

(Agregar también `HF_TOKEN` si vas a usar los scripts de MLS en el mismo VPS.)

### 8. Abrir una sesión `tmux` persistente

```bash
tmux new -s cv-download
```

Todo lo que corras dentro de esta sesión sigue vivo aunque se corte el SSH.

### 9. Verificar espacio disponible

```bash
df -h .
```

Confirmar que hay al menos 150 GB libres antes de arrancar.

### 10. Ejecutar la descarga + extracción

Dentro de la sesión `tmux`:

```bash
uv run descargar_cv.py
```

(Si instalaste [go-task](https://taskfile.dev/installation/), el atajo equivalente es `task descargar-cv`. No es necesario — `uv run` alcanza.)

Qué hace el script, en orden:

1. Pide a la API de Mozilla Data Collective una URL de descarga firmada (válida 12 h)
2. Descarga en chunks de 8 MB a `./data/cv-raw.tar.gz.part`, con hasta 10 reintentos automáticos y backoff exponencial ante cortes de conexión
3. Verifica el checksum SHA-256 del archivo completo
4. Extrae el `.tar.gz` a `./data/cv-raw/`
5. Borra el `.tar.gz.part` solo si la extracción terminó bien

### 11. Salir de `tmux` sin matar el proceso

`Ctrl+B`, soltar, luego `D` (detach). Ahí ya podés cerrar la sesión SSH tranquilamente — el proceso sigue en el VPS.

### 12. Monitorear progreso

Reconectar en cualquier momento:

```bash
ssh usuario@ip-del-vps
tmux attach -t cv-download
```

O, desde otra sesión SSH sin interrumpir la que corre el proceso:

```bash
# Durante la descarga
du -sh ./data/cv-raw.tar.gz.part

# Durante/después de la extracción
du -sh ./data/cv-raw/
```

Evitá correr `find ./data/cv-raw -type f | wc -l` mientras la extracción está en curso — con cientos de miles de archivos ese comando también puede tardar y compite por I/O con el proceso de extracción.

### 13. Si la conexión se corta

- **Durante la descarga:** simplemente volvé a correr `uv run descargar_cv.py`. Detecta el `.tar.gz.part` existente y reanuda desde el offset con un header `Range`, pidiendo una URL firmada nueva si la anterior venció (12 h).
- **Durante la extracción:** el script no es reanudable archivo-por-archivo. Si se interrumpe, al volver a correrlo detecta que `./data/cv-raw/` quedó a medio extraer, la borra, re-verifica el checksum del `.tar.gz.part` (ya completo, no hace falta re-descargar) y **reinicia la extracción desde cero**. Es seguro, solo repite ese paso.

### 14. Confirmar que terminó bien

El script imprime `Corpus disponible en ...` al final, y `./data/cv-raw.tar.gz.part` ya no debe existir.

```bash
ls ./data/cv-raw/
```

Debe verse un `.tsv` de metadatos y una carpeta `clips/` con los MP3.

### 15. Analizar y exportar directamente en el VPS

No hace falta traer el corpus completo a tu máquina — analizá y exportá ahí mismo:

```bash
uv run analizar_cv.py
uv run exportar_cv.py --speaker <client_id>
```

El resultado (`./data/<client_id[:8]>/wavs/` + `metadata.csv`) es mucho más chico que el corpus crudo — es lo único que conviene transferir de vuelta.

### 16. Traer el resultado final a tu máquina local

Desde tu máquina local (no desde el VPS):

```bash
scp -r usuario@ip-del-vps:~/common-voice-bookbot/data/<client_id[:8]> ./data/
# o, si preferís rsync (más robusto ante cortes):
rsync -avz usuario@ip-del-vps:~/common-voice-bookbot/data/<client_id[:8]>/ ./data/<client_id[:8]>/
```

### 17. Limpieza en el VPS (opcional)

Si no vas a exportar más hablantes de este corpus, podés liberar los ~50 GB del corpus crudo:

```bash
rm -rf ./data/cv-raw/
```

No uses `task clean` para esto — borra todo `./data/`, incluyendo datasets ya exportados que todavía no hayas transferido.

## Troubleshooting

| Síntoma | Causa / solución |
|---|---|
| `Error: MDC_API_KEY no encontrado en .env` | Falta crear/completar `.env` en el VPS (paso 7) |
| `El servidor no soporta reanudación (HTTP ... en lugar de 206)` | El `Range` header fue ignorado. Borrar `./data/cv-raw.tar.gz.part` y volver a correr desde cero |
| `Error: checksum no coincide. Archivo eliminado.` | El script ya borró el `.part` corrupto — simplemente volver a correr |
| Extracción lenta también en el VPS | Verificar que el disco no sea un volumen de red lento; chequear con `iostat 1` o `top` que el proceso siga consumiendo I/O y no esté colgado |
| Falla a mitad de extracción por espacio | Confirmar con `df -h` antes de empezar (paso 9); necesitás los ~150 GB del punto anterior |

## Resumen de comandos

```bash
sudo apt update && sudo apt install -y git tmux build-essential curl
curl -LsSf https://astral.sh/uv/install.sh | sh && source "$HOME/.local/bin/env"
git clone <url-de-tu-repo> common-voice-bookbot && cd common-voice-bookbot
uv sync
nano .env   # MDC_API_KEY=...
df -h .
tmux new -s cv-download
uv run descargar_cv.py
# Ctrl+B, D para hacer detach
# tmux attach -t cv-download   para reconectar
uv run analizar_cv.py
uv run exportar_cv.py --speaker <client_id>
# desde tu máquina local:
rsync -avz usuario@ip-del-vps:~/common-voice-bookbot/data/<client_id[:8]>/ ./data/<client_id[:8]>/
```
