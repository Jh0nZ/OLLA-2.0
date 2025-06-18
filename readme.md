# Guía Rápida: Configuración de Entorno de Desarrollo
Sigue estos pasos para preparar tu entorno de desarrollo para este proyecto.


## Requisitos Previos
- Python 3.12.x instalado (para backend).
- Node.js y npm instalados (para frontend).

## Guía para Backend (Python)

### 1. Crear el Entorno Virtual

Abre la terminal en la raíz del proyecto y ejecuta:

```bash
python -m venv venv
```

Esto creará una carpeta `venv` con el entorno virtual.

### 2. Activar el Entorno Virtual

- **Windows:**
    ```bash
    .\venv\Scripts\activate
    ```
- **macOS/Linux:**
    ```bash
    source venv/bin/activate
    ```

El prompt de la terminal cambiará indicando que el entorno está activo.

### 3. Instalar Dependencias

Con el entorno virtual activo, instala las dependencias:

```bash
pip install -r requirements.txt
```

### 4. (Opcional) Configurar Soporte para CUDA

Si deseas usar la GPU para acelerar el procesamiento:

1. Descarga e instala la versión de CUDA compatible con tu GPU desde el [archivo de CUDA](https://developer.nvidia.com/cuda-toolkit-archive).
2. Instala PyTorch con soporte CUDA (ajusta la versión según tu instalación):

    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    ```

Consulta la [guía oficial de PyTorch](https://pytorch.org/get-started/locally/) para más detalles.

### 5. Desactivar el Entorno Virtual

Cuando termines, desactiva el entorno con:

```bash
deactivate
```

### 6. Comandos Útiles

- **Entrenar el modelo:**
    ```bash
    python -m backend.gan.train
    ```
- **Ejecutar el modelo:**
    ```bash
    python -m backend.gan.user
    ```

---
## Guía para Backend
En la carpeta `/backend`, inicia:

```bash
cd backend
fastapi dev main.py
```

## Guía para Frontend (Node.js)

### 1. Instalar Dependencias

En la carpeta `/frontend`, instala las dependencias:

```bash
cd frontend
npm install
```

### 2. Ejecutar el Servidor de Desarrollo

Inicia el frontend en modo desarrollo:

```bash
npm run dev
```

Consulta la terminal para la URL local donde se ejecuta la aplicación.


## Dependencias
protobuf
torch
transformers
fastapi