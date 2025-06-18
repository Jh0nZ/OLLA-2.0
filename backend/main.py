from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import time
from .utils.file_manager import cargar_dataset, guardar_dataset
from .gan.model_manager import ModelManager

app = FastAPI(title="OLLA 2.0 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMAGES_DIR = Path("datos/images")
(IMAGES_DIR / "recetas").mkdir(parents=True, exist_ok=True)
(IMAGES_DIR / "ingredientes").mkdir(parents=True, exist_ok=True)

app.mount("/datos/images", StaticFiles(directory="datos/images"), name="imagenes")

def save_image(file: UploadFile, folder: str) -> str:
    timestamp = int(time.time())
    ext = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
    filename = f"{timestamp}.{ext}"
    
    with open(IMAGES_DIR / folder / filename, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    return f"{folder}/{filename}"

@app.post("/ingredientes")
async def agregar_ingrediente(nombre: str = Form(...), imagen: UploadFile = File(...)):
    data = cargar_dataset()
    
    if any(ing["nombre"] == nombre for ing in data["ingredientes"]):
        return {"mensaje": "El ingrediente ya existe"}
    
    imagen_path = save_image(imagen, "ingredientes")
    ingrediente = {"nombre": nombre, "imagen": imagen_path}
    data["ingredientes"].append(ingrediente)
    guardar_dataset(data)
    
    return {"mensaje": "Ingrediente agregado correctamente", "ingrediente": ingrediente}

@app.get("/ingredientes")
def listar_ingredientes():
    return cargar_dataset()["ingredientes"]

@app.post("/recetas")
async def agregar_receta(
    receta: str = Form(...),
    procedimiento: str = Form(...),
    ingredientes: str = Form(...),
    imagen: UploadFile = File(...)
):
    data = cargar_dataset()
    imagen_path = save_image(imagen, "recetas")
    
    nueva_receta = {
        "receta": receta,
        "procedimiento": procedimiento,
        "ingredientes": [i.strip() for i in ingredientes.split(",")],
        "imagen": imagen_path
    }
    
    data["recetas"].append(nueva_receta)
    guardar_dataset(data)
    
    return {"mensaje": "Receta guardada correctamente", "receta": nueva_receta}

@app.get("/recetas")
def listar_recetas():
    return cargar_dataset()["recetas"]

@app.get("/models")
def obtener_labels_modelos():
    return [model["label"] for model in ModelManager.load_models_registry().get("models", [])]

@app.post("/generar-receta")
def generar_receta(
    ingredientes: str = Form(...),
    modelo: str = Form(...),
    temperatura: float = Form(0.96),
    max_length: int = Form(500)
):
    generator, _ = ModelManager.load_generator_and_discriminator(label=modelo)
    
    if generator is None:
        raise HTTPException(status_code=404, detail=f"Modelo '{modelo}' no encontrado")
    
    receta_generada = generator.generate(ingredientes, max_length=max_length, temperature=temperatura)
    
    return {
        "receta": receta_generada,
        "modelo_usado": modelo,
        "ingredientes": ingredientes,
        "parametros": {"temperatura": temperatura, "max_length": max_length}
    }