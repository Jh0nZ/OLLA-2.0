# Entrenar un nuevo modelo
from backend.gan.train import train
train(
    epochs=50,
    model_label="olla-gpt2-v2.0", 
    model_description="Modelo de recetas en español",
)

# # Listar todos los modelos
# from backend.gan.model_manager import list_models
# list_models()

# # Cargar un modelo específico
# from backend.gan.model_manager import load_model
# generator, discriminator = load_model("Modelo_Recetas_Espanol_v1")

# # Generar una receta
# if generator:
#     receta = generator.generate("tomate, cebolla, carne de res, arroz, huevo, comino molido", max_length=500, temperature=0.96)
#     print(receta)

# from backend.gan.train import load_dataset

# test = load_dataset()
# print(test)