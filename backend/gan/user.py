from .generator import RecipeGenerator
import os

# Cargar el modelo más reciente
def get_latest_model(path):
    """Busca el modelo más reciente en el directorio especificado"""
    if not os.path.exists(path):
        return None
    
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    folders.sort(reverse=True)  # Ordenar por fecha descendente
    return os.path.join(path, folders[0]) if folders else None

def interactive_recipe_generator():
    """Interfaz interactiva para generar recetas"""
    # Rutas actualizadas para Windows
    models_base_path = "models/generator"
    latest_gen_path = get_latest_model(models_base_path)
    
    print("=== GENERADOR DE RECETAS OLLA 2.0 ===")
    print(f"Buscando modelos en: {models_base_path}")
    
    # Inicializar generador
    generator = RecipeGenerator()
    
    if latest_gen_path and os.path.exists(latest_gen_path):
        print(f"✓ Modelo encontrado: {latest_gen_path}")
        try:
            generator.load(latest_gen_path)
            print("✓ Modelo cargado exitosamente")
        except Exception as e:
            print(f"⚠ Error al cargar modelo: {e}")
            print("Usando modelo base en español...")
    else:
        print("⚠ No se encontró modelo entrenado personalizado")
        print("Usando modelo GPT-2 en español base...")
    
    print("\n" + "="*50)
    print("Instrucciones:")
    print("- Ingrese ingredientes separados por comas")
    print("- Escriba 'salir' para terminar")
    print("- Escriba 'config' para cambiar configuración")
    print("="*50)
    
    # Configuración por defecto
    config = {
        'max_length': 200,
        'temperature': 0.8
    }
    
    while True:
        try:
            print("\n" + "-"*30)
            user_input = input("🥘 Ingredientes: ").strip()
            
            if user_input.lower() in ['salir', 'exit', 'quit']:
                print("¡Hasta luego! 👋")
                break
            
            elif user_input.lower() == 'config':
                print("\n--- Configuración actual ---")
                print(f"Longitud máxima: {config['max_length']}")
                print(f"Temperatura (creatividad): {config['temperature']}")
                
                try:
                    new_length = input(f"Nueva longitud máxima [{config['max_length']}]: ").strip()
                    if new_length:
                        config['max_length'] = int(new_length)
                    
                    new_temp = input(f"Nueva temperatura (0.1-1.0) [{config['temperature']}]: ").strip()
                    if new_temp:
                        config['temperature'] = float(new_temp)
                        if config['temperature'] < 0.1 or config['temperature'] > 1.0:
                            config['temperature'] = 0.8
                            print("⚠ Temperatura fuera de rango, usando 0.8")
                    
                    print("✓ Configuración actualizada")
                except ValueError:
                    print("⚠ Valores inválidos, manteniendo configuración actual")
                continue
            
            elif not user_input:
                print("⚠ Por favor ingrese algunos ingredientes")
                continue
            
            # Generar receta
            print("\n🤖 Generando receta...")
            
            receta = generator.generate(
                user_input, 
                max_length=config['max_length'],
                temperature=config['temperature']
            )
            
            # Mostrar resultado
            print("\n" + "="*50)
            print("🍽️  RECETA GENERADA")
            print("="*50)
            print(f"📋 Ingredientes: {user_input}")
            print(f"👨‍🍳 Procedimiento:\n")
            print(receta)
            print("="*50)
            
            # Opción de guardar
            save_option = input("\n¿Guardar esta receta? (s/n): ").strip().lower()
            if save_option in ['s', 'si', 'sí', 'y', 'yes']:
                save_recipe(user_input, receta)
        
        except KeyboardInterrupt:
            print("\n\n¡Hasta luego! 👋")
            break
        except Exception as e:
            print(f"⚠ Error: {e}")
            print("Intente nuevamente...")

def save_recipe(ingredientes, receta):
    """Guarda la receta generada en un archivo"""
    try:
        recipes_dir = "generated_recipes"
        os.makedirs(recipes_dir, exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"receta_{timestamp}.txt"
        filepath = os.path.join(recipes_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=== RECETA GENERADA OLLA 2.0 ===\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Ingredientes: {ingredientes}\n")
            f.write("="*40 + "\n")
            f.write("PROCEDIMIENTO:\n")
            f.write(receta)
            f.write("\n" + "="*40)
        
        print(f"✓ Receta guardada en: {filepath}")
    
    except Exception as e:
        print(f"⚠ Error al guardar: {e}")

def generate_single_recipe(ingredientes, max_length=200, temperature=0.8):
    """Función para generar una sola receta (para uso programático)"""
    models_base_path = "models/generator"
    latest_gen_path = get_latest_model(models_base_path)
    
    generator = RecipeGenerator()
    
    if latest_gen_path and os.path.exists(latest_gen_path):
        try:
            generator.load(latest_gen_path)
        except:
            pass  # Usar modelo base si hay error
    
    return generator.generate(ingredientes, max_length=max_length, temperature=temperature)

if __name__ == "__main__":
    interactive_recipe_generator()
