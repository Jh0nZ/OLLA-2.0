import IngredienteCard from "./IngredienteCard";

interface Ingrediente {
  nombre: string;
  imagen: string;
}

interface IngredientesListProps {
  ingredientes: Ingrediente[];
  apiUrl: string;
  className?: string;
}

export default function IngredientesList({
  ingredientes,
  apiUrl,
  className,
}: IngredientesListProps) {
  return (
    <div className={className}>
      <h2 className="text-xl font-semibold mb-4">Lista de Ingredientes</h2>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4 justify-items-center overflow-y-auto">
        {ingredientes.map((ingrediente, index) => (
          <IngredienteCard
            key={index}
            nombre={ingrediente.nombre}
            imagen={`${apiUrl}/datos/images/${ingrediente.imagen}`}
          />
        ))}
      </div>
    </div>
  );
}
