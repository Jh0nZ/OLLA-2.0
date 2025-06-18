interface IngredienteCardProps {
  nombre: string;
  imagen: string;
  onClick?: () => void;
  disabled?: boolean;
}

export default function IngredienteCard({
  nombre,
  imagen,
  onClick,
  disabled = false,
}: IngredienteCardProps) {
  return (
    <div
      className={`flex flex-col items-center justify-end h-32 p-4 rounded-lg transition-colors w-full ${
        disabled
          ? "opacity-50 cursor-not-allowed"
          : "hover:bg-gray-700 cursor-pointer"
      }`}
      style={{
        background: `linear-gradient(to bottom, rgba(0,0,0,0.3), rgba(0,0,0,0.7)), url(${imagen}) center/cover`,
      }}
      onClick={disabled ? undefined : onClick}
    >
      <span className="text-white text-sm text-center font-medium">
        {nombre}
      </span>
    </div>
  );
}
