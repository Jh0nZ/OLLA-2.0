import { useState } from "react";
import ImageDropdown from "./ImageDropdown";

interface IngredienteFormProps {
  onSubmit: (formData: FormData) => Promise<void>;
  loading: boolean;
  className?: string;
}

export default function IngredienteForm({
  onSubmit,
  loading,
  className,
}: IngredienteFormProps) {
  const [form, setForm] = useState({
    name: "",
    imageFile: null as File | null,
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    const formData = new FormData();
    formData.append("nombre", form.name);
    if (form.imageFile) formData.append("imagen", form.imageFile);

    await onSubmit(formData);
    setForm({ name: "", imageFile: null });
  };

  return (
    <div className={className}>
      <h2 className="text-xl font-semibold mb-4">Agregar Ingrediente</h2>
      <form onSubmit={handleSubmit} className="space-y-3">
        <input
          type="text"
          value={form.name}
          onChange={(e) => setForm({ ...form, name: e.target.value })}
          placeholder="Nombre del ingrediente"
          className="w-full border rounded px-2 py-1 text-sm focus:outline-none"
          required
        />
        <ImageDropdown
          image = {form.imageFile}
          setImage={(file) => setForm({ ...form, imageFile: file })}
        />
        <button
          type="submit"
          disabled={loading || !form.imageFile}
          className="w-full bg-gray-600 text-white py-1 px-3 text-sm rounded hover:bg-gray-700 disabled:bg-gray-400"
        >
          {loading ? "Agregando..." : "Agregar"}
        </button>
      </form>
    </div>
  );
}
