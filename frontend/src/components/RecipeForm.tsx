"use client";

import React, { useState, useCallback } from "react";
import ImageDropdown from "./ImageDropdown";
import IngredientSelector from "./IngredientSelector";
import type { Receta, Ingredient } from "@/app/creador/page";

interface RecipeFormProps {
  onRecipeAdded: (receta: Receta) => void;
  className?: string;
}

export default function RecipeForm({
  onRecipeAdded,
  className,
}: RecipeFormProps) {
  const [selectedIngredients, setSelectedIngredients] = useState<Ingredient[]>(
    []
  );
  const [formData, setFormData] = useState<Receta>({
    receta: "",
    ingredientes: [],
    procedimiento: "",
    imagen: undefined,
  });
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleInputChange = useCallback(
    (field: keyof Receta, value: string | File | undefined) => {
      setFormData((prev) => ({ ...prev, [field]: value }));
    },
    []
  );

  const handleIngredientsChange = useCallback((ingredients: Ingredient[]) => {
    setSelectedIngredients(ingredients);
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    if (selectedIngredients.length === 0) {
      setError("Debe seleccionar al menos un ingrediente");
      setLoading(false);
      return;
    }

    try {
      const form = new FormData();
      form.append("receta", formData.receta);
      form.append(
        "ingredientes",
        selectedIngredients.map((i) => i.nombre).join(",")
      );
      form.append("procedimiento", formData.procedimiento);
      if (formData.imagen instanceof File) {
        form.append("imagen", formData.imagen);
      }

      const res = await fetch("http://localhost:8000/recetas", {
        method: "POST",
        body: form,
      });
      if (!res.ok) throw new Error("Error al guardar la receta");

      const newReceta = await res.json();
      onRecipeAdded(newReceta.receta);
      setFormData({
        receta: "",
        ingredientes: [],
        procedimiento: "",
        imagen: undefined,
      });
      setSelectedIngredients([]);
    } catch (error) {
      console.error(error);
      setError("Error al guardar la receta");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={className}>
      <h1 className="text-2xl font-bold mb-4">Formulario de Recetas</h1>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block mb-1 font-semibold">Imagen</label>
          <ImageDropdown
            image={formData.imagen instanceof File ? formData.imagen : null}
            setImage={(file) =>
              handleInputChange("imagen", file || undefined)
            }
          />
        </div>

        <div>
          <label className="block mb-1 font-semibold">
            Nombre de la Receta
          </label>
          <input
            type="text"
            value={formData.receta}
            onChange={(e) => handleInputChange("receta", e.target.value)}
            required
            className="border p-1 rounded w-full"
          />
        </div>

        <IngredientSelector
          selectedIngredients={selectedIngredients}
          onIngredientsChange={handleIngredientsChange}
        />

        <div>
          <label className="block mb-1 font-semibold">Procedimiento</label>
          <textarea
            value={formData.procedimiento}
            onChange={(e) => handleInputChange("procedimiento", e.target.value)}
            required
            className="border p-2 rounded w-full min-h-[100px]"
          />
        </div>

        {error && <p className="text-red-600">{error}</p>}

        <button
          type="submit"
          disabled={loading}
          className={`px-4 py-2 rounded text-white ${
            loading
              ? "bg-gray-400 cursor-not-allowed"
              : "bg-blue-600 hover:bg-blue-700"
          }`}
        >
          {loading ? "Guardando..." : "Agregar Receta"}
        </button>
      </form>
    </div>
  );
}
