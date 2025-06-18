"use client";

import React, { useEffect, useState, useCallback } from "react";
import RecipeForm from "@/components/RecipeForm";
import RecipeList from "@/components/RecipeList";

export interface Receta {
  ingredientes: string[];
  receta: string;
  procedimiento: string;
  imagen?: string | File;
}

export interface Ingredient {
  nombre: string;
  imagen: string;
}

const API_BASE_URL = "http://localhost:8000";

export default function CreadorPage() {
  const [recetas, setRecetas] = useState<Receta[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchRecetas = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(`${API_BASE_URL}/recetas`);
      if (!response.ok) throw new Error("Error fetching recetas");

      const data = await response.json();
      setRecetas(data);
    } catch (err) {
      console.error("Fetch error:", err);
      setError("Error al cargar recetas");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchRecetas();
  }, [fetchRecetas]);

  const handleRecipeAdded = useCallback((newReceta: Receta) => {
    setRecetas((prev) => [...prev, newReceta]);
  }, []);

  const handleRecipeDeleted = useCallback((index: number) => {
    setRecetas((prev) => prev.filter((_, i) => i !== index));
  }, []);

  if (loading) {
    return (
      <div className="container mx-auto p-4 md:p-8">
        <div className="text-center">Cargando...</div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-4 md:p-8">
      {error && (
        <div className="text-red-600 text-center mb-4 p-4 bg-red-50 rounded">
          {error}
          <button
            onClick={fetchRecetas}
            className="ml-4 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
          >
            Reintentar
          </button>
        </div>
      )}
      <div className="flex flex-col lg:flex-row gap-4 md:gap-8 p-2 md:p-4 m-2 md:m-8 justify-center items-start">
        <RecipeForm
          onRecipeAdded={handleRecipeAdded}
          className="w-full lg:w-[30%]"
        />
        <RecipeList
          recetas={recetas}
          onRecipeDeleted={handleRecipeDeleted}
          className="w-full lg:w-[70%]"
        />
      </div>
    </div>
  );
}
