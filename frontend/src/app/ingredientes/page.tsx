"use client";
import { useState, useEffect } from "react";
import IngredientesList from "@/components/IngredientesList";
import IngredienteForm from "@/components/IngredienteForm";

interface Ingrediente {
  nombre: string;
  imagen: string;
}

const API_URL = "http://localhost:8000";

export default function IngredientesPage() {
  const [ingredientes, setIngredientes] = useState<Ingrediente[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchIngredientes();
  }, []);

  const fetchIngredientes = async () => {
    try {
      const response = await fetch(`${API_URL}/ingredientes`);
      const data = await response.json();
      setIngredientes(data);
    } catch (error) {
      console.error("Error obteniendo los ingredientes:", error);
    }
  };

  const handleFormSubmit = async (formData: FormData) => {
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/ingredientes`, {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        fetchIngredientes();
      }
    } catch (error) {
      console.error("Error al crear ingrediente:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col lg:flex-row p-4 gap-4 items-start w-full justify-center">
      <IngredienteForm
        onSubmit={handleFormSubmit}
        loading={loading}
        className="p-4 w-full lg:flex-[0.3]"
      />
      <IngredientesList
        ingredientes={ingredientes}
        apiUrl={API_URL}
        className="p-4 w-full lg:flex-[0.7]"
      />
    </div>
  );
}
