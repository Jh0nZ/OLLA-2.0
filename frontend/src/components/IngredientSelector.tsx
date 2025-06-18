"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { X, Plus } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface Ingredient {
  nombre: string;
  imagen: string;
}

interface Props {
  selectedIngredients: Ingredient[];
  onIngredientsChange: (ingredients: Ingredient[]) => void;
  apiUrl?: string;
}

export default function IngredientSelector({
  selectedIngredients,
  onIngredientsChange,
  apiUrl = "http://localhost:8000",
}: Props) {
  const [search, setSearch] = useState("");
  const [ingredientes, setIngredientes] = useState<Ingredient[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchIngredientes = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${apiUrl}/ingredientes`);
      if (response.ok) {
        const data = await response.json();
        setIngredientes(data);
      }
    } catch (error) {
      console.error("Error fetching ingredientes:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchIngredientes();
  }, [apiUrl]);

  const add = (nombre: string) => {
    const ingredient = ingredientes.find((i) => i.nombre === nombre);
    if (ingredient && !selectedIngredients.find((i) => i.nombre === nombre)) {
      onIngredientsChange([...selectedIngredients, ingredient]);
    }
  };

  const remove = (nombre: string) => {
    onIngredientsChange(selectedIngredients.filter((i) => i.nombre !== nombre));
  };

  const filtered = ingredientes.filter(
    (i) =>
      !selectedIngredients.find((s) => s.nombre === i.nombre) &&
      i.nombre.toLowerCase().includes(search.toLowerCase())
  );

  if (ingredientes.length === 0 && !loading) {
    return (
      <div className="p-4 border text-center space-y-2">
        <p className="text-sm text-gray-500">No hay ingredientes</p>
        <Button
          size="sm"
          onClick={() => window.open("/ingredientes", "_blank")}
        >
          <Plus className="w-4 h-4 mr-1" />
          Crear ingredientes
        </Button>
      </div>
    );
  }

  return (
    <div className="">
      {selectedIngredients.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {selectedIngredients.map((i) => (
            <div
              key={i.nombre}
              className="mb-2 flex items-center gap-1 px-2 py-1 bg-gradient-to-r from-emerald-600 to-emerald-700 text-white shadow-lg shadow-emerald-600/25 text-xs rounded-md"
            >
              {i.nombre}
              <X
                className="w-3 h-3 cursor-pointer"
                onClick={() => remove(i.nombre)}
              />
            </div>
          ))}
        </div>
      )}
      <Select
        onValueChange={add}
        value=""
        onOpenChange={(open) => open && fetchIngredientes()}
      >
        <SelectTrigger>
          <SelectValue
            placeholder={loading ? "Cargando..." : "Agregar ingrediente"}
          />
        </SelectTrigger>
        <SelectContent>
          <div className="p-2">
            <Input
              placeholder="Buscar..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              onKeyDown={(e) => e.stopPropagation()}
              className="h-8"
              autoFocus
            />
          </div>
          <div className="max-h-32 overflow-y-auto">
            {loading ? (
              <div className="p-2 text-center text-xs text-gray-500">
                Cargando...
              </div>
            ) : filtered.length > 0 ? (
              filtered.map((i) => (
                <SelectItem key={i.nombre} value={i.nombre}>
                  {i.nombre}
                </SelectItem>
              ))
            ) : (
              <div className="p-2 text-center">
                <p className="text-xs text-gray-500 mb-1">Sin resultados</p>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => window.open("/ingredientes", "_blank")}
                >
                  <Plus className="w-3 h-3" />
                  Nuevo
                </Button>
              </div>
            )}
          </div>
        </SelectContent>
      </Select>
    </div>
  );
}
