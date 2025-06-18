"use client"

import { Send, ChefHat, Apple, Pizza } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import IngredientSelector from "@/components/IngredientSelector"
import type { Ingredient } from "@/app/page"

interface ActionSectionProps {
  selectedIngredients: Ingredient[]
  selectedModel: string
  isLoading: boolean
  aiModels: Array<{ value: string; label: string }>
  onIngredientsChange: (ingredients: Ingredient[]) => void
  onGenerateRecipe: () => void
  onModelChange: (model: string) => void
}

export default function ActionSection({
  selectedIngredients,
  selectedModel,
  isLoading,
  aiModels,
  onIngredientsChange,
  onGenerateRecipe,
  onModelChange,
}: ActionSectionProps) {
  return (
    <div className="bg-white/95 backdrop-blur-sm rounded-xl shadow-lg border border-emerald-200/50 p-6 space-y-6">
      {/* Selector de Ingredientes */}
      <div>
        <h3 className="text-sm font-semibold text-emerald-800 mb-4 flex items-center gap-2">
          <ChefHat className="h-4 w-4" />
          Selecciona tus ingredientes
        </h3>
        <IngredientSelector selectedIngredients={selectedIngredients} onIngredientsChange={onIngredientsChange} />
      </div>

      {/* Separador visual */}
      <div className="border-t border-emerald-100"></div>

      {/* Barra de Acciones */}
      <div className="flex gap-3 items-end">
        <div className="flex-1">
          <div className="flex gap-2">
            <Button
              onClick={onGenerateRecipe}
              disabled={selectedIngredients.length === 0 || isLoading}
              className="flex-1 bg-gradient-to-r from-emerald-600 to-emerald-700 hover:from-emerald-700 hover:to-emerald-800 text-white shadow-lg shadow-emerald-600/25 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed border-0"
              size="lg"
            >
              <Send className="h-4 w-4 mr-2" />
              {isLoading ? "Generando..." : `Generar Receta (${selectedIngredients.length})`}
              <ChefHat className="h-4 w-4 ml-2" />
            </Button>

            <Select value={selectedModel} onValueChange={onModelChange}>
              <SelectTrigger className="w-32 border-emerald-300 focus:border-emerald-500 focus:ring-emerald-500 bg-white/90">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {aiModels.map((model) => (
                  <SelectItem key={model.value} value={model.value}>
                    {model.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {selectedIngredients.length === 0 && (
            <p className="text-xs text-emerald-600 mt-2 text-center flex items-center justify-center gap-1">
              <Apple className="h-3 w-3" />
              Selecciona al menos un ingrediente para generar una receta
              <Pizza className="h-3 w-3" />
            </p>
          )}
        </div>
      </div>
    </div>
  )
}
