import { ChefHat } from "lucide-react"
import IngredientSelector from "@/components/IngredientSelector"
import type { Ingredient } from "@/app/page"

interface IngredientSelectorCardProps {
  selectedIngredients: Ingredient[]
  onIngredientsChange: (ingredients: Ingredient[]) => void
}

export default function IngredientSelectorCard({
  selectedIngredients,
  onIngredientsChange,
}: IngredientSelectorCardProps) {
  return (
    <div className="bg-white/95 backdrop-blur-sm rounded-xl p-4 shadow-lg border border-emerald-200/50">
      <h3 className="text-sm font-semibold text-emerald-800 mb-3 flex items-center gap-2">
        <ChefHat className="h-4 w-4" />
        Selecciona tus ingredientes
      </h3>
      <IngredientSelector selectedIngredients={selectedIngredients} onIngredientsChange={onIngredientsChange} />
    </div>
  )
}
