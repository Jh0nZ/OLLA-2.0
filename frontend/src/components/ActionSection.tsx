"use client"

import { Send, ChefHat, Apple, Pizza, Thermometer, FileText, Plus, Minus } from "lucide-react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import IngredientSelector from "@/components/IngredientSelector"
import type { Ingredient } from "@/app/page"

interface ActionSectionProps {
  selectedIngredients: Ingredient[]
  selectedModel: string
  temperature: number
  maxLength: number
  isLoading: boolean
  aiModels: Array<{ value: string; label: string }>
  onIngredientsChange: (ingredients: Ingredient[]) => void
  onGenerateRecipe: () => void
  onModelChange: (model: string) => void
  onTemperatureChange: (temperature: number) => void
  onMaxLengthChange: (maxLength: number) => void
}

export default function ActionSection({
  selectedIngredients,
  selectedModel,
  temperature,
  maxLength,
  isLoading,
  aiModels,
  onIngredientsChange,
  onGenerateRecipe,
  onModelChange,
  onTemperatureChange,
  onMaxLengthChange,
}: ActionSectionProps) {
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false)

  const handleMaxLengthChange = (value: string) => {
    const numValue = Number.parseInt(value) || 0
    if (numValue >= 50 && numValue <= 2000) {
      onMaxLengthChange(numValue)
    }
  }

  return (
    <div className="bg-white/95 backdrop-blur-sm rounded-xl shadow-lg border border-emerald-200/50 p-6 space-y-6">
      <div>
        <h3 className="text-sm font-semibold text-emerald-800 mb-4 flex items-center gap-2">
          <ChefHat className="h-4 w-4" />
          Selecciona tus ingredientes
        </h3>
        <IngredientSelector selectedIngredients={selectedIngredients} onIngredientsChange={onIngredientsChange} />
      </div>
      <div className="flex justify-center">
        <button
          type="button"
          onClick={() => setShowAdvancedSettings(!showAdvancedSettings)}
          className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-emerald-700 bg-emerald-50 hover:bg-emerald-100 rounded-lg border border-emerald-200 transition-colors"
        >
          {showAdvancedSettings ? (
            <>
              <Minus className="h-4 w-4" />
              Ocultar configuración avanzada
            </>
          ) : (
            <>
              <Plus className="h-4 w-4" />
              Mostrar configuración avanzada
            </>
          )}
        </button>
      </div>
      {showAdvancedSettings && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-in slide-in-from-top-2 duration-300">
          <div>
            <h3 className="text-sm font-semibold text-emerald-800 mb-4 flex items-center gap-2">
              <Thermometer className="h-4 w-4" />
              Nivel de creatividad
            </h3>
            <div className="bg-emerald-50/50 rounded-lg p-4 border border-emerald-100">
              <div className="flex items-center gap-4">
                <label className="text-sm font-medium text-emerald-700 whitespace-nowrap">Creatividad:</label>
                <div className="flex-1 flex items-center gap-3">
                  <span className="text-xs text-emerald-600">Conservador</span>
                  <div className="flex-1 relative">
                    <input
                      type="range"
                      min="0.1"
                      max="2.0"
                      step="0.1"
                      value={temperature}
                      onChange={(e) => onTemperatureChange(Number(e.target.value))}
                      className="w-full h-2 bg-emerald-200 rounded-lg appearance-none cursor-pointer slider focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-opacity-50"
                      style={{
                        background: `linear-gradient(to right, #10b981 0%, #10b981 ${
                          ((temperature - 0.1) / (2.0 - 0.1)) * 100
                        }%, #d1fae5 ${((temperature - 0.1) / (2.0 - 0.1)) * 100}%, #d1fae5 100%)`,
                      }}
                    />
                    <style jsx>{`
                      .slider::-webkit-slider-thumb {
                        appearance: none;
                        height: 20px;
                        width: 20px;
                        border-radius: 50%;
                        background: #059669;
                        cursor: pointer;
                        border: 2px solid #ffffff;
                        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                      }
                      .slider::-moz-range-thumb {
                        height: 20px;
                        width: 20px;
                        border-radius: 50%;
                        background: #059669;
                        cursor: pointer;
                        border: 2px solid #ffffff;
                        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                      }
                    `}</style>
                  </div>
                  <span className="text-xs text-emerald-600">Creativo</span>
                  <span className="text-sm font-medium text-emerald-700 min-w-[2.5rem] bg-emerald-100 px-2 py-1 rounded-md">
                    {temperature.toFixed(1)}
                  </span>
                </div>
              </div>
              <p className="text-xs text-emerald-600 mt-2 text-center">
                {temperature < 0.5
                  ? "🎯 Recetas tradicionales y precisas"
                  : temperature < 1.0
                    ? "⚖️ Balance entre tradición e innovación"
                    : temperature < 1.5
                      ? "🎨 Recetas creativas y originales"
                      : "🚀 Máxima creatividad culinaria"}
              </p>
            </div>
          </div>

          <div>
            <h3 className="text-sm font-semibold text-emerald-800 mb-4 flex items-center gap-2">
              <FileText className="h-4 w-4" />
              Longitud de la receta
            </h3>
            <div className="bg-emerald-50/50 rounded-lg p-4 border border-emerald-100">
              <div className="space-y-3">
                <div className="flex items-center gap-3">
                  <label className="text-sm font-medium text-emerald-700 whitespace-nowrap">Caracteres máximos:</label>
                  <div className="flex-1 flex items-center gap-2">
                    <Input
                      type="number"
                      min="50"
                      max="2000"
                      value={maxLength}
                      onChange={(e) => handleMaxLengthChange(e.target.value)}
                      className="flex-1 border-emerald-300 focus:border-emerald-500 focus:ring-emerald-500 bg-white/90"
                      placeholder="256"
                    />
                    <span className="text-sm font-medium text-emerald-700 bg-emerald-100 px-2 py-1 rounded-md min-w-[3rem] text-center">
                      {maxLength}
                    </span>
                  </div>
                </div>
                <div className="flex gap-2 flex-wrap">
                  <button
                    type="button"
                    onClick={() => onMaxLengthChange(128)}
                    className={`px-3 py-1 text-xs rounded-md transition-colors ${
                      maxLength === 128
                        ? "bg-emerald-500 text-white"
                        : "bg-emerald-100 text-emerald-700 hover:bg-emerald-200"
                    }`}
                  >
                    Corta (128)
                  </button>
                  <button
                    type="button"
                    onClick={() => onMaxLengthChange(256)}
                    className={`px-3 py-1 text-xs rounded-md transition-colors ${
                      maxLength === 256
                        ? "bg-emerald-500 text-white"
                        : "bg-emerald-100 text-emerald-700 hover:bg-emerald-200"
                    }`}
                  >
                    Media (256)
                  </button>
                  <button
                    type="button"
                    onClick={() => onMaxLengthChange(512)}
                    className={`px-3 py-1 text-xs rounded-md transition-colors ${
                      maxLength === 512
                        ? "bg-emerald-500 text-white"
                        : "bg-emerald-100 text-emerald-700 hover:bg-emerald-200"
                    }`}
                  >
                    Larga (512)
                  </button>
                </div>
              </div>

              <p className="text-xs text-emerald-600 mt-2 text-center">
                {maxLength <= 150
                  ? "📝 Receta básica y concisa"
                  : maxLength <= 300
                    ? "📋 Receta con detalles esenciales"
                    : maxLength <= 600
                      ? "📖 Receta completa con explicaciones"
                      : "📚 Receta muy detallada con tips y variaciones"}
              </p>
            </div>
          </div>
        </div>
      )}
      <div className="border-t border-emerald-100"></div>
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
