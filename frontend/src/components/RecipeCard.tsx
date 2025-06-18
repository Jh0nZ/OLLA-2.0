"use client"

import React from "react"
import type { Receta } from '@/app/creador/page'

interface RecipeCardProps {
  receta: Receta
  onDelete: () => void
}

export default function RecipeCard({ receta, onDelete }: RecipeCardProps) {
  return (
    <div className="bg-white rounded-lg shadow-lg p-4 mb-4 hover:shadow-xl transition-shadow border">
      {receta.imagen && typeof receta.imagen === "string" && (
        <div className="mb-3">
          <img
            src={`http://127.0.0.1:8000/datos/images/${receta.imagen}`}
            alt={receta.receta}
            className="w-full h-32 object-cover rounded-lg"
          />
        </div>
      )}
      <div className="space-y-3">
        <div className="flex justify-between items-center">
          <h3 className="text-lg font-bold text-gray-800 border-b border-gray-200 pb-2">
            {receta.receta}
          </h3>
          <button
            onClick={onDelete}
            className="text-red-500 hover:text-red-700 font-bold text-xl"
            aria-label="Eliminar receta"
          >
            ×
          </button>
        </div>
        
        <div className="bg-gray-50 p-3 rounded-lg">
          <p className="font-semibold text-gray-700 mb-2">Ingredientes:</p>
          <div className="grid grid-cols-3 gap-1 text-xs">
            {Array.isArray(receta.ingredientes) && receta.ingredientes.map((ingrediente, idx) => (
              <span key={idx} className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-center">
                {ingrediente}
              </span>
            ))}
          </div>
        </div>
        
        <div className="bg-gray-50 p-3 rounded-lg">
          <p className="font-semibold text-gray-700 mb-2">Procedimiento:</p>
          <p className="text-gray-600 text-sm leading-relaxed">{receta.procedimiento}</p>
        </div>
      </div>
    </div>
  )
}