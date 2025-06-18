"use client"

import React from "react"
import type { Ingredient } from '@/app/creador/page'

interface IngredientModalProps {
  ingredientes: Ingredient[]
  searchTerm: string
  setSearchTerm: React.Dispatch<React.SetStateAction<string>>
  addIngredient: (ingredient: Ingredient) => void
  onClose: () => void
}

export default function IngredientModal({
  ingredientes,
  searchTerm,
  setSearchTerm,
  addIngredient,
  onClose,
}: IngredientModalProps) {
  const filteredIngredients = ingredientes.filter(i =>
    i.nombre.toLowerCase().includes(searchTerm.toLowerCase())
  )

  return (
    <div className="modal-backdrop p-4 bg-white border rounded shadow max-w-md">
      <input
        type="text"
        placeholder="Buscar ingredientes..."
        value={searchTerm}
        onChange={e => setSearchTerm(e.target.value)}
        className="w-full mb-2 p-1 border rounded"
      />
      <div style={{ maxHeight: 200, overflowY: "auto" }}>
        {filteredIngredients.length === 0 ? (
          <p>No se encontraron ingredientes</p>
        ) : (
          filteredIngredients.map(i => (
            <div
              key={i.nombre}
              onClick={() => addIngredient(i)}
              style={{ cursor: "pointer" }}
              className="p-1 hover:bg-gray-200 rounded"
            >
              {i.nombre}
            </div>
          ))
        )}
      </div>
      <button
        type="button"
        onClick={onClose}
        className="mt-2 px-3 py-1 bg-red-500 text-white rounded"
      >
        Cerrar
      </button>
    </div>
  )
}