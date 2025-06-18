"use client"

import React from "react"
import RecipeCard from './RecipeCard'
import type { Receta } from '@/app/creador/page'

interface RecipeListProps {
    recetas: Receta[]
    onRecipeDeleted: (index: number) => void,
    className?: string
}

export default function RecipeList({ recetas, onRecipeDeleted, className }: RecipeListProps) {
    return (
        <div className={className}>
            <h2 className="text-xl font-semibold mb-4">
                Recetas Guardadas ({recetas.length})
            </h2>
            <div className="overflow-y-auto max-h-184">
                {recetas.length === 0 ? (
                    <p>No hay recetas guardadas</p>
                ) : (
                    recetas.map((receta, index) => (
                        <RecipeCard
                            key={index}
                            receta={receta}
                            onDelete={() => onRecipeDeleted(index)}
                        />
                    ))
                )}
            </div>
        </div>
    )
}