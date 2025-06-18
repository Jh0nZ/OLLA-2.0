"use client"

import { useState, useEffect } from "react"
import AnimatedBackground from "@/components/AnimatedBackground"
import Header from "@/components/Header"
import ChatArea from "@/components/ChatArea"
import ActionSection from "@/components/ActionSection"

export interface Ingredient {
  nombre: string
  imagen: string
}

export interface Message {
  id: number
  type: "user" | "assistant"
  content: string
  ingredients?: Ingredient[]
}

export default function RecipeChat() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      type: "assistant",
      content: "¡Hola! Selecciona los ingredientes que tienes disponibles y te ayudo a crear una receta deliciosa.",
    },
  ])
  const [aiModels, setAiModels] = useState<{ value: string; label: string }[]>([])
  const [selectedIngredients, setSelectedIngredients] = useState<Ingredient[]>([])
  const [selectedModel, setSelectedModel] = useState<string>("")
  const [temperature, setTemperature] = useState<number>(0.96)
  const [maxLength, setMaxLength] = useState<number>(256)
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    fetch("http://localhost:8000/models")
      .then((res) => res.json())
      .then((data: string[]) => {
        const formatted = data.map((model) => ({
          value: model,
          label: model.charAt(0).toUpperCase() + model.slice(1),
        }))
        setAiModels(formatted)
        if (formatted.length > 0) setSelectedModel(formatted[0].value)
      })
      .catch((err) => console.error("Error al cargar modelos:", err))
  }, [])

  const generateRecipe = async () => {
    if (!selectedIngredients.length) return

    const userMessage: Message = {
      id: Date.now(),
      type: "user",
      content: "Quiero una receta con los siguientes ingredientes: ",
      ingredients: selectedIngredients,
    }

    setMessages((prev) => [...prev, userMessage])
    setIsLoading(true)

    try {
      const formData = new URLSearchParams({
        ingredientes: selectedIngredients.map((i) => i.nombre).join(", "),
        modelo: selectedModel,
        temperatura: temperature.toString(),
        max_length: maxLength.toString(),
      })

      const response = await fetch("http://localhost:8000/generar-receta", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: formData,
      })

      const data = await response.json()

      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          type: "assistant",
          content: data.receta || "No se pudo generar la receta. Intenta de nuevo.",
        },
      ])
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          type: "assistant",
          content: "Error al generar la receta. Verifica la conexión con el servidor.",
        },
      ])
    } finally {
      setIsLoading(false)
      setSelectedIngredients([])
    }
  }

  return (
    <div className="min-h-screen relative overflow-hidden">
      <AnimatedBackground />
      <div className="container mx-auto max-w-4xl h-screen flex flex-col relative z-10">
        <div className="flex-1 flex flex-col p-6 gap-6 overflow-hidden">
          <ChatArea messages={messages} isLoading={isLoading} />
          <ActionSection
            selectedIngredients={selectedIngredients}
            selectedModel={selectedModel}
            temperature={temperature}
            maxLength={maxLength}
            isLoading={isLoading}
            aiModels={aiModels}
            onIngredientsChange={setSelectedIngredients}
            onGenerateRecipe={generateRecipe}
            onModelChange={setSelectedModel}
            onTemperatureChange={setTemperature}
            onMaxLengthChange={setMaxLength}
          />
        </div>
      </div>
    </div>
  )
}
