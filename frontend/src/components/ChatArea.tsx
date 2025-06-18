import { Coffee, Soup } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import type { Message } from "@/app/page"

interface ChatAreaProps {
  messages: Message[]
  isLoading: boolean
}

export default function ChatArea({ messages, isLoading }: ChatAreaProps) {
  return (
    <div className="flex-1 bg-white/95 backdrop-blur-sm rounded-xl shadow-lg border border-emerald-200/50 flex flex-col overflow-hidden">
      <div className="p-4 border-b border-emerald-100 bg-emerald-50/50">
        <h2 className="font-semibold text-emerald-800 text-sm flex items-center gap-2">
          <Coffee className="h-4 w-4" />
          Conversación
        </h2>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${
              message.type === "user" ? "justify-end" : "justify-start"
            } animate-in slide-in-from-bottom-2 duration-300`}
          >
            <div
              className={`max-w-[80%] px-4 py-3 rounded-2xl transition-all duration-200 ${
                message.type === "user"
                  ? "bg-gradient-to-r from-emerald-600 to-emerald-700 text-white shadow-lg shadow-emerald-600/25"
                  : "bg-emerald-50 border border-emerald-200 text-emerald-900 shadow-sm"
              }`}
            >
                            {typeof message.content === "string" ? (
                <p className="text-sm leading-relaxed whitespace-pre-line">
                  {message.content}
                </p>
              ) : (
                <div className="text-sm leading-relaxed whitespace-pre-line space-y-2">
                  <h3 className="font-semibold text-emerald-800">
                    {message.content.nombre}
                  </h3>
                  <p>{message.content.procedimiento}</p>
                  <p>{message.content.texto_completo}</p>
                </div>
              )}
              {message.ingredients && message.ingredients.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-2">
                  {message.ingredients.map((ingredient, index) => (
                    <Badge
                      key={index}
                      variant="secondary"
                      className={`text-xs ${
                        message.type === "user"
                          ? "bg-white/20 text-white border-white/30"
                          : "bg-emerald-100 text-emerald-700 border-emerald-200"
                      }`}
                    >
                      {ingredient.nombre}
                    </Badge>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start animate-in slide-in-from-bottom-2 duration-300">
            <div className="bg-emerald-50 border border-emerald-200 rounded-2xl px-4 py-3 shadow-sm">
              <div className="flex items-center space-x-2">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-emerald-500 rounded-full animate-bounce"></div>
                  <div
                    className="w-2 h-2 bg-emerald-500 rounded-full animate-bounce"
                    style={{ animationDelay: "0.1s" }}
                  ></div>
                  <div
                    className="w-2 h-2 bg-emerald-500 rounded-full animate-bounce"
                    style={{ animationDelay: "0.2s" }}
                  ></div>
                </div>
                <span className="text-sm text-emerald-700">Generando receta...</span>
                <Soup className="h-4 w-4 text-emerald-500 animate-pulse" />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
