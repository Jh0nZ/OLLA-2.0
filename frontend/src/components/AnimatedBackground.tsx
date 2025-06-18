"use client"

import type React from "react"
import { ChefHat, UtensilsCrossed, Coffee, Cookie, Pizza, Soup, Cake, Apple } from "lucide-react"

// Componente para iconos flotantes animados
const FloatingIcon = ({
  Icon,
  className,
  delay = 0,
}: {
  Icon: React.ComponentType<any>
  className: string
  delay?: number
}) => (
  <div
    className={`absolute opacity-10 ${className}`}
    style={{
      animation: `float 6s ease-in-out infinite`,
      animationDelay: `${delay}s`,
    }}
  >
    <Icon size={24} />
  </div>
)

export default function AnimatedBackground() {
  return (
    <>
      {/* Estilos CSS para animaciones */}
      <style jsx>{`
        @keyframes float {
          0%, 100% { transform: translateY(0px) rotate(0deg); }
          25% { transform: translateY(-20px) rotate(5deg); }
          50% { transform: translateY(-10px) rotate(-5deg); }
          75% { transform: translateY(-15px) rotate(3deg); }
        }
        
        @keyframes drift {
          0% { transform: translateX(0px); }
          50% { transform: translateX(30px); }
          100% { transform: translateX(0px); }
        }
        
        @keyframes pulse-slow {
          0%, 100% { opacity: 0.05; }
          50% { opacity: 0.15; }
        }
        
        .drift-animation {
          animation: drift 8s ease-in-out infinite;
        }
        
        .pulse-slow {
          animation: pulse-slow 4s ease-in-out infinite;
        }
      `}</style>

      <div className="absolute inset-0 bg-gradient-to-br from-emerald-800 via-emerald-700 to-green-800">
        {/* Iconos flotantes animados */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          {/* Primera capa de iconos */}
          <FloatingIcon Icon={ChefHat} className="top-10 left-10 text-emerald-300" delay={0} />
          <FloatingIcon Icon={UtensilsCrossed} className="top-20 right-20 text-emerald-200" delay={1} />
          <FloatingIcon Icon={Coffee} className="top-40 left-1/4 text-emerald-400" delay={2} />
          <FloatingIcon Icon={Pizza} className="top-60 right-1/3 text-emerald-300" delay={0.5} />
          <FloatingIcon Icon={Cookie} className="bottom-40 left-20 text-emerald-200" delay={1.5} />
          <FloatingIcon Icon={Soup} className="bottom-60 right-10 text-emerald-400" delay={2.5} />
          <FloatingIcon Icon={Cake} className="top-1/2 left-10 text-emerald-300" delay={3} />
          <FloatingIcon Icon={Apple} className="bottom-20 right-1/4 text-emerald-200" delay={1.8} />

          {/* Segunda capa con animación drift */}
          <div className="drift-animation">
            <FloatingIcon Icon={ChefHat} className="top-32 right-1/4 text-emerald-200" delay={2.2} />
            <FloatingIcon Icon={UtensilsCrossed} className="bottom-32 left-1/3 text-emerald-300" delay={0.8} />
            <FloatingIcon Icon={Coffee} className="top-3/4 right-20 text-emerald-400" delay={1.2} />
          </div>

          {/* Tercera capa con pulse lento */}
          <div className="pulse-slow">
            <div className="absolute top-1/4 left-1/2 transform -translate-x-1/2">
              <Pizza size={32} className="text-emerald-200" />
            </div>
            <div className="absolute bottom-1/4 left-1/4">
              <Soup size={28} className="text-emerald-300" />
            </div>
            <div className="absolute top-1/3 right-1/4">
              <Cookie size={30} className="text-emerald-400" />
            </div>
          </div>

          {/* Elementos decorativos adicionales */}
          <div className="absolute top-0 left-0 w-full h-full">
            <div
              className="absolute top-16 left-1/2 w-2 h-2 bg-emerald-300 rounded-full opacity-20 animate-ping"
              style={{ animationDelay: "1s" }}
            ></div>
            <div
              className="absolute bottom-32 right-1/3 w-3 h-3 bg-emerald-200 rounded-full opacity-15 animate-ping"
              style={{ animationDelay: "2s" }}
            ></div>
            <div
              className="absolute top-1/2 left-1/4 w-1 h-1 bg-emerald-400 rounded-full opacity-25 animate-ping"
              style={{ animationDelay: "0.5s" }}
            ></div>
          </div>
        </div>
      </div>
    </>
  )
}
