"use client"

import type React from "react"

import { useState } from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { ChefHat, Menu, X, Plus, Home, Database, Apple, TestTube } from "lucide-react"

const Navbar: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false)
  const pathname = usePathname()
  const isActive = (href: string) => pathname === href

  const toggleMenu = () => setIsOpen(!isOpen)

  const navLinks = [
    { href: "/", label: "Inicio", icon: Home },
    { href: "/creador", label: "Crear dataset", icon: Database },
    { href: "/ingredientes", label: "Ingredientes", icon: Apple },
    { href: "/test", label: "Test", icon: TestTube },
  ]

  const actionButtons = [
    {
      href: "/generador",
      label: "Crear una receta",
      icon: Plus,
      className:
        "text-white bg-gradient-to-r from-emerald-600 to-emerald-700 hover:from-emerald-700 hover:to-emerald-800 shadow-lg shadow-emerald-600/25",
    },
  ]

  return (
    <nav className="sticky top-0 z-50 bg-white/90 backdrop-blur-md border-b border-emerald-200/50 shadow-lg">
      <div className="max-w-screen-xl flex flex-wrap items-center justify-between mx-auto p-4">
        {/* Logo */}
        <Link href="/" className="flex items-center space-x-3 rtl:space-x-reverse group">
          <div className="relative">
            <img
              src="/images/recursos/logo.png"
              className="h-8 transition-transform group-hover:scale-105"
              alt="Olla 2.0 Logo"
            />
            <ChefHat className="absolute -top-1 -right-1 h-4 w-4 text-emerald-600 opacity-0 group-hover:opacity-100 transition-opacity" />
          </div>
          <span className="self-center text-2xl font-bold bg-gradient-to-r from-emerald-700 to-green-600 bg-clip-text text-transparent">
            Olla 2.0
          </span>
        </Link>

        <div className="flex md:order-2 space-x-3 md:space-x-0 rtl:space-x-reverse">
          <div className="hidden md:flex">
            {actionButtons.map((button, index) => (
              <Link key={index} href={button.href}>
                <button
                  type="button"
                  className={`${button.className} focus:ring-4 focus:outline-none focus:ring-emerald-300 font-medium rounded-lg text-sm px-4 py-2 text-center transition-all duration-200 flex items-center gap-2`}
                >
                  <button.icon className="h-4 w-4" />
                  {button.label}
                </button>
              </Link>
            ))}
          </div>

          {/* Mobile menu button */}
          <button
            onClick={toggleMenu}
            type="button"
            className="inline-flex items-center p-2 w-10 h-10 justify-center text-emerald-600 rounded-lg md:hidden hover:bg-emerald-50 focus:outline-none focus:ring-2 focus:ring-emerald-200 transition-colors duration-200"
            aria-controls="navbar-sticky"
            aria-expanded={isOpen}
            aria-label="Toggle navigation menu"
          >
            {isOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        </div>

        {/* Navigation menu */}
        <div
          className={`items-center justify-between w-full md:flex md:w-auto md:order-1 transition-all duration-300 ${
            isOpen ? "block" : "hidden"
          }`}
          id="navbar-sticky"
        >
          <ul className="flex flex-col p-4 md:p-0 mt-4 font-medium border border-emerald-200/50 rounded-lg bg-white/80 backdrop-blur-sm md:space-x-8 rtl:space-x-reverse md:flex-row md:mt-0 md:border-0 md:bg-transparent shadow-lg md:shadow-none">
            {navLinks.map((link, index) => (
              <li key={index}>
                <Link
                  href={link.href}
                  className={`flex items-center gap-2 py-2 px-3 rounded-lg transition-all duration-200 ${
                    isActive(link.href)
                      ? "text-white bg-gradient-to-r from-emerald-600 to-emerald-700 shadow-md rounded-lg md:rounded-full md:bg-gradient-to-r md:from-emerald-600 md:to-emerald-700 md:text-white"
                      : "text-emerald-800 hover:bg-emerald-50 md:hover:bg-transparent md:hover:text-emerald-600 md:hover:border-b-2 md:hover:border-emerald-300 md:rounded-none md:pb-1 md:border-b-2 md:border-transparent"
                  }`}
                  {...(isActive(link.href) && { "aria-current": "page" })}
                >
                  <link.icon className="h-4 w-4" />
                  {link.label}
                </Link>
              </li>
            ))}
          </ul>
          <div className="md:hidden mt-4 space-y-2 px-4">
            {actionButtons.map((button, index) => (
              <Link key={index} href={button.href} className="block">
                <button
                  type="button"
                  className={`${button.className} focus:ring-4 focus:outline-none focus:ring-emerald-300 font-medium rounded-lg text-sm px-4 py-2 text-center w-full transition-all duration-200 flex items-center justify-center gap-2`}
                >   
                  <button.icon className="h-4 w-4" />
                  {button.label}
                </button>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </nav>
  )
}

export default Navbar
