"use client";
import React, { useCallback, useEffect } from "react";
import { useDropzone } from "react-dropzone";

interface PropiedadesImagenDropdown {
  image: File | null;
  setImage: (archivo: File | null) => void;
}

const ImagenDropdown: React.FC<PropiedadesImagenDropdown> = ({
  image,
  setImage,
}) => {
  const manejarCambioImagen = useCallback(
    (archivo: File | null) => {
      setImage(archivo);
    },
    [setImage]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: (archivos) => {
      const archivoImagen = archivos.find((f) => f.type.startsWith("image/"));
      if (archivoImagen) manejarCambioImagen(archivoImagen);
    },
    accept: { "image/*": [] },
    maxFiles: 1,
  });

  useEffect(() => {
    const manejarPegar = (e: ClipboardEvent) => {
      const elementoImagen = Array.from(e.clipboardData?.items || []).find(
        (item) => item.type.startsWith("image/")
      );

      if (elementoImagen) {
        const archivo = elementoImagen.getAsFile();
        if (archivo) {
          e.preventDefault();
          manejarCambioImagen(archivo);
        }
      }
    };

    document.addEventListener("paste", manejarPegar);
    return () => document.removeEventListener("paste", manejarPegar);
  }, [manejarCambioImagen]);

  return (
    <div
      {...getRootProps()}
      className={`border border-gray-300 rounded p-4 cursor-pointer bg-gray-50 hover:bg-gray-100 min-h-[100px] flex flex-col items-center justify-center gap-2 text-sm text-gray-600 ${
        isDragActive ? "border-blue-400 bg-blue-50" : ""
      }`}
    >
      <input {...getInputProps()} />
      {image ? (
        <div className="relative">
          <img
            src={URL.createObjectURL(image)}
            alt={image.name}
            className="w-16 h-16 object-cover rounded border"
          />
          <button
            onClick={(e) => {
              e.stopPropagation();
              manejarCambioImagen(null);
            }}
            className="absolute -top-1 -right-1 bg-red-500 text-white rounded-full w-5 h-5 text-xs hover:bg-red-600"
          >
            ×
          </button>
          <div className="text-xs text-gray-500 mt-1 max-w-[64px] truncate">
            {image.name}
          </div>
        </div>
      ) : (
        <>
          📁{isDragActive ? "Suelta aquí..." : "Clic, arrastra o pega (Ctrl+V)"}
        </>
      )}
    </div>
  );
};

export default ImagenDropdown;
