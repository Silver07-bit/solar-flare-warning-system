import React, { useState, useRef } from "react";
import { Upload, X, CheckCircle, AlertTriangle, Image as ImageIcon, Sparkles, RefreshCw } from "lucide-react";
import { uploadAndPredictCustomImages, type CustomUploadResult } from "../services/api";

interface CustomImageUploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUploadSuccess: (
    result: CustomUploadResult,
    customLabel: string,
    rawImages: string[],
    activeRegion: string,
    observationTime?: string
  ) => void;
}

export const CustomImageUploadModal: React.FC<CustomImageUploadModalProps> = ({
  isOpen,
  onClose,
  onUploadSuccess,
}) => {
  const [images, setImages] = useState<(string | null)[]>([null, null, null, null]);
  const [activeRegion, setActiveRegion] = useState("NOAA AR-14299");
  const [observationTime, setObservationTime] = useState("2025-12-07T11:00:00Z");
  const [isProcessing, setIsProcessing] = useState(false);
  const [progressText, setProgressText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [activeSlot, setActiveSlot] = useState<number | null>(null);

  if (!isOpen) return null;

  const slotLabels = ["T - 9 hrs", "T - 6 hrs", "T - 3 hrs", "T_0 (Now)"];

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const fileList = Array.from(files);
    setError(null);

    // If uploading to a specific slot
    if (activeSlot !== null) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const base64 = event.target?.result as string;
        setImages((prev) => {
          const updated = [...prev];
          updated[activeSlot] = base64;
          return updated;
        });
        setActiveSlot(null);
      };
      reader.readAsDataURL(fileList[0]);
    } else {
      // Multi-file bulk upload
      const readPromises = fileList.slice(0, 4).map((file) => {
        return new Promise<string>((resolve) => {
          const reader = new FileReader();
          reader.onload = (event) => resolve(event.target?.result as string);
          reader.readAsDataURL(file);
        });
      });

      Promise.all(readPromises).then((loadedImages) => {
        setImages((prev) => {
          const updated = [...prev];
          loadedImages.forEach((img, idx) => {
            if (idx < 4) updated[idx] = img;
          });
          return updated;
        });
      });
    }

    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleSlotClick = (index: number) => {
    setActiveSlot(index);
    fileInputRef.current?.click();
  };

  const handleRemoveSlot = (index: number, e: React.MouseEvent) => {
    e.stopPropagation();
    setImages((prev) => {
      const updated = [...prev];
      updated[index] = null;
      return updated;
    });
  };

  const handleRunInference = async () => {
    const validImages = images.filter((img): img is string => img !== null);
    if (validImages.length === 0) {
      setError("Please upload at least 1 solar observation image to run AI inference.");
      return;
    }

    setIsProcessing(true);
    setError(null);
    setProgressText("Normalizing solar continuum & synthesizing 4-channel tensor...");

    try {
      await new Promise((r) => setTimeout(r, 600));
      setProgressText("Executing PyTorch Spatio-Temporal ConvLSTM forward pass...");

      const result = await uploadAndPredictCustomImages(
        validImages,
        activeRegion || "CUSTOM-AR",
        observationTime || undefined
      );

      await new Promise((r) => setTimeout(r, 600));
      setProgressText("Computing PyTorch Autograd Grad-CAM attention attribution...");

      await new Promise((r) => setTimeout(r, 400));
      onUploadSuccess(
        result,
        activeRegion || "CUSTOM-AR",
        validImages,
        activeRegion || "CUSTOM-AR",
        observationTime || undefined
      );
      onClose();
    } catch (err: any) {
      console.error(err);
      setError(err?.message || "Failed to execute AI prediction on uploaded images.");
    } finally {
      setIsProcessing(false);
      setProgressText("");
    }
  };

  const uploadedCount = images.filter(Boolean).length;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-fade-in">
      <div className="relative w-full max-w-3xl bg-slate-900/95 border border-cyan-500/30 rounded-2xl shadow-2xl shadow-cyan-950/60 p-6 md:p-8 overflow-hidden">
        {/* Glow corner */}
        <div className="absolute top-0 right-0 w-64 h-64 bg-cyan-500/10 rounded-full blur-3xl -z-10 pointer-events-none" />

        {/* Modal Header */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-4 mb-6">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-cyan-950/80 border border-cyan-500/40 text-cyan-400">
              <Upload className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white tracking-wide flex items-center gap-2">
                Custom Solar Sequence Ingestion
                <span className="text-xs px-2 py-0.5 rounded-full bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                  Live AI Inference
                </span>
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Upload 1 to 4 solar images (PNG/JPG/FITS) to extract physical shear and compute live flare predictions.
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            disabled={isProcessing}
            className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="mb-4 p-3 rounded-lg bg-red-950/50 border border-red-500/50 text-red-200 text-xs flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Hidden File Input */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*,.fits,.fts"
          multiple={activeSlot === null}
          onChange={handleFileChange}
          className="hidden"
        />

        {/* Bulk Dropzone / Trigger */}
        <div
          onClick={() => {
            setActiveSlot(null);
            fileInputRef.current?.click();
          }}
          className="mb-6 p-4 border-2 border-dashed border-slate-700 hover:border-cyan-500/60 rounded-xl bg-slate-950/40 hover:bg-cyan-950/20 transition-all cursor-pointer flex flex-col items-center justify-center text-center group"
        >
          <div className="p-3 rounded-full bg-slate-800 group-hover:bg-cyan-900/40 group-hover:text-cyan-400 text-slate-400 transition-colors mb-2">
            <ImageIcon className="w-6 h-6" />
          </div>
          <p className="text-sm font-medium text-slate-200">
            Click here to browse or drag & drop 1 to 4 solar images
          </p>
          <p className="text-xs text-slate-400 mt-1">
            Supports SDO/AIA, Helioviewer screenshots, SUIT UV FITS, and PNG/JPEG formats.
          </p>
        </div>

        {/* 4 Sequence Frame Slots */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
          {slotLabels.map((label, idx) => {
            const imgData = images[idx];
            return (
              <div
                key={label}
                onClick={() => handleSlotClick(idx)}
                className={`relative aspect-square rounded-xl border transition-all cursor-pointer overflow-hidden flex flex-col items-center justify-center p-2 group ${
                  imgData
                    ? "border-cyan-500/60 bg-slate-950/80 shadow-md shadow-cyan-950/40"
                    : "border-slate-800 hover:border-slate-600 bg-slate-950/30"
                }`}
              >
                {imgData ? (
                  <>
                    <img
                      src={imgData}
                      alt={label}
                      className="w-full h-full object-cover rounded-lg group-hover:scale-105 transition-transform duration-300"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-black/40 pointer-events-none" />
                    <button
                      onClick={(e) => handleRemoveSlot(idx, e)}
                      className="absolute top-1.5 right-1.5 p-1 rounded-md bg-black/60 hover:bg-red-600/80 text-white transition-colors"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                    <div className="absolute bottom-1.5 left-2 right-2 flex items-center justify-between text-[11px] font-mono text-cyan-300">
                      <span>{label}</span>
                      <CheckCircle className="w-3.5 h-3.5 text-cyan-400" />
                    </div>
                  </>
                ) : (
                  <>
                    <div className="p-2 rounded-lg bg-slate-800/60 text-slate-400 group-hover:text-cyan-400 transition-colors mb-1.5">
                      <Upload className="w-4 h-4" />
                    </div>
                    <span className="text-xs font-medium text-slate-300">{label}</span>
                    <span className="text-[10px] text-slate-400 mt-0.5">Click to upload</span>
                  </>
                )}
              </div>
            );
          })}
        </div>

        {/* Active Region & Observation Time Inputs */}
        <div className="mb-6 bg-slate-950/60 border border-slate-800/80 rounded-xl p-3.5 space-y-3">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="flex items-center gap-2 text-xs text-slate-300 font-mono">
              <span className="text-slate-400 whitespace-nowrap">Target AR:</span>
              <input
                type="text"
                value={activeRegion}
                onChange={(e) => setActiveRegion(e.target.value)}
                placeholder="e.g. NOAA AR-14299"
                className="w-full bg-slate-900 border border-slate-700 focus:border-cyan-500 rounded-lg px-2.5 py-1 text-cyan-300 font-bold outline-none uppercase text-xs"
              />
            </div>
            <div className="flex items-center gap-2 text-xs text-slate-300 font-mono">
              <span className="text-slate-400 whitespace-nowrap">Timestamp (T_0):</span>
              <input
                type="text"
                value={observationTime}
                onChange={(e) => setObservationTime(e.target.value)}
                placeholder="2025-12-07T11:00:00Z"
                className="w-full bg-slate-900 border border-slate-700 focus:border-cyan-500 rounded-lg px-2.5 py-1 text-amber-300 font-mono outline-none text-xs"
              />
            </div>
          </div>

          {/* Quick NOAA Presets */}
          <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-slate-800/60 text-[11px] font-mono">
            <span className="text-slate-500 text-[10px]">Quick Presets:</span>
            <button
              type="button"
              onClick={() => {
                setActiveRegion("NOAA AR-13664");
                setObservationTime("2024-05-14T17:10:00Z");
              }}
              className="px-2 py-0.5 rounded bg-slate-800/80 hover:bg-rose-950 text-slate-300 hover:text-rose-300 border border-slate-700 hover:border-rose-500/40 transition-colors"
            >
              May 14 (X8.7 Superflare)
            </button>
            <button
              type="button"
              onClick={() => {
                setActiveRegion("NOAA AR-14299");
                setObservationTime("2025-12-07T11:00:00Z");
              }}
              className="px-2 py-0.5 rounded bg-slate-800/80 hover:bg-amber-950 text-slate-300 hover:text-amber-300 border border-slate-700 hover:border-amber-500/40 transition-colors"
            >
              Dec 7 '25 (M8.1 Flare)
            </button>
            <button
              type="button"
              onClick={() => {
                setActiveRegion("NOAA AR-13664");
                setObservationTime("2024-05-10T17:00:00Z");
              }}
              className="px-2 py-0.5 rounded bg-slate-800/80 hover:bg-cyan-950 text-slate-300 hover:text-cyan-300 border border-slate-700 hover:border-cyan-500/40 transition-colors"
            >
              May 10 '24 (X2.8 Storm)
            </button>
            <button
              type="button"
              onClick={() => {
                setActiveRegion("NOAA AR-13842");
                setObservationTime("2024-10-03T12:18:00Z");
              }}
              className="px-2 py-0.5 rounded bg-slate-800/80 hover:bg-rose-950 text-slate-300 hover:text-rose-300 border border-slate-700 hover:border-rose-500/40 transition-colors"
            >
              Oct 3 '24 (X9.0 Flare)
            </button>
            <button
              type="button"
              onClick={() => {
                setActiveRegion("NOAA AR-12673");
                setObservationTime("2017-09-06T12:00:00Z");
              }}
              className="px-2 py-0.5 rounded bg-slate-800/80 hover:bg-rose-950 text-slate-300 hover:text-rose-300 border border-slate-700 hover:border-rose-500/40 transition-colors"
            >
              Sept 6 '17 (X9.3 Eruption)
            </button>
            <button
              type="button"
              onClick={() => {
                setActiveRegion("NOAA AR-13100");
                setObservationTime("2026-08-25T00:00:00Z");
              }}
              className="px-2 py-0.5 rounded bg-slate-800/80 hover:bg-emerald-950 text-slate-300 hover:text-emerald-300 border border-slate-700 hover:border-emerald-500/40 transition-colors"
            >
              Quiet Sun Baseline
            </button>
          </div>

        </div>

        {/* Progress Display */}
        {isProcessing && (
          <div className="mb-6 p-4 rounded-xl bg-cyan-950/40 border border-cyan-500/40 animate-pulse">
            <div className="flex items-center gap-3 text-cyan-300 text-xs font-mono mb-2">
              <RefreshCw className="w-4 h-4 animate-spin text-cyan-400" />
              <span>{progressText}</span>
            </div>
            <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-cyan-500 to-amber-400" style={{ width: "100%" }} />
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="flex items-center justify-end gap-3 border-t border-slate-800 pt-4">
          <button
            type="button"
            onClick={onClose}
            disabled={isProcessing}
            className="px-4 py-2 rounded-xl text-xs font-medium text-slate-300 hover:text-white hover:bg-slate-800 transition-colors"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleRunInference}
            disabled={isProcessing || uploadedCount === 0}
            className={`px-5 py-2.5 rounded-xl text-xs font-bold tracking-wide flex items-center gap-2 transition-all ${
              uploadedCount === 0 || isProcessing
                ? "bg-slate-800 text-slate-400 cursor-not-allowed border border-slate-700"
                : "bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 shadow-lg shadow-cyan-500/25 border border-cyan-400"
            }`}
          >
            {isProcessing ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                Analyzing Solar Sequence...
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" />
                Run AI Flare Prediction
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
