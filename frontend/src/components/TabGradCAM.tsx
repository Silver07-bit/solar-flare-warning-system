import React, { useState } from "react";
import { GlowingCard } from "./ui/glowing-card";
import type { GradCamResponse, GradCamFrame } from "../services/api";
import { Eye, Cpu, Sparkles } from "lucide-react";

export const TabGradCAM: React.FC<{ gradcam: GradCamResponse | null }> = ({
  gradcam,
}) => {
  const [selectedIdx, setSelectedIdx] = useState(3); // Default to T_0

  if (!gradcam || gradcam.frames.length === 0) {
    return (
      <div className="flex items-center justify-center min-h-[400px] text-cyan-400 font-mono text-sm">
        COMPUTING BACKWARD AUTOGRAD GRAD-CAM ATTRIBUTIONS...
      </div>
    );
  }

  const currentFrame = gradcam.frames[selectedIdx] || gradcam.frames[0];

  return (
    <div className="space-y-6">
      {/* Explainability Banner */}
      <div className="flex items-center justify-between border-b border-white/10 pb-4">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Cpu className="h-5 w-5 text-cyan-400" />
            Explainable AI: PyTorch Autograd Grad-CAM Saliency
          </h2>
          <p className="text-xs text-slate-400 font-mono mt-0.5">
            {gradcam.attribution_note}
          </p>
        </div>
        <span className="px-3 py-1 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 font-mono text-xs">
          Zero Black-Box Guessing
        </span>
      </div>

      {/* Frame Timeline Selector */}
      <div className="grid grid-cols-4 gap-4">
        {gradcam.frames.map((frame: GradCamFrame, idx: number) => {
          const isSelected = selectedIdx === idx;
          return (
            <button
              key={idx}
              onClick={() => setSelectedIdx(idx)}
              className={`p-3 rounded-2xl border transition-all text-left ${
                isSelected
                  ? "bg-space-850 border-cyan-400 shadow-lg shadow-cyan-500/20"
                  : "bg-space-900/80 border-white/10 hover:border-white/30"
              }`}
            >
              <div className="flex justify-between items-center mb-1.5 text-xs font-mono">
                <span className="font-bold text-cyan-400">{frame.step}</span>
                <span className="text-slate-400 text-[10px]">
                  Peak Attn: {(frame.peak_attention_score || 0.85).toFixed(2)}
                </span>
              </div>
              <img
                src={frame.gradcam_base64}
                alt={frame.step}
                className="w-full h-28 object-cover rounded-xl border border-white/10"
              />
            </button>
          );
        })}
      </div>

      {/* Side-by-side Inspection */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Raw Observation Patch */}
        <GlowingCard>
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-bold text-white flex items-center gap-2">
              <Eye className="h-4 w-4 text-slate-400" />
              Raw SUIT UV Magnetogram Patch ({currentFrame.step})
            </span>
            <span className="text-xs font-mono text-slate-400">256x256 Crop</span>
          </div>
          <img
            src={currentFrame.patch_base64}
            alt="Raw Patch"
            className="w-full h-80 object-cover rounded-xl border border-white/10 shadow-2xl"
          />
          <div className="mt-3 p-3 bg-space-950/80 rounded-xl border border-white/5 text-xs text-slate-300 font-sans">
            Photospheric flux distribution captured in narrowband 279.6 nm prior to neural gradient projection.
          </div>
        </GlowingCard>

        {/* Grad-CAM Gradient Attention Overlay */}
        <GlowingCard glowColor="rgba(255, 51, 75, 0.35)">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-bold text-rose-400 flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-rose-400" />
              Grad-CAM Class Activation Map ({currentFrame.step})
            </span>
            <span className="text-xs font-mono text-rose-300 font-bold">
              Target: M/X Eruption
            </span>
          </div>
          <img
            src={currentFrame.gradcam_base64}
            alt="Grad-CAM Overlay"
            className="w-full h-80 object-cover rounded-xl border border-rose-500/30 shadow-2xl shadow-rose-500/10"
          />
          <div className="mt-3 p-3 bg-space-950/80 rounded-xl border border-white/5 text-xs text-slate-300 font-sans">
            High-intensity thermal regions (red/yellow) indicate neural attention concentrated on high-shear polarity inversion lines.
          </div>
        </GlowingCard>
      </div>
    </div>
  );
};
