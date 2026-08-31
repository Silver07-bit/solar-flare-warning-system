import React from "react";
import { motion } from "framer-motion";
import { GlowingCard } from "./ui/glowing-card";
import type { PredictResponse, GradCamResponse, GradCamFrame } from "../services/api";
import { Zap, Activity, Flame, BarChart3 } from "lucide-react";

interface TabMissionControlProps {
  prediction: PredictResponse | null;
  gradcam: GradCamResponse | null;
  loading: boolean;
}

export const TabMissionControl: React.FC<TabMissionControlProps> = ({
  prediction,
  gradcam,
  loading,
}) => {
  if (loading || !prediction) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] text-cyan-400 gap-3">
        <div className="h-10 w-10 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
        <span className="font-mono text-sm tracking-wider">
          SYNCHRONIZING ADITYA-L1 TELEMETRY...
        </span>
      </div>
    );
  }

  const p24 = prediction.mx_probability_24h;
  const p48 = prediction.mx_probability_48h;
  const mc = prediction.multiclass_distribution;

  const classData = [
    { label: "Quiet / B", val: mc.Quiet_B, color: "bg-emerald-500", glow: "shadow-emerald-500/30" },
    { label: "C-Class", val: mc.C_Class, color: "bg-blue-500", glow: "shadow-blue-500/30" },
    { label: "M-Class", val: mc.M_Class, color: "bg-amber-500", glow: "shadow-amber-500/30" },
    { label: "X-Class", val: mc.X_Class, color: "bg-rose-500", glow: "shadow-rose-500/30" },
  ];

  return (
    <div className="space-y-6">
      {/* Top Banner Alert */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="rounded-2xl border border-cyan-500/20 bg-gradient-to-r from-space-900 via-space-850 to-space-900 p-4 flex flex-wrap items-center justify-between gap-4 shadow-xl"
      >
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
            <Activity className="h-5 w-5" />
          </div>
          <div>
            <div className="text-xs text-slate-400 font-mono">
              OBSERVATION TIMESTAMP: {prediction.observation_time}
            </div>
            <div className="text-sm font-bold text-white flex items-center gap-2">
              Target: <span className="text-cyan-400">{prediction.target_active_region}</span>
              <span className="text-slate-500">|</span>
              Forecast Window: <span className="text-slate-300 font-mono text-xs">{prediction.forecast_window.start_utc} → {prediction.forecast_window.end_utc}</span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2 font-mono text-xs">
          <span className="px-3 py-1 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            PRADAN Pipeline Ready
          </span>
          <span className="px-3 py-1 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
            Calibrated T = 0.254
          </span>
        </div>
      </motion.div>

      {/* Main Control Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Observation Reel */}
        <div className="lg:col-span-7 space-y-4">
          <GlowingCard className="h-full">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2 text-white font-bold text-base">
                <Flame className="h-5 w-5 text-amber-400" />
                Spatio-Temporal Observation Reel (T-3 to T_0)
              </div>
              <span className="text-xs font-mono text-slate-400">
                SUIT 279.6 nm Filter
              </span>
            </div>

            {/* Reel Images */}
            <div className="grid grid-cols-4 gap-3 mb-4">
              {gradcam?.frames.map((frame: GradCamFrame, idx: number) => (
                <div
                  key={idx}
                  className="group relative rounded-xl overflow-hidden border border-white/10 bg-space-950/80 p-1.5 transition-transform hover:scale-105"
                >
                  <img
                    src={frame.patch_base64}
                    alt={frame.step}
                    className="w-full h-24 object-cover rounded-lg"
                  />
                  <div className="mt-1.5 flex items-center justify-between text-[10px] font-mono text-slate-300 px-1">
                    <span className="text-cyan-400 font-bold">{frame.step}</span>
                    <span className="text-slate-400">256x256</span>
                  </div>
                </div>
              ))}
            </div>

            {/* Physical Telemetry Bar */}
            <div className="grid grid-cols-3 gap-3 pt-3 border-t border-white/10 font-mono text-xs">
              <div className="bg-space-950/60 p-2.5 rounded-xl border border-white/5">
                <span className="text-slate-400 block text-[10px]">PEAK FLUX PROXY</span>
                <span className="text-white font-bold">{prediction.optical_proxies?.peak_intensity?.toFixed(2) || "1.00"}</span>
              </div>
              <div className="bg-space-950/60 p-2.5 rounded-xl border border-white/5">
                <span className="text-slate-400 block text-[10px]">SHEAR GRADIENT (|∇I|)</span>
                <span className="text-cyan-400 font-bold">{prediction.optical_proxies?.max_gradient?.toFixed(2) || "0.85"}</span>
              </div>
              <div className="bg-space-950/60 p-2.5 rounded-xl border border-white/5">
                <span className="text-slate-400 block text-[10px]">COMPLEXITY INDEX</span>
                <span className="text-amber-400 font-bold">{prediction.optical_proxies?.complexity_index?.toFixed(2) || "1.42"}</span>
              </div>
            </div>
          </GlowingCard>
        </div>

        {/* Prediction Gauges & Distribution */}
        <div className="lg:col-span-5 space-y-4">
          <GlowingCard glowColor="rgba(0, 229, 255, 0.4)">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2 text-white font-bold text-base">
                <Zap className="h-5 w-5 text-cyan-400" />
                24h & 48h Eruption Probability
              </div>
              <span className="px-2 py-0.5 rounded text-[11px] font-bold bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                {prediction.predicted_class}
              </span>
            </div>

            {/* Gauge meters */}
            <div className="grid grid-cols-2 gap-4 text-center my-3">
              <div className="relative p-4 rounded-2xl bg-space-950/90 border border-cyan-500/30 shadow-lg shadow-cyan-500/10">
                <div className="text-[11px] font-mono text-slate-400 mb-1">
                  24-HOUR FORECAST
                </div>
                <div className="text-3xl font-extrabold text-cyan-400 tracking-tight">
                  {p24.toFixed(1)}%
                </div>
                <div className="text-[10px] text-slate-400 mt-1 font-mono">
                  Calibrated Platt Scaled
                </div>
              </div>

              <div className="relative p-4 rounded-2xl bg-space-950/90 border border-amber-500/30 shadow-lg shadow-amber-500/10">
                <div className="text-[11px] font-mono text-slate-400 mb-1">
                  48-HOUR ACCUMULATED
                </div>
                <div className="text-3xl font-extrabold text-amber-400 tracking-tight">
                  {p48.toFixed(1)}%
                </div>
                <div className="text-[10px] text-slate-400 mt-1 font-mono">
                  Forward Temporal Window
                </div>
              </div>
            </div>

            {/* Peak Flux & Confidence */}
            <div className="flex items-center justify-between bg-space-950/80 rounded-xl p-3 border border-white/5 text-xs font-mono mb-4">
              <div>
                <span className="text-slate-400 block text-[10px]">ESTIMATED PEAK X-RAY FLUX</span>
                <span className="text-rose-400 font-bold text-sm">{prediction.estimated_peak_flux}</span>
              </div>
              <div className="text-right">
                <span className="text-slate-400 block text-[10px]">MODEL CONFIDENCE</span>
                <span className="text-white font-bold">{prediction.model_confidence.toFixed(1)}%</span>
              </div>
            </div>

            {/* NOAA Category Probabilities */}
            <div className="space-y-2 pt-2 border-t border-white/10">
              <div className="flex items-center justify-between text-xs text-slate-300 font-semibold mb-1">
                <span>NOAA Category Distribution</span>
                <BarChart3 className="h-3.5 w-3.5 text-slate-400" />
              </div>
              {classData.map((cd, idx) => (
                <div key={idx} className="space-y-1">
                  <div className="flex justify-between text-[11px] font-mono">
                    <span className="text-slate-300">{cd.label}</span>
                    <span className="text-white font-bold">{cd.val.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-space-950 rounded-full h-2 overflow-hidden border border-white/5">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.min(100, cd.val)}%` }}
                      transition={{ duration: 0.8, delay: idx * 0.1 }}
                      className={`h-full rounded-full ${cd.color}`}
                    />
                  </div>
                </div>
              ))}
            </div>
          </GlowingCard>
        </div>
      </div>
    </div>
  );
};