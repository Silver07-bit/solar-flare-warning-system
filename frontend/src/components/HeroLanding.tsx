import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Vortex } from "./ui/vortex";
import { EarthGlobe3D } from "./EarthGlobe3D";
import { Sun, ArrowRight, ShieldCheck, Activity, Radio, Info } from "lucide-react";

interface HeroLandingProps {
  onEnterDashboard: () => void;
  isFlareActive: boolean;
  activeRegion: string;
}

export const HeroLanding: React.FC<HeroLandingProps> = ({
  onEnterDashboard,
  isFlareActive,
  activeRegion,
}) => {
  const [showTelemetryHover, setShowTelemetryHover] = useState(false);

  const threatScore = isFlareActive ? 86 : 8;
  const threatLevelLabel = isFlareActive
    ? "NOAA SCALE: R4 (Severe HF Blackout) | G5 (Extreme Storm) | S3 (Radiation Hazard)"
    : "NOAA SCALE: R0 / G0 / S0 (Quiet Photospheric Baseline)";

  // SVG Sparkline coordinates for 24h X-ray flux curve
  const sparklinePoints = isFlareActive
    ? "0,35 20,33 40,34 60,30 80,28 100,12 110,4 120,8 140,16 160,24 180,28 200,32"
    : "0,35 30,34 60,35 90,34 120,35 150,34 180,35 200,35";

  return (
    <div className="relative min-h-[92vh] flex flex-col justify-center items-center overflow-hidden py-14 px-4 md:px-8">
      {/* Vortex Cosmic Particle Flow */}
      <Vortex
        isFlareActive={isFlareActive}
        className="flex flex-col items-center justify-center w-full max-w-7xl mx-auto"
      >
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 lg:gap-12 items-center w-full">
          {/* Left Column: Mission Hook & Threat Assessment */}
          <div className="lg:col-span-7 space-y-6 text-left">
            {/* Top Mission Identification Badge */}
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-space-900/90 border border-cyan-500/30 backdrop-blur-xl shadow-lg shadow-cyan-500/10 text-xs font-mono text-cyan-300"
            >
              <Sun className="h-4 w-4 text-orange-400 animate-spin-slow" />
              <span>ISRO ADITYA-L1 SUIT PAYLOAD (279.6 nm)</span>
              <span className="text-white/30">•</span>
              <span className="text-orange-400 font-bold">SMART INDIA HACKATHON 2026</span>
            </motion.div>

            {/* 🌟 4. Real-Time NOAA Threat Level Gauge */}
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.05 }}
              className="rounded-2xl border border-white/10 bg-space-950/80 p-3.5 backdrop-blur-xl shadow-xl space-y-2"
            >
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-slate-400 flex items-center gap-1.5">
                  <Radio className={`h-3.5 w-3.5 ${isFlareActive ? "text-rose-400 animate-pulse" : "text-emerald-400"}`} />
                  {threatLevelLabel}
                </span>
                <span className={`font-bold px-2 py-0.5 rounded text-[10px] ${
                  isFlareActive
                    ? "bg-rose-500/20 text-rose-300 border border-rose-500/40 animate-pulse"
                    : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                }`}>
                  {isFlareActive ? "DANGER ZONE (86%)" : "NOMINAL (8%)"}
                </span>
              </div>

              {/* Animated Danger Slider Gauge */}
              <div className="w-full bg-space-900 rounded-full h-2.5 overflow-hidden border border-white/10 p-[1px]">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${threatScore}%` }}
                  transition={{ duration: 1.2, ease: "easeOut" }}
                  className={`h-full rounded-full transition-colors duration-500 ${
                    isFlareActive
                      ? "bg-gradient-to-r from-amber-500 via-rose-500 to-red-600 shadow-lg shadow-rose-500/50"
                      : "bg-gradient-to-r from-emerald-500 to-cyan-500"
                  }`}
                />
              </div>
            </motion.div>

            {/* 📐 3. Main Headline with Improved Line-Height */}
            <motion.h1
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="text-3xl sm:text-4xl md:text-5xl lg:text-[54px] font-extrabold text-white tracking-tight leading-[1.2] md:leading-[1.18]"
            >
              Predicting Solar Eruptions{" "}
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 via-sky-300 to-blue-400">
                24 to 48 Hours
              </span>{" "}
              Before Earth Impact.
            </motion.h1>

            {/* 📐 3. Structured Subtitle with Bold Architectural Highlights */}
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.15 }}
              className="text-sm md:text-[15px] text-slate-300 leading-relaxed max-w-2xl font-sans space-y-2"
            >
              <p>
                Transforming reactive space mitigation into proactive national asset defense.
                The system analyzes multi-spectral solar magnetic shear to anticipate M/X-class eruptions prior to ionospheric arrival.
              </p>
              <div className="flex flex-wrap gap-2 pt-1 font-mono text-xs">
                <span className="px-2.5 py-1 rounded-lg bg-cyan-500/10 text-cyan-300 border border-cyan-500/30">
                  ⚡ 4-Channel ConvLSTM
                </span>
                <span className="px-2.5 py-1 rounded-lg bg-blue-500/10 text-blue-300 border border-blue-500/30">
                  🔍 PyTorch Grad-CAM XAI
                </span>
                <span className="px-2.5 py-1 rounded-lg bg-emerald-500/10 text-emerald-300 border border-emerald-500/30">
                  📊 12-Fold LORO-CV Rigor
                </span>
              </div>
            </motion.div>

            {/* 📊 2. Upgraded Active Target Live Stream Chip & CTA */}
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.25 }}
              className="pt-2 flex flex-wrap items-center gap-4 relative"
            >
              {/* Primary Glowing Action Button */}
              <button
                onClick={onEnterDashboard}
                className="relative group/btn overflow-hidden rounded-2xl p-[2px] cursor-pointer transition-transform hover:scale-105 active:scale-95 shadow-2xl shadow-cyan-500/25"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-cyan-500 via-blue-500 to-teal-400 animate-shimmer" />
                <div className="relative px-7 py-3.5 bg-space-950 rounded-2xl flex items-center gap-3 text-sm md:text-base font-bold text-white transition-colors group-hover/btn:bg-space-900">
                  <span>LAUNCH MISSION DASHBOARD</span>
                  <ArrowRight className="h-5 w-5 text-cyan-400 transition-transform group-hover/btn:translate-x-1" />
                </div>
              </button>

              {/* 📊 2. Live Telemetry Target Card with Hover Sparkline Modal */}
              <div
                className="relative"
                onMouseEnter={() => setShowTelemetryHover(true)}
                onMouseLeave={() => setShowTelemetryHover(false)}
              >
                <div className="flex items-center gap-2 text-xs font-mono text-slate-300 bg-space-900/90 hover:bg-space-850 px-4 py-3 rounded-2xl border border-white/15 cursor-pointer shadow-lg transition-all">
                  <span className="relative flex h-2.5 w-2.5">
                    <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
                      isFlareActive ? "bg-rose-400" : "bg-emerald-400"
                    }`} />
                    <span className={`relative inline-flex rounded-full h-2.5 w-2.5 ${
                      isFlareActive ? "bg-rose-500" : "bg-emerald-500"
                    }`} />
                  </span>
                  <span>
                    Target: <strong className="text-cyan-300">{activeRegion}</strong>
                  </span>
                  <Info className="h-3.5 w-3.5 text-slate-400 ml-0.5" />
                </div>

                {/* Micro Hover Popover: Live X-Ray Flux Sparkline */}
                <AnimatePresence>
                  {showTelemetryHover && (
                    <motion.div
                      initial={{ opacity: 0, y: 8, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: 8, scale: 0.95 }}
                      transition={{ duration: 0.18 }}
                      className="absolute bottom-full left-0 mb-3 w-72 rounded-2xl border border-cyan-500/40 bg-space-950/95 p-4 backdrop-blur-2xl shadow-2xl z-50 pointer-events-none font-mono text-xs"
                    >
                      <div className="flex items-center justify-between pb-2 border-b border-white/10 mb-2">
                        <span className="font-bold text-white flex items-center gap-1.5">
                          <Activity className="h-3.5 w-3.5 text-cyan-400" />
                          Live GOES Flux Telemetry
                        </span>
                        <span className="text-[10px] text-emerald-400 font-extrabold">● STREAMING</span>
                      </div>

                      {/* SVG Sparkline */}
                      <div className="bg-space-900/90 rounded-xl p-2 border border-white/5 mb-3">
                        <div className="flex justify-between text-[10px] text-slate-400 mb-1">
                          <span>24h X-Ray Flux</span>
                          <span className={isFlareActive ? "text-rose-400 font-bold" : "text-emerald-400"}>
                            {isFlareActive ? "X1.2 Peak Spike" : "B1.0 Baseline"}
                          </span>
                        </div>
                        <svg viewBox="0 0 200 40" className="w-full h-10 overflow-visible">
                          <polyline
                            fill="none"
                            stroke={isFlareActive ? "#ff334b" : "#00e5ff"}
                            strokeWidth="2.5"
                            points={sparklinePoints}
                          />
                        </svg>
                      </div>

                      <div className="grid grid-cols-2 gap-2 text-[10px] text-slate-300">
                        <div>
                          <span className="text-slate-500 block">MAGNETIC SHEAR</span>
                          <span className="text-white font-bold">{isFlareActive ? "0.85 max |∇I|" : "0.12 max |∇I|"}</span>
                        </div>
                        <div>
                          <span className="text-slate-500 block">HALE CLASS</span>
                          <span className="text-amber-300 font-bold">{isFlareActive ? "β-γ-δ Inversion" : "α Unipolar"}</span>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </motion.div>

            {/* Quick Metrics Strip */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.35 }}
              className="grid grid-cols-3 gap-3 pt-3 border-t border-white/10 font-mono text-xs"
            >
              <div className="bg-space-950/70 p-3 rounded-xl border border-white/5">
                <span className="text-slate-500 block text-[10px]">SCIENTIFIC RIGOR</span>
                <span className="text-cyan-400 font-bold">12-Fold LORO-CV</span>
              </div>
              <div className="bg-space-950/70 p-3 rounded-xl border border-white/5">
                <span className="text-slate-500 block text-[10px]">PRECURSOR WINDOW</span>
                <span className="text-amber-400 font-bold">24-48 Hours</span>
              </div>
              <div className="bg-space-950/70 p-3 rounded-xl border border-white/5">
                <span className="text-slate-500 block text-[10px]">EXPLAINABLE AI</span>
                <span className="text-emerald-400 font-bold">PyTorch Autograd</span>
              </div>
            </motion.div>
          </div>

          {/* Right Column: 🚨 1. Interactive 3D Earth Globe & Orbit System */}
          <div className="lg:col-span-5 relative flex flex-col items-center justify-center">
            <div className="relative w-full rounded-3xl border border-white/10 bg-space-900/50 p-4 md:p-5 backdrop-blur-2xl shadow-2xl">
              <div className="flex items-center justify-between px-2 mb-1 font-mono text-xs text-slate-400">
                <span className="flex items-center gap-1.5">
                  <ShieldCheck className={`h-4 w-4 ${isFlareActive ? "text-rose-400" : "text-cyan-400"}`} />
                  3D Magnetosphere Shield & Orbits
                </span>
                <span className={`px-2.5 py-0.5 rounded text-[10px] font-bold ${
                  isFlareActive
                    ? "bg-rose-500/20 text-rose-300 border border-rose-500/30 animate-pulse"
                    : "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30"
                }`}>
                  {isFlareActive ? "SHIELD COMPRESSED" : "GEO SHIELD NOMINAL"}
                </span>
              </div>

              {/* Three.js Interactive Earth Component */}
              <EarthGlobe3D isCritical={isFlareActive} />

              <div className="mt-2 text-center text-[10px] text-slate-400 font-mono">
                Click & Drag to rotate • Select orbit filters below to inspect satellite altitude.
              </div>
            </div>
          </div>
        </div>
      </Vortex>
    </div>
  );
};