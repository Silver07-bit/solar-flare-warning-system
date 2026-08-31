import React, { useState, useEffect } from "react";
import { AlertTriangle, ShieldCheck, Volume2, VolumeX, Flame, Clock } from "lucide-react";
import { motion } from "framer-motion";

interface AlertBannerProps {
  riskLevel: string;
  predictedClass: string;
  estimatedPeakFlux: string;
  isFlareActive: boolean;
}

export const AlertBanner: React.FC<AlertBannerProps> = ({
  riskLevel,
  predictedClass,
  estimatedPeakFlux,
  isFlareActive,
}) => {
  const [timeLeft, setTimeLeft] = useState({ hours: 34, minutes: 18, seconds: 45 });
  const [audioEnabled, setAudioEnabled] = useState(false);

  useEffect(() => {
    const timer = setInterval(() => {
      setTimeLeft((prev) => {
        if (prev.seconds > 0) return { ...prev, seconds: prev.seconds - 1 };
        if (prev.minutes > 0) return { ...prev, minutes: 59, seconds: 59 };
        if (prev.hours > 0) return { hours: prev.hours - 1, minutes: 59, seconds: 59 };
        return { hours: 36, minutes: 0, seconds: 0 };
      });
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  const playAlertChime = () => {
    try {
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = isFlareActive ? "sawtooth" : "sine";
      osc.frequency.setValueAtTime(isFlareActive ? 880 : 440, audioCtx.currentTime);
      gain.gain.setValueAtTime(0.1, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.5);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start();
      osc.stop(audioCtx.currentTime + 0.5);
    } catch (e) {
      console.warn("AudioContext not permitted", e);
    }
  };

  const handleAudioToggle = () => {
    const next = !audioEnabled;
    setAudioEnabled(next);
    if (next) playAlertChime();
  };

  const isCritical = riskLevel === "CRITICAL" || isFlareActive;

  return (
    <motion.div
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      className={`rounded-2xl border p-4 transition-all duration-500 backdrop-blur-xl ${
        isCritical
          ? "border-rose-500/50 bg-gradient-to-r from-rose-950/80 via-space-900/90 to-rose-950/80 shadow-2xl shadow-rose-500/20"
          : "border-emerald-500/30 bg-gradient-to-r from-space-900 via-space-850 to-space-900"
      }`}
    >
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        {/* Left: Alert Status */}
        <div className="flex items-center gap-3">
          <div
            className={`h-11 w-11 rounded-xl flex items-center justify-center border shadow-lg ${
              isCritical
                ? "bg-rose-500/20 border-rose-500/40 text-rose-400 animate-pulse"
                : "bg-emerald-500/20 border-emerald-500/40 text-emerald-400"
            }`}
          >
            {isCritical ? (
              <AlertTriangle className="h-6 w-6" />
            ) : (
              <ShieldCheck className="h-6 w-6" />
            )}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span
                className={`text-[11px] font-mono font-extrabold px-2 py-0.5 rounded border ${
                  isCritical
                    ? "bg-rose-500 text-black border-rose-400 animate-bounce"
                    : "bg-emerald-500/20 text-emerald-300 border-emerald-500/30"
                }`}
              >
                {isCritical ? "DEFCON 1 — CRITICAL WARNING" : "DEFCON 4 — NOMINAL"}
              </span>
              <span className="text-xs text-slate-300 font-mono">
                Aditya-L1 SUIT Automated Trigger
              </span>
            </div>
            <div className="text-sm md:text-base font-bold text-white mt-0.5 flex items-center gap-2">
              {isCritical ? (
                <>
                  <Flame className="h-4 w-4 text-rose-400" />
                  <span>{predictedClass} ERUPTION DETECTED — PEAK FLUX: {estimatedPeakFlux}</span>
                </>
              ) : (
                <span>Solar Baseline Stable (Quiet Photospheric Magnetism)</span>
              )}
            </div>
          </div>
        </div>

        {/* Right: Countdown & Audio Control */}
        <div className="flex items-center gap-4">
          {/* Countdown Clock */}
          <div className="flex items-center gap-2 bg-space-950/80 border border-white/10 px-4 py-2 rounded-xl font-mono text-xs shadow-inner">
            <Clock className={`h-4 w-4 ${isCritical ? "text-rose-400 animate-spin" : "text-cyan-400"}`} />
            <div>
              <span className="text-[9px] text-slate-400 block uppercase">
                {isCritical ? "EST. EARTH IMPACT WINDOW" : "NEXT OBSERVATION STEP"}
              </span>
              <span className="text-sm font-bold text-white tracking-widest">
                T- {String(timeLeft.hours).padStart(2, "0")}h:
                {String(timeLeft.minutes).padStart(2, "0")}m:
                {String(timeLeft.seconds).padStart(2, "0")}s
              </span>
            </div>
          </div>

          {/* Sound Synthesizer Chime Toggle */}
          <button
            onClick={handleAudioToggle}
            className={`p-2.5 rounded-xl border transition-all cursor-pointer ${
              audioEnabled
                ? "bg-cyan-500/20 border-cyan-400 text-cyan-300 shadow-lg shadow-cyan-500/20"
                : "bg-space-900 border-white/10 text-slate-400 hover:text-white"
            }`}
            title="Toggle Mission Alert Chime"
          >
            {audioEnabled ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
          </button>
        </div>
      </div>
    </motion.div>
  );
};