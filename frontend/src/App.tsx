import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { BlanketMesh } from "./components/ui/blanket-mesh";
import { Spotlight } from "./components/ui/spotlight";
import { Header } from "./components/Header";
import { HeroLanding } from "./components/HeroLanding";
import { AlertBanner } from "./components/AlertBanner";
import { TabMissionControl } from "./components/TabMissionControl";
import { TabDiagnostics } from "./components/TabDiagnostics";
import { TabGradCAM } from "./components/TabGradCAM";
import { TabGridSimulation } from "./components/TabGridSimulation";
import { TabTelemetryBulletin } from "./components/TabTelemetryBulletin";
import {
  fetchPrediction,
  fetchGradCam,
  FALLBACK_PREDICTIONS,
  FALLBACK_GRADCAM,
  type PredictResponse,
  type GradCamResponse,
} from "./services/api";
import { Activity, Layers, Cpu, ShieldAlert, Award, Compass, Eye } from "lucide-react";

export function App() {
  const [viewMode, setViewMode] = useState<"hero" | "dashboard">("hero");
  const [activeTab, setActiveTab] = useState<number>(0);
  const [scenario, setScenario] = useState<string>("AR3664_Impending_X_Flare");
  const [prediction, setPrediction] = useState<PredictResponse>(
    FALLBACK_PREDICTIONS["AR3664_Impending_X_Flare"]
  );
  const [gradcam, setGradcam] = useState<GradCamResponse>(FALLBACK_GRADCAM["default"]);
  const [loading, setLoading] = useState<boolean>(false);

  const loadData = async (sc: string) => {
    // Set fallback immediately for 0ms lag
    if (FALLBACK_PREDICTIONS[sc]) {
      setPrediction(FALLBACK_PREDICTIONS[sc]);
    }
    try {
      const [predRes, gradRes] = await Promise.all([
        fetchPrediction(sc),
        fetchGradCam(sc),
      ]);
      setPrediction(predRes);
      setGradcam(gradRes);
    } catch (err) {
      console.warn("Using offline fallback telemetry", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData(scenario);
  }, [scenario]);

  const isFlareActive = scenario !== "AR3670_Quiet_Sun";

  const tabs = [
    { id: 0, label: "Live Sun & Forecast", icon: <Activity className="h-4 w-4" /> },
    { id: 1, label: "Multi-Spectral Tensors", icon: <Layers className="h-4 w-4" /> },
    { id: 2, label: "Explainable AI (Grad-CAM)", icon: <Cpu className="h-4 w-4" /> },
    { id: 3, label: "Grid Impact Simulation", icon: <ShieldAlert className="h-4 w-4" /> },
    { id: 4, label: "ISSDC Advisory & CV Scores", icon: <Award className="h-4 w-4" /> },
  ];

  return (
    <div className="min-h-screen bg-space-950 text-slate-100 relative selection:bg-cyan-500 selection:text-black font-sans">
      {/* Interactive Elastic Blanket Mesh Physics */}
      <BlanketMesh isFlareActive={isFlareActive} />

      {/* Radial Spotlight */}
      <Spotlight className="-top-40 left-0 md:left-60 md:-top-20" fill="rgba(0, 176, 255, 0.04)" />

      {/* Top Mission Header */}
      <Header
        scenario={scenario}
        setScenario={setScenario}
        riskLevel={prediction?.risk_level || "MODERATE"}
        targetAR={prediction?.target_active_region || "AR-13664"}
      />

      {/* Top View Mode Switcher */}
      <div className="max-w-7xl mx-auto px-6 pt-4 flex items-center justify-between">
        <div className="flex items-center gap-2 bg-space-900/90 p-1 rounded-xl border border-white/10 text-xs font-mono">
          <button
            onClick={() => setViewMode("hero")}
            className={`px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition-all cursor-pointer ${
              viewMode === "hero"
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 font-bold"
                : "text-slate-400 hover:text-white"
            }`}
          >
            <Compass className="h-3.5 w-3.5" />
            <span>Mission Overview</span>
          </button>
          <button
            onClick={() => setViewMode("dashboard")}
            className={`px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition-all cursor-pointer ${
              viewMode === "dashboard"
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 font-bold"
                : "text-slate-400 hover:text-white"
            }`}
          >
            <Eye className="h-3.5 w-3.5" />
            <span>Telemetry Dashboard</span>
          </button>
        </div>

        <div className="hidden md:flex items-center gap-3 text-xs font-mono text-slate-400">
          <span>Active Orbit: <strong className="text-cyan-400">Halo L1 (1.5M km)</strong></span>
          <span>•</span>
          <span>Filter: <strong className="text-amber-400">SUIT NB 279.6 nm</strong></span>
        </div>
      </div>

      {/* Main View Area */}
      <main className="relative z-10 max-w-7xl mx-auto px-6 py-4 space-y-6">
        <AnimatePresence mode="wait">
          {viewMode === "hero" ? (
            <motion.div
              key="hero-view"
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              transition={{ duration: 0.3 }}
            >
              <HeroLanding
                onEnterDashboard={() => setViewMode("dashboard")}
                isFlareActive={isFlareActive}
                activeRegion={prediction?.target_active_region || "AR-13664"}
              />
            </motion.div>
          ) : (
            <motion.div
              key="dashboard-view"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -12 }}
              transition={{ duration: 0.3 }}
              className="space-y-6"
            >
              {/* Urgent DEFCON Countdown Alert Banner */}
              <AlertBanner
                riskLevel={prediction?.risk_level || "MODERATE"}
                predictedClass={prediction?.predicted_class || "M/X Class"}
                estimatedPeakFlux={prediction?.estimated_peak_flux || "X1.2 (1.2e-4 W/m²)"}
                isFlareActive={isFlareActive}
              />

              {/* Navigation Tabs */}
              <div className="flex items-center gap-2 overflow-x-auto pb-2 border-b border-white/10 no-scrollbar">
                {tabs.map((tab) => {
                  const isActive = activeTab === tab.id;
                  return (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id)}
                      className={`relative px-4 py-2.5 rounded-xl font-medium text-xs md:text-sm transition-all duration-300 flex items-center gap-2 whitespace-nowrap cursor-pointer ${
                        isActive
                          ? "text-cyan-300 bg-space-900/90 border border-cyan-500/40 shadow-lg shadow-cyan-500/10"
                          : "text-slate-400 hover:text-white hover:bg-space-900/50 border border-transparent"
                      }`}
                    >
                      {tab.icon}
                      <span>{tab.label}</span>
                      {isActive && (
                        <motion.div
                          layoutId="activeTabGlow"
                          className="absolute inset-0 rounded-xl bg-cyan-500/10 -z-10"
                          transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                        />
                      )}
                    </button>
                  );
                })}
              </div>

              {/* Tab Content Panels */}
              <div className="min-h-[500px]">
                <AnimatePresence mode="wait">
                  <motion.div
                    key={activeTab + scenario}
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -12 }}
                    transition={{ duration: 0.3 }}
                  >
                    {activeTab === 0 && (
                      <TabMissionControl
                        prediction={prediction}
                        gradcam={gradcam}
                        loading={loading}
                      />
                    )}
                    {activeTab === 1 && <TabDiagnostics scenario={scenario} />}
                    {activeTab === 2 && <TabGradCAM gradcam={gradcam} />}
                    {activeTab === 3 && (
                      <TabGridSimulation
                        prediction={prediction}
                        isFlareActive={isFlareActive}
                      />
                    )}
                    {activeTab === 4 && <TabTelemetryBulletin />}
                  </motion.div>
                </AnimatePresence>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Footer */}
      <footer className="relative z-10 border-t border-white/10 bg-space-950/90 py-6 text-center text-xs text-slate-400 font-mono">
        <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row items-center justify-between gap-3">
          <div>
            ISRO Aditya-L1 Space Weather Warning System | Smart India Hackathon (SIH) 2026
          </div>
          <div className="flex items-center gap-4 text-slate-400">
            <span className="text-cyan-400 font-bold">FastAPI Backend: localhost:8000</span>
            <span>•</span>
            <span className="text-emerald-400 font-bold">12-Fold LORO-CV Verified</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
