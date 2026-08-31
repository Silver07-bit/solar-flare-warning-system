import React, { useState, useEffect } from "react";
import { GlowingCard } from "./ui/glowing-card";
import { fetchSolarChannels, type SolarChannelsResponse, type SolarChannel } from "../services/api";
import { Layers, Compass, Eye, Sparkles } from "lucide-react";

export const TabDiagnostics: React.FC<{ scenario: string }> = ({ scenario }) => {
  const [data, setData] = useState<SolarChannelsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState("ch0");

  useEffect(() => {
    setLoading(true);
    fetchSolarChannels(scenario)
      .then((res: SolarChannelsResponse) => setData(res))
      .catch((err: unknown) => console.error(err))
      .finally(() => setLoading(false));
  }, [scenario]);

  if (loading || !data) {
    return (
      <div className="flex items-center justify-center min-h-[400px] text-cyan-400 font-mono text-sm">
        EXTRACTING 4-CHANNEL MULTI-SPECTRAL TENSORS...
      </div>
    );
  }

  const selectedChannel = data.channels.find((c: SolarChannel) => c.id === activeTab) || data.channels[0];

  return (
    <div className="space-y-6">
      {/* Intro Header */}
      <div className="flex items-center justify-between border-b border-white/10 pb-4">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Layers className="h-5 w-5 text-cyan-400" />
            4-Channel Spatio-Temporal Tensor Synthesis
          </h2>
          <p className="text-xs text-slate-400 font-mono mt-0.5">
            Physics-informed deep learning input representation: [Batch, Sequence=4, Channels=4, 256, 256]
          </p>
        </div>
        <span className="px-3 py-1 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 font-mono text-xs">
          Zero Grayscale Loss
        </span>
      </div>

      {/* Grid of 4 Channels */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {data.channels.map((ch: SolarChannel) => {
          const isSelected = activeTab === ch.id;
          return (
            <button
              key={ch.id}
              onClick={() => setActiveTab(ch.id)}
              className={`text-left p-4 rounded-2xl border transition-all duration-300 ${
                isSelected
                  ? "bg-space-850 border-cyan-400/80 shadow-lg shadow-cyan-500/20"
                  : "bg-space-900/80 border-white/10 hover:border-white/30"
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-mono font-bold text-cyan-400 uppercase">
                  {ch.id.toUpperCase()}
                </span>
                {isSelected && <Sparkles className="h-4 w-4 text-cyan-400 animate-pulse" />}
              </div>
              <img
                src={ch.image_base64}
                alt={ch.name}
                className="w-full h-36 object-cover rounded-xl mb-3 border border-white/10"
              />
              <div className="text-xs font-bold text-white truncate">{ch.name}</div>
              <p className="text-[11px] text-slate-400 line-clamp-2 mt-1 font-sans">
                {ch.description}
              </p>
            </button>
          );
        })}
      </div>

      {/* Focused Channel Deep Dive */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-6">
          <GlowingCard>
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-bold text-white flex items-center gap-2">
                <Eye className="h-4 w-4 text-cyan-400" />
                Active Region High-Res Channel Inspection
              </span>
              <span className="text-xs font-mono text-cyan-400 font-semibold">
                {selectedChannel.id.toUpperCase()}
              </span>
            </div>
            <img
              src={selectedChannel.image_base64}
              alt={selectedChannel.name}
              className="w-full h-72 object-cover rounded-xl border border-white/10 shadow-2xl"
            />
            <div className="mt-4 p-3 bg-space-950/80 rounded-xl border border-white/5 font-mono text-xs text-slate-300">
              <span className="text-cyan-400 font-bold block mb-1">
                {selectedChannel.name}
              </span>
              {selectedChannel.description}
            </div>
          </GlowingCard>
        </div>

        <div className="lg:col-span-6">
          <GlowingCard>
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-bold text-white flex items-center gap-2">
                <Compass className="h-4 w-4 text-amber-400" />
                Calibrated Full Solar Disk Context (SUIT FOV)
              </span>
              <span className="text-xs font-mono text-slate-400">
                1024x1024 Heliographic
              </span>
            </div>
            <img
              src={data.full_disk}
              alt="Full Solar Disk"
              className="w-full h-72 object-contain rounded-xl border border-white/10 bg-black/40"
            />
            <div className="mt-4 grid grid-cols-3 gap-2 text-center font-mono text-xs">
              <div className="bg-space-950/70 p-2.5 rounded-xl border border-white/5">
                <span className="text-slate-400 block text-[10px]">WAVELENGTH</span>
                <span className="text-cyan-300 font-bold">279.6 nm</span>
              </div>
              <div className="bg-space-950/70 p-2.5 rounded-xl border border-white/5">
                <span className="text-slate-400 block text-[10px]">ORBIT POINT</span>
                <span className="text-white font-bold">Halo L1</span>
              </div>
              <div className="bg-space-950/70 p-2.5 rounded-xl border border-white/5">
                <span className="text-slate-400 block text-[10px]">SAMPLING</span>
                <span className="text-amber-300 font-bold">3.0 Hours</span>
              </div>
            </div>
          </GlowingCard>
        </div>
      </div>
    </div>
  );
};
