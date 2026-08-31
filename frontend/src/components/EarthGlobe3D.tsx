import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { ShieldAlert, Navigation, Users, Radio } from "lucide-react";

interface EarthGlobe3DProps {
  isCritical?: boolean;
  className?: string;
}

export const EarthGlobe3D = ({
  isCritical = false,
  className = "",
}: EarthGlobe3DProps) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const [selectedOrbit, setSelectedOrbit] = useState<"all" | "isro" | "navic" | "gaganyaan" | "shield">("all");
  const [orbitStats, setOrbitStats] = useState({
    altitude: "36,000 km (GSO)",
    satCount: "7 Satellites Active",
    shieldStatus: isCritical ? "Compressed (6.2 Re Bow Shock)" : "Nominal (10.5 Re Bow Shock)",
    radiationDose: isCritical ? "142 mSv/hr (Elevated)" : "2.4 mSv/day (Safe)",
  });

  const handleOrbitSelect = (orbit: "all" | "isro" | "navic" | "gaganyaan" | "shield") => {
    setSelectedOrbit(orbit);
    if (orbit === "isro") {
      setOrbitStats({
        altitude: "Ground Station (Hassan / Bengaluru)",
        satCount: "Master Control Facility (MCF)",
        shieldStatus: "Coordinates: 21°N, 78°E (India)",
        radiationDose: "Active Telemetry Uplink (Aditya-L1 / NavIC)",
      });
    } else if (orbit === "navic") {
      setOrbitStats({
        altitude: "36,000 km (GSO Nodes)",
        satCount: "7 Satellites (3 Geostationary, 4 Inclined)",
        shieldStatus: isCritical ? "Ionospheric Delay High" : "Standard Ephemeris Margin",
        radiationDose: isCritical ? "Payload Safeguard Armed" : "Nominal Telemetry",
      });
    } else if (orbit === "gaganyaan") {
      setOrbitStats({
        altitude: "400 km (Low Earth Orbit)",
        satCount: "Gaganyaan Crew Module",
        shieldStatus: isCritical ? "EVA Inhibit (S3 Proton Flux)" : "EVA Permitted (Nominal)",
        radiationDose: isCritical ? "Storm Shelter Armed" : "Safe Baseline (2.4 mSv/day)",
      });
    } else {
      setOrbitStats({
        altitude: "Magnetosphere (6 - 10.5 Re)",
        satCount: "Earth Magnetic Shield",
        shieldStatus: isCritical ? "Compressed (6.2 Re Bow Shock)" : "Nominal (10.5 Re Bow Shock)",
        radiationDose: isCritical ? "142 mSv/hr (Elevated)" : "2.4 mSv/day (Safe)",
      });
    }
  };

  useEffect(() => {
    setOrbitStats((prev) => ({
      ...prev,
      shieldStatus: isCritical ? "Compressed (6.2 Re Bow Shock)" : "Nominal (10.5 Re Bow Shock)",
      radiationDose: isCritical ? "142 mSv/hr (Elevated)" : "2.4 mSv/day (Safe)",
    }));
  }, [isCritical]);

  useEffect(() => {
    const currentMount = mountRef.current;
    if (!currentMount) return;

    const width = currentMount.clientWidth || 460;
    const height = currentMount.clientHeight || 460;

    // Scene, Camera, Renderer
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(0, 0.15, 7.6);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    currentMount.appendChild(renderer.domElement);

    const globeGroup = new THREE.Group();
    globeGroup.position.set(0, 0.1, 0);
    scene.add(globeGroup);

    // 1. Earth Core Sphere (Scaled to fit perfectly centered in viewport)
    const earthRadius = 1.6;
    const earthGeo = new THREE.SphereGeometry(earthRadius, 48, 48);
    const earthMat = new THREE.MeshBasicMaterial({
      color: 0x06122c,
      wireframe: true,
      transparent: true,
      opacity: 0.45,
    });
    const earthMesh = new THREE.Mesh(earthGeo, earthMat);
    globeGroup.add(earthMesh);

    // Solid inner core
    const innerGeo = new THREE.SphereGeometry(earthRadius * 0.985, 36, 36);
    const innerMat = new THREE.MeshBasicMaterial({ color: 0x020716 });
    const innerMesh = new THREE.Mesh(innerGeo, innerMat);
    globeGroup.add(innerMesh);

    // 2. Latitude & Longitude Coordinate Lines (Neutral Electric Cyan)
    const gridMat = new THREE.LineBasicMaterial({
      color: 0x00e5ff,
      transparent: true,
      opacity: 0.22,
    });

    for (let lat = -60; lat <= 60; lat += 30) {
      const radius = (earthRadius + 0.01) * Math.cos((lat * Math.PI) / 180);
      const y = (earthRadius + 0.01) * Math.sin((lat * Math.PI) / 180);
      const circleGeo = new THREE.BufferGeometry();
      const pts: THREE.Vector3[] = [];
      for (let i = 0; i <= 64; i++) {
        const theta = (i / 64) * Math.PI * 2;
        pts.push(new THREE.Vector3(radius * Math.cos(theta), y, radius * Math.sin(theta)));
      }
      circleGeo.setFromPoints(pts);
      const latLine = new THREE.Line(circleGeo, gridMat);
      globeGroup.add(latLine);
    }

    // 3. India Highlight Marker with Pulsing Ring (ISRO Ground Pin)
    const phi = (90 - 21) * (Math.PI / 180);
    const theta = (78 + 180) * (Math.PI / 180);
    const pinRadius = earthRadius + 0.05;
    const pinX = -(pinRadius * Math.sin(phi) * Math.cos(theta));
    const pinZ = pinRadius * Math.sin(phi) * Math.sin(theta);
    const pinY = pinRadius * Math.cos(phi);

    const pinGeo = new THREE.SphereGeometry(0.08, 16, 16);
    const pinMat = new THREE.MeshBasicMaterial({ color: 0x00e5ff });
    const pinMesh = new THREE.Mesh(pinGeo, pinMat);
    pinMesh.position.set(pinX, pinY, pinZ);
    globeGroup.add(pinMesh);

    // 4. Magnetosphere Bow Shock & Magnetopause Lines (Neutral Cyber Palette: Cyan / Sky-Blue / Silver)
    const shieldRings: THREE.Mesh[] = [];
    const shieldColors = [0x00e5ff, 0x38bdf8, 0x00b0ff, 0x7dd3fc, 0xe2e8f0];
    for (let i = 0; i < 5; i++) {
      const ringGeo = new THREE.TorusGeometry(2.1 + i * 0.25, 0.02, 16, 64);
      const ringMat = new THREE.MeshBasicMaterial({
        color: shieldColors[i % shieldColors.length],
        transparent: true,
        opacity: isCritical ? 0.42 : 0.26,
        wireframe: true,
      });
      const ring = new THREE.Mesh(ringGeo, ringMat);
      ring.rotation.x = Math.PI / 2 + (i - 2) * 0.18;
      ring.rotation.y = (i - 2) * 0.12;
      globeGroup.add(ring);
      shieldRings.push(ring);
    }

    // 5. NavIC GSO Inclined Orbits & Satellites (Crisp Solar Gold / Amber)
    const navicOrbits: THREE.Line[] = [];
    const navicSats: THREE.Mesh[] = [];
    const inclinations = [0.45, -0.45, 0.0];
    const orbitRadius = 2.85;

    inclinations.forEach((inc) => {
      const orbitGeo = new THREE.BufferGeometry();
      const points: THREE.Vector3[] = [];
      for (let i = 0; i <= 64; i++) {
        const a = (i / 64) * Math.PI * 2;
        points.push(
          new THREE.Vector3(
            Math.cos(a) * orbitRadius,
            Math.sin(a) * (orbitRadius * inc),
            Math.sin(a) * orbitRadius
          )
        );
      }
      orbitGeo.setFromPoints(points);
      const orbitLine = new THREE.Line(
        orbitGeo,
        new THREE.LineBasicMaterial({
          color: 0xffb300,
          transparent: true,
          opacity: 0.5,
        })
      );
      globeGroup.add(orbitLine);
      navicOrbits.push(orbitLine);

      // Satellite Box Model
      const sat = new THREE.Mesh(
        new THREE.BoxGeometry(0.12, 0.12, 0.12),
        new THREE.MeshBasicMaterial({ color: 0xffb300 })
      );
      globeGroup.add(sat);
      navicSats.push(sat);
    });

    // 6. Gaganyaan LEO Satellite
    const leoOrbitGeo = new THREE.BufferGeometry();
    const leoPts: THREE.Vector3[] = [];
    const leoR = 1.95;
    for (let i = 0; i <= 64; i++) {
      const a = (i / 64) * Math.PI * 2;
      leoPts.push(new THREE.Vector3(Math.cos(a) * leoR, Math.sin(a) * leoR * 0.6, Math.sin(a) * leoR));
    }
    leoOrbitGeo.setFromPoints(leoPts);
    const leoLine = new THREE.Line(
      leoOrbitGeo,
      new THREE.LineBasicMaterial({ color: 0x00e676, transparent: true, opacity: 0.6 })
    );
    globeGroup.add(leoLine);

    const leoSat = new THREE.Mesh(
      new THREE.SphereGeometry(0.07, 16, 16),
      new THREE.MeshBasicMaterial({ color: 0x00e676 })
    );
    globeGroup.add(leoSat);

    // 7. Interactive Drag and Inertia Physics
    let isDragging = false;
    let prevMouse = { x: 0, y: 0 };
    let velX = 0;
    let velY = 0;

    const onMouseDown = (e: MouseEvent) => {
      isDragging = true;
      prevMouse = { x: e.clientX, y: e.clientY };
    };

    const onMouseMove = (e: MouseEvent) => {
      if (!isDragging) return;
      const dx = e.clientX - prevMouse.x;
      const dy = e.clientY - prevMouse.y;
      velX = dx * 0.005;
      velY = dy * 0.005;
      globeGroup.rotation.y += velX;
      globeGroup.rotation.x += velY;
      prevMouse = { x: e.clientX, y: e.clientY };
    };

    const onMouseUp = () => {
      isDragging = false;
    };

    currentMount.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);

    // Animation Loop
    let angle = 0;
    let animId: number;

    const animate = () => {
      animId = requestAnimationFrame(animate);

      if (!isDragging) {
        velX *= 0.95;
        velY *= 0.95;
        globeGroup.rotation.y += 0.003 + velX;
        globeGroup.rotation.x += velY;
      }

      angle += 0.015;

      // Move NavIC sats
      navicSats.forEach((sat, idx) => {
        const offset = (idx * Math.PI * 2) / 3;
        const inc = inclinations[idx];
        sat.position.x = Math.cos(angle + offset) * orbitRadius;
        sat.position.y = Math.sin(angle + offset) * (orbitRadius * inc);
        sat.position.z = Math.sin(angle + offset) * orbitRadius;
      });

      // Move Gaganyaan LEO Sat
      leoSat.position.x = Math.cos(angle * 2.8) * leoR;
      leoSat.position.y = Math.sin(angle * 2.8) * leoR * 0.6;
      leoSat.position.z = Math.sin(angle * 2.8) * leoR;

      // Flare pulse compression effect
      if (isCritical) {
        const pulse = 1 + Math.sin(Date.now() * 0.009) * 0.08;
        shieldRings.forEach((r, i) => {
          const factor = (1 - i * 0.04) * pulse;
          r.scale.set(factor, factor, factor);
        });
      } else {
        shieldRings.forEach((r) => r.scale.set(1, 1, 1));
      }

      renderer.render(scene, camera);
    };

    animate();

    const handleResize = () => {
      if (!currentMount) return;
      const w = currentMount.clientWidth;
      const h = currentMount.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };

    window.addEventListener("resize", handleResize);

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", handleResize);
      currentMount.removeEventListener("mousedown", onMouseDown);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
      if (currentMount.contains(renderer.domElement)) {
        currentMount.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, [isCritical]);

  return (
    <div className={`relative flex flex-col items-center justify-center ${className}`}>
      {/* 3D Canvas Mounting Point */}
      <div
        ref={mountRef}
        className="w-full h-[300px] sm:h-[340px] md:h-[380px] cursor-grab active:cursor-grabbing select-none flex items-center justify-center"
      />

      {/* Orbit Filter Chips & Live Telemetry Inspector */}
      <div className="w-full mt-2 space-y-2">
        <div className="flex flex-wrap items-center justify-center gap-1.5">
          <button
            onClick={() => handleOrbitSelect("isro")}
            className={`px-2.5 py-1 rounded-lg text-[11px] font-mono flex items-center gap-1.5 border transition-all cursor-pointer ${
              selectedOrbit === "isro"
                ? "bg-cyan-500/25 text-cyan-300 border-cyan-400 shadow-lg shadow-cyan-500/20 font-bold"
                : "bg-space-900/80 text-slate-400 border-white/10 hover:text-white"
            }`}
          >
            <Radio className="h-3 w-3 text-cyan-400 animate-pulse" />
            <span>ISRO Ground Pin</span>
          </button>

          <button
            onClick={() => handleOrbitSelect("navic")}
            className={`px-2.5 py-1 rounded-lg text-[11px] font-mono flex items-center gap-1.5 border transition-all cursor-pointer ${
              selectedOrbit === "navic"
                ? "bg-amber-500/25 text-amber-300 border-amber-400 shadow-lg shadow-amber-500/20 font-bold"
                : "bg-space-900/80 text-slate-400 border-white/10 hover:text-white"
            }`}
          >
            <Navigation className="h-3 w-3 text-amber-400" />
            <span>NavIC GSO</span>
          </button>

          <button
            onClick={() => handleOrbitSelect("gaganyaan")}
            className={`px-2.5 py-1 rounded-lg text-[11px] font-mono flex items-center gap-1.5 border transition-all cursor-pointer ${
              selectedOrbit === "gaganyaan"
                ? "bg-emerald-500/25 text-emerald-300 border-emerald-400 shadow-lg shadow-emerald-500/20 font-bold"
                : "bg-space-900/80 text-slate-400 border-white/10 hover:text-white"
            }`}
          >
            <Users className="h-3 w-3 text-emerald-400" />
            <span>Gaganyaan LEO</span>
          </button>

          <button
            onClick={() => handleOrbitSelect("shield")}
            className={`px-2.5 py-1 rounded-lg text-[11px] font-mono flex items-center gap-1.5 border transition-all cursor-pointer ${
              selectedOrbit === "shield"
                ? isCritical
                  ? "bg-rose-500/25 text-rose-300 border-rose-400 shadow-lg shadow-rose-500/20 font-bold"
                  : "bg-cyan-500/25 text-cyan-300 border-cyan-400"
                : "bg-space-900/80 text-slate-400 border-white/10 hover:text-white"
            }`}
          >
            <ShieldAlert className={`h-3 w-3 ${isCritical ? "text-rose-400 animate-pulse" : "text-cyan-400"}`} />
            <span>{isCritical ? "Shield Shockwave" : "Magnetosphere"}</span>
          </button>
        </div>

        {/* Dynamic Telemetry Box */}
        <div className="bg-space-950/85 border border-white/10 rounded-xl p-2.5 font-mono text-[10px] text-slate-300 grid grid-cols-2 gap-2 shadow-inner">
          <div>
            <span className="text-slate-500 block">TARGET REGION / ASSET</span>
            <span className="text-white font-bold">{orbitStats.altitude}</span>
          </div>
          <div>
            <span className="text-slate-500 block">ASSET TYPE / COUNT</span>
            <span className="text-cyan-300 font-bold">{orbitStats.satCount}</span>
          </div>
          <div className="col-span-2 pt-1 border-t border-white/5 flex items-center justify-between">
            <span className="text-slate-500">OPERATIONAL STATUS:</span>
            <span className={`font-bold ${isCritical && selectedOrbit === "shield" ? "text-rose-400" : "text-emerald-400"}`}>
              {orbitStats.shieldStatus}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};