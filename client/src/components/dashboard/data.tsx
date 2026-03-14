"use client";
import { useState } from "react";
import { useBuoyStore } from "@/stores/buoys";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

export default function Data() {
  const { selectedBuoy } = useBuoyStore();
  const [activeTab, setActiveTab] = useState<
    "temperature" | "salinity" | "density" | "depth"
  >("temperature");

  if (!selectedBuoy) {
    return (
      <div className="border-t border-slate-700/50 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-sm text-slate-400 h-[30%] flex flex-col items-center justify-center relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-violet-500/5 via-purple-500/5 to-fuchsia-500/5"></div>
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-violet-600/10 via-transparent to-transparent"></div>
        
        <div className="relative z-10 text-center">
          <div className="w-20 h-20 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-slate-800/80 to-slate-700/80 border border-slate-600/30 flex items-center justify-center backdrop-blur-sm shadow-2xl">
            <span className="text-3xl">📊</span>
          </div>
          <p className="text-xl font-bold text-slate-200 mb-3 tracking-wide">
            No Buoy Selected
          </p>
          <p className="text-slate-400 font-medium">
            Click on a buoy marker to view detailed analytics
          </p>
        </div>
      </div>
    );
  }

  const metrics = ["temperature", "salinity", "density", "depth"] as const;
  const metricConfig = {
    temperature: {
      icon: "🌡️",
      color: "from-orange-400 via-red-400 to-pink-400",
      bgColor: "bg-gradient-to-br from-orange-500/15 to-red-500/15",
      borderColor: "border-orange-400/30",
      shadowColor: "shadow-orange-500/20",
    },
    salinity: {
      icon: "🧂",
      color: "from-indigo-400 via-purple-400 to-violet-400",
      bgColor: "bg-gradient-to-br from-indigo-500/15 to-purple-500/15",
      borderColor: "border-indigo-400/30",
      shadowColor: "shadow-indigo-500/20",
    },
    density: {
      icon: "⚖️",
      color: "from-emerald-400 via-teal-400 to-cyan-400",
      bgColor: "bg-gradient-to-br from-emerald-500/15 to-teal-500/15",
      borderColor: "border-emerald-400/30",
      shadowColor: "shadow-emerald-500/20",
    },
    depth: {
      icon: "🌊",
      color: "from-amber-400 via-yellow-400 to-orange-400",
      bgColor: "bg-gradient-to-br from-amber-500/15 to-yellow-500/15",
      borderColor: "border-amber-400/30",
      shadowColor: "shadow-amber-500/20",
    },
  };

  const labels = Object.entries(selectedBuoy.yearly).flatMap(
    ([year, records]) => records.map((_, idx) => `${year}-${idx + 1}`)
  );

  const chartDatasets = metrics.map((metric) => {
    let borderColor = "";
    let backgroundColor = "";
    switch (metric) {
      case "temperature":
        borderColor = "rgba(251, 146, 60, 1)";
        backgroundColor = "rgba(251, 146, 60, 0.15)";
        break;
      case "salinity":
        borderColor = "rgba(129, 140, 248, 1)";
        backgroundColor = "rgba(129, 140, 248, 0.15)";
        break;
      case "density":
        borderColor = "rgba(52, 211, 153, 1)";
        backgroundColor = "rgba(52, 211, 153, 0.15)";
        break;
      case "depth":
        borderColor = "rgba(251, 191, 36, 1)";
        backgroundColor = "rgba(251, 191, 36, 0.15)";
        break;
    }
    return {
      label: metric.charAt(0).toUpperCase() + metric.slice(1),
      data: Object.entries(selectedBuoy.yearly).flatMap(([_, records]) =>
        records.map((r) => r[metric])
      ),
      borderColor,
      backgroundColor,
      tension: 0.4,
      borderWidth: 3,
      pointRadius: 5,
      pointHoverRadius: 8,
      pointBackgroundColor: borderColor,
      pointBorderColor: "#0f172a",
      pointBorderWidth: 3,
      fill: true,
    };
  });

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: "rgba(2, 6, 23, 0.95)",
        titleColor: "#f8fafc",
        bodyColor: "#e2e8f0",
        borderColor: "#475569",
        borderWidth: 1,
        cornerRadius: 12,
        padding: 16,
        displayColors: false,
        titleFont: { size: 14, weight: 600 },
        bodyFont: { size: 13 },
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: "Time (record index per year)",
          color: "#94a3b8",
          font: { size: 13, weight: 500 },
        },
        grid: { 
          color: "rgba(148, 163, 184, 0.08)",
          lineWidth: 1,
        },
        ticks: { 
          color: "#64748b", 
          font: { size: 11 },
          padding: 8,
        },
      },
      y: {
        title: {
          display: true,
          text: "Value",
          color: "#94a3b8",
          font: { size: 13, weight: 500 },
        },
        grid: { 
          color: "rgba(148, 163, 184, 0.08)",
          lineWidth: 1,
        },
        ticks: { 
          color: "#64748b", 
          font: { size: 11 },
          padding: 8,
        },
      },
    },
  };

  return (
    <div className="border-t border-slate-700/50 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 h-[30%] flex relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-r from-violet-500/5 via-purple-500/5 to-fuchsia-500/5"></div>
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_bottom_right,_var(--tw-gradient-stops))] from-indigo-600/10 via-transparent to-transparent"></div>

      <div className="relative z-10 w-1/5 p-4 border-r border-slate-700/50 bg-slate-900/40 backdrop-blur-sm">
        <div className="mb-6">
          <h2 className="text-2xl font-bold bg-gradient-to-r from-violet-400 via-purple-400 to-fuchsia-400 bg-clip-text text-transparent mb-3 tracking-wide">
            Argo Data
          </h2>
          <div className="w-16 h-1 bg-gradient-to-r from-violet-400 via-purple-400 to-fuchsia-400 rounded-full mb-4 shadow-lg shadow-violet-500/25"></div>
        </div>

        <div className="space-y-3 text-sm">
          <div className="flex justify-between items-center py-2 px-3 rounded-lg bg-slate-800/30 backdrop-blur-sm border border-slate-600/20">
            <span className="text-slate-300 font-semibold">ID:</span>
            <span className="text-white font-mono bg-slate-700/50 px-2 py-1 rounded text-xs">
              {selectedBuoy.buoyId}
            </span>
          </div>
          <div className="flex justify-between items-center py-2 px-3 rounded-lg bg-slate-800/30 backdrop-blur-sm border border-slate-600/20">
            <span className="text-slate-300 font-semibold">Status:</span>
            <span
              className={`px-3 py-1 rounded-full text-xs font-bold tracking-wide ${
                selectedBuoy.status === "active"
                  ? "bg-gradient-to-r from-emerald-500/20 to-green-500/20 text-emerald-300 border border-emerald-400/40 shadow-lg shadow-emerald-500/20"
                  : "bg-gradient-to-r from-slate-500/20 to-gray-500/20 text-slate-300 border border-slate-400/40"
              }`}
            >
              {selectedBuoy.status.toUpperCase()}
            </span>
          </div>
          <div className="flex justify-between items-center py-2 px-3 rounded-lg bg-slate-800/30 backdrop-blur-sm border border-slate-600/20">
            <span className="text-slate-300 font-semibold">Latitude:</span>
            <span className="text-white font-mono text-xs">
              {selectedBuoy.latitude.toFixed(4)}°
            </span>
          </div>
          <div className="flex justify-between items-center py-2 px-3 rounded-lg bg-slate-800/30 backdrop-blur-sm border border-slate-600/20">
            <span className="text-slate-300 font-semibold">Longitude:</span>
            <span className="text-white font-mono text-xs">
              {selectedBuoy.longitude.toFixed(4)}°
            </span>
          </div>
        </div>
      </div>

      <div className="relative z-10 flex-1 flex flex-col w-[70%] p-4 pb-2">
        <div className="h-full bg-gradient-to-br from-slate-900/60 to-slate-800/60 backdrop-blur-sm rounded-2xl border border-slate-600/30 p-4 shadow-2xl shadow-slate-900/50">
          <Line
            data={{
              labels,
              datasets: [chartDatasets[metrics.indexOf(activeTab)]],
            }}
            options={{
              ...options,
              plugins: {
                ...options.plugins,
                title: {
                  display: true,
                  text: `${
                    activeTab.charAt(0).toUpperCase() + activeTab.slice(1)
                  } Analytics`,
                  color: "#f8fafc",
                  font: { size: 18, weight: 700 },
                  padding: { bottom: 20 },
                },
              },
            }}
          />
        </div>
      </div>

      <div className="relative z-10 flex flex-col w-[10%] border-l border-slate-700/50">
        {metrics.map((metric, index) => (
          <button
            key={metric}
            className={`h-1/4 text-xs font-bold transition-all duration-500 relative group overflow-hidden ${
              activeTab === metric
                ? `${metricConfig[metric].bgColor} text-white border-l-4 ${metricConfig[metric].borderColor} shadow-xl ${metricConfig[metric].shadowColor}`
                : "text-slate-400 hover:text-white hover:bg-gradient-to-r hover:from-slate-700/30 hover:to-slate-600/30"
            }`}
            onClick={() => setActiveTab(metric)}
          >
            <div className={`absolute inset-0 bg-gradient-to-r ${metricConfig[metric].color} opacity-0 group-hover:opacity-10 transition-all duration-500`}></div>
            <div className="relative z-10 flex items-center justify-start h-full px-3 space-x-2">
              <span className={`text-xl transition-transform duration-300 ${activeTab === metric ? 'scale-110' : 'group-hover:scale-105'}`}>
                {metricConfig[metric].icon}
              </span>
              <span className="capitalize leading-tight tracking-wide">
                {metric}
              </span>
            </div>
            {activeTab === metric && (
              <div
                className={`absolute left-0 top-0 w-1 h-full bg-gradient-to-b ${metricConfig[metric].color} shadow-lg`}
              ></div>
            )}
            <div className={`absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 bg-gradient-to-r ${metricConfig[metric].color} blur-xl scale-150 -z-10`}></div>
          </button>
        ))}
      </div>
    </div>
  );
}
