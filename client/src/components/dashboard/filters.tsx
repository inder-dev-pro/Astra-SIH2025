"use client";
import { useBuoyStore } from "@/stores/buoys";

export default function Filters() {
  const { filters, setFilters } = useBuoyStore();
  const metrics = [
    {
      key: "Temperature",
      min: "minTemp",
      max: "maxTemp",
      unit: "°C",
      color: "from-orange-400 via-red-400 to-pink-400",
      bgColor: "from-orange-500/10 to-red-500/10",
      icon: "🌡️",
    },
    {
      key: "Salinity",
      min: "minSalinity",
      max: "maxSalinity",
      unit: "PSU",
      color: "from-indigo-400 via-purple-400 to-violet-400",
      bgColor: "from-indigo-500/10 to-purple-500/10",
      icon: "🧂",
    },
    {
      key: "Density",
      min: "minDensity",
      max: "maxDensity",
      unit: "kg/m³",
      color: "from-emerald-400 via-teal-400 to-cyan-400",
      bgColor: "from-emerald-500/10 to-teal-500/10",
      icon: "⚖️",
    },
  ] as const;

  return (
    <section className="w-1/5 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 border-r border-slate-700/50 h-full overflow-y-auto backdrop-blur-xl relative">
      {/* Background Effects */}
      <div className="absolute inset-0 bg-gradient-to-b from-violet-500/5 via-purple-500/5 to-fuchsia-500/5"></div>
      <div className="absolute top-0 right-0 w-32 h-32 bg-violet-500/10 rounded-full blur-3xl animate-pulse"></div>
      <div className="absolute bottom-0 left-0 w-24 h-24 bg-purple-500/10 rounded-full blur-2xl animate-pulse" style={{ animationDelay: "2s" }}></div>
      
      <div className="relative z-10 p-6">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-violet-500/20 to-purple-500/20 rounded-2xl border border-violet-400/30 backdrop-blur-sm shadow-2xl shadow-violet-500/20 mb-4">
            <span className="text-2xl">🎛️</span>
          </div>
          <h2 className="text-3xl font-black bg-gradient-to-r from-violet-400 via-purple-400 to-fuchsia-400 bg-clip-text text-transparent mb-3 tracking-wide">
            FILTERS
          </h2>
          <div className="w-20 h-1 bg-gradient-to-r from-violet-400 via-purple-400 to-fuchsia-400 mx-auto rounded-full shadow-lg shadow-violet-500/25"></div>
        </div>

        <div className="space-y-6">
          {metrics.map((metric, index) => (
            <div
              key={metric.key}
              className="relative group animate-fadeIn"
              style={{ animationDelay: `${index * 150}ms` }}
            >
              <div className={`bg-gradient-to-br ${metric.bgColor} backdrop-blur-sm rounded-2xl p-5 border border-slate-600/30 hover:border-slate-500/50 transition-all duration-500 hover:shadow-2xl hover:shadow-slate-900/30 relative overflow-hidden`}>
                {/* Glow effect */}
                <div className={`absolute inset-0 bg-gradient-to-r ${metric.color} opacity-0 group-hover:opacity-5 rounded-2xl transition-opacity duration-500`}></div>
                
                <div className="flex items-center space-x-3 mb-4 relative z-10">
                  <div className="w-10 h-10 bg-slate-800/60 rounded-xl flex items-center justify-center border border-slate-600/40 shadow-lg">
                    <span className="text-lg">{metric.icon}</span>
                  </div>
                  <div>
                    <span
                      className={`text-lg font-bold bg-gradient-to-r ${metric.color} bg-clip-text text-transparent tracking-wide`}
                    >
                      {metric.key}
                    </span>
                    <div className="text-xs text-slate-400 font-medium">{metric.unit}</div>
                  </div>
                </div>

                <div className="space-y-4 relative z-10">
                  <div className="relative">
                    <label className="block text-xs text-slate-300 mb-2 font-bold tracking-wide uppercase">
                      Minimum Value
                    </label>
                    <div className="relative">
                      <input
                        type="number"
                        value={filters[metric.min as keyof typeof filters] || 0}
                        onChange={(e) =>
                          setFilters({ [metric.min]: Number(e.target.value) })
                        }
                        className="w-full rounded-xl p-3 text-white bg-slate-800/60 border border-slate-600/40 focus:border-slate-400/60 focus:ring-2 focus:ring-violet-500/20 transition-all duration-300 text-sm placeholder-slate-500 backdrop-blur-sm shadow-inner"
                        placeholder="Min"
                      />
                      <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-transparent via-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"></div>
                    </div>
                  </div>

                  <div className="relative">
                    <label className="block text-xs text-slate-300 mb-2 font-bold tracking-wide uppercase">
                      Maximum Value
                    </label>
                    <div className="relative">
                      <input
                        type="number"
                        value={filters[metric.max as keyof typeof filters] || 0}
                        onChange={(e) =>
                          setFilters({ [metric.max]: Number(e.target.value) })
                        }
                        className="w-full rounded-xl p-3 text-white bg-slate-800/60 border border-slate-600/40 focus:border-slate-400/60 focus:ring-2 focus:ring-violet-500/20 transition-all duration-300 text-sm placeholder-slate-500 backdrop-blur-sm shadow-inner"
                        placeholder="Max"
                      />
                      <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-transparent via-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"></div>
                    </div>
                  </div>
                </div>

                {/* Background shine effect */}
                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-white/20 to-transparent rounded-t-2xl"></div>
              </div>
            </div>
          ))}
        </div>

        {/* Bottom decoration */}
        <div className="mt-8 text-center">
          <div className="inline-flex items-center justify-center space-x-2 text-xs text-slate-500 font-medium">
            <div className="w-2 h-2 bg-gradient-to-r from-violet-400 to-purple-400 rounded-full"></div>
            <span>Real-time Ocean Data</span>
            <div className="w-2 h-2 bg-gradient-to-r from-purple-400 to-fuchsia-400 rounded-full"></div>
          </div>
        </div>
      </div>
    </section>
  );
}
