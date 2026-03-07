"use client";

import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import { useBuoyStore } from "@/stores/buoys";
import { groupedData } from "@/utils/groupBuoyData";

// Custom marker icons with modern colors
const activeIcon = L.divIcon({
  html: `
    <div class="relative">
      <div class="absolute -inset-2 bg-gradient-to-r from-orange-400 to-red-400 rounded-full blur-sm opacity-60 animate-pulse"></div>
      <div class="relative w-6 h-6 bg-gradient-to-br from-orange-500 to-red-500 rounded-full border-2 border-white shadow-lg flex items-center justify-center">
        <div class="w-2 h-2 bg-white rounded-full"></div>
      </div>
    </div>
  `,
  className: 'custom-marker',
  iconSize: [24, 24],
  iconAnchor: [12, 12],
});

const inactiveIcon = L.divIcon({
  html: `
    <div class="relative">
      <div class="absolute -inset-2 bg-gradient-to-r from-slate-400 to-slate-500 rounded-full blur-sm opacity-40"></div>
      <div class="relative w-6 h-6 bg-gradient-to-br from-slate-500 to-slate-600 rounded-full border-2 border-white shadow-lg flex items-center justify-center">
        <div class="w-2 h-2 bg-white rounded-full opacity-60"></div>
      </div>
    </div>
  `,
  className: 'custom-marker',
  iconSize: [24, 24],
  iconAnchor: [12, 12],
});

export default function Map() {
  const { setSelectedBuoy, filters } = useBuoyStore();

  const filteredBuoys = Object.values(groupedData).filter((b) => {
    const latestYear = Math.max(...Object.keys(b.yearly).map(Number));
    const latestRecords = b.yearly[latestYear];

    const record = latestRecords[0];

    return (
      record.temperature >= filters.minTemp &&
      record.temperature <= filters.maxTemp &&
      record.salinity >= filters.minSalinity &&
      record.salinity <= filters.maxSalinity &&
      record.density >= filters.minDensity &&
      record.density <= filters.maxDensity
    );
  });

  return (
    <div className="h-[70%] relative overflow-hidden rounded-t-none">
      {/* Map overlay with modern styling */}
      <div className="absolute top-4 left-4 z-[1000] bg-slate-900/90 backdrop-blur-md rounded-xl p-4 border border-slate-700/50 shadow-2xl">
        <div className="flex items-center space-x-3 mb-3">
          <div className="w-8 h-8 bg-gradient-to-br from-violet-500/20 to-purple-500/20 rounded-lg flex items-center justify-center border border-violet-400/30">
            <span className="text-sm">🗺️</span>
          </div>
          <h3 className="text-sm font-bold text-white tracking-wide">Ocean Buoys</h3>
        </div>
        <div className="space-y-2 text-xs">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-gradient-to-br from-orange-500 to-red-500 rounded-full border border-white shadow-sm"></div>
            <span className="text-slate-300 font-medium">Active ({filteredBuoys.filter(b => b.status === 'active').length})</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-gradient-to-br from-slate-500 to-slate-600 rounded-full border border-white shadow-sm"></div>
            <span className="text-slate-300 font-medium">Inactive ({filteredBuoys.filter(b => b.status !== 'active').length})</span>
          </div>
        </div>
      </div>

      {/* Stats overlay */}
      <div className="absolute top-4 right-4 z-[1000] bg-slate-900/90 backdrop-blur-md rounded-xl p-4 border border-slate-700/50 shadow-2xl">
        <div className="text-center">
          <div className="text-2xl font-black text-white mb-1">{filteredBuoys.length}</div>
          <div className="text-xs text-slate-400 font-medium tracking-wide uppercase">Total Buoys</div>
        </div>
      </div>

      <MapContainer
        center={[10, 80]}
        zoom={6}
        style={{ height: "100%", width: "100%" }}
        className="rounded-none"
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {filteredBuoys.map((b) => (
          <Marker
            key={b.latitude + "-" + b.longitude}
            position={[b.latitude, b.longitude]}
            eventHandlers={{
              click: () => setSelectedBuoy(b),
            }}
            icon={b.status === 'active' ? activeIcon : inactiveIcon}
          >
            <Popup className="custom-popup">
              <div className="bg-slate-900 text-white p-4 rounded-lg border border-slate-700 min-w-[200px]">
                <div className="flex items-center space-x-2 mb-3">
                  <div className={`w-3 h-3 rounded-full ${
                    b.status === 'active' 
                      ? 'bg-gradient-to-r from-orange-400 to-red-400' 
                      : 'bg-gradient-to-r from-slate-400 to-slate-500'
                  }`}></div>
                  <span className="text-sm font-bold text-slate-200">
                    Buoy {Object.keys(groupedData).find(id => groupedData[id].latitude === b.latitude)}
                  </span>
                </div>
                
                <div className="space-y-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Status:</span>
                    <span className={`font-medium ${
                      b.status === 'active' ? 'text-orange-400' : 'text-slate-400'
                    }`}>
                      {b.status}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Coordinates:</span>
                    <span className="text-slate-200 font-mono">
                      {b.latitude.toFixed(3)}, {b.longitude.toFixed(3)}
                    </span>
                  </div>
                </div>
                
                <div className="mt-3 pt-3 border-t border-slate-700">
                  <button 
                    onClick={() => setSelectedBuoy(b)}
                    className="w-full bg-gradient-to-r from-violet-500 to-purple-500 hover:from-violet-600 hover:to-purple-600 text-white text-xs font-medium py-2 px-3 rounded-lg transition-all duration-200 shadow-lg hover:shadow-violet-500/25"
                  >
                    View Details
                  </button>
                </div>
              </div>
            </Popup>
          </Marker>
        ))}
      </MapContainer>
      
      <style jsx global>{`
        .custom-popup .leaflet-popup-content-wrapper {
          background: transparent !important;
          box-shadow: none !important;
          border-radius: 12px !important;
          padding: 0 !important;
        }
        
        .custom-popup .leaflet-popup-content {
          margin: 0 !important;
          border-radius: 12px !important;
        }
        
        .custom-popup .leaflet-popup-tip {
          background: rgb(15 23 42) !important;
          border: 1px solid rgb(51 65 85) !important;
        }
        
        .custom-marker {
          background: transparent !important;
          border: none !important;
        }
        
        .leaflet-container {
          background: rgb(2 6 23) !important;
        }
      `}</style>
    </div>
  );
}
