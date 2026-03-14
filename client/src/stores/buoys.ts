import { create } from "zustand";

export interface BuoyRecord {
  latitude: number;
  longitude: number;
  status: string;
  buoyId: string;
  temperature: number;
  depth: number;
  density: number;
  salinity: number;
}

export interface GroupedBuoy {
  buoyId: string;
  latitude: number;
  longitude: number;
  status: string;
  yearly: {
    [year: string]: BuoyRecord[];
  };
}

interface Filters {
  minTemp: number;
  maxTemp: number;
  minSalinity: number;
  maxSalinity: number;
  minDensity: number;
  maxDensity: number;
}

interface StoreState {
  selectedBuoy: GroupedBuoy | null;
  setSelectedBuoy: (buoy: GroupedBuoy | null) => void;

  filters: Filters;
  setFilters: (filters: Partial<Filters>) => void;
}

export const useBuoyStore = create<StoreState>((set) => ({
  selectedBuoy: null,
  setSelectedBuoy: (buoy) => set({ selectedBuoy: buoy }),

  filters: {
    minTemp: 0,
    maxTemp: 40,
    minSalinity: 30,
    maxSalinity: 40,
    minDensity: 0,
    maxDensity: 1100,
  },
  setFilters: (filters) =>
    set((state) => ({
      filters: { ...state.filters, ...filters },
    })),
}));