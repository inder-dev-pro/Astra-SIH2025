import { GroupedBuoy } from "@/stores/buoys";
import data_2001 from "@/assets/data/final_output_2001.json";
import data_2002 from "@/assets/data/final_output_2002.json";
import data_2003 from "@/assets/data/final_output_2003.json";
import data_2004 from "@/assets/data/final_output_2004.json";
import data_2005 from "@/assets/data/final_output_2005.json";
import data_2006 from "@/assets/data/final_output_2006.json";
import data_2007 from "@/assets/data/final_output_2007.json";
import data_2008 from "@/assets/data/final_output_2008.json";
import data_2009 from "@/assets/data/final_output_2009.json";
import data_2010 from "@/assets/data/final_output_2010.json";
import data_2011 from "@/assets/data/final_output_2011.json";
import data_2012 from "@/assets/data/final_output_2012.json";
import data_2013 from "@/assets/data/final_output_2013.json";
import data_2014 from "@/assets/data/final_output_2014.json";
import data_2015 from "@/assets/data/final_output_2015.json";
import data_2016 from "@/assets/data/final_output_2016.json";
import data_2017 from "@/assets/data/final_output_2017.json";

type BuoyRecord = {
  latitude: number;
  longitude: number;
  status: string;
  buoyId: string;
  temperature: number;
  depth: number;
  density: number;
  salinity: number;
};

type GroupedBuoyData = {
  [buoyId: string]: GroupedBuoy;
};

export function groupBuoyData(dataByYear: { [year: string]: BuoyRecord[] }) {
  const grouped: GroupedBuoyData = {};

  for (const [year, records] of Object.entries(dataByYear)) {
    records.forEach((record) => {
      if (!grouped[record.buoyId]) {
        grouped[record.buoyId] = {
          buoyId: record.buoyId,
          latitude: record.latitude,
          longitude: record.longitude,
          status: record.status,
          yearly: {},
        };
      }

      if (!grouped[record.buoyId].yearly[year]) {
        grouped[record.buoyId].yearly[year] = [];
      }

      grouped[record.buoyId].yearly[year].push(record);
    });
  }

  return grouped;
}

// create groupedData from 2001 to 2017
export const groupedData = groupBuoyData({
  "2001": data_2001,
  "2002": data_2002,
  "2003": data_2003,
  "2004": data_2004,
  "2005": data_2005,
  "2006": data_2006,
  "2007": data_2007,
  "2008": data_2008,
  "2009": data_2009,
  "2010": data_2010,
  "2011": data_2011,
  "2012": data_2012,
  "2013": data_2013,
  "2014": data_2014,
  "2015": data_2015,
  "2016": data_2016,
  "2017": data_2017,
});
