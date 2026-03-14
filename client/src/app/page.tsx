import Data from "@/components/dashboard/data";
import Filters from "@/components/dashboard/filters";
import Map from "@/components/dashboard/map";

export default function Home() {
  return (
    <main className="flex h-[91dvh]">
      <Filters />
      <section className="w-4/5 flex flex-col">
        <Map />
        <Data />
      </section>
    </main>
  );
}