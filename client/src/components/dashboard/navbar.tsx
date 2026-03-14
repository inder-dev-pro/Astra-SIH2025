import logo from "@/assets/logo.png";
import Image from "next/image";
import Link from "next/link";
import github from "@/assets/github.png";

export default function Navbar() {
  return (
    <header className="border-b border-slate-700/50 bg-gradient-to-r from-slate-950 via-slate-900 to-slate-950 backdrop-blur-xl flex justify-between items-center h-[9dvh] relative overflow-hidden">
      {/* Advanced background effects */}
      <div className="absolute inset-0 bg-gradient-to-r from-violet-500/8 via-purple-500/6 to-fuchsia-500/8"></div>
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-violet-600/15 via-transparent to-transparent"></div>
      
      {/* Animated particles */}
      <div className="absolute top-0 left-1/4 w-32 h-32 bg-violet-500/8 rounded-full blur-3xl animate-pulse"></div>
      <div 
        className="absolute bottom-0 right-1/3 w-24 h-24 bg-purple-500/8 rounded-full blur-2xl animate-pulse" 
        style={{ animationDelay: "1s" }}
      ></div>
      <div 
        className="absolute top-1/2 left-1/2 w-16 h-16 bg-fuchsia-500/6 rounded-full blur-xl animate-pulse" 
        style={{ animationDelay: "2s" }}
      ></div>

      <section className="relative z-10 flex items-center ml-6 group">
        <div className="relative mr-4 transform transition-all duration-500 group-hover:scale-110 group-hover:rotate-3">
          <div className="absolute inset-0 bg-gradient-to-r from-violet-400 via-purple-400 to-fuchsia-400 rounded-2xl blur-xl opacity-20 group-hover:opacity-50 transition-all duration-500 scale-150"></div>
          <div className="relative z-10 w-12 h-12 bg-gradient-to-br from-slate-800/80 to-slate-700/80 rounded-2xl border border-slate-600/50 backdrop-blur-sm shadow-2xl flex items-center justify-center group-hover:shadow-violet-500/25">
            <Image
              src={logo}
              alt="FloatChat Logo"
              height={28}
              className="drop-shadow-lg filter brightness-110"
            />
          </div>
        </div>
        <div className="text-3xl font-black bg-gradient-to-r from-violet-400 via-purple-400 to-fuchsia-400 bg-clip-text text-transparent tracking-tight group-hover:tracking-wide transition-all duration-300">
          FloatChat
        </div>
        <div className="ml-2 px-2 py-1 bg-gradient-to-r from-violet-500/10 to-purple-500/10 rounded-lg border border-violet-400/20 text-xs text-violet-300 font-medium">
          v2.0
        </div>
      </section>

      <section className="relative z-10 flex items-center mr-6 space-x-3">
        <nav className="flex items-center space-x-1">
          {[
            { href: "/", label: "Dashboard", icon: "📊" },
            { href: "/chat", label: "Chat", icon: "💬" },
            { href: "/aboutus", label: "About Us", icon: "👥" },
          ].map((item, index) => (
            <Link
              key={item.href}
              href={item.href}
              className="group relative px-5 py-3 text-slate-300 hover:text-white transition-all duration-500 rounded-xl hover:bg-gradient-to-r hover:from-slate-800/60 hover:to-slate-700/60 border border-transparent hover:border-slate-600/40 backdrop-blur-sm transform hover:scale-105"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <div className="flex items-center space-x-2">
                <span className="text-sm opacity-70 group-hover:opacity-100 transition-all duration-300 transform group-hover:scale-110">
                  {item.icon}
                </span>
                <span className="text-sm font-bold relative tracking-wide">
                  {item.label}
                  <div className="absolute -bottom-1 left-0 w-0 h-0.5 bg-gradient-to-r from-violet-400 via-purple-400 to-fuchsia-400 group-hover:w-full transition-all duration-500 rounded-full shadow-lg shadow-violet-500/50"></div>
                </span>
              </div>
              {/* Hover glow effect */}
              <div className="absolute inset-0 bg-gradient-to-r from-violet-500/10 via-purple-500/10 to-fuchsia-500/10 opacity-0 group-hover:opacity-100 rounded-xl transition-all duration-500 -z-10 blur-sm"></div>
            </Link>
          ))}
        </nav>

        <div className="w-px h-8 bg-gradient-to-b from-transparent via-slate-500/50 to-transparent mx-3"></div>

        <a
          className="group relative p-3 rounded-xl hover:bg-gradient-to-r hover:from-slate-800/60 hover:to-slate-700/60 border border-transparent hover:border-slate-600/40 transition-all duration-500 transform hover:scale-110"
          href="https://github.com/inder-dev-pro/Astra-SIH2025"
          target="_blank"
          rel="noopener noreferrer"
        >
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-slate-400 to-slate-300 rounded-xl blur-lg opacity-0 group-hover:opacity-30 transition-all duration-500 scale-150"></div>
            <div className="relative z-10 w-8 h-8 bg-gradient-to-br from-slate-700/80 to-slate-600/80 rounded-lg border border-slate-500/30 backdrop-blur-sm shadow-lg flex items-center justify-center group-hover:shadow-slate-400/25">
              <Image
                src={github}
                alt="GitHub Repository"
                height={20}
                className="transition-all duration-500 group-hover:brightness-125 filter"
              />
            </div>
          </div>
          <div className="absolute -top-10 left-1/2 transform -translate-x-1/2 bg-slate-800/90 backdrop-blur-sm text-white text-xs px-3 py-2 rounded-lg opacity-0 group-hover:opacity-100 transition-all duration-300 pointer-events-none whitespace-nowrap border border-slate-600/50 shadow-xl">
            View Source Code
            <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-4 border-r-4 border-t-4 border-transparent border-t-slate-800/90"></div>
          </div>
        </a>

        {/* Status indicator */}
        <div className="flex items-center space-x-2 px-3 py-2 bg-gradient-to-r from-emerald-500/10 to-green-500/10 rounded-lg border border-emerald-400/20">
          <div className="w-2 h-2 bg-gradient-to-r from-emerald-400 to-green-400 rounded-full animate-pulse shadow-lg shadow-emerald-500/50"></div>
          <span className="text-xs text-emerald-300 font-bold tracking-wide">LIVE</span>
        </div>
      </section>
    </header>
  );
}
