'use client';
import { useState, FormEvent, useEffect, useRef } from 'react';
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
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

type Role = 'user' | 'assistant';
type Message = { 
  role: Role; 
  content: string; 
  graphData?: any; 
};
const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL;

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([
    { role: 'assistant', content: 'Hi! I\'m FloatChat AI. Ask me anything about ocean data, buoys, or marine analytics.' },
  ]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  async function handleSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!input.trim() || sending) return;

    const userText = input.trim();
    setMessages((m) => [...m, { role: 'user', content: userText }]);
    setInput('');
    setSending(true);

    // Add typing indicator
    setMessages((m) => [...m, { role: 'assistant', content: '…' }]);

    try {
      const res = await fetch(`${backendUrl}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_prompt: userText,
          history: messages,
        }),
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      const { generated_answer, graph_data } = await res.json();

      // Replace the typing indicator with actual response including graph data
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last?.role === 'assistant') {
          next[next.length - 1] = {
            ...last,
            content: (generated_answer ?? '').trim() || 'I couldn\'t generate a response.',
            graphData: graph_data ?? null,
          };
        }
        return next;
      });
    } catch (err: any) {
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last?.role === 'assistant') {
          next[next.length - 1] = {
            ...last,
            content: 'Sorry, I encountered an error. Please try again.',
          };
        }
        return next;
      });
      console.error(err);
    } finally {
      setSending(false);
    }
  }

  const formatTime = () => {
    return new Date().toLocaleTimeString('en-US', {
      hour12: true,
      hour: 'numeric',
      minute: '2-digit'
    });
  };

  // Component for rendering graph within message
  const GraphVisualization = ({ graphData }: { graphData: any }) => {
    const chartData = {
      labels: graphData.coordinates.map((coord: any) => coord.x.toString()),
      datasets: [
        {
          label: `${graphData.y_title} vs ${graphData.x_title}`,
          data: graphData.coordinates.map((coord: any) => ({
            x: coord.x,
            y: coord.y
          })),
          borderColor: 'rgba(139, 92, 246, 1)',
          backgroundColor: 'rgba(139, 92, 246, 0.1)',
          pointBackgroundColor: 'rgba(139, 92, 246, 1)',
          pointBorderColor: '#0f172a',
          pointBorderWidth: 2,
          pointRadius: 4,
          pointHoverRadius: 6,
          borderWidth: 2,
          tension: 0.4,
          fill: true,
        }
      ]
    };

    const chartOptions = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false,
        },
        title: {
          display: true,
          text: `${graphData.y_title} vs ${graphData.x_title}`,
          color: '#f8fafc',
          font: { size: 14, weight: 700 },
          padding: { bottom: 15 },
        },
        tooltip: {
          backgroundColor: 'rgba(2, 6, 23, 0.95)',
          titleColor: '#f8fafc',
          bodyColor: '#e2e8f0',
          borderColor: '#475569',
          borderWidth: 1,
          cornerRadius: 8,
          padding: 12,
          displayColors: false,
          titleFont: { size: 12, weight: 600 },
          bodyFont: { size: 11 },
          callbacks: {
            title: function(context: any) {
              return `${graphData?.x_title}: ${context[0].parsed.x}`;
            },
            label: function(context: any) {
              return `${graphData?.y_title}: ${context.parsed.y}`;
            }
          }
        },
      },
      scales: {
        x: {
          type: 'linear' as const,
          title: {
            display: true,
            text: graphData?.x_title || 'X-axis',
            color: '#94a3b8',
            font: { size: 11, weight: 500 },
          },
          grid: { 
            color: "rgba(148, 163, 184, 0.08)",
            lineWidth: 1,
          },
          ticks: { 
            color: "#64748b", 
            font: { size: 10 },
            padding: 6,
          },
        },
        y: {
          title: {
            display: true,
            text: graphData?.y_title || 'Y-axis',
            color: '#94a3b8',
            font: { size: 11, weight: 500 },
          },
          grid: { 
            color: "rgba(148, 163, 184, 0.08)",
            lineWidth: 1,
          },
          ticks: { 
            color: "#64748b", 
            font: { size: 10 },
            padding: 6,
          },
        },
      },
    };

    return (
      <div className="mt-4 bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-600/30 p-4">
        <div className="flex items-center space-x-2 mb-3">
          <div className="w-6 h-6 bg-gradient-to-br from-violet-500/20 to-purple-500/20 rounded-lg flex items-center justify-center border border-violet-400/30">
            <span className="text-sm">📊</span>
          </div>
          <h4 className="text-sm font-semibold bg-gradient-to-r from-violet-400 via-purple-400 to-fuchsia-400 bg-clip-text text-transparent">
            Ocean Data Visualization
          </h4>
        </div>
        
        <div className="h-64 bg-slate-900/30 rounded-lg border border-slate-600/20 p-3">
          <Line data={chartData} options={chartOptions} />
        </div>
        
        {/* Data summary */}
        <div className="mt-3 grid grid-cols-2 gap-3 text-xs">
          <div className="bg-slate-700/30 rounded-md p-2 border border-slate-600/20">
            <span className="text-slate-400">Data Points:</span>
            <span className="ml-1 text-white font-semibold">{graphData.coordinates.length}</span>
          </div>
          <div className="bg-slate-700/30 rounded-md p-2 border border-slate-600/20">
            <span className="text-slate-400">Range:</span>
            <span className="ml-1 text-white font-semibold">
              {Math.min(...graphData.coordinates.map((c: any) => c.x)).toFixed(1)} - {Math.max(...graphData.coordinates.map((c: any) => c.x)).toFixed(1)}
            </span>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 relative overflow-hidden">
      {/* Background Effects */}
      <div className="absolute inset-0 bg-gradient-to-r from-violet-500/5 via-purple-500/5 to-fuchsia-500/5"></div>
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-violet-600/10 via-transparent to-transparent"></div>

      {/* Animated background particles */}
      <div className="absolute top-1/4 left-1/4 w-32 h-32 bg-violet-500/8 rounded-full blur-3xl animate-pulse"></div>
      <div className="absolute bottom-1/3 right-1/4 w-24 h-24 bg-purple-500/8 rounded-full blur-2xl animate-pulse" style={{ animationDelay: "2s" }}></div>

      {/* Header */}
      <div className="relative z-10 bg-slate-900/80 backdrop-blur-xl border-b border-slate-700/50 p-4">
        <div className="flex items-center space-x-4 max-w-4xl mx-auto">
          <div className="w-12 h-12 bg-gradient-to-br from-violet-500/20 to-purple-500/20 rounded-2xl border border-violet-400/30 flex items-center justify-center backdrop-blur-sm shadow-2xl shadow-violet-500/20">
            <span className="text-2xl">🤖</span>
          </div>
          <div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-violet-400 via-purple-400 to-fuchsia-400 bg-clip-text text-transparent">
              FloatChat AI
            </h1>
            <p className="text-sm text-slate-400 font-medium">Ocean Data Intelligence Assistant</p>
          </div>
          <div className="ml-auto flex items-center space-x-2">
            <div className="flex items-center space-x-2 px-3 py-2 bg-gradient-to-r from-emerald-500/10 to-green-500/10 rounded-lg border border-emerald-400/20">
              <div className="w-2 h-2 bg-gradient-to-r from-emerald-400 to-green-400 rounded-full animate-pulse"></div>
              <span className="text-xs text-emerald-300 font-bold">ONLINE</span>
            </div>
          </div>
        </div>
      </div>

      {/* Messages Container */}
      <div className="relative z-10 flex-1 overflow-y-auto p-4 pb-32 max-w-4xl mx-auto h-[calc(100vh-140px)]">
        <div className="space-y-4">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`max-w-[80%] ${message.role === 'user' ? 'order-2' : 'order-1'}`}>
                {/* Message bubble */}
                <div
                  className={`relative px-4 py-3 rounded-2xl shadow-lg ${message.role === 'user'
                    ? 'bg-gradient-to-r from-violet-600 to-purple-600 text-white rounded-br-md'
                    : 'bg-slate-800/80 backdrop-blur-sm border border-slate-700/50 text-slate-100 rounded-bl-md'
                    }`}
                >
                  {/* Typing animation for assistant */}
                  {message.content === '…' && message.role === 'assistant' ? (
                    <div className="flex items-center space-x-1 py-2">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-slate-400 rounded-full animate-pulse"></div>
                        <div className="w-2 h-2 bg-slate-400 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
                        <div className="w-2 h-2 bg-slate-400 rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
                      </div>
                      <span className="text-xs text-slate-400 ml-2">AI is thinking...</span>
                    </div>
                  ) : (
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={{
                        p: ({ node, ...props }) => (
                          <p className="prose prose-invert max-w-none text-sm leading-relaxed" {...props} />
                        ),
                      }}
                    >
                      {message.content}
                    </ReactMarkdown>
                  )}

                  {/* Graph visualization inline with message */}
                  {message.role === 'assistant' && message.graphData && (
                    <GraphVisualization graphData={message.graphData} />
                  )}

                  {/* Message tail */}
                  <div
                    className={`absolute bottom-0 w-4 h-4 ${message.role === 'user'
                      ? 'right-0 bg-gradient-to-r from-violet-600 to-purple-600'
                      : 'left-0 bg-slate-800/80 border-l border-b border-slate-700/50'
                      } transform rotate-45 translate-y-2 ${message.role === 'user' ? 'translate-x-2' : '-translate-x-2'
                      }`}
                  ></div>
                </div>

                {/* Timestamp */}
                <div className={`text-xs text-slate-500 mt-1 px-1 ${message.role === 'user' ? 'text-right' : 'text-left'
                  }`}>
                  {formatTime()}
                </div>
              </div>

              {/* Avatar */}
              <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold ${message.role === 'user'
                ? 'order-1 ml-3 bg-gradient-to-br from-violet-500 to-purple-500 text-white'
                : 'order-2 mr-3 bg-gradient-to-br from-slate-700 to-slate-600 text-slate-300'
                }`}>
                {message.role === 'user' ? '👤' : '🤖'}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Fixed Input at Bottom */}
      <div className="fixed bottom-0 left-0 right-0 z-20 bg-slate-900/95 backdrop-blur-xl border-t border-slate-700/50 p-4">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
          <div className="flex items-end space-x-3">
            <div className="flex-1 relative">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e as any);
                  }
                }}
                placeholder="Ask about ocean data, buoys, marine analytics..."
                disabled={sending}
                rows={1}
                className="w-full resize-none rounded-2xl bg-slate-800/80 backdrop-blur-sm border border-slate-600/50 text-white placeholder-slate-400 px-4 py-3 pr-12 focus:outline-none focus:ring-2 focus:ring-violet-500/50 focus:border-violet-500/50 transition-all duration-300 text-sm leading-relaxed"
                style={{ minHeight: '44px', maxHeight: '120px' }}
              />

              {/* Input decorations */}
              <div className="absolute right-3 bottom-3 flex items-center space-x-2">
                {input.trim() && (
                  <div className="text-xs text-slate-500 font-mono">
                    {input.length}
                  </div>
                )}
              </div>
            </div>

            <button
              type="submit"
              disabled={sending || !input.trim()}
              className="w-12 h-12 bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-700 hover:to-purple-700 disabled:from-slate-600 disabled:to-slate-700 text-white rounded-2xl flex items-center justify-center transition-all duration-300 shadow-lg hover:shadow-violet-500/25 disabled:opacity-50 disabled:cursor-not-allowed transform hover:scale-105 disabled:hover:scale-100"
            >
              {sending ? (
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              ) : (
                <span className="text-lg">🚀</span>
              )}
            </button>
          </div>

          {/* Quick actions */}
          <div className="flex items-center space-x-2 mt-3">
            {['Show buoy data', 'Ocean temperature trends', 'Salinity analysis'].map((suggestion) => (
              <button
                key={suggestion}
                onClick={() => setInput(suggestion)}
                className="px-3 py-1.5 bg-slate-800/60 hover:bg-slate-700/60 text-slate-300 hover:text-white text-xs rounded-lg border border-slate-600/30 hover:border-slate-500/50 transition-all duration-200"
              >
                {suggestion}
              </button>
            ))}
          </div>
        </form>
      </div>
    </div>
  );
}