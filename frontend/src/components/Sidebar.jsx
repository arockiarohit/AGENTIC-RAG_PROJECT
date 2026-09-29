import { useState, useEffect } from 'react';
import { FiDatabase, FiActivity, FiMenu, FiX, FiInfo } from 'react-icons/fi';
import { checkHealth } from '../api/api';

export default function Sidebar() {
  const [health, setHealth] = useState(null);
  const [isOpen, setIsOpen] = useState(true);

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  const fetchHealth = async () => {
    try {
      const data = await checkHealth();
      setHealth(data);
    } catch {
      setHealth(null);
    }
  };

  return (
    <>
      {/* Mobile Toggle */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="lg:hidden fixed top-4 left-4 z-50 w-10 h-10 rounded-xl glass flex items-center justify-center text-dark-200"
      >
        {isOpen ? <FiX /> : <FiMenu />}
      </button>

      {/* Sidebar */}
      <div className={`
        fixed lg:relative inset-y-0 left-0 z-40
        w-80 glass border-r border-brand-500/10
        flex flex-col
        transition-transform duration-300
        ${isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
      `}>
        {/* Header */}
        <div className="p-6 border-b border-brand-500/10">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-brand-500 to-brand-700 flex items-center justify-center shadow-lg shadow-brand-500/30">
              <span className="text-xl">🏛️</span>
            </div>
            <div>
              <h1 className="text-sm font-bold gradient-text">District RAG</h1>
              <p className="text-xs text-brand-300/80">Knowledge Chatbot</p>
            </div>
          </div>
        </div>

        {/* Status */}
        <div className="px-6 py-4 border-b border-brand-500/10 bg-brand-900/10">
          <div className="flex items-center gap-2 mb-3">
            <FiActivity className="text-brand-400" size={14} />
            <span className="text-xs font-semibold text-brand-200 uppercase tracking-wider">System Status</span>
          </div>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs text-dark-300">Backend API</span>
              <span className={`text-xs font-medium flex items-center gap-1.5 ${health ? 'text-emerald-400' : 'text-red-400'}`}>
                <span className={`w-1.5 h-1.5 rounded-full ${health ? 'bg-emerald-400' : 'bg-red-400'} animate-pulse`} />
                {health ? 'Connected' : 'Offline'}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-dark-300">Vector Index</span>
              <span className="text-xs font-medium text-brand-300 flex items-center gap-1.5">
                <FiDatabase size={11} />
                {health?.total_vectors || 0}
              </span>
            </div>
          </div>
        </div>

        {/* Knowledge Base Overview */}
        <div className="flex-1 overflow-y-auto px-6 py-6">
          <div className="glass-light rounded-xl p-4 border border-brand-500/20 glow">
            <div className="flex items-center gap-2 mb-3">
              <FiInfo className="text-brand-400" size={16} />
              <h3 className="text-sm font-semibold text-white">Knowledge Base</h3>
            </div>
            <p className="text-xs text-dark-300 leading-relaxed mb-4">
              The chatbot is pre-loaded with comprehensive district data. It uses advanced RAG technology to provide accurate answers.
            </p>
            
            <div className="space-y-2">
              <div className="text-xs flex justify-between items-center p-2 rounded-lg bg-dark-900/50">
                <span className="text-dark-400">Total Chunks</span>
                <span className="text-brand-300 font-mono">{health?.total_documents || 0}</span>
              </div>
              <div className="text-xs flex justify-between items-center p-2 rounded-lg bg-dark-900/50">
                <span className="text-dark-400">Data Source</span>
                <span className="text-brand-300 truncate max-w-[100px]" title="theni_complete_large_rag_dataset.json">Project Data</span>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-brand-500/10 bg-brand-950/30">
          <p className="text-xs font-medium text-brand-400/60 text-center tracking-widest uppercase">
            Llama-3.1 • Hybrid RAG
          </p>
        </div>
      </div>
    </>
  );
}
