import { useState } from 'react';
import { FiSend } from 'react-icons/fi';

export default function ChatInput({ onSend, disabled }) {
  const [message, setMessage] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (message.trim() && !disabled) {
      onSend(message.trim());
      setMessage('');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="px-6 py-4">
      <form onSubmit={handleSubmit} className="relative">
        <div className="glass rounded-[1.25rem] flex items-center gap-3 px-5 py-3.5 glow border border-brand-500/30 shadow-lg shadow-brand-500/10 focus-within:border-brand-400 focus-within:shadow-brand-500/20 transition-all duration-300 bg-brand-950/40">
          <input
            type="text"
            id="chat-input"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={disabled ? "Waiting for response..." : "Ask about the district..."}
            disabled={disabled}
            className="flex-1 bg-transparent text-sm text-brand-50 placeholder-brand-300/50 outline-none disabled:opacity-50"
            autoComplete="off"
          />
          <button
            type="submit"
            disabled={disabled || !message.trim()}
            className="w-10 h-10 rounded-xl bg-gradient-to-r from-brand-600 to-brand-500 hover:from-brand-500 hover:to-brand-400 disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center transition-all duration-300 hover:scale-105 active:scale-95 shadow-md shadow-brand-500/30"
          >
            <FiSend className="text-white text-sm" />
          </button>
        </div>
        <p className="text-center text-xs text-brand-200/50 mt-3 font-medium tracking-wide">
          Powered by Llama-3.1 via Groq • Answers based on the official district knowledge base
        </p>
      </form>
    </div>
  );
}
