import { useRef, useEffect } from 'react';
import { FiUser, FiCpu, FiFileText } from 'react-icons/fi';

function TypingIndicator() {
  return (
    <div className="flex items-start gap-3 animate-fade-in">
      <div className="w-8 h-8 rounded-xl bg-brand-500/20 flex items-center justify-center flex-shrink-0 mt-1">
        <FiCpu className="text-brand-400 text-sm" />
      </div>
      <div className="msg-bot px-5 py-4">
        <div className="flex items-center gap-1">
          <span className="typing-dot"></span>
          <span className="typing-dot"></span>
          <span className="typing-dot"></span>
        </div>
      </div>
    </div>
  );
}

function SourceBadge({ source }) {
  return (
    <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-brand-500/10 border border-brand-500/20 text-xs text-brand-300">
      <FiFileText size={11} />
      <span className="truncate max-w-[120px]">{source.document}</span>
      <span className="text-brand-400/60">({source.relevance})</span>
    </div>
  );
}

function getCleanTextContent(message) {
  if (!message) return '';
  let content = message.content || '';

  if (typeof content === 'string' && content.trim().startsWith('{') && content.trim().endsWith('}')) {
    try {
      const parsed = JSON.parse(content);
      if (parsed && parsed.answer !== undefined) {
        return parsed.answer;
      }
    } catch (e) {
      return content;
    }
  }
  return content;
}

function getImageUrl(message) {
  if (!message) return null;

  let url = message.image_url || message.image_path || null;

  if (!url && typeof message.content === 'string' && message.content.trim().startsWith('{') && message.content.trim().endsWith('}')) {
    try {
      const parsed = JSON.parse(message.content);
      if (parsed && (parsed.image_url || parsed.image_path)) {
        url = parsed.image_url || parsed.image_path;
      }
    } catch (e) {
      // Ignore
    }
  }

  if (!url) return null;

  if (url.startsWith('http://') || url.startsWith('https://')) {
    return url;
  }

  if (!url.startsWith('/')) {
    url = '/' + url;
  }

  const apiBase = (import.meta.env?.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');
  return `${apiBase}${url}`;
}

function MessageBubble({ message }) {
  const isUser = message.role === 'user';
  const imageUrl = getImageUrl(message);
  const displayContent = getCleanTextContent(message);

  return (
    <div className={`flex items-start gap-3 animate-slide-up ${isUser ? 'flex-row-reverse' : ''}`}>
      {/* Avatar */}
      <div className={`w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0 mt-1 ${
        isUser ? 'bg-brand-500' : 'bg-brand-500/20'
      }`}>
        {isUser ? (
          <FiUser className="text-white text-sm" />
        ) : (
          <FiCpu className="text-brand-400 text-sm" />
        )}
      </div>

      {/* Message Content */}
      <div className={`max-w-[75%] ${isUser ? 'msg-user' : 'msg-bot'} px-5 py-3.5 space-y-3`}>
        {displayContent && (
          <p className="text-sm leading-relaxed whitespace-pre-wrap">{displayContent}</p>
        )}

        {imageUrl && (
          <div className="mt-2 overflow-hidden rounded-xl border border-brand-500/20 bg-black/20">
            <img
              src={imageUrl}
              alt="AI generated image"
              className="max-w-full h-auto rounded-lg object-contain max-h-[400px] shadow-lg"
              onError={(e) => {
                console.error("Image failed to load:", imageUrl);
                const backendFallback = `http://localhost:8000${imageUrl.startsWith('/') ? '' : '/'}${imageUrl.replace(/^https?:\/\/[^/]+/, '')}`;
                if (e.target.src !== backendFallback) {
                  e.target.src = backendFallback;
                  return;
                }
                e.target.onerror = null;
              }}
            />
          </div>
        )}
      </div>
    </div>
  );
}

export default function ChatWindow({ messages, isLoading }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  return (
    <div className="flex-1 overflow-y-auto px-6 py-4 space-y-5">
      {/* Welcome Message */}
      {messages.length === 0 && (
        <div className="flex flex-col items-center justify-center h-full text-center animate-fade-in">
          <div className="w-24 h-24 rounded-[2rem] bg-gradient-to-br from-brand-500/20 to-brand-700/20 flex items-center justify-center mb-8 glow pulse-ring shadow-2xl shadow-brand-500/20 border border-brand-500/30 backdrop-blur-xl">
            <span className="text-5xl">🏛️</span>
          </div>
          <h2 className="text-3xl font-bold gradient-text mb-4 tracking-tight">District Knowledge Assistant</h2>
          <p className="text-brand-200/70 text-sm max-w-md leading-relaxed">
            I am loaded with comprehensive district data. Ask me anything and I will provide accurate answers based on the official records.
          </p>
          <div className="mt-8 grid grid-cols-2 gap-3 max-w-lg">
            {[
              "What is the population?",
              "Major tourist attractions?",
              "Government welfare schemes?",
              "What is the literacy rate?",
            ].map((q, i) => (
              <div key={i} className="glass-light rounded-xl px-4 py-3.5 text-xs text-brand-100/80 cursor-default hover:border-brand-400/50 hover:bg-brand-500/10 hover:shadow-lg hover:shadow-brand-500/10 transition-all duration-300 transform hover:-translate-y-1">
                💬 {q}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Messages */}
      {messages.map((msg, i) => (
        <MessageBubble key={i} message={msg} />
      ))}

      {/* Typing Indicator */}
      {isLoading && <TypingIndicator />}

      <div ref={bottomRef} />
    </div>
  );
}
