import { useState } from 'react';
import Sidebar from './components/Sidebar';
import ChatWindow from './components/ChatWindow';
import ChatInput from './components/ChatInput';
import { sendMessage } from './api/api';

export default function App() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleSend = async (question) => {
    // Add user message
    const userMsg = { role: 'user', content: question };
    setMessages(prev => [...prev, userMsg]);
    setIsLoading(true);

    try {
      const data = await sendMessage(question);

      let content = '';
      let image_url = null;
      let sources = [];
      let route = 'general';

      if (data && typeof data === 'object') {
        content = data.answer || '';
        image_url = data.image_url || null;
        sources = data.sources || [];
        route = data.route || 'general';
      } else if (typeof data === 'string') {
        try {
          const parsed = JSON.parse(data);
          content = parsed.answer || data;
          image_url = parsed.image_url || null;
          sources = parsed.sources || [];
          route = parsed.route || 'general';
        } catch (e) {
          content = data;
        }
      }

      const assistantMsg = {
        role: 'assistant',
        content,
        image_url,
        sources,
        route
      };

      setMessages(prev => [...prev, assistantMsg]);
    } catch (err) {
      console.error(err);
      const errorMsg = {
        role: 'assistant',
        content: '❌ Failed to get response. Please make sure the backend is running.',
        sources: [],
        image_url: null,
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };


  return (
    <div className="flex h-screen bg-dark-950">
      {/* Sidebar */}
      <Sidebar />

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="px-6 py-4 border-b border-brand-500/10 glass">
          <div className="flex items-center justify-between">
            <div className="pl-12 lg:pl-0">
              <h1 className="text-lg font-bold text-dark-100">District Knowledge Chat</h1>
              <p className="text-xs text-dark-400">Ask questions about Theni district</p>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs text-dark-400 hidden sm:inline">
                {messages.filter(m => m.role === 'user').length} questions asked
              </span>
            </div>
          </div>
        </div>

        {/* Chat Window */}
        <ChatWindow messages={messages} isLoading={isLoading} />

        {/* Chat Input */}
        <ChatInput onSend={handleSend} disabled={isLoading} />
      </div>
    </div>
  );
}
