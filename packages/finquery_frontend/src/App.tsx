/* App.tsx */
import React, { useState, useEffect, useRef } from 'react';
import { Send, Upload, RefreshCw, Database, Bot, User, Loader2, Server, AlertTriangle } from 'lucide-react';

// --- Configuration & Types ---
const API_BASE_URL = 'http://localhost:5001/api';

interface ChatMessage {
  id: string;
  sender: 'user' | 'assistant';
  content: string;
  type: 'ask' | 'answer' | 'error' | 'loading';
}

interface DbStatus {
  collection_name: string;
  total_chunks: number;
}

// --- Child Components for Better Structure ---

const MessageBubble: React.FC<{ msg: ChatMessage }> = ({ msg }) => (
  <div key={msg.id} className={`flex items-start gap-3 ${msg.sender === 'user' ? 'justify-end' : ''}`}>
    {msg.sender === 'assistant' && (
      <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white flex-shrink-0">
        <Bot size={20} />
      </div>
    )}
    <div className={`max-w-lg p-3 rounded-xl shadow-sm ${msg.sender === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-100 text-gray-800'}`}>
      {msg.type === 'loading' ? (
        <div className="flex items-center justify-center p-2">
          <Loader2 className="animate-spin" size={20} />
        </div>
      ) : msg.type === 'error' ? (
        <div className="flex items-center gap-2 text-red-700 bg-red-100 p-2 rounded-md">
          <AlertTriangle size={16} />
          <p className="text-sm">{msg.content}</p>
        </div>
      ) : (
        <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
      )}
    </div>
    {msg.sender === 'user' && (
      <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center text-gray-600 flex-shrink-0">
        <User size={20} />
      </div>
    )}
  </div>
);

const ChatPanel: React.FC<{
  messages: ChatMessage[];
  userInput: string;
  setUserInput: (value: string) => void;
  isLoading: boolean;
  handleSendMessage: (e: React.FormEvent) => void;
}> = ({ messages, userInput, setUserInput, isLoading, handleSendMessage }) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <main className="flex-1 flex flex-col bg-white">
      <header className="p-4 border-b border-gray-200 flex-shrink-0">
        <h1 className="text-xl font-semibold text-gray-900">FinQuery Assistant</h1>
      </header>
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.map((msg) => <MessageBubble key={msg.id} msg={msg} />)}
        <div ref={messagesEndRef} />
      </div>
      <footer className="p-4 border-t border-gray-200 bg-white flex-shrink-0">
        <form onSubmit={handleSendMessage} className="flex items-center gap-3">
          <input
            type="text"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            placeholder="Ask a question..."
            className="flex-1 p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none transition"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !userInput.trim()}
            className="p-2 bg-blue-500 text-white rounded-lg disabled:bg-blue-300 disabled:cursor-not-allowed hover:bg-blue-600 transition flex items-center justify-center w-10 h-10"
          >
            {isLoading ? <Loader2 className="animate-spin" /> : <Send size={20} />}
          </button>
        </form>
      </footer>
    </main>
  );
};

const ControlPanel: React.FC<{
  dbStatus: DbStatus | null;
  fetchDbStatus: () => void;
  selectedFile: File | null;
  setSelectedFile: (file: File | null) => void;
  handleFileUpload: () => void;
  triggerIngestion: () => void;
  controlPanelMessage: { type: 'info' | 'error', text: string } | null;
}> = ({ dbStatus, fetchDbStatus, selectedFile, setSelectedFile, handleFileUpload, triggerIngestion, controlPanelMessage }) => (
  <aside className="hidden md:flex w-96 bg-gray-100 p-6 border-l border-gray-200 flex-col gap-8 overflow-y-auto">
    <div className="space-y-2">
      <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2"><Server size={20} /> Control Panel</h2>
      <p className="text-sm text-gray-500">Manage your documents and data ingestion.</p>
    </div>
    <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
      <h3 className="font-semibold flex items-center gap-2 mb-2"><Database size={16} /> Database Status</h3>
      {dbStatus ? (
        <div className="text-sm space-y-1">
          <p>Collection: <span className="font-medium text-gray-600">{dbStatus.collection_name}</span></p>
          <p>Total Chunks: <span className="font-medium text-gray-600">{dbStatus.total_chunks}</span></p>
        </div>
      ) : <p className="text-sm text-gray-500">Loading status...</p>}
      <button onClick={fetchDbStatus} className="text-xs text-blue-500 hover:underline mt-2">Refresh</button>
    </div>
    <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-200 space-y-3">
      <h3 className="font-semibold flex items-center gap-2"><Upload size={16} /> Upload Document</h3>
      <input
        id="file-upload" type="file" accept=".pdf"
        onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
        className="text-sm file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 w-full"
      />
      <button onClick={handleFileUpload} disabled={!selectedFile} className="w-full bg-blue-500 text-white text-sm font-medium py-2 rounded-lg hover:bg-blue-600 transition disabled:bg-blue-300 disabled:cursor-not-allowed">
        Upload File
      </button>
    </div>
    <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-200 space-y-3">
      <h3 className="font-semibold flex items-center gap-2"><RefreshCw size={16} /> Ingest Data</h3>
      <p className="text-sm text-gray-500">Process uploaded documents and add them to the database.</p>
      <button onClick={triggerIngestion} className="w-full bg-green-500 text-white text-sm font-medium py-2 rounded-lg hover:bg-green-600 transition">
        Start Ingestion
      </button>
    </div>
    {controlPanelMessage && (
      <div className={`p-3 rounded-lg text-sm transition-opacity duration-300 ${controlPanelMessage.type === 'error' ? 'bg-red-100 text-red-800' : 'bg-blue-100 text-blue-800'}`}>
        {controlPanelMessage.text}
      </div>
    )}
  </aside>
);

// --- Main App Component (Logic) ---
const App: React.FC = () => {
  const [sessionId, setSessionId] = useState<string>('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [userInput, setUserInput] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [dbStatus, setDbStatus] = useState<DbStatus | null>(null);
  const [controlPanelMessage, setControlPanelMessage] = useState<{type: 'info' | 'error', text: string} | null>(null);

  useEffect(() => {
    setSessionId(`session_${crypto.randomUUID()}`);
    setMessages([{
      id: crypto.randomUUID(),
      sender: 'assistant',
      content: "Hello! I'm FinQuery. Ask me a question about your financial documents.",
      type: 'ask'
    }]);
    fetchDbStatus();
  }, []);

  useEffect(() => {
    if (controlPanelMessage) {
      const timer = setTimeout(() => setControlPanelMessage(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [controlPanelMessage]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!userInput.trim() || isLoading) return;

    const userMessage: ChatMessage = { id: crypto.randomUUID(), sender: 'user', content: userInput, type: 'answer' };
    setMessages(prev => [...prev, userMessage]);
    setUserInput('');
    setIsLoading(true);

    const loadingMessageId = crypto.randomUUID();
    const loadingMessage: ChatMessage = { id: loadingMessageId, sender: 'assistant', content: '', type: 'loading' };
    setMessages(prev => [...prev, loadingMessage]);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message: userInput }),
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || 'API request failed');
      }

      const data = await response.json();

      if (data.type === 'ask') {
          setMessages(prev => prev.map(msg => msg.id === loadingMessageId ? { ...msg, content: data.message, type: 'ask' } : msg));
      } else if (data.type === 'answer') {
          setMessages(prev => prev.map(msg => msg.id === loadingMessageId ? { ...msg, content: data.answer, type: 'answer' } : msg));
      } else {
          throw new Error("Received an unknown response type from the API.");
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred.';
      setMessages(prev => prev.map(msg => msg.id === loadingMessageId ? { ...msg, content: `Error: ${errorMessage}`, type: 'error' } : msg));
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async () => {
    if (!selectedFile) return;
    setControlPanelMessage({ type: 'info', text: 'Uploading file...' });
    try {
      const fd = new FormData();
      fd.append('file', selectedFile);
      const res = await fetch(`${API_BASE_URL}/upload`, { method: 'POST', body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Upload failed');
      setControlPanelMessage({ type: 'info', text: data.message });
      setSelectedFile(null);
      const fileInput = document.getElementById('file-upload') as HTMLInputElement;
      if (fileInput) fileInput.value = '';
    } catch (e) {
      setControlPanelMessage({ type: 'error', text: e instanceof Error ? e.message : 'Upload failed' });
    }
  };

  const triggerIngestion = async () => {
    setControlPanelMessage({ type: 'info', text: 'Starting ingestion process...' });
    try {
      const res = await fetch(`${API_BASE_URL}/ingest`, { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Ingestion failed');
      setControlPanelMessage({ type: 'info', text: data.message });
    } catch (e) {
      setControlPanelMessage({ type: 'error', text: e instanceof Error ? e.message : 'Ingestion failed' });
    }
  };

  const fetchDbStatus = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/db/status`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to fetch DB status');
      setDbStatus(data);
    } catch (e) {
       setControlPanelMessage({ type: 'error', text: e instanceof Error ? e.message : 'Failed to fetch DB status' });
    }
  };

  return (
    <>
      <style>{`
        html, body, #root {
          height: 100%;
          margin: 0;
          padding: 0;
          overflow: hidden; /* Prevents scrolling on the body */
        }
      `}</style>
      <div className="font-sans bg-gray-50 h-full flex text-gray-800" style={{ width: '1515px' }}>
        <ChatPanel
          messages={messages}
          userInput={userInput}
          setUserInput={setUserInput}
          isLoading={isLoading}
          handleSendMessage={handleSendMessage}
        />
        <ControlPanel
          dbStatus={dbStatus}
          fetchDbStatus={fetchDbStatus}
          selectedFile={selectedFile}
          setSelectedFile={setSelectedFile}
          handleFileUpload={handleFileUpload}
          triggerIngestion={triggerIngestion}
          controlPanelMessage={controlPanelMessage}
        />
      </div>
    </>
  );
};

export default App;
