/* App.tsx */
import React, { useState, useEffect } from 'react';
import {
  FileText,
  Upload,
  Search,
  Database,
  TestTube,
  Trash2,
  Activity,
  AlertCircle,
  CheckCircle,
} from 'lucide-react';

const API_BASE_URL = 'http://localhost:5001/api';

// ---------- types ----------
interface DbStatus {
  collection_name: string;
  total_chunks: number;
  status?: string;
}
interface DocumentsResponse {
  processed_documents: string[];
  pending_ingestion: string[];
}
interface TestQuestion {
  question: string;
  category?: string;
}
interface QueryResult {
  content: string;
  metadata: Record<string, unknown>;
}
interface ApiResponse {
  message?: string;
  error?: string;
  status?: string;
  query?: string;
  answer?: string;
  results?: QueryResult[];
  count?: number;
  questions?: TestQuestion[];
  collection_name?: string;
  total_chunks?: number;
  processed_documents?: string[];
  pending_ingestion?: string[];
}
type ResultValue =
  | string
  | QueryResult[]
  | TestQuestion[]
  | DocumentsResponse
  | DbStatus
  | null;

// ---------- Presentational Components (Moved outside App) ----------
const ActionCard: React.FC<{
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}> = ({ title, icon, children }) => (
  <div
    style={{
      backgroundColor: 'white',
      borderRadius: 8,
      boxShadow: '0 2px 4px rgba(0,0,0,.1)',
      padding: 24,
      border: '1px solid #e2e8f0',
      minWidth: 280,
    }}
  >
    <div style={{ display: 'flex', alignItems: 'center', marginBottom: 16 }}>
      {icon}
      <h2 style={{ fontSize: '1.25rem', fontWeight: 600, marginLeft: 12 }}>
        {title}
      </h2>
    </div>
    {children}
  </div>
);

const ResultDisplay: React.FC<{ result: ResultValue; error?: string }> = ({
  result,
  error,
}) => {
  if (error)
    return (
      <div
        style={{
          marginTop: 16,
          padding: 12,
          backgroundColor: '#fef2f2',
          border: '1px solid #fecaca',
          borderRadius: 6,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <AlertCircle
            style={{ height: 20, width: 20, color: '#ef4444', marginRight: 8 }}
          />
          <span style={{ color: '#b91c1c' }}>{error}</span>
        </div>
      </div>
    );

  if (result)
    return (
      <div
        style={{
          marginTop: 16,
          padding: 12,
          backgroundColor: '#f0fdf4',
          border: '1px solid #dcfce7',
          borderRadius: 6,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: 8 }}>
          <CheckCircle
            style={{ height: 20, width: 20, color: '#22c55e', marginRight: 8 }}
          />
          <span style={{ color: '#15803d', fontWeight: 500 }}>Success</span>
        </div>
        <pre
          style={{
            fontSize: '.875rem',
            color: '#374151',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-all', // Ensure long text wraps
            maxHeight: '16rem',
            overflowY: 'auto',
          }}
        >
          {typeof result === 'string'
            ? result
            : JSON.stringify(result, null, 2)}
        </pre>
      </div>
    );
  return null;
};

// ---------- component ----------
const App: React.FC = () => {
  // ----- state -----
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [queryText, setQueryText] = useState('');
  const [questionText, setQuestionText] = useState('');
  const [documents, setDocuments] = useState<DocumentsResponse | null>(null);
  const [dbStatus, setDbStatus] = useState<DbStatus | null>(null);
  const [testQuestions, setTestQuestions] = useState<TestQuestion[]>([]);
  const [loading, setLoading] = useState<Record<string, boolean>>({});
  const [results, setResults] = useState<Record<string, ResultValue>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});

  // ----- helpers -----
  const setLoadingState = (k: string, v: boolean) =>
    setLoading((p) => ({ ...p, [k]: v }));

  const setResult = (k: string, v: ResultValue) =>
    setResults((p) => ({ ...p, [k]: v }));

  const setError = (k: string, v: string) =>
    setErrors((p) => ({ ...p, [k]: v }));

  const clearError = (k: string) =>
    setErrors((p) => {
      const clone = { ...p };
      delete clone[k];
      return clone;
    });

  const apiCall = async (
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse> => {
    const res = await fetch(`${API_BASE_URL}${endpoint}`, {
      headers: { 'Content-Type': 'application/json', ...options.headers },
      ...options,
    });
    if (!res.ok) {
      const err = (await res.json()) as ApiResponse;
      throw new Error(err.error || `HTTP ${res.status}`);
    }
    return (await res.json()) as ApiResponse;
  };

  // ----- actions -----
  const handleFileUpload = async () => {
    if (!selectedFile) return;
    setLoadingState('upload', true);
    clearError('upload');

    try {
      const fd = new FormData();
      fd.append('file', selectedFile);
      const res = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: fd,
      });

      if (!res.ok) {
        const err = (await res.json()) as ApiResponse;
        throw new Error(err.error || 'Upload failed');
      }
      const data = (await res.json()) as ApiResponse;
      setResult('upload', data.message || 'Upload successful');
      setSelectedFile(null);
      // Clear the file input visually after upload
      const fileInput = document.getElementById('file-upload-input') as HTMLInputElement;
      if (fileInput) fileInput.value = '';

      await loadDocuments();
    } catch (e) {
      setError('upload', e instanceof Error ? e.message : 'Upload failed');
    } finally {
      setLoadingState('upload', false);
    }
  };

  const triggerIngestion = async () => {
    setLoadingState('ingest', true);
    clearError('ingest');
    try {
      const data = await apiCall('/ingest', { method: 'POST' });
      setResult('ingest', data.message || 'Ingestion started');
    } catch (e) {
      setError('ingest', e instanceof Error ? e.message : 'Ingestion failed');
    } finally {
      setLoadingState('ingest', false);
    }
  };

  const executeQuery = async () => {
    if (!queryText.trim()) return;
    setLoadingState('query', true);
    clearError('query');
    try {
      const data = await apiCall('/query', {
        method: 'POST',
        body: JSON.stringify({ query_text: queryText, num_to_fetch: 10 }),
      });
      setResult('query', data.results || []);
    } catch (e) {
      setError('query', e instanceof Error ? e.message : 'Query failed');
    } finally {
      setLoadingState('query', false);
    }
  };

  const askQuestion = async () => {
    if (!questionText.trim()) return;
    setLoadingState('question', true);
    clearError('question');
    try {
      const data = await apiCall('/question', {
        method: 'POST',
        body: JSON.stringify({ query_text: questionText }),
      });
      setResult('question', data.answer || 'No answer returned');
    } catch (e) {
      setError('question', e instanceof Error ? e.message : 'Question failed');
    } finally {
      setLoadingState('question', false);
    }
  };

  const loadDocuments = async () => {
    setLoadingState('documents', true);
    try {
      const data = await apiCall('/documents');
      setDocuments({
        processed_documents: data.processed_documents || [],
        pending_ingestion: data.pending_ingestion || [],
      });
    } catch (e) {
      setError(
        'documents',
        e instanceof Error ? e.message : 'Failed to load documents'
      );
    } finally {
      setLoadingState('documents', false);
    }
  };

  const loadDbStatus = async () => {
    setLoadingState('dbStatus', true);
    try {
      const data = await apiCall('/status/db');
      setDbStatus({
        collection_name: data.collection_name || 'unknown',
        total_chunks: data.total_chunks || 0,
        status: data.status,
      });
    } catch (e) {
      setError(
        'dbStatus',
        e instanceof Error ? e.message : 'Failed to load DB status'
      );
    } finally {
      setLoadingState('dbStatus', false);
    }
  };

  const loadTestQuestions = async () => {
    setLoadingState('testQuestions', true);
    try {
      const data = await apiCall('/testing/questions');
      setTestQuestions(data.questions || []);
    } catch (e) {
      setError(
        'testQuestions',
        e instanceof Error ? e.message : 'Failed to load test questions'
      );
    } finally {
      setLoadingState('testQuestions', false);
    }
  };

  const deleteCollection = async () => {
    if (
      !window.confirm(
        'Are you sure you want to delete the entire collection? This cannot be undone.'
      )
    )
      return;
    setLoadingState('delete', true);
    clearError('delete');
    try {
      const data = await apiCall('/admin/collection', { method: 'DELETE' });
      setResult('delete', data.message || 'Collection deleted');
      setDbStatus(null);
      setDocuments(null);
    } catch (e) {
      setError('delete', e instanceof Error ? e.message : 'Delete failed');
    } finally {
      setLoadingState('delete', false);
    }
  };

  const checkHealth = async () => {
    setLoadingState('health', true);
    try {
      const data = await apiCall('/health');
      setResult('health', data.message || 'API is healthy');
    } catch (e) {
      setError('health', e instanceof Error ? e.message : 'Health check failed');
    } finally {
      setLoadingState('health', false);
    }
  };

  // ----- init -----
  useEffect(() => {
    (async () => {
      await Promise.all([loadDocuments(), loadDbStatus(), checkHealth()]);
    })().catch(console.error);
  }, []);

  // ----- render -----
  return (
    // FIX: This new page shell creates a robust 3-column grid to center the content.
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'minmax(32px, 1fr) minmax(0, 1800px) minmax(32px, 1fr)',
      backgroundColor: '#f9fafb',
      minHeight: '100vh',
    }}>
      {/* FIX: This wrapper lives inside the stable middle column of the grid. */}
      <div style={{
        gridColumn: 2,
        paddingTop: 32,
        paddingBottom: 32,
        display: 'flex',
        flexDirection: 'column',
        gap: 32,
      }}>
        {/* ---------- HEADER ---------- */}
        <header style={{ textAlign: 'center' }}>
          <h1
            style={{
              fontSize: '2.25rem',
              fontWeight: 700,
              color: '#1f2937',
              marginBottom: 8,
            }}
          >
            FinQuery Demo
          </h1>
          <p style={{ color: '#4b5563' }}>Financial Document Query System</p>
        </header>

        {/* ---------- HEALTH ---------- */}
        <section style={{ textAlign: 'center' }}>
          <button
            type="button"
            onClick={checkHealth}
            disabled={loading.health}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              padding: '8px 16px',
              backgroundColor: '#3b82f6',
              color: 'white',
              borderRadius: 6,
              cursor: loading.health ? 'not-allowed' : 'pointer',
              opacity: loading.health ? 0.5 : 1,
              border: 'none',
              marginBottom: 8,
            }}
          >
            <Activity style={{ height: 16, width: 16, marginRight: 8 }} />
            {loading.health ? 'Checking...' : 'Check API Health'}
          </button>
          <ResultDisplay result={results.health} error={errors.health} />
        </section>

        {/* ---------- GRID ---------- */}
        <section
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit,minmax(320px,1fr))',
            gap: 24,
            width: '100%',
          }}
        >
          {/* ---------- UPLOAD ---------- */}
          <ActionCard title="Upload File" icon={<Upload size={20} />}>
            <input
              id="file-upload-input"
              type="file"
              onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
              style={{ display: 'block', width: '100%', marginBottom: 12 }}
            />
            <button
              type="button"
              onClick={handleFileUpload}
              disabled={!selectedFile || loading.upload}
              style={{
                display: 'flex',
                alignItems: 'center',
                backgroundColor: '#10b981',
                color: 'white',
                padding: '6px 12px',
                border: 'none',
                borderRadius: 4,
                cursor:
                  !selectedFile || loading.upload ? 'not-allowed' : 'pointer',
                opacity: !selectedFile || loading.upload ? 0.6 : 1,
              }}
            >
              <FileText size={16} style={{ marginRight: 6 }} />
              {loading.upload ? 'Uploading…' : 'Upload'}
            </button>
            <ResultDisplay result={results.upload} error={errors.upload} />
          </ActionCard>

          {/* ---------- DOCUMENTS ---------- */}
          <ActionCard title="Documents" icon={<Database size={20} />}>
            <button
              type="button"
              onClick={loadDocuments}
              disabled={loading.documents}
              style={{
                backgroundColor: '#4b5563',
                color: 'white',
                border: 'none',
                borderRadius: 4,
                padding: '6px 12px',
                cursor: loading.documents ? 'not-allowed' : 'pointer',
                opacity: loading.documents ? 0.6 : 1,
                marginBottom: 8,
              }}
            >
              {loading.documents ? 'Refreshing…' : 'Refresh'}
            </button>
            <ResultDisplay result={documents} error={errors.documents} />
          </ActionCard>

          {/* ---------- INGESTION ---------- */}
          <ActionCard title="Start Ingestion" icon={<Upload size={20} />}>
            <button
              type="button"
              onClick={triggerIngestion}
              disabled={loading.ingest}
              style={{
                backgroundColor: '#2563eb',
                color: 'white',
                border: 'none',
                borderRadius: 4,
                padding: '6px 12px',
                cursor: loading.ingest ? 'not-allowed' : 'pointer',
                opacity: loading.ingest ? 0.6 : 1,
              }}
            >
              {loading.ingest ? 'Starting…' : 'Start'}
            </button>
            <ResultDisplay result={results.ingest} error={errors.ingest} />
          </ActionCard>

          {/* ---------- QUERY (no <form>) ---------- */}
          <ActionCard title="Run Query" icon={<Search size={20} />}>
            <input
              type="text"
              value={queryText}
              placeholder="Enter query…"
              onChange={(e) => setQueryText(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && executeQuery()}
              style={{
                display: 'block',
                width: '100%',
                boxSizing: 'border-box',
                padding: 8,
                border: '1px solid #d1d5db',
                borderRadius: 4,
                marginBottom: 12,
              }}
            />
            <button
              type="button"
              onClick={executeQuery}
              disabled={!queryText.trim() || loading.query}
              style={{
                backgroundColor: '#3b82f6',
                color: 'white',
                border: 'none',
                borderRadius: 4,
                padding: '6px 12px',
                cursor: !queryText.trim() || loading.query ? 'not-allowed' : 'pointer',
                opacity: !queryText.trim() || loading.query ? 0.6 : 1,
              }}
            >
              {loading.query ? 'Running…' : 'Run'}
            </button>
            <ResultDisplay result={results.query} error={errors.query} />
          </ActionCard>

          {/* ---------- ASK QUESTION (no <form>) ---------- */}
          <ActionCard title="Ask Question" icon={<TestTube size={20} />}>
            <input
              type="text"
              value={questionText}
              placeholder="Ask your question…"
              onChange={(e) => setQuestionText(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && askQuestion()}
              style={{
                display: 'block',
                width: '100%',
                boxSizing: 'border-box',
                padding: 8,
                border: '1px solid #d1d5db',
                borderRadius: 4,
                marginBottom: 12,
              }}
            />
            <button
              type="button"
              onClick={askQuestion}
              disabled={!questionText.trim() || loading.question}
              style={{
                backgroundColor: '#10b981',
                color: 'white',
                border: 'none',
                borderRadius: 4,
                padding: '6px 12px',
                cursor: !questionText.trim() || loading.question ? 'not-allowed' : 'pointer',
                opacity: !questionText.trim() || loading.question ? 0.6 : 1,
              }}
            >
              {loading.question ? 'Asking…' : 'Ask'}
            </button>
            <ResultDisplay result={results.question} error={errors.question} />
          </ActionCard>

          {/* ---------- TEST QUESTIONS ---------- */}
          <ActionCard title="Get Test Questions" icon={<TestTube size={20} />}>
            <button
              type="button"
              onClick={loadTestQuestions}
              disabled={loading.testQuestions}
              style={{
                backgroundColor: '#0ea5e9',
                color: 'white',
                border: 'none',
                borderRadius: 4,
                padding: '6px 12px',
                cursor: loading.testQuestions ? 'not-allowed' : 'pointer',
                opacity: loading.testQuestions ? 0.6 : 1,
                marginBottom: 8,
              }}
            >
              {loading.testQuestions ? 'Loading…' : 'Load'}
            </button>
            <ResultDisplay
              result={testQuestions}
              error={errors.testQuestions}
            />
          </ActionCard>

          {/* ---------- DB STATUS ---------- */}
          <ActionCard title="DB Status" icon={<Database size={20} />}>
            <button
              type="button"
              onClick={loadDbStatus}
              disabled={loading.dbStatus}
              style={{
                backgroundColor: '#6b7280',
                color: 'white',
                border: 'none',
                borderRadius: 4,
                padding: '6px 12px',
                cursor: loading.dbStatus ? 'not-allowed' : 'pointer',
                opacity: loading.dbStatus ? 0.6 : 1,
                marginBottom: 8,
              }}
            >
              {loading.dbStatus ? 'Refreshing…' : 'Refresh'}
            </button>
            <ResultDisplay result={dbStatus} error={errors.dbStatus} />
          </ActionCard>

          {/* ---------- DELETE COLLECTION ---------- */}
          <ActionCard title="Delete Collection" icon={<Trash2 size={20} />}>
            <button
              type="button"
              onClick={deleteCollection}
              disabled={loading.delete}
              style={{
                backgroundColor: '#ef4444',
                color: 'white',
                border: 'none',
                borderRadius: 4,
                padding: '6px 12px',
                cursor: loading.delete ? 'not-allowed' : 'pointer',
                opacity: loading.delete ? 0.6 : 1,
              }}
            >
              {loading.delete ? 'Deleting…' : 'Delete'}
            </button>
            <ResultDisplay result={results.delete} error={errors.delete} />
          </ActionCard>
        </section>
      </div>
    </div>
  );
};

export default App;