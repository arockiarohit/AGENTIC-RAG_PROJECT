import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export const uploadDocument = async (file, onProgress) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await axios.post(`${API_BASE}/upload`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (progressEvent) => {
      if (onProgress) {
        const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        onProgress(percent);
      }
    },
  });
  return response.data;
};

let sessionId = null;
const getSessionId = () => {
  if (!sessionId) {
    sessionId = 'sess-' + Math.random().toString(36).substring(2, 9);
  }
  return sessionId;
};

export const sendMessage = async (question) => {
  const response = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, session_id: getSessionId() })
  });

  if (!response.ok) {
    throw new Error('Network response was not ok');
  }

  const contentType = response.headers.get('content-type');
  if (contentType && contentType.includes('application/json')) {
    return await response.json();
  }

  const text = await response.text();
  try {
    return JSON.parse(text);
  } catch (e) {
    return { answer: text, sources: [], route: 'general', image_url: null };
  }
};

export const sendMessageStream = async (question, onChunk) => {
  const data = await sendMessage(question);
  if (onChunk) {
    onChunk(data);
  }
  return data;
};

export const checkHealth = async () => {
  const response = await axios.get(`${API_BASE}/health`);
  return response.data;
};
