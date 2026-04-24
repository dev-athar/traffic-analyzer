import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:8000",
});

export async function uploadVideo(
  file,
  reidMode,
  lineMode,
  lineSinglePos,
  lineDualA,
  lineDualB
) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("reid_mode", reidMode);
  formData.append("line_mode", lineMode);
  formData.append("line_single_pos", lineSinglePos);
  formData.append("line_dual_a_pos", lineDualA);
  formData.append("line_dual_b_pos", lineDualB);

  const response = await api.post("/upload", formData, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
  return response.data;
}

export async function getStatus(jobId) {
  const response = await api.get(`/status/${jobId}`);
  return response.data;
}

export async function getResult(jobId) {
  const response = await api.get(`/result/${jobId}`);
  return response.data;
}

export function connectProgressWS(jobId, onMessage, onError) {
  const ws = new WebSocket(`ws://localhost:8000/ws/${jobId}`);

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch {
      onError?.();
    }
  };

  ws.onerror = () => {
    onError?.();
  };

  ws.onclose = () => {
    onError?.();
  };

  return ws;
}
