import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useLocation, useNavigate, useParams } from "react-router-dom";
import { connectProgressWS, getStatus } from "../api";

export default function Progress() {
  const { jobId } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const [status, setStatus] = useState("pending");
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState("");
  const [reconnecting, setReconnecting] = useState(false);
  const [frames, setFrames] = useState({ processed: null, total: null });
  const [meta, setMeta] = useState({
    reidMode: location.state?.reidMode || "fast",
    lineMode: location.state?.lineMode || "dual",
    lineSinglePos: location.state?.lineSinglePos,
    lineDualA: location.state?.lineDualA,
    lineDualB: location.state?.lineDualB,
  });
  const timeoutRef = useRef(null);
  const metaRef = useRef(meta);

  useEffect(() => {
    metaRef.current = meta;
  }, [meta]);

  const statusMessage = useMemo(() => {
    if (status === "pending") {
      return "INITIALIZING MODEL...";
    }
    if (status === "processing") {
      return progress >= 95 ? "GENERATING REPORT..." : "PROCESSING FRAMES...";
    }
    if (status === "complete") {
      return "ANALYSIS COMPLETE";
    }
    if (status === "error") {
      return "PROCESSING FAILED";
    }
    return "GENERATING REPORT...";
  }, [status, progress]);

  useEffect(() => {
    let mounted = true;
    let ws;

    const hydrateStatus = async () => {
      try {
        const current = await getStatus(jobId);
        if (!mounted) {
          return;
        }
        setStatus(current.status || "pending");
        setProgress(Math.round(current.progress || 0));
        if (current.error) {
          setError(current.error);
        }
      } catch {
        // If status fetch fails, websocket still handles updates.
      }
    };

    const connect = () => {
      ws = connectProgressWS(
        jobId,
        (data) => {
          if (!mounted) {
            return;
          }
          setReconnecting(false);
          if (data.reid_mode || data.line_mode) {
            setMeta((prev) => ({
              reidMode: data.reid_mode || prev.reidMode,
              lineMode: data.line_mode || prev.lineMode,
            }));
          }
          if (typeof data.progress === "number") {
            setProgress(Math.round(data.progress));
          }
          if (data.status) {
            setStatus(data.status);
          }
          if (data.error) {
            setError(data.error);
          }
          if (data.status === "complete") {
            timeoutRef.current = window.setTimeout(() => {
              navigate(`/result/${jobId}`, { state: metaRef.current });
            }, 1500);
          }
          if (
            typeof data.processed_frames === "number" &&
            typeof data.total_frames === "number"
          ) {
            setFrames({ processed: data.processed_frames, total: data.total_frames });
          }
        },
        () => {
          if (!mounted) {
            return;
          }
          setReconnecting(true);
        }
      );
    };

    hydrateStatus();
    connect();
    return () => {
      mounted = false;
      ws?.close();
      if (timeoutRef.current) {
        window.clearTimeout(timeoutRef.current);
      }
    };
  }, [jobId, navigate]);

  return (
    <main className="page page--narrow">
      <Link className="back-link" to="/">
        {"<- BACK TO UPLOAD"}
      </Link>

      <div className="processing-head">
        <h1 className="section-heading__title">PROCESSING</h1>
        <div className="job-tag">JOB ID: {jobId}</div>
      </div>

      <div className="badge-row">
        <div className="small-tag">
          MODE: {meta.reidMode === "accurate" ? "ACCURATE" : "FAST"}
        </div>
        <div className="small-tag">
          LINES: {meta.lineMode === "dual" ? "DUAL" : "SINGLE"}
        </div>
      </div>

      {status === "error" ? (
        <div className="fail-box">
          <h2 className="fail-title">PROCESSING FAILED</h2>
          <p className="status-line">{error || "Unknown processing error."}</p>
          <button
            type="button"
            className="nb-btn"
            style={{ marginTop: "12px", maxWidth: "220px" }}
            onClick={() => navigate("/")}
          >
            UPLOAD NEW VIDEO
          </button>
        </div>
      ) : (
        <>
          <div className="progress-percent">
            <span className="progress-percent__num">{progress}</span>
            <span className="progress-percent__unit">%</span>
          </div>
          <div className="progress-bar">
            <div
              className="progress-bar__fill"
              style={{ width: `${Math.min(100, Math.max(progress, 0))}%` }}
            />
          </div>
          <p className="status-line">{statusMessage}</p>
          {typeof frames.processed === "number" && typeof frames.total === "number" && (
            <p className="frame-line">
              FRAMES: {frames.processed} / {frames.total}
            </p>
          )}
          {reconnecting && <p className="reconnecting">WS RECONNECTING...</p>}
        </>
      )}
    </main>
  );
}
