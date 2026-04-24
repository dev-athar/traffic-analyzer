import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import { getResult } from "../api";

export default function Results() {
  const { jobId } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let mounted = true;
    const fetchData = async () => {
      try {
        const data = await getResult(jobId);
        if (!mounted) {
          return;
        }
        setResult(data);
      } catch (err) {
        setError(err.response?.data?.detail || "Result not available.");
      }
    };
    fetchData();
    return () => {
      mounted = false;
    };
  }, [jobId]);

  const classCounts = useMemo(() => {
    const raw = result?.count_by_class || {};
    return {
      car: raw.car || 0,
      truck: raw.truck || 0,
      bus: raw.bus || 0,
      motorcycle: raw.motorcycle || 0,
    };
  }, [result]);

  const modeLabel = (result?.reid_mode || location.state?.reidMode || "fast").toUpperCase();

  const lineConfig =
    result?.line_mode === "dual"
      ? `dual ${Math.round((location.state?.lineDualA ?? 0.45) * 100)}%/${Math.round(
          (location.state?.lineDualB ?? 0.80) * 100
        )}%`
      : `single ${Math.round((location.state?.lineSinglePos ?? 0.5) * 100)}%`;

  if (error) {
    return (
      <main className="page page--wide">
        <div className="fail-box">
          <h2 className="fail-title">RESULT LOAD FAILED</h2>
          <p className="status-line">{error}</p>
          <button
            type="button"
            className="nb-btn"
            style={{ marginTop: "12px", maxWidth: "220px" }}
            onClick={() => navigate("/")}
          >
            UPLOAD NEW VIDEO
          </button>
        </div>
      </main>
    );
  }

  if (!result) {
    return (
      <main className="page page--wide">
        <p className="status-line">LOADING RESULT...</p>
      </main>
    );
  }

  return (
    <main className="page page--wide">
      <div className="result-grid">
        <section className="video-card">
          <video
            className="video-player"
            controls
            src={`http://localhost:8000/static/${jobId}_processed.mp4`}
          />
          <div className="download-row">
            <button
              type="button"
              className="nb-btn"
              onClick={() =>
                window.open(`/api/report/${jobId}?format=csv`, "_blank")
              }
            >
              ↓ EXPORT CSV
            </button>
            <button
              type="button"
              className="nb-btn nb-btn--selected"
              onClick={() =>
                window.open(`/api/report/${jobId}?format=excel`, "_blank")
              }
            >
              ↓ EXPORT EXCEL
            </button>
          </div>
        </section>

        <aside className="stats-card">
          <h2 className="stats-header">ANALYSIS REPORT</h2>
          <p className="meta-strip">
            JOB: {jobId} | MODE: {modeLabel} | LINES: {result.line_mode}
          </p>

          <div className="hero-stat">
            <div className="hero-stat__num">{result.total_unique || 0}</div>
            <div className="hero-stat__label">TOTAL VEHICLES COUNTED</div>
          </div>

          <div className="class-grid">
            <div className="class-cell" style={{ borderLeftColor: "var(--green)" }}>
              <div className="class-cell__count">{classCounts.car}</div>
              <div className="class-cell__label">CAR</div>
            </div>
            <div className="class-cell" style={{ borderLeftColor: "var(--red)" }}>
              <div className="class-cell__count">{classCounts.truck}</div>
              <div className="class-cell__label">TRUCK</div>
            </div>
            <div className="class-cell" style={{ borderLeftColor: "var(--yellow)" }}>
              <div className="class-cell__count">{classCounts.bus}</div>
              <div className="class-cell__label">BUS</div>
            </div>
            <div className="class-cell" style={{ borderLeftColor: "var(--asphalt)" }}>
              <div className="class-cell__count">{classCounts.motorcycle}</div>
              <div className="class-cell__label">MOTORCYCLE</div>
            </div>
          </div>

          <div className="meta-section">
            <h3 className="meta-title">RUN METADATA</h3>
            <div className="meta-list">
              <div className="meta-item">DURATION {result.processing_duration_sec?.toFixed(1)}s</div>
              <div className="meta-item">FPS {result.fps}</div>
              <div className="meta-item">
                RESOLUTION {result.width}×{result.height}
              </div>
              <div className="meta-item">
                FRAMES {result.processed_frames}/{result.total_frames}
              </div>
              <div className="meta-item">MODEL MODE {modeLabel === "ACCURATE" ? "YOLOV8S" : "YOLOV8N"}</div>
              <div className="meta-item">LINE CONFIG {lineConfig}</div>
            </div>
          </div>

          <button
            type="button"
            className="main-upload-btn new-analysis-btn"
            onClick={() => navigate("/")}
          >
            ▣ START NEW ANALYSIS
          </button>
        </aside>
      </div>
    </main>
  );
}
