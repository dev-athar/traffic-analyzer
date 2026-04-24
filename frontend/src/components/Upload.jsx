import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { uploadVideo } from "../api";

function LinePreview({ frameDataUrl, aspectRatio, lineMode, lineSinglePos, lineDualA, lineDualB }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !frameDataUrl) return;
    const img = new Image();
    img.src = frameDataUrl;
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext("2d");

      ctx.drawImage(img, 0, 0);

      if (lineMode === "single") {
        const y = Math.floor(img.height * lineSinglePos);
        ctx.strokeStyle = "#FFFFFF";
        ctx.lineWidth = 15;
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(img.width, y);
        ctx.stroke();
        ctx.fillStyle = "rgba(0,0,0,0.55)";
        ctx.fillRect(0, y - 100, 1000, 100);
        ctx.fillStyle = "#F5C518";
        ctx.font = "bold 55px JetBrains Mono, monospace";
        ctx.fillText(
          `COUNTING LINE — ${Math.round(lineSinglePos * 100)}% FROM TOP`,
          40, y - 50
        );
      } else {
        const yA = Math.floor(img.height * lineDualA);
        ctx.strokeStyle = "#F5C518";
        ctx.lineWidth = 15;
        ctx.beginPath();
        ctx.moveTo(0, yA);
        ctx.lineTo(img.width, yA);
        ctx.stroke();
        ctx.fillStyle = "rgba(0,0,0,0.55)";
        ctx.fillRect(0, yA - 100, 900, 100);
        ctx.fillStyle = "#F5C518";
        ctx.font = "bold 55px JetBrains Mono, monospace";
        ctx.fillText(
          `LINE A — ${Math.round(lineDualA * 100)}% FROM TOP`,
          40, yA - 50
        );

        const yB = Math.floor(img.height * lineDualB);
        ctx.strokeStyle = "#FFFFFF";
        ctx.lineWidth = 15;
        ctx.beginPath();
        ctx.moveTo(0, yB);
        ctx.lineTo(img.width, yB);
        ctx.stroke();
        ctx.fillStyle = "rgba(0,0,0,0.55)";
        ctx.fillRect(0, yB + 14, 900, 100);
        ctx.fillStyle = "#FFFFFF";
        ctx.font = "bold 55px JetBrains Mono, monospace";
        ctx.fillText(
          `LINE B — ${Math.round(lineDualB * 100)}% FROM TOP`,
          40, yB + 78
        );
      }
    };
  }, [frameDataUrl, lineMode, lineSinglePos, lineDualA, lineDualB]);

  return <canvas ref={canvasRef} style={{ width: "100%" }} />;
}

export default function Upload() {
  const [file, setFile] = useState(null);
  const [reidMode, setReidMode] = useState("fast");
  const [lineMode, setLineMode] = useState("dual");
  const [lineSinglePos, setLineSinglePos] = useState(50);
  const [lineDualA, setLineDualA] = useState(45);
  const [lineDualB, setLineDualB] = useState(80);
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadDots, setUploadDots] = useState(".");
  const [dualLineWarning, setDualLineWarning] = useState(false);
  const [error, setError] = useState("");
  const [previewFrame, setPreviewFrame] = useState(null);
  const [aspectRatio, setAspectRatio] = useState(null);
  const fileInputRef = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    if (!uploading) {
      setUploadDots(".");
      return undefined;
    }
    const id = window.setInterval(() => {
      setUploadDots((prev) => (prev.length >= 3 ? "." : `${prev}.`));
    }, 350);
    return () => window.clearInterval(id);
  }, [uploading]);

  function onSelectFile(nextFile) {
    if (!nextFile) return;
    if (!nextFile.name.toLowerCase().endsWith(".mp4")) {
      setError("Only .mp4 files are accepted.");
      return;
    }
    setError("");
    setFile(nextFile);

    const videoEl = document.createElement("video");
    videoEl.src = URL.createObjectURL(nextFile);
    videoEl.muted = true;
    videoEl.preload = "metadata";

    videoEl.addEventListener("loadedmetadata", () => {
      videoEl.currentTime = videoEl.duration * 0.10;
    });

    videoEl.addEventListener("seeked", () => {
      const canvas = document.createElement("canvas");
      canvas.width = videoEl.videoWidth;
      canvas.height = videoEl.videoHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(videoEl, 0, 0);
      const frameDataUrl = canvas.toDataURL("image/jpeg");
      setPreviewFrame(frameDataUrl);
      setAspectRatio(videoEl.videoWidth / videoEl.videoHeight);
      URL.revokeObjectURL(videoEl.src);
    });
  }

  function clearFile() {
    setFile(null);
    setPreviewFrame(null);
    setAspectRatio(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  function formatBytes(bytes = 0) {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function updateDualA(value) {
    const next = Number(value);
    setDualLineWarning(next >= lineDualB);
    setLineDualA(Math.min(next, lineDualB - 1));
  }

  function updateDualB(value) {
    const next = Number(value);
    setDualLineWarning(next <= lineDualA);
    setLineDualB(Math.max(next, lineDualA + 1));
  }

  async function handleUpload() {
    if (!file) {
      setError("Please select an .mp4 file.");
      return;
    }
    setError("");
    setUploading(true);
    try {
      const data = await uploadVideo(
        file,
        reidMode,
        lineMode,
        lineSinglePos / 100,
        lineDualA / 100,
        lineDualB / 100
      );
      navigate(`/processing/${data.job_id}`, {
        state: {
          reidMode,
          lineMode,
          lineSinglePos: lineSinglePos / 100,
          lineDualA: lineDualA / 100,
          lineDualB: lineDualB / 100,
        },
      });
    } catch (err) {
      setError(err.response?.data?.detail ?? "Upload failed. Please try again.");
    } finally {
      setUploading(false);
    }
  }

  return (
    <main className="page page--narrow">
      <section className="section-heading">
        <h1 className="section-heading__title">UPLOAD FOOTAGE</h1>
        <p className="section-heading__subtitle">
          Supported format: .MP4 - Max recommended: 2GB
        </p>
      </section>

      <label
        className={`drop-zone ${dragActive ? "drop-zone--drag" : ""}`}
        style={previewFrame ? { minHeight: "unset" } : {}}
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragActive(false);
          onSelectFile(e.dataTransfer.files?.[0]);
        }}
      >
        {previewFrame ? (
          <>
            <LinePreview
              frameDataUrl={previewFrame}
              aspectRatio={aspectRatio}
              lineMode={lineMode}
              lineSinglePos={lineSinglePos / 100}
              lineDualA={lineDualA / 100}
              lineDualB={lineDualB / 100}
            />
            <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "11px", marginTop: "8px", color: "var(--asphalt, #2A2A2A)" }}>
              {file.name} — {formatBytes(file.size)}
            </div>
            <button
              type="button"
              onClick={(e) => { e.preventDefault(); clearFile(); }}
              style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "11px", marginTop: "4px", background: "none", border: "none", cursor: "pointer", color: "var(--asphalt, #2A2A2A)", textDecoration: "underline" }}
            >
              ✕ REMOVE
            </button>
          </>
        ) : (
          <>
            <svg className="drone-icon" width="72" height="52" viewBox="0 0 72 52">
              <rect x="28" y="20" width="16" height="12" fill="none" stroke="var(--black)" strokeWidth="2" />
              <line x1="20" y1="10" x2="28" y2="20" stroke="var(--black)" strokeWidth="2" />
              <line x1="52" y1="10" x2="44" y2="20" stroke="var(--black)" strokeWidth="2" />
              <line x1="20" y1="42" x2="28" y2="32" stroke="var(--black)" strokeWidth="2" />
              <line x1="52" y1="42" x2="44" y2="32" stroke="var(--black)" strokeWidth="2" />
              <rect x="8" y="2" width="12" height="12" fill="none" stroke="var(--black)" strokeWidth="2" />
              <rect x="52" y="2" width="12" height="12" fill="none" stroke="var(--black)" strokeWidth="2" />
              <rect x="8" y="38" width="12" height="12" fill="none" stroke="var(--black)" strokeWidth="2" />
              <rect x="52" y="38" width="12" height="12" fill="none" stroke="var(--black)" strokeWidth="2" />
            </svg>
            <div className="drop-zone__title">DROP .MP4 HERE</div>
            <div className="drop-zone__hint">or click to browse</div>
          </>
        )}
        <input
          ref={fileInputRef}
          type="file"
          accept=".mp4"
          onChange={(e) => onSelectFile(e.target.files[0])}
          hidden
        />
      </label>

      {file && !previewFrame && (
        <div className="file-tag">
          {file.name} - {formatBytes(file.size)}
        </div>
      )}

      <section className="section-block">
        <div className="section-heading">
          <h2 className="section-heading__title" style={{ fontSize: "22px" }}>
            PROCESSING CONFIG
          </h2>
        </div>

        <div className="segmented">
          <button
            type="button"
            className={`nb-btn ${reidMode === "fast" ? "nb-btn--selected" : ""}`}
            onClick={() => setReidMode("fast")}
          >
            FAST
          </button>
          <button
            type="button"
            className={`nb-btn ${reidMode === "accurate" ? "nb-btn--selected" : ""}`}
            onClick={() => setReidMode("accurate")}
          >
            ACCURATE
          </button>
        </div>

        {reidMode === "accurate" && (
          <div className="disclaimer">
            High-accuracy model enabled. Expect significantly longer processing time.
          </div>
        )}

        <div className="slider-wrap">
          <div style={{ fontSize: "12px", marginBottom: "8px", fontWeight: "700" }}>
            COUNTING LINE MODE
          </div>
          <div className="segmented">
            <button
              type="button"
              className={`nb-btn ${lineMode === "single" ? "nb-btn--selected" : ""}`}
              onClick={() => setLineMode("single")}
            >
              SINGLE LINE
            </button>
            <button
              type="button"
              className={`nb-btn ${lineMode === "dual" ? "nb-btn--selected" : ""}`}
              onClick={() => setLineMode("dual")}
            >
              DUAL LINE
            </button>
          </div>
        </div>

        {lineMode === "single" ? (
          <div className="slider-row">
            <div className="slider-label">
              <span>LINE POSITION</span>
              <span>{lineSinglePos}% FROM TOP</span>
            </div>
            <input
              className="custom-slider"
              style={{ "--progress": `${lineSinglePos}%` }}
              type="range"
              min="10"
              max="90"
              value={lineSinglePos}
              onChange={(e) => setLineSinglePos(Number(e.target.value))}
            />
            <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "11px", color: "var(--asphalt, #2A2A2A)", borderLeft: "4px solid var(--concrete, #C8C8BE)", paddingLeft: "10px", marginTop: "6px" }}>
              This line acts as the vehicle trigger. Any vehicle whose path crosses this line is counted once. Position it across the main traffic flow for best accuracy.
            </div>
          </div>
        ) : (
          <div className="slider-row">
            <div className="slider-label">
              <span>LINE A POSITION</span>
              <span>{lineDualA}% FROM TOP</span>
            </div>
            <input
              className="custom-slider"
              style={{ "--progress": `${lineDualA}%` }}
              type="range"
              min="10"
              max="90"
              value={lineDualA}
              onChange={(e) => updateDualA(e.target.value)}
            />
            <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "11px", color: "var(--asphalt, #2A2A2A)", borderLeft: "4px solid var(--concrete, #C8C8BE)", paddingLeft: "10px", marginTop: "6px" }}>
              LINE A acts as the entry trigger. Vehicles crossing from above are counted here. Place above the main traffic zone.
            </div>

            <div className="slider-label" style={{ marginTop: "12px" }}>
              <span>LINE B POSITION</span>
              <span>{lineDualB}% FROM TOP</span>
            </div>
            <input
              className="custom-slider"
              style={{ "--progress": `${lineDualB}%` }}
              type="range"
              min="10"
              max="90"
              value={lineDualB}
              onChange={(e) => updateDualB(e.target.value)}
            />
            <div style={{ fontFamily: "JetBrains Mono, monospace", fontSize: "11px", color: "var(--asphalt, #2A2A2A)", borderLeft: "4px solid var(--concrete, #C8C8BE)", paddingLeft: "10px", marginTop: "6px" }}>
              LINE B acts as the exit trigger. Vehicles that missed Line A (entered mid-frame) are caught here. Place below the main traffic zone.
            </div>

            {dualLineWarning && (
              <div className="warning-tag">LINE A MUST BE ABOVE LINE B</div>
            )}
          </div>
        )}
      </section>

      <button
        type="button"
        className="main-upload-btn"
        onClick={handleUpload}
        disabled={!file || uploading}
      >
        {uploading ? `UPLOADING${uploadDots}` : "INITIALIZE ANALYSIS"}
      </button>

      {error && <div className="error-bar">{error}</div>}
    </main>
  );
}
