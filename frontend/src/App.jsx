import { BrowserRouter, Routes, Route } from "react-router-dom";
import Upload from "./components/Upload";
import Progress from "./components/Progress";
import Results from "./components/Results";

export default function App() {
  return (
    <div className="app-shell">
      <BrowserRouter>
        <nav className="top-nav">
          <div className="top-nav__brand">▣ DRONE TRAFFIC ANALYZER</div>
          <div className="sys-badge">
            <span className="sys-badge__dot" />
            <span>SYS:ONLINE</span>
          </div>
        </nav>
        <Routes>
          <Route path="/" element={<Upload />} />
          <Route path="/processing/:jobId" element={<Progress />} />
          <Route path="/result/:jobId" element={<Results />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}
