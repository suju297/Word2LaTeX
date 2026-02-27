"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";

type Stage = "idle" | "uploading" | "converting" | "packaging" | "done" | "error";

type Stats = {
  visitorCount: number;
  conversionCount: number;
};

const DEFAULT_PROFILE = "auto";

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB"];
  const base = 1024;
  const index = Math.min(
    units.length - 1,
    Math.floor(Math.log(bytes) / Math.log(base)),
  );
  const value = bytes / Math.pow(base, index);
  const precision = value >= 10 || index === 0 ? 0 : 1;
  return `${value.toFixed(precision)} ${units[index]}`;
}

const STEPS = [
  {
    title: "Upload",
    desc: "Securely send the DOCX for conversion.",
  },
  {
    title: "Convert",
    desc: "Preserve structure while mapping to LaTeX.",
  },
  {
    title: "Package",
    desc: "Package everything into a ZIP download.",
  },
];

const STAGE_TO_INDEX: Record<Stage, number> = {
  idle: -1,
  uploading: 0,
  converting: 1,
  packaging: 2,
  done: 2,
  error: 2,
};

const STAGE_PROGRESS: Record<Stage, number> = {
  idle: 0,
  uploading: 33,
  converting: 66,
  packaging: 99,
  done: 100,
  error: 80,
};

const STATUS_LABELS: Record<Stage, string> = {
  idle: "Idle",
  uploading: "Uploading",
  converting: "Converting",
  packaging: "Packaging",
  done: "Done",
  error: "Error",
};

const STATUS_CLASSES: Record<Stage, string> = {
  idle: "pill",
  uploading: "pill active",
  converting: "pill active",
  packaging: "pill active",
  done: "pill done",
  error: "pill error",
};

function getDownloadName(
  disposition: string | null,
  fallback: string,
): string {
  if (!disposition) {
    return fallback;
  }
  const match =
    /filename\*=UTF-8''([^;]+)|filename=\"?([^\";]+)\"?/i.exec(disposition);
  const rawName = match?.[1] || match?.[2];
  if (!rawName) {
    return fallback;
  }
  try {
    return decodeURIComponent(rawName);
  } catch {
    return rawName;
  }
}

export default function Home() {
  const [stats, setStats] = useState<Stats>({
    visitorCount: 0,
    conversionCount: 0,
  });
  const [stage, setStage] = useState<Stage>("idle");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const [downloadUrl, setDownloadUrl] = useState("");
  const [downloadName, setDownloadName] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const activeIndex = STAGE_TO_INDEX[stage];
  const progress = STAGE_PROGRESS[stage];
  const statusLabel = STATUS_LABELS[stage];
  const statusClass = STATUS_CLASSES[stage];

  useEffect(() => {
    const controller = new AbortController();
    const loadStats = async () => {
      try {
        const response = await fetch("/api/stats", {
          cache: "no-store",
          signal: controller.signal,
        });
        if (!response.ok) {
          return;
        }
        const data = await response.json();
        setStats((prev) => ({
          visitorCount: data.visitor_count ?? prev.visitorCount,
          conversionCount: data.conversion_count ?? prev.conversionCount,
        }));
      } catch {
        // Ignore stats errors on the client.
      }
    };
    loadStats();
    return () => controller.abort();
  }, []);

  useEffect(() => {
    return () => {
      if (downloadUrl) {
        URL.revokeObjectURL(downloadUrl);
      }
    };
  }, [downloadUrl]);

  const resetForm = () => {
    setSelectedFile(null);
    setStage("idle");
    setErrorMessage("");
    setSuccessMessage("");
    setDownloadName("");
    if (downloadUrl) {
      URL.revokeObjectURL(downloadUrl);
      setDownloadUrl("");
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!selectedFile || isSubmitting) {
      return;
    }
    setIsSubmitting(true);
    setErrorMessage("");
    setSuccessMessage("");
    setDownloadName("");
    if (downloadUrl) {
      URL.revokeObjectURL(downloadUrl);
      setDownloadUrl("");
    }

    const formData = new FormData();
    formData.append("docx", selectedFile);
    formData.append(
      "options_json",
      JSON.stringify({
        profile: DEFAULT_PROFILE,
        dynamic: true,
        header_fallback: true,
      }),
    );

    try {
      setStage("uploading");
      const responsePromise = fetch("/api/convert", {
        method: "POST",
        body: formData,
      });
      setStage("converting");
      const response = await responsePromise;

      if (!response.ok) {
        const errorPayload = await response.json().catch(() => null);
        setErrorMessage(
          errorPayload?.message ||
            errorPayload?.detail ||
            "Conversion failed. Please check your file and try again.",
        );
        setStage("error");
        return;
      }

      setStage("packaging");
      const fallbackExt = ".zip";
      const baseName =
        selectedFile.name.replace(/\.docx$/i, "") || "wordtolatex-output";
      const fallbackName = `${baseName}${fallbackExt}`;
      const filename = getDownloadName(
        response.headers.get("content-disposition"),
        fallbackName,
      );

      const statsPromise = fetch("/api/stats", { cache: "no-store" })
        .then((statsResponse) =>
          statsResponse.ok ? statsResponse.json() : null,
        )
        .catch(() => null);

      const [blob, statsData] = await Promise.all([
        response.blob(),
        statsPromise,
      ]);
      const url = URL.createObjectURL(blob);
      setDownloadUrl(url);
      setDownloadName(filename);
      setSuccessMessage("Your ZIP is ready to download.");
      setStage("done");

      if (statsData) {
        setStats((prev) => ({
          visitorCount: statsData.visitor_count ?? prev.visitorCount,
          conversionCount: statsData.conversion_count ?? prev.conversionCount,
        }));
      }
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "Something went wrong.",
      );
      setStage("error");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="page">
      <div className="stack">
        <header className="hero">
          <span className="eyebrow">Word to LaTeX</span>
          <h1>Forge polished LaTeX from DOCX files.</h1>
          <p>
            Upload a Word document and receive a clean, editable LaTeX file
            without exposing any pipeline details.
          </p>
          <div className="stats">
            <div className="stat">
              <div className="stat-value">{stats.visitorCount}</div>
              <div className="stat-label">Visitors</div>
            </div>
            <div className="stat">
              <div className="stat-value">{stats.conversionCount}</div>
              <div className="stat-label">Conversions</div>
            </div>
          </div>
        </header>

        <main className="layout">
          <section className="card upload-card">
            <h2>Upload DOCX</h2>
            <p className="hint">
              Drop a Word file and we will return a ZIP containing the .tex
              (and any images). Layout is detected automatically.
            </p>

            <form onSubmit={handleSubmit}>
              <div className="field">
                <label htmlFor="docx">DOCX file</label>
                <input
                  ref={fileInputRef}
                  id="docx"
                  name="docx"
                  type="file"
                  accept=".docx"
                  required
                  className="file-input"
                  onChange={(event) =>
                    setSelectedFile(event.target.files?.[0] ?? null)
                  }
                />
                {selectedFile ? (
                  <div className="file-meta">
                    <span className="file-name">{selectedFile.name}</span>
                    <span className="file-size">
                      {formatBytes(selectedFile.size)}
                    </span>
                  </div>
                ) : null}
              </div>

              <div className="actions">
                <button
                  className="primary"
                  type="submit"
                  disabled={!selectedFile || isSubmitting}
                >
                  Convert to LaTeX
                </button>
                <button className="secondary" type="button" onClick={resetForm}>
                  Reset
                </button>
              </div>

              <div
                className="progress"
                style={{ ["--progress" as string]: `${progress}%` }}
              >
                <div className="progress-header">
                  <strong>Processing</strong>
                  <span className={statusClass}>{statusLabel}</span>
                </div>
                <div className="progress-track">
                  <div className="progress-bar" />
                </div>
                <div className="steps">
                  {STEPS.map((step, index) => {
                    let state: "pending" | "active" | "done" | "error" =
                      "pending";
                    if (stage === "done") {
                      state = "done";
                    } else if (stage === "error") {
                      state = index < activeIndex ? "done" : "error";
                    } else if (index < activeIndex) {
                      state = "done";
                    } else if (index === activeIndex) {
                      state = "active";
                    }
                    return (
                      <div key={step.title} className="step" data-state={state}>
                        <span className="step-dot" />
                        <div>
                          <div className="step-label">{step.title}</div>
                          <div className="step-desc">{step.desc}</div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {errorMessage ? (
                <div className="status error">
                  <div className="status-title">Conversion failed</div>
                  <p className="hint">{errorMessage}</p>
                  {errorMessage.includes("Reference PDF generation failed") ? (
                    <p className="hint">
                      Install LibreOffice or set REFERENCE_PDF_COMMAND on the
                      server to enable reference PDF creation.
                    </p>
                  ) : null}
                </div>
              ) : null}

              {successMessage && downloadUrl ? (
                <div className="status success">
                  <div className="status-title">{successMessage}</div>
                  <div className="download">
                    <a className="primary" href={downloadUrl} download={downloadName}>
                      Download {downloadName}
                    </a>
                    <button className="ghost" type="button" onClick={resetForm}>
                      Convert another file
                    </button>
                  </div>
                </div>
              ) : null}
            </form>
          </section>

          <section className="card expectations">
            <h3>What to expect</h3>
            <ul>
              <li>ZIP download containing an editable .tex file (plus images if needed).</li>
              <li>Structure-preserving conversion for resumes and reports.</li>
              <li>Compile with XeLaTeX or LuaLaTeX (Overleaf: switch compiler).</li>
              <li>Per-IP cooldown: 30 minutes between conversions.</li>
              <li>No login required.</li>
            </ul>
            <div className="badge">Built for fast iteration</div>
          </section>
        </main>

        <footer className="footer">
          <span>Get a clean ZIP with your .tex in seconds.</span>
          <span>Questions or feedback? Reach out.</span>
        </footer>
      </div>
    </div>
  );
}
