import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';

type TrafficLight = 'red' | 'yellow' | 'green' | null;
type LaneState = 'steady' | 'changing-left' | 'changing-right' | null;
type Gear = 'P' | 'R' | 'N' | 'D';
type AlertTone = 'danger' | 'caution' | 'neutral' | 'go';
type Tab = 'live' | 'image' | 'video' | 'compare';

type DetectionBox = {
  label: string;
  x: number;
  y: number;
  width: number;
  height: number;
  confidence?: number;
  ttc_s?: number | null;
};

export type DetectionPacket = {
  source?: 'model';
  vehicle?: {
    detected: boolean;
    distanceM: number;
    position: 'front' | 'rear-left' | 'rear-right';
  } | null;
  rain?: boolean;
  lane?: LaneState;
  trafficLight?: TrafficLight;
  speedLimit?: number | null;
  roadSurface?: 'dry' | 'wet' | null;
  gear?: Gear;
  speedKph?: number;
  objects?: DetectionBox[];
};

type ActiveAlert = {
  key: string;
  tone: AlertTone;
  category: string;
  title: string;
  message: string;
};

const EMPTY_DETECTION: DetectionPacket = {
  source: 'model',
  vehicle: null,
  rain: false,
  lane: null,
  trafficLight: null,
  speedLimit: null,
  roadSurface: null,
  gear: 'D',
  speedKph: 0,
  objects: [],
};

function getAlert(detection: DetectionPacket): ActiveAlert | null {
  if (detection.vehicle?.detected) {
    const position = detection.vehicle.position === 'rear-right' ? 'Rear right' : detection.vehicle.position === 'rear-left' ? 'Rear left' : 'Ahead';
    return {
      key: 'vehicle', tone: 'danger', category: 'Collision warning',
      title: 'Vehicle approaching', message: `${position} / ${detection.vehicle.distanceM.toFixed(1)} m`,
    };
  }
  if (detection.rain) {
    return { key: 'rain', tone: 'caution', category: 'Weather warning', title: 'Rain detected', message: 'Slow down and increase following distance' };
  }
  if (detection.lane === 'changing-left' || detection.lane === 'changing-right') {
    return { key: 'lane', tone: 'neutral', category: 'Lane assistance', title: 'Lane change detected', message: 'Check mirrors and maintain a safe gap' };
  }
  if (detection.trafficLight === 'red') {
    return { key: 'red-light', tone: 'danger', category: 'Traffic signal', title: 'Red light up ahead', message: 'Stop at the line' };
  }
  if (detection.trafficLight === 'yellow') {
    return { key: 'yellow-light', tone: 'caution', category: 'Traffic signal', title: 'Yellow light incoming', message: 'Slow down and prepare to stop' };
  }
  if (detection.trafficLight === 'green') {
    return { key: 'green-light', tone: 'go', category: 'Traffic signal', title: 'Green light', message: 'Go when the intersection is clear' };
  }
  return null;
}

/* ------------ small panel helpers (styled to match the cabin) ------------ */
function StatChip({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <b>{value}</b>
    </div>
  );
}

function TaskGrid({ tasks }: { tasks: Record<string, any> | null }) {
  if (!tasks) return null;
  return (
    <div className="task-grid">
      {Object.entries(tasks).map(([k, v]) => (
        <div className="task-cell" key={k}>
          <div className="task-head">{k}</div>
          {renderTaskValue(k, v)}
        </div>
      ))}
    </div>
  );
}

function renderTaskValue(k: string, v: any) {
  const row = <span className="muted">—</span>;
  if (!v) return row;
  if (v.status === 'error') return <span className="err">{String(v.error)}</span>;
  if (k === 'weather') {
    const top = v.results?.top_k || [];
    return (
      <>
        <div className="kv"><span>prediction</span><b>{v.prediction ?? '—'}</b></div>
        <div className="kv"><span>confidence</span><b>{v.confidence != null ? Math.round(v.confidence * 100) + '%' : '—'}</b></div>
        {top.length > 0 && (
          <div className="chips">{top.map((t: any, i: number) => <span className="chip" key={i}>{t.class} {Math.round((t.confidence || 0) * 100)}%</span>)}</div>
        )}
      </>
    );
  }
  if (k === 'objects') {
    const d = v.results?.detections || [];
    return (
      <>
        <div className="kv"><span>count</span><b>{d.length}</b></div>
        <div className="chips">{d.slice(0, 12).map((o: any, i: number) => <span className="chip" key={i}>{o.class} {Math.round(o.confidence * 100)}%</span>)}</div>
      </>
    );
  }
  if (k === 'lanes') {
    return (
      <>
        <div className="kv"><span>lanes found</span><b>{v.results?.num_lanes ?? 0}</b></div>
        <div className="kv"><span>mode</span><b>{v.results?.mode ?? 'classical'}</b></div>
      </>
    );
  }
  if (k === 'signs') {
    const top = v.results?.top_k || [];
    return (
      <>
        <div className="kv"><span>prediction</span><b>{v.prediction ?? '—'}</b></div>
        <div className="kv"><span>confidence</span><b>{v.confidence != null ? Math.round(v.confidence * 100) + '%' : '—'}</b></div>
        {top.length > 0 && <div className="chips">{top.slice(0, 3).map((t: any, i: number) => <span className="chip" key={i}>{t.class_name}</span>)}</div>}
      </>
    );
  }
  if (k === 'seg') {
    const cl = v.results?.classes || [];
    return (
      <>
        <div className="kv"><span>classes</span><b>{v.results?.num_classes ?? 0}</b></div>
        <div className="chips">{cl.slice(0, 10).map((c: any, i: number) => <span className="chip" key={i}>{c.class_name} {c.percent}%</span>)}</div>
      </>
    );
  }
  return <pre className="raw">{JSON.stringify(v).slice(0, 300)}</pre>;
}

/* sample dataset manifest shape */
type Sample = { dir: string; file: string; url: string };

/* ---------- map real backend task results -> DetectionPacket ----------
   Mirrors the reference repo's ml_bridge.py mapping (from_ai_and_sensors) so
   the bottom "detection status" footer shows REAL results on every page. */
const COCO_TO_LABEL: Record<string, string> = {
  car: 'Vehicle', truck: 'Truck', bus: 'Bus', motorcycle: 'Motorbike',
  motorbike: 'Motorbike', bicycle: 'Cyclist', person: 'Person',
  'traffic light': 'Signal', 'stop sign': 'Stop sign',
};
const GTSRB_SPEED_LIMITS: Record<number, number> = {
  0: 20, 1: 30, 2: 50, 3: 60, 4: 70, 5: 80, 7: 100, 8: 120,
};
const isVehicleCls = (c: string) => /car|truck|bus|motorcycle|motorbike|vehicle/i.test(c);
const isTrafficLightCls = (c: string) => /traffic light|signal/i.test(c);

function laneFromCount(count: number | undefined | null): LaneState {
  if (count == null) return null;
  return count <= 1 ? 'changing-left' : 'steady';
}

function deriveFromTasks(tasks: Record<string, any> | null | undefined): DetectionPacket {
  const t = tasks || {};
  const weather = t.weather?.prediction as string | undefined;
  const weatherConf = t.weather?.confidence;
  const dets: any[] = t.objects?.results?.detections || [];
  const numLanes = t.lanes?.results?.num_lanes;
  const signs = t.signs || {};
  const signId = signs.results?.class_id;
  const vehicleDetected = dets.some((d) => isVehicleCls(d.class));
  const hasTlight = dets.some((d) => isTrafficLightCls(d.class));
  const rain = weather ? /rain/i.test(weather) : false;
  const speedLimit = typeof signId === 'number' && signId in GTSRB_SPEED_LIMITS
    ? GTSRB_SPEED_LIMITS[signId] : null;
  return {
    source: 'model',
    vehicle: vehicleDetected ? { detected: true, distanceM: 0, position: 'front' } : null,
    rain,
    lane: laneFromCount(numLanes),
    trafficLight: hasTlight ? 'green' : null,
    speedLimit,
    roadSurface: rain ? 'wet' : (weather && /fog|cloud|clear/i.test(weather) ? 'dry' : null),
    gear: 'D',
    speedKph: 0,
    objects: dets.slice(0, 8).map((d) => ({
      label: COCO_TO_LABEL[d.class] || String(d.class || 'Object'),
      x: 0, y: 0, width: 0, height: 0,
    })),
    weather: weather ?? null,
    weatherConf: weatherConf ?? null,
  };
}

function deriveFromVideo(stats: any): DetectionPacket {
  const counts = stats?.weather_counts || {};
  const entries = Object.entries(counts);
  const weather = entries.length ? entries.sort((a, b) => Number(b[1]) - Number(a[1]))[0]?.[0] : null;
  const objFrames = Number(stats?.object_frames || 0);
  return {
    source: 'model',
    vehicle: objFrames > 0 ? { detected: true, distanceM: 0, position: 'front' } : null,
    rain: weather ? /rain/i.test(String(weather)) : false,
    lane: null,
    trafficLight: null,
    speedLimit: null,
    roadSurface: weather && /rain/i.test(String(weather)) ? 'wet' : (weather ? 'dry' : null),
    gear: 'D',
    speedKph: 0,
    objects: objFrames > 0 ? [{ label: 'Object', x: 0, y: 0, width: 0, height: 0 }] : [],
    weather: weather ?? null,
    weatherConf: null,
  };
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const pollRef = useRef<number | null>(null);

  const [tab, setTab] = useState<Tab>('live');
  const [cameraState, setCameraState] = useState<'idle' | 'starting' | 'active' | 'error'>('idle');
  const [cameraError, setCameraError] = useState('');
  const [detection, setDetection] = useState<DetectionPacket>(EMPTY_DETECTION);
  const [dismissedAlert, setDismissedAlert] = useState<string | null>(null);
  const [clock, setClock] = useState('10:42');

  // image tab
  const [imgFile, setImgFile] = useState<File | null>(null);
  const [imgPreview, setImgPreview] = useState<string | null>(null);
  const [imgBusy, setImgBusy] = useState(false);
  const [imgProgress, setImgProgress] = useState(0);
  const [imgStatus, setImgStatus] = useState('');
  const [imgResult, setImgResult] = useState<any>(null);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [sampleOpen, setSampleOpen] = useState(false);
  const [sampleRefreshing, setSampleRefreshing] = useState(false);
  const [loadingSample, setLoadingSample] = useState(false);

  // video tab
  const [vidFile, setVidFile] = useState<File | null>(null);
  const [vidPreview, setVidPreview] = useState<string | null>(null);
  const [vidBusy, setVidBusy] = useState(false);
  const [vidProgress, setVidProgress] = useState(0);
  const [vidStatus, setVidStatus] = useState('');
  const [vidResult, setVidResult] = useState<any>(null);

  // comparison tab
  const [comparison, setComparison] = useState<any[]>([]);
  const [modelsStatus, setModelsStatus] = useState<any>(null);
  const [compareNote, setCompareNote] = useState('');
  const [compareBusy, setCompareBusy] = useState(false);

  /* clock + detection event wiring */
  useEffect(() => {
    const updateClock = () => setClock(new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
    updateClock();
    const timer = window.setInterval(updateClock, 30000);

    const handleModelDetection = (event: Event) => {
      const packet = (event as CustomEvent<DetectionPacket>).detail;
      if (!packet || typeof packet !== 'object') return;
      setDetection({ ...EMPTY_DETECTION, ...packet, source: 'model' });
    };
    window.addEventListener('adas-detection', handleModelDetection);

    return () => {
      window.clearInterval(timer);
      window.removeEventListener('adas-detection', handleModelDetection);
    };
  }, []);

  /* active alert key (declared from `detection`, declared above) */
  const activeAlert = useMemo(() => getAlert(detection), [detection]);
  const activeAlertKey = activeAlert?.key ?? null;
  useEffect(() => {
    if (activeAlertKey !== dismissedAlert) setDismissedAlert(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeAlertKey]);

  /* -------- live camera: reference-repo polling of /detect -------- */
  const stopCamera = useCallback(() => {
    if (pollRef.current) { window.clearInterval(pollRef.current); pollRef.current = null; }
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraState('idle');
  }, []);

  const startCamera = useCallback(async () => {
    setCameraState('starting');
    setCameraError('');
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access is not available in this browser.');
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setDetection(EMPTY_DETECTION);
      setCameraState('active');
      if (pollRef.current) window.clearInterval(pollRef.current);
      pollRef.current = window.setInterval(sendFrame, 700);
    } catch (error) {
      setCameraError(error instanceof Error ? error.message : 'Camera permission was not granted.');
      setCameraState('error');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function sendFrame() {
    const vid = videoRef.current;
    if (!vid || vid.readyState < 2) return;
    const cnv = document.createElement('canvas');
    cnv.width = vid.videoWidth || 640;
    cnv.height = vid.videoHeight || 480;
    cnv.getContext('2d')!.drawImage(vid, 0, 0, cnv.width, cnv.height);
    cnv.toBlob(async (blob) => {
      if (!blob) return;
      const fd = new FormData();
      fd.append('file', blob, 'frame.jpg');
      try {
        const resp = await fetch('/detect', { method: 'POST', body: fd });
        if (!resp.ok) return;
        const packet = await resp.json();
        packet.source = packet.source || 'model';
        window.dispatchEvent(new CustomEvent('adas-detection', { detail: packet }));
      } catch { /* backend down — keep last frame */ }
    }, 'image/jpeg', 0.7);
  }

  /* reactive display values */
  const proximity = detection.vehicle?.detected
    ? Math.min(1, Math.max(0.16, (12 - detection.vehicle.distanceM) / 10))
    : 0;
  const edgeStrength = activeAlert?.key === 'vehicle'
    ? Math.min(1, proximity + (detection.gear === 'R' ? 0.12 : 0))
    : activeAlert?.tone === 'danger' ? 0.58
      : activeAlert?.tone === 'caution' ? 0.42
        : activeAlert?.tone === 'go' ? 0.22 : 0;
  const interfaceStyle = { '--edge-strength': edgeStrength } as CSSProperties;
  const tone = activeAlert?.tone ?? 'idle';
  const inputReady = cameraState === 'active';
  // Show real results regardless of which tab is active (camera not required).
  const hasResults = detection.vehicle?.detected || (detection.objects?.length ?? 0) > 0 ||
    detection.lane != null || detection.trafficLight != null || detection.speedLimit != null ||
    detection.rain || detection.roadSurface != null;
  const detectionActive = inputReady || tab !== 'live';
  const objectSummary = detection.vehicle?.detected
    ? `Vehicle / ${detection.vehicle.distanceM.toFixed(1)} m`
    : detection.objects?.length
      ? `${detection.objects.length} object${detection.objects.length === 1 ? '' : 's'}`
      : hasResults ? 'No object hazard' : detectionActive ? 'No object hazard' : 'Standby';
  const laneSummary = detection.lane === 'changing-left' || detection.lane === 'changing-right'
    ? 'Change detected' : detection.lane === 'steady' ? 'Lane tracking' : 'Standby';

  /* simple XHR upload with progress */
  function uploadWithProgress(url: string, fd: FormData, onProgress: (p: number) => void) {
    return new Promise<any>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', url);
      xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100)); };
      xhr.onload = () => {
        try {
          const body = JSON.parse(xhr.responseText);
          resolve(body);
        } catch { reject(new Error('Invalid server response')); }
      };
      xhr.onerror = () => reject(new Error('Network error'));
      xhr.send(fd);
    });
  }

  /* image tab */
  function onPickImage(file: File | null) {
    if (!file) return;
    setImgFile(file);
    setImgPreview(URL.createObjectURL(file));
    setImgResult(null);
    setImgStatus('');
  }
  // Load a dataset sample, turning it into a real File so "Analyse image" works.
  // Fetch a fresh random set of demo images (2 per folder) for the gallery.
  async function refreshSamples() {
    setSampleRefreshing(true);
    try {
      const r = await fetch('/dataset/samples?random=true');
      const j = await r.json();
      setSamples(j.samples || []);
    } catch (e) {
      setImgStatus('could not refresh samples: ' + (e instanceof Error ? e.message : e));
    } finally {
      setSampleRefreshing(false);
    }
  }
  async function loadSample(s: Sample) {
    setLoadingSample(true);
    setImgStatus('loading sample…');
    try {
      const resp = await fetch(s.url);
      const blob = await resp.blob();
      const file = new File([blob], s.file || 'sample.jpg', { type: blob.type || 'image/jpeg' });
      setImgFile(file);
      if (imgPreview) URL.revokeObjectURL(imgPreview);
      setImgPreview(URL.createObjectURL(file));
      setImgResult(null);
      setImgStatus(`sample loaded: ${s.dir}`);
    } catch (e) {
      setImgStatus('error loading sample: ' + (e instanceof Error ? e.message : e));
    } finally {
      setLoadingSample(false);
    }
  }
  async function analyzeImage(file: File) {
    setImgBusy(true); setImgProgress(0); setImgStatus('uploading… (first run loads models)');
    const fd = new FormData(); fd.append('file', file);
    try {
      const body = await uploadWithProgress('/analyze', fd, setImgProgress);
      setImgResult(body); setImgStatus('done');
      // feed the real per-task results into the footer detection status
      if (body?.tasks) setDetection(deriveFromTasks(body.tasks));
    } catch (e) { setImgStatus('error: ' + (e instanceof Error ? e.message : e)); }
    setImgBusy(false);
  }

  /* video tab */
  function onPickVideo(file: File | null) {
    if (!file) return;
    setVidFile(file);
    setVidPreview(URL.createObjectURL(file));
    setVidResult(null);
    setVidStatus('');
  }
  async function analyzeVideo(file: File, maxFrames: number) {
    setVidBusy(true); setVidProgress(0); setVidStatus('uploading… (can take a while on CPU)');
    const fd = new FormData(); fd.append('file', file);
    if (maxFrames > 0) fd.append('max_frames', String(maxFrames));
    try {
      const body = await uploadWithProgress('/analyze_video', fd, setVidProgress);
      setVidResult(body); setVidStatus('done');
      if (body) setDetection(deriveFromVideo(body));
    } catch (e) { setVidStatus('error: ' + (e instanceof Error ? e.message : e)); }
    setVidBusy(false);
  }

  /* comparison tab: load stored results + samples */
  useEffect(() => {
    (async () => {
      try {
        const r = await fetch('/comparison'); const j = await r.json();
        setComparison(j.rows || j || []);
        setCompareNote('Stored results from the last run.');
      } catch (e) { setCompareNote('comparison unavailable: ' + (e instanceof Error ? e.message : e)); }
      try {
        const r = await fetch('/dataset/samples'); const j = await r.json();
        setSamples(j.samples || []);
      } catch { /* no dataset tab */ }
      try {
        const r = await fetch('/models'); const j = await r.json();
        setModelsStatus(j || null);
      } catch { /* models status unavailable */ }
    })();
    // keep model status fresh while the compare tab is open
    const t = setInterval(async () => {
      try {
        const r = await fetch('/models'); const j = await r.json();
        setModelsStatus(j || null);
      } catch { /* ignore */ }
    }, 4000);
    return () => clearInterval(t);
  }, []);

  async function runComparison() {
    setCompareBusy(true);
    setCompareNote('Running comparison — this trains / evaluates the four backbones. Please wait.');
    try {
      const r = await fetch('/comparison_run', { method: 'POST' });
      const j = await r.json();
      if (j.rows && j.rows.length) setComparison(j.rows);
      setCompareNote(j.note || 'Run complete.');
    } catch (e) { setCompareNote('error: ' + (e instanceof Error ? e.message : e)); }
    setCompareBusy(false);
  }

  const bestAcc = comparison.length ? Math.max(...comparison.map((r) => r.accuracy), 0) : 0;

  return (
    <main className={`vehicle-cabin tone-${tone}`} style={interfaceStyle}>
      <div className="cabin-trim" aria-hidden="true" />
      <section className="display-unit" aria-label="ADAS camera-based driver assistance display">
        <header className="system-bar">
          <div className="brand">
            <div className="product-name"><span className="product-mark">A</span><span>ADAS</span></div>
            <div className="mode-description">Camera-based driver assistance</div>
          </div>
          <div className="center-cluster">
            <time className="clock">{clock}</time>
            <nav className="tabs" aria-label="ADAS views">
              <button type="button" className={`tab ${tab === 'live' ? 'active' : ''}`} onClick={() => setTab('live')}>Live</button>
              <button type="button" className={`tab ${tab === 'image' ? 'active' : ''}`} onClick={() => setTab('image')}>Image</button>
              <button type="button" className={`tab ${tab === 'video' ? 'active' : ''}`} onClick={() => setTab('video')}>Video</button>
              <button type="button" className={`tab ${tab === 'compare' ? 'active' : ''}`} onClick={() => setTab('compare')}>Compare</button>
            </nav>
          </div>
          <div className="system-actions">
            <button type="button" className={`camera-control ${cameraState === 'active' ? 'active' : ''}`}
              onClick={cameraState === 'active' ? stopCamera : startCamera} disabled={cameraState === 'starting'}>
              <i />{cameraState === 'active' ? 'Camera live' : cameraState === 'starting' ? 'Connecting' : 'Camera on'}
            </button>
          </div>
        </header>

        {/* ---------- LIVE : the ADAS cabin camera view ---------- */}
        {tab === 'live' && (
          <section className={`camera-view ${inputReady ? 'input-ready' : ''}`} aria-label="Laptop camera perception feed">
            <video ref={videoRef} autoPlay muted playsInline aria-label="Laptop camera feed" />
            {!inputReady && (
              <div className="camera-empty">
                <div className="camera-symbol" aria-hidden="true"><i /></div>
                <h1>Connect the perception camera</h1>
                <p>The laptop camera is used as the live video input. Detection results appear directly on this screen.</p>
                <div className="empty-actions">
                  <button type="button" className="primary-action" onClick={startCamera} disabled={cameraState === 'starting'}>
                    {cameraState === 'starting' ? 'Connecting camera...' : 'Connect laptop camera'}
                  </button>
                </div>
                {cameraError && <p className="camera-error">{cameraError}</p>}
              </div>
            )}
            {inputReady && (
              <div className="input-label">
                <i />
                <span><b>Laptop camera</b><small>Waiting for model detections</small></span>
              </div>
            )}
            {inputReady && detection.roadSurface && (
              <div className={`road-segmentation surface-${detection.roadSurface}`} aria-label={`${detection.roadSurface} drivable road area`}>
                <span>Drivable area</span>
              </div>
            )}
            {inputReady && detection.lane && (
              <div className={`lane-overlay lane-${detection.lane}`} aria-label={`Lane detection: ${detection.lane}`}>
                <i className="detected-lane lane-left" /><i className="detected-lane lane-right" />
              </div>
            )}
            {inputReady && detection.objects?.map((object, index) => (
              <div className={`detection-frame ${detection.vehicle?.detected ? 'hazard-object' : ''}`}
                key={`${object.label}-${index}`}
                style={{ left: `${object.x}%`, top: `${object.y}%`, width: `${object.width}%`, height: `${object.height}%` }}>
                <span>{object.label}</span>
              </div>
            ))}
            {inputReady && detection.speedLimit && (
              <div className="speed-sign" aria-label={`Detected speed limit ${detection.speedLimit}`}><span>{detection.speedLimit}</span><small>Speed limit</small></div>
            )}
            {inputReady && detection.trafficLight && (
              <div className={`traffic-detection signal-${detection.trafficLight}`} aria-label={`${detection.trafficLight} traffic light detected`}>
                <div className="signal-housing"><i /><i /><i /></div>
                <span>{detection.trafficLight} light</span>
              </div>
            )}
            {activeAlert && dismissedAlert !== activeAlert.key && (
              <aside className={`warning-panel warning-${activeAlert.tone}`} role="alert" aria-live="assertive">
                <span className="warning-symbol">!</span>
                <div className="warning-copy"><small>{activeAlert.category}</small><strong>{activeAlert.title}</strong><p>{activeAlert.message}</p></div>
                <button type="button" className="dismiss-warning" onClick={() => setDismissedAlert(activeAlert.key)} aria-label="Dismiss alert">Close</button>
              </aside>
            )}
            <div className="drive-summary">
              <span className="gear-indicator">{detection.gear ?? 'D'}</span>
              <strong>{String(detection.speedKph ?? 0).padStart(2, '0')}</strong>
              <small>km/h</small>
            </div>
            <div className="model-state">
              <span className={inputReady ? 'ready' : ''} />
              {inputReady ? 'Perception input active' : 'Perception input standby'}
            </div>
          </section>
        )}

        {/* ---------- IMAGE upload ---------- */}
        {tab === 'image' && (
          <section className="tool-view" aria-label="Image analysis">
            <div className="tool-col">
              <h2>Upload a road image</h2>
              <p className="muted">Runs the full pipeline: weather, YOLO objects, lane, traffic-sign and segmentation.</p>
              <input type="file" accept="image/*" id="imgupload" className="file-input"
                onChange={(e) => onPickImage(e.target.files?.[0] ?? null)} />
              <div className="row">
                <label htmlFor="imgupload" className="file-label">Choose a .jpg / .png</label>
                <button type="button" className="ghost-action" disabled={loadingSample}
                  onClick={() => setSampleOpen((v) => !v)}>
                  {loadingSample ? 'Loading sample…' : 'Load dataset sample'}
                </button>
              </div>
              {sampleOpen && (
                <div className="sample-picker" aria-label="Dataset sample images">
                  <div className="sample-picker-title">
                    <span>Pick a sample (Day / Night / Fog / Rain)</span>
                    <button type="button" className="ghost shuffle" disabled={sampleRefreshing}
                      onClick={refreshSamples}>
                      {sampleRefreshing ? 'Loading…' : '↻ Shuffle samples'}
                    </button>
                  </div>
                  <div className="sample-row">
                    {samples.map((s) => (
                      <button type="button" key={s.url} className={`sample-chip ${imgPreview && imgPreview.includes(encodeURIComponent(s.file)) ? 'sel' : ''}`}
                        onClick={() => loadSample(s)}>
                        <img src={s.url} alt={`${s.dir} sample`} loading="lazy" />
                        <span>{s.dir}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
              <div className="row">
                <button type="button" className="primary-action run" disabled={imgBusy || (!imgFile)}
                  onClick={() => imgFile && analyzeImage(imgFile)}>
                  {imgBusy ? 'Analysing…' : 'Analyse image'}
                </button>
              </div>
              {imgBusy && <div className="progress-bar"><i style={{ width: `${imgProgress}%` }} /></div>}
              {imgStatus && <div className={`status ${imgStatus.startsWith('error') ? 'err' : ''}`}>{imgStatus}</div>}
            </div>
            <div className="tool-col">
              <h2>Preview</h2>
              {imgPreview ? (
                <div className="preview-frame">
                  <img className="result-img" src={imgPreview} alt="uploaded preview" />
                  {imgResult?.image_base64 && <span className="annotated-tag">annotated</span>}
                </div>
              ) : (
                <div className="muted">No image selected yet.</div>
              )}
              {imgResult?.image_base64 && (
                <>
                  <div className="grid-metrics">
                    <StatChip label="weather" value={imgResult.info?.weather || '—'} />
                    <StatChip label="objects" value={String(imgResult.info?.objects ?? '—')} />
                    <StatChip label="lanes" value={String(imgResult.info?.lanes ?? '—')} />
                    <StatChip label="inference" value={imgResult.inference_time_ms != null ? imgResult.inference_time_ms + ' ms' : '—'} />
                  </div>
                  <TaskGrid tasks={imgResult.tasks} />
                </>
              )}
            </div>
          </section>
        )}

        {/* ---------- VIDEO upload ---------- */}
        {tab === 'video' && (
          <section className="tool-view" aria-label="Video analysis">
            <div className="tool-col">
              <h2>Upload a driving clip</h2>
              <p className="muted">Processes each frame through the pipeline and writes an annotated MP4. Limit frames for a quick test.</p>
              <div className="row">
                <input type="file" accept="video/*" id="vidupload" className="file-input"
                  onChange={(e) => onPickVideo(e.target.files?.[0] ?? null)} />
                <label htmlFor="vidupload" className="file-label">Choose a .mp4</label>
                <label className="muted">Max frames <input type="number" id="vidmax" defaultValue={120} min={1} style={{ width: 70 }} /></label>
              </div>
              <div className="row">
                <button type="button" className="primary-action run" disabled={vidBusy || (!vidFile)}
                  onClick={() => {
                    const max = Number((document.getElementById('vidmax') as HTMLInputElement).value) || 120;
                    if (vidFile) analyzeVideo(vidFile, max);
                  }}>
                  {vidBusy ? 'Analysing…' : 'Analyse video'}
                </button>
              </div>
              {vidBusy && <div className="progress-bar"><i style={{ width: `${vidProgress}%` }} /></div>}
              {vidStatus && <div className={`status ${vidStatus.startsWith('error') ? 'err' : ''}`}>{vidStatus}</div>}
            </div>
            <div className="tool-col">
              <h2>Preview</h2>
              {vidPreview ? (
                <div className="preview-frame">
                  <video className="result-vid" src={vidPreview} controls muted playsInline />
                  {vidResult?.sample_base64 ? (
                    <div className="annotated-thumb"><img className="result-img" src={`data:image/jpeg;base64,${vidResult.sample_base64}`} alt="sample annotated frame" /><span className="annotated-tag">annotated frame</span></div>
                  ) : null}
                </div>
              ) : (
                <div className="muted">No video selected yet.</div>
              )}
              {vidResult && (
                <>
                  <div className="grid-metrics">
                    <StatChip label="frames" value={String(vidResult.frames)} />
                    <StatChip label="avg ms" value={String(vidResult.avg_ms)} />
                    <StatChip label="~FPS" value={String(Math.round(1000 / Math.max(vidResult.avg_ms, 1)))} />
                    <StatChip label="obj frames" value={String(vidResult.object_frames)} />
                  </div>
                  <div className="chips">
                    {Object.entries(vidResult.weather_counts || {}).map(([k, v]) => <span className="chip" key={k}>{k}: {String(v)}</span>)}
                  </div>
                  {vidResult.video_url && <a className="url-link" href={vidResult.video_url} target="_blank" rel="noreferrer">Open annotated video</a>}
                </>
              )}
            </div>
          </section>
        )}

        {/* ---------- COMPARISON ---------- */}
        {tab === 'compare' && (
          <section className="tool-view compare-view" aria-label="Model comparison">
            <div className="tool-col wide">
              <div className="compare-head">
                <h2>Four-backbone comparison</h2>
                <button type="button" className="primary-action run" disabled={compareBusy} onClick={runComparison}>
                  {compareBusy ? 'Running…' : 'Start comparison'}
                </button>
              </div>
              <p className="muted">Same split, same preprocessing, same budget — only the backbone differs.</p>

              {modelsStatus && (
                <div className="model-status" aria-label="Running models">
                  <h3>Running models</h3>
                  <div className="status-grid">
                    {Object.entries(modelsStatus.tasks || {}).map(([task, state]) => (
                      <div key={task} className={`status-chip ${state || 'lazy'}`}>
                        <span className="dot" />
                        <span className="task">{task}</span>
                        <span className="state">{state}</span>
                      </div>
                    ))}
                  </div>
                  {modelsStatus.checkpoints && (
                    <p className="muted ckpt-note">
                      checkpoints: {' '}
                      {Object.entries(modelsStatus.checkpoints).map(([m, has]) => (
                        <span key={m} className={has ? 'ready' : 'missing'}>{m}{has ? ' ✓' : ' ✗'}</span>
                      )).reduce<React.ReactNode[]>((acc, el, i) =>
                        i === 0 ? [el] : [...acc, <span key={'sep'+i}> · </span>, el], [])}
                    </p>
                  )}
                </div>
              )}

              {comparison.length === 0 && <div className="muted">{compareNote || 'No rows yet — run the comparison.'}</div>}
              {comparison.length > 0 && (
                <table className="compare-table">
                  <thead><tr>
                    <th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>Latency</th><th>FPS</th><th>Size</th>
                  </tr></thead>
                  <tbody>
                    {comparison.map((r) => (
                      <tr key={r.model} className={r.accuracy === bestAcc ? 'hi' : ''}>
                        <td><b>{r.model}</b></td>
                        <td className="num">{Math.round(r.accuracy * 100)}%</td>
                        <td className="num">{Math.round(r.precision * 100)}%</td>
                        <td className="num">{Math.round(r.recall * 100)}%</td>
                        <td className="num">{Math.round(r.f1 * 100)}%</td>
                        <td className="num">{r.inference_time_ms} ms</td>
                        <td className="num">{r.fps}</td>
                        <td className="num">{r.model_size_mb} MB</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              <div className="muted">{compareNote}</div>
            </div>
          </section>
        )}

        <footer className="vehicle-status" aria-label="Current detection status">
          <div><small>Object detection</small><strong>{objectSummary}</strong></div>
          <div><small>Lane detection</small><strong>{laneSummary}</strong></div>
          <div><small>Traffic light</small><strong className={detection.trafficLight ? `text-${detection.trafficLight}` : ''}>{detection.trafficLight ? `${detection.trafficLight[0].toUpperCase()}${detection.trafficLight.slice(1)}` : 'None'}</strong></div>
          <div><small>Traffic sign</small><strong>{detection.speedLimit ? `${detection.speedLimit} km/h` : 'None'}</strong></div>
          <div><small>Road condition</small><strong>{detection.rain ? 'Rain / slow down' : detection.roadSurface === 'wet' ? 'Wet' : detection.roadSurface === 'dry' ? 'Dry' : 'Standby'}</strong></div>
          <div><small>Weather</small><strong>{detection.weather ? `${detection.weather[0].toUpperCase()}${detection.weather.slice(1)}${detection.weatherConf != null ? ` · ${Math.round(detection.weatherConf * 100)}%` : ''}` : 'None'}</strong></div>
        </footer>
      </section>
      <p className="prototype-note">Frontend prototype. Camera detections are supplied through the ADAS model event bridge.</p>
    </main>
  );
}
