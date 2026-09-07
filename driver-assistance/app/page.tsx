'use client';

import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';

type TrafficLight = 'red' | 'yellow' | 'green' | null;
type LaneState = 'steady' | 'changing-left' | 'changing-right' | null;
type Gear = 'P' | 'R' | 'N' | 'D';
type AlertTone = 'danger' | 'caution' | 'neutral' | 'go';

type DetectionBox = {
  label: string;
  x: number;
  y: number;
  width: number;
  height: number;
};

export type DetectionPacket = {
  source?: 'model' | 'demo';
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

const DEMO_SEQUENCE: DetectionPacket[] = [
  { source: 'demo', gear: 'D', speedKph: 38, lane: 'steady', roadSurface: 'dry', speedLimit: 60, objects: [{ label: 'Vehicle', x: 44, y: 40, width: 13, height: 20 }] },
  { source: 'demo', gear: 'R', speedKph: 4, lane: 'steady', roadSurface: 'dry', vehicle: { detected: true, distanceM: 9.2, position: 'rear-right' }, objects: [{ label: 'Vehicle', x: 67, y: 38, width: 15, height: 27 }] },
  { source: 'demo', gear: 'R', speedKph: 3, lane: 'steady', roadSurface: 'dry', vehicle: { detected: true, distanceM: 5.4, position: 'rear-right' }, objects: [{ label: 'Vehicle', x: 63, y: 34, width: 21, height: 35 }] },
  { source: 'demo', gear: 'R', speedKph: 2, lane: 'steady', roadSurface: 'dry', vehicle: { detected: true, distanceM: 2.6, position: 'rear-right' }, objects: [{ label: 'Vehicle', x: 57, y: 26, width: 31, height: 48 }] },
  { source: 'demo', gear: 'D', speedKph: 28, lane: 'steady', roadSurface: 'wet', rain: true, objects: [{ label: 'Vehicle', x: 45, y: 41, width: 12, height: 19 }] },
  { source: 'demo', gear: 'D', speedKph: 41, lane: 'changing-right', roadSurface: 'dry', objects: [{ label: 'Vehicle', x: 43, y: 41, width: 13, height: 20 }] },
  { source: 'demo', gear: 'D', speedKph: 24, lane: 'steady', roadSurface: 'dry', trafficLight: 'red', speedLimit: 60, objects: [] },
  { source: 'demo', gear: 'D', speedKph: 19, lane: 'steady', roadSurface: 'dry', trafficLight: 'yellow', speedLimit: 60, objects: [] },
  { source: 'demo', gear: 'D', speedKph: 27, lane: 'steady', roadSurface: 'dry', trafficLight: 'green', speedLimit: 60, objects: [] },
];

function getAlert(detection: DetectionPacket): ActiveAlert | null {
  if (detection.vehicle?.detected) {
    const position = detection.vehicle.position === 'rear-right' ? 'Rear right' : detection.vehicle.position === 'rear-left' ? 'Rear left' : 'Ahead';
    return {
      key: 'vehicle',
      tone: 'danger',
      category: 'Collision warning',
      title: 'Vehicle approaching',
      message: `${position} / ${detection.vehicle.distanceM.toFixed(1)} m`,
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

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [cameraState, setCameraState] = useState<'idle' | 'starting' | 'active' | 'error'>('idle');
  const [cameraError, setCameraError] = useState('');
  const [demoActive, setDemoActive] = useState(false);
  const [detection, setDetection] = useState<DetectionPacket>(EMPTY_DETECTION);
  const [dismissedAlert, setDismissedAlert] = useState<string | null>(null);
  const [clock, setClock] = useState('10:42');

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraState('idle');
  };

  const startCamera = async () => {
    setCameraState('starting');
    setCameraError('');
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access is not available in this browser.');
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setDemoActive(false);
      setDetection(EMPTY_DETECTION);
      setCameraState('active');
    } catch (error) {
      setCameraError(error instanceof Error ? error.message : 'Camera permission was not granted.');
      setCameraState('error');
    }
  };

  useEffect(() => {
    const updateClock = () => setClock(new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
    updateClock();
    const clockTimer = window.setInterval(updateClock, 30000);

    // ML integration point:
    // window.dispatchEvent(new CustomEvent('adas-detection', { detail: detectionPacket }));
    const handleModelDetection = (event: Event) => {
      const packet = (event as CustomEvent<DetectionPacket>).detail;
      if (!packet || typeof packet !== 'object') return;
      setDemoActive(false);
      setDetection({ ...EMPTY_DETECTION, ...packet, source: 'model' });
    };
    window.addEventListener('adas-detection', handleModelDetection);

    return () => {
      window.clearInterval(clockTimer);
      window.removeEventListener('adas-detection', handleModelDetection);
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);

  useEffect(() => {
    if (!demoActive) return;
    let index = 0;
    setDetection(DEMO_SEQUENCE[index]);
    const timer = window.setInterval(() => {
      index = (index + 1) % DEMO_SEQUENCE.length;
      setDetection(DEMO_SEQUENCE[index]);
    }, 3200);
    return () => window.clearInterval(timer);
  }, [demoActive]);

  const activeAlert = useMemo(() => getAlert(detection), [detection]);

  useEffect(() => {
    if (activeAlert?.key !== dismissedAlert) setDismissedAlert(null);
    // Only the alert category resets dismissal; changing vehicle distance does not.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeAlert?.key]);

  const proximity = detection.vehicle?.detected
    ? Math.min(1, Math.max(0.16, (12 - detection.vehicle.distanceM) / 10))
    : 0;
  const edgeStrength = activeAlert?.key === 'vehicle'
    ? Math.min(1, proximity + (detection.gear === 'R' ? 0.12 : 0))
    : activeAlert?.tone === 'danger'
      ? 0.58
      : activeAlert?.tone === 'caution'
        ? 0.42
        : activeAlert?.tone === 'go'
          ? 0.22
          : 0;

  const interfaceStyle = { '--edge-strength': edgeStrength } as CSSProperties;
  const tone = activeAlert?.tone ?? 'idle';
  const inputReady = cameraState === 'active' || demoActive;
  const objectSummary = detection.vehicle?.detected
    ? `Vehicle / ${detection.vehicle.distanceM.toFixed(1)} m`
    : detection.objects?.length
      ? `${detection.objects.length} object${detection.objects.length === 1 ? '' : 's'}`
      : inputReady
        ? 'No object hazard'
        : 'Standby';
  const laneSummary = detection.lane === 'changing-left' || detection.lane === 'changing-right'
    ? 'Change detected'
    : detection.lane === 'steady'
      ? 'Lane tracking'
      : 'Standby';

  return (
    <main className={`vehicle-cabin tone-${tone}`} style={interfaceStyle}>
      <div className="cabin-trim" aria-hidden="true" />

      <section className="display-unit" aria-label="ARES Drive camera-based driver assistance display">
        <header className="system-bar">
          <div className="product-name"><span className="product-mark">A</span><span>ARES Drive</span></div>
          <div className="mode-description">Camera-based driver assistance</div>
          <div className="system-actions">
            <button type="button" className={`demo-control ${demoActive ? 'active' : ''}`} onClick={() => setDemoActive((value) => !value)}>
              {demoActive ? 'Stop demo' : 'Automatic demo'}
            </button>
            <button type="button" className={`camera-control ${cameraState === 'active' ? 'active' : ''}`} onClick={cameraState === 'active' ? stopCamera : startCamera} disabled={cameraState === 'starting'}>
              <i />{cameraState === 'active' ? 'Camera live' : cameraState === 'starting' ? 'Connecting' : 'Camera'}
            </button>
            <time>{clock}</time>
          </div>
        </header>

        <section className={`camera-view ${inputReady ? 'input-ready' : ''}`} aria-label="Laptop camera perception feed">
          <video ref={videoRef} autoPlay muted playsInline aria-label="Laptop camera feed" />

          {!inputReady && (
            <div className="camera-empty">
              <div className="camera-symbol" aria-hidden="true"><i /></div>
              <h1>Connect the perception camera</h1>
              <p>The laptop camera is used as the live video input. Detection results will appear directly on this screen.</p>
              <div className="empty-actions">
                <button type="button" className="primary-action" onClick={startCamera} disabled={cameraState === 'starting'}>
                  {cameraState === 'starting' ? 'Connecting camera...' : 'Connect laptop camera'}
                </button>
                <button type="button" className="secondary-action" onClick={() => setDemoActive(true)}>Preview automatic alerts</button>
              </div>
              {cameraError && <p className="camera-error">{cameraError}</p>}
            </div>
          )}

          {demoActive && cameraState !== 'active' && (
            <div className="demo-camera" aria-hidden="true">
              <div className="demo-horizon" />
              <div className="demo-road"><i className="road-line left" /><i className="road-line right" /><i className="road-centre" /></div>
            </div>
          )}

          {inputReady && (
            <div className="input-label">
              <i />
              <span><b>{demoActive ? 'Simulated camera input' : 'Laptop camera'}</b><small>{demoActive ? 'Automatic detection sequence' : 'Waiting for model detections'}</small></span>
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
            <div
              className={`detection-frame ${detection.vehicle?.detected ? 'hazard-object' : ''}`}
              key={`${object.label}-${index}`}
              style={{ left: `${object.x}%`, top: `${object.y}%`, width: `${object.width}%`, height: `${object.height}%` }}
            >
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

        <footer className="vehicle-status" aria-label="Current detection status">
          <div><small>Object detection</small><strong>{objectSummary}</strong></div>
          <div><small>Lane detection</small><strong>{laneSummary}</strong></div>
          <div><small>Traffic light</small><strong className={detection.trafficLight ? `text-${detection.trafficLight}` : ''}>{detection.trafficLight ? `${detection.trafficLight[0].toUpperCase()}${detection.trafficLight.slice(1)}` : 'None'}</strong></div>
          <div><small>Traffic sign</small><strong>{detection.speedLimit ? `${detection.speedLimit} km/h` : 'None'}</strong></div>
          <div><small>Road condition</small><strong>{detection.rain ? 'Rain / slow down' : detection.roadSurface === 'wet' ? 'Wet' : detection.roadSurface === 'dry' ? 'Dry' : 'Standby'}</strong></div>
        </footer>
      </section>

      <p className="prototype-note">Frontend prototype. Camera detections are supplied through the ADAS model event bridge.</p>
    </main>
  );
}
