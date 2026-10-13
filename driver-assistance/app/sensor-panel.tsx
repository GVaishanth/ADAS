'use client';

/* ------------------------------------------------------------------------- *
 *  Live ESP32 sensor panel
 * ------------------------------------------------------------------------- *
 *  Polls the existing backend endpoint `GET /sensor` (FastAPI, :7000) and
 *  renders the physical readings in the dashboard's own visual language.
 *
 *  Design notes
 *  ------------
 *  * Nothing is written to the backend / bridge: this file only reads the
 *    already-working REST endpoint, using the same config convention as the
 *    rest of the app (`window.__ADAS_SENSOR_URL__`, defaulting to
 *    http://127.0.0.1:7000/sensor — see app/layout.tsx).
 *  * Every value is validated before it is rendered. Missing / null / malformed
 *    fields render as `N/A`; a null is never coerced to 0 and no value is ever
 *    invented.
 *  * Polling is 1 s, never overlapping, and is aborted + cleared on unmount.
 *  * The reading is only labelled "Live" while fresh data keeps arriving;
 *    otherwise the panel reports Stale / Disconnected and still shows the
 *    timestamp of the last successful update.
 * ------------------------------------------------------------------------- */

import { useEffect, useState } from 'react';

/* ----------------------------- configuration ----------------------------- */

const DEFAULT_SENSOR_URL = 'http://127.0.0.1:7000/sensor';

/** One request per second (matches the ESP32 push rate). */
const POLL_INTERVAL_MS = 1000;
/** A hung request is aborted so the next tick can take over. */
const REQUEST_TIMEOUT_MS = 2500;
/** A reading older than this is no longer "live". */
const FRESHNESS_MS = 3500;
/** Consecutive failures after which the link is reported as disconnected. */
const FAILURE_LIMIT = 3;

/** `connecting` only exists until the first request settles. */
export type SensorStatus = 'connecting' | 'live' | 'stale' | 'disconnected';

export type SensorReading = {
  distanceCm: number | null;
  rain: boolean | null;
  temperatureC: number | null;
  humidityPercent: number | null;
  ambientLight: number | null;
  speedKph: number | null;
  gear: string | null;
  /** When the reading was produced (backend `_ts` if usable, else local receive time). */
  observedAt: number;
  /** False when the payload carries no real measurement (backend cache untouched). */
  hasMeasurement: boolean;
};

type SensorState = {
  reading: SensorReading | null;
  status: SensorStatus;
  /** Local time of the last successful fetch that carried a measurement. */
  updatedAt: number | null;
};

type SensorTone = 'caution' | null;

const EMPTY_STATE: SensorState = { reading: null, status: 'connecting', updatedAt: null };

const STATUS_LABEL: Record<SensorStatus, string> = {
  connecting: 'Connecting',
  live: 'Live',
  stale: 'Stale',
  disconnected: 'Disconnected',
};

const STATUS_DETAIL: Record<SensorStatus, string> = {
  connecting: 'contacting backend',
  live: 'receiving sensor data',
  stale: 'updates interrupted',
  disconnected: 'no sensor data',
};

/* ------------------------------- validation ------------------------------ */

function sensorEndpoint(): string {
  if (typeof window !== 'undefined') {
    const configured = (window as { __ADAS_SENSOR_URL__?: unknown }).__ADAS_SENSOR_URL__;
    if (typeof configured === 'string' && configured.trim()) return configured.trim();
  }
  return DEFAULT_SENSOR_URL;
}

function asNumber(value: unknown): number | null {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function asFlag(value: unknown): boolean | null {
  if (typeof value === 'boolean') return value;
  if (value === 0 || value === 1) return value === 1; // numeric 0/1 flag, still the real value
  return null;
}

function asGear(value: unknown): string | null {
  if (typeof value !== 'string') return null;
  const gear = value.trim();
  return gear.length > 0 && gear.length <= 8 ? gear : null;
}

/**
 * The backend cache is created as
 * `{distance_cm:null, rain:null, temperature_c:null, humidity_percent:null,
 *   ambient_light:null, speed_kph:0, gear:'D'}` and its POST handler updates
 * every field (including nulls) on the first ESP32 packet. So that exact,
 * untouched shape means "no packet has arrived yet" — showing it as
 * `0 km/h / gear D` would be a fabricated measurement.
 */
function isUntouchedCache(raw: Record<string, unknown>): boolean {
  return (
    raw.distance_cm == null &&
    raw.rain == null &&
    raw.temperature_c == null &&
    raw.humidity_percent == null &&
    raw.ambient_light == null &&
    (raw.speed_kph == null || asNumber(raw.speed_kph) === 0) &&
    (raw.gear == null || asGear(raw.gear) === 'D')
  );
}

/**
 * Validates one `GET /sensor` payload. Returns null when the response is not a
 * usable object, so a broken response is treated as a failed request instead of
 * being rendered.
 */
export function parseSensorPayload(payload: unknown, receivedAt = Date.now()): SensorReading | null {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) return null;
  const raw = payload as Record<string, unknown>;

  const serverTs = asNumber(raw._ts);
  const observedAt = serverTs != null && serverTs > 0
    ? (serverTs > 1e11 ? serverTs : serverTs * 1000) // backend sends epoch seconds
    : receivedAt;

  const hasMeasurement = !isUntouchedCache(raw);

  return {
    // Unavailable fields stay null; they are never defaulted here.
    distanceCm: hasMeasurement ? asNumber(raw.distance_cm) : null,
    rain: hasMeasurement ? asFlag(raw.rain) : null,
    temperatureC: hasMeasurement ? asNumber(raw.temperature_c) : null,
    humidityPercent: hasMeasurement ? asNumber(raw.humidity_percent) : null,
    ambientLight: hasMeasurement ? asNumber(raw.ambient_light) : null,
    speedKph: hasMeasurement ? asNumber(raw.speed_kph) : null,
    gear: hasMeasurement ? asGear(raw.gear) : null,
    observedAt,
    hasMeasurement,
  };
}

/* -------------------------------- polling -------------------------------- */

/**
 * Polls `/sensor` every second. Requests never overlap, a hung request is
 * aborted, the timers and the in-flight request are cleaned up on unmount, and
 * network errors only change the reported status — they never throw into the
 * render tree and never replace the last good reading.
 */
export function useSensorReadings(): SensorState & { ageMs: number | null } {
  const [state, setState] = useState<SensorState>(EMPTY_STATE);
  const [now, setNow] = useState<number | null>(null);

  useEffect(() => {
    let disposed = false;
    let inFlight = false;
    let failures = 0;
    let controller: AbortController | null = null;

    const poll = async () => {
      if (disposed || inFlight) return; // 1 s ticks never stack up requests
      inFlight = true;
      controller = new AbortController();
      const timer = window.setTimeout(() => controller?.abort(), REQUEST_TIMEOUT_MS);
      const receivedAt = Date.now();

      try {
        const response = await fetch(sensorEndpoint(), {
          signal: controller.signal,
          cache: 'no-store',
          headers: { Accept: 'application/json' },
        });
        if (!response.ok) throw new Error(`sensor endpoint returned HTTP ${response.status}`);
        const reading = parseSensorPayload(await response.json(), receivedAt);
        if (!reading) throw new Error('unexpected sensor payload');
        if (disposed) return;

        failures = 0;
        setState((previous) => {
          if (!reading.hasMeasurement) {
            // Backend reachable but it holds no measurement (nothing sent yet,
            // or the cache was reset): no valid reading, so report that — while
            // keeping whatever was received earlier on screen.
            return { reading: previous.reading, status: 'disconnected', updatedAt: previous.updatedAt };
          }
          return { reading, status: 'live', updatedAt: receivedAt };
        });
      } catch {
        if (disposed) return;
        failures += 1;
        setState((previous) => ({
          reading: previous.reading,
          status: previous.reading
            ? (failures >= FAILURE_LIMIT ? 'disconnected' : 'stale')
            : 'disconnected',
          updatedAt: previous.updatedAt,
        }));
      } finally {
        window.clearTimeout(timer);
        inFlight = false;
      }
    };

    poll();
    const pollTimer = window.setInterval(poll, POLL_INTERVAL_MS);
    // Independent tick so a reading that stops refreshing eventually reads as
    // stale (relative age is derived from it below).
    const ageTimer = window.setInterval(() => setNow(Date.now()), POLL_INTERVAL_MS);

    return () => {
      disposed = true;
      window.clearInterval(pollTimer);
      window.clearInterval(ageTimer);
      controller?.abort();
    };
  }, []);

  const reading = state.reading;
  // `now` is only set by the 1 s tick, i.e. after mount: the first (server)
  // render stays deterministic.
  const ageMs = reading && now != null ? now - reading.observedAt : null;
  const status: SensorStatus =
    state.status === 'live' && ageMs != null && ageMs > FRESHNESS_MS ? 'stale' : state.status;

  return { reading, status, updatedAt: state.updatedAt, ageMs };
}

/* ------------------------------- formatting ------------------------------ */

function formatNumber(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

function formatClockTime(ms: number): string {
  return new Date(ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function formatAge(ms: number): string {
  const seconds = Math.max(0, Math.round(ms / 1000));
  return seconds < 60 ? `${seconds}s ago` : `${Math.round(seconds / 60)}m ago`;
}

/* --------------------------------- view ---------------------------------- */

function SensorTile({
  label,
  value,
  unit,
  primary = false,
  tone = null,
}: {
  label: string;
  value: string | null;
  unit?: string;
  primary?: boolean;
  tone?: SensorTone;
}) {
  const available = value != null;
  return (
    <div
      className={`sensor-tile${primary ? ' is-primary' : ''}${available ? '' : ' is-unavailable'}${tone ? ` tone-${tone}` : ''}`}
    >
      <small>{label}</small>
      <strong>
        {available ? value : 'N/A'}
        {available && unit ? <span className="sensor-unit">{unit}</span> : null}
      </strong>
    </div>
  );
}

export default function SensorPanel() {
  const { reading, status, updatedAt, ageMs } = useSensorReadings();

  const distance = reading?.distanceCm ?? null;
  const temperature = reading?.temperatureC ?? null;
  const humidity = reading?.humidityPercent ?? null;
  const rain = reading?.rain ?? null;
  const light = reading?.ambientLight ?? null;
  const speed = reading?.speedKph ?? null;
  const gear = reading?.gear ?? null;

  const updatedLabel = updatedAt != null
    ? `Updated ${formatClockTime(updatedAt)}${status === 'live' ? '' : ` · ${formatAge(ageMs ?? 0)}`}`
    : 'No reading received yet';

  return (
    <section className="sensor-deck" aria-label="Live ESP32 sensor readings">
      <div className="sensor-state-block">
        <div className="sensor-title">ESP32 sensors</div>
        <div className={`sensor-state state-${status}`} role="status" aria-live="polite">
          <i aria-hidden="true" />
          <b>{STATUS_LABEL[status]}</b>
          <span>{STATUS_DETAIL[status]}</span>
        </div>
        <small className="sensor-updated">{updatedLabel}</small>
      </div>

      <div className="sensor-tiles">
        <SensorTile label="Obstacle distance" value={distance != null ? formatNumber(distance) : null} unit="cm" primary />
        <SensorTile label="Temperature" value={temperature != null ? formatNumber(temperature) : null} unit="°C" primary />
        <SensorTile label="Humidity" value={humidity != null ? formatNumber(humidity) : null} unit="%" primary />
        <SensorTile
          label="Rain"
          value={rain == null ? null : rain ? 'Detected' : 'Clear'}
          tone={rain ? 'caution' : null}
        />
        {/* The backend defines no unit for ambient_light, so the raw value is shown as-is. */}
        <SensorTile label="Ambient light" value={light != null ? formatNumber(light) : null} />
        <SensorTile label="Vehicle speed" value={speed != null ? formatNumber(speed) : null} unit="km/h" />
        <SensorTile label="Gear" value={gear} />
      </div>
    </section>
  );
}
