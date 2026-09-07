/*
 * adas_bridge.js
 * --------------
 * Tiny browser-side script that polls our FastAPI server's /detect
 * endpoint and dispatches the returned packet as a CustomEvent on the
 * window object.
 *
 * The frontend repo you cloned (frontend_repo/, UNTOUCHED) already
 * listens for the 'adas-detection' event - the relevant TypeScript is
 * in app/page.tsx:
 *
 *   useEffect(() => { ... window.addEventListener('adas-detection',
 *     handleModelDetection); ... }, []);
 *
 * To wire the frontend to this backend, copy the contents of this file
 * into the frontend's app/layout.tsx (or import it via a <script> tag).
 *
 * No network calls happen inside app/page.tsx today - this file is the
 * only thing needed.
 */

(function () {
  var POST_URL = (window.__ADAS_BACKEND__ || 'http://localhost:7000') + '/detect';
  var POLL_MS  = window.__ADAS_POLL_MS__ || 700;     // send a frame every 700 ms
  var SOURCE_LABEL = window.__ADAS_SOURCE__ || 'model';

  // Enable with: window.__ADAS_BRIDGE_ENABLED__ = true;   in DevTools, OR
  // we enable automatically if /health responds.
  function openStream() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      console.warn('[ADAS bridge] getUserMedia not available');
      return;
    }
    return navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
  }

  function start() {
    var stream = openStream();
    if (!stream) return;
    var tracks = stream.getTracks();
    var video = document.createElement('video');
    video.setAttribute('playsinline', '');
    video.srcObject = stream;
    video.play();

    var canvas = document.createElement('canvas');

    async function loop() {
      if (video.readyState < 2) {
        return setTimeout(loop, 200);
      }
      canvas.width  = video.videoWidth  || 640;
      canvas.height = video.videoHeight || 480;
      var ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(async function (blob) {
        if (!blob) return;
        var fd = new FormData();
        fd.append('file', blob, 'frame.jpg');
        try {
          var resp = await fetch(POST_URL, { method: 'POST', body: fd });
          if (!resp.ok) {
            console.warn('[ADAS bridge] /detect HTTP', resp.status);
            return;
          }
          var packet = await resp.json();
          packet = Object.assign({}, packet, { source: SOURCE_LABEL });
          window.dispatchEvent(new CustomEvent('adas-detection', { detail: packet }));
        } catch (e) {
          // backend not up - silently retry; the frontend will just show "Standby"
        }
      }, 'image/jpeg', 0.7);
    }
    setInterval(loop, POLL_MS);
    loop();
    window.addEventListener('beforeunload', function () {
      tracks.forEach(function (t) { return t.stop(); });
    });
  }

  // Auto-enable in production; allow manual disabling via __ADAS_DISABLE__ === true
  if (window.__ADAS_DISABLE__ === true) return;
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();
