import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  metadataBase: new URL('https://ares-drive-adas.shivanipatodia11.chatgpt.site'),
  title: 'ARES Drive | Camera-Based Driver Assistance',
  description: 'A camera-first driver-assistance frontend for lane, object, road, sign, weather, and traffic-light detections.',
  openGraph: {
    title: 'ARES Drive | Camera-Based Driver Assistance',
    description: 'A restrained automotive interface built around live camera detections and clear driver warnings.',
    type: 'website',
    images: [{
      url: 'https://ares-drive-adas.shivanipatodia11.chatgpt.site/og.png',
      width: 1200,
      height: 630,
      alt: 'ARES Drive camera-based driver assistance display',
    }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'ARES Drive | Camera-Based Driver Assistance',
    description: 'A restrained automotive interface built around live camera detections and clear driver warnings.',
    images: ['https://ares-drive-adas.shivanipatodia11.chatgpt.site/og.png'],
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <head>
        {/* ADAS bridge: polls the backend /detect and emits the
            'adas-detection' event that page.tsx listens for. The UI is
            untouched — only this wiring is added so the React app consumes
            real model outputs instead of the demo sequence. */}
        <script
          dangerouslySetInnerHTML={{
            __html: `window.__ADAS_BACKEND__ = (window.__ADAS_BACKEND__ || 'http://localhost:8000');
window.__ADAS_POLL_MS__ = 700;`,
          }}
        />
        <script src="http://localhost:8000/adas_bridge.js" defer />
      </head>
      <body>{children}</body>
    </html>
  );
}
