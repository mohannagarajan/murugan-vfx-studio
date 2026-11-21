import React from 'react';
import {registerRoot, Composition, AbsoluteFill, Audio, Img, staticFile, useCurrentFrame, useVideoConfig, interpolate} from 'remotion';
import data from '../public/manifest.json';

const h = React.createElement;

// Floating divine golden dust / embers particle overlay
const GoldenParticles: React.FC<{frame: number}> = ({frame}) => {
  const particles = Array.from({length: 24}, (_, i) => {
    const speed = 0.6 + (i % 5) * 0.25;
    const xBase = ((i * 197) % 1080);
    const x = xBase + Math.sin((frame * 0.02) + i) * 35;
    const y = (1920 - ((frame * speed * 2.2 + i * 110) % 2000));
    const size = 4 + (i % 4) * 3;
    const opacity = 0.25 + Math.sin(frame * 0.05 + i) * 0.2;

    return h('div', {
      key: i,
      style: {
        position: 'absolute',
        left: `${x}px`,
        top: `${y}px`,
        width: `${size}px`,
        height: `${size}px`,
        borderRadius: '50%',
        backgroundColor: '#ffd166',
        boxShadow: '0 0 12px 3px rgba(255, 195, 50, 0.85)',
        opacity: Math.max(0, opacity),
      },
    });
  });

  return h(AbsoluteFill, {style: {pointerEvents: 'none'}}, ...particles);
};

const SceneLayer: React.FC<{
  scene: {start: number; end: number; image: string; depth: string};
  index: number;
  frame: number;
  fps: number;
}> = ({scene, index, frame, fps}) => {
  const startF = Math.floor(scene.start * fps);
  const endF = Math.floor(scene.end * fps);
  const durF = Math.max(1, endF - startF);
  const localF = frame - startF;

  // Render slightly beyond endF for smooth cross-dissolve overlap
  if (localF < -24 || localF > durF + 24) return null;

  // Smooth 24-frame (0.4s) cross-dissolve between scenes
  const fadeIn = index === 0 ? 1 : interpolate(localF, [0, 24], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const fadeOut = index === data.scenes.length - 1 ? 1 : interpolate(localF, [durF, durF + 24], [1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const opacity = Math.min(fadeIn, fadeOut);

  // Alternate smooth 3D camera push-in / pull-out + gentle orbital pan
  const progress = Math.min(1.2, Math.max(0, localF / durF));
  const scale = index % 2 === 0
    ? interpolate(progress, [0, 1], [1.02, 1.15])
    : interpolate(progress, [0, 1], [1.15, 1.03]);

  const panX = (index % 2 === 0 ? 1 : -1) * interpolate(progress, [0, 1], [-22, 22]);
  const panY = interpolate(progress, [0, 1], [14, -14]);

  // Foreground depth-masked parallax boost (uses Depth-Anything-V2 map as luminance mask)
  const fgScale = scale * 1.045;
  const fgPanX = panX * 1.65;
  const fgPanY = panY * 1.45;

  // Subtle volumetric sunbeam shimmer
  const rayPulse = 0.14 + Math.sin(localF * 0.04) * 0.05;

  return h(
    AbsoluteFill,
    {style: {opacity, overflow: 'hidden'}},
    // Base full realistic scene (Background depth plane)
    h(
      AbsoluteFill,
      {
        style: {
          transform: `translate(${panX}px, ${panY}px) scale(${scale})`,
          transformOrigin: 'center center',
        },
      },
      h(Img, {src: staticFile(scene.image), style: {width: '100%', height: '100%', objectFit: 'cover'}})
    ),
    // Foreground 3D Parallax layer masked by AI Depth Map (creates true 3D camera separation with ZERO cutout artifacts)
    h(
      AbsoluteFill,
      {
        style: {
          transform: `translate(${fgPanX}px, ${fgPanY}px) scale(${fgScale})`,
          transformOrigin: 'center center',
          WebkitMaskImage: `url(${staticFile(scene.depth)})`,
          maskImage: `url(${staticFile(scene.depth)})`,
          WebkitMaskMode: 'luminance',
          maskMode: 'luminance',
          WebkitMaskSize: 'cover',
          maskSize: 'cover',
          opacity: 0.72,
        },
      },
      h(Img, {src: staticFile(scene.image), style: {width: '100%', height: '100%', objectFit: 'cover'}})
    ),
    // Warm Volumetric Sun-Glow Overlay
    h(AbsoluteFill, {
      style: {
        background: 'radial-gradient(circle at 50% 28%, rgba(255, 214, 102, 0.45) 0%, rgba(255, 140, 0, 0.12) 45%, rgba(0, 0, 0, 0.45) 100%)',
        mixBlendMode: 'screen',
        opacity: rayPulse,
        pointerEvents: 'none',
      },
    })
  );
};

const VfxScene: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  return h(
    AbsoluteFill,
    {style: {backgroundColor: '#090402', overflow: 'hidden'}},
    h(Audio, {src: staticFile('voice.mp3')}),
    ...data.scenes.map((scene, idx) =>
      h(SceneLayer, {key: scene.image, scene, index: idx, frame, fps})
    ),
    h(GoldenParticles, {frame}),
    // Cinematic film vignette
    h(AbsoluteFill, {
      style: {
        background: 'radial-gradient(circle, rgba(0,0,0,0) 58%, rgba(0,0,0,0.62) 100%)',
        pointerEvents: 'none',
      },
    })
  );
};

const Root: React.FC = () =>
  h(Composition, {
    id: 'VfxScene',
    component: VfxScene,
    durationInFrames: data.durationInFrames || 2280,
    fps: 60,
    width: 1080,
    height: 1920,
  });

registerRoot(Root);
