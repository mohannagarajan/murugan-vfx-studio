import React from 'react';
import {registerRoot, Composition, AbsoluteFill, Audio, Img, staticFile, useCurrentFrame, useVideoConfig, spring, interpolate} from 'remotion';
import data from '../public/manifest.json';

const h = React.createElement;

const VfxScene: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const t = frame / fps;

  let idx = data.scenes.findIndex((s) => t >= s.start && t < s.end);
  if (idx === -1) idx = data.scenes.length - 1;
  const scene = data.scenes[idx];

  const startF = Math.floor(scene.start * fps);
  const durF = Math.max(1, Math.floor((scene.end - scene.start) * fps));
  const localF = Math.max(0, frame - startF);

  // 1. Sky + Sun: Gentle cinematic zoom & warm glow
  const skyScale = interpolate(localF, [0, durF], [1.0, 1.07]);

  // 2. Clouds: Slide in smoothly from left/right (frames 4 to 28) & drift continuously
  const cloudIn = spring({frame: Math.max(0, localF - 4), fps, config: {damping: 16, stiffness: 95}});
  const cloudX = interpolate(cloudIn, [0, 1], [-180, 0]) + Math.sin(localF / 40) * 14;

  // 3. Paper Hills / Ground: Rise up like a pop-up book stage (frames 12 to 38)
  const hillIn = spring({frame: Math.max(0, localF - 12), fps, config: {damping: 13, stiffness: 115, mass: 0.9}});
  const hillY = interpolate(hillIn, [0, 1], [520, 0]);
  const hillScale = interpolate(localF, [0, durF], [1.02, 1.10]);

  // 4. Characters: Pop-up sequentially AFTER the stage builds (starting frame 26, spaced by 14 frames)
  const charLayers = scene.chars.map((charSrc, i) => {
    const pop = spring({
      frame: Math.max(0, localF - (26 + i * 14)),
      fps,
      config: {damping: 10, stiffness: 150, mass: 0.78},
    });
    const y = interpolate(pop, [0, 1], [760, 0]) + Math.sin((localF + i * 24) / 28) * 3.5;
    const scale = interpolate(pop, [0, 1], [0.72, 1.0]);
    const rot = interpolate(pop, [0, 1], [i % 2 === 0 ? -4 : 4, 0]);

    return h(
      AbsoluteFill,
      {
        key: charSrc,
        style: {
          transform: `translateY(${y}px) scale(${scale}) rotate(${rot}deg)`,
          filter: 'drop-shadow(0px 22px 24px rgba(0,0,0,0.62))',
        },
      },
      h(Img, {src: staticFile(charSrc), style: {width: '100%', height: '100%', objectFit: 'cover'}})
    );
  });

  return h(
    AbsoluteFill,
    {style: {backgroundColor: '#1a0d05', overflow: 'hidden'}},
    h(Audio, {src: staticFile('voice.mp3')}),
    // Layer 1: Sky + Sun Rays
    h(
      AbsoluteFill,
      {style: {transform: `scale(${skyScale})`}},
      h(Img, {src: staticFile(scene.sky), style: {width: '100%', height: '100%', objectFit: 'cover'}})
    ),
    // Layer 2: Drifting Paper Clouds
    h(
      AbsoluteFill,
      {
        style: {
          transform: `translateX(${cloudX}px) scale(1.06)`,
          filter: 'drop-shadow(0px 10px 14px rgba(0,0,0,0.35))',
        },
      },
      h(Img, {src: staticFile(scene.clouds), style: {width: '100%', height: '100%', objectFit: 'cover'}})
    ),
    // Layer 3: Rising Paper Hills / Foreground Stage
    h(
      AbsoluteFill,
      {
        style: {
          transform: `translateY(${hillY}px) scale(${hillScale})`,
          filter: 'drop-shadow(0px -14px 20px rgba(0,0,0,0.45))',
        },
      },
      h(Img, {src: staticFile(scene.hills), style: {width: '100%', height: '100%', objectFit: 'cover'}})
    ),
    // Layer 4: Pop-up Sticker Characters
    ...charLayers
  );
};

const Root: React.FC = () =>
  h(Composition, {
    id: 'VfxScene',
    component: VfxScene,
    durationInFrames: data.durationInFrames || 1800,
    fps: 60,
    width: 1080,
    height: 1920,
  });

registerRoot(Root);
