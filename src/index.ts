import React from 'react';
import {registerRoot, Composition, AbsoluteFill, Audio, Img, staticFile, useCurrentFrame, useVideoConfig, spring, interpolate} from 'remotion';
import data from '../public/manifest.json';

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

  const bgScale = interpolate(localF, [0, durF], [1.02, 1.09]);
  const bgRotate = Math.sin(localF / 44) * 0.65;

  return (
    <AbsoluteFill style={{backgroundColor: '#140b04', overflow: 'hidden', perspective: 1400}}>
      <Audio src={staticFile('voice.mp3')} />
      <AbsoluteFill style={{transform: `scale(${bgScale}) rotateZ(${bgRotate}deg)`}}>
        <Img src={staticFile(scene.bg)} style={{width: '100%', height: '100%', objectFit: 'cover'}} />
      </AbsoluteFill>
      {scene.chars.map((charSrc, i) => {
        const pop = spring({
          frame: Math.max(0, localF - (6 + i * 14)),
          fps,
          config: {damping: 10, stiffness: 145, mass: 0.85},
        });
        const y = interpolate(pop, [0, 1], [680, 0]) + Math.sin((localF + i * 28) / 26) * 4;
        const scale = interpolate(pop, [0, 1], [0.85, 1.0]);

        return (
          <AbsoluteFill
            key={charSrc}
            style={{
              transform: `translateY(${y}px) scale(${scale})`,
              filter: 'drop-shadow(0px 24px 28px rgba(0,0,0,0.68))',
            }}
          >
            <Img src={staticFile(charSrc)} style={{width: '100%', height: '100%', objectFit: 'cover'}} />
          </AbsoluteFill>
        );
      })}
    </AbsoluteFill>
  );
};

const Root: React.FC = () => (
  <Composition
    id="VfxScene"
    component={VfxScene}
    durationInFrames={data.durationInFrames || 1800}
    fps={60}
    width={1080}
    height={1920}
  />
);

registerRoot(Root);
