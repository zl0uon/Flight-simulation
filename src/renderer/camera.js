// 웹캠을 켜서 <video> 에 연결 (좌우반전은 style.css 에서 화면에만 적용)
export async function startCamera(video) {
  // 카메라가 지원하면 60fps 로 (프레임이 많을수록 지연이 줄어듦)
  video.srcObject = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 360, frameRate: { ideal: 60 } },
    audio: false,
  });
  await new Promise((resolve) => (video.onloadedmetadata = resolve));
  await video.play();
  return { width: video.videoWidth, height: video.videoHeight };
}
