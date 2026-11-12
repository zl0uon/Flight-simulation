// PiP: 카메라 영상 + 손 인식 오버레이 + 상태 표시를 한 캔버스에 합성해서 PiP 창으로 띄운다
import { COLORS } from './draw.js';

const BAR = 44;

export function setupPip({ video, overlay, button, pipVideo, label }) {
  const canvas = document.createElement('canvas');
  canvas.width = overlay.width;
  canvas.height = overlay.height + BAR;
  const ctx = canvas.getContext('2d');
  pipVideo.srcObject = canvas.captureStream();
  let starting = false;
  let last = { state: '', mode: 'none', fps: '' };

  function render(state = last.state, mode = last.mode, fps = last.fps) {
    last = { state, mode, fps };
    if (!document.pictureInPictureElement && !starting) return;
    const { width: W, height: H } = overlay;
    ctx.save();
    ctx.translate(W, 0);
    ctx.scale(-1, 1); // 메인 화면과 같은 좌우반전
    ctx.drawImage(video, 0, 0, W, H);
    ctx.restore();
    ctx.drawImage(overlay, 0, 0); // 점·선은 이미 반전 좌표로 그려져 있음

    ctx.font = '600 18px -apple-system, sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'top';
    ctx.fillStyle = '#fff';
    ctx.fillText(fps, W - 10, 10);

    ctx.fillStyle = '#111';
    ctx.fillRect(0, H, W, BAR);
    ctx.fillStyle = COLORS[mode] ?? '#eee';
    ctx.fillRect(0, H, 8, BAR);
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    ctx.font = '600 22px -apple-system, "Apple SD Gothic Neo", sans-serif';
    ctx.fillStyle = '#eee';
    ctx.fillText(state, 20, H + BAR / 2);
    ctx.textAlign = 'right';
    ctx.font = '16px -apple-system, "Apple SD Gothic Neo", sans-serif';
    ctx.fillStyle = '#9e9e9e';
    ctx.fillText(label, W - 12, H + BAR / 2);
  }

  const setUI = () => (button.textContent = document.pictureInPictureElement ? 'PiP 닫기' : 'PiP');
  button.addEventListener('click', async () => {
    if (document.pictureInPictureElement) {
      await document.exitPictureInPicture();
      return;
    }
    starting = true;
    try {
      render(); // 첫 프레임이 있어야 PiP 를 열 수 있음
      await pipVideo.play();
      await pipVideo.requestPictureInPicture();
    } catch (err) {
      alert(`PiP 를 열 수 없어요: ${err.message}`);
    } finally {
      starting = false;
    }
  });
  pipVideo.addEventListener('enterpictureinpicture', setUI);
  pipVideo.addEventListener('leavepictureinpicture', setUI);
  button.hidden = !document.pictureInPictureEnabled;

  return render;
}
