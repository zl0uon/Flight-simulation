// 1초마다 처리한 카메라 프레임 수를 세서 오른쪽 위에 표시
export class FpsCounter {
  constructor(el) {
    this.el = el;
    this.frames = 0;
    this.since = performance.now();
    this.value = 0;
  }

  tick(now) {
    this.frames++;
    if (now - this.since < 1000) return;
    this.value = this.frames / ((now - this.since) / 1000);
    this.frames = 0;
    this.since = now;
    this.el.textContent = `${Math.round(this.value)} FPS`;
    this.el.classList.toggle('slow', this.value < 24); // 어두우면 카메라가 느려짐
    this.el.title = this.value < 24 ? '어두우면 카메라 프레임이 느려져요' : '';
  }
}
