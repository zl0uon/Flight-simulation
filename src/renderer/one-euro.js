// One Euro Filter: 느리게 움직일 땐 떨림을 강하게 줄이고, 빠르게 움직일 땐 지연을 줄인다
// https://gery.casiez.net/1euro/
class LowPass {
  constructor() {
    this.y = null;
  }
  filter(x, alpha) {
    this.y = this.y === null ? x : alpha * x + (1 - alpha) * this.y;
    return this.y;
  }
}

const alpha = (cutoff, dt) => {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dt);
};

export class OneEuro {
  constructor({ minCutoff = 1.0, beta = 0.0, dCutoff = 1.0 } = {}) {
    Object.assign(this, { minCutoff, beta, dCutoff });
    this.reset();
  }

  reset() {
    this.x = new LowPass();
    this.dx = new LowPass();
    this.lastT = null;
  }

  filter(value, t) {
    const dt = this.lastT === null ? 1 / 30 : Math.max(1e-3, t - this.lastT);
    const rawDx = this.x.y === null ? 0 : (value - this.x.y) / dt;
    const edx = this.dx.filter(rawDx, alpha(this.dCutoff, dt));
    const cutoff = this.minCutoff + this.beta * Math.abs(edx);
    this.lastT = t;
    return this.x.filter(value, alpha(cutoff, dt));
  }
}
