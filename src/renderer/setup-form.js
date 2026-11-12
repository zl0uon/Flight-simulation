// 실행 시 참가자 ID / 세션 번호 / 손 입력 폼
export const HAND_KO = { Left: '좌', Right: '우' };

export function askSessionInfo() {
  const form = document.getElementById('setup-form');
  return new Promise((resolve) => {
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      const data = new FormData(form);
      document.getElementById('setup').remove();
      resolve({
        participant: data.get('participant').trim(),
        session: Number(data.get('session')),
        hand: data.get('hand'), // 'Left' | 'Right'
      });
    });
  });
}

export const describe = (meta) => `${meta.participant} · 세션 ${meta.session} · ${HAND_KO[meta.hand]}손`;
