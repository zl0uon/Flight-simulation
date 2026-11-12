const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('handMouse', {
  move: (nx, ny, vx, vy) => ipcRenderer.send('mouse:move', nx, ny, vx, vy),
  stop: () => ipcRenderer.send('mouse:stop'),
  left: (down) => ipcRenderer.send('mouse:left', down),
  rightClick: () => ipcRenderer.send('mouse:right'),
  scroll: (dy) => ipcRenderer.send('mouse:scroll', dy),
  windowGrab: (nx, ny) => ipcRenderer.invoke('window:grab', nx, ny),
  windowMove: (nx, ny) => ipcRenderer.send('window:move', nx, ny),
  windowRelease: () => ipcRenderer.send('window:release'),
  isAccessibilityTrusted: () => ipcRenderer.invoke('accessibility'),
  onTogglePause: (callback) => ipcRenderer.on('toggle-pause', callback),
});
