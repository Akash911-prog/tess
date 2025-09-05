const electron = require('electron')

electron.contextBridge.exposeInMainWorld('API', {
    hello: () => ipcInvoke('hello'),
})

function ipcOn(key, callback) {
    const cb = (_, payload) => callback(payload)
    electron.ipcRenderer.on(key, cb);
    return () => { electron.ipcRenderer.off(key, cb) };
}

function ipcInvoke(key) {
    electron.ipcRenderer.invoke(key);
}