import { app, BrowserWindow } from 'electron';
import { ipcHandle, isDev } from './utils.js';
import { getUiPath, getPreloadPath } from './pathResolver.js';

app.whenReady().then(() => {
    const mainWindow = new BrowserWindow({
        width: 1000,
        height: 830,
        webPreferences: {
            preload: getPreloadPath()
        }
    })
    if (isDev) {
        mainWindow.loadURL('http://localhost:5125')
    }
    else {
        mainWindow.loadFile(getUiPath())
    }
    ipcHandle('hello', () => console.log('hello'))
})
