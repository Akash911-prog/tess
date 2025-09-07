import { app, BrowserWindow } from 'electron';
import { ipcHandle, isDev } from './Utils/utils.js';
import { getUiPath, getPreloadPath } from './pathResolver.js';
import { pollResource } from './Utils/resourceUsage.js';

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
    pollResource(mainWindow)
})
