import { getUiPath } from "../pathResolver.js";
import { pathToFileURL } from 'url';
import { ipcMain } from "electron";

export function isDev() {
    return process.env.NODE_ENV === 'development';
}

export function validateEventFrame(frame) {
    console.log(frame.url);
    if (isDev() && new URL(frame.url).host === 'localhost:5125') {
        return;
    }
    if (frame.url !== pathToFileURL(getUiPath()).toString()) {
        throw new Error('malicious event');
    }
}

export function ipcHandle(key, handler) {
    ipcMain.handle(key, () => handler());
} 