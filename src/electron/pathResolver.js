import { app } from "electron";
import path from 'path';
import { isDev } from "./Utils/utils.js";

export function getPreloadPath() {
    // The path to the preload script depends on whether we are running in
    // development or production mode.
    //
    // In development mode, the path is relative to the currently running script.
    // This means that the preload script is in the same directory as the main
    // script.
    //
    // In production mode, the path is relative to the packaged application.
    // This means that the preload script is in the 'src/electron' directory.
    return path.join(
        app.getAppPath(),
        isDev() ? '.' : '..',
        'src\\electron\\preload.cjs'
    );
}

export function getUiPath() {
    return path.join(app.getAppPath(), '/dist-react/index.html');
}