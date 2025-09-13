import { spawn } from 'child_process';

const py = spawn('src/assistant/python/python.exe', ['src/assistant/scripts/worker.py']);

py.stdout.on('data', (data) => console.log(data.toString()));

py.stderr.on('data', (data) => console.error(data.toString()));

export function sendCommand(command) {
    py.stdin.write(JSON.stringify(command) + '\n');
}

export function workerStop() {
    py.kill();
}