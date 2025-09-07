
import osUtils from 'os-utils';
import fs from 'fs';
import { exec } from 'child_process';
const POLLING_INTERVAL = 500;
const PLATFORM = process.platform;

export function pollResource(mainWindow) {
    setInterval(async () => {
        const cpuUsage = await getCpuUsage();
        const ramUsage = getRamUsage();
        const diskUsage = (await getStorageData()).usage;
        mainWindow.webContents.send('satistics', { cpuUsage, ramUsage, diskUsage });
    }, POLLING_INTERVAL);
}

function getCpuUsage() {
    return new Promise((resolve) => {
        osUtils.cpuUsage(resolve)
    })
}

function getRamUsage() {
    return 1 - osUtils.freememPercentage();
}
function getDiskUsage() {
    return new Promise((resolve, reject) => {
        exec('wmic logicaldisk get name,freespace,size', (error, stdout, stderr) => {
            if (error) {
                return reject(error);
            }

            const lines = stdout.trim().split('\n');
            let totalSize = 0;
            let totalFree = 0;

            // Skip header (first line)
            for (let i = 1; i < lines.length; i++) {
                const parts = lines[i].trim().split(/\s+/); // split by whitespace
                if (parts.length >= 3) {
                    const free = parseInt(parts[0], 10);
                    const size = parseInt(parts[2], 10);

                    if (!isNaN(size)) totalSize += size;
                    if (!isNaN(free)) totalFree += free;
                }
            }

            resolve({
                total: totalSize,
                free: totalFree,
                used: totalSize - totalFree,
                usage: 1 - totalFree / totalSize
            });
        });
    });
}
async function getStorageData() {

    let free;
    let total;
    //require node 18
    if (PLATFORM === 'win32') {
        const diskInfo = await getDiskUsage()
        free = diskInfo.free;
        total = diskInfo.total;
    }
    else {
        const stats = fs.statfsSync('/');
        total = stats.bsize * stats.blocks;
        free = stats.bsize * stats.bfree;
    }
    return {
        total: Math.floor(total / 1_000_000_000),
        usage: 1 - free / total
    };
}