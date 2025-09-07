
import osUtils from 'os-utils';
import fs from 'fs';
import os from 'os';
import { exec } from 'child_process';
const POLLING_INTERVAL = 500;
const PLATFORM = process.platform;

export function pollResource(mainWin) {
    setInterval(async () => {
        const cpuUsage = await getCpuUsage();
        const ramUsage = getRamUsage();
        const diskUsage = getStorageData().usage;
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
function getDiskUsage(callback) {
    exec('wmic logicaldisk get name,freespace,size', (error, stdout, stderr) => {
        if (error) {
            console.error(`Error executing wmic: ${error.message}`);
            return callback(error);
        }
        if (stderr) {
            console.error(`wmic stderr: ${stderr}`);
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

        callback(null, {
            total: totalSize,
            free: totalFree,
            used: totalSize - totalFree,
            usage: 1 - totalFree / totalSize
        });
    });
}

function getStorageData() {

    let free;
    let total;
    //require node 18
    if (PLATFORM === 'win32') {
        const diskInfo = getDiskUsage((err, data) => {
            if (err) {
                console.error('Failed to get total disk storage:', err);
            } else {
                const { total, free, used, usage } = data;
                return { total, free, used, usage };
            }
            console.log(total);
        })

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