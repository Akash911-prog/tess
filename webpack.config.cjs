// webpack.config.cjs
const path = require("path");

const isDev = process.env.NODE_ENV !== "production";

module.exports = {
    mode: isDev ? "development" : "production",
    target: "electron-main",
    entry: path.resolve(__dirname, "src/electron/main.js"),
    output: {
        path: path.resolve(__dirname, "dist-electron"),
        filename: "main.js",
        library: { type: "module" },  // <-- make it ESM
        module: true,
        environment: {
            module: true
        }
    },
    experiments: {
        outputModule: true // <-- enable ESM output
    },
    module: {
        rules: [
            {
                test: /\.m?js$/,
                exclude: /node_modules/
            }
        ]
    },
    node: {
        __dirname: false,
        __filename: false
    },
    devtool: isDev ? "inline-source-map" : false
};
