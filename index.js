const fs = require('fs');
const path = require('path');

const files = fs.readdirSync('./dataset');
const numFiles = files.length;

for (let i = 0; i < numFiles; i++) {
    const file = files[i];
    if (file === '.DS_Store') {
        continue;
    }

    const extensionMatch = file.match(/.+\.(jpg|jpeg|png)$/i);
    if (!extensionMatch) {
        console.error(`Failed to get extension for ${file}`);
    }

    const extension = extensionMatch[1];

    const originalPath = path.join(__dirname, 'dataset', file);
    const newPath = path.join(__dirname, 'dataset', `${i}.${extension}`);

    // console.log([originalPath, newPath]);
    try {
        fs.renameSync(originalPath, newPath);
    } catch (e) {
        console.error(`Failed to rename file: ${originalPath}`);
    }
}