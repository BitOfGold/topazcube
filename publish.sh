#!/bin/bash
npm run build
git add .
git commit -m "Publish"
git push
npm publish
