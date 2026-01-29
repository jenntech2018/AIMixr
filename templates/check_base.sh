#!/bin/bash

MISSING=0

echo "Scanning current directory for missing '{% extends \"base.html\" %}'..."

for file in *.html; do
    if ! grep -q '{% extends "base.html" %}' "$file"; then
        echo "❌ Missing extends in: $file"
        MISSING=1
    fi
done

if [ $MISSING -eq 0 ]; then
    echo "✅ All templates extend base.html"
else
    echo "⚠️ Some templates are missing the extends line"
fi
