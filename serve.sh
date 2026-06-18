#!/usr/bin/env bash
# Start the Jekyll site locally at http://127.0.0.1:4000/ with live reload.
# Usage: ./serve.sh
set -euo pipefail

# Use rbenv's Ruby 3.1.6 (system Ruby 2.6 is too old for github-pages/jekyll).
if command -v rbenv >/dev/null 2>&1; then
  eval "$(rbenv init - bash)"
  rbenv shell 3.1.6
fi

echo "Ruby: $(ruby -v)"

# Install gems if anything is missing (no-op once installed).
bundle check >/dev/null 2>&1 || bundle install

echo "Serving at http://127.0.0.1:4000/  (Ctrl+C to stop)"
exec bundle exec jekyll serve --livereload
