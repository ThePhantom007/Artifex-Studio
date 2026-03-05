# ──────────────────────────────────────────────────────────────────
# ArtifexStudio — Frontend (Nginx static file server)
# Serves index.html, styles.css, and script.js.
# The browser JS calls the backend API directly on port 8000.
# ──────────────────────────────────────────────────────────────────
FROM nginx:1.27-alpine

# Remove the default Nginx placeholder page
RUN rm -rf /usr/share/nginx/html/*

# Copy static frontend assets
COPY index.html  /usr/share/nginx/html/
COPY styles.css  /usr/share/nginx/html/
COPY script.js   /usr/share/nginx/html/

# Custom Nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=20s --timeout=5s --retries=3 \
    CMD wget -qO- http://localhost/ > /dev/null || exit 1