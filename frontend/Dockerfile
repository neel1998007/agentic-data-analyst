# ── Stage 1: Build ───────────────────────────────────────────────────────────
# Compile the React/TypeScript app with Vite inside a Node container.
# The built output lands in /app/dist.
FROM node:20-alpine AS build

WORKDIR /app

# Copy manifests first so npm ci is cached separately from source.
# .dockerignore excludes node_modules so Windows-compiled binaries never
# enter the build context — npm ci installs fresh Linux-native packages.
COPY package.json package-lock.json ./
RUN npm ci

# Copy source and build.
COPY . .
RUN npm run build

# ── Stage 2: Serve ───────────────────────────────────────────────────────────
# Copy only the compiled static files into a lean Nginx image.
# The node_modules and full source tree are discarded at this stage —
# the final image contains only /dist output and Nginx.
FROM nginx:alpine

# Replace the default Nginx config with our SPA + proxy config.
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy the Vite build output from Stage 1.
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
