# Frontend Dockerfile for Railway deployment
# Vite + React 19 with Express server

FROM node:20-slim AS builder

# Install pnpm
RUN corepack enable && corepack prepare pnpm@10.4.1 --activate

WORKDIR /app

# Copy package files
COPY package.json pnpm-lock.yaml* ./
COPY patches ./patches

# Install dependencies
RUN pnpm install --frozen-lockfile

# Copy source
COPY . .

# Build
RUN pnpm build

# Production stage
FROM node:20-slim AS runner

RUN corepack enable && corepack prepare pnpm@10.4.1 --activate

WORKDIR /app

ENV NODE_ENV=production

# Copy built files and production dependencies
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package.json ./
COPY --from=builder /app/pnpm-lock.yaml* ./
COPY --from=builder /app/patches ./patches

# Install production dependencies only
RUN pnpm install --prod --frozen-lockfile

EXPOSE 3000

CMD ["node", "dist/index.js"]
