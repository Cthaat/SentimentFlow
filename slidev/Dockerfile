FROM node:22-bookworm-slim AS deps

WORKDIR /app
ARG YARN_REGISTRY=https://registry.npmjs.org

COPY package.json yarn.lock ./
RUN yarn config set registry "$YARN_REGISTRY" \
    && yarn install --frozen-lockfile

FROM node:22-bookworm-slim AS builder

WORKDIR /app
ENV NODE_ENV=production

COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN yarn build

FROM nginx:1.27-alpine AS runner

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 3030

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD wget -qO- http://127.0.0.1:3030/ >/dev/null || exit 1
