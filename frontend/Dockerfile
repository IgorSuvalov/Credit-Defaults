# Build Stage
FROM node:18-alpine AS build
WORKDIR /app

ARG VITE_API_URL
ENV VITE_API_URL=${VITE_API_URL}

COPY frontend/package*.json ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
RUN npm run build


FROM nginx:stable-alpine AS production

COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]