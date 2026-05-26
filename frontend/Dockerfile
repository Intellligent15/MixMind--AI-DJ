FROM node:20-alpine

WORKDIR /app

# NEXT_PUBLIC_API_BASE is read by Next.js at build/serve time and inlined
# into the client bundle. Compose passes it via build args so prod can
# point at the droplet's public IP while dev keeps localhost.
ARG NEXT_PUBLIC_API_BASE=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE=${NEXT_PUBLIC_API_BASE}

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

EXPOSE 3000

CMD ["npm", "run", "dev"]
