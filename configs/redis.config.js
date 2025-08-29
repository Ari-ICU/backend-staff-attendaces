// configs/redis.config.js
const redis = require('redis');

class RedisClient {
    constructor(url = process.env.REDIS_URL) {
        if (!url) {
            throw new Error("REDIS_URL is not defined in environment variables");
        }
        this.client = redis.createClient({ url });
        this._connect();
    }

    async _connect() {
        this.client.on('error', (err) => {
            console.error('Redis Client Error:', err);
        });

        try {
            await this.client.connect();
            const { host, port } = this.client.options.socket;
            console.log(`Redis connected successfully: ${host}:${port}`);
        } catch (err) {
            console.error('Redis connection failed:', err);
        }
    }

    async set(key, value, options = {}) {
        return this.client.set(key, JSON.stringify(value), options);
    }

    async get(key) {
        const data = await this.client.get(key);
        return data ? JSON.parse(data) : null;
    }

    async del(key) {
        return this.client.del(key);
    }

    async ping() {
        return this.client.ping();
    }

    async testSpeed(key, value) {
        const start = Date.now();
        await this.set(key, value);
        await this.get(key);
        const end = Date.now();
        return end - start;
    }
}

module.exports = RedisClient;
