const express = require('express')
const bodyParser = require('body-parser');
const path = require('path');
const fs = require('fs');
const uploadDir = path.join(__dirname, 'Uploads');

if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir, { recursive: true });
    console.log("Uploads folder created");
}

// npm install dotenv node-cache
dotenv = require('dotenv')
cachest = require('node-cache')

dotenv.config()
const app = express()

// import cors middleware
const corsMiddleware = require('./middlewares/cors.middlewares')

// import ratelimit middleware
const rateLimitMiddleware = require('./middlewares/ratelimit.middlewares')

// import mongodb connection
const connectDB = require('./configs/db.config')

// import redis connection
const RedisClient = require('./configs/redis.config')

// port
const port = process.env.PORT || 3000

// checking port
try {
    if (!port) {
        throw new Error("PORT is not defined in environment variables");
    }
} catch (error) {
    console.error(error.message);
    process.exit(1);
}

//check database connection
connectDB();

//check redis
const redis = new RedisClient();

(async () => {
    try {
        // Ping Redis
        const pong = await redis.ping();
        console.log(`Redis ping successful: ${pong}`);

        // Test round-trip speed
        const speedMs = await redis.testSpeed('speed_test', { test: 'data' });
        console.log(`Redis round-trip (set + get) took ${speedMs} ms`);
    } catch (err) {
        console.error('Redis check failed:', err.message);
        process.exit(1);
    }
})();

// Middleware
app.use(express.json())

app.use(bodyParser.json());


// check core middleware
try {
    app.use(corsMiddleware);
    corsMiddleware && console.log("CORS middleware applied successfully.");
} catch (error) {
    console.error(error.message);
    console.log("Failed to apply CORS middleware.");
    process.exit(1);
}

// check rate limiting middleware
try {
    app.use(rateLimitMiddleware);
    rateLimitMiddleware && console.log("Rate limiting middleware applied successfully.");
} catch (error) {
    console.error(error.message);
    console.log("Failed to apply rate limiting middleware.");
    process.exit(1);
}


app.get('/', (req, res) => {
    //index.html
    res.sendFile(path.join(__dirname, 'public', 'index.html'));

})


app.use(express.static(path.join(__dirname, 'public')));
app.use('/uploads', express.static(path.join(__dirname, 'Uploads')));

// global API rate limiting
// app.use('/api', rateLimitMiddleware)

// Staff routes
const staffRoutes = require('./routes/staff.route');
app.use('/api/staff', staffRoutes);

// Attendance routes
const attendanceRoutes = require('./routes/attendance.route');
app.use('/api/attendance', attendanceRoutes);


// Start the server
app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
