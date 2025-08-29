// configs/db.config.js
const mongoose = require('mongoose');

const connectDB = async () => {
    try {
        const conn = await mongoose.connect(process.env.MONGO_URI, {
            useNewUrlParser: true,
            useUnifiedTopology: true,
            // useFindAndModify: false, // no longer needed in latest mongoose
            // useCreateIndex: true,    // no longer needed in latest mongoose
        });
        console.log(`MongoDB connected: ${conn.connection.host}:${conn.connection.port}`);
    } catch (error) {
        console.error(`Error: ${error.message}`);
        process.exit(1); // Exit process with failure
    }
};

module.exports = connectDB;
