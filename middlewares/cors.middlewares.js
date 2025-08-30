// middleware/cors.js
const cors = require('cors');

const corsOptions = {
  origin: '*', // or restrict to ['http://localhost:5000']
  methods: ['GET', 'POST', 'PUT', 'DELETE'],
  allowedHeaders: ['Content-Type', 'Authorization'],
};

module.exports = cors(corsOptions);
