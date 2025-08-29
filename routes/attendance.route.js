// routes/attendance.route.js
const express = require('express');
const router = express.Router();
const AttendanceController = require('../controllers/attendance.controller');

router.post('/', AttendanceController.createRecord.bind(AttendanceController));
router.get('/', AttendanceController.getAllRecords.bind(AttendanceController));

module.exports = router;
