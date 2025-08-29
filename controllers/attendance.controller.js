// controllers/attendance.controller.js
const AttendanceService = require('../services/attendance.service');
const ApiResponse = require('../utils/api.response');

const VALID_TYPES = ['check-in', 'check-out', 'leave'];

class AttendanceController {
    async createRecord(req, res) {
        console.log("POST body received:", req.body); // debug
        try {
            const { staffId, name, type, note } = req.body;

            if (!staffId || !name) {
                return ApiResponse.error(res, 'Staff ID and name are required', 400);
            }

            if (!type || !VALID_TYPES.includes(type)) {
                return ApiResponse.error(res, `Type must be one of: ${VALID_TYPES.join(', ')}`, 400);
            }

            const record = await AttendanceService.createRecord({ staffId, name, type, note });
            ApiResponse.success(res, record, 'Attendance recorded', 201);
        } catch (err) {
            ApiResponse.error(res, err.message);
        }
    }

    async getAllRecords(req, res) {
        try {
            const records = await AttendanceService.getAllRecords();
            ApiResponse.success(res, records, 'Attendance records retrieved');
        } catch (err) {
            ApiResponse.error(res, err.message);
        }
    }

    // Optionally: Get records by type
    async getRecordsByType(req, res) {
        try {
            const { type } = req.params;

            if (!VALID_TYPES.includes(type)) {
                return ApiResponse.error(res, `Type must be one of: ${VALID_TYPES.join(', ')}`, 400);
            }

            const records = await AttendanceService.getRecordsByType(type);
            ApiResponse.success(res, records, `Attendance records of type ${type} retrieved`);
        } catch (err) {
            ApiResponse.error(res, err.message);
        }
    }
}

module.exports = new AttendanceController();
