// services/attendance.service.js
const Attendance = require('../models/attendance.model');

class AttendanceService {
    // Record attendance
    async createRecord(data) {
        const record = new Attendance({
            staffId: data.staffId,
            name: data.name,
            type: data.type,   // must be one of 'check-in', 'check-out', 'leave'
            note: data.note || ''
        });
        return await record.save();
    }

    // Get all attendance records
    async getAllRecords(filter = {}) {
        return await Attendance.find(filter).sort({ createdAt: -1 });
    }

    // Optionally: Get records by type
    async getRecordsByType(type) {
        return await this.getAllRecords({ type });
    }
}

module.exports = new AttendanceService();
