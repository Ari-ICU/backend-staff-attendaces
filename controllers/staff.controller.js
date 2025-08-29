const staffService = require('../services/staff.service');
const ApiResponse = require('../utils/api.response');
const { createStaffSchema, updateStaffSchema } = require('../validations/staff.validate');

class StaffController {
    async getAllStaff(req, res) {
        try {
            const { page, limit } = req.query;
            const staffData = await staffService.getAllStaff({ page, limit });
            ApiResponse.success(res, staffData, 'Staff list fetched');
        } catch (err) {
            ApiResponse.error(res, err.message);
        }
    }

    async getStaffById(req, res) {
        try {
            const staff = await staffService.getStaffById(req.params.id);
            if (!staff) return ApiResponse.error(res, 'Staff not found', 404);
            ApiResponse.success(res, staff, 'Staff fetched');
        } catch (err) {
            ApiResponse.error(res, err.message);
        }
    }

    async createStaff(req, res) {
        try {
            const { error } = createStaffSchema.validate(req.body);
            if (error) return ApiResponse.error(res, error.details[0].message, 400);
            const staff = await staffService.createStaff(req.body, req.file);
            ApiResponse.success(res, staff, 'Staff created', 201);
        } catch (err) {
            ApiResponse.error(res, err.message);
        }
    }

    async updateStaff(req, res) {
        try {
            const { error } = updateStaffSchema.validate(req.body);
            if (error) return ApiResponse.error(res, error.details[0].message, 400);
            const staff = await staffService.updateStaff(req.params.id, req.body, req.file);
            if (!staff) return ApiResponse.error(res, 'Staff not found', 404);
            ApiResponse.success(res, staff, 'Staff updated');
        } catch (err) {
            ApiResponse.error(res, err.message);
        }
    }

    async deleteStaff(req, res) {
        try {
            const staff = await staffService.deleteStaff(req.params.id);
            if (!staff) return ApiResponse.error(res, 'Staff not found', 404);
            ApiResponse.success(res, { id: req.params.id }, 'Staff deleted');
        } catch (err) {
            ApiResponse.error(res, err.message);
        }
    }


    async verifyFace(req, res) {
        try {
            const { encoding } = req.body;
            if (!encoding || !Array.isArray(encoding)) {
                return ApiResponse.error(res, 'Invalid face encoding', 400);
            }
            const result = await staffService.verifyFaceEncoding(encoding);
            ApiResponse.success(res, result, 'Face verification completed');
        } catch (err) {
            ApiResponse.error(res, err.message);
        }
    }
}

module.exports = new StaffController();
