const Staff = require('../models/staff.models');
const path = require('path');
const fs = require('fs').promises;

class StaffService {
    async createStaff(data, file) {
        try {
            if (file) {
                data.imageUrl = `http://localhost:5000/uploads/${file.filename}`;
            }
            const staff = new Staff(data);
            return await staff.save();
        } catch (err) {
            if (file) {
                await fs.unlink(path.join(__dirname, '..', 'Uploads', file.filename)).catch(() => { });
            }
            if (err.code === 11000) {
                throw new Error('Email already exists');
            }
            throw new Error('Error creating staff: ' + err.message);
        }
    }

    async getAllStaff({ page = 1, limit = 10 }) {
        try {
            const skip = (page - 1) * limit;
            const staff = await Staff.find()
                .select('name email phone role permissions imageUrl faceEncoding createdAt updatedAt')
                .skip(skip)
                .limit(parseInt(limit));
            const total = await Staff.countDocuments();
            return { staff, total, page: parseInt(page), limit: parseInt(limit) };
        } catch (err) {
            throw new Error('Error fetching staff: ' + err.message);
        }
    }

    async getStaffById(id) {
        try {
            if (!mongoose.Types.ObjectId.isValid(id)) {
                throw new Error('Invalid staff ID');
            }
            return await Staff.findById(id);
        } catch (err) {
            if (err.name === 'CastError') {
                throw new Error('Invalid staff ID format');
            }
            throw new Error('Error fetching staff by ID: ' + err.message);
        }
    }

    async updateStaff(id, data, file) {
        try {
            if (!mongoose.Types.ObjectId.isValid(id)) {
                throw new Error('Invalid staff ID');
            }
            if (file) {
                data.imageUrl = `http://localhost:5000/uploads/${file.filename}`;
                const oldStaff = await Staff.findById(id);
                if (oldStaff && oldStaff.imageUrl) {
                    const oldImagePath = path.join(
                        __dirname,
                        '..',
                        'Uploads',
                        oldStaff.imageUrl.split('/uploads/')[1]
                    );
                    await fs.unlink(oldImagePath).catch(() => { });
                }
            }
            return await Staff.findByIdAndUpdate(
                id,
                { ...data, updatedAt: Date.now() },
                { new: true }
            );
        } catch (err) {
            if (file) {
                await fs.unlink(path.join(__dirname, '..', 'Uploads', file.filename)).catch(() => { });
            }
            if (err.name === 'CastError') {
                throw new Error('Invalid staff ID format');
            }
            throw new Error('Error updating staff: ' + err.message);
        }
    }

    async deleteStaff(id) {
        try {
            if (!mongoose.Types.ObjectId.isValid(id)) {
                throw new Error('Invalid staff ID');
            }
            const staff = await Staff.findByIdAndDelete(id);
            if (staff && staff.imageUrl) {
                const imagePath = path.join(__dirname, '..', 'Uploads', staff.imageUrl.split('/uploads/')[1]);
                await fs.unlink(imagePath).catch(() => { });
            }
            return staff;
        } catch (err) {
            if (err.name === 'CastError') {
                throw new Error('Invalid staff ID format');
            }
            throw new Error('Error deleting staff: ' + err.message);
        }
    }



    async checkPermissions(staffId, requiredPermission) {
        try {
            const staff = await this.getStaffById(staffId);
            if (!staff) throw new Error('Staff not found');
            if (!staff.permissions || !staff.permissions.includes(requiredPermission)) {
                throw new Error('Permission denied');
            }
            return true;
        } catch (err) {
            throw new Error('Error checking permissions: ' + err.message);
        }
    }

    async verifyFaceEncoding(encoding) {
        if (!encoding || !Array.isArray(encoding)) {
            throw new Error('Invalid encoding');
        }
        const staffList = await this.getAllStaff({ page: 1, limit: 1000 }); // Fetch all for verification
        const euclideanDistance = (a, b) => {
            let sum = 0;
            for (let i = 0; i < a.length; i++) {
                sum += (a[i] - b[i]) ** 2;
            }
            return Math.sqrt(sum);
        };
        const matches = staffList.staff
            .map(staff => {
                if (!staff.faceEncoding) return null;
                const distance = euclideanDistance(staff.faceEncoding, encoding);
                return { name: staff.name, distance };
            })
            .filter(x => x && x.distance < 0.6);
        if (matches.length > 0) {
            matches.sort((a, b) => a.distance - b.distance);
            return { matched: true, name: matches[0].name };
        }
        return { matched: false };
    }
}

module.exports = new StaffService();