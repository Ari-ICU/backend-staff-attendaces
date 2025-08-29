const express = require('express');
const router = express.Router();
const staffController = require('../controllers/staff.controller');
const multer = require('multer');

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'Uploads/');
    },
    filename: (req, file, cb) => {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
        cb(null, `${uniqueSuffix}-${file.originalname}`);
    },
});
const upload = multer({
    storage,
    fileFilter: (req, file, cb) => {
        if (file.mimetype.startsWith('image/')) {
            cb(null, true);
        } else {
            cb(new Error('Only images are allowed'), false);
        }
    },
    limits: { fileSize: 5 * 1024 * 1024 },
});

// CRUD routes
router.post('/', upload.single('image'), staffController.createStaff.bind(staffController));
router.get('/', staffController.getAllStaff.bind(staffController));
router.get('/:id', staffController.getStaffById.bind(staffController));
router.put('/:id', upload.single('image'), staffController.updateStaff.bind(staffController));
router.delete('/:id', staffController.deleteStaff.bind(staffController));
router.post('/verify', staffController.verifyFace.bind(staffController));

module.exports = router;