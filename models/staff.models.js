const mongoose = require('mongoose');

const staffSchema = new mongoose.Schema({
    name: {
        type: String,
        required: true,
        minlength: 2,
        maxlength: 100,
    },
    email: {
        type: String,
        required: true,
        unique: true,
    },
    phone: {
        type: String,
        match: /^\+?[0-9\s-]{8,15}$/,
    },
    role: {
        type: String,
        enum: ['staff', 'admin'],
        default: 'staff',
    },
    imageUrl: {
        type: String,
        validate: {
            validator: function (v) {
                return !v || /^(https?:\/\/[^\s$.?#].[^\s]*)$/.test(v);
            },
            message: 'Invalid URL format for imageUrl',
        },
    },
    createdAt: {
        type: Date,
        default: Date.now,
    },
    updatedAt: {
        type: Date,
    },
});

// Indexes for faster queries
staffSchema.index({ email: 1 });
staffSchema.index({ name: 1 });

module.exports = mongoose.model('Staff', staffSchema);