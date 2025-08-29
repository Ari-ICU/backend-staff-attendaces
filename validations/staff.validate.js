const Joi = require('joi');

const createStaffSchema = Joi.object({
    name: Joi.string().required(),
    email: Joi.string().email().required(),
    phone: Joi.string().optional(),
    role: Joi.string().valid('staff', 'admin').required(),
    // Remove permissions from schema
    // permissions: Joi.array().items(Joi.string()).optional(),
});

const updateStaffSchema = Joi.object({
    name: Joi.string().optional(),
    email: Joi.string().email().optional(),
    phone: Joi.string().optional(),
    role: Joi.string().valid('staff', 'admin').optional(),
});

module.exports = {
    createStaffSchema,
    updateStaffSchema,
};
