// utils/api.response.js

class ApiResponse {
    static success(res, data = null, message = 'Success', statusCode = 200) {
        return res.status(statusCode).json({
            status: true,
            message,
            data
        });
    }

    static error(res, message = 'Error', statusCode = 500, data = null) {
        return res.status(statusCode).json({
            status: false,
            message,
        });
    }
}

module.exports = ApiResponse;
