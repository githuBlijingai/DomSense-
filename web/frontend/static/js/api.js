/**
 * api.js - 后端 API 封装
 * 所有接口调用集中在此文件，方便维护
 */

const API_BASE = '/api/v1';

/** 触发下次训练所需的反馈条数阈值 */
const FEEDBACK_THRESHOLD = 50;

/**
 * 调用模型预测接口
 * @param {string} stateText 决策场景文本
 * @param {string} actions 选项文本（每行一个，空则用默认3个）
 * @returns {Promise<Object>} 预测结果
 */
async function predict(stateText, actions) {
    const resp = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ state_text: stateText, actions: actions })
    });
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || err.message || `预测失败 (${resp.status})`);
    }
    return resp.json();
}

/**
 * 提交用户反馈
 * @param {Object} data 反馈数据
 * @returns {Promise<Object>} 提交结果（含累计条数）
 */
async function submitFeedback(data) {
    const resp = await fetch(`${API_BASE}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || err.message || `提交反馈失败 (${resp.status})`);
    }
    return resp.json();
}

/**
 * 获取反馈统计
 * @returns {Promise<Object>} 统计信息（累计 / 阈值 / 距离下次训练）
 */
async function getFeedbackStats() {
    const resp = await fetch(`${API_BASE}/feedback/stats`);
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || err.message || `获取统计失败 (${resp.status})`);
    }
    return resp.json();
}

/**
 * 上传 CSV 标注文件
 * @param {File} file CSV 文件
 * @returns {Promise<Object>} 上传结果
 */
async function uploadCSV(file) {
    const formData = new FormData();
    formData.append('file', file);
    const resp = await fetch(`${API_BASE}/admin/upload-csv`, {
        method: 'POST',
        body: formData
    });
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || err.message || `上传失败 (${resp.status})`);
    }
    return resp.json();
}

/**
 * 触发模型训练
 * @param {number} maxSteps 最大训练步数
 * @param {string} runName 运行名称
 * @returns {Promise<Object>} 训练任务信息
 */
async function triggerTraining(maxSteps, runName) {
    const resp = await fetch(`${API_BASE}/admin/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ max_steps: maxSteps, run_name: runName })
    });
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || err.message || `触发训练失败 (${resp.status})`);
    }
    return resp.json();
}

/**
 * 获取训练记录列表
 * @param {number} limit 数量上限
 * @param {number} offset 偏移量
 * @returns {Promise<Object>} 记录列表
 */
async function getTrainingRecords(limit = 20, offset = 0) {
    const resp = await fetch(`${API_BASE}/admin/training-records?limit=${limit}&offset=${offset}`);
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || err.message || `获取记录失败 (${resp.status})`);
    }
    return resp.json();
}

/**
 * 删除单条训练记录
 * @param {number} recordId 训练记录 ID
 * @returns {Promise<Object>} 删除结果
 */
async function deleteTrainingRecord(recordId) {
    const resp = await fetch(`${API_BASE}/admin/training-records/${recordId}`, {
        method: 'DELETE'
    });
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || err.message || `删除记录失败 (${resp.status})`);
    }
    return resp.json();
}

/**
 * 获取当前模型信息
 * @returns {Promise<Object>} 模型信息
 */
async function getModelInfo() {
    const resp = await fetch(`${API_BASE}/admin/model/info`);
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || err.message || `获取模型信息失败 (${resp.status})`);
    }
    return resp.json();
}

/**
 * 显示 Toast 通知
 * @param {string} message 提示文字
 * @param {number} duration 显示时长（毫秒）
 */
function showToast(message, duration = 3000) {
    // 移除已有 toast
    const existing = document.querySelector('.toast');
    if (existing) {
        existing.remove();
    }
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.3s';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}
