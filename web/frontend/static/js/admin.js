/**
 * admin.js - 管理员页逻辑
 * 处理训练记录、模型信息、反馈统计、CSV 上传、手动训练
 */

let refreshTimer = null;

document.addEventListener('DOMContentLoaded', function () {
    // 初始化粒子背景
    if (window.Particles) {
        Particles.init('#particles-js', {
            color: '#7c3aed',
            count: 70,
            maxDistance: 120,
            speed: 0.4
        });
    }

    // 加载初始数据
    loadModelInfo();
    loadFeedbackStats();
    loadTrainingRecords();

    // CSV 上传相关事件
    setupCSVUpload();

    // 手动训练按钮
    const btnTrain = document.getElementById('btn-train');
    if (btnTrain) btnTrain.addEventListener('click', onTrain);

    // 每 10 秒自动刷新训练记录和统计
    refreshTimer = setInterval(() => {
        loadTrainingRecords();
        loadFeedbackStats();
    }, 10000);
});

/**
 * 加载模型信息
 */
async function loadModelInfo() {
    const el = document.getElementById('model-info');
    if (!el) return;
    el.innerHTML = '<div style="color:#9ca3af;">加载中...</div>';
    try {
        const info = await getModelInfo();
        const items = [
            ['版本', info.version || info.model_version || '-'],
            ['路径', info.path || info.model_path || '-'],
            ['Loss', info.loss !== undefined && info.loss !== null ? Number(info.loss).toFixed(4) : '-'],
            ['参数量', info.params !== undefined && info.params !== null ? info.params.toLocaleString() : '-'],
            ['最后更新', info.updated_at || info.last_updated || '-']
        ];
        el.innerHTML = items.map(([k, v]) => `
            <div class="info-row">
                <span class="info-key">${k}</span>
                <span class="info-value">${v}</span>
            </div>
        `).join('');
    } catch (err) {
        el.innerHTML = `<div style="color:#ef4444;">加载失败：${err.message}</div>`;
    }
}

/**
 * 加载反馈统计并渲染进度环
 */
async function loadFeedbackStats() {
    const textEl = document.getElementById('feedback-stats-text');
    const ringEl = document.querySelector('.progress-ring');
    if (!textEl) return;
    try {
        const stats = await getFeedbackStats();
        // 后端返回 accumulation 用 unused 字段（未用于训练的反馈数）
        const accumulated = stats.accumulated !== undefined
            ? stats.accumulated
            : (stats.unused !== undefined ? stats.unused : (stats.count || 0));
        const threshold = stats.threshold !== undefined ? stats.threshold : FEEDBACK_THRESHOLD;
        const remaining = Math.max(0, threshold - accumulated);
        const pct = threshold > 0 ? Math.min(100, (accumulated / threshold) * 100) : 0;

        // 更新进度环
        if (ringEl) {
            ringEl.style.setProperty('--progress', pct.toFixed(1) + '%');
            const inner = ringEl.querySelector('.progress-ring-inner');
            if (inner) {
                inner.innerHTML = `
                    <div class="progress-ring-text">${accumulated}/${threshold}</div>
                    <div class="progress-ring-label">累计反馈</div>
                `;
            }
        }

        textEl.innerHTML = `
            <div class="info-row"><span class="info-key">已累计</span><span class="info-value">${accumulated} 条</span></div>
            <div class="info-row"><span class="info-key">训练阈值</span><span class="info-value">${threshold} 条</span></div>
            <div class="info-row"><span class="info-key">距离下次训练</span><span class="info-value">${remaining} 条</span></div>
            <div class="info-row"><span class="info-key">进度</span><span class="info-value">${pct.toFixed(1)}%</span></div>
        `;
    } catch (err) {
        textEl.innerHTML = `<div style="color:#ef4444;">加载失败：${err.message}</div>`;
    }
}

/**
 * 加载训练记录表
 */
async function loadTrainingRecords() {
    const table = document.getElementById('training-records-table');
    if (!table) return;
    try {
        const data = await getTrainingRecords(20, 0);
        const records = data.records || data.items || data || [];
        if (!Array.isArray(records) || records.length === 0) {
            table.innerHTML = `
                <thead><tr><th>暂无训练记录</th></tr></thead>
                <tbody><tr><td style="text-align:center; color:#9ca3af;">点击"立即触发训练"开始</td></tr></tbody>
            `;
            return;
        }
        table.innerHTML = `
            <thead>
                <tr>
                    <th>运行名称</th>
                    <th>状态</th>
                    <th>样本数</th>
                    <th>Loss</th>
                    <th>开始时间</th>
                    <th>操作</th>
                </tr>
            </thead>
            <tbody>
                ${records.map(r => {
                    const id = r.id;
                    const status = r.status || 'pending';
                    const statusClass = `badge badge-${status}`;
                    const loss = r.best_val_loss !== undefined && r.best_val_loss !== null
                        ? Number(r.best_val_loss).toFixed(4) : '-';
                    const samples = r.num_samples !== undefined && r.num_samples !== null
                        ? r.num_samples : '-';
                    const time = r.start_time || r.created_at || '-';
                    return `
                        <tr>
                            <td>${r.run_name || r.name || '-'}</td>
                            <td><span class="${statusClass}">${status}</span></td>
                            <td>${samples}</td>
                            <td>${loss}</td>
                            <td>${time}</td>
                            <td>
                                <button class="btn btn-sm btn-danger"
                                        data-delete-id="${id}"
                                        onclick="handleDeleteRecord(${id})">删除</button>
                            </td>
                        </tr>
                    `;
                }).join('')}
            </tbody>
        `;
    } catch (err) {
        table.innerHTML = `<thead><tr><th>加载失败</th></tr></thead><tbody><tr><td style="color:#ef4444;">${err.message}</td></tr></tbody>`;
    }
}

/**
 * 删除单条训练记录（带确认）
 * @param {number} recordId 训练记录 ID
 */
async function handleDeleteRecord(recordId) {
    if (!confirm(`确定删除训练记录 #${recordId} 吗？此操作不可撤销。`)) {
        return;
    }
    try {
        await deleteTrainingRecord(recordId);
        showToast(`已删除训练记录 #${recordId}`);
        loadTrainingRecords();
    } catch (err) {
        showToast('删除失败：' + err.message);
    }
}

/**
 * 配置 CSV 上传交互
 */
function setupCSVUpload() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('csv-file');
    const btnUpload = document.getElementById('btn-upload');
    if (!dropZone || !fileInput) return;

    // 点击拖拽区触发文件选择
    dropZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            dropZone.querySelector('.drop-text') && (dropZone.querySelector('.drop-text').textContent = fileInput.files[0].name);
        }
    });

    // 拖拽事件
    ['dragenter', 'dragover'].forEach(evt => {
        dropZone.addEventListener(evt, e => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
    });
    ['dragleave', 'drop'].forEach(evt => {
        dropZone.addEventListener(evt, e => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
        });
    });
    dropZone.addEventListener('drop', e => {
        if (e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            const dropText = dropZone.querySelector('.drop-text');
            if (dropText) dropText.textContent = e.dataTransfer.files[0].name;
            else dropZone.querySelector('.drop-text') && (dropZone.querySelector('.drop-text').textContent = e.dataTransfer.files[0].name);
        }
    });

    if (btnUpload) btnUpload.addEventListener('click', onUploadCSV);
}

/**
 * 上传 CSV 文件
 */
async function onUploadCSV() {
    const fileInput = document.getElementById('csv-file');
    if (!fileInput.files || fileInput.files.length === 0) {
        showToast('请先选择 CSV 文件');
        return;
    }
    const file = fileInput.files[0];
    if (!file.name.toLowerCase().endsWith('.csv')) {
        showToast('仅支持 .csv 文件');
        return;
    }

    const btn = document.getElementById('btn-upload');
    const originalText = btn.textContent;
    btn.textContent = '上传中...';
    btn.disabled = true;

    try {
        const result = await uploadCSV(file);
        const count = result.count !== undefined ? result.count : (result.rows || 0);
        showToast(`上传成功，共 ${count} 条标注数据`);
        // 上传后刷新统计
        loadFeedbackStats();
    } catch (err) {
        showToast('上传失败：' + err.message);
    } finally {
        btn.textContent = originalText;
        btn.disabled = false;
    }
}

/**
 * 触发手动训练
 */
async function onTrain() {
    const maxSteps = parseInt(document.getElementById('max-steps').value, 10) || 200;
    const runName = document.getElementById('run-name').value.trim() || `手动训练_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}`;

    const btn = document.getElementById('btn-train');
    const originalText = btn.textContent;
    btn.textContent = '触发中...';
    btn.disabled = true;

    try {
        const result = await triggerTraining(maxSteps, runName);
        showToast('训练已触发：' + (result.run_name || runName));
        // 刷新记录
        loadTrainingRecords();
        loadFeedbackStats();
    } catch (err) {
        showToast('触发失败：' + err.message);
    } finally {
        btn.textContent = originalText;
        btn.disabled = false;
    }
}

// 页面卸载时清理定时器
window.addEventListener('beforeunload', () => {
    if (refreshTimer) clearInterval(refreshTimer);
});
