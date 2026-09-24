/**
 * app.js - 用户决策页逻辑
 * 选项采用动态添加卡片输入，反馈采用点击选择
 */

let currentPrediction = null;
let optionIdCounter = 0;

document.addEventListener('DOMContentLoaded', function () {
    if (window.Particles) {
        Particles.init('#particles-js', {
            color: '#00d4ff',
            count: 80,
            maxDistance: 130,
            speed: 0.5
        });
    }

    const btnAdd = document.getElementById('btn-add-option');
    const btnPredict = document.getElementById('btn-predict');
    const btnFeedback = document.getElementById('btn-feedback');
    const btnReset = document.getElementById('btn-reset');

    if (btnAdd) btnAdd.addEventListener('click', () => addOption());
    if (btnPredict) btnPredict.addEventListener('click', onPredict);
    if (btnFeedback) btnFeedback.addEventListener('click', onFeedback);
    if (btnReset) btnReset.addEventListener('click', onReset);

    // 默认添加 3 个选项
    for (let i = 0; i < 3; i++) {
        addOption(`选项${i + 1}`);
    }

    // 加载首页反馈累计进度（真实数据）
    loadHomeFeedbackStats();
});

/**
 * 加载/刷新首页反馈累计进度：使用后端 /feedback/stats 的真实 unused 数据
 */
async function loadHomeFeedbackStats() {
    const accumEl = document.getElementById('home-accum');
    const threshEl = document.getElementById('home-threshold');
    if (!accumEl) return;
    try {
        const stats = await getFeedbackStats();
        // 真实数据：考核自动训练进度的累计数 = 未用反馈数（unused）
        const accumulated = stats.accumulated !== undefined
            ? stats.accumulated
            : (stats.unused !== undefined ? stats.unused : 0);
        const threshold = stats.threshold !== undefined ? stats.threshold : FEEDBACK_THRESHOLD;
        if (accumEl) accumEl.textContent = accumulated;
        if (threshEl) threshEl.textContent = threshold;
    } catch (err) {
        // 失败时保留占位，不阻塞页面
    }
}

function addOption(name) {
    const container = document.getElementById('options-container');
    const id = ++optionIdCounter;
    const div = document.createElement('div');
    div.className = 'option-card';
    div.dataset.optionId = id;
    div.innerHTML = `
        <div class="option-card-header">
            <span class="option-card-label">选项 ${container.children.length + 1}</span>
            <button class="btn-cyber btn-sm btn-danger" onclick="removeOption(this)">✕ 删除</button>
        </div>
        <div class="form-group">
            <label>选项名称</label>
            <input class="option-name" type="text" value="${name || ''}" placeholder="如：购买、观望、抛售...">
        </div>
        <div class="option-probs-group">
            <label>转移概率（两个结局，留空由模型预测）</label>
            <div class="probs-input-row">
                <input class="option-prob" type="text" placeholder="好结局概率" data-outcome="0">
                <input class="option-prob" type="text" placeholder="差结局概率" data-outcome="1">
            </div>
        </div>
        <div class="form-group">
            <label>期望回报（留空由模型预测）</label>
            <input class="option-return" type="text" placeholder="如：0.85">
        </div>
    `;
    container.appendChild(div);
}

function removeOption(btn) {
    const card = btn.closest('.option-card');
    card.remove();
    // 重新编号
    const container = document.getElementById('options-container');
    const labels = container.querySelectorAll('.option-card-label');
    labels.forEach((el, i) => {
        el.textContent = `选项 ${i + 1}`;
    });
}

function getOptions() {
    const container = document.getElementById('options-container');
    const cards = container.querySelectorAll('.option-card');
    const actions = [];
    let allTransitionsFilled = true;
    let allReturnsFilled = true;

    cards.forEach(card => {
        const name = card.querySelector('.option-name').value.trim();
        actions.push(name || `选项${actions.length + 1}`);

        const probInputs = card.querySelectorAll('.option-prob');
        const p0 = probInputs[0].value.trim();
        const p1 = probInputs[1].value.trim();
        if (!p0 || !p1) allTransitionsFilled = false;

        const r = card.querySelector('.option-return').value.trim();
        if (!r) allReturnsFilled = false;
    });

    // 只有所有选项都填齐某字段时才提交该字段，否则传 null（由后端用模型预测填充）
    let transition_probs = null;
    if (allTransitionsFilled && cards.length > 0) {
        transition_probs = [];
        cards.forEach(card => {
            const probInputs = card.querySelectorAll('.option-prob');
            transition_probs.push([parseFloat(probInputs[0].value), parseFloat(probInputs[1].value)]);
        });
    }

    let expected_returns = null;
    if (allReturnsFilled && cards.length > 0) {
        expected_returns = [];
        cards.forEach(card => {
            expected_returns.push(parseFloat(card.querySelector('.option-return').value));
        });
    }

    return { actions, transition_probs, expected_returns };
}

async function onPredict() {
    const stateText = document.getElementById('state-text').value.trim();
    if (!stateText) {
        showToast('请输入决策场景描述');
        return;
    }

    const { actions } = getOptions();
    if (actions.length < 1) {
        showToast('请至少添加一个选项');
        return;
    }

    const btn = document.getElementById('btn-predict');
    const originalText = btn.textContent;
    btn.textContent = '生成中...';
    btn.disabled = true;

    try {
        const result = await predict(stateText, actions);
        currentPrediction = result;
        renderResult(result, actions);
        renderFeedback(actions);
        document.getElementById('result-card').classList.remove('hidden');
        document.getElementById('feedback-card').classList.remove('hidden');
        showToast('决策生成完成');
    } catch (err) {
        showToast('生成失败：' + err.message);
    } finally {
        btn.textContent = originalText;
        btn.disabled = false;
    }
}

function renderResult(result, actions) {
    const choiceEl = document.getElementById('result-choice');
    const choice = result.choice !== undefined ? result.choice : 0;
    const label = actions[choice] || `选项${choice + 1}`;
    choiceEl.textContent = '→ ' + label;

    // 置信度
    const confidence = typeof result.confidence === 'number' ? result.confidence : 0;
    const meter = document.getElementById('confidence-meter');
    meter.innerHTML = `
        <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
            <span style="color:#00d4ff; font-size:13px;">置信度</span>
            <span style="color:#00d4ff; font-size:13px;">${(confidence * 100).toFixed(1)}%</span>
        </div>
        <div class="confidence-meter">
            <div class="confidence-fill" style="width:${(confidence * 100).toFixed(1)}%"></div>
        </div>
    `;

    // 概率条形图
    const probs = result.choice_probs || [];
    const probsEl = document.getElementById('result-probs');
    probsEl.innerHTML = '<div style="color:#9ca3af; font-size:13px; margin-bottom:8px;">各选项概率</div>';
    probs.forEach((prob, idx) => {
        const label = actions[idx] || `选项${idx + 1}`;
        const pct = Math.max(0, Math.min(1, prob)) * 100;
        const row = document.createElement('div');
        row.className = 'prob-bar-container';
        row.innerHTML = `
            <div class="prob-label">${label}</div>
            <div class="prob-bar-track">
                <div class="prob-bar" style="width:${pct.toFixed(1)}%"></div>
            </div>
            <div class="prob-value">${(prob * 100).toFixed(1)}%</div>
        `;
        probsEl.appendChild(row);
    });

    // 期望回报
    const returns = result.expected_returns || [];
    const returnsEl = document.getElementById('result-returns');
    if (returns.length > 0) {
        const items = returns.map((r, idx) => {
            const label = actions[idx] || `选项${idx + 1}`;
            return `${label}: <span style="color:#00d4ff">${Number(r).toFixed(3)}</span>`;
        }).join('　|　');
        returnsEl.innerHTML = `<div style="color:#9ca3af; font-size:13px;">期望回报</div><div style="margin-top:4px;">${items}</div>`;
    } else {
        returnsEl.innerHTML = '';
    }
}

function renderFeedback(actions) {
    const container = document.getElementById('feedback-options');
    container.innerHTML = '';
    actions.forEach((label, idx) => {
        const div = document.createElement('div');
        div.className = 'feedback-option';
        div.dataset.fbIdx = idx;
        div.innerHTML = `<span>${label}</span><span class="fb-check">✓</span>`;
        div.addEventListener('click', function () {
            container.querySelectorAll('.feedback-option').forEach(el => el.classList.remove('selected'));
            this.classList.add('selected');
        });
        container.appendChild(div);
    });
    // 默认选中第一个
    const first = container.querySelector('.feedback-option');
    if (first) first.classList.add('selected');
}

async function onFeedback() {
    if (!currentPrediction) {
        showToast('请先生成决策结果');
        return;
    }

    const selected = document.querySelector('#feedback-options .feedback-option.selected');
    if (!selected) {
        showToast('请选择你认为正确的选项');
        return;
    }
    const correctIdx = parseInt(selected.dataset.fbIdx, 10);

    const stateText = document.getElementById('state-text').value.trim();
    const { actions, transition_probs, expected_returns } = getOptions();

    const data = {
        state_text: stateText,
        actions: actions,
        optimal_action: correctIdx,
        predicted_choice: currentPrediction.choice !== undefined ? currentPrediction.choice : 0,
        transition_probs: transition_probs,
        expected_returns: expected_returns
    };

    const btn = document.getElementById('btn-feedback');
    const originalText = btn.textContent;
    btn.textContent = '提交中...';
    btn.disabled = true;

    try {
        const result = await submitFeedback(data);
        // 用后端返回的真实累计数（未用反馈数），而非写死 0
        const accumulated = result.accumulated_count !== undefined ? result.accumulated_count
            : (result.accumulated !== undefined ? result.accumulated : (result.count || 0));
        const threshold = (result.threshold !== undefined ? result.threshold : FEEDBACK_THRESHOLD);
        const remaining = Math.max(0, threshold - accumulated);
        showToast(`反馈已提交！已积累 ${accumulated}/${threshold}，距离下次训练 ${remaining} 条`);
        // 同步刷新首页进度条为真实数据
        loadHomeFeedbackStats();
    } catch (err) {
        showToast('提交失败：' + err.message);
    } finally {
        btn.textContent = originalText;
        btn.disabled = false;
    }
}

function onReset() {
    document.getElementById('state-text').value = '';
    const container = document.getElementById('options-container');
    container.innerHTML = '';
    optionIdCounter = 0;
    for (let i = 0; i < 3; i++) {
        addOption(`选项${i + 1}`);
    }
    document.getElementById('result-card').classList.add('hidden');
    document.getElementById('feedback-card').classList.add('hidden');
    document.getElementById('feedback-options').innerHTML = '';
    currentPrediction = null;
    showToast('已清空');
}
